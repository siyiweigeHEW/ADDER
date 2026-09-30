"""TVM backend for the API verifier: static index, call extraction, resolution.

TVM's Python source cannot be imported in the analysis environment, so the API list is
built by **statically scanning the source tree**: for each namespace, parse the
package's ``__init__.py`` re-exports (``from .X import (a, b, ...)``) plus the ``def``
names defined in the package's own modules.

Modules that can be imported at runtime (numpy / functools / math / onnx / ...) are
handled by the verifier's runtime path, not here.

Consumed by ``api_check.py``; see that module for the interface a backend must provide.
"""
import ast
import difflib
import importlib
import inspect
import os
import re

NAME = "TVM"

ROOT_ENV = "TVM_PYTHON_ROOT"
ROOT_HINT = "the `python/tvm` directory of a TVM checkout"
ROOT_MARKER = ("relax", "op", "__init__.py")

REVIEW_SYSTEM = (
    "You are a meticulous TVM frontend developer reviewing whether Relax/Topi/NumPy "
    "APIs are used correctly. Return only the JSON object."
)

# LLM-facing API name -> relative directory in the tvm source tree.
SEED_ALIASES = [
    ("relax.op", "relax/op"),
    ("relax", "relax"),
    ("topi.nn", "topi/nn"),
    ("topi", "topi"),
    ("tirx", "tirx"),
]

# These namespaces are fully enumerated (init re-exports + in-package defs), so the
# missing verdict is authoritative; for other namespaces (e.g., relax, which aggregates
# many level-0 re-exports) a leaf not found degrades to unverifiable, to avoid false
# positives on real APIs that happen not to be enumerated.
MISSING_STRICT_NS = {"relax.op", "relax.op.nn", "topi", "topi.nn", "tirx"}

# LLM-facing name -> real module name
RUNTIME_ROOTS = {
    "np": "numpy",
    "_np": "numpy",
    "numpy": "numpy",
    "functools": "functools",
    "math": "math",
    "operator": "operator",
    "onnx": "onnx",
    "re": "re",
    "os": "os",
    "warnings": "warnings",
    "json": "json",
    "collections": "collections",
    "collections.abc": "collections.abc",
}

ROOT_WHITELIST = set(RUNTIME_ROOTS) | {"relax", "topi", "tirx", "tvm"}

SELF_TEST_CASES = [
    ("relax.op.add", "ok"),
    ("relax.op.zeros", "ok"),
    ("relax.op.nn.prelu", "ok"),
    ("relax.op.foobar", "missing"),
    ("relax.const", "ok"),
    ("topi.nn.pad", "ok"),
    ("topi.nn.foobar", "missing"),
    ("tirx.IntImm", "ok"),
    ("np.mean", "ok"),
    ("_np.stack", "ok"),
    ("functools.reduce", "ok"),
    ("relax.op.foo.bar", "unverifiable"),
]


# ---------------------------------------------------------------- root


def detect_root():
    """The TVM `python/tvm` directory, or None."""
    env = os.environ.get(ROOT_ENV)
    if env and os.path.isfile(os.path.join(env, *ROOT_MARKER)):
        return env
    candidate = "[your tvm source root]"
    if os.path.isfile(os.path.join(candidate, *ROOT_MARKER)):
        return candidate
    return None


# ---------------------------------------------------------------- static index


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def scan_defs(filepath):
    """Return a list of (name, filepath): all top-level `def name` in the file
    (respecting __all__)."""
    if not os.path.isfile(filepath):
        return []
    try:
        tree = ast.parse(_read(filepath))
    except SyntaxError:
        return []
    all_names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ) and node.targets[0].id == "__all__":
            try:
                all_names = [e.value for e in node.value.elts]
            except AttributeError:
                all_names = None
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if all_names is None or node.name in all_names:
                out.append((node.name, filepath))
    return out


def parse_reexports(pkg_dir):
    """Parse the re-exports of a package's __init__.py, building the set of API names
    and a name -> source-file map.

    Handles two styles:
      from .binary import (add, subtract, ...)    # explicit list
      from .conv1d import *                       # star: scan all defs in the imported file
    Also adds: all top-level defs in every *.py of the package (to cover APIs that are
    reachable as namespace attributes even if not re-exported).
    Returns (names:set, src_map:dict[name->filepath]).
    """
    names = set()
    src_map = {}
    init = os.path.join(pkg_dir, "__init__.py")
    if os.path.isfile(init):
        try:
            tree = ast.parse(_read(init))
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    if node.module:  # from .sub import (...)
                        sub = node.module.replace(".", "/")
                        for alias in node.names:
                            if alias.name == "*":
                                for nm, f in scan_defs(os.path.join(pkg_dir, sub + ".py")):
                                    names.add(nm)
                                    src_map.setdefault(nm, f)
                            else:
                                names.add(alias.name)
                                src_map.setdefault(
                                    alias.name, os.path.join(pkg_dir, sub + ".py")
                                )
                    else:  # from . import a, b, c (submodule names)
                        for alias in node.names:
                            names.add(alias.name)
    # Supplement: all defs in the package's top-level files (including non-re-exported),
    # to reduce false negatives
    if os.path.isdir(pkg_dir):
        for f in sorted(os.listdir(pkg_dir)):
            if f.endswith(".py"):
                for nm, fp in scan_defs(os.path.join(pkg_dir, f)):
                    names.add(nm)
                    src_map.setdefault(nm, fp)
    return names, src_map


def build_index(root):
    """Build a namespace -> {names, src, dir} index from the TVM source tree.

    Sub-package directories (with __init__.py) under each seed namespace are indexed
    too, so `relax.op` also yields `relax.op.image` and so on.
    """
    namespaces = {}

    def add_ns(ns_name, rel):
        pkg_dir = os.path.join(root, rel)
        if not os.path.isdir(pkg_dir):
            return
        names, src = parse_reexports(pkg_dir)
        namespaces[ns_name] = {"names": names, "src": src, "dir": pkg_dir}

    for alias, rel in SEED_ALIASES:
        add_ns(alias, rel)
        pkg_dir = os.path.join(root, rel)
        if os.path.isdir(pkg_dir):
            for sub in sorted(os.listdir(pkg_dir)):
                subdir = os.path.join(pkg_dir, sub)
                if os.path.isdir(subdir) and os.path.isfile(
                    os.path.join(subdir, "__init__.py")
                ):
                    add_ns(f"{alias}.{sub}", f"{rel}/{sub}")
    return namespaces


def describe_index(index):
    """One-line summary for the report and the self-test."""
    total = sum(len(rec["names"]) for rec in index.values())
    return f"{len(index)} namespaces, {total} names"


# ---------------------------------------------------------------- resolution


def resolve_runtime(name):
    """Resolve at runtime via inspect (numpy/functools and other importable modules).

    Returns (status, ns, leaf, info, close) -- the same 5-tuple shape as `resolve`.
    """
    parts = name.split(".")
    root = RUNTIME_ROOTS.get(parts[0], parts[0])
    try:
        obj = importlib.import_module(root)
    except ImportError:
        return ("unverifiable", root, parts[0], None, None)
    for p in parts[1:]:
        if not hasattr(obj, p):
            return ("missing", root, p, None, None)
        obj = getattr(obj, p)
    return ("ok", root, parts[-1], obj, None)


def resolve(name, index):
    """Resolve a dotted API name. Returns (status, ns, leaf, info, close).

    Always a 5-tuple, whatever the status, so callers can unpack unconditionally.

    status: ok / missing / unverifiable
      ok          the API exists in the namespace
      missing     the namespace resolved, but the leaf is not in its API list
      unverifiable the namespace is not indexed, or the intermediate level cannot be
                   confirmed (to avoid false positives)

    A `missing` verdict is only issued for a namespace in MISSING_STRICT_NS; elsewhere
    the leaf may simply not have been enumerated.
    """
    if name in index:
        return ("ok", name, name, index[name], None)
    ns_list = sorted(index.keys(), key=len, reverse=True)
    for ns in ns_list:
        if name.startswith(ns + "."):
            rest = name[len(ns):].lstrip(".")
            info = index[ns]
            parts = rest.split(".")
            if len(parts) == 1:
                if parts[0] in info["names"]:
                    return ("ok", ns, parts[0], info, None)
                # difflib needs an ordered sequence; a set would make the suggestions
                # differ between runs.
                close = difflib.get_close_matches(parts[0], sorted(info["names"]), n=3)
                if ns in MISSING_STRICT_NS:
                    return ("missing", ns, parts[0], info, close)
                return ("unverifiable", ns, parts[0], info, None)
            # Chained: the whole chain must be within the namespace to be ok, otherwise
            # do not confirm (to avoid false positives)
            if all(p in info["names"] for p in parts):
                return ("ok", ns, parts[-1], info, None)
            return ("unverifiable", ns, parts[0], info, None)
    root = name.split(".")[0]
    if root in ROOT_WHITELIST:
        return resolve_runtime(name)
    return ("unverifiable", None, root, None, None)


def describe(name, index):
    """Return (signature, docstring summary, source) for an existing API."""
    st, ns, leaf, info, _close = resolve(name, index)
    if st != "ok":
        return (None, None, st)
    if isinstance(info, dict) and "src" in info:  # static namespace
        fp = info["src"].get(leaf)
        if fp:
            res = extract_def_doc(fp, leaf)
            if res:
                sig, doc, lineno = res
                return (sig, doc, f"{os.path.relpath(fp, info['dir'])}:{lineno}")
        return (None, None, "static-no-def")
    if info is not None:  # runtime object
        try:
            doc = inspect.getdoc(info) or ""
            return (f"def {leaf}(...)", doc, f"runtime {ns}")
        except Exception:
            return (None, None, "runtime-nodoc")
    return (None, None, st)


def extract_def_doc(filepath, name):
    """Extract the signature and docstring of `def name` from the source file.
    Returns (sig, doc, lineno) or None."""
    if not os.path.isfile(filepath):
        return None
    try:
        tree = ast.parse(_read(filepath))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            sig = ast.unparse(node.args)
            doc = ast.get_docstring(node) or ""
            return (f"def {name}({sig})", doc, node.lineno)
    return None


# ---------------------------------------------------------------- call extraction

DOTTED_CALL_RE = re.compile(r"(?<![.\w])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(")


def extract_calls(code, index=None):
    """Extract dotted API call names (e.g. `relax.op.add(`) from Python code.

    Only keeps calls whose root is in ROOT_WHITELIST; runtime objects such as
    bb./attr./cls./x. are skipped.
    """
    out = []
    for m in DOTTED_CALL_RE.finditer(code):
        name = m.group(1)
        if name.split(".")[0] in ROOT_WHITELIST:
            out.append(name)
    return out

