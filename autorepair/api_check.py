#!/usr/bin/env python3
"""API verification of LLM fix patches: namespace existence check + misuse review +
correction loop.

Targets two kinds of LLM mistakes on APIs: 'fabrication' (using an API that does not
exist) and 'misuse' (using an existing API incorrectly):

1. After the patch is generated, extract `a.b.c(...)`-style API calls from the
   **added lines** of the fixed code;
2. Resolve the API's namespace (e.g., `relax.op.add` -> `relax.op`), get the **full
   list of APIs** in that namespace, and check whether the API is among them
   (existence check);
3. For existing APIs, extract their signature/docstring (API brief docs) and ask the
   LLM to review whether they are misused;
4. When an API is 'nonexistent' or 'misused', design the detected information
   (with evidence) into a prompt and feed it back to the LLM, asking it to re-output
   the fixed code after modification; iterate until it passes or reaches the round limit.

Implementation notes (static vs runtime inspect):
- tvm cannot be `import`ed locally (lib build issue; the local TVM builds raise
  AttributeError), so TVM namespaces (relax.op / topi.nn / tirx / ...) build the API
  list by **statically scanning the source tree**: parse each package's `__init__.py`
  re-exports (`from .X import (a, b, ...)`) plus in-package `def` names.
- Modules that can be imported at runtime (numpy / functools / math / onnx / ...) use
  the standard `inspect` mechanism for the existence check and doc extraction.

Self-test: `python api_check.py --self-test` (no LLM; only verifies the static index
and the resolution logic).
"""
import ast
import difflib
import importlib
import inspect
import json
import os
import re

from llm_client import DeepseekV4FlashClient  # noqa: E402

# ---------------------------------------------------------------- constants

# LLM-facing API name -> relative directory in the tvm source tree (rooted at detect_tvm_root())
SEED_ALIASES = [
    ("relax.op", "relax/op"),
    ("relax", "relax"),
    ("topi.nn", "topi/nn"),
    ("topi", "topi"),
    ("tirx", "tirx"),
]

# These 5 namespaces are fully enumerated (init re-exports + in-package defs), so the
# missing verdict is authoritative; for other namespaces (e.g., relax, which aggregates
# many level-0 re-exports) a leaf not found degrades to unverifiable, to avoid false
# positives on real APIs that happen not to be enumerated.
MISSING_STRICT_NS = {"relax.op", "relax.op.nn", "topi", "topi.nn", "tirx"}

# Runtime inspect roots (LLM-facing name -> real module name)
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

# Dotted calls in the fixed code whose root is not in these lists (e.g., bb./attr./cls./x.)
# are skipped
ROOT_WHITELIST = set(RUNTIME_ROOTS) | {"relax", "topi", "tirx", "tvm"}

DOTTED_CALL_RE = re.compile(r"(?<![.\w])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(")

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def _clean_block(text):
    """Strip leading/trailing blank lines and trailing whitespace, but preserve the
    first line's indentation (same as repair.py's clean_block)."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extract_code(response):
    """Extract the ```python code block from the response; return the whole response if
    there is no fence (same as repair.py)."""
    m = CODE_FENCE_RE.search(response)
    return (_clean_block(m.group(1)) + "\n") if m else _clean_block(response) + "\n"


REVIEW_SYSTEM = (
    "You are a meticulous TVM frontend developer reviewing whether Relax/Topi/NumPy "
    "APIs are used correctly. Return only the JSON object."
)

# ---------------------------------------------------------------- tvm root


def detect_tvm_root():
    """Automatically locate the tvm source python/tvm directory (for static scanning)."""
    env = os.environ.get("TVM_PYTHON_ROOT")
    if env and os.path.isfile(os.path.join(env, "relax", "op", "__init__.py")):
        return env
    candidates = [
        # Point this at the `python/tvm` directory of the TVM checkout you want
        # scanned, or leave it unset and export TVM_PYTHON_ROOT instead.
        "[your tvm source root]",
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(os.path.join(c, "relax", "op", "__init__.py")):
            return c
    return None


# ---------------------------------------------------------------- static index

_INDEX_CACHE = {}


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


def build_index(tvm_root):
    """Build a namespace -> {names, src, dir} index.

    Sub-package directories (with __init__.py) under the seed namespaces are also
    indexed automatically, e.g., image/memory/nn/vision under relax/op ->
    relax.op.image, etc.
    Returns a dict: ns_name -> info.
    """
    namespaces = {}

    def add_ns(ns_name, rel):
        pkg_dir = os.path.join(tvm_root, rel)
        if not os.path.isdir(pkg_dir):
            return
        names, src = parse_reexports(pkg_dir)
        namespaces[ns_name] = {"names": names, "src": src, "dir": pkg_dir}

    for alias, rel in SEED_ALIASES:
        add_ns(alias, rel)
        pkg_dir = os.path.join(tvm_root, rel)
        if os.path.isdir(pkg_dir):
            for sub in sorted(os.listdir(pkg_dir)):
                subdir = os.path.join(pkg_dir, sub)
                if os.path.isdir(subdir) and os.path.isfile(
                    os.path.join(subdir, "__init__.py")
                ):
                    add_ns(f"{alias}.{sub}", f"{rel}/{sub}")
    return namespaces


def get_index(tvm_root=None):
    root = tvm_root or detect_tvm_root()
    if root not in _INDEX_CACHE:
        _INDEX_CACHE[root] = build_index(root)
    return _INDEX_CACHE[root]


# ---------------------------------------------------------------- resolution/existence


def resolve_runtime(name):
    """Resolve at runtime via inspect (numpy/functools and other importable modules).
    Returns (status, detail...)."""
    parts = name.split(".")
    root = RUNTIME_ROOTS.get(parts[0], parts[0])
    try:
        obj = importlib.import_module(root)
    except ImportError:
        return ("unverifiable", root, parts[0], None)
    for p in parts[1:]:
        if not hasattr(obj, p):
            return ("missing", root, p, None)
        obj = getattr(obj, p)
    return ("ok", root, parts[-1], obj)


def resolve(name, index):
    """Resolve a dotted API name. Returns (status, ns, leaf, info).

    status: ok / missing / unverifiable
      ok          the API exists in the namespace
      missing     the namespace resolved, but the leaf is not in its API list (fabrication)
      unverifiable the namespace is not indexed or the intermediate level cannot be
                   confirmed (to avoid false positives)
    """
    if name in index:
        info = index[name]
        return ("ok", name, name, info)
    ns_list = sorted(index.keys(), key=len, reverse=True)
    for ns in ns_list:
        if name.startswith(ns + "."):
            rest = name[len(ns):].lstrip(".")
            info = index[ns]
            parts = rest.split(".")
            if len(parts) == 1:
                if parts[0] in info["names"]:
                    return ("ok", ns, parts[0], info)
                close = difflib.get_close_matches(parts[0], info["names"], n=3)
                if ns in MISSING_STRICT_NS:
                    return ("missing", ns, parts[0], info, close)
                return ("unverifiable", ns, parts[0], info, None)
            # Chained: the whole chain must be within the namespace to be ok, otherwise
            # do not confirm (to avoid false positives)
            if all(p in info["names"] for p in parts):
                return ("ok", ns, parts[-1], info)
            return ("unverifiable", ns, parts[0], info, None)
    root = name.split(".")[0]
    if root in ROOT_WHITELIST:
        st, mod, leaf, obj = resolve_runtime(name)
        return (st, mod, leaf, obj)
    return ("unverifiable", None, root, None, None)


def extract_api_calls(code):
    """Extract API call names of the form `relax.op.add(` / `topi.nn.pad(` from code.

    Only keeps dotted calls whose root is in ROOT_WHITELIST; runtime objects such as
    bb./attr./cls./x./slope. are skipped.
    """
    out = []
    for m in DOTTED_CALL_RE.finditer(code):
        name = m.group(1)
        root = name.split(".")[0]
        if root in ROOT_WHITELIST:
            out.append(name)
    return out


def added_lines(original, fixed):
    """Return the lines added/changed in fixed relative to original (the + lines of the
    diff)."""
    orig_lines = original.splitlines()
    fix_lines = fixed.splitlines()
    sm = difflib.SequenceMatcher(a=orig_lines, b=fix_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for line in fix_lines[j1:j2]:
                yield line


def extract_api_calls_from_lines(lines):
    return extract_api_calls("\n".join(lines))


# ---------------------------------------------------------------- API docs


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


def get_api_doc_text(name, index):
    """Return the API's (signature, docstring summary, source). For feeding to the
    review LLM."""
    r = resolve(name, index)
    st, ns, leaf, info = r[0], r[1], r[2], r[3]
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
    if isinstance(info, type) or info is not None:  # runtime object
        try:
            doc = inspect.getdoc(info) or ""
            return (f"def {leaf}(...)", doc, f"runtime {ns}")
        except Exception:
            return (None, None, "runtime-nodoc")
    return (None, None, st)


# ---------------------------------------------------------------- misuse review (LLM)


def _parse_verdicts(text):
    """Tolerantly parse the JSON returned by the review LLM (code block or raw JSON)."""
    if not text:
        return []
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    payload = m.group(1) if m else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        m2 = re.search(r"\{.*\}", payload, re.DOTALL)
        if not m2:
            return []
        try:
            data = json.loads(m2.group(0))
        except json.JSONDecodeError:
            return []
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    return [v for v in verdicts if isinstance(v, dict) and v.get("api")]


def review_usage(original, fixed, used_apis, index, client):
    """Ask the LLM to review, against the API brief docs, whether there is any misuse.
    Returns (verdicts, raw_response)."""
    added = "\n".join(added_lines(original, fixed)) or fixed
    doc_blocks = []
    for name in used_apis:
        sig, doc, src = get_api_doc_text(name, index)
        if sig is None:
            doc_blocks.append(f"### {name}\n(document unavailable: {src})")
        else:
            excerpt = doc[:800].replace("`", "'")
            doc_blocks.append(f"### {name}\nsignature: `{sig}`\nsource: {src}\n"
                              f"summary: {excerpt}")

    prompt = f"""# API usage review (autorepair automatic verification)

The added/changed lines of the fixed code call the following APIs. Based on each API's
signature and summary, judge whether its usage is correct.

## Original code (the fix scope; use it to infer codebase conventions)
```python
{original}
```

## Added/changed lines of the fixed code
```python
{added}
```

## Documentation of the used APIs (signature + summary)
{chr(10).join(doc_blocks)}

## Judging criteria (any hit -> correct=false)
1. Parameter name, type, or count does not match the signature;
2. Return type/semantics do not match what the call site expects;
3. Clearly deviates from codebase conventions (e.g., passing a DataType object where a
   string dtype is expected, using a Python built-in instead of a relax operator,
   hard-coding behavior that should be parameterized);
4. The API does not exist in this version.

## Output (strict JSON, only JSON)
```json
{{"verdicts": [
  {{"api": "<full API name>", "correct": true, "issue": "", "suggestion": ""}}
]}}
```
"""
    resp = client.model_prediction(prompt, system=REVIEW_SYSTEM)
    return _parse_verdicts(resp), resp


# ---------------------------------------------------------------- correction loop


def _format_issue(issue):
    """issue = (api, kind, detail). kind: missing | misuse."""
    api, kind, detail = issue
    if kind == "missing":
        st, ns, leaf, info, close = detail
        extra = f"; similar APIs: {', '.join(close)}" if close else ""
        ns_size = len(info["names"]) if isinstance(info, dict) else 0
        return (
            f"❌ **nonexistent API**: `{api}` -- not found in namespace `{ns}` "
            f"({ns_size} APIs){extra}. Use a real API or an equivalent low-level "
            f"implementation."
        )
    # misuse
    return (
        f"⚠ **suspected misuse**: `{api}` -- {detail.get('issue', '')}"
        + (f"  suggestion: {detail.get('suggestion', '')}"
           if detail.get("suggestion") else "")
    )


def build_correction_prompt(original, previous_fixed, issues):
    """Turn the detected API problems into a prompt asking the LLM to output the
    revised complete code."""
    issues_text = "\n".join(f"{i + 1}. {_format_issue(it)}" for i, it in enumerate(issues))
    return f"""# Fix the API problems in the previous repair round (autorepair automatic
verification feedback)

Your previous repair round has the following API problems. Fix each one, then re-output
the revised **complete code**.

## Detected problems
{issues_text}

## Original code (the fix scope; the output must stay aligned to this scope, keep every
other line byte-for-byte)
```python
{original}
```

## Your previously output fixed code (contains the above problems)
```python
{previous_fixed}
```

## Correction constraints
1. Modify only the lines directly related to the above problems; preserve every other
   line (comments, blank lines, indentation, function signatures) byte-for-byte.
2. Only use APIs that actually exist in this version; do not introduce a nonexistent
   API or misuse an API again.
3. For a nonexistent API: use an equivalent existing API or an implementation approach
   already used in the codebase.
4. For a misused API: fix the parameter types/count/semantics or the return handling
   per the suggestion.

## Output format
A one-sentence explanation outside the code block + a ```python code block
(the corrected complete code)
"""


def run_verification(original, fixed, system, opts=None):
    """Main API verification flow. Returns (final fixed_code, report text)."""
    opts = opts or {}
    client = DeepseekV4FlashClient()
    index = get_index(opts.get("tvm_root"))
    tvm_root = opts.get("tvm_root") or detect_tvm_root()
    max_rounds = int(opts.get("rounds", 2))
    no_review = bool(opts.get("no_review"))

    report = []
    report.append("# API Verification Report (autorepair)\n")
    report.append(f"- tvm source root: `{tvm_root}`")
    report.append(f"- verification method: existence check (static scan of the source "
                  f"tree + runtime inspect)"
                  f"{' + LLM misuse review' if not no_review else ' (LLM review skipped)'}")
    report.append(f"- correction round limit: {max_rounds}\n")

    cur = fixed
    done_ok = False
    for round_idx in range(max_rounds + 1):
        added = list(added_lines(original, cur))
        calls = extract_api_calls_from_lines(added) or extract_api_calls(cur)
        calls = sorted(set(calls))

        report.append(f"## Round {round_idx} check\n")
        if not calls:
            report.append("The added/changed lines introduce no new dotted API calls; "
                          "nothing to verify.\n")
            done_ok = True
            break

        results = [resolve(c, index) for c in calls]
        missing = []
        used = []
        report.append("| API | Result | Note |")
        report.append("|---|---|---|")
        for c, r in zip(calls, results):
            if r[0] == "ok":
                used.append(c)
                report.append(f"| `{c}` | ✓ exists | namespace `{r[1]}` |")
            elif r[0] == "missing":
                st, ns, leaf, info, close = r
                ns_size = len(info["names"]) if isinstance(info, dict) else 0
                close_txt = f"; similar: {', '.join(close)}" if close else ""
                missing.append((c, "missing", r))
                report.append(f"| `{c}` | ❌ **nonexistent** | namespace `{ns}` has "
                              f"{ns_size} APIs; `{leaf}` not found{close_txt} |")
            else:
                report.append(f"| `{c}` | ? unverifiable | namespace not indexed / "
                              f"intermediate level unknown |")

        verdicts = []
        if not no_review and used:
            verdicts, raw = review_usage(original, cur, used, index, client)
            report.append(f"\n### LLM misuse review ({len(used)} APIs)\n")
            for v in verdicts:
                ok = v.get("correct", True)
                mark = "✓ correct usage" if ok else "⚠ **suspected misuse**"
                report.append(f"- `{v['api']}` {mark}"
                              + (f": {v.get('issue', '')}" if v.get("issue") else "")
                              + (f"  suggestion: {v.get('suggestion', '')}"
                                 if v.get("suggestion") else ""))
        missing_apis = {c for c, _, _ in missing}
        misused = [
            (v["api"], "misuse", v)
            for v in verdicts
            if not v.get("correct", True) and v["api"] not in missing_apis
        ]

        issues = missing + misused
        if not issues:
            done_ok = True
            report.append("\n**Passed**: no missing or misused APIs.\n")
            break

        report.append(f"\n**Found {len(issues)} API problems; starting correction**\n")
        if round_idx >= max_rounds:
            report.append("(reached the correction round limit; keeping this round's "
                          "result)\n")
            break

        corr_prompt = build_correction_prompt(original, cur, issues)
        report.append("### Correction prompt\n```text\n" + corr_prompt[:2000] + "\n```\n")
        resp = client.model_prediction(corr_prompt, system=system)
        new = _extract_code(resp)
        if not new or new.strip() == cur.strip():
            report.append("The LLM correction did not change the code; stopping the "
                          "iteration.\n")
            break
        cur = new

    if not done_ok:
        report.append("Not fully passed in the end (missing/misused APIs remain or the "
                      "round limit was reached).\n")

    report_text = "\n".join(report)
    return cur, report_text


# ---------------------------------------------------------------- self-test


def self_test():
    root = detect_tvm_root()
    print(f"tvm source root: {root}")
    if not root:
        print("!! tvm source not found; cannot self-test")
        return 1
    idx = get_index(root)
    print(f"indexed namespaces: {len(idx)}")
    for ns in sorted(idx.keys()):
        print(f"  {ns:22s} APIs={len(idx[ns]['names'])}")

    cases = [
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
    print("\n== resolution self-test ==")
    fail = 0
    for name, expect in cases:
        r = resolve(name, idx)
        st = r[0]
        tag = "PASS" if st == expect else "FAIL"
        if st != expect:
            fail += 1
        print(f"  [{tag}] {name:22s} -> {st:12s} (expected {expect})")
    print(f"\n{len(cases) - fail}/{len(cases)} passed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        sys.exit(self_test())
