"""OpenVINO backend for the API verifier: static index, call extraction, resolution.

OpenVINO is C++ and its sources are not importable either, but they do declare
everything we need, so the index is built by **statically scanning declaration sites**:

* ``ov::op::vN::X``  -- one header per operator under ``src/core/include/openvino/op/``.
  The header's innermost ``namespace vN`` *is* the opset, which is what converters
  write after ``using namespace ov::op;``, so the namespace maps straight across.
* ``ov::op::util::X``, ``ov::core::X``, ``ov::runtime::X``, ...  -- the rest of
  ``src/core/include/openvino/``.
* frontend helpers -- ``src/frontends/{onnx,pytorch,paddle}`` and
  ``src/frontends/common_translators``. These are called unqualified
  (``normalize_axis(...)``) or scope-qualified
  (``common_translators::translate_atan2_util(...)``), so every function name found
  there also goes into a flat map used for bare-call lookup.

Calls on an object (``node.get_ov_inputs()``) are out of scope, the same way the TVM
backend skips ``bb.`` / ``attr.`` / ``cls.``.

Consumed by ``api_check.py``; see that module for the interface a backend must provide.
"""
import difflib
import os
import re

NAME = "OpenVINO"

ROOT_ENV = "OPENVINO_SRC_ROOT"
ROOT_HINT = "the root of an OpenVINO checkout (the directory containing src/frontends/)"
ROOT_MARKER = ("src", "core", "include", "openvino", "op")

REVIEW_SYSTEM = (
    "You are a meticulous OpenVINO frontend developer reviewing whether ov:: and "
    "frontend helper APIs are used correctly. Return only the JSON object."
)

# Namespaces whose enumeration is complete, so a missing leaf is authoritative.
# `ov.op.*` comes from one header per operator and the frontend helper namespaces
# from every source file under them; the aggregate namespaces (ov.core, ...) only
# carry what their headers declare, so a miss there degrades to unverifiable.
STRICT_NS_PREFIXES = ("ov.op", "ov.frontend.common_translators",
                      "ov.frontend.onnx", "ov.frontend.pytorch", "ov.frontend.paddle")

MAX_FILES = 4000          # guard against pointing ROOT_ENV at something huge

_SPECIFIERS = r"(?:template\s*<.*?>\s*)?(?:(?:static|inline|virtual|explicit|constexpr|extern)\s+)*"

# Anchored at line start, so call sites -- which follow '=', '(' or a return type they
# are not -- stay out.
_DECL_RE = re.compile(
    _SPECIFIERS
    + r"(?:[\w:]+(?:<[^<>]*>)?[\s*&]+)+"
    + r"([a-z_]\w*)\s*\("
)

_CLASS_RE = re.compile(
    r"^\s*(?:template\s*<.*?>\s*)?(?:class|struct)\s+(?:OPENVINO_API\s+)?([A-Z]\w*)"
)

_NAMESPACE_OPEN_RE = re.compile(r"^\s*namespace\s+([A-Za-z_]\w*)\s*\{")

_USING_RE = re.compile(r"^\s*using\s+([A-Za-z_]\w*)\s*=")
_NAMESPACE_CLOSE_RE = re.compile(r"^\s*\}\s*(?://.*)?$")

_CPP_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "alignof", "decltype",
    "static_cast", "dynamic_cast", "const_cast", "reinterpret_cast", "new", "delete",
    "throw", "assert", "defined", "operator", "typeid", "noexcept", "and", "or", "not",
}

SELF_TEST_CASES = [
    ("ov::op::v1::Add", "ok"),
    ("ov::op::v0::Constant", "ok"),
    ("ov::op::v1::Foo", "missing"),
    ("v1::Add", "ok"),
    ("v9::NonexistentOp", "missing"),
    ("common_translators::translate_atan2_util", "ok"),
    ("common_translators::not_a_real_helper", "missing"),
    ("normalize_axis", "ok"),
    ("definitely_not_a_helper_name", "missing"),
    ("std::vector", "unverifiable"),
]

# ---------------------------------------------------------------- root


def detect_root():
    """The OpenVINO checkout root, or None."""
    env = os.environ.get(ROOT_ENV)
    if env and os.path.isdir(os.path.join(env, *ROOT_MARKER)):
        return env
    candidate = "[your openvino source root]"
    if os.path.isdir(os.path.join(candidate, *ROOT_MARKER)):
        return candidate
    return None


# ---------------------------------------------------------------- declaration scan


def _iter_files(*dirs, exts=(".hpp", ".h", ".cpp")):
    """Yield source files under `dirs`, deterministic order, with a global cap."""
    seen = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, subdirs, files in os.walk(d):
            subdirs[:] = sorted(s for s in subdirs if s not in {"tests", "__pycache__"})
            for f in sorted(files):
                if not f.endswith(exts):
                    continue
                p = os.path.join(root, f)
                if p in seen:
                    continue
                seen.add(p)
                yield p
                if len(seen) >= MAX_FILES:
                    return


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def scan_declarations(path):
    """Return a list of (namespace tuple, name) for every declaration in one file.

    The namespace is tracked by **brace depth**, not by looking for a closing line: a
    bare `}` ends any function or class body too, so treating every one of them as a
    namespace close empties the stack part-way through a file and misfiles everything
    after it. Names are recorded under the namespace they actually declare into, which
    is not always the one the directory suggests -- a frontend header may legitimately
    declare into ``ov::op::util``.

    Declarations spanning several lines (a parameter list broken across lines, say) are
    joined before matching, because the closing `;` is what marks the end and the head
    of the first line is all the regex needs.
    """
    records = []
    stack = []          # (namespace, brace depth it was opened at)
    depth = 0
    buf = ""

    for line in _read_code(path).split("\n"):
        m = _NAMESPACE_OPEN_RE.match(line)
        if m:
            buf = ""
            stack.append((m.group(1), depth))
            depth += line.count("{") - line.count("}")
            continue

        depth += line.count("{") - line.count("}")
        while stack and stack[-1][1] >= depth:
            stack.pop()

        stripped = line.strip()
        if not stripped:
            buf = ""
            continue
        buf = f"{buf} {stripped}" if buf else stripped
        if not buf.endswith((";", "{")):
            if len(buf) > 800:      # not a declaration after all; do not grow forever
                buf = ""
            continue

        ns = tuple(name for name, _ in stack)
        m = _CLASS_RE.match(buf)
        if m:
            records.append((ns, m.group(1)))
        m = _USING_RE.match(buf)
        if m:
            records.append((ns, m.group(1)))
        m = _DECL_RE.match(buf)
        if m and m.group(1) not in _CPP_KEYWORDS:
            records.append((ns, m.group(1)))
        buf = ""

    return records


# ---------------------------------------------------------------- index


def build_index(root):
    """Build the OpenVINO API index.

    Returns {"namespaces": {ns: {"names", "src", "dir"}}, "bare": {name: path}}.

    Every declaration is filed under the namespace it declares itself into, so a name
    is reachable by the same qualified path the converters write. Declarations with no
    enclosing namespace (file-local statics) only land in `bare`, which is what
    unqualified calls are resolved against.
    """
    core = os.path.join(root, "src", "core", "include", "openvino")
    frontends = os.path.join(root, "src", "frontends")

    namespaces = {}
    bare = {}

    scan_dirs = [os.path.join(core, "op"), core]
    scan_dirs += [os.path.join(frontends, f)
                  for f in ("onnx", "pytorch", "paddle", "common_translators")]

    for d in scan_dirs:
        for path in _iter_files(d, exts=(".hpp", ".h")):
            for ns_tuple, name in scan_declarations(path):
                bare.setdefault(name, path)
                if not ns_tuple:
                    continue
                ns_name = ".".join(ns_tuple)
                rec = namespaces.setdefault(
                    ns_name, {"names": set(), "src": {}, "dir": d})
                rec["names"].add(name)
                rec["src"].setdefault(name, path)

    # Implementation files matter for the flat map even when they declare nothing new:
    # a helper defined in a .cpp is still a legitimate bare call.
    for d in scan_dirs:
        for path in _iter_files(d, exts=(".cpp",)):
            for _ns_tuple, name in scan_declarations(path):
                bare.setdefault(name, path)

    # `ov.op` aggregates every opset: a call naming an opset that has no namespace of
    # its own (`v9::Foo`) is then judged against the whole operator set instead of
    # being reported missing just because that opset does not exist.
    op_root = namespaces.setdefault("ov.op", {"names": set(), "src": {}, "dir": os.path.join(core, "op")})
    for ns_name, rec in namespaces.items():
        if ns_name.startswith("ov.op."):
            op_root["names"] |= rec["names"]
            for name, path in rec["src"].items():
                op_root["src"].setdefault(name, path)

    return {"namespaces": namespaces, "bare": bare}


def describe_index(index):
    """One-line summary for the report and the self-test."""
    ns = index.get("namespaces", {})
    total = sum(len(v["names"]) for v in ns.values())
    return f"{len(ns)} namespaces, {total} names, {len(index.get('bare', {}))} bare"


# ---------------------------------------------------------------- resolution

_OPSET_RE = re.compile(r"v\d+")

_SCOPED_CALL_RE = re.compile(r"(?<![\w:])((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)\s*\(")

# Ops are usually constructed through a factory rather than called, so the type
# argument of make_shared<...> is where a fabricated op class shows up.
_MAKE_SHARED_RE = re.compile(
    r"\bmake_shared\s*<\s*((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)\s*[>,]")
_BARE_CALL_RE = re.compile(r"(?<![\w:.>])([a-z_]\w*)\s*\(")

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')

# Language types and keywords: `int(x)` is a functional cast, not an API call.
_NON_API_NAMES = _CPP_KEYWORDS | {
    "int", "long", "short", "char", "bool", "float", "double", "void", "unsigned",
    "signed", "size_t", "auto", "const", "static", "wchar_t", "uint8_t", "uint16_t",
    "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t", "string",
}


def strip_comments(code):
    """Remove C++ comments so English prose in them is not read as a call."""
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", code))


def _read_code(path):
    """File text with comments and string literals neutralised, line count preserved.

    Brace counting drives the namespace tracking, so a `{` inside a comment or a string
    must not be counted, and a block comment must not glue two lines together.
    """
    text = _BLOCK_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), _read(path))
    text = _LINE_COMMENT_RE.sub("", text)
    return _STRING_RE.sub('""', text)


def _lookup(name, index):
    """Resolve a dotted name (`ov.op.v1.Add`) against the indexed namespaces.

    Returns (status, ns, leaf, info, close) or None when no namespace matches.
    """
    ns_map = index["namespaces"]
    ns_list = sorted(ns_map.keys(), key=len, reverse=True)
    for ns in ns_list:
        if not name.startswith(ns + "."):
            continue
        rest = name[len(ns) + 1:]
        if "." in rest:
            return None
        info = ns_map[ns]
        if rest in info["names"]:
            return ("ok", ns, rest, info, None)
        close = difflib.get_close_matches(rest, sorted(info["names"]), n=3)
        if ns.startswith(STRICT_NS_PREFIXES):
            return ("missing", ns, rest, info, close)
        return ("unverifiable", ns, rest, info, None)
    return None


def _resolve_qualified(dotted, index):
    """Resolve a `::`-qualified call, or say it cannot be resolved.

    Only the forms a converter actually writes are expanded: a fully qualified
    `ov::...` path, an opset-relative `vN::X`, and `common_translators::X`. Anything
    qualified by a short name (`detail::conv(`, `internal::X(`, `common::X(`) is
    relative to the namespace of the file the snippet came from, which the verifier
    never sees; expanding it would land in a same-named namespace somewhere else and
    report a real API as fabricated, so it answers unverifiable instead.

    A trailing segment may be a static member call (`ov::op::v0::Constant::create`), so
    successively shorter prefixes are tried: the class the call hangs off has to exist
    even when the member itself is not indexed.
    """
    parts = dotted.split(".")
    first = parts[0]

    if first == "ov":
        candidates = [dotted]
    elif _OPSET_RE.fullmatch(first):
        candidates = ["ov.op." + dotted] + ["ov.op." + ".".join(parts[:cut])
                                            for cut in range(len(parts) - 1, 0, -1)]
    elif first == "common_translators":
        candidates = ["ov.frontend.common_translators." + ".".join(parts[1:])]
    else:
        return ("unverifiable", None, parts[-1], None, None)

    for cand in candidates:
        hit = _lookup(cand, index)
        if hit is not None and hit[0] == "ok":
            return hit
    for cand in candidates:
        hit = _lookup(cand, index)
        if hit is not None:
            return hit
    return ("unverifiable", None, parts[-1], None, None)


def _defined_in(code, name):
    """True when `name` is declared or assigned inside `code` itself.

    A call to a helper the fix defines locally is not a fabricated API. The token in
    front of the name has to look like a return type or a declaration head, and a
    statement keyword in that position (`return foo(`) does not count as one.
    """
    esc = re.escape(name)
    head = (r"(?:auto|void|bool|int|long|short|unsigned|signed|float|double|char|"
            r"size_t|static|const|inline|explicit|constexpr)\b")
    looks_declared = re.search(rf"\b{head}\s+{esc}\s*[=(]", code) or re.search(
        rf"\b[A-Za-z_]\w*(?:::\w+)*(?:<[^<>]*>)?\s+{esc}\s*\(", code)
    if not looks_declared:
        return False
    return not re.search(
        rf"\b(?:return|if|while|for|switch|case|throw|delete|sizeof|assert|"
        rf"and|or|not)\s+{esc}\s*[=(]", code)


def resolve(name, index):
    """Resolve a call name. Returns (status, ns, leaf, info, close).

    A `::`-qualified call is resolved against the namespaces; a bare call is resolved
    against the flat name map the scan produced. Bare names that the fixed code itself
    defines are not API uses and are not reported.
    """
    if "::" in name:
        dotted = name.replace("::", ".")
        hit = _resolve_qualified(dotted, index)
        if hit is not None:
            return hit
        return ("unverifiable", None, dotted.split(".")[-1], None, None)

    bare = index.get("bare", {})
    if name in bare:
        info = {"names": {name}, "src": {name: bare[name]}, "dir": os.path.dirname(bare[name])}
        return ("ok", "(frontend)", name, info, None)
    if name in _NON_API_NAMES or name.startswith("_"):
        return ("unverifiable", None, name, None, None)
    close = difflib.get_close_matches(name, sorted(bare), n=3) if bare else []
    if bare:
        # The index was built, so the name is not declared anywhere in the checkouts we
        # scanned. In these frontends an unqualified call is a frontend helper rather
        # than standard library (which is written `std::...`), so this is a fabrication.
        return ("missing", "(frontend)", name, None, close)
    return ("unverifiable", None, name, None, None)


# ---------------------------------------------------------------- call extraction


def extract_calls(code, index=None):
    """Extract API call names from C++ code.

    Two shapes are collected: `::`-qualified calls (`ov::op::v1::Add(`, `v1::Add(`,
    `common_translators::translate_atan2_util(`), and bare calls to names the index
    knows about. Method calls (`node.get_ov_inputs()`) are skipped, as are names the
    snippet defines itself.
    """
    code = strip_comments(code)
    out = []
    for m in _SCOPED_CALL_RE.finditer(code):
        out.append(m.group(1))
    for m in _MAKE_SHARED_RE.finditer(code):
        out.append(m.group(1))
    for m in _BARE_CALL_RE.finditer(code):
        name = m.group(1)
        if name in _NON_API_NAMES or name.startswith("_"):
            continue
        if _defined_in(code, name):
            continue
        # Emitted whether or not the index knows it: resolve() is what decides between
        # "exists", "fabricated" and "cannot tell".
        out.append(name)
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


# ---------------------------------------------------------------- API docs


def _extract_decl_block(path, name):
    """Return the declaration of `name` plus its preceding comment block, or None."""
    if not os.path.isfile(path):
        return None
    lines = _read(path).split("\n")
    for i, line in enumerate(lines):
        if not re.search(r"\b" + re.escape(name) + r"\s*\(", line):
            continue
        if not (line.strip().endswith(";") or line.strip().endswith("{")):
            continue
        start = i
        j = i - 1
        while j >= 0 and (lines[j].strip().startswith(("//", "*", "/*"))
                          or not lines[j].strip()
                          or lines[j].strip().startswith("template")):
            start = j
            j -= 1
        return "\n".join(lines[start:i + 1]).strip()
    return None


def describe(name, index):
    """Return (declaration, docstring, source location) for an existing API.

    The API brief handed to the review LLM is the C++ declaration plus the comment
    block directly above it; C++ has no docstring object to read.
    """
    st, ns, leaf, info, _close = resolve(name, index)
    if st != "ok":
        return (None, None, st)

    path = index.get("bare", {}).get(name)
    if path is None and isinstance(info, dict):
        path = (info.get("src") or {}).get(leaf)
    if not path:
        return (None, None, "static-no-decl")

    decl = _extract_decl_block(path, leaf)
    if decl is None:
        return (None, None, "static-no-decl")
    try:
        rel = os.path.relpath(path)
    except ValueError:
        rel = path
    return (decl, "", rel)
