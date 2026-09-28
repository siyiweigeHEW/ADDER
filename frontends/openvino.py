"""OpenVINO frontend rules (C++ converter implementations).

OpenVINO converter implementations are free functions and classes spread over
``.cpp`` files. The operator-to-implementation mapping is implicit in the file
and function names, so extraction is a directory scan and a dependency is a
function or class the body delegates to.
"""
import os
import re

NAME = "OpenVINO"
CODE_LANGUAGE = "cpp"
COMMENT_PREFIX = "//"
NAME_SEPARATOR = "::"
TARGET_FRONTS = ("onnx", "torch", "paddle")

# A dependency is looked up first in the headers that define it.
FILE_EXTENSION_GROUPS = ((".cpp",), (".hpp", ".h"))

SOURCE_ROOT_ENV = "OPENVINO_SRC_ROOT"

ANALYSIS_RULES = """1. Check whether the function body calls functions/classes defined in other files.
2. Pay special attention to: a function that directly returns the result of another function (e.g., `return reverse_op(node);`) and calls to helper functions in other files.
3. Do NOT list standard-library functions (with the std:: prefix) or OpenVINO framework built-in classes (with the ov:: prefix, NodeContext, etc.).
4. The external dependencies to list may be defined in: other files in the same operator directory (src/op), utils.cpp/utils.hpp under the frontend src directory (e.g., get_inputs_with_promoted_types, get_shape_rank, normalize_axis, make_list_construct, etc.), or the shared directory common_translators (e.g., common_translators::translate_atan2_util).
5. If a function is just a thin wrapper whose real logic lives in some shared utility function, be sure to list that utility function as a dependency."""

PROMPT_VALUES = {
    "ToolStack": "OpenVINO",
    "CodeLanguage": CODE_LANGUAGE,
    "OperatorNoun": "operator code",
    "AnalysisRules": ANALYSIS_RULES,
    "SharedHelperExamples": "common_translators::xxx, utils::xxx, etc.",
}


# --------------------------------------------------------------------------
# API verification profile
# --------------------------------------------------------------------------
# Read by autorepair/api_check.py, so that the verifier itself names no tool
# stack. Fill in the fields below for your own backend; nothing is pre-filled,
# because which namespaces are worth indexing depends on the tree you audit.

API_VERIFY = {
    # Tool stack name, used in the verification report and the review prompt.
    "tool_stack": "OpenVINO",

    # Where the API source tree is. `root_env` is the environment variable that
    # points at it; `root_marker` is a path relative to that root whose presence
    # confirms it is the right tree, e.g. ("pkg", "sub", "__init__.py").
    # Set `root_marker` to None if the backend has no Python package tree to
    # scan: the static existence check is then skipped, and only the runtime
    # check and the LLM review run.
    "root_env": "OPENVINO_SRC_ROOT",
    "root_marker": None,

    # Namespaces to index, as (namespace, directory relative to the root) pairs.
    # For each one the verifier collects the API names that the package's
    # __init__.py re-exports, plus the defs in that package's own modules.
    # Example: [("pkg.op", "pkg/op"), ("pkg", "pkg")].
    # Leave empty to disable the static existence check.
    "namespaces": [],

    # Subset of `namespaces` whose enumeration is complete. A leaf missing from
    # one of these is reported as a fabrication; a leaf missing anywhere else
    # degrades to "unverifiable", so a partially enumerated namespace cannot
    # produce a false positive. Leave empty to never report a fabrication.
    "strict_namespaces": set(),

    # Extra roots to resolve dotted calls under, beyond the runtime modules the
    # verifier already handles (numpy, functools, math, ...).
    "extra_call_roots": set(),

    # (api_name, expected_status) pairs exercised by `api_check.py --self-test`.
    # expected_status is one of "ok" / "missing" / "unverifiable".
    "self_test_cases": [],
}


# Relative to the checkout root: the directory holding each frontend's
# operator implementations.
_FRONT_DIRS = {
    "onnx": os.path.join("src", "frontends", "onnx", "frontend", "src", "op"),
    "torch": os.path.join("src", "frontends", "pytorch", "src", "op"),
    "paddle": os.path.join("src", "frontends", "paddle", "src", "op"),
}


def source_root():
    """Root of the OpenVINO checkout."""
    root = os.environ.get(SOURCE_ROOT_ENV)
    if not root:
        raise SystemExit(
            f"{SOURCE_ROOT_ENV} is not set. Point it at the root of an OpenVINO "
            f"checkout (the directory containing src/frontends/)."
        )
    return root


def front_dirs(target_fronts=None):
    """Map frontend name -> directory holding its operator implementations."""
    fronts = target_fronts or TARGET_FRONTS
    root = source_root()
    return {f: os.path.join(root, _FRONT_DIRS[f]) for f in fronts if f in _FRONT_DIRS}


def get_all_front_converters(target_fronts=None):
    """Extract every converter body, keyed by operator name.

    Returns ``{front_name: (code_body_dict, name_mapping_dict)}`` where both
    dicts map an operator identity to its source text and to itself (C++ has no
    separate mapping indirection, unlike the Python frontends).
    """
    all_front_converter = {}
    for front_name, front_path in front_dirs(target_fronts).items():
        if not os.path.isdir(front_path):
            print(f"[SKIP] Frontend '{front_name}' directory not found: {front_path}")
            continue

        print("*" * 30 + f"  {front_name}  " + "*" * 30)

        code_body_dict = {}
        for root, _, files in os.walk(front_path):
            for file in files:
                if not file.endswith(".cpp"):
                    continue
                op_identity = os.path.splitext(file)[0].lower()
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        code_body_dict[op_identity] = f.read()
                except OSError:
                    continue

        if code_body_dict:
            all_front_converter[front_name] = (code_body_dict, dict(code_body_dict))
            print(f"  [OK] Extracted {front_name} frontend: {len(code_body_dict)} ops")

    return all_front_converter


def dependency_search_dirs(front_op_dir):
    """Directories to search for dependency code, in priority order.

    Helpers frequently live outside the operator directory, so the search is
    widened to the frontend src root and the cross-frontend
    ``common_translators`` shared directory. Restricting the search to the
    operator directory makes converters that delegate to a shared helper look
    "broken" and produces false Bug verdicts.
    """
    dirs = []
    seen = set()

    def add(d):
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            dirs.append(d)

    add(front_op_dir)
    parent = os.path.dirname(os.path.abspath(front_op_dir))
    add(parent)
    add(os.path.join(parent, "utils"))
    add(os.path.join(parent, "core"))
    add(os.path.join(parent, "common"))

    parts = os.path.abspath(front_op_dir).split(os.sep)
    if "frontends" in parts:
        frontends_root = os.sep.join(parts[: parts.index("frontends") + 1])
        common_dir = os.path.join(frontends_root, "common_translators")
        add(common_dir)
        add(os.path.join(common_dir, "src"))
        add(os.path.join(common_dir, "include"))

    return dirs


# Token that, in the line preceding a name, marks it as a call site rather than
# a definition.
_CALL_CONTEXT_TOKENS = (
    "return", "=", "(", ",", ":", "->", "&&", "||", "!", "emit",
    "{", "[", "?", ".", "new", "delete",
)


def _find_definition_line(lines, clean_name):
    """Return the 0-based index of the line defining ``clean_name``, or None.

    Distinguishes a definition from a call site. A name that starts a line is
    only a definition when the previous code line is a dangling return type
    (template types are commonly split across two lines).
    """
    for i, line in enumerate(lines):
        m = re.search(rf"\b{re.escape(clean_name)}\s*\(", line)
        if not m:
            continue
        before = line[: m.start()].strip()
        if before == "":
            j = i - 1
            while j >= 0:
                prev = lines[j].strip()
                if prev and not prev.startswith(("//", "/*", "*", "#")):
                    break
                j -= 1
            prev = lines[j].strip() if j >= 0 else ""
            if prev and not prev.endswith(("{", ";", "return", "=", "(", ",", "->", ")")):
                if re.search(r"\w[>\])]?$", prev):
                    return i
            continue
        last_tok = before.split()[-1]
        if last_tok in _CALL_CONTEXT_TOKENS:
            continue
        if before.endswith(("return", "=", "(", ",", ":", "->", "{", "[", "?")):
            continue
        return i
    return None


def _extract_braced_block(lines, start):
    """Return the source block from ``start`` through its balanced closing brace."""
    # Back up over a dangling return-type line (e.g. 'std::tuple<...>' alone).
    while start > 0:
        prev = lines[start - 1].strip()
        if prev and not prev.startswith(("//", "/*", "*", "#")) and \
           not prev.endswith(("{", ";", "}")) and re.search(r"\w[>\])]?$", prev):
            start -= 1
        else:
            break
    depth = 0
    started = False
    end = start
    for j in range(start, min(start + 800, len(lines))):
        sline = re.sub(r"//.*", "", lines[j])
        prev_depth = depth
        depth += sline.count("{") - sline.count("}")
        if depth > 0:
            started = True
        if started and depth <= 0 and prev_depth > 0:
            end = j
            break
        end = j
    return "\n".join(lines[start:end + 1])


def extract_dependency_snippet(content, clean_name):
    """Return a bounded snippet of the function/class defining ``clean_name``.

    A snippet rather than the whole file keeps the expanded prompt small enough
    to stay under the token limit.
    """
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if re.match(rf"\s*(?:class|struct)\s+{re.escape(clean_name)}\b", line):
            return _extract_braced_block(lines, i)
    idx = _find_definition_line(lines, clean_name)
    if idx is None:
        return None
    return _extract_braced_block(lines, idx)
