"""TVM Relax frontend rules (Python converter implementations).

Unlike the C++ frontends, a TVM frontend declares an explicit mapping from
operator name to the function or class that converts it (``convert_map``). The
mapping is parsed out of the source and used to select the converter bodies, so
extraction needs both a regex for the mapping literal and an AST pass for the
definitions it names.
"""
import ast
import os
import re

NAME = "TVM"
CODE_LANGUAGE = "python"
COMMENT_PREFIX = "#"
NAME_SEPARATOR = "."
TARGET_FRONTS = ("onnx", "torch")

FILE_EXTENSION_GROUPS = ((".py",),)

SOURCE_ROOT_ENV = "TVM_RELAX_FRONTEND"

ANALYSIS_RULES = """1. If it is a class definition (class X(BaseY)), check whether the base class BaseY needs to be expanded; if BaseY itself inherits from other classes, list them as well.
2. Check whether the function/class body calls other helper functions or classes defined within the same package (excluding built-in functions and TVM framework functions).
3. Do NOT list standard-library modules (os, math, typing, etc.) or TVM built-in modules (relax.op, tvm, tir, etc.).
4. The external dependencies to list may be defined in: other files in the same frontend directory (e.g., onnx_frontend.py), or shared modules under the frontend root relax/frontend (e.g., utility functions in common.py).
5. If a function/class is just a thin wrapper whose real logic lives in some shared utility function or base class, be sure to list that utility function/base class as a dependency.
6. Only list base class names or helper function names essential for understanding the core logic."""

PROMPT_VALUES = {
    "ToolStack": "TVM Relax",
    "CodeLanguage": CODE_LANGUAGE,
    "OperatorNoun": "operator conversion code",
    "AnalysisRules": ANALYSIS_RULES,
    "SharedHelperExamples": "utility functions in common.py under relax/frontend, base classes in onnx_frontend.py, etc.",
}


# --------------------------------------------------------------------------
# API verification profile
# --------------------------------------------------------------------------
# Read by autorepair/api_check.py, so that the verifier itself names no tool
# stack. Fill in the fields below for your own backend; nothing is pre-filled,
# because which namespaces are worth indexing depends on the tree you audit.

API_VERIFY = {
    # Tool stack name, used in the verification report and the review prompt.
    "tool_stack": "TVM",

    # Where the API source tree is. `root_env` is the environment variable that
    # points at it; `root_marker` is a path relative to that root whose presence
    # confirms it is the right tree, e.g. ("pkg", "sub", "__init__.py").
    # Set `root_marker` to None if the backend has no Python package tree to
    # scan: the static existence check is then skipped, and only the runtime
    # check and the LLM review run.
    "root_env": "TVM_PYTHON_ROOT",
    "root_marker": ("relax", "op", "__init__.py"),

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


# Frontends present in the TVM tree but not audited by this pipeline.
_SKIP_FRONTS = {"darknet", "caffe2", "qnn_torch", "tensorflow2_ops"}


def source_root():
    """The ``python/tvm/relax/frontend`` directory of a TVM checkout."""
    root = os.environ.get(SOURCE_ROOT_ENV)
    if not root:
        raise SystemExit(
            f"{SOURCE_ROOT_ENV} is not set. Point it at the python/tvm/relax/frontend "
            f"directory of a TVM checkout."
        )
    return root


# --------------------------------------------------------------------------
# convert_map parsing
# --------------------------------------------------------------------------

def preprocess_api_map(ori_map, flag):
    """Normalise a ``convert_map`` literal so it can be evaluated as a dict.

    Keys and values are frequently bare names (``Add``, ``nn.Linear``) rather
    than string literals, and torch values are converter factories of the form
    ``X.get_converter``. Both are rewritten to quoted strings.
    """
    final_map = "{"
    match_content = re.search(r"\{(.*)\}", ori_map, re.DOTALL)
    content = match_content.group(1) if match_content else ori_map

    for line in content.split("\n"):
        line = line.strip()
        if not line or ": " not in line:
            continue

        line = line.replace("self.", "")
        parts = line.split(": ")
        dll_name = parts[0].strip()
        dlc_name = parts[1].strip().rstrip(",")

        if not (dll_name.startswith("'") or dll_name.startswith('"')):
            dll_name = f"'{dll_name}'"

        if not (dlc_name.startswith("'") or dlc_name.startswith('"')):
            if flag == "func":
                dlc_name = dlc_name.split(".get_converter")[0]
            dlc_name = f"'{dlc_name}'"

        final_map += f"{dll_name}: {dlc_name},\n"

    final_map += "}"
    return final_map


def get_convert_map(source_code):
    """Return ``(normalised_map_literal, flag)`` for the first mapping found.

    Handles three shapes: a ``convert_map`` assignment, a ``return { ... }``
    inside a converter-factory function (torch), and a ``get_convert_map``
    function definition. ``flag`` is ``'func'`` when the values are converter
    factories rather than implementations.
    """
    patterns = [
        r"_?convert_map\s*=\s*\{.*?\}",
        r"return\s+\{.*?\}",
    ]
    for p in patterns:
        matches = re.findall(p, source_code, flags=re.DOTALL)
        if matches:
            target_str = matches[-1]
            flag = "func" if "get_converter" in target_str else "map"
            return preprocess_api_map(target_str, flag), flag

    pattern_func = r"def\s+_?get_convert_map\(.*?\}\n"
    match_func = re.findall(pattern_func, source_code, flags=re.DOTALL | re.IGNORECASE)
    if match_func:
        return preprocess_api_map(match_func[-1], "func"), "func"

    return None, None


def _collect_definitions(source_code, api_dict):
    """Map every name referenced by ``api_dict`` to its source segment.

    Both keys and values are collected, because torch keeps the implementation
    name in the value.
    """
    valid_names = set()
    for k, v in api_dict.items():
        valid_names.add(str(k))
        valid_names.add(str(v))

    op_converter = {}
    for node in ast.walk(ast.parse(source_code)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in valid_names:
            op_converter[node.name] = ast.get_source_segment(source_code, node)
    return op_converter


def _read_sources(path):
    """Concatenate the Python sources under ``path`` (a file or a directory)."""
    if os.path.isdir(path):
        chunks = []
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.endswith(".py"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        chunks.append(f.read())
        return "\n".join(chunks)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _fronts_in(base_dir):
    """Map frontend name -> path for every entry of the frontend root."""
    if not os.path.exists(base_dir):
        return {}
    fronts = {}
    for entry in os.listdir(base_dir):
        if entry.startswith(".") or entry == "__pycache__":
            continue
        name, _ = os.path.splitext(entry)
        fronts[name] = os.path.join(base_dir, entry)
    return fronts


# The torch frontend's mapping lives in fx_translator.py, but its converters
# inherit from base_fx_graph_translator.py, so both are read. They are read in
# that order so the mapping literal is taken from fx_translator.py.
_TORCH_SOURCES = ("base_fx_graph_translator.py", "fx_translator.py")


def _extract_front(path, front):
    """Return ``(function_dict, api_convert_map)`` for one frontend."""
    if front == "torch":
        sources = []
        for filename in _TORCH_SOURCES:
            candidate = os.path.join(path, filename)
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    sources.append(f.read())
        source_code = "\n".join(sources)
    else:
        source_code = _read_sources(path)

    api_map_str, _ = get_convert_map(source_code)
    if not api_map_str:
        print(f"[SKIP] No convert_map found in: {path}")
        return None, None

    try:
        api_dict = eval(api_map_str)
    except Exception as e:
        print(f"[ERROR] Could not evaluate convert_map for {path}: {e}")
        return None, None

    try:
        op_converter = _collect_definitions(source_code, api_dict)
    except SyntaxError as e:
        print(f"[ERROR] AST parse failed for {path}: {e}")
        return None, None

    return op_converter, api_dict


def get_all_front_converters(target_fronts=None):
    """Extract converter bodies for each selected frontend.

    Returns ``{front_name: (code_body_dict, api_map_dict)}``.
    """
    targets = target_fronts or TARGET_FRONTS
    all_front_converter = {}
    for front, path in _fronts_in(source_root()).items():
        if front in _SKIP_FRONTS or front not in targets:
            continue

        print("*" * 30 + f"  {front}  " + "*" * 30)
        function_dict, api_convert_map = _extract_front(path, front)
        if function_dict:
            all_front_converter[front] = [function_dict, api_convert_map]

    return all_front_converter


# --------------------------------------------------------------------------
# Dependency search
# --------------------------------------------------------------------------

def dependency_search_dirs(front_dir):
    """Directories to search for dependency code, in priority order.

    Widened beyond the operator directory to the frontend root (where common.py
    and shared helpers live) and sibling utility packages.
    """
    dirs = []
    seen = set()

    def add(d):
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            dirs.append(d)

    add(front_dir)
    parent = os.path.dirname(os.path.abspath(front_dir))
    add(parent)
    add(os.path.join(parent, "utils"))
    add(os.path.join(parent, "core"))
    add(os.path.join(parent, "common"))

    return dirs


def extract_dependency_snippet(content, clean_name):
    """Return the source segment defining ``clean_name``, or None.

    The Python counterpart of brace matching: class definitions, function
    definitions and module-level assignments (e.g. ``np_add = _np.add``).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == clean_name:
            return ast.get_source_segment(content, node)
        if isinstance(node, ast.FunctionDef) and node.name == clean_name:
            return ast.get_source_segment(content, node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == clean_name:
                    return ast.get_source_segment(content, node)
    return None
