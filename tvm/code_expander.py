"""
Code expansion module for TVM.
Analyzes operator converter code to determine if external helper code is needed
for complete semantic understanding, then retrieves and appends it.

Mirrors OPENVINO's code_expander.py approach, adapted for TVM's Python source:
  * Directory-based search (frontend op dir + frontend root + shared dirs)
    instead of a fixed file list, so helpers living outside the op dir are found.
  * AST-based definition extraction (Python's analog of C++ brace matching),
    returning a bounded source segment (exact class/function/assignment).
  * visited_paths tracking so a file is read at most once for multiple deps.

Example:
    class Add(BinaryBase):
        @classmethod
        def _impl_v1(cls, bb, inputs, attr, params):
            return cls.base_impl(bb, inputs, attr, params)

    This module detects BinaryBase is defined externally and appends its code
    so the LLM can perform a complete semantic comparison.
"""
import ast
import os
from llm_client import global_model


def get_dependency_search_dirs(front_dir):
    """Candidate dirs to search for dependency code, ordered by priority.

    Includes the frontend op dir itself, its parent (the frontend root, where
    common.py and shared helpers live), and sibling utils/core/common dirs.
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
    add(os.path.join(parent, 'utils'))
    add(os.path.join(parent, 'core'))
    add(os.path.join(parent, 'common'))

    return dirs


def analyze_dependencies_with_llm(code_body, front_name):
    """
    Use LLM to determine which external helper classes/functions are needed.
    Returns list of dependency names (empty if none needed).
    """
    prompt = f"""You are a TVM Relax compiler frontend code analysis expert. Please analyze whether the operator conversion code of the following {front_name} frontend requires helper code defined in other files to gain complete semantics:

```python
{code_body}
```

[Analysis Rules]
1. If it is a class definition (class X(BaseY)), check whether the base class BaseY needs to be expanded; if BaseY itself inherits from other classes, list them as well.
2. Check whether the function/class body calls other helper functions or classes defined within the same package (excluding built-in functions and TVM framework functions).
3. Do NOT list standard-library modules (os, math, typing, etc.) or TVM built-in modules (relax.op, tvm, tir, etc.).
4. The external dependencies to list may be defined in: other files in the same frontend directory (e.g., onnx_frontend.py), or shared modules under the frontend root relax/frontend (e.g., utility functions in common.py).
5. If a function/class is just a thin wrapper whose real logic lives in some shared utility function or base class, be sure to list that utility function/base class as a dependency.
6. Only list base class names or helper function names essential for understanding the core logic.

[Output Format]
If there are key external dependencies, output one line: `[DEPS]: dep_name1, dep_name2`
If not needed, output: `[DEPS]: NONE`
"""
    try:
        response = global_model.model_prediction(prompt)
        for line in response.split('\n'):
            if '[DEPS]:' in line:
                deps_str = line.split('[DEPS]:')[1].strip()
                if deps_str.upper() == 'NONE':
                    return []
                deps = [d.strip() for d in deps_str.split(',') if d.strip()]
                return deps
        return []
    except Exception as e:
        print(f"  [WARN] Dependency analysis LLM call failed: {e}")
        return []


def _extract_dependency_snippet(content, clean_name):
    """Return the source segment defining `clean_name`.

    Handles class definitions, function definitions, and module-level
    assignments (e.g. np_add = _np.add). Returns None if not found.
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


def search_dependency_code(dep_name, front_dir, visited_paths=None):
    """
    Search for dependency definition across the expanded dir set.
    Searches .py files only, in dir priority order. Returns a bounded snippet
    of the matching class/function/assignment (not the whole file).
    `visited_paths`: set of already-appended file paths; matching files there
    are skipped to avoid re-reading the same file for two deps.

    Returns (snippet, full_path) or (None, None).
    """
    clean_name = dep_name.split('.')[-1]
    visited_paths = visited_paths if visited_paths is not None else set()
    search_dirs = get_dependency_search_dirs(front_dir)

    for root in search_dirs:
        for dirpath, _, files in os.walk(root):
            for file in sorted(files):
                if not file.endswith('.py'):
                    continue
                full_path = os.path.join(dirpath, file)
                if full_path in visited_paths:
                    continue
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception:
                    continue
                snippet = _extract_dependency_snippet(content, clean_name)
                if snippet:
                    return snippet, full_path
    return None, None


def expand_code_body(code_body, front_name, front_dir):
    """
    Analyze and expand code body with dependencies.

    Args:
        code_body: Extracted Python function/class body
        front_name: 'onnx' or 'torch'
        front_dir: Path to the frontend op directory (e.g. relax/frontend/onnx)

    Returns:
        expanded_code: Original code + dependency code (if any)
        dep_info:     Comma-separated dependency names (empty string if none)
    """
    dependency_names = analyze_dependencies_with_llm(code_body, front_name)

    if not dependency_names:
        return code_body, ""

    dep_codes = []
    found_deps = []
    visited_paths = set()

    for dep_name in dependency_names:
        dep_code, full_path = search_dependency_code(dep_name, front_dir, visited_paths)
        if dep_code is not None:
            if full_path:
                visited_paths.add(full_path)
            dep_codes.append(f"# ===== Dependency: {dep_name} =====\n{dep_code}")
            found_deps.append(dep_name)
            print(f"  [EXPAND] Found dependency '{dep_name}' for {front_name} in {full_path}")
        else:
            print(f"  [WARN] Dependency '{dep_name}' not found under {front_dir}")

    if not dep_codes:
        return code_body, ""

    dep_section = "\n\n".join(dep_codes)
    expanded = (
        "# ===== BEGIN: Dependency code (for complete semantics) =====\n"
        f"{dep_section}\n"
        "# ===== END: Dependency code =====\n"
        f"# === Original code ===\n{code_body}"
    )
    dep_str = ", ".join(found_deps)
    return expanded, dep_str
