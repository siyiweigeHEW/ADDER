"""Step 2: expand converter code with the helper code it delegates to.

Many converters are thin wrappers whose real logic lives in another file. Passing
only the wrapper body to the LLM hides that logic, so a delegated implementation
is reported as missing and turns into a false Bug verdict. This module asks the
LLM which helpers are needed, locates their definitions, and appends them so the
comparison in Step 3 sees self-contained code.

Only the definition search and the prompt wording differ between backends; both
come from the active frontend module.
"""
import os

import prompts
from frontends import get_frontend
from llm_client import global_model


def parse_dependencies(response):
    """Extract the ``[DEPS]: a, b`` line from a dependency-analysis response."""
    for line in response.split("\n"):
        if "[DEPS]:" in line:
            deps_str = line.split("[DEPS]:")[1].strip()
            if deps_str.upper() == "NONE":
                return []
            return [d.strip() for d in deps_str.split(",") if d.strip()]
    return []


def analyze_dependencies_with_llm(code_body, front_name, backend=None):
    """Ask the LLM which external definitions ``code_body`` needs.

    Returns a list of dependency names, empty when none are needed or the call
    fails.
    """
    fe = get_frontend(backend)
    template = prompts.render(prompts.PROMPT_2, fe.PROMPT_VALUES)
    prompt = template.format(front_name=front_name, code_body=code_body)

    try:
        response = global_model.model_prediction(prompt)
    except Exception as e:
        print(f"  [WARN] Dependency analysis LLM call failed: {e}")
        return []
    return parse_dependencies(response)


def search_dependency_code(dep_name, front_dir, backend=None, visited_paths=None):
    """Locate the definition of ``dep_name`` near ``front_dir``.

    Searches the backend's candidate directories in priority order and returns a
    bounded snippet rather than a whole file, to keep the expanded prompt under
    the token limit. Paths in ``visited_paths`` are skipped so two dependencies
    defined in the same file do not duplicate it.

    Returns ``(snippet, full_path)``, or ``(None, None)`` when not found.
    """
    fe = get_frontend(backend)
    clean_name = dep_name.split(fe.NAME_SEPARATOR)[-1]
    visited_paths = visited_paths if visited_paths is not None else set()

    for extensions in fe.FILE_EXTENSION_GROUPS:
        for root in fe.dependency_search_dirs(front_dir):
            for dirpath, _, files in os.walk(root):
                for file in sorted(files):
                    if not file.endswith(extensions):
                        continue
                    full_path = os.path.join(dirpath, file)
                    if full_path in visited_paths:
                        continue
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except OSError:
                        continue
                    snippet = fe.extract_dependency_snippet(content, clean_name)
                    if snippet:
                        return snippet, full_path
    return None, None


def expand_code_body(code_body, front_name, front_dir, backend=None):
    """Return ``(expanded_code, dependency_names)`` for one converter body.

    ``expanded_code`` is the original body with each located dependency prepended;
    ``dependency_names`` is a comma-separated string, empty when nothing was added.
    """
    fe = get_frontend(backend)
    dependency_names = analyze_dependencies_with_llm(code_body, front_name, backend)
    if not dependency_names:
        return code_body, ""

    prefix = fe.COMMENT_PREFIX
    dep_codes = []
    found_deps = []
    visited_paths = set()

    for dep_name in dependency_names:
        dep_code, full_path = search_dependency_code(dep_name, front_dir, backend, visited_paths)
        if dep_code is None:
            print(f"  [WARN] Dependency '{dep_name}' not found under {front_dir}")
            continue
        visited_paths.add(full_path)
        dep_codes.append(f"{prefix} ===== Dependency: {dep_name} =====\n{dep_code}")
        found_deps.append(dep_name)
        print(f"  [EXPAND] Found dependency '{dep_name}' for {front_name} in {full_path}")

    if not dep_codes:
        return code_body, ""

    dep_section = "\n\n".join(dep_codes)
    expanded = (
        f"{prefix} ===== BEGIN: Dependency code (for complete semantics) =====\n"
        f"{dep_section}\n"
        f"{prefix} ===== END: Dependency code =====\n"
        f"{prefix} === Original code ===\n{code_body}"
    )
    return expanded, ", ".join(found_deps)
