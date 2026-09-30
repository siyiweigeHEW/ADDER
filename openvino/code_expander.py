"""Code expansion for OpenVINO: find the definitions a converter delegates to and
append them, so a thin wrapper is not read as missing logic.
"""
import os
import re
from llm_client import global_model


def get_dependency_search_dirs(front_op_dir):
    """Candidate dirs to search, in priority order."""
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
    add(os.path.join(parent, 'utils'))
    add(os.path.join(parent, 'core'))
    add(os.path.join(parent, 'common'))

    parts = os.path.abspath(front_op_dir).split(os.sep)
    if 'frontends' in parts:
        frontends_root = os.sep.join(parts[:parts.index('frontends') + 1])
        common_dir = os.path.join(frontends_root, 'common_translators')
        add(common_dir)
        add(os.path.join(common_dir, 'src'))
        add(os.path.join(common_dir, 'include'))

    return dirs


def analyze_dependencies_with_llm(code_body, front_name):
    """External helper names the code needs, or []."""
    prompt = f"""You are an OpenVINO compiler code analysis expert. Please analyze whether the operator code of the following {front_name} frontend requires helper code defined in other files to gain complete semantics:

```cpp
{code_body}
```

[Analysis Rules]
1. Check whether the function body calls functions/classes defined in other files.
2. Pay special attention to: a function that directly returns the result of another function (e.g., `return reverse_op(node);`) and calls to helper functions in other files.
3. Do NOT list standard-library functions (with the std:: prefix) or OpenVINO framework built-in classes (with the ov:: prefix, NodeContext, etc.).
4. The external dependencies to list may be defined in: other files in the same operator directory (src/op), utils.cpp/utils.hpp under the frontend src directory (e.g., get_inputs_with_promoted_types, get_shape_rank, normalize_axis, make_list_construct, etc.), or the shared directory common_translators (e.g., common_translators::translate_atan2_util).
5. If a function is just a thin wrapper whose real logic lives in some shared utility function, be sure to list that utility function as a dependency.

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


_CALL_CONTEXT_TOKENS = ('return', '=', '(', ',', ':', '->', '&&', '||', '!', 'emit', '{', '[', '?', '.', 'new', 'delete')


def _prev_code_line(content, line_start):
    """Return the last non-empty, non-comment line before `line_start`."""
    i = line_start - 1
    while i > 0:
        seg_start = content.rfind('\n', 0, i) + 1
        seg = content[seg_start:i].strip()
        if seg and not seg.startswith(('//', '/*', '*', '#')):
            return seg
        i = seg_start - 1
    return ''


def _find_dependency_in(content, clean_name):
    """True when `clean_name` is defined here rather than merely called."""
    if re.search(rf'(?:^|\n)\s*(?:class|struct|using|typedef)\s+{re.escape(clean_name)}\b',
                 content, re.MULTILINE):
        return True
    for m in re.finditer(rf'\b{re.escape(clean_name)}\s*\(', content):
        start = m.start()
        line_start = content.rfind('\n', 0, start) + 1
        before = content[line_start:start].strip()
        if before == '':
            # A name starting the line is a definition only if the previous code
            # line is a dangling return type.
            prev_line = _prev_code_line(content, line_start)
            if prev_line and not prev_line.endswith(('{', ';', 'return', '=', '(', ',', '->', ')')):
                if re.search(r'\w[>\])]?$', prev_line):
                    return True
            continue
        last_tok = before.split()[-1]
        if last_tok in _CALL_CONTEXT_TOKENS:
            continue
        if before.endswith(('return', '=', '(', ',', ':', '->', '{', '[', '?')):
            continue
        return True
    return False


def _find_definition_line(lines, clean_name):
    """Return 0-based index of the line defining `clean_name`, or None."""
    for i, line in enumerate(lines):
        m = re.search(rf'\b{re.escape(clean_name)}\s*\(', line)
        if not m:
            continue
        before = line[:m.start()].strip()
        if before == '':
            # A name starting the line: check the previous code line.
            j = i - 1
            while j >= 0:
                prev = lines[j].strip()
                if prev and not prev.startswith(('//', '/*', '*', '#')):
                    break
                j -= 1
            prev = lines[j].strip() if j >= 0 else ''
            if prev and not prev.endswith(('{', ';', 'return', '=', '(', ',', '->', ')')):
                if re.search(r'\w[>\])]?$', prev):
                    return i
            continue
        last_tok = before.split()[-1]
        if last_tok in _CALL_CONTEXT_TOKENS:
            continue
        if before.endswith(('return', '=', '(', ',', ':', '->', '{', '[', '?')):
            continue
        return i
    return None


def _extract_dependency_snippet(content, clean_name):
    """Return a bounded snippet of the function/class that defines clean_name."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.match(rf'\s*(?:class|struct)\s+{re.escape(clean_name)}\b', line):
            return _extract_braced_block(lines, i)
    idx = _find_definition_line(lines, clean_name)
    if idx is None:
        return None
    return _extract_braced_block(lines, idx)


def _extract_braced_block(lines, start):
    """Return the source block from `start` through its balanced closing brace."""
    # back up over a dangling return-type line (e.g. 'std::tuple<...>' alone)
    while start > 0:
        prev = lines[start - 1].strip()
        if prev and not prev.startswith(('//', '/*', '*', '#')) and \
           not prev.endswith(('{', ';', '}')) and re.search(r'\w[>\])]?$', prev):
            start -= 1
        else:
            break
    depth = 0
    started = False
    end = start
    for j in range(start, min(start + 800, len(lines))):
        sline = re.sub(r'//.*', '', lines[j])
        prev_depth = depth
        depth += sline.count('{') - sline.count('}')
        if depth > 0:
            started = True
        if started and depth <= 0 and prev_depth > 0:
            end = j
            break
        end = j
    return '\n'.join(lines[start:end + 1])


def search_dependency_code(dep_name, front_op_dir, visited_paths=None):
    """Find `dep_name` across the search dirs; returns (snippet, path) or (None, None)."""
    clean_name = dep_name.split('::')[-1]
    visited_paths = visited_paths if visited_paths is not None else set()
    search_dirs = get_dependency_search_dirs(front_op_dir)

    for exts in (('.cpp',), ('.hpp', '.h')):
        for root in search_dirs:
            for dirpath, _, files in os.walk(root):
                for file in sorted(files):
                    if not file.endswith(exts):
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


def expand_code_body(code_body, front_name, front_op_dir):
    """Returns (expanded_code, dependency names)."""
    dependency_names = analyze_dependencies_with_llm(code_body, front_name)

    if not dependency_names:
        return code_body, ""

    dep_codes = []
    found_deps = []
    visited_paths = set()

    for dep_name in dependency_names:
        dep_code, full_path = search_dependency_code(dep_name, front_op_dir, visited_paths)
        if dep_code is not None:
            if full_path:
                visited_paths.add(full_path)
            dep_codes.append(f"// ===== Dependency: {dep_name} =====\n{dep_code}")
            found_deps.append(dep_name)
            print(f"  [EXPAND] Found dependency '{dep_name}' for {front_name} in {full_path}")
        else:
            print(f"  [WARN] Dependency '{dep_name}' not found under {front_op_dir}")

    if not dep_codes:
        return code_body, ""

    dep_section = "\n\n".join(dep_codes)
    expanded = (
        "// ===== BEGIN: Dependency code (for complete semantics) =====\n"
        f"{dep_section}\n"
        "// ===== END: Dependency code =====\n"
        f"// === Original code ===\n{code_body}"
    )
    dep_str = ", ".join(found_deps)
    return expanded, dep_str
