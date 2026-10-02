import os
import ast
import re

_IMPL_VERSION_RE = re.compile(r"_impl_v(\d+)")


def supported_opset(code_body):
    """The newest ONNX opset this converter declares, or None when it declares none.

    The frontend dispatches a model to the newest `_impl_vN` not above the model's opset,
    so this is the newest specification revision the converter implements.
    """
    versions = [int(v) for v in _IMPL_VERSION_RE.findall(code_body or "")]
    return max(versions) if versions else None


def preprocess_api_map(ori_map, flag):
    final_map = '{'
    match_content = re.search(r'\{(.*)\}', ori_map, re.DOTALL)
    content = match_content.group(1) if match_content else ori_map

    for line in content.split("\n"):
        line = line.strip()
        if not line or not (': ' in line):
            continue

        line = line.replace('self.', '')
        parts = line.split(': ')
        dll_name = parts[0].strip()
        dlc_name = parts[1].strip().rstrip(',')

        # --- Core change: handle unquoted keys (e.g., nn.Linear) ---
        if not (dll_name.startswith("'") or dll_name.startswith('"')):
            dll_name = f"'{dll_name}'"

        # --- Core change: handle unquoted values (e.g., _linear_module or Add) ---
        if not (dlc_name.startswith("'") or dlc_name.startswith('"')):
            if flag == 'func':
                dlc_name = dlc_name.split('.get_converter')[0]
            dlc_name = f"'{dlc_name}'"

        final_map += f"{dll_name}: {dlc_name},\n"

    final_map += '}'
    return final_map


def get_convert_map(source_code):
    # Pattern 1: traditional assignment (ONNX/Relay)
    # Pattern 2: function return statement (Torch Relax: return { ... })
    patterns = [
        r"_?convert_map\s*=\s*\{.*?\}",
        r"return\s+\{.*?\}"
    ]

    for p in patterns:
        matches = re.findall(p, source_code, flags=re.DOTALL)
        if matches:
            target_str = matches[-1]
            flag = 'func' if 'get_converter' in target_str else 'map'
            return preprocess_api_map(target_str, flag), flag

    # Pattern 3: traditional get_convert_map function definition
    pattern_func = r"def\s+_?get_convert_map\(.*?\}\n"
    match_func = re.findall(pattern_func, source_code, flags=re.DOTALL | re.IGNORECASE)
    if match_func:
        return preprocess_api_map(match_func[-1], 'func'), 'func'

    return None, None


def get_function(front_path):
    source_code = ""
    if os.path.isdir(front_path):
        for root, _, files in os.walk(front_path):
            for file in files:
                if file.endswith('.py'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        source_code += f.read() + "\n"
    else:
        with open(front_path, 'r', encoding='utf-8') as front_f:
            source_code = front_f.read()

    api_convert_map_str, api_converter_flag = get_convert_map(source_code)
    if not api_convert_map_str:
        print(f"[SKIP] the file: {front_path}")
        return None, None

    try:
        this_front_api_dict = eval(api_convert_map_str)
    except Exception as e:
        print(f"[ERROR] Eval failed for {front_path}: {e}")
        return None, None

    op_converter = {}
    tree = ast.parse(source_code)

    # Collect all values in the dict, because Torch's implementation function names are in the values
    valid_names = set()
    for k, v in this_front_api_dict.items():
        valid_names.add(str(k))
        valid_names.add(str(v))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name in valid_names:
                op_converter[node.name] = ast.get_source_segment(source_code, node)
        # Extract class definitions (ONNX commonly uses classes as converters)
        elif isinstance(node, ast.ClassDef):
            if node.name in valid_names:
                op_converter[node.name] = ast.get_source_segment(source_code, node)

    return op_converter, this_front_api_dict


def get_function_from_sources(source_list):
    """
    Extract convert_map and function bodies from multiple source strings.
    Used when the Torch frontend needs to read multiple files.
    """
    combined_code = "\n".join(source_list)

    api_convert_map_str, api_converter_flag = get_convert_map(combined_code)
    if not api_convert_map_str:
        return None, None

    try:
        this_front_api_dict = eval(api_convert_map_str)
    except Exception as e:
        print(f"[ERROR] Eval failed for combined torch sources: {e}")
        return None, None

    op_converter = {}
    try:
        tree = ast.parse(combined_code)
    except SyntaxError as e:
        print(f"[ERROR] AST parse failed for combined torch sources: {e}")
        return None, None

    valid_names = set()
    for k, v in this_front_api_dict.items():
        valid_names.add(str(k))
        valid_names.add(str(v))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name in valid_names:
                op_converter[node.name] = ast.get_source_segment(combined_code, node)
        elif isinstance(node, ast.ClassDef):
            if node.name in valid_names:
                op_converter[node.name] = ast.get_source_segment(combined_code, node)

    return op_converter, this_front_api_dict


def get_all_front_files(base_dir):
    all_frontend_dict = {}
    if not os.path.exists(base_dir): return {}
    for dll_proj_file in os.listdir(base_dir):
        if dll_proj_file.startswith('.') or dll_proj_file == "__pycache__":
            continue
        dll_proj, _ = os.path.splitext(dll_proj_file)
        dll_proj_path = os.path.join(base_dir, dll_proj_file)
        all_frontend_dict[dll_proj] = dll_proj_path
    return all_frontend_dict


def get_all_front_converters(base_dir, target_fronts):
    all_fronts_dict = get_all_front_files(base_dir)
    all_front_converter = {}
    for front, path in all_fronts_dict.items():
        if front in ['darknet', 'caffe2', 'qnn_torch', 'tensorflow2_ops']:
            continue
        if front not in target_fronts:
            continue
        print('*' * 30 + f"  {front}  " + '*' * 30)

        # For the Torch frontend, read fx_translator.py and base_fx_graph_translator.py,
        # skipping exported_program_translator.py (its keys use overload-style names like "round.default").
        # Note: base_fx_graph_translator.py comes first, then fx_translator.py,
        # so the regex picks up the return { ... } (convert_map) in fx_translator.py.
        if front == 'torch':
            base_path = os.path.join(path, 'base_fx_graph_translator.py')
            fx_path = os.path.join(path, 'fx_translator.py')
            sources = []
            if os.path.exists(base_path):
                with open(base_path, 'r', encoding='utf-8') as f:
                    sources.append(f.read())
            if os.path.exists(fx_path):
                with open(fx_path, 'r', encoding='utf-8') as f:
                    sources.append(f.read())
            function_dict, api_convert_map = get_function_from_sources(sources)
            if function_dict:
                all_front_converter[front] = [function_dict, api_convert_map]
            continue

        function_dict, api_convert_map = get_function(front_path=path)
        if function_dict:
            all_front_converter[front] = [function_dict, api_convert_map]
    return all_front_converter


if __name__ == '__main__':
    base_dir = r'[your tvm source root]/python/tvm/relax/frontend'
    bug_record_file = 'results_inconsistency_.txt'
    if os.path.exists(bug_record_file):
        os.remove(bug_record_file)

    all_front_converter = get_all_front_converters(base_dir,
                                                   target_fronts=['onnx', 'torch'])

    for k, v in all_front_converter.items():
        print(f"Framework: {k}, functions extracted: {len(v[0])}, mapping entries: {len(v[1])}")
