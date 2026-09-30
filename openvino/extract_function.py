import os


def get_all_front_converters(front_op_dir_map, target_fronts):
    """
    Scan the OpenVINO frontend op directories for .cpp files and extract converter code.

    Args:
        front_op_dir_map: dict, frontend name -> op directory path
                          e.g. {'onnx': '/path/to/onnx/op', 'torch': '/path/to/torch/op'}
        target_fronts: list, frontends to process e.g. ['onnx', 'torch']

    Returns:
        all_front_converter: dict
            { front_name: (code_body_dict, name_mapping_dict) }
            - code_body_dict: { op_identity: file_content }
            - name_mapping_dict: { op_identity: op_identity }  (identity map)
    """
    all_front_converter = {}
    for front_name in target_fronts:
        front_path = front_op_dir_map.get(front_name)
        if not front_path or not os.path.isdir(front_path):
            print(f"[SKIP] Frontend '{front_name}' directory not found: {front_path}")
            continue

        print('*' * 30 + f"  {front_name}  " + '*' * 30)

        code_body_dict = {}
        name_mapping_dict = {}
        for root, _, files in os.walk(front_path):
            for file in files:
                if file.endswith('.cpp'):
                    op_identity = os.path.splitext(file)[0].lower()
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            code_body_dict[op_identity] = f.read()
                        name_mapping_dict[op_identity] = op_identity
                    except:
                        continue

        if code_body_dict:
            all_front_converter[front_name] = (code_body_dict, name_mapping_dict)
            print(f"  [OK] Extracted {front_name} frontend: {len(code_body_dict)} ops")

    return all_front_converter
