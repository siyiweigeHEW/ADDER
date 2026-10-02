import os
import re

_OPSET_DECL_RE = re.compile(
    r"OPSET_RANGE\(\s*(\d+)\s*,\s*(\d+)\s*\)"   # this opset up to that one, both inclusive
    r"|OPSET_IN\(\s*(\d+)\s*\)"                 # that opset only
    r"|OPSET_SINCE\(\s*(\d+)\s*\)"              # that opset onward
)


def supported_opset(code_body):
    """The newest ONNX opset this converter declares, or None when it declares none.

    `ONNX_OP("Abs", OPSET_RANGE(1, 5), ...)` covers opsets 1 to 5, `OPSET_IN(13)` names 13
    alone and `OPSET_SINCE(13)` covers 13 onward, so the newest opset any declaration names
    is the newest specification revision the file implements.
    """
    versions = []
    for _lo, hi, only, since in _OPSET_DECL_RE.findall(code_body or ""):
        versions.append(int(hi or only or since))
    return max(versions) if versions else None


def get_all_front_converters(front_op_dir_map, target_fronts):
    """Returns {front_name: (code_body_dict, name_mapping_dict)}."""
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
