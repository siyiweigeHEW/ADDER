from llm_client import global_model
import re


def parse_gpt_answer4func(answer):
    pattern = f'`:.*`'
    pure_api_list = re.findall(pattern, answer)
    match_len = len(pure_api_list)
    if match_len == 1:
        pure_api = pure_api_list[0]
        pure_api = pure_api[2:-1]
    elif match_len == 0:
        pure_api = ''
    else:
        for api in pure_api_list:
            if 'function' not in api:
                pure_api = api[2:-1]
    return pure_api


def parse_gpt_answer4consistency(answer):
    explanation = answer
    print('******original answer', answer)
    # Only extract the [JUDGMENT] tag from the last line to prevent body-text mismatches
    last_line = answer.strip().split('\n')[-1].strip().lower()
    if '[judgment] nonequivalent' in last_line:
        return 'nonequivalent', explanation
    elif '[judgment] equivalent' in last_line:
        return 'equivalent', explanation
    else:
        return 'skipped', ''


def compare_code_consistency(source_front, source_func, target_front, target_func):
    compare_consistency_prompt = f"Are the following two function/class logic equivalent?\n" \
                                 f"The converter function/class in TVM {source_front} frontend is:" \
                                 f"\n```{source_func}\n```\n" \
                                 f"The converter function/class in TVM {target_front} frontend is:" \
                                 f"\n```{target_func}\n```\n" \
                                 f"\nOutput format:\n" \
                                 f"- If equivalent, end with: [JUDGMENT] EQUIVALENT\n" \
                                 f"- If not equivalent, end with: [JUDGMENT] NONEQUIVALENT\n"

    token_length = len(compare_consistency_prompt.split(" "))
    if token_length > 10000:
        print(f'token length: {token_length} may exceed the max_token, skip it!')
        return 'skipped', '', ''
    ori_answer = global_model.model_prediction(compare_consistency_prompt)
    result, explanation = parse_gpt_answer4consistency(ori_answer)
    return result, explanation, compare_consistency_prompt
