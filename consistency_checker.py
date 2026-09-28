"""Step 3: first-round consistency comparison.

The LLM is shown the two converter bodies and asked whether they are logically
equivalent. Pairs judged equivalent pass straight through; the rest continue to
the documentation-based audit in Step 5.
"""
import prompts
from frontends import get_frontend
from llm_client import global_model

# Prompts larger than this are skipped rather than truncated.
MAX_PROMPT_WORDS = 10000


def parse_gpt_answer4consistency(answer):
    """Read the verdict off the last line, where the output contract puts it."""
    last_line = answer.strip().split("\n")[-1].strip().lower()
    if "[judgment] nonequivalent" in last_line:
        return "nonequivalent", answer
    if "[judgment] equivalent" in last_line:
        return "equivalent", answer
    return "skipped", ""


def compare_code_consistency(source_front, source_func, target_front, target_func,
                             backend=None):
    """Compare two converters.

    Returns ``(result, explanation, prompt)`` where ``result`` is
    ``'equivalent'``, ``'nonequivalent'`` or ``'skipped'``. ``prompt`` is empty
    when the pair was skipped.
    """
    fe = get_frontend(backend)
    template = prompts.render(prompts.PROMPT_1, fe.PROMPT_VALUES)
    prompt = template.format(
        source_front=source_front,
        source_func=source_func,
        target_front=target_front,
        target_func=target_func,
    )

    token_length = len(prompt.split(" "))
    if token_length > MAX_PROMPT_WORDS:
        print(f"token length: {token_length} may exceed the max_token, skip it!")
        return "skipped", "", ""

    answer = global_model.model_prediction(prompt)
    result, explanation = parse_gpt_answer4consistency(answer)
    return result, explanation, prompt
