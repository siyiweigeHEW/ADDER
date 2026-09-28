"""Step 5: documentation-based deep audit.

A code-level inconsistency is only a Bug candidate. This step puts the two
implementations next to their own framework's documentation and asks the LLM to
extract the full constraint set on both sides, then attribute the difference to
a genuine defect or to something that is not a defect. The audit criteria in the
prompt exist to suppress the false positives the code-only comparison produces.
"""
import re

import prompts
from frontends import get_frontend
from llm_client import global_model


def analyze_with_docs(op_pair_info, A_doc, B_doc, source_code="", target_code="",
                      backend=None):
    """Return the LLM's raw second-round audit response."""
    fe = get_frontend(backend)

    code_section = ""
    if source_code and target_code:
        section = prompts.render(prompts.PROMPT_3_CODE_SECTION, fe.PROMPT_VALUES)
        code_section = section.format(source_code=source_code, target_code=target_code)

    template = prompts.render(prompts.PROMPT_3, fe.PROMPT_VALUES)
    prompt = template.format(
        op_pair_info=op_pair_info,
        code_section=code_section,
        A_doc=A_doc,
        B_doc=B_doc,
    )

    try:
        return global_model.model_prediction(prompt)
    except Exception as e:
        return f"Deep analysis call failed: {str(e)}"


_VERDICT_LABELS = (
    ("inconclusive", "INCONCLUSIVE"),
    ("bug", "BUG"),
    ("standard_gap", "STANDARD GAP"),
    ("optimization", "OPTIMIZATION"),
)


def parse_final_verdict(text):
    """Extract the verdict from an audit response.

    Returns one of ``'bug'``, ``'standard_gap'``, ``'optimization'``,
    ``'inconclusive'`` or ``'unparsed'``. Only text after the last
    ``[FINAL CONCLUSION]`` marker is considered, so reasoning that mentions
    "bug" earlier is not mistaken for the verdict.
    """
    if not text:
        return "unparsed"
    upper = text.upper()
    idx = upper.rfind("[FINAL CONCLUSION]")
    if idx == -1:
        # No conclusion marker: non-conforming response, do not guess.
        return "unparsed"
    tail = upper[idx + len("[FINAL CONCLUSION]"):]

    # Prefer an explicit bracketed label right after the marker.
    m = re.search(r"\[(INCONCLUSIVE|BUG|STANDARD\s*GAP|OPTIMIZATION(?:\s*DIFFERENCE)?)\]", tail)
    if m:
        label = m.group(1)
        return "inconclusive" if "INCONCLUSIVE" in label else \
               "bug" if "BUG" in label else \
               "standard_gap" if "STANDARD" in label else "optimization"

    # Otherwise take the first verdict keyword, guarding against negated wording
    # such as "not a bug" / "no bug".
    first_pos = -1
    verdict = "unparsed"
    for key, word in _VERDICT_LABELS:
        pos = tail.find(word)
        if pos == -1:
            continue
        prefix = tail[max(0, pos - 6):pos]
        if key == "bug" and re.search(r"(?:NOT|NO)\s*$", prefix):
            continue
        if first_pos == -1 or pos < first_pos:
            first_pos, verdict = pos, key
    return verdict
