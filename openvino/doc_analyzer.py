import re

from llm_client import global_model


def analyze_with_docs(op_pair_info, A_doc, B_doc, source_code="", target_code=""):
    code_section = ""
    if source_code and target_code:
        code_section = f"""
    [Frontend A complete code (including expanded dependencies)]:
    ```cpp
    {source_code}
    ```

    [Frontend B complete code (including expanded dependencies)]:
    ```cpp
    {target_code}
    ```
    """

    prompt = f"""
    You are a compiler compliance audit expert, responsible for verifying that the implementation in the AI inference deployment and acceleration tool stack (OpenVINO) fully conforms to its official standard specification.

    [Audit Task]
    We detected an inconsistency between frontend A and frontend B in OpenVINO.
    [Preliminary Code Difference Conclusion]:
    {op_pair_info}
    {code_section}

    [Reference Documents]
    Frontend A framework documentation: {A_doc}
    Frontend B framework documentation: {B_doc}

    [Audit Criteria (strictly enforced to prevent false positives)]
    1. [Hard prerequisite for a Bug verdict]: Only when you can point out, in the code, a [concrete behavior] of one side that deviates from the behavior defined in its own documentation, on [inputs that conform to its own documentation] (e.g., a missing branch, an unhandled pattern, a parameter silently ignored and producing wrong output), may you mark [Bug]. You must cite a specific code location or construct.
    2. [An extension is not a Bug]: If one side's implementation supports types, input forms, or attributes beyond its documented specification (i.e., the implementation is a [superset] of the specification), and behaves correctly on standard inputs conforming to the specification, this is an [intentional extension / compatibility handling]; mark [Standard Gap] or [Optimization Difference], and [must NOT] mark [Bug].
    3. [Delegation / thin wrapper is not a Bug]: If one side's implementation is merely a thin wrapper that delegates its core logic to a shared utility function (e.g., common_translators::xxx, utils::xxx, etc.), and that utility function's implementation is not included in the code provided, so you [cannot confirm its logic], you [must NOT] mark [Bug] merely because you "did not see the implementation." Mark [Inconclusive] or [Standard Gap].
    4. [A framework-spec difference is a Standard Gap]: When the behavioral difference between the two frontends stems entirely from differences between the two frameworks' specifications themselves (e.g., argument order, opset version semantics, naming), and each side's code faithfully implements its own framework's specification, mark [Standard Gap].
    5. [When evidence is insufficient]: When the evidence is insufficient to determine that one side has an implementation defect, mark [Inconclusive]; do not guess, and do not force a Bug just to "find an inconsistency".

    [Execution Steps]
    Step 1: [Full constraint set extraction]
    Extract from documents A and B the operator's [complete semantic constraint set] (inputs/outputs, types, attributes, edge-case behavior).
    Step 2: [Difference attribution]
    Compare the constraint sets extracted in Step 1 against the code:
    1. Logic coverage self-check: Does code A cover every point in constraint set A? Does code B cover every point in constraint set B?
    2. Classify the nature of the difference according to the [Audit Criteria] above: Bug / Standard Gap / Optimization Difference / Inconclusive.

    [Output Format]
    1. [Full constraint comparison table]: Briefly describe the complete behavior required by documents A and B.
    2. [Implementation defect identification]: Point out which side's implementation fails to align with its own documentation constraints, and [cite the specific code location or construct]; if there is no defect, state so explicitly.
    3. [Final conclusion]:
    Format requirement (must end with this): [Final Conclusion]: [Bug] or [Standard Gap] or [Optimization Difference] or [Inconclusive].
    """

    try:
        response = global_model.model_prediction(prompt)
        return response
    except Exception as e:
        return f"Deep analysis call failed: {str(e)}"


_VERDICT_LABELS = (
    ('inconclusive', 'INCONCLUSIVE'),
    ('bug', 'BUG'),
    ('standard_gap', 'STANDARD GAP'),
    ('optimization', 'OPTIMIZATION'),
)


def parse_final_verdict(text):
    """Robustly extract the final verdict from a doc-analysis response.

    Returns one of: 'bug', 'standard_gap', 'optimization', 'inconclusive',
    'unparsed'. Mirrors the verdict labels in the audit prompt. Prefers the
    label that appears immediately after the (last) [FINAL CONCLUSION] marker,
    so reasoning text mentioning "bug" earlier is not misread.
    """
    if not text:
        return 'unparsed'
    upper = text.upper()
    idx = upper.rfind('[FINAL CONCLUSION]')
    if idx == -1:
        # No conclusion marker -> non-conforming response; do not guess.
        return 'unparsed'
    tail = upper[idx + len('[FINAL CONCLUSION]'):]

    # Prefer an explicit bracketed verdict label right after the marker.
    m = re.search(r'\[(INCONCLUSIVE|BUG|STANDARD\s*GAP|OPTIMIZATION(?:\s*DIFFERENCE)?)\]', tail)
    if m:
        label = m.group(1)
        return 'inconclusive' if 'INCONCLUSIVE' in label else \
               'bug' if 'BUG' in label else \
               'standard_gap' if 'STANDARD' in label else 'optimization'

    # Fallback: first verdict keyword in the conclusion tail, guarding against
    # negated wording such as "not a bug" / "no bug".
    first_pos = -1
    verdict = 'unparsed'
    for key, word in _VERDICT_LABELS:
        pos = tail.find(word)
        if pos == -1:
            continue
        prefix = tail[max(0, pos - 6):pos]
        if key == 'bug' and re.search(r'(?:NOT|NO)\s*$', prefix):
            continue
        if first_pos == -1 or pos < first_pos:
            first_pos, verdict = pos, key
    return verdict
