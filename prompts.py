"""Prompt templates shared by every backend.

Text that differs between tool stacks is left as a ``[Placeholder]`` token and
filled at call time from the active frontend module's ``PROMPT_VALUES`` (see
``frontends/``). Large per-backend blocks -- the Prompt 2 analysis rules, for
instance -- are carried whole in ``[AnalysisRules]`` rather than patched word by
word.

``render`` substitutes the tokens; the caller then applies ``str.format`` for
the per-call values. Braces never appear in the substituted blocks, and the
``[JUDGMENT]`` / ``[DEPS]`` markers used by the output contract do not collide
with any placeholder name.
"""

SYSTEM_PROMPT = "You are a compiler expert. Compare code logic carefully."

# Prompt 1 - code consistency comparison (Step 3)
PROMPT_1 = """Are the following two function/class logic equivalent?
The converter function/class in [ToolStack] {source_front} frontend is:
```[CodeLanguage]
{source_func}
```

The converter function/class in [ToolStack] {target_front} frontend is:
```[CodeLanguage]
{target_func}
```

Output format:
- If equivalent, end with: [JUDGMENT] EQUIVALENT
- If not equivalent, end with: [JUDGMENT] NONEQUIVALENT
"""

# Prompt 2 - dependency code identification (Step 2)
PROMPT_2 = """You are a [ToolStack] compiler frontend code analysis expert. Please analyze whether the [OperatorNoun] of the following {front_name} frontend requires helper code defined in other files to gain complete semantics:

```[CodeLanguage]
{code_body}
```

[Analysis Rules]
[AnalysisRules]

[Output Format]
If there are key external dependencies, output one line: `[DEPS]: dep_name1, dep_name2`
If not needed, output: `[DEPS]: NONE`
"""

# Prompt 3 - documentation-based deep audit (Step 5)
PROMPT_3 = """You are a compiler compliance audit expert, responsible for verifying that the frontend converters of the AI inference deployment and acceleration tool stack ([ToolStack]) fully conform to their official standard specifications.

[Audit Task]
We detected an inconsistency between frontend A and frontend B in [ToolStack].
[Preliminary Code Difference Conclusion]:
{op_pair_info}
{code_section}

[Reference Documents]
Frontend A framework documentation: {A_doc}
Frontend B framework documentation: {B_doc}

[Audit Criteria (strictly enforced to prevent false positives)]
1. [Hard prerequisite for a Bug verdict]: Only when you can point out, in the code, a [concrete behavior] of one side that deviates from the behavior defined in its own documentation, on [inputs that conform to its own documentation] (e.g., a missing branch, an unhandled pattern, a parameter silently ignored and producing wrong output), may you mark [Bug]. You must cite a specific code location or construct.
2. [An extension is not a Bug]: If one side's implementation supports types, input forms, or attributes beyond its documented specification (i.e., the implementation is a [superset] of the specification), and behaves correctly on standard inputs conforming to the specification, this is an [intentional extension / compatibility handling]; mark [Standard Gap] or [Optimization Difference], and [must NOT] mark [Bug].
3. [Delegation / thin wrapper is not a Bug]: If one side's implementation is merely a thin wrapper that delegates its core logic to a shared utility function or base class (e.g., [SharedHelperExamples]), and that utility function's implementation is not included in the code provided, so you [cannot confirm its logic], you [must NOT] mark [Bug] merely because you "did not see the implementation." Mark [Inconclusive] or [Standard Gap].
4. [A framework-spec difference is a Standard Gap]: When the behavioral difference between the two frontends stems entirely from differences between the two frameworks' specifications themselves (e.g., argument order, opset version semantics, naming), and each side's code faithfully implements its own framework's specification, mark [Standard Gap].
5. [When evidence is insufficient]: When the evidence is insufficient to determine that one side has an implementation defect, mark [Inconclusive]; do not guess, and do not force a Bug just to "find an inconsistency".

[Execution Steps]
Step 1: [Full constraint set extraction]
Extract from documents A and B the operator's [complete semantic constraint set] (inputs/outputs, types, attributes, edge-case behavior).
Step 2: [Difference attribution]
Compare the constraints set extracted in Step 1 against the code:
1. Logic coverage self-check: Does code A cover every point in constraint set A? Does code B cover every point in constraint set B?
2. Classify the nature of the difference according to the [Audit Criteria] above: Bug / Standard Gap / Optimization Difference / Inconclusive.

[Output Format]
1. [Full constraint comparison table]: Briefly describe the complete behavior required by documents A and B.
2. [Implementation defect identification]: Point out which side's implementation fails to align with its own documentation constraints, and [cite the specific code location or construct]; if there is no defect, state so explicitly.
3. [Final conclusion]:
Format requirement (must end with this): [Final Conclusion]: [Bug] or [Standard Gap] or [Optimization Difference] or [Inconclusive].
"""

# Code-block header emitted when the expanded code is embedded in Prompt 3.
PROMPT_3_CODE_SECTION = """
    [Frontend A complete code (including expanded dependencies)]:
    ```[CodeLanguage]
    {source_code}
    ```

    [Frontend B complete code (including expanded dependencies)]:
    ```[CodeLanguage]
    {target_code}
    ```
    """


def render(template, values):
    """Replace ``[Name]`` tokens with the matching entry of ``values``."""
    out = template
    for key, val in (values or {}).items():
        out = out.replace(f"[{key}]", str(val))
    return out
