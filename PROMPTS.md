# Prompts

Every prompt the tool sends, verbatim. `{...}` marks a value substituted at call time;
`[...]` marks a piece of text that differs between the two tool stacks, with the values
tabulated below each template.

Two pipelines share all of these. They differ only inside Prompt 2, which is therefore
written out once per tool stack, and in four spots in Prompt 3, marked where they occur.

## Contents

| | |
|---|---|
| [1. System prompt](#1-system-prompt) | sent with every call |
| [2. Prompt 1: code consistency](#2-prompt-1-code-consistency) | Step 3 |
| [3. Prompt 2: dependency identification](#3-prompt-2-dependency-identification) | Step 2 |
| [4. Prompt 3: documentation audit](#4-prompt-3-documentation-audit) | Step 5 |
| [5. Repair prompts](#5-repair-prompts) | `autorepair/` |

---

## 1. System prompt

Sent with every call in both pipelines.

    You are a compiler expert. Compare code logic carefully.

---

## 2. Prompt 1: code consistency

Step 3. Asks the model to judge whether two converters of the same operator, from two
frontends of one tool stack, are logically equivalent.

    Are the following two function/class logic equivalent?
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

| Placeholder | TVM | OpenVINO |
|---|---|---|
| `[ToolStack]` | `TVM` | `OpenVINO` |
| `[CodeLanguage]` | `python` | `cpp` |

A prompt over 10,000 words is skipped rather than sent, and the pair is flagged `9`.

---

## 3. Prompt 2: dependency identification

Step 2. Asks which definitions outside the converter's own file the converter relies on,
so Step 2 can append them before the comparison. The two tool stacks get a different rule
set, so the prompt is written out once per stack.

**TVM** (`tvm/code_expander.py`), six rules:

    You are a TVM Relax compiler frontend code analysis expert. Please analyze whether the operator conversion code of the following {front_name} frontend requires helper code defined in other files to gain complete semantics:

    ```python
    {code_body}
    ```

    [Analysis Rules]
    1. If it is a class definition (class X(BaseY)), check whether the base class BaseY needs to be expanded; if BaseY itself inherits from other classes, list them as well.
    2. Check whether the function/class body calls other helper functions or classes defined within the same package (excluding built-in functions and TVM framework functions).
    3. Do NOT list standard-library modules (os, math, typing, etc.) or TVM built-in modules (relax.op, tvm, tir, etc.).
    4. The external dependencies to list may be defined in: other files in the same frontend directory (e.g., onnx_frontend.py), or shared modules under the frontend root relax/frontend (e.g., utility functions in common.py).
    5. If a function/class is just a thin wrapper whose real logic lives in some shared utility function or base class, be sure to list that utility function/base class as a dependency.
    6. Only list base class names or helper function names essential for understanding the core logic.

    [Output Format]
    If there are key external dependencies, output one line: `[DEPS]: dep_name1, dep_name2`
    If not needed, output: `[DEPS]: NONE`

**OpenVINO** (`openvino/code_expander.py`), five rules — there is no C++ counterpart to
the base-class rule, the call-detection rule carries a concrete example, and the closing
rule is absent:

    You are an OpenVINO compiler code analysis expert. Please analyze whether the operator code of the following {front_name} frontend requires helper code defined in other files to gain complete semantics:

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

**Why they differ.** A TVM converter is often a class whose body is inherited
(`class Add(BinaryBase)`), which the C++ frontends do not have, so only TVM needs the
base-class rule; conversely OpenVINO's rule 2 carries a call-shaped example
(`return reverse_op(node);`) with no Python counterpart.

---

## 4. Prompt 3: documentation audit

Step 5, run only for pairs Prompt 1 judged non-equivalent. Shown with the TVM values; the
four places OpenVINO differs are listed underneath.

    You are a compiler compliance audit expert, responsible for verifying that the deep learning frontend converters (Converter) in the AI inference deployment and acceleration tool stack (TVM) fully conform to their official standard specifications.

    [Audit Task]
    We detected an inconsistency between frontend A and frontend B in TVM.
    [Preliminary Code Difference Conclusion]:
    {op_pair_info}
    {code_section}

    [Reference Documents]
    Frontend A framework documentation: {A_doc}
    Frontend B framework documentation: {B_doc}

    [Audit Criteria (strictly enforced to prevent false positives)]
    1. [Hard prerequisite for a Bug verdict]: Only when you can point out, in the code, a [concrete behavior] of one side that deviates from the behavior defined in its own documentation, on [inputs that conform to its own documentation] (e.g., a missing branch, an unhandled pattern, a parameter silently ignored and producing wrong output), may you mark [Bug]. You must cite a specific code location or construct.
    2. [An extension is not a Bug]: If one side's implementation supports types, input forms, or attributes beyond its documented specification (i.e., the implementation is a [superset] of the specification), and behaves correctly on standard inputs conforming to the specification, this is an [intentional extension / compatibility handling]; mark [Standard Gap] or [Optimization Difference], and [must NOT] mark [Bug].
    3. [Delegation / thin wrapper is not a Bug]: If one side's implementation is merely a thin wrapper that delegates its core logic to a shared utility function or base class (e.g., utility functions in common.py under relax/frontend, base classes in onnx_frontend.py, etc.), and that utility function's implementation is not included in the code provided, so you [cannot confirm its logic], you [must NOT] mark [Bug] merely because you "did not see the implementation." Mark [Inconclusive] or [Standard Gap].
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

`{op_pair_info}` is the first round's conclusion and explanation. `{code_section}` is the two
expanded implementations, fenced with `[CodeLanguage]`.

The OpenVINO version differs in exactly four places:

| | TVM | OpenVINO |
|---|---|---|
| both code fences | `python` | `cpp` |
| opening sentence | "the deep learning frontend converters (Converter) in the … tool stack (**TVM**) fully conform to their official standard specifications." | "the implementation in the … tool stack (**OpenVINO**) fully conforms to its official standard specification." |
| audit task | "…between frontend A and frontend B in **TVM**." | "…in **OpenVINO**." |
| criterion 3's example | utility functions in `common.py` under `relax/frontend`, base classes in `onnx_frontend.py` | `common_translators::xxx`, `utils::xxx` |

---

## 5. Repair prompts

Four more prompts, all in `autorepair/`. The second and third are the two complementary
checks a candidate must pass before it is returned.

### 5.1 Fix prompt

Asks for the fixed code. It deliberately never asks for a diff — the patch is computed by
the program from the real source file, because a hand-written diff is usually not
applicable.

    # Fix the defect in the {backend}-{frontend} frontend converter "{op}"

    ## 1. Root cause of the defect (attribution)
    {cause_block}

    ## 2. Original code (to be fixed; the fix must preserve the full extent of this code)
    ```{lang}
    {code}
    ```

    ## 3. Key documentation information (only what is needed for the fix)
    {doc}

    ## 4. Output: the complete fixed code
    The patch is generated automatically by the program; you only need to output the code
    itself. Please follow this contract:
    1. Scope alignment: output the complete code in the same scope as the "Original code";
       modify only the lines directly related to the root cause; preserve every other line
       (comments, blank lines, indentation, function signatures) byte-for-byte; do not
       reorder, rename, or refactor along the way.
    2. Only use operators/interfaces that actually exist in this {backend} version. Prefer
       the concrete implementation approach given in "Key documentation information"; if
       unsure whether an API exists, use an equivalent low-level implementation (e.g., a
       topi operator + `bb.emit_te`), and note it outside the code block.
    3. Outside the code block, first explain the root cause and the change in one sentence.
    4. If the information is insufficient to determine the fix, state outside the code block
       what is missing; do not fabricate.

    Output format: an explanation outside the code block + a ```{lang} code block
    (the complete fixed code)

Its system prompt:

    You are a meticulous {backend} frontend converter developer. Write minimal surgical
    fixes: preserve every untouched line byte-for-byte, never rename or restructure code,
    and never use APIs that may not exist in this {backend} version.

### 5.2 Misuse review

Given the original code, the added lines, and for each API the declaration and comment
block found at its definition site, the model answers with JSON:

    # API usage review (autorepair automatic verification)

    The added/changed lines of the fixed code call the following APIs. Based on each API's
    signature and summary, judge whether its usage is correct.

    ## Original code (the fix scope; use it to infer codebase conventions)
    ```{code_lang}
    {original}
    ```

    ## Added/changed lines of the fixed code
    ```{code_lang}
    {added}
    ```

    ## Documentation of the used APIs (signature + summary)
    {declarations}

    ## Judging criteria (any hit -> correct=false)
    1. Parameter name, type, or count does not match the signature;
    2. Return type/semantics do not match what the call site expects;
    3. Clearly deviates from codebase conventions (e.g., passing a DataType object where a
       string dtype is expected, calling a language built-in where a {backend} API is
       expected, hard-coding behavior that should be parameterized);
    4. The API does not exist in this version.

    ## Output (strict JSON, only JSON)
    ```json
    {"verdicts": [
      {"api": "<full API name>", "correct": true, "issue": "", "suggestion": ""}
    ]}
    ```

Its system prompt is `You are a meticulous {backend} frontend developer reviewing whether
<the APIs a patch calls> are used correctly. Return only the JSON object.`

### 5.3 Semantic correctness review

The second of the two checks a candidate must pass, and the only user of the source
requirement. `{requirement}` is the documentation supplied for the fix and `{cause_block}`
is the diagnosis;

    # Semantic correctness review (autorepair automatic verification)

    A patch was generated to repair the violation below. Judge whether the patched code now
    satisfies the source operator definition it was meant to conform to.

    ## Diagnosed violation
    {cause_block}

    ## Source requirement (the behaviour the patched code must satisfy)
    {requirement}

    ## Original code (before the patch)
    ```{code_lang}
    {original}
    ```

    ## Patched code (the candidate under review)
    ```{code_lang}
    {fixed}
    ```

    ## Judging criteria (any hit -> satisfied=false)
    1. The patched behaviour still violates the requirement above;
    2. The patch touches only the area the diagnosis named but does not restore the required
       behaviour;
    3. The patch introduces a new conflict between the patched behaviour and the requirement;
    4. The patch suppresses the symptom without satisfying the requirement.

    Do not report a violation for behaviour the requirement does not cover, and do not
    restate code-validity problems (nonexistent or misused APIs) here.

    ## Output (strict JSON, only JSON)
    ```json
    {"satisfied": true, "feedback": "", "conflicts": []}
    ```

Its system prompt is `You are a meticulous {backend} frontend converter developer
reviewing whether a patch makes a converter satisfy the operator semantics its source
framework defines. Return only the JSON object.`

The check runs only when a source requirement was supplied; a candidate passes it only
when the review answers `satisfied: true`. An unparsable review is reported as
inconclusive and stops the loop rather than being counted as a failure.

### 5.4 Correction prompt

Sent when either check found a problem. Each code validity problem is rendered as
`❌ **nonexistent API**: <name> -- <where>`, `⚠ **suspected misuse**: <name> -- <issue>` or
`❌ **syntax error**: <diagnostics>` before being listed; the unresolved-requirement section
is present only when the semantic review rejected the candidate. The syntax check itself is
a tool invocation, not a prompt, so it has no template here.

    # Fix the problems in the previous repair round (autorepair automatic
    verification feedback)

    Your previous repair round has the problems below. Fix each one, then re-output the
    revised **complete code**.

    ## Detected problems
    {issues_text}

    ## Unresolved requirement (semantic review)
    {semantic_feedback}

    ## Original code (the fix scope; the output must stay aligned to this scope, keep every
    other line byte-for-byte)
    ```{code_lang}
    {original}
    ```

    ## Your previously output fixed code (contains the above problems)
    ```{code_lang}
    {previous_fixed}
    ```

    ## Correction constraints
    1. Modify only the lines directly related to the above problems; preserve every other
       line (comments, blank lines, indentation, function signatures) byte-for-byte.
    2. Only use APIs that actually exist in this version; do not introduce a nonexistent
       API or misuse an API again.
    3. For a nonexistent API: use an equivalent existing API or an implementation approach
       already used in the codebase.
    4. For a misused API: fix the parameter types/count/semantics or the return handling
       per the suggestion.
    5. If an unresolved requirement is listed above, change the behaviour so that the patched
       code satisfies it.

    ## Output format
    A one-sentence explanation outside the code block + a ```{code_lang} code block
    (the corrected complete code)

It reuses the misuse-review system prompt above.
