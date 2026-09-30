# Frontend Converter Consistency Audit — tool and experimental setup

This repository provides the tool and the experimental setup for our work on auditing
the **frontend converters** of AI inference tool stacks for consistency: the same
operator is implemented once per frontend (ONNX / PyTorch / Paddle), those
implementations are compared, and the differences that survive a code-level comparison
are checked against each framework's own documentation, so a genuine compliance bug can
be told apart from a specification gap and from a false positive.

Two audit pipelines and one repair component are shipped:

| | Tool stack | Language | Frontends |
|---|---|---|---|
| `tvm/` | TVM Relax | Python | onnx / torch |
| `openvino/` | OpenVINO | C++ | onnx / torch / paddle |
| `autorepair/` | repair and API verification, both tool stacks | | |

## Contents

| Document | Covers |
|---|---|
| **this file** | the method, every prompt verbatim, and the walkthrough from a bare checkout to results |
| [`tvm/README.md`](tvm/README.md) | the TVM pipeline: what it reads, how extraction and expansion work for Python frontends, how to run it |
| [`openvino/README.md`](openvino/README.md) | the OpenVINO pipeline: what it reads, how extraction and expansion work for C++ frontends, how to run it |
| [`autorepair/README.md`](autorepair/README.md) | the repair component: inputs, usage, output, and its verification backend |

---

## 1. Method

### 1.1 Converters are paired, then judged twice

Step 1 extracts the converter implementation of every operator from each frontend of the
tool stack under audit, and pairs same-name operators across frontends by text
similarity on the operator name (LCS ≥ 0.85). Each pair then goes through two LLM
judgments:

- **Prompt 1 — code consistency.** The two implementations are put side by side and the
  LLM answers whether they are logically equivalent.
- **Prompt 3 — documentation-based deep audit.** Run only for pairs judged *not*
  equivalent. Each implementation is read against its own framework's documentation, the
  full semantic constraint set is extracted on both sides, and the difference is
  attributed to Bug / Standard Gap / Optimization Difference / Inconclusive.

### 1.2 Code expansion sits between them

Many converters are thin wrappers: the body delegates to a helper defined in another
file, or to a base class. Given only the wrapper, the LLM reads the delegation as missing
logic and reports a false Bug, so Step 2 asks the LLM which external definitions a
converter needs (Prompt 2), locates those definitions in the source tree deterministically
— regex and brace matching for C++, AST for Python — and appends them under a comment
banner before the original body. Both sides of a pair are expanded, so both judgments see
code with complete semantics.

### 1.3 Repair

`autorepair/` takes an attributed root cause, the code to fix and the relevant
documentation, asks the LLM for the fixed code, generates the patch itself with
`difflib` (the prompt never asks the LLM for a diff format), and then verifies the APIs
the fix uses before accepting it.

---

## 2. Reproducibility

### 2.0 Repository layout

    .
    ├── tvm/                    audit pipeline for TVM          (Python frontends)
    ├── openvino/               audit pipeline for OpenVINO     (C++ frontends)
    ├── autorepair/             repair + API verification
    ├── requirements.txt        pinned dependencies
    └── README.md

The two pipelines are independent, self-contained copies: each has its own `main.py`,
`llm_client.py` and helper modules, so either can be run without the other. Files the
runs read or write are **not** shipped and are listed as you go below.

### 2.1 Build the environment

Python ≥ 3.9 (`autorepair/tvm/api_index.py` uses `ast.unparse`).

    pip install -r requirements.txt

The pins matter: the audit verdicts come from an LLM, so a different client version is a
different experiment. See 2.6.

### 2.2 Supply the inputs

Three things are needed before a run; none of them are in this repository.

**(a) A source checkout of the tool stack.** The pipeline reads this tree directly —
it never imports the framework.

- **OpenVINO**: the `openvino` repository, any recent checkout. The pipeline reads
  `src/frontends/{onnx,pytorch,paddle}/…/op`.
- **TVM**: the `apache/tvm` repository. The pipeline reads `python/tvm/relax/frontend`,
  which holds the ONNX and PyTorch frontends.

**(b) Documentation dumps.** Step 4 looks up each operator in its framework's reference
documentation, one plain-text dump per frontend. Build them and place them like this:

    openvino/docxes/  onnxdoc.txt  torchdoc.txt  paddledoc.txt  jaxdoc.txt
    tvm/docxes/       onnxdoc.txt  torchdoc.txt

  The format is: operator reference pages concatenated, each block introduced by its URL
  and separated from the next by a line of exactly 50 `=` characters.

      operator reference dump
      ==================================================
      https://example.invalid/operators/Add.html
      ==================================================
      Add computes element-wise addition of two inputs. Both inputs must have the
      same shape (numpy-style broadcasting since opset 7).
      ==================================================
      https://example.invalid/operators/Sub.html
      ==================================================
      Sub computes element-wise subtraction. Inputs must share a shape.
      ==================================================

  The operator name is taken from the URL's last path segment, so the URL has to end in
  the operator name (`.../Add.html`, or `.../add__Add.html` under the ONNX convention).
  Lookups are lenient — names are lowercased, underscores and trailing digits stripped,
  and a small table of cross-framework variants (`cum_sum`/`cumsum`, `reshape2`/`reshape`,
  `BatchNormalization`/`batch_norm`, …) is consulted — and the highest-scoring block
  wins. A frontend with no dump is not an error: Step 4 records the pair as
  documentation-not-matched, which is flag `8` in Section 4.

**(c) An API key.** Each client class in `llm_client.py` ships with the placeholder
`[Your own API key]`; replace it with your key for that provider. `autorepair/` also
accepts `DEEPSEEK_API_KEY` from the environment.

### 2.3 Point the code at your checkouts

The pipelines are the authors' working copies with machine-specific paths replaced by
placeholders. Nothing runs until they are filled in:

| Placeholder | In | Replace with |
|---|---|---|
| `[your tvm source root]` | `tvm/main.py`, `tvm/extract_function.py`, `autorepair/tvm/api_index.py` | the `python/tvm` directory of your TVM checkout |
| `[your openvino source root]` | `openvino/main.py` | the root of your OpenVINO checkout, i.e. the directory holding `src/frontends/` |
| `[this directory]` | `tvm/main.py`, `openvino/main.py`, `tvm/batch_run.py`, `openvino/batch_run.py` | the absolute path of the directory that file lives in; it is how `docxes/` and `results/` are found |
| `[Your own API key]` | `tvm/llm_client.py`, `openvino/llm_client.py`, `autorepair/llm_client.py` | your key for that provider |

### 2.4 Run an audit

Model numbers, used by `run_with_model.py` and by `batch_run.py`:

| # | Model | Endpoint |
|---|---|---|
| 1 | DeepSeek Chat | `api.deepseek.com` |
| 2 | Qwen3.7-Max | DashScope |
| 3 | Qwen3.5-Flash | DashScope |
| 4 | DeepSeek v4 Flash | `api.deepseek.com` |
| 5 | GPT-5.4-mini | kamiapi.top |

Pick a tool stack and run it. Interactive, choosing the model from a menu:

    python tvm/main.py
    python openvino/main.py

Non-interactive, one model, one run:

    python tvm/run_with_model.py 4
    python openvino/run_with_model.py 4

The batch runner used for the paper: three models (4 = DeepSeek v4 Flash, 3 =
Qwen3.5-Flash, 5 = GPT-5.4-mini), five runs each, the three models in parallel:

    python tvm/batch_run.py
    python openvino/batch_run.py

It streams everything to `batch_run.log` in the same directory. A full run is
LLM-bound: each pair costs one Prompt 1 call, and every pair that is judged
non-equivalent costs a Prompt 2 call on each side plus a Prompt 3 call.

### 2.5 Read the results

Each run writes, next to the pipeline:

    <tvm|openvino>/results/<model_name>/<timestamp>/
        pairs_result.txt      one line per operator pair:  <pair> <code_match> <doc_match>
        results_detail.txt    the full log: the Prompt 1 input and answer, the doc
                              retrieval result, and the Prompt 3 output

`pairs_result.txt` is the table the paper's numbers are counted from; the third field is
the verdict code, defined in Section 4. For example, the bug candidates of a run are the
lines with `doc_match == 0`:

    awk '$3 == 0 {print $1}' tvm/results/deepseek-v4-flash/<timestamp>/pairs_result.txt

### 2.6 Run the repair

`autorepair/` is driven by hand. Its inputs:

| Input | Description |
|---|---|
| `backend` | tool stack under repair: `TVM` or `OPENVINO` |
| `frontend` | frontend: onnx / torch / paddle … |
| `op` | operator name, e.g. `Flatten` |
| `code` | the original code to fix — only the part related to the buggy operator |
| `doc` | the documentation relevant to the fix (optional) |
| `cause` | the root cause attributed by the audit (optional; without it the repair model reasons from the code and docs itself) |
| `--src-file` | path of the real source file (optional): anchors the patch to it so the line numbers are real |

    cd autorepair

    # build the prompt and print it, without calling the model
    python repair.py --dry-run

    # fully interactive: enter each input, multiline values ended by a line EOF
    python repair.py

    # non-interactive
    python repair.py --backend TVM --frontend onnx --op Flatten \
                     --code-file code.py --doc-file doc.txt --cause-file cause.txt \
                     --src-file path/to/onnx_frontend.py

Pass `--src-file` as a source-root-relative path: the patch header is rewritten to the
same relative form, so it applies with `patch -p1` from the source root. Each case goes to
`repairs/{backend}_{frontend}_{op}/`: `prompt.txt` is written as soon as the prompt is
built — `--dry-run` stops there — and a real run adds `response.md`, `fixed.py`,
`fix.patch` and, unless `--no-verify` was given, `api_check.md`. The script prints
`applyable=`;
`True` means the patch carries real line numbers. Being applyable does not make a fix
correct — verify it independently by differential or reproduction testing.

API verification runs automatically after the patch is generated (Section 5). It needs a
source tree to scan:

    export TVM_PYTHON_ROOT=/path/to/tvm/python/tvm       # for --backend TVM
    export OPENVINO_SRC_ROOT=/path/to/openvino           # for --backend OPENVINO

or pass `--api-root`. Without one, the static half is skipped. `--no-verify` skips
verification entirely; `--no-review` keeps the deterministic check but drops the LLM
misuse review; `--verify-rounds N` bounds the correction rounds (default 2). You can
exercise the index and the resolution logic without any LLM call:

    python api_check.py --self-test tvm
    python api_check.py --self-test openvino

---

## 3. Prompts

All four audit calls share one system prompt. `{...}` marks a value substituted at call
time. Where the two tool stacks differ — which is only in Prompt 2 at any length, plus
four spots in Prompt 3 — both versions are given.

### 3.0 System prompt (all calls)

    You are a compiler expert. Compare code logic carefully.

### 3.1 Prompt 1 — code consistency comparison (Step 3)

    Are the following two function/class logic equivalent?
    The converter function/class in {ToolStack} {source_front} frontend is:
    ```{CodeLanguage}
    {source_func}
    ```

    The converter function/class in {ToolStack} {target_front} frontend is:
    ```{CodeLanguage}
    {target_func}
    ```

    Output format:
    - If equivalent, end with: [JUDGMENT] EQUIVALENT
    - If not equivalent, end with: [JUDGMENT] NONEQUIVALENT

The prompt is skipped, and the pair flagged `9`, when it exceeds 10,000 words.

### 3.2 Prompt 2 — dependency identification (Step 2)

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

### 3.3 Prompt 3 — documentation-based deep audit (Step 5)

Shown for TVM. The OpenVINO version differs in exactly four places: the two code fences
say `cpp`, the opening sentence names OpenVINO and reads "the implementation … fully
conforms to its official standard specification", the audit task names OpenVINO, and the
example in criterion 3 is `common_translators::xxx, utils::xxx, etc.` instead of
"utility functions in common.py under relax/frontend, base classes in onnx_frontend.py".

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

`{code_section}` is the two expanded implementations, fenced with `{CodeLanguage}`.

### 3.4 Repair prompts

**Fix prompt** (`autorepair/repair.py`). Note that it asks for code, never for a diff:

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

Its system prompt is `You are a meticulous {backend} frontend converter developer. Write
minimal surgical fixes: preserve every untouched line byte-for-byte, never rename or
restructure code, and never use APIs that may not exist in this {backend} version.`

**Misuse review** (`autorepair/api_check.py`). Given the original code, the added lines,
and for each API the declaration and comment block found at its definition site, the
model answers with JSON. Its criteria are: a parameter name, type or count that does not
match the signature; a return type or semantics the call site does not expect; a
deviation from codebase conventions; or an API that does not exist in this version.

**Correction prompt** (`autorepair/api_check.py`). Lists the problems found in the
previous round with their evidence, repeats the original code and the previous output,
and constrains the model to modify only the lines related to those problems, to use only
APIs that exist, and to re-output the complete code.

---

## 4. Verdict flags

`pairs_result.txt` carries two flags per pair, `code_match` and `doc_match`.

First round, code consistency:

| LLM output | Meaning | `code_match` |
|---|---|---|
| `[JUDGMENT] EQUIVALENT` | logically equivalent, no second round | 1 |
| `[JUDGMENT] NONEQUIVALENT` | not equivalent, go to the second round | 0 |
| (oversize skip) | prompt over 10,000 words | 9 |

Second round, documentation audit:

| Final conclusion | Meaning | `doc_match` |
|---|---|---|
| `[Bug]` | not aligned with its own documentation — **bug candidate** | 0 |
| `[Standard Gap]` / `[Optimization Difference]` | specification or optimization difference, not a bug | 1 |
| (documentation not found) | no matching operator in the dump | 8 |
| `[Inconclusive]` / unparsed | insufficient evidence | 9 |

Pairs that passed the first round are written as `(1, 1)` — their `doc_match` is a
pass-through and does not mean "standard gap". Count non-bug differences with
`code_match == 0 and doc_match == 1`, or they are mixed in.

---

## 5. API verification in `autorepair/`

After a patch is generated, the calls on its added lines are resolved against a static
index of the tool stack's own source tree — read, never imported, since these frameworks
are normally not importable in the analysis environment. Two shapes are extracted: for
TVM, dotted calls (`relax.op.add(`); for OpenVINO, `::`-qualified calls (`ov::op::v1::Add(`,
`v1::Add(`) plus bare frontend helper names and the type argument of
`std::make_shared<v0::Clamp>`. Method calls (`node.get_ov_inputs()`) and names the snippet
defines itself are skipped.

Each call resolves to one of three states:

- **ok** — it exists, and its declaration goes to the LLM for the misuse review;
- **missing** — the namespace is indexed and complete enough for this verdict, so the
  call is a fabrication;
- **unverifiable** — the index cannot settle it. Reported instead of a guess, so a real
  API that merely was not enumerated is never called fabricated. The OpenVINO backend
  answers this for a call qualified by a short name (`detail::conv(`): that qualifier is
  relative to the file the snippet came from, which the verifier never sees.

Anything found missing or misused is fed back for a correction round, up to
`--verify-rounds`.

The index is built from the checkout named by `TVM_PYTHON_ROOT` or `OPENVINO_SRC_ROOT`
(2.6). For TVM it comes from the package re-exports in each `__init__.py` plus the defs
in the package's own modules; for OpenVINO, from C++ declaration sites — the operator
headers under `src/core/include/openvino/op/`, where the header's innermost `namespace vN`
is the opset the converters refer to as `v1::Add`, the rest of
`src/core/include/openvino/`, and the frontend helpers under
`src/frontends/{onnx,pytorch,paddle,common_translators}`.

**Verification boundary.** It checks that an API exists and is used consistently with its
signature; it does not catch a semantic regression in a shared code path — a fix that
hard-codes a base class's behaviour for one operator still passes. That belongs to the
fix prompt's scope contract, not to this check.

---

## 6. Code structure

    README.md                       this file
    tvm/                            audit pipeline, TVM (Python frontends)
      README.md                     the TVM pipeline in more detail
      main.py                       pipeline: Steps 0-5
      extract_function.py           Step 1  parse convert_map + AST
      code_expander.py              Step 2  dependency expansion (Prompt 2)
      consistency_checker.py        Step 3  consistency comparison (Prompt 1)
      doc_retriever.py              Step 4  documentation lookup
      doc_analyzer.py               Step 5  documentation audit (Prompt 3)
      llm_client.py                 model clients behind one proxy
      batch_run.py                  batch runner
      run_with_model.py             non-interactive single run

    openvino/                       audit pipeline, OpenVINO (C++ frontends)
      README.md                     the OpenVINO pipeline in more detail
      ...                           same module layout as tvm/

    autorepair/                     repair component
      README.md                     inputs, usage, output, verification
      repair.py                     prompt → model → code → patch → verification
      api_check.py                  verification driver
      tvm/api_index.py              index and call extraction for TVM
      openvino/api_index.py         index and call extraction for OpenVINO
      llm_client.py                 one client, with backoff

Not shipped, because you supply them or the runs produce them:

    <tvm|openvino>/docxes/          documentation dumps read by Step 4      (2.2b)
    <tvm|openvino>/results/         audit results, one directory per run    (2.5)
    <tvm|openvino>/batch_run.log    batch runner log                        (2.4)
    autorepair/repairs/             repair cases                            (2.6)
