# Frontend Converter Consistency Audit Method (LLM-Based Two-Stage Audit)

## 1. Method Overview

This method uses a large language model (LLM) to audit the **frontend converters (Converter)** of AI inference deployment and acceleration tool stacks for consistency: it automatically extracts the converter implementations of the same operator across different frontends (ONNX / PyTorch / Paddle, etc.), first runs an LLM-based **code-level consistency comparison**, and for operator pairs judged "inconsistent," retrieves the official documentation for a **documentation-based deep audit**. The difference is finally classified as Bug / Standard Gap / Optimization Difference / Inconclusive, thereby locating potential compliance bugs in the frontend implementation while suppressing false positives.

Beyond detection, the method also includes an automated repair component — **autorepair** (`autorepair/`) — that feeds the LLM-attributed error cause, the original code to fix, and key documentation back into the LLM to produce a fix patch, with deterministic patch generation and automatic API verification (see Section 5).

The audit method has been deployed and validated on two tool stacks:

| Test Object | Source Language | Covered Frontends |
|---|---|---|
| `openvino/` | C++ (`cpp`) | onnx / torch / paddle |
| `tvm/` | Python (`python`) | onnx / torch |

## 2. Audit Pipeline

| Step | Action | Key Output |
|---|---|---|
| 0 | Select the LLM backend (DeepSeek / Qwen / GPT) | — |
| 1 | Extract converters: parse the `convert_map` / API mapping table in the frontend source, extract the function/class implementations of same-name operators, and pair them by text similarity (>= 0.85) | Operator pairs |
| 2 | Code expansion: the LLM determines whether the implementation depends on external helper code (Prompt 2), then searches and appends the dependency implementations in the source | Code with complete semantics |
| 3 | First judgment: the LLM directly compares whether the two pieces of logic are equivalent (Prompt 1) | `[JUDGMENT]` |
| 4 | Documentation retrieval: look up the documentation block matching the operator name in that frontend's documentation dump (`docxes/*.txt`, which you supply — see 7.3) | Documentation snippet |
| 5 | Second judgment: the LLM combines both pieces of code with both documents, extracts the complete semantic constraint set, and classifies the difference per the audit criteria (Prompt 3) | `[Final Conclusion]` |

Steps 4 and 5 run only when Step 3 judges "not equivalent"; equivalent pairs pass directly.

## 3. Prompts (Core)

All LLM calls share the same system prompt; the main body consists of three prompt templates. The **generic templates** below use `{...}` placeholders; the differences between the two test objects are limited to how the placeholders are filled in (see 3.4).

### 3.0 System Prompt (Shared)

    You are a compiler expert. Compare code logic carefully.

### 3.1 Prompt 1: Code Consistency Comparison (Step 3)

    Are the following two function/class logic equivalent?
    The converter function/class in {ToolStack} {FrontendA} frontend is:
    ```{CodeLanguage}
    {Function/Class A Code (with Expanded Dependencies)}
    ```

    The converter function/class in {ToolStack} {FrontendB} frontend is:
    ```{CodeLanguage}
    {Function/Class B Code (with Expanded Dependencies)}
    ```

    Output format:
    - If equivalent, end with: [JUDGMENT] EQUIVALENT
    - If not equivalent, end with: [JUDGMENT] NONEQUIVALENT

> Prompts exceeding 10,000 tokens are skipped (flagged as `skipped`).

### 3.2 Prompt 2: Dependency Code Identification (Step 2)

    You are a {ToolStack} compiler frontend code analysis expert. Please analyze whether the operator conversion code of the following {Frontend} frontend requires helper code defined in other files to gain complete semantics:

    ```{CodeLanguage}
    {OperatorCode}
    ```

    [Analysis Rules]
    1. Check whether the function/class body calls helper functions or base classes defined in other files (excluding the standard library and framework built-ins).
    2. If it is a class definition `class X(BaseY)`, check whether the base class `BaseY` needs to be expanded; if `BaseY` itself inherits from other classes, list them as well.
    3. Pay special attention to "thin wrappers": a function that directly `return`s the result of another function, or delegates its core logic to a shared utility function / base class.
    4. Do NOT list: the standard library ({StandardLibraryExamples}), or framework built-ins ({FrameworkBuiltinExamples}).
    5. External dependencies may be defined in: other files in the same operator directory, or shared modules under the frontend root ({SharedHelperExamples}).
    6. Only list dependencies essential for understanding the core logic.

    [Output Format]
    If there are key external dependencies, output one line: `[DEPS]: dep_name1, dep_name2`
    If not needed, output: `[DEPS]: NONE`

### 3.3 Prompt 3: Documentation-Based Deep Audit (Step 5)

    You are a compiler compliance audit expert, responsible for verifying that the deep learning frontend converters (Converter) in the AI inference deployment and acceleration tool stack ({ToolStack}) fully conform to their official standard specifications.

    [Audit Task]
    We detected an inconsistency between frontend A and frontend B in {ToolStack}.
    [Preliminary Code Difference Conclusion]:
    {First-Round Consistency Conclusion & Explanation}
    {Code Snippets (with Expanded Dependencies)}

    [Reference Documents]
    Frontend A framework documentation: {Frontend A Documentation}
    Frontend B framework documentation: {Frontend B Documentation}

    [Audit Criteria (strictly enforced to prevent false positives)]
    1. [Hard prerequisite for a Bug verdict]: Only when you can point out, in the code, a [concrete behavior] of one side that deviates from the behavior defined in its own documentation, on [inputs that conform to its own documentation] (e.g., a missing branch, an unhandled pattern, a parameter silently ignored and producing wrong output), may you mark [Bug]. You must cite a specific code location or construct.
    2. [An extension is not a Bug]: If one side's implementation supports types, input forms, or attributes beyond its documented specification (i.e., the implementation is a [superset] of the specification), and behaves correctly on standard inputs conforming to the specification, this is an [intentional extension / compatibility handling]; mark [Standard Gap] or [Optimization Difference], and [must NOT] mark [Bug].
    3. [Delegation / thin wrapper is not a Bug]: If one side's implementation is merely a thin wrapper that delegates its core logic to a shared utility function or base class (e.g., {SharedHelperExamples}), and that utility function's implementation is not included in the code provided, so you [cannot confirm its logic], you [must NOT] mark [Bug] merely because you "did not see the implementation." Mark [Inconclusive] or [Standard Gap].
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

### 3.4 Instantiating the Generic Templates (OpenVINO vs TVM)

| Placeholder | OpenVINO | TVM |
|---|---|---|
| {ToolStack} | OpenVINO | TVM (Relax) |
| {CodeLanguage} | cpp | python |
| {StandardLibraryExamples} | `std::`-prefixed functions | os / math / typing, etc. |
| {FrameworkBuiltinExamples} | `ov::`-prefixed classes, NodeContext, etc. | relax.op, tvm, tir, etc. |
| {SharedHelperExamples} | `utils.cpp` / `utils.hpp` under the frontend src (get_inputs_with_promoted_types, normalize_axis, etc.), shared directory `common_translators` (e.g., `common_translators::translate_atan2_util`) | utility functions in `common.py` under `relax/frontend`, base classes in `onnx_frontend.py` (e.g., `BinaryBase`) |
| Code extraction method | Scan `.cpp` files by directory | Parse `convert_map` + AST to extract functions/classes |

All other rules, execution steps, and output formats are identical and require no changes.

## 4. Verdicts and Result Flags

### First Round (Code Consistency, code_match)

| LLM Output | Meaning | Flag |
|---|---|---|
| `[JUDGMENT] EQUIVALENT` | Logically equivalent | 1 |
| `[JUDGMENT] NONEQUIVALENT` | Not equivalent, proceed to the second round | 0 |
| (Oversize skip) | Prompt > 10,000 tokens | 9 |

### Second Round (Documentation Deep Audit, doc_match)

| Final Conclusion | Meaning | Flag |
|---|---|---|
| `[Bug]` | One side's implementation not aligned with its own documentation, **Bug candidate** | 0 |
| `[Standard Gap]` / `[Optimization Difference]` | Specification / optimization difference, not a Bug | 1 |
| (Documentation not found) | No matching operator in the doc dump | 8 |
| `[Inconclusive]` / unparsed | Insufficient evidence | 9 |

## 5. Repair: autorepair

**autorepair** (`autorepair/`) is the repair component: it feeds the **error cause attributed by the LLM + the original code to fix + key documentation information** into the LLM, which calls DeepSeek v4 Flash to repair TVM / OpenVINO frontend converter defects.

**Design point**: the prompt **contains no diff-format requirements** — it only asks the LLM to output the 'fixed code'. The patch format (`--- a/` file headers, hunk line numbers) is generated automatically by the program from the real source file with difflib, deterministically, so it does not depend on the LLM following format instructions (avoiding the problem of non-applicable diffs).

### 5.1 Files

| File | Role |
|---|---|
| `repair.py` | Main script: manual input → build prompt → call LLM → code → patch → API verification |
| `api_check.py` | API verification: namespace existence check (static scan of the source tree + runtime inspect) + LLM misuse review + correction loop |
| `llm_client.py` | `DeepseekV4FlashClient` + exponential-backoff retry |

### 5.2 Manual inputs

| Input | Description |
|---|---|
| `backend` | Backend: TVM / OPENVINO |
| `frontend` | Frontend: onnx / torch / paddle ... |
| `op` | Operator name: e.g., Flatten |
| `code` | Original code to fix (**only the parts related to the buggy operator, i.e., what needs modification**) |
| `doc` | Key documentation information (**only what is needed for the fix**, optional) |
| `cause` | Error cause attributed by the LLM (optional; if omitted, the repair LLM analyzes it on its own) |
| `--src-file` | Path to the real source file (optional): anchors the patch to the real file and automatically computes line numbers to generate an applicable diff |

### 5.3 Usage

```bash
# 1) Fully interactive: enter each item (multiline code/doc terminated by a line of EOF)
python repair.py

# 2) Only build and print the prompt, no LLM call (inspect the prompt design first)
python repair.py --dry-run

# 3) All arguments given (non-interactive); code/doc/cause can be passed as files
python repair.py --backend TVM --frontend onnx --op Flatten \
                 --code-file code.py --doc-file doc.txt --cause-file cause.txt \
                 --src-file python/tvm/relax/frontend/onnx/onnx_frontend.py
```

`--backend` defaults to `TVM`; if omitted, it is entered interactively. `--code-file` and the other file arguments are mutually exclusive with interactive input. Pass `--src-file` preferably as a **source-root-relative path** (e.g., `python/tvm/relax/frontend/onnx/onnx_frontend.py`); the patch header is converted to a source-root-relative path so it can be applied with `patch -p1` from the source root.

API-verification arguments:
- `--no-verify`: skip API verification entirely (existence check + misuse review + correction loop).
- `--no-review`: only run the static existence check; skip the LLM misuse review/correction (saves LLM calls).
- `--verify-rounds N`: correction round limit (default 2).
- `--tvm-root PATH`: tvm source `python/tvm` directory scanned by the existence check (defaults to the `TVM_PYTHON_ROOT` environment variable; `--no-verify` skips the check).

### 5.4 Output

`repairs/{backend}_{frontend}_{op}/`:

| File | Content |
|---|---|
| `prompt.txt` | The full prompt sent to the LLM (reproducible, no format requirements) |
| `response.md` | Raw LLM response |
| `fixed.py` | The complete code block fixed by the LLM |
| `fix.patch` | The unified diff generated by the program with difflib |
| `api_check.md` | API verification report (per-round existence results / misuse review / correction rounds) |

The script prints `applyable=`: `True` means the patch is anchored to the real source file with correct line numbers (`patch -p1` can apply it); `False` means no `--src-file` was given or the original code snippet was not matched in the source file (falls back to a snippet-level diff). Fix correctness must be verified independently by differential/reproduction testing (see `repairs/TVM_onnx_Flatten/verify_fix.py`).

### 5.5 API verification (automatic; disabled with `--no-verify`)

Against the LLM's **fabrication** of APIs (using APIs that do not exist) and **misuse** (using existing APIs incorrectly), verification + correction runs automatically after the patch is generated:

1. **Extraction**: extract `a.b.c(...)`-style dotted API calls from the **added/changed lines** of the fixed code (the `+` lines of the diff).
2. **Existence check (deterministic, no LLM)**: resolve each call's namespace (e.g., `relax.op.add` → `relax.op`), get the **full API list** of that namespace, and see whether `add` is among them.
   - Because tvm cannot be `import`ed locally, TVM namespaces (`relax.op` / `topi.nn` / `tirx`, 231 / 113 / 320 APIs) build the list by **statically scanning the source tree**: parse each package's `__init__.py` re-exports (`from .X import (a, b, ...)`) plus in-package `def` names (`topi`'s `from .X import *` star exports are expanded by scanning the imported files).
   - Modules that can be imported (numpy / functools / ...) are checked with the standard `inspect` mechanism.
   - Runtime objects such as `attr.` / `bb.` are out of scope (`attr` fabrication is caught by the heuristic check below).
3. **Misuse review (LLM)**: for each used API, extract its signature + docstring (AST, from the definition file), and have the LLM judge, together with the original code and the added lines, whether the parameter names/types/count/semantics match and whether the usage deviates from codebase conventions.
4. **Correction loop**: when there is a 'nonexistent API' or 'suspected misuse', design the detected information (including the namespace's API count, similar APIs, and the LLM's suggestion) into a prompt and feed it back to the LLM, asking it to re-output the complete code after modification, then verify again; iterate until it passes or reaches `--verify-rounds` (default 2).

Regression-verified against known weak points:
- `relax.op.addd` (fabricated) → the static check returns ❌ nonexistent (similar: add); after correction, changed to `relax.op.add`;
- `relax.op.zeros(x_shape)` (missing the required `dtype`) → the review returns ⚠ misuse; after correction, `dtype='float32'` was added;
- `relax.op.zeros(x_shape, x.ty.dtype)` → the review returns ✓ correct (the `zeros` signature is `dtype: str | DataType`; passing a DataType object is legal) — resolving the earlier unverified question of "whether zeros accepts it";
- `attr.get_int_tuple` (fabricated) → caught by the `check_attr_accessors` heuristic (below).

**Verification boundary (no false positives)**: verification targets only "whether the API exists / is misused". It does **not** catch semantic regressions in shared code paths (e.g., the Mean fix hard-coding `cls.numpy_op` to `np.mean`, where `np.mean` itself is real and used correctly) — such issues belong to the prompt-contract layer (which must constrain "not changing the semantics of the shared base class for other operators"), see `repairs/test/SUMMARY.md`.

#### Heuristic check (existing)

If the fixed code introduces `attr.<method>(` calls not used by the original code (e.g., `attr.get_int_tuple()` / `attr.get_int()`), a `[warn]` is printed. Such methods are often fabricated by the LLM (a frontend attr is usually a dict, only `attr.get(...)` exists); when hit, confirm manually, and use `attr.get(...)` if unsure.

### 5.6 Repair cases

`repairs/` holds the repair cases produced by the method:

```
repairs/
├── TVM_onnx_PRelu/
├── TVM_onnx_ConvTranspose/
├── TVM_onnx_Flatten/        # e.g. prompt.txt, response.md, fixed.py, fix.patch, api_check.md, verify_fix.py
├── TVM_onnx_Mean/
├── TVM_onnx_Pad/
└── test/                    # regression tests / summaries (see repairs/test/SUMMARY.md)
```

## 6. Code Structure

    .
    ├── tvm/                       # Test object 1: TVM (Python frontends)
    │   ├── main.py                # Main flow
    │   ├── extract_function.py    # Step 1: Extract converters
    │   ├── code_expander.py       # Step 2: Dependency code expansion (Prompt 2)
    │   ├── consistency_checker.py # Step 3: Consistency comparison (Prompt 1)
    │   ├── doc_retriever.py       # Step 4: Documentation retrieval
    │   ├── doc_analyzer.py        # Step 5: Documentation-based deep audit (Prompt 3)
    │   ├── llm_client.py          # Unified LLM backend interface
    │   ├── batch_run.py           # Batch runner (3 models in parallel)
    │   └── run_with_model.py      # Non-interactive run with a specified model
    ├── openvino/                  # Test object 2: OpenVINO (C++ frontends)
    │   └── ...                    # Same structure as tvm/
    └── autorepair/                # Repair component
        ├── repair.py              # prompt → LLM → code → patch → API verification
        ├── api_check.py           # existence check + misuse review + correction loop
        ├── llm_client.py          # DeepseekV4FlashClient + exponential-backoff retry
        └── repairs/               # Repair cases produced by the method

Not shipped, because they are inputs you supply or outputs you produce:

    <tvm|openvino>/docxes/         # Documentation dumps read by Step 4 (see 7.3)
    <tvm|openvino>/results/        # Audit results, written by main.py, one dir per run

## 7. Setup

### 7.1 Replace the placeholders

This is the authors' working copy. Machine-specific values have been replaced by
placeholders, and nothing runs until you edit them:

| Placeholder | In | Replace with |
|---|---|---|
| `[your openvino source root]` | `openvino/main.py` | Root of an OpenVINO checkout — the directory containing `src/frontends/` |
| `[your tvm source root]` | `tvm/main.py`, `tvm/extract_function.py`, `autorepair/api_check.py` | The `python/tvm` directory of a TVM checkout |
| `[this directory]` | `tvm/main.py`, `openvino/main.py`, `tvm/batch_run.py`, `openvino/batch_run.py` | Absolute path of the directory that file lives in; it is how `docxes/` and `results/` are located |
| `[Your own API key]` | `tvm/llm_client.py`, `openvino/llm_client.py`, `autorepair/llm_client.py` | Your key for the provider that client talks to |

### 7.2 Source checkouts

Both pipelines read the framework source directly rather than importing it:

- **OpenVINO** — a checkout of the `openvino` repository, any recent version. `main.py`
  points at `src/frontends/{onnx,pytorch,paddle}/…/op`.
- **TVM** — a checkout of `apache/tvm`. `main.py` points at `python/tvm/relax/frontend`,
  which is where the ONNX and PyTorch frontends live.

`autorepair/` needs a TVM checkout too, but only for the API existence check; it scans
the tree statically rather than importing it.

### 7.3 Documentation dumps

Step 4 reads one dump per frontend. The dumps are not shipped here — they are large
scraped snapshots of each framework's reference documentation — so build them and place
them next to the code:

    openvino/docxes/  onnxdoc.txt  torchdoc.txt  paddledoc.txt  jaxdoc.txt
    tvm/docxes/       onnxdoc.txt  torchdoc.txt

A dump is plain text: operator reference pages concatenated, each block separated by a
line of exactly 50 `=` characters and introduced by the page's URL. The operator name is
read from the URL's last path segment, so the URL must end in the operator name
(`.../Add.html`, or `.../add__Add.html` under the ONNX convention).

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

Names are matched leniently — lowercased, underscores and trailing digits stripped,
plus a table of known cross-framework variants such as `cum_sum`/`cumsum` — and the
highest-scoring block wins. A frontend whose dump is missing is not an error: Step 4
reports the pair as documentation-not-matched, which is flag `8` in Section 4.

### 7.4 Dependencies

    pip install openai python-Levenshtein

`openai` is used by both pipelines and by `autorepair/`; `python-Levenshtein` by
`main.py`, for operator-name similarity.

### 7.5 API keys

Keys are not stored in this repository. Each client class in `llm_client.py` ships with
the `[Your own API key]` placeholder from 7.1 to replace. `autorepair/llm_client.py`
additionally reads `DEEPSEEK_API_KEY` from the environment and falls back to the same
placeholder.

## 8. Running

    # TVM
    python tvm/main.py                      # interactive model selection
    python tvm/run_with_model.py 4          # non-interactive, model 4
    python tvm/batch_run.py                 # 3 models x 5 runs, in parallel

    # OpenVINO
    python openvino/main.py
    python openvino/batch_run.py

    # Repair
    cd autorepair
    python repair.py --dry-run                          # build the prompt, no LLM call
    python repair.py --backend TVM --frontend onnx --op Flatten \
                     --code-file code.py --src-file path/to/onnx_frontend.py
