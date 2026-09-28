# Frontend Converter Consistency Audit Method (LLM-Based Two-Stage Audit)

## 1. Method Overview

This method uses a large language model (LLM) to audit the **frontend converters (Converter)** of AI inference deployment and acceleration tool stacks for consistency: it automatically extracts the converter implementations of the same operator across different frontends (ONNX / PyTorch / Paddle, etc.), first runs an LLM-based **code-level consistency comparison**, and for operator pairs judged "inconsistent," retrieves the official documentation for a **documentation-based deep audit**. The difference is finally classified as Bug / Standard Gap / Optimization Difference / Inconclusive, thereby locating potential compliance bugs in the frontend implementation while suppressing false positives.

Beyond detection, the method also includes an automated repair component — **autorepair** (`autorepair/`) — that feeds the LLM-attributed error cause, the original code to fix, and key documentation back into the LLM to produce a fix patch, with deterministic patch generation and automatic API verification (see Section 5).

The audit method has been deployed and validated on two tool stacks. Everything that is genuinely specific to a tool stack lives in one module per backend under `frontends/`; the pipeline itself is shared and selects a backend with `--backend`.

| Backend (`--backend`) | Source Checkout | Language | Covered Frontends |
|---|---|---|---|
| `openvino` | OpenVINO, `src/frontends/` | C++ | onnx / torch / paddle |
| `tvm` | TVM, `python/tvm/relax/frontend` | Python | onnx / torch |

## 2. Audit Pipeline

| Step | Action | Key Output |
|---|---|---|
| 0 | Select the LLM backend (DeepSeek / Qwen / GPT) | — |
| 1 | Extract converters: parse the `convert_map` / API mapping table in the frontend source, extract the function/class implementations of same-name operators, and pair them by text similarity (>= 0.85) | Operator pairs |
| 2 | Code expansion: the LLM determines whether the implementation depends on external helper code (Prompt 2), then searches and appends the dependency implementations in the source | Code with complete semantics |
| 3 | First judgment: the LLM directly compares whether the two pieces of logic are equivalent (Prompt 1) | `[JUDGMENT]` |
| 4 | Documentation retrieval: look up the documentation block matching the operator name in that frontend's official documentation dump | Documentation snippet |
| 5 | Second judgment: the LLM combines both pieces of code with both documents, extracts the complete semantic constraint set, and classifies the difference per the audit criteria (Prompt 3) | `[Final Conclusion]` |

Steps 4 and 5 run only when Step 3 judges "not equivalent"; equivalent pairs pass directly.

### 2.1 Step 2 in detail: code expansion

Many converters are thin wrappers: the body is a line or two, and the real logic lives in a helper function or base class defined in **another file**. OpenVINO's `flip.cpp` is `return reverse_op(node);`, with the logic in `reverse_op`; TVM's `class Add(BinaryBase)` inherits its body from `BinaryBase`. Given only the wrapper, the LLM cannot see the implementation and reads the delegation as missing logic, which surfaces as a **false Bug**. Code expansion collects the definitions the converter actually calls and places them next to the original body, so the comparison sees a self-contained block.

It runs in three stages:

1. **Name the dependencies** (`analyze_dependencies_with_llm`). The body goes to the LLM (Prompt 2), which must answer with a single line, `[DEPS]: dep1, dep2`. Transitive dependencies are flattened into that line: a base class that itself inherits, and the shared utility behind a thin wrapper, must all be named.
2. **Find the definition sites deterministically** (`search_dependency_code`). Candidate directories are searched in priority order — the operator directory, the frontend root, `utils`/`core`/`common`, and for OpenVINO the shared `common_translators` tree — because helpers often live outside the operator directory. What is returned must be a *definition* rather than a call site, and only a bounded snippet is taken instead of a whole file, to keep the prompt small.
3. **Assemble one self-contained block** (`expand_code_body`). The dependency code is placed **before** the original function and delimited by comments rather than inlined into the body:

```cpp
// ===== BEGIN: Dependency code (for complete semantics) =====
// ===== Dependency: BinaryBase =====
<BinaryBase source segment>
...
// ===== END: Dependency code =====
// === Original code ===
<original converter body>
```

Files already read are skipped, a dependency that cannot be found logs `[WARN]` without aborting the run, and the function returns `(expanded_code, dep_str)`. Both the source and the target converter are expanded, so both rounds of judgement see complete semantics.

The responsibilities are split deliberately: deciding *which* dependencies exist is a semantic judgement and is left to the LLM, while deciding *where to find them and how much to take* is deterministic code — a regex heuristic for C++, an AST match for Python — so the LLM never invents code.

The C++/Python split of stage 2 lives in `frontends/`: comment prefix, name separator, file extensions, search directories, snippet extractor. The driver is shared.

## 3. Prompts (Core)

All LLM calls share the same system prompt; the main body consists of three prompt templates, defined in `prompts.py`. Text that differs between tool stacks is written as a `[Placeholder]` token and filled at call time from the active frontend module's `PROMPT_VALUES` (see 3.4). The templates are shown below with the placeholders left unfilled.

### 3.0 System Prompt (Shared)

    You are a compiler expert. Compare code logic carefully.

### 3.1 Prompt 1: Code Consistency Comparison (Step 3)

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

> Prompts exceeding 10,000 words are skipped (flagged as `skipped`).

### 3.2 Prompt 2: Dependency Code Identification (Step 2)

    You are a [ToolStack] compiler frontend code analysis expert. Please analyze whether the [OperatorNoun] of the following {front_name} frontend requires helper code defined in other files to gain complete semantics:

    ```[CodeLanguage]
    {code_body}
    ```

    [Analysis Rules]
    [AnalysisRules]

    [Output Format]
    If there are key external dependencies, output one line: `[DEPS]: dep_name1, dep_name2`
    If not needed, output: `[DEPS]: NONE`

> The rule block differs substantially between backends — OpenVINO has five rules, TVM six, and the wording of the call-detection rule differs — so it is carried whole in `[AnalysisRules]` rather than patched word by word. The stock rule blocks are `frontends/openvino.py` and `frontends/tvm.py`.

### 3.3 Prompt 3: Documentation-Based Deep Audit (Step 5)

    You are a compiler compliance audit expert, responsible for verifying that the frontend converters of the AI inference deployment and acceleration tool stack ([ToolStack]) fully conform to their official standard specifications.

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
    Compare the constraint sets extracted in Step 1 against the code:
    1. Logic coverage self-check: Does code A cover every point in constraint set A? Does code B cover every point in constraint set B?
    2. Classify the nature of the difference according to the [Audit Criteria] above: Bug / Standard Gap / Optimization Difference / Inconclusive.

    [Output Format]
    1. [Full constraint comparison table]: Briefly describe the complete behavior required by documents A and B.
    2. [Implementation defect identification]: Point out which side's implementation fails to align with its own documentation constraints, and [cite the specific code location or construct]; if there is no defect, state so explicitly.
    3. [Final conclusion]:
    Format requirement (must end with this): [Final Conclusion]: [Bug] or [Standard Gap] or [Optimization Difference] or [Inconclusive].

### 3.4 Placeholder Values per Backend

A backend declares its substitutions in `PROMPT_VALUES`, and `prompts.render()` applies them. The values shipped here:

| Placeholder | `openvino` | `tvm` |
|---|---|---|
| `[ToolStack]` | `OpenVINO` | `TVM Relax` |
| `[CodeLanguage]` | `cpp` | `python` |
| `[OperatorNoun]` | `operator code` | `operator conversion code` |
| `[AnalysisRules]` | 5 rules | 6 rules |
| `[SharedHelperExamples]` | `common_translators::xxx, utils::xxx, etc.` | `utility functions in common.py under relax/frontend, base classes in onnx_frontend.py, etc.` |

The rule blocks are stored verbatim in `frontends/openvino.py` and `frontends/tvm.py`, so each backend is judged under the same prompt contract as before the merge. The two blocks differ in more than wording: the OpenVINO block has no base-class expansion rule (C++ has no equivalent pattern) and no closing "only list what is essential" rule, while its call-detection rule carries a concrete example, `return reverse_op(node);`, that the TVM block does not. Carrying the whole block in one placeholder keeps those differences visible in one place instead of scattering them across the template.

### 3.5 Backend Behaviour Outside the Prompts

Prompts are not the only per-backend difference. `extract_function.py` and `code_expander.py` are shared drivers that delegate the following to the backend module:

| Concern | `openvino` | `tvm` |
|---|---|---|
| Converter discovery | scan each operator directory for `.cpp` files | parse the `convert_map` literal, then take the named `FunctionDef` / `ClassDef` bodies |
| Helper lookup order | `.cpp`, then `.hpp` / `.h` | `.py` |
| Definition test | regex heuristic over call context, then brace matching | AST node match (`ClassDef` / `FunctionDef` / module-level `Assign`) |
| Name separator | `::` | `.` |
| Comment prefix | `//` | `#` |

Steps 1 and 2 are the only stages that need this; the remaining stages operate on prompt text and code strings and are backend-agnostic.

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

Flags actually written to `pairs_result.txt` (as `(code_match, doc_match)`):

- Round 1 `EQUIVALENT` → `(1, 1)`. These pairs never reach round 2, so their `doc_match = 1` is a pass-through, **not** a Standard Gap / Optimization Difference verdict. `doc_match = 1` thus carries two meanings; filter on `code_match == 0` when counting round-2 non-Bug differences.
- Round 1 oversize skip → `(9, 9)`.
- Documentation block not matched → `(0, 8)`.
- Round 1 `NONEQUIVALENT` → `code_match = 0`, then round 2 sets `doc_match` to `0` / `1` / `9`.

## 5. Repair: autorepair

**autorepair** (`autorepair/`) is the repair component: it feeds the **error cause attributed by the LLM + the original code to fix + key documentation information** into the LLM, which calls DeepSeek v4 Flash to repair frontend converter defects. The tool stack is selected the same way as in the audit pipeline.

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
- `--api-root PATH`: the API source tree scanned by the existence check. Equivalent to the active backend's `root_env` variable, and required by one of the two whenever that backend declares namespaces (see 5.5 and 7.2).

### 5.4 Output

No repair cases are shipped. On its first run `repair.py` creates `repairs/{backend}_{frontend}_{op}/` next to itself and writes:

| File | Content |
|---|---|
| `prompt.txt` | The full prompt sent to the LLM (reproducible, no format requirements) |
| `response.md` | Raw LLM response |
| `fixed.py` | The complete code block fixed by the LLM |
| `fix.patch` | The unified diff generated by the program with difflib |
| `api_check.md` | API verification report (per-round existence results / misuse review / correction rounds) |

The script prints `applyable=`: `True` means the patch is anchored to the real source file with correct line numbers (`patch -p1` can apply it); `False` means no `--src-file` was given or the original code snippet was not matched in the source file (falls back to a snippet-level diff). A patch being applyable does not make it correct — verify the fix independently by differential or reproduction testing.

### 5.5 API verification (automatic; disabled with `--no-verify`)

Against the LLM's **fabrication** of APIs (using APIs that do not exist) and **misuse** (using existing APIs incorrectly), verification + correction runs automatically after the patch is generated.

The existence check is driven by a per-backend `API_VERIFY` profile that ships **empty** — which namespaces are worth indexing depends on the tree you audit — so the static half is off until you fill it in. See 7.5 for the fields; the alternative is to skip verification with `--no-verify`.

1. **Extraction**: extract `a.b.c(...)`-style dotted API calls from the **added/changed lines** of the fixed code (the `+` lines of the diff).
2. **Existence check (deterministic, no LLM)**: resolve each call's namespace (e.g., `pkg.op.add` → `pkg.op`), get the **full API list** of that namespace, and see whether the leaf is among them. Two sources are consulted:
   - Backends that declare namespaces in their `API_VERIFY` profile get a **static index**: for each declared package the verifier parses the `__init__.py` re-exports (`from .X import (a, b, ...)`) plus in-package `def` names, expanding star exports by scanning the imported files. The tree is read rather than imported, because these frameworks are normally not importable in the analysis environment, and it is located through the profile's `root_env` variable — nothing is guessed.
   - Modules that can be imported (numpy / functools / ...) are checked with the standard `inspect` mechanism.
   - Runtime objects such as `attr.` / `bb.` are out of scope (`attr` fabrication is caught by the heuristic check below).

   A missing leaf is only reported as a fabrication for namespaces the profile lists under `strict_namespaces`; anywhere else it degrades to `unverifiable`, so a namespace that was not enumerated completely cannot produce a false positive. A backend that declares no namespaces — a C++ frontend, for instance — gets no static check at all, and verification falls back to the runtime check and the LLM review.
3. **Misuse review (LLM)**: for each used API, extract its signature + docstring (AST, from the definition file), and have the LLM judge, together with the original code and the added lines, whether the parameter names/types/count/semantics match and whether the usage deviates from codebase conventions.
4. **Correction loop**: when there is a 'nonexistent API' or 'suspected misuse', design the detected information (including the namespace's API count, similar APIs, and the LLM's suggestion) into a prompt and feed it back to the LLM, asking it to re-output the complete code after modification, then verify again; iterate until it passes or reaches `--verify-rounds` (default 2).

Regression-verified against known weak points (from the authors' run, with the profile's namespaces populated):
- `relax.op.addd` (fabricated) → the static check returns ❌ nonexistent (similar: add); after correction, changed to `relax.op.add`;
- `relax.op.zeros(x_shape)` (missing the required `dtype`) → the review returns ⚠ misuse; after correction, `dtype='float32'` was added;
- `relax.op.zeros(x_shape, x.ty.dtype)` → the review returns ✓ correct (the `zeros` signature is `dtype: str | DataType`; passing a DataType object is legal) — resolving the earlier unverified question of "whether zeros accepts it";
- `attr.get_int_tuple` (fabricated) → caught by the `check_attr_accessors` heuristic (below).

**Verification boundary (no false positives)**: verification targets only "whether the API exists / is misused". It does **not** catch semantic regressions in shared code paths — for example a fix that hard-codes `cls.numpy_op` to `np.mean`, where `np.mean` itself is real and used correctly, but the base class it lives in is shared by other operators. Such issues belong to the prompt-contract layer, which must constrain "do not change the semantics of the shared base class for the other operators that use it".

#### Heuristic check (existing)

If the fixed code introduces `attr.<method>(` calls not used by the original code (e.g., `attr.get_int_tuple()` / `attr.get_int()`), a `[warn]` is printed. Such methods are often fabricated by the LLM (a frontend attr is usually a dict, only `attr.get(...)` exists); when hit, confirm manually, and use `attr.get(...)` if unsure.

## 6. Repository Layout

```
.
├── README.md                  # This document
├── main.py                    # Audit entry point; --backend selects the tool stack
├── settings.py                # Where the documentation dumps live
├── prompts.py                 # Prompt 1 / 2 / 3 templates and the [Placeholder] renderer
├── extract_function.py        # Step 1: extract converters        (delegates to frontends/)
├── code_expander.py           # Step 2: dependency code expansion (delegates to frontends/)
├── consistency_checker.py     # Step 3: consistency comparison (Prompt 1)
├── doc_retriever.py           # Step 4: documentation retrieval
├── doc_analyzer.py            # Step 5: documentation-based deep audit (Prompt 3)
├── llm_client.py              # One client per provider/model + global proxy
├── batch_run.py               # Batch runner (several models in parallel)
├── run_with_model.py          # Non-interactive run with a specified model
├── frontends/                 # Everything tool-stack specific
│   ├── __init__.py            #   registry: name -> module
│   ├── openvino.py            #   C++: .cpp scan, regex + brace matching, 5-rule Prompt 2
│   └── tvm.py                 #   Python: convert_map + AST, .py search, 6-rule Prompt 2
└── autorepair/                # Repair component
    ├── repair.py              # prompt → LLM → code → patch → API verification
    ├── api_check.py           # existence check + misuse review + correction loop
    ├── llm_client.py          # DeepseekV4FlashClient + exponential-backoff retry
    └── repairs/               # Written at runtime by repair.py (not tracked)
```

`frontends/` is the only place a tool stack is named. Adding a backend means adding one module there and registering it in `frontends/__init__.py`. `autorepair/` reads the same module for the backend's `API_VERIFY` profile, so it carries no tool-stack names of its own either — which is why it needs the repository root on its import path rather than being copyable on its own.

**Audit output.** No audit results are shipped. A run creates `results/<model>/<timestamp>/` next to `main.py`, holding `results_detail.txt` (full per-pair log) and `pairs_result.txt` (one line per operator pair with its `code_match` / `doc_match` flags — see Section 4). `batch_run.py` appends to `batch_run.log` in the same directory. Both are run artifacts and are not tracked.

## 7. Setup

### 7.1 Python dependencies

| Package | Needed by |
|---|---|
| `openai` | `llm_client.py`, `autorepair/llm_client.py` |
| `python-Levenshtein` | `main.py` (operator-name similarity) |

Everything else is standard library.

```bash
pip install openai python-Levenshtein
```

### 7.2 Environment

| Variable | Required | Meaning |
|---|---|---|
| `OPENVINO_SRC_ROOT` | for `--backend openvino` | Root of the OpenVINO checkout — the directory containing `src/frontends/` |
| `TVM_RELAX_FRONTEND` | for `--backend tvm` | The `python/tvm/relax/frontend` directory of the TVM checkout |
| `FRONTEND_AUDIT_DOC_ROOT` | always | Directory holding one `<frontend>doc.txt` documentation dump per frontend (see 7.4) |
| `AUDIT_BACKEND` | no | Default backend when `--backend` is not given |
| the backend's `root_env` (see 7.5) | only once that backend's profile declares namespaces | The API source tree scanned by the existence check; also settable with `--api-root`. The `tvm` profile reads `TVM_PYTHON_ROOT` (note: not `TVM_RELAX_FRONTEND`) and the `openvino` profile reuses `OPENVINO_SRC_ROOT`. While the shipped profiles are empty, nothing reads these |

### 7.3 API keys

API keys are **not** stored in this repository. Every client class in `llm_client.py` ships with a `[Your own API key]` placeholder; replace it with your own key before running.

`autorepair/llm_client.py` accepts `DEEPSEEK_API_KEY` from the environment and falls back to the same placeholder when it is unset.

### 7.4 Documentation dumps

Step 4 needs an official documentation dump per frontend. The dumps are not shipped here — they are large scraped snapshots of each framework's reference documentation — so point `FRONTEND_AUDIT_DOC_ROOT` at a directory laid out as:

```
onnxdoc.txt  torchdoc.txt  paddledoc.txt  jaxdoc.txt    # --backend openvino
onnxdoc.txt  torchdoc.txt                              # --backend tvm
```

Each file is plain text. Step 4 splits it on a line of exactly 50 `=` characters and
reads alternating blocks: a URL line, then that operator's documentation body. The op name
is taken from the URL's last path segment, so the URL must end in the operator name
(`.../Add.html`, or `.../add__Add.html` for the ONNX convention).

```
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
```

A dump is produced by concatenating the operator reference pages of one framework in that
form; the scraper itself is not part of this repository. Names are matched leniently
(lowercased, underscores and trailing digits stripped, plus a table of known cross-framework
variants such as `cum_sum`/`cumsum`), and the highest-scoring block wins. A frontend whose
dump is missing is not an error: Step 4 reports the pair as documentation-not-matched, which
is flag `8` in Section 4.

### 7.5 API verification profile

`autorepair/`'s existence check is configured per backend by the `API_VERIFY` dict in each
`frontends/` module. It ships with every list empty: the verifier is a generic driver, and
which API namespaces are worth indexing depends on the tree you audit, so populating the
profile is your job. Until you do, the static half is skipped and only the runtime check
and the LLM misuse review run — the `❌ nonexistent API` verdict cannot fire.

| Field | Meaning |
|---|---|
| `tool_stack` | Name used in the verification report and the review prompt. |
| `root_env` | Environment variable that points at the API source tree (e.g. `TVM_PYTHON_ROOT`). |
| `root_marker` | Path relative to that root whose presence confirms it is the right tree, e.g. `("pkg", "sub", "__init__.py")`. Set to `None` for a backend with no Python package tree to scan — the static check is then off by construction. |
| `namespaces` | `(namespace, directory relative to the root)` pairs to index. For each one the verifier collects the names that the package's `__init__.py` re-exports plus the `def`s in that package's own modules. Empty disables the static check. |
| `strict_namespaces` | Subset of `namespaces` whose enumeration is complete. A leaf missing from one of these is reported as a fabrication; a leaf missing anywhere else degrades to `unverifiable`, so a namespace that was not enumerated completely cannot produce a false positive. Empty means never report a fabrication. |
| `extra_call_roots` | Extra dotted-call roots to resolve, beyond the runtime modules the verifier already handles (`numpy`, `functools`, `math`, ...). |
| `self_test_cases` | `(api_name, expected_status)` pairs exercised by `python autorepair/api_check.py --self-test [backend]`; `expected_status` is `ok` / `missing` / `unverifiable`. |

A filled-in profile for a Python frontend whose packages live under `<root>/pkg/`:

```python
API_VERIFY = {
    "tool_stack": "MyFramework",
    "root_env": "MYFRAMEWORK_PYTHON_ROOT",
    "root_marker": ("pkg", "op", "__init__.py"),
    "namespaces": [("pkg.op", "pkg/op"), ("pkg", "pkg")],
    "strict_namespaces": {"pkg.op"},
    "extra_call_roots": {"pkg"},
    "self_test_cases": [("pkg.op.add", "ok"), ("pkg.op.nope", "missing")],
}
```

A call under one of the declared namespaces is resolved against the static index; anything
else falls through to the runtime path. Only list a namespace under `strict_namespaces` if
the index really does enumerate its API list completely — that field is the single switch
between "this leaf is fabricated" and "we cannot tell", and over-claiming it is what
produces false positives.

## 8. Running

```bash
export OPENVINO_SRC_ROOT=/path/to/openvino
export FRONTEND_AUDIT_DOC_ROOT=/path/to/doc-dumps

python main.py --backend openvino            # interactive model selection
python run_with_model.py 4 --backend tvm     # non-interactive, model 4
python batch_run.py --backend openvino       # 3 models x 5 runs, in parallel
```

`autorepair/` needs no configuration beyond the API source tree of the backend you pass to `--backend`.
