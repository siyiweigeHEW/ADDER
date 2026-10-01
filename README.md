# ADDER — auditing frontend converters for consistency

ADDER is our method for auditing the **frontend converters** of AI inference tool stacks
for consistency. This repository ships the tool and the experimental setup behind the
paper; the method itself is described there.

Two audit pipelines and one repair component are shipped:

| | Tool stack | Language | Frontends |
|---|---|---|---|
| `tvm/` | TVM Relax | Python | onnx / torch |
| `openvino/` | OpenVINO | C++ | onnx / torch / paddle |
| `autorepair/` | repair and API verification, for both tool stacks | | |

## Contents

| Document | Covers |
|---|---|
| **this file** | building the environment, running an audit, reading the results |
| [`PROMPTS.md`](PROMPTS.md) | every prompt the tool sends, verbatim |
| [`tvm/README.md`](tvm/README.md) | the TVM pipeline in detail |
| [`openvino/README.md`](openvino/README.md) | the OpenVINO pipeline in detail |
| [`autorepair/README.md`](autorepair/README.md) | the repair component and its API verification |
| [`bugs.md`](bugs.md) | the bugs the audit found and reported upstream, with links |

---

## Reproducibility

### 1. Build the environment

Python ≥ 3.9 (`autorepair/tvm/api_index.py` uses `ast.unparse`).

    pip install -r requirements.txt

The pins matter: the verdicts come from an LLM, so a different client version is a
different experiment.

### 2. Supply the inputs

Three things are needed before a run, and none of them are in this repository.

**(a) A source checkout of the tool stack.** The pipeline reads the tree directly and never
imports the framework.

- **OpenVINO** — the `openvino` repository. The pipeline reads
  `src/frontends/{onnx,pytorch,paddle}/…/op`.
- **TVM** — the `apache/tvm` repository. The pipeline reads `python/tvm/relax/frontend`,
  which holds the ONNX and PyTorch frontends.

**(b) Documentation dumps**, one per frontend, built **at the version the compiler
implements**. The audit judges each side against its own framework's specification, so a
dump from a newer version than the converters implement turns intended behaviour into a
false Bug, and an older one hides real gaps. The two pipelines need not agree: their ONNX
frontends reach opset 23 (TVM) and opset 19 (OpenVINO) — read your own off the highest
`_impl_vN` method in the frontend, or the `OPSET_RANGE(lo, hi)` a converter registers with.
Torch and Paddle have no opset, so take the framework release the frontend was written
against.

`get_doc_dumps.py` builds them, given the reference index page to walk (a placeholder in
that script). Its dependencies are separate, since only this one-off script needs them:

    pip install -r requirements-doc-dumps.txt
    python get_doc_dumps.py --out tvm/docxes onnx torch
    python get_doc_dumps.py --out openvino/docxes onnx torch paddle jax

**(c) An API key.** Each client class in `llm_client.py` ships with the placeholder
`[Your own API key]`; replace it with your key for that provider. `autorepair/` also accepts
`DEEPSEEK_API_KEY` from the environment.

### 3. Point the code at your checkouts

The pipelines are the authors' working copies with machine-specific paths replaced by
placeholders. Nothing runs until they are filled in:

| Placeholder | In | Replace with |
|---|---|---|
| `[your tvm source root]` | `tvm/main.py`, `tvm/extract_function.py`, `autorepair/tvm/api_index.py` | the `python/tvm` directory of your TVM checkout |
| `[your openvino source root]` | `openvino/main.py` | the root of your OpenVINO checkout, i.e. the directory holding `src/frontends/` |
| `[this directory]` | `tvm/main.py`, `openvino/main.py`, `tvm/batch_run.py`, `openvino/batch_run.py` | the absolute path of the directory that file lives in; it is how `docxes/` and `results/` are found |
| `[Your own API key]` | `tvm/llm_client.py`, `openvino/llm_client.py`, `autorepair/llm_client.py` | your key for that provider |

### 4. Run an audit

The paper's results were produced with DeepSeek v4 Flash, model `4`. The other numbers —
`1` DeepSeek Chat, `2` Qwen3.7-Max, `3` Qwen3.5-Flash, `5` GPT-5.4-mini — are only there to
offer alternatives.

    python tvm/main.py                     # interactive, choose from a menu
    python tvm/run_with_model.py 4         # non-interactive
    python openvino/main.py
    python openvino/run_with_model.py 4

`batch_run.py` runs several models in parallel, streaming to `batch_run.log` beside it. A
run is LLM-bound: every pair costs one Prompt 1 call, and each pair judged non-equivalent
costs a Prompt 2 call on each side plus a Prompt 3 call.

### 5. Read the results

Each run writes, next to the pipeline:

    <tvm|openvino>/results/<model_name>/<timestamp>/
        pairs_result.txt      one line per operator pair:  <pair> <code_match> <doc_match>
        results_detail.txt    the full log: Prompt 1's input and answer, the documentation
                              retrieved, and Prompt 3's output

`pairs_result.txt` is what the paper's numbers are counted from; the two flags are defined
in the next section. The bug candidates of a run are the lines with `doc_match == 0`:

    awk '$3 == 0 {print $1}' tvm/results/deepseek-v4-flash/<timestamp>/pairs_result.txt

### 6. Run the repair

[`autorepair/README.md`](autorepair/README.md) lists its inputs; in short:

    cd autorepair

    # build the prompt and print it, without calling the model
    python repair.py --dry-run

    # non-interactive
    python repair.py --backend TVM --frontend onnx --op Flatten \
                     --code-file code.py --doc-file doc.txt --cause-file cause.txt \
                     --src-file path/to/onnx_frontend.py

Each case goes to `repairs/{backend}_{frontend}_{op}/` — `prompt.txt`, and from a real run
`response.md`, `fixed.py`, `fix.patch` and, unless `--no-verify` was given, `api_check.md`.
Verification is on by default and needs a source tree to scan:

    export TVM_PYTHON_ROOT=/path/to/tvm/python/tvm       # for --backend TVM
    export OPENVINO_SRC_ROOT=/path/to/openvino           # for --backend OPENVINO

or pass `--api-root`. It can be exercised without any model call:

    python api_check.py --self-test tvm
    python api_check.py --self-test openvino

---

## Verdict flags

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

## Code structure

    README.md                       this file
    PROMPTS.md                      every prompt, verbatim
    bugs.md                         the bugs reported upstream, with links
    get_doc_dumps.py                builds the documentation dumps (2b)
    requirements.txt                dependencies of the pipelines
    requirements-doc-dumps.txt      dependencies of the scraper only

    tvm/                            audit pipeline, TVM (Python frontends)
      README.md                     this pipeline in more detail
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
      README.md                     this pipeline in more detail
      ...                           same module layout as tvm/

    autorepair/                     repair component
      README.md                     inputs, usage, output, verification
      repair.py                     prompt → model → code → patch → verification
      api_check.py                  verification driver
      tvm/api_index.py              index and call extraction for TVM
      openvino/api_index.py         index and call extraction for OpenVINO
      llm_client.py                 one client, with backoff

Not shipped, because you supply them or the runs produce them:

    <tvm|openvino>/docxes/          documentation dumps read by Step 4      (2b)
    <tvm|openvino>/results/         audit results, one directory per run    (5)
    <tvm|openvino>/batch_run.log    batch runner log                        (4)
    autorepair/repairs/             repair cases                            (6)
