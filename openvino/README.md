# `openvino/` — audit pipeline for OpenVINO

ADDER's OpenVINO pipeline, one of the two audit pipelines in this repository. The walkthrough from a bare checkout to results is in the
[root README](../README.md) and the prompts in [../PROMPTS.md](../PROMPTS.md); this file is
the reference for what this pipeline does differently.

- Language: C++
- Frontends covered: **onnx**, **torch**, **paddle**

## What it reads

An OpenVINO checkout, at the path held by `[your openvino source root]` in `main.py` —
see [3](../README.md#3-point-the-code-at-your-checkouts). It reads
`src/frontends/{onnx,pytorch,paddle}/…/op`.

## Step 1 — extracting converters

OpenVINO implementations are free functions and classes spread over `.cpp` files, and the
operator-to-implementation mapping is implicit in the file names, so extraction walks each
frontend's operator directory and takes the contents of every `.cpp` it finds, keyed by
the file name. Files under the same operator directory that are not converters — shared
headers and the like — are not picked up, since only `.cpp` is read.

Converters are then paired across frontends by name similarity, and the pair goes through
the two judgments the pipeline names Step 3 and Step 5.

## Step 2 — code expansion

This matters more here than on the Python side: a great many OpenVINO converters are thin
wrappers, so the body alone frequently says nothing about what the operator does. Search
scope is therefore wider than the operator directory:

- the operator directory itself (`src/op`);
- the frontend's `src` root, plus its `utils` / `core` / `common` subdirectories, where
  helpers such as `get_inputs_with_promoted_types` and `normalize_axis` live;
- `src/frontends/common_translators`, the cross-frontend shared directory (e.g.
  `common_translators::translate_atan2_util`).

A dependency counts as found when brace matching lands on a definition — a call site is
explicitly rejected by looking at the token before the name — and only the balanced block
is appended, so a single header's worth of text does not enter the prompt.

## Step 4 — documentation lookup

Only the ONNX frontend has versioned semantics, so only it is looked up by version. The
dump keeps one block per opset, with the anchor naming it kept in the block's URL. The
opset to judge against is the one the frontend supports — the newest opset named by any of
its `ONNX_OP` registrations: `OPSET_RANGE(lo, hi)` up to `hi`, `OPSET_IN(n)`, or
`OPSET_SINCE(n)` from `n` onward — and Step 4 returns, per operator, the newest definition
not above it.

The bound is the frontend's, not the individual converter's: a converter never updated for
a later opset is exactly the gap being looked for. The PyTorch and Paddle converters
register no opsets, so there is no version to select on: the newest few definitions are
returned together instead, which keeps an operator documented rather than judged against
whichever block the page listed first.

## Modules

| File | Role |
|---|---|
| `main.py` | the pipeline: Steps 0–5 |
| `extract_function.py` | Step 1 — directory scan for `.cpp` converters |
| `code_expander.py` | Step 2 — dependency expansion (Prompt 2) |
| `consistency_checker.py` | Step 3 — consistency comparison (Prompt 1) |
| `doc_retriever.py` | Step 4 — documentation lookup |
| `doc_analyzer.py` | Step 5 — documentation audit (Prompt 3) |
| `llm_client.py` | one client per provider, behind a shared proxy |
| `batch_run.py` | batch runner |
| `run_with_model.py` | non-interactive single run |

## Running

    python openvino/main.py                 # interactive model selection
    python openvino/run_with_model.py 4     # non-interactive, model 4
    python openvino/batch_run.py            # 3 models x 5 runs, in parallel

Model numbers are in [4](../README.md#4-run-an-audit); the results layout and how to
count bug candidates are in [5](../README.md#5-read-the-results). This run also needs
`openvino/docxes/` — see [2](../README.md#2-supply-the-inputs).
