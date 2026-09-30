# `tvm/` — audit pipeline for TVM Relax

One of the two audit pipelines in this repository. The walkthrough from a bare checkout to results is in the
[root README](../README.md) and the prompts in [../PROMPTS.md](../PROMPTS.md); this file is
the reference for what this pipeline does differently.

- Language: Python
- Frontends covered: **onnx**, **torch**

## What it reads

A TVM checkout, at the path held by `[your tvm source root]` in `main.py` and
`extract_function.py` — see [3](../README.md#3-point-the-code-at-your-checkouts). It
points at `python/tvm/relax/frontend`.

## Step 1 — extracting converters

A TVM frontend declares its converters in an explicit mapping, so extraction reads that
mapping instead of scanning files:

- `convert_map` is pulled out of the frontend source with a regex. Three shapes are
  handled: a `convert_map = { … }` assignment, the `return { … }` form the torch frontend
  uses, and a `get_convert_map()` definition. The literal is normalised first, because
  keys and values are usually bare names (`Add`, `nn.Linear`) and torch values are
  converter factories of the form `X.get_converter` — both have to be quoted before the
  mapping can be evaluated.
- The names the mapping refers to are then located in the module with `ast`, and the
  `FunctionDef` / `ClassDef` body is taken as that operator's converter. ONNX frontends
  use classes, torch frontends use functions.

The torch frontend is read as `base_fx_graph_translator.py` followed by
`fx_translator.py`: its converters inherit from the first, and its mapping lives in the
second, so both are needed and the order is what makes the regex pick the right literal.

Converters are then paired across frontends by name similarity, and the pair goes through
the two judgments the pipeline names Step 3 and Step 5.

## Step 2 — code expansion

Dependency definitions are searched in the operator directory, its parent (the frontend
root, where `common.py` and shared helpers live), and the sibling `utils` / `core` /
`common` directories. A name counts as found when an AST match lands on a `ClassDef`,
`FunctionDef` or module-level `Assign` — a definition, not a call site — and only that
source segment is appended, so the expanded prompt stays within the token budget.

## Modules

| File | Role |
|---|---|
| `main.py` | the pipeline: Steps 0–5 |
| `extract_function.py` | Step 1 — `convert_map` + AST extraction |
| `code_expander.py` | Step 2 — dependency expansion (Prompt 2) |
| `consistency_checker.py` | Step 3 — consistency comparison (Prompt 1) |
| `doc_retriever.py` | Step 4 — documentation lookup |
| `doc_analyzer.py` | Step 5 — documentation audit (Prompt 3) |
| `llm_client.py` | one client per provider, behind a shared proxy |
| `batch_run.py` | batch runner |
| `run_with_model.py` | non-interactive single run |

## Running

    python tvm/main.py                      # interactive model selection
    python tvm/run_with_model.py 4          # non-interactive, model 4
    python tvm/batch_run.py                 # 3 models x 5 runs, in parallel

Model numbers are in [4](../README.md#4-run-an-audit); the results layout and how to
count bug candidates are in [5](../README.md#5-read-the-results). This run also needs
`tvm/docxes/` — see [2](../README.md#2-supply-the-inputs).
