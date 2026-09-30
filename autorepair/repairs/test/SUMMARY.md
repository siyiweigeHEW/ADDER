# autorepair test summary: 4 operators vs upstream reference PRs

Test date: 2026-08-18. Protocol: **run the tool independently first (without looking
at the reference PRs), then fetch the PRs for comparison**.
Each operator directory records separately: `input_code/doc/cause` (inputs),
`tool_prompt.txt` (the prompt of this round), `tool_fixed.py` / `tool_fix.patch`
(tool outputs), `tool_response.md` (raw LLM response), `ref_PR*.patch` (reference PR),
`compare.md` (equivalence verdict).

| Operator | Reference PR | Tool patch applyable | Equivalence verdict | Notes |
|---|---|---|---|---|
| Flatten | #20145 | ✅ dry-run passed | **Equivalent ✓** | rank + normalization + ValueError; differs only in wording |
| Mean | #20147 | ✅ dry-run passed | **Not fully equivalent ⚠** | correct for Mean, but hard-codes `np.mean` -> breaks the shared path of Sum/Min/Max (multi-constant inputs would be computed as the mean) |
| PRelu | #20149 | ✅ applyable | **Partially equivalent ⚠** | round 1 invalid due to an input-extraction error (re-ran); fallback moved outside `s_ndim<=ndim` (s_ndim>ndim no longer raises); the dtype argument of `relax.op.zeros(x_shape, x.ty.dtype)` deviates from the codebase convention |
| Pad | #20152 | ✅ dry-run passed | **Functionally equivalent ✓ (reference more rigorous)** | tool modified v2/v11 to add wrap+axes; reference added _impl_v18; tool over-permissive and lacks length/negative-axis validation |

## Conclusions

1. **The tool can independently produce high-quality patches**: Flatten (fully
   equivalent) and Pad (functionally equivalent) produced fixes semantically consistent
   with the human PRs without any information from the reference PRs; all patches are
   applyable.
2. **Two real weak points** (not formatting issues, but semantic/API-consistency
   issues):
   - **Shared code path**: the Mean fix hard-codes `np.mean` instead of reusing
     `cls.numpy_op`, ignoring that MultiInputBase is shared by Sum/Min/Max — fixing a
     single operator introduces a regression in sibling operators. The prompt contract
     already has a "minimal surgical" constraint, but it does not constrain "not
     changing the behavior semantics of the fixed function / not assuming the function
     only serves the current operator".
   - **API convention deviation**: PRelu uses `relax.op.zeros(x_shape, x.ty.dtype)`,
     while the codebase convention is `data.ty.dtype.dtype` (onnx_frontend.py:4126);
     the `s_ndim > ndim` guard was removed.
3. Structural differences (Pad modifying old methods vs adding new methods) are a
   reasonable design divergence; functionally equivalent.

## Suggested tool improvements (if needed)

- Add a clause to the prompt contract: **the fix must not change the semantics of the
  callers/shared base class for other operators**; if the fixed code belongs to a shared
  code path, its generalized interface must be preserved (e.g., `cls.numpy_op`); the
  behavior must not be hard-coded to the current operator.
- On the code side, add a heuristic check for "newly introduced relax op calls" (similar
  to the existing attr check), warning on API usage that deviates from the codebase
  convention (e.g., passing `x.ty.dtype` rather than `x.ty.dtype.dtype` to an op).
