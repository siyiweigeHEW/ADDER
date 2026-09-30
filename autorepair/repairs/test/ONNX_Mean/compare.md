# ONNX_Mean Equivalence Comparison

- Reference PR: [#20147](https://github.com/apache/tvm/pull/20147) (+14 -7)
- Prompt of this round: [tool_prompt.txt](tool_prompt.txt)
- Tool patch: [tool_fix.patch](tool_fix.patch) (applyable=True, dry-run passed)
- Reference patch: [ref_PR20147.patch](ref_PR20147.patch)

## Verdict: correct for Mean itself, but introduces a sibling-operator regression — not fully equivalent ⚠

| | Tool patch | Reference #20147 |
|---|---|---|
| Single input | `return relax.const(np_inputs[0], dtype)` returned unchanged ✓ | `np.broadcast_arrays(x) -> stack(axis=0) -> reduce(axis=0)` = x ✓ |
| Multiple inputs | `compute_broadcast_shape` -> `np.broadcast_to` -> `np.stack` -> `np.mean(stacked, axis=0)` | `np.broadcast_arrays` -> `np.stack` -> `cls.numpy_op(stacked, axis=0)` |
| Reduction function | **hard-coded `np.mean`** | **`cls.numpy_op`** |

**Key difference (a real defect)**: `MultiInputBase` is the shared base class of
Mean/Sum/Min/Max (`numpy_op` = np.mean/np.sum/np.min/np.max).
- The tool hard-codes the multi-input reduction to `np.mean` -> after the fix, the
  multi-constant-input path of Sum/Min/Max would compute the mean (a newly introduced
  regression).
- The reference uses `cls.numpy_op(stacked, axis=0)` -> all 4 operators are correct.
- The single-input branch is equivalent in both (returning the input unchanged is
  correct for all 4 operators).

**Suggestion**: if the tool reverted the constant-path `np.mean` to `cls.numpy_op`, it
would be fully equivalent to the reference (the root-cause analysis already noted that
Sum/Min/Max share this path).
