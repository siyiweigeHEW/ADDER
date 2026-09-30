# ONNX_Flatten Equivalence Comparison

- Reference PR: [#20145](https://github.com/apache/tvm/pull/20145) (+13 -0)
- Prompt of this round: [tool_prompt.txt](tool_prompt.txt); inputs in input_code.py / input_doc.txt / input_cause.txt
- Tool patch: [tool_fix.patch](tool_fix.patch) (applyable=True, dry-run passed)
- Reference patch: [ref_PR20145.patch](ref_PR20145.patch)

## Verdict: Equivalent ✓

| | Tool patch | Reference #20145 |
|---|---|---|
| Rank | `r = len(data_shape)` | `rank = len(data_shape)` |
| Out-of-range check | `if axis < -r or axis > r: raise ValueError` (before normalization) | `if axis < 0: axis += rank`, then `if not 0 <= axis <= rank: raise ValueError` |
| Negative axis | `if axis < 0: axis += r` | same as left |
| Error message | `Flatten axis {axis} out of range [-{r},{r}]` | `Flatten axis {attr.get('axis',1)} is out of range [-{rank},{rank}]` |

**Equivalence notes**:
- Both accept exactly the same set of axes (`[-r, r]` endpoints → normalized `[0, r]`
  endpoints); out-of-range axes raise `ValueError`.
- The order of checks differs (the tool checks `-r..r` before normalizing; the
  reference normalizes first, then checks `0..r`), which is mathematically equivalent.
- Only the wording of the error messages differs (cosmetic).
- The already-supported shapes (legal axes, including negative axes) are unchanged.
