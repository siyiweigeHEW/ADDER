# ONNX_PRelu Equivalence Comparison

- Reference PR: [#20149](https://github.com/apache/tvm/pull/20149) (+18 -10)
- Prompt of this round: [tool_prompt.txt](tool_prompt.txt)
- Tool patch: [tool_fix.patch](tool_fix.patch) (applyable=True, dry-run passed with `patch -p1` from `/`)
- Reference patch: [ref_PR20149.patch](ref_PR20149.patch)

## Round log
- **round 1**: invalid. The root cause was a **test-input construction error** (not a
  tool problem) — when extracting the code, the declaration line of the next class
  `class ThresholdedRelu(OnnxOpConverter):` was mixed into the input; the LLM
  reasonably treated it as part of the PRelu scope and dropped it, so the patch deleted
  that line (leaving a dangling docstring). After fixing the input (without that line),
  re-ran.
- **round 2**: scope alignment correct; this is the archived round.

## Verdict: the main legal shapes are equivalent, with two deviations ⚠

| | Tool patch | Reference #20149 |
|---|---|---|
| Single non-broadcast axis | `nn.prelu(x, slope, axis)` ✓ | same as left ✓ |
| Multiple non-broadcast axes / slope same as X | `where(less(x, zeros(x_shape)), multiply(slope, x), x)` | `where(less(x, const(0, dtype)), multiply(x, slope), x)` |
| Fallback position | **outside the `if s_ndim <= ndim:` block** | **inside the block** |
| s_ndim > ndim | goes to fallback (no longer raises) | keeps `raise ValueError` |
| 0 constant | `relax.op.zeros(x_shape, x.ty.dtype)` | `relax.const(0, dtype)`, `dtype = x.ty.dtype.dtype` |

**Deviation 1 (behavior)**: the tool moves the where/multiply fallback outside the
`if s_ndim <= ndim:` block -> `s_ndim > ndim` (slope rank greater than X; not a legal
unidirectional broadcast shape) no longer raises ValueError but instead attempts
`multiply(slope, x)`, which fails at runtime broadcast or produces a wrong shape. The
reference keeps the raise, which is safer.

**Deviation 2 (API validity)**: `relax.op.zeros` exists (onnx_frontend.py:4126), but
the codebase convention is `data.ty.dtype.dtype` (passing a string dtype); the tool
passes `x.ty.dtype` directly (a DataType object), and whether zeros accepts it is
unverified. The reference uses `relax.const(0, dtype)` without this risk.

**Equivalent parts**: the single-non-broadcast-axis path is verbatim identical; the
where-multiply semantics for multiple non-broadcast axes / slope same as X are correct
(equivalent to the reference on legal inputs).
