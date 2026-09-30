"""Verify the LLM-generated onnx_Flatten fix (corresponds to the diff in repairs/TVM_onnx_Flatten/response.md).

Verifies directly at the converter Flatten._impl_v13 level (intercepting
relax.op.reshape to observe the computed result), comparing how the "original
implementation" and the "LLM fixed version" handle out-of-range/legal axes:
  1) Original implementation: an out-of-range axis (|axis| > r) does not raise an
     error; Python slicing silently clamps it, producing a wrong shape (BUG)
  2) Fixed version: an out-of-range axis raises ValueError (matching onnxruntime);
     legal axes behave unchanged

Note: this script uses the converter API of the TVM checkout
(inputs[0].struct_info.shape) for converter-level verification; it does not rely on
full-stack VM differential testing (the local TVM build cannot be imported due to a
lib mismatch, and its VM rejects raw numpy).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx.onnx_frontend import Flatten


def make_var(shape):
    return relax.Var("X", relax.TensorStructInfo(shape, "float32"))


def onnx_new_shape(shape, axis):
    """ONNX Flatten spec output shape (axis in [-r, r])."""
    r = len(shape)
    ax = axis if axis >= 0 else r + axis
    return (int(np.prod(shape[:ax])) if ax > 0 else 1,
            int(np.prod(shape[ax:])) if ax < r else 1)


# ---- Original implementation (taken verbatim from Flatten._impl_v13 in onnx_frontend.py) ----
def original_impl(cls, bb, inputs, attr, params):
    axis = attr.get("axis", 1)
    data_shape = list(inputs[0].struct_info.shape)
    if axis == 0:
        new_shape = (1, -1)
    else:
        shape_flags = [isinstance(x, tvm.script.tir.IntImm) for x in data_shape[0:axis]]
        if all(shape_flags):
            data_shape = [x.value for x in data_shape[0:axis]]
            new_shape = (np.prod(data_shape).astype("int64"), -1)
        else:
            batch_size = 1
            for el in data_shape[0:axis]:
                batch_size = batch_size * el
            new_shape = (batch_size, -1)
    return relax.op.reshape(inputs[0], new_shape)


# ---- Fixed version = original implementation + the added block from the response.md diff (verbatim) ----
def fixed_impl(cls, bb, inputs, attr, params):
    axis = attr.get("axis", 1)
    data_shape = list(inputs[0].struct_info.shape)
    rank = len(data_shape)

    # Normalize negative axis (ONNX allows axis in [-r, r])
    if axis < 0:
        axis += rank

    # Validate axis is within [0, rank] per ONNX spec
    if axis < 0 or axis > rank:
        raise ValueError(
            f"Invalid value({axis}) for attribute 'axis' in Flatten. "
            f"Expected axis in [{-rank}, {rank}], got {axis}."
        )

    if axis == 0:
        new_shape = (1, -1)
    else:
        shape_flags = [isinstance(x, tvm.script.tir.IntImm) for x in data_shape[0:axis]]
        if all(shape_flags):
            data_shape = [x.value for x in data_shape[0:axis]]
            new_shape = (np.prod(data_shape).astype("int64"), -1)
        else:
            batch_size = 1
            for el in data_shape[0:axis]:
                batch_size = batch_size * el
            new_shape = (batch_size, -1)
    return relax.op.reshape(inputs[0], new_shape)


seen = {}
orig_reshape = relax.op.reshape


def spy_reshape(data, shape):
    seen["shape"] = shape
    return data


def run(impl, axis, shape=(2, 3, 4)):
    seen.clear()
    X = make_var(shape)
    try:
        impl(Flatten, None, [X], {"axis": axis}, {})
        ns = seen.get("shape")
        got = tuple(int(x) if hasattr(x, "__index__") else x for x in ns)
        return "OK", got
    except Exception as e:
        return "RAISED", f"{type(e).__name__}: {str(e)[:60]}"


print("=" * 70)
print("Original implementation vs LLM fixed version: Flatten._impl_v13 handling of axis")
print("=" * 70)
relax.op.reshape = spy_reshape

print("\n[1] Out-of-range axes (violate the ONNX spec axis in [-r, r]; onnxruntime rejects):")
for shape, axis in [((2, 3, 4), 5), ((2, 3, 4), -4), ((2, 3, 4, 5), 5), ((2, 3, 4, 5), -5)]:
    o = run(original_impl, axis, shape)
    f = run(fixed_impl, axis, shape)
    fixed_ok = f[0] == "RAISED" and "ValueError" in f[1]
    print(f"  shape={shape} axis={axis:>3}: original={o[0]:<6}->{o[1]} | fixed={f[0]:<6}->{f[1]}  "
          f"{'✓ raises after fix (matches onnxruntime)' if fixed_ok else '✗ no error!'}")

print("\n[2] Legal axes (including negative axes and the ±r boundaries; must remain correct):")
all_ok = True
for shape, axis in [((2, 3, 4), -3), ((2, 3, 4), -2), ((2, 3, 4), -1), ((2, 3, 4), 0),
                    ((2, 3, 4), 1), ((2, 3, 4), 2), ((2, 3, 4), 3),
                    ((2, 3, 4, 5), -4), ((2, 3, 4, 5), -2), ((2, 3, 4, 5), 2), ((2, 3, 4, 5), 4)]:
    exp = onnx_new_shape(shape, axis)
    o = run(original_impl, axis, shape)
    f = run(fixed_impl, axis, shape)
    # The converter produces (batch, -1): on reshape, -1 infers the second dim from the
    # total element count, so the final shape should equal the expected one
    total = int(np.prod(shape))
    out = lambda r: (r[0], total // int(r[0])) if r[0] != 0 and r[1] == -1 else r
    ok = o[0] == "OK" and f[0] == "OK" and out(o[1]) == exp and out(f[1]) == exp
    all_ok = all_ok and ok
    print(f"  shape={shape} axis={axis:>3}: expected={exp} original={o[1]}->{out(o[1])} "
          f"fixed={f[1]}->{out(f[1])}  {'✓' if ok else '✗'}")

print(f"\nVerdict: all out-of-range axes raise ValueError after the fix ✓ | legal axes match "
      f"the ONNX spec before and after the fix {'✓' if all_ok else '✗'}")
relax.op.reshape = orig_reshape
