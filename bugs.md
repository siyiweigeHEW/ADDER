# Bugs uncovered by ADDER

> [!IMPORTANT]
>
> **Summary of Bugs**
>
> | Tool stack | Frontend | #Confirmed | #Fixed | #Fixing | #Waiting |
> |---|---|---|--:|--:|--:|--:|
> | TVM | ONNX | 13 | 12 | 1 | 0 |
> | TVM | PyTorch | 8 | 7 | 1 | 0 |
> | TVM | Paddle | 1 | 1 | 0 | 0 |
> | TVM | OneFlow | 2 | 2 | 0 | 0 |
> | TVM | Keras | 1 | 1 | 0 | 0 |
> | OpenVINO | ONNX | 9 | 3 | 5 | 0 |
> | OpenVINO | PyTorch | 4 | 0 | 4 | 0 |
> | OpenVINO | Paddle | 9 | 0 | 9 | 1 |

> [!NOTE]
>
> - **All entries were found and reported independently by us.** Bugs the upstream maintainers
>   already knew about (reported by others) are excluded, as are false positives and rejections
>   we did not file upstream.
> - **#Confirmed counts every report upstream acted on**, i.e. fixed + fix in progress + confirmed;
>   the `#Fixed` / `#Fixing` columns break it down. ✅ fixed upstream; 🚧 fix in progress;
>   🔵 confirmed by upstream; ⏳ waiting on upstream (unsupported).
> - **Symptom**: 💥 crash or conversion failure; 🧮 wrong result (silent semantic bug)
>
> **Table of Contents**
>
> * [**TVM**](#tvm)
> * [**OpenVINO**](#openvino)

## [TVM](https://github.com/apache/tvm)

### ONNX frontend

* ✅💥 [`ConvTranspose` — ConvTranspose with `auto_pad` (or `output_shape`) but without the optional `strides` attribute aborts conversion with a KeyError, because the converter reads ... · #15868 · apache/tvm](https://github.com/apache/tvm/pull/15868)
* ✅🧮 [`Flatten` — Flatten silently accepts an out-of-range axis in from_onnx, whereas onnxruntime rejects the same model · #20144 · apache/tvm](https://github.com/apache/tvm/issues/20144)
* ✅🧮 [`Gelu` — The GELU operator does not support the `approximate="tanh"` attribute (approximate computation) · #18750 · apache/tvm](https://github.com/apache/tvm/issues/18750)
* ✅🧮 [`Mean` — ONNX Mean is handled incorrectly when all inputs are constants (initializers): a single input returns a 0-D scalar (the global mean) and multiple inputs raise TypeError · #20146 · apache/tvm](https://github.com/apache/tvm/issues/20146)
* ✅💥 [`Pad` — The opset18+ features of ONNX Pad, namely mode="wrap" and the optional axes input, are rejected or ignored by from_onnx · #20150 · apache/tvm](https://github.com/apache/tvm/issues/20150)
* ✅💥 [`Prelu` — ONNX PRelu with a low-rank broadcastable slope shape (for example (64,1,1)) is rejected by from_onnx with "ValueError: Unsupported PRelu slope shape" · #20148 · apache/tvm](https://github.com/apache/tvm/issues/20148)
* ✅🧮 [`Reshape` — ONNX Reshape mishandles zeros in shape: the constant-folding path treats 0 as a literal zero dimension (it should copy the corresponding input dimension), and ... · #20151 · apache/tvm](https://github.com/apache/tvm/issues/20151)
* ✅🧮 [`Scatter` — ONNX Scatter (opset 9/10) silently produces wrong numerical output when indices and updates have broadcastable lower rank · #20182 · apache/tvm](https://github.com/apache/tvm/issues/20182)
* ✅🧮 [`Softplus` — ONNX Softplus is injected with a threshold that does not exist in the ONNX specification, so that for float64 the range 20 < x <= 30 is approximated linearly and ... · #20184 · apache/tvm](https://github.com/apache/tvm/issues/20184)
* 🚧💥 [`Split` — Split cannot handle uneven splits described by `num_outputs` · #18751 · apache/tvm](https://github.com/apache/tvm/issues/18751)
* ✅🧮 [`Squeeze` — ONNX Squeeze silently ignores axes that select non-unit dimensions (it should raise an error), violating the ONNX specification · #20185 · apache/tvm](https://github.com/apache/tvm/issues/20185)
* ✅💥 [`Tile` — The Tile operator incorrectly requires the repeats argument to be a constant tensor · #18752 · apache/tvm](https://github.com/apache/tvm/issues/18752)
* ✅💥 [`Where` — ONNX Where mishandles ShapeExpr (shape tensor) inputs: it rejects legal size-1 broadcasting and crashes on mixed tensor/ShapeExpr inputs · #20186 · apache/tvm](https://github.com/apache/tvm/issues/20186)

### PyTorch frontend

* ✅💥 [`Einsum` — torch.einsum with repeated subscripts (diagonals and traces, for example 'ii->i') contains aten.diagonal after decomposition and cannot be converted · #20228 · apache/tvm](https://github.com/apache/tvm/issues/20228)
* ✅💥 [`Flatten` — torch.flatten with illegal arguments (start_dim > end_dim) crashes from_fx with an internal "TypeError: reduce() of empty iterable" instead of reporting clear invalid ... · #20227 · apache/tvm](https://github.com/apache/tvm/issues/20227)
* ✅🧮 [`Flip` — The flip converter only reverses along the first element of the axes list (`axis[0]`), so torch.flip(x, [0, 1]) reverses only dim 0 and returns a wrong result whenever ... · #15752 · apache/tvm](https://github.com/apache/tvm/pull/15752)
* ✅🧮 [`Mean` — The dtype argument of torch.Tensor.mean(..., dtype=...) is silently ignored and the output keeps the input dtype (accumulation also follows the input type) · #20230 · apache/tvm](https://github.com/apache/tvm/issues/20230)
* ✅💥 [`One` — The _one_hot of F.one_hot lacks a num_classes > 0 check: non-positive values (0, -1, -2) are passed through and trigger a low-level C++ ICHECK, raising an opaque ... · #20319 · apache/tvm](https://github.com/apache/tvm/issues/20319)
* ✅🧮 [`Round` — torch.round rounds half values away from zero, whereas banker's rounding (ties-to-even) is required; round(x, decimals=...) also cannot be converted through ... · #20231 · apache/tvm](https://github.com/apache/tvm/issues/20231)
* ✅🧮 [`Split` — x.split(split_size) produces wrong output block shapes when split_size does not divide D and is greater than D/2 (the per-block size is mistaken for a segment count) · #20232 · apache/tvm](https://github.com/apache/tvm/issues/20232)
* 🚧🧮 [`Squeeze` — Missing dim validation in the torch frontend _squeeze: when dim is a list whose entries are all positively out of range, valid_dims is empty and the implementation falls ... · #20321 · apache/tvm](https://github.com/apache/tvm/issues/20321)

### Paddle frontend

* ✅🧮 [`Softplus` — The Softplus converter in the PaddlePaddle frontend ignores the `threshold` attribute: it always returns log(exp(x*beta)+1)/beta, while PaddlePaddle returns x unchanged ... · #14845 · apache/tvm](https://github.com/apache/tvm/pull/14845)

### OneFlow frontend

* ✅🧮 [`Softplus` — The Softplus converter in the OneFlow frontend ignores both `beta` and `threshold`: it returns log(exp(x)+1) with no beta scaling and no threshold branch, so the result ... · #15717 · apache/tvm](https://github.com/apache/tvm/pull/15717)
* ✅🧮 [`Threshold` — The `threshold` operator converter is a copy of ThresholdedReLU: it reads a non-existent `alpha` attribute and computes x*(x>alpha), instead of where(x>threshold_val, x ... · #15715 · apache/tvm](https://github.com/apache/tvm/pull/15715)

### Keras frontend

* ✅🧮 [`GRU, SimpleRNN` — The GRU and SimpleRNN converters never read the layer property `go_backwards`, so a layer created with go_backwards=True is converted as if it ran forward and the ... · #15829 · apache/tvm](https://github.com/apache/tvm/pull/15829)

## [OpenVINO](https://github.com/openvinotoolkit/openvino)

### ONNX frontend

* 🚧🧮 [`Cast` — The ONNX frontend casts a float tensor to int4/uint4 by rounding ties away from zero instead of ties-to-even, so exact .5 inputs are converted to the wrong integer (2.5 ... · #38158 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/38158)
* 🔵🧮 [`Div` — ONNX Div produces wrong results for int64 inputs: values outside the int32 range return 0, and values inside the int32 range are slightly inaccurate or even negative · #37766 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37766)
* ✅💥 [`Dropout` — ONNX Dropout with training_mode=true cannot be converted by the frontend (missing feature; onnxruntime executes the same model correctly) · #37767 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37767)
* 🚧💥 [`Grid` — 5-D (volumetric) ONNX GridSample (opset 16, 20 and 22) cannot be converted by the frontend (missing feature) · #37770 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37770)
* 🚧💥 [`Pad` — ONNX Pad with mode="wrap" (opset 19 and later) cannot be converted by the frontend (missing feature) · #37772 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37772)
* 🚧🧮 [`Reduce` — The ONNX frontend ignores the `axes` attribute of `ReduceL1` / `ReduceLogSum` / `ReduceLogSumExp`, so opset 13-17 models silently reduce over all axes · #38238 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/38238)
* ✅🧮 [`Rnn` — The layout=1 attribute (opset 14 and later) of ONNX RNN/LSTM/GRU is ignored, so sequences are silently processed under the wrong layout · #37820 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37820)
* 🚧🧮 [`Roi` — The default value of sampling_ratio in ONNX RoiAlign is taken as 1, whereas the specification default is 0 (adaptive sampling), so the result is wrong when the attribute ... · #37826 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37826)
* ✅🧮 [`Softmax` — ONNX Softmax for opset 11/12 does not reshape inputs of rank > 2 into 2-D first, so the result is wrong when axis is not the last dimension · #37823 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37823)

### PyTorch frontend

* 🚧🧮 [`Distance` — The PyTorch frontend computes `sum((x-y)^p)^(1/p)` for `aten::pairwise_distance` and `aten::cdist`, omitting the `Abs()`, so every `p != 2` is wrong or NaN · #38239 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/38239)
* 🚧🧮 [`Instance` — The PyTorch frontend ignores use_input_stats in instance_norm: even with use_input_stats=True, the running statistics are used whenever running_mean and running_var are ... · #37954 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37954)
* 🚧🧮 [`Round` — The PyTorch frontend ignores the decimals parameter of torch.round, so the output is silently rounded to an integer · #37953 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37953)
* 🚧💥 [`Unique` — torch.unique(x, dim=0) fails to convert with OpConversionFailure: No conversion rule found for aten::unique_dim (the frontend does not handle dim and fails loudly at ... · #38284 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/pull/38284)

### Paddle frontend

* 🚧🧮 [`Argmax` — Paddle argmax ignores the keepdims attribute and unconditionally squeezes the reduction axis when keepdim=True · #37827 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37827)
* 🚧🧮 [`Atan2` — Paddle atan2 returns +pi/2 when the first input (y) is negative and the second input (x) is 0, whereas the correct value is -pi/2 (wrong quadrant) · #37831 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37831)
* 🚧💥 [`Dropout` — The exported model fails to convert at all: with a float64 or float16 input and dropout_implementation="downscale_in_infer", convert_model raises OpConversionFailure ... · #38095 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/pull/38095)
* 🚧 [`Group` — The documentation declares six data_format values while the code handles only NCHW/NHWC and does not cover the remaining layouts · #38276 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/pull/38276)
* 🚧 [`Leakyrelu` — Handling of the alpha type in the fp16 case · #38269 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/pull/38269)
* 🚧🧮 [`Linspace` — paddle.linspace declared with dtype=float64 is silently converted to float32 output, losing float64 precision · #37949 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37949)
* ⏳💥 [`Rnn` — The Paddle frontend cannot convert GRU or SimpleRNN models: the rnn operator converter supports only mode==LSTM · #37950 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37950)
* 🚧🧮 [`Scatter` — With overwrite=False the frontend zeroes every row not covered by index, whereas Paddle zeroes only the indexed positions: for x=[[1,1],[2,2],[3,3]], index=[2,1] and ... · #38185 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/38185)
* 🚧💥 [`Softplus` — The Paddle frontend cannot convert softplus with beta != 1 or threshold != 20 and accepts only the default parameters · #37951 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37951)
* 🚧🧮 [`Sum` — The Paddle frontend ignores out_dtype and boolean inputs of the reduce operators: paddle.sum(dtype=...) returns the wrong output element type, and paddle.sum(bool) ... · #37952 · openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/issues/37952)
