根因：`Flatten._impl_v13` 未校验 `axis` 范围、未归一化负 `axis`，导致越界或负越界时 Python 切片静默钳制产生错误 shape。修复方案：在切片前校验 `axis ∈ [-r, r]` 并归一化为 `[0, r]`，确保合法负 axis 行为不变，越界时抛异常。

```python
    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 1)
        data_shape = list(inputs[0].ty.shape)
        rank = len(data_shape)

        # Validate and normalize axis per ONNX spec: axis ∈ [-r, r]
        if axis < -rank or axis > rank:
            raise ValueError(
                f"Flatten axis {axis} out of range [-{rank}, {rank}] for rank {rank} input"
            )
        if axis < 0:
            axis = rank + axis

        if axis == 0:
            new_shape = (1, -1)
        else:
            shape_flags = [isinstance(x, tvm.script.tirx.IntImm) for x in data_shape[0:axis]]

            if all(shape_flags):
                data_shape = [x.value for x in data_shape[0:axis]]
                new_shape = (_np.prod(data_shape).astype("int64"), -1)
            else:
                batch_size = 1

                for el in data_shape[0:axis]:
                    batch_size = batch_size * el
```