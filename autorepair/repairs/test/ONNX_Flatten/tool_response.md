根因：`Flatten._impl_v13` 未校验 `axis` 范围、未归一化负 `axis`，导致越界或负越界时 Python 切片静默钳制产生错误 shape。修复为：先获取输入秩 `r`，校验 `axis ∈ [-r, r]`，将负 `axis` 归一化为 `axis + r`，再按规范切片。

```python
    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 1)
        data_shape = list(inputs[0].ty.shape)
        r = len(data_shape)

        if axis < -r or axis > r:
            raise ValueError(
                f"Flatten axis {axis} out of range [-{r}, {r}] for input rank {r}"
            )
        if axis < 0:
            axis = axis + r

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