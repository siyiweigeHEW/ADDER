根因：`np.mean` 等归约函数将第二个位置参数当作 `axis`，导致单输入被错误归约、多输入报错；修复为单输入直接返回，多输入先广播再堆叠后沿新轴求均值，与非常量路径行为一致。

```python
    relax_op: Callable = None

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if cls.numpy_op is None or cls.relax_op is None:
            raise NotImplementedError("numpy_op and relax_op must be defined for MultiInputBase")
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            np_inputs = [inp.data.numpy() for inp in inputs]
            if len(np_inputs) == 1:
                # 单输入：均值即输入本身，原样返回
                return relax.const(np_inputs[0], np_inputs[0].dtype)
            # 多输入：先广播到共同形状，再堆叠后沿新轴求均值
            target_shape = functools.reduce(compute_broadcast_shape, [inp.shape for inp in np_inputs])
            broadcasted = [np.broadcast_to(inp, target_shape) for inp in np_inputs]
            stacked = np.stack(broadcasted, axis=0)
            output = np.mean(stacked, axis=0)
            return relax.const(output, output.dtype)

        input_shapes = [inp.ty.shape for inp in inputs]
        target_shape = functools.reduce(compute_broadcast_shape, input_shapes)

        # broadcast_to, stack them, then perform minimum over the new axis.
        inputs = [bb.normalize(relax.op.broadcast_to(i, target_shape)) for i in inputs]
        stacked_tensor = bb.normalize(relax.op.stack(inputs, axis=0))
        return cls.relax_op(stacked_tensor, axis=0)  # pylint: disable=not-callable
```