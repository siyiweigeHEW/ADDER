    relax_op: Callable = None

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if cls.numpy_op is None or cls.relax_op is None:
            raise NotImplementedError("numpy_op and relax_op must be defined for MultiInputBase")
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            np_inputs = [inp.data.numpy() for inp in inputs]
            output = cls.numpy_op(*np_inputs)  # pylint: disable=not-callable
            return relax.const(output, output.dtype)

        input_shapes = [inp.ty.shape for inp in inputs]
        target_shape = functools.reduce(compute_broadcast_shape, input_shapes)

        # broadcast_to, stack them, then perform minimum over the new axis.
        inputs = [bb.normalize(relax.op.broadcast_to(i, target_shape)) for i in inputs]
        stacked_tensor = bb.normalize(relax.op.stack(inputs, axis=0))
        return cls.relax_op(stacked_tensor, axis=0)  # pylint: disable=not-callable
