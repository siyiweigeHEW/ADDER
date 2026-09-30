class PRelu(OnnxOpConverter):
    """Converts an onnx PRelu node into an equivalent Relax expression.

    f(x) = slope * x for x < 0, x for x >= 0
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        x = inputs[0]
        slope = inputs[1]

        x_shape = x.ty.shape
        slope_shape = slope.ty.shape

        ndim = len(x_shape)
        s_ndim = len(slope_shape)

        if all(ss == 1 for ss in slope_shape):
            slope = relax.op.reshape(slope, (1,))
            return relax.op.nn.prelu(x, slope, ndim - 1)

        if s_ndim == 1:
            slope = relax.op.reshape(slope, (slope_shape[0],))
            return relax.op.nn.prelu(x, slope, ndim - 1)

        if s_ndim <= ndim:
            non_one_axes = [i for i, ss in enumerate(slope_shape) if ss != 1]

            # Must have only ONE non-broadcast axis
            if len(non_one_axes) == 1:
                relative_axis = non_one_axes[0]
                axis = ndim - s_ndim + relative_axis

                slope = relax.op.reshape(slope, (slope_shape[relative_axis],))
                return relax.op.nn.prelu(x, slope, axis)

        # Fallback: use generic broadcast via where/less/multiply
        # This handles: multiple non-broadcast dims, slope same shape as X,
        # lower-rank slope, and scalar (rank-0) slope.
        zero = relax.op.zeros(x_shape, x.ty.dtype)
        mask = relax.op.less(x, zero)
        neg_part = relax.op.multiply(slope, x)
        return relax.op.where(mask, neg_part, x)
