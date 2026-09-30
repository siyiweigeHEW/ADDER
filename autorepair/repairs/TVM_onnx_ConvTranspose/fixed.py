    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if hasattr(inputs[0].ty, "ndim"):
            ndim = inputs[0].ty.ndim
        else:
            ndim = len(inputs[0].ty.shape)

        if ndim == 3:
            op = relax.op.nn.conv1d_transpose
            data_layout = "NCW"
            kernel_layout = "IOW"
        elif ndim == 4:
            op = relax.op.nn.conv2d_transpose
            data_layout = "NCHW"
            kernel_layout = "IOHW"
        elif ndim == 5:
            # 3D 转置卷积：relax.op.nn 无高层算子，走底层 topi 实现
            strides = attr.get_int_tuple("strides")
            padding = attr.get_int_tuple("pads")
            output_padding = attr.get_int_tuple("output_padding")
            dilation = attr.get_int_tuple("dilations")
            groups = attr.get_int("group", 1)
            data = inputs[0]
            weight = inputs[1]
            out = bb.emit_te(
                topi.nn.conv3d_transpose.group_conv3d_transpose_ncdhw,
                data,
                weight,
                strides,
                padding,
                output_padding,
                dilation,
                groups,
            )
            return out
        else:
            raise NotImplementedError("Ndim > 5 not supported for convolution.")
