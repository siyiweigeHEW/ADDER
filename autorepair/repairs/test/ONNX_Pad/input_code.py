    """Converts an onnx Pad node into an equivalent Relax expression."""

    @classmethod
    def _impl_v2(cls, bb, inputs, attr, params):
        pads = attr.get("pads")
        pads = relax.const(_np.array(pads), inputs[0].ty.shape[0].ty)
        constant_value = attr.get("value")
        if constant_value is None:
            constant_value = 0.0

        if isinstance(pads, relax.Constant):
            pad_before, pad_after = _np.split(pads.data.numpy(), 2)
            pad_before = _np.ndarray.tolist(pad_before)
            pad_after = _np.ndarray.tolist(pad_after)
        else:
            raise ValueError("Dynamic pads are not supported yet.")

        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if pad_mode not in ["constant", "edge", "reflect"]:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        if pad_mode == "constant":
            return bb.emit_te(topi.nn.pad, inputs[0], pad_before, pad_after, constant_value)
        elif pad_mode == "reflect":
            return bb.emit_te(topi.nn.mirror_pad, inputs[0], pad_before, pad_after, "REFLECT")
        else:
            # edge mode - replicate border values
            return bb.emit_te(topi.nn.replicate_pad, inputs[0], pad_before, pad_after)

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        pads = get_constant(inputs[1], params)
        constant_value = get_constant(inputs[2], params)
        if constant_value is not None:
            constant_value = constant_value.data.numpy().item()
        else:
            constant_value = 0.0

        if isinstance(pads, relax.Constant):
            pad_before, pad_after = _np.split(pads.data.numpy(), 2)
            pad_before = _np.ndarray.tolist(pad_before)
            pad_after = _np.ndarray.tolist(pad_after)
        else:
            raise ValueError("Dynamic pads are not supported yet.")

        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if pad_mode not in ["constant", "edge", "reflect"]:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        if pad_mode == "constant":
            return bb.emit_te(topi.nn.pad, inputs[0], pad_before, pad_after, constant_value)
        elif pad_mode == "reflect":
            return bb.emit_te(topi.nn.mirror_pad, inputs[0], pad_before, pad_after, "REFLECT")
        else:
            # edge mode - replicate border values
            return bb.emit_te(topi.nn.replicate_pad, inputs[0], pad_before, pad_after)

