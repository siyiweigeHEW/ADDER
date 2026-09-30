
    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 1)
        data_shape = list(inputs[0].ty.shape)

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

