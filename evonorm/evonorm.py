import tensorflow as tf

DEFAULT_EPSILON_VALUE = 1e-5


def instance_std(x, eps=DEFAULT_EPSILON_VALUE):
    _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    return tf.sqrt(var + eps)


def group_std(inputs, groups=2, eps=DEFAULT_EPSILON_VALUE, axis=-1):
    input_shape = tf.keras.backend.int_shape(inputs)
    tensor_input_shape = tf.shape(inputs)
    group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
    group_shape[axis] = input_shape[axis] // groups
    group_shape.insert(axis, groups)
    group_shape = tf.stack(group_shape)
    reshaped_inputs = tf.reshape(inputs, group_shape)

    _, var = tf.nn.moments(reshaped_inputs, [1, 2, 4], keepdims=True)

    std = tf.sqrt(var + eps)
    return tf.reshape(std, tensor_input_shape)


class EvoNormB0(tf.keras.layers.Layer):
    pass


class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(EvoNormS0, self).__init__()

        self.gamma = self.add_weight(name="gamma", shape=(1, 1, channels))
        self.beta = self.add_weight(name="beta", shape=(1, 1, channels))
        self.v_1 = self.add_weight(name="v1", shape=(1, 1, channels))

    def call(self, inputs, training=True):
        return (inputs * tf.sigmoid(self.v_1 * inputs)) / group_std(inputs) * self.gamma + self.beta


if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(416, 416, 3), filters=16, kernel_size=3, padding="same"),
        EvoNormS0(channels=16)
    ])
    model.summary()
