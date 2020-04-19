import tensorflow as tf

DEFAULT_EPSILON_VALUE = 1e-5


def instance_std(x, eps=DEFAULT_EPSILON_VALUE):
    _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    return tf.sqrt(var + eps)


# def group_std(x, groups=32, eps=1e-5):
#     N, H, W, C = tf.shape(x)
#     x = tf.reshape(x, [N, H, W, groups, C // groups])
#     _, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
#     std = tf.sqrt(var + eps)
#     std = tf.broadcast_to(std, x.shape)
#     return tf.reshape(std, (N, H, W, C))


def group_std(inputs, groups=32, eps=DEFAULT_EPSILON_VALUE, axis=-1):
    groups = min(inputs.shape[axis], groups)

    input_shape = tf.shape(inputs)
    group_shape = [input_shape[i] for i in range(4)]
    group_shape[axis] = input_shape[axis] // groups
    group_shape.insert(axis, groups)
    group_shape = tf.stack(group_shape)
    grouped_inputs = tf.reshape(inputs, group_shape)
    _, var = tf.nn.moments(grouped_inputs, [1, 2, 4], keepdims=True)

    std = tf.sqrt(var + eps)
    std = tf.broadcast_to(std, tf.shape(grouped_inputs))
    return tf.reshape(std, input_shape)


class EvoNormB0(tf.keras.layers.Layer):
    pass

class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, channels, groups=8):
        super(EvoNormS0, self).__init__()

        self.groups = groups

        self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=tf.initializers.Constant(1.))
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=tf.initializers.Constant(0.))
        self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, channels), initializer=tf.initializers.Constant(1.))

    def call(self, inputs, training=True):
        return (inputs * tf.sigmoid(self.v_1 * inputs)) / group_std(inputs, groups=self.groups) * self.gamma + self.beta


if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(416, 416, 3), filters=32, kernel_size=3, padding="same"),
        EvoNormS0(channels=32, groups=16)
    ])
    model.summary()
