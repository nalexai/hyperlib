import tensorflow as tf
from tensorflow import keras
import ..utils.math as hmath

class LinearHyperbolic(keras.layers.Layer):
    """
    Implementation of a hyperbolic linear layer for a neural network, that inherits from the keras Layer class
    """

    def __init__(self, units, c, activation=None, use_bias=True):
        super().__init__()
        self.units = units
        self.c = tf.Variable([c], dtype="float64")
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, batch_input_shape):
        w_init = tf.random_normal_initializer()
        self.kernel = tf.Variable(
            initial_value=w_init(shape=(batch_input_shape[-1], self.units), dtype="float64"), dtype="float64", trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units),
                initializer="zeros",
                dtype=tf.float64,
                trainable=True,
            )

        super().build(batch_input_shape)  # must be at the end

    def call(self, inputs):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """
        # TODO: remove casting and instead recommend setting default tfd values to float64
        inputs = tf.cast(inputs, tf.float64)
        mv = hmath.mobius_matvec(self.kernel, inputs, self.c)
        res = hmath.proj(mv, self.c)

        if self.use_bias:
            hyp_bias = hmath.expmap0(self.bias, self.c)
            hyp_bias = hmath.proj(hyp_bias, self.c)
            res = hmath.mobius_add(res, hyp_bias, c=self.c)
            res = hmath.proj(res, self.c)

        return self.activation(res)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "curvature": self.c
        }
