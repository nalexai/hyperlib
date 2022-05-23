import tensorflow as tf
from tensorflow import keras


class LinearHyperbolic(keras.layers.Layer):
    """
    Implementation of a hyperbolic linear layer for a neural network, that inherits from the keras Layer class
    """

    def __init__(self, units, manifold, c, use_bias=True):
        super().__init__()
        self.units = units
        self.c = tf.Variable([c])
        self.manifold = manifold
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=((input_shape[-1], self.units)), 
            initializer="random_normal",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(1,self.units),
                initializer="zeros",
                trainable=True,
            )

    def call(self, inputs):
        """
        Called during forward pass of a neural network. Uses hyperbolic matrix multiplication
        """
        # TODO: remove casting and instead recommend setting default tfd values to float64
        # inputs = tf.cast(inputs, tf.float64)

        mv = self.manifold.mobius_matvec(self.kernel, inputs, self.c)
        res = self.manifold.proj(mv, c=self.c)

        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias, c=self.c)
            hyp_bias = self.manifold.proj(hyp_bias, c=self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, c=self.c)

        return res

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "manifold": self.manifold.name,
            "curvature": self.c
        }

class ActivationHyperbolic(keras.layers.Layer):
    def __init__(self, manifold, c_in, c_out, activation):
        super().__init__()
        self.activation = keras.activations.get(activation)
        self.c_in = c_in
        self.c_out = c_out
        self.manifold = manifold

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        inputs_tan = self.activation(self.manifold.logmap0(inputs, c=self.c_in))
        inputs_tan = self.manifold.proj_tan0(inputs_tan, self.activation(inputs))
        out = self.manifold.expmap0(inputs_tan, c=self.c_out)
        return self.manifold.proj(out, c=self.c_out)

    def get_config(self):
        return {
            "activation": keras.activations.serialize(self.activation),
            "c_in": self.c_in,
            "c_out": self.c_out,
            "manifold": self.manifold.name,
        }
