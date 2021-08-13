import tensorflow as tf
from tensorflow.keras import backend


class MinimalHYPRNNCell(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 c,
                 activation,
                 manifold,
                 dtype=tf.float64, use_bias=True, hyperbolic_input=False, hyperbolic_weights=True, **kwargs):
        self.units = units
        self.state_size = units
        self.manifold = manifold
        self.__dtype = dtype
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_weights = hyperbolic_weights
        self.activation = tf.keras.activations.get(activation)
        self.c = tf.Variable([c], dtype="float64")
        self.matrix_initializer = tf.initializers.GlorotUniform()
        self.use_bias = use_bias
        tf.keras.backend.set_floatx('float64')
        tf.executing_eagerly()

        super(MinimalHYPRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units),
                initializer="zeros",
                dtype=tf.float64,
                trainable=True,
            )

        self.built = True
        super().build(input_shape)


    # Performs the hyperbolic version of the operation Wh + Ux + b.
    def one_rnn_transform(self, W, h, U, x, b):
        if not self.hyperbolic_input:
            h =  self.manifold.expmap0(h, c=self.c)
            h = self.manifold.proj(h, self.c)
        if not self.hyperbolic_weights:
            W =  self.manifold.expmap0(W, c=self.c)
            W = self.manifold.proj(W, self.c)

        W_otimes_h = self.manifold.mobius_matvec(W, h, self.c)
        U_otimes_x = self.manifold.mobius_matvec(U, x, self.c)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, self.c)
        hyp_bias = self.manifold.expmap0(b, self.c)
        hyp_bias = self.manifold.proj(hyp_bias, self.c)
        res = self.manifold.mobius_add(Wh_plus_Ux, hyp_bias, self.c)
        result = self.manifold.proj(res, self.c)
        return result

    def call(self, inputs, states):
        previous_output = states[0]
        new_h = self.one_rnn_transform(self.kernel, previous_output, self.recurrent_kernel, inputs, self.bias)
        output = self.manifold.hyp_act(self.activation, new_h, c_in=self.c, c_out=self.c)
        return output, [output]
