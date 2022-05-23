import tensorflow as tf
from tensorflow import keras

from .linear import LinearHyperbolic, ActivationHyperbolic

class ConvHyperbolic(keras.layers.Layer):
    def __init__(
        self, 
        units, 
        manifold, 
        c_in, 
        c_out, 
        activation,
        use_bias=True
    ):
        super().__init__()
        self.linear = LinearHyperbolic(units, manifold, c_in, use_bias=use_bias)
        self.activation = ActivationHyperbolic(manifold, c_in, c_out, activation)
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out

    def call(self, inputs):
        x, adj = inputs
        x = self.linear(x)

        # simple uniform aggregation
        x_tan = self.manifold.logmap0(x, c=self.c_in)
        x_agg = tf.sparse.sparse_dense_matmul(adj, x_tan)
        
        out = self.manifold.expmap0(x_tan, c=self.c_in)
        out = self.manifold.proj(out, c=self.c_in)
        out = self.activation(out)
        return out, adj

