import tensorflow as tf
from tensorflow import keras

from .linear import LinearHyperbolic, ActivationHyperbolic


class HyperbolicAggregation(keras.Layer):

    def _init_(self, manifold, c):
        self.manifold = manifold
        self.c = c

    def call(self, inputs):
        x_tangent, adj = inputs
        support_t = tf.sparse.sparse_dense_matmul(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

class HGCLayer(keras.Layer):
    def __init__(self, linear_layer, aggregation_layer, activation_layer):
        super().__init__()

        self.manifold = Lorentz()
        self.c = tf.Variable([0.4], trainable=True)
        self.linear_layer = linear_layer
        self.aggregation_layer = aggregation_layer
        self.activation_layer = activation_layeractivation_layer

    def call(self, inputs):
        # Step 1 (hyperbolic feature transform)
        x, adj = inputs
        x = self.manifold.logmap0(x, c=self.c)

        # Step 2 (attention-based neighborhood aggregation)
        x = linear(inputs)
        x = aggregation_layer((x, adj))

        # Step 3 (non-linear activation with different curvatures)
        x = activation_layer(x)

        # Notes
        # Note 1:  Hyperbolic embeddings at the last layer can then be used to predict node attributes or links
        # Note 2: For link prediction we use the Fermi-Dirac decoder , a generalization of sigmoid,
        #         to compute probability scores for edges. We then train HGCN by minimizing the
        #         cross-entropy loss using negative sampling
        # Note 3: For node classification  map the output of the last HGCN layer to tangent space of the origin with the
        #         logarithmic map and then perform Euclidean multinomial logistic regression. Note that another possibility
        #         is to directly classify points on the hyperboloid manifold using the hyperbolic multinomial logistic loss.
        #         This method performs similarly to Euclidean classification. Finally, we also add a link prediction
        #         regularization objective in node classification tasks, to encourage embeddings at the last layer to
        #         preserve the graph structure

        return x


class HGCN(keras.Model):


    def call(self, inputs):

        # Map euclidean features to Hyperbolic space

        # Stack multiple hyperbolic graph convolution layers
