import tensorflow as tf
from tensorflow import keras

from .linear import LinearHyperbolic, ActivationHyperbolic
from hyperlib.manifold.lorentz import Lorentz
from hyperlib.manifold.poincare import Poincare

class HyperbolicAggregation(keras.layers.Layer):

    def __init__(self, manifold, c):
        super().__init__()
        self.manifold = manifold
        self.c = c

    def call(self, inputs):
        x_tangent, adj = inputs
        support_t = tf.sparse.sparse_dense_matmul(adj, x_tangent)
        #support_t = tf.linalg.matmul(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

class HGCLayer(keras.layers.Layer):
    def __init__(self, manifold, input_size, c, activation):
        super().__init__()

        self.manifold = manifold
        self.c = tf.Variable([c], trainable=False)
        #self.linear_layer = LinearHyperbolic(input_size, self.manifold, self.c, activation=None)
        self.linear_layer = LinearHyperbolic(1433, self.manifold, 1.0, activation=None)
        self.aggregation_layer = HyperbolicAggregation(self.manifold, self.c)
        self.activation_layer = ActivationHyperbolic(self.manifold, self.c, self.c, activation)

    def call(self, inputs):
        # Step 1 (hyperbolic feature transform)
        x, adj = inputs
        x = self.manifold.logmap0(x, c=self.c)

        # Step 2 (attention-based neighborhood aggregation)
        print('HGCLayer x shape', x.shape)
        x = self.linear_layer(x)
        x = self.aggregation_layer((x, adj))

        # Step 3 (non-linear activation with different curvatures)
        x = self.activation_layer(x)

        return x


class HGCNLP(keras.Model):

    def __init__(self, input_size, dropout=0.4):
        super().__init__()

        self.input_size = input_size

        self.manifold = Lorentz()
        self.c_map = tf.Variable([0.4], trainable=False)
        self.c0 = tf.Variable([0.4], trainable=False)
        self.c1 = tf.Variable([0.4], trainable=False)
        self.c2 = tf.Variable([0.4], trainable=False)

        self.conv0 = HGCLayer(self.manifold, self.input_size, self.c0, activation="relu")
        self.conv1 = HGCLayer(self.manifold, self.input_size, self.c0, activation="relu")
        self.conv2 = HGCLayer(self.manifold, self.input_size, self.c0, activation="relu")

    def call(self, inputs):
        x, adj = inputs
        print('HGCNLP x shape', x.shape)
        # Map euclidean features to Hyperbolic space
        x = self.manifold.expmap0(x, c=self.c_map)
        # Stack multiple hyperbolic graph convolution layers
        x, adj = self.conv0((x, adj))
        x, adj = self.conv1((x, adj))
        x, adj = self.conv2((x, adj))

        # TODO - add link prediction/node classification code as described
        # in the notes below
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
        return


    def decode(self, emb_in, emb_out):
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        # fermi dirac to comput edge probabilities
        1. / (tf.exp((sqdist - self.r) / self.t) + 1.0)
        return probs
