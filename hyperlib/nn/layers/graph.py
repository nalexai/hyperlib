import tensorflow as tf
from tensorflow import keras

from .linear import LinearHyperbolic, ActivationHyperbolic


class HyperbolicAggregation(keras.Layer):

    def call():
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

class HGCLayer(keras.Layer):
    def __init__(self):
        super().__init__()

        self.manifold = Lorentz()
        self.c = tf.Variable([0.4], trainable=True)

    def call(self, inputs):
        # Step 1 (hyperbolic feature transform)
        inputs = self.manifold.logmap0(inputs, c=self.c)

        # Step 2 (attention-based neighborhood aggregation)
        x = LinearHyperbolic(inputs)

        # Step 3 (non-linear activation with different curvatures)
        x = ActivationHyperbolic(x)

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


class HGCN(keras.Model):


    def call(self, inputs):

        # Map euclidean features to Hyperbolic space

        # Stack multiple hyperbolic graph convolution layers
