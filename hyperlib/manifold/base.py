import tensorflow as tf

class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.name = "manifold"
        self.min_norm = 1e-15
        self.eps = {tf.float32: 4e-3, tf.float64: 1e-5}


    def expmap(self, u, p, c):
        raise NotImplementedError

    def expmap0(self, u, c):
        """
        Hyperbolic exponential map at zero.
          Args:
            u: tensor of size B x dimension representing tangent vectors.
            c: tensor of size 1 representing the hyperbolic curvature.
          Returns:
            Tensor of shape B x dimension.
          """
        raise NotImplementedError

    def logmap0(self, p, c):
        """
        Hyperbolic logarithmic map at zero.
        Args:
          p: tensor of size B x dimension representing hyperbolic points.
          c: tensor of size 1 representing the hyperbolic curvature.
        Returns:
          Tensor of shape B x dimension.
        """
        raise NotImplementedError

    def proj(self, x, c):
        """
        Safe projection on the manifold for numerical stability. This was mentioned in [1]

        Args:
            x : Tensor point on the Poincare ball
            c : Tensor of size 1 representing the hyperbolic curvature.

        Returns:
            Projected vector on the manifold

        References:
            [1] Hyperbolic Neural Networks, NIPS2018
            https://arxiv.org/abs/1805.09112
        """

        raise NotImplementedError

    def hyp_act(self, act, c_in, c_out):
        """Apply an activation function to a tensor in the hyperbolic space"""
        raise NotImplementedError
