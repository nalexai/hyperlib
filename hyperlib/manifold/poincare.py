import tensorflow as tf
from math import sqrt
from ..utils.math import tanh, atanh_


class Poincare:
    """
    Implementation of the poincare manifold,. This class can be used for mathematical functions on the poincare manifold.
    """

    def __init__(self,):
        super(Poincare, self).__init__()
        self.name = "PoincareBall"
        self.min_norm = 1e-15
        self.eps = {tf.float32: 4e-3, tf.float64: 1e-5}

    def mobius_matvec(self, m, x, c=1.0):
        """
        Generalization for matrix-vector multiplication to hyperbolic space defined as
        math::
            M \otimes_c x = (1/\sqrt{c}) \tanh\left(
                \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
            \right)\frac{Mx}{\|Mx\|_2}
        Args:
            m : Tensor for multiplication
            x : Tensor point on poincare ball
            c : Tensor of size 1 representing the hyperbolic curvature.
        Returns
            Mobius matvec result
        """

        sqrt_c = sqrt(c) 
        x_norm = self.clipped_norm(x)
        mx = x @ m
        mx_norm = self.clipped_norm(x)

        res_c = (
            tanh(mx_norm / x_norm * atanh_(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        )
        cond = tf.reduce_prod(
            tf.cast((mx == 0), tf.uint8, name=None), axis=-1, keepdims=True
        )
        res_0 = tf.zeros(1, dtype=res_c.dtype)
        res = tf.where(tf.cast(cond, tf.bool), res_0, res_c)
        return res

    def clipped_norm(self, x):
        x_norm = tf.norm(x, axis=-1, ord=2, keepdims=True)
        max_num = tf.math.reduce_max(x_norm)
        x_norm = tf.clip_by_value(
                x_norm, clip_value_min=self.min_norm, clip_value_max=max_num
                )
        return x_norm

    def expmap(self, u, p, c=1.0):
        sqrt_c = sqrt(c) 
        u_norm = self.clipped_norm(u) 
        second_term = (
            tanh(sqrt_c / 2 * self.lambda_x(p, c) * u_norm) * u / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def lambda_x(self, x, c=1.0):
        x_norm = self.clipped_norm(x) 
        return 2.0 / (1.0 - c*x_norm)

    def expmap0(self, u, c=1.0):
        """
        Hyperbolic exponential map at zero in the Poincare ball model.
          Args:
            u: tensor of size B x dimension representing tangent vectors.
            c: tensor of size 1 representing the hyperbolic curvature.
          Returns:
            Tensor of shape B x dimension.
          """
        sqrt_c = sqrt(c) 
        max_num = tf.math.reduce_max(u)
        u_norm = self.clipped_norm(u)
        gamma_1 = tf.math.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c=1.0):
        """
        Hyperbolic logarithmic map at zero in the Poincare ball model.
        Args:
          p: tensor of size B x dimension representing hyperbolic points.
          c: tensor of size 1 representing the hyperbolic curvature.
        Returns:
          Tensor of shape B x dimension.
        """
        sqrt_c = sqrt(c) 
        p_norm = self.clipped_norm(p)
        scale = 1.0 / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def proj(self, x, c=1.0):
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

        norm = self.clipped_norm(x)
        maxnorm = (1 - self.eps[x.dtype]) / sqrt(c)  # tf.math.reduce_max(x)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return tf.where(cond, projected, x)

    def mobius_add(self, x, y, c=1.0):
        """Element-wise Mobius addition.
      Args:
        x: Tensor of size B x dimension representing hyperbolic points.
        y: Tensor of size B x dimension representing hyperbolic points.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.
      Returns:
        Tensor of shape B x dimension representing the element-wise Mobius addition
        of x and y.
      """
        cx2 = c * tf.reduce_sum(x * x, axis=-1, keepdims=True)
        cy2 = c * tf.reduce_sum(y * y, axis=-1, keepdims=True)
        cxy = c * tf.reduce_sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
        denom = 1 + 2 * cxy + cx2 * cy2
        return self.proj(num / tf.maximum(denom, self.min_norm), c)
    
    def gyr(x, y, z, c=1.0):
        """
        Ungar's gryation operation defined in [1].
        math::
            gyr[x,y]z = \ominus (x \oplus y)\oplus(x \oplus (y \oplus z))
            
            where \oplus is Mobius addition and \ominus is the left inverse.

        Args:
            x, y, z: Tensors of size B x dim in the Poincare ball of curvature c
        Returns:
            Tensor of size B x dim
        Reference:
           [1] A. Ungar, A Gryovector Space Approach to Hyperbolic Geometry
        """
        xy = tf.reduce_sum( x*y, axis=-1, keepdims=True)
        yz = tf.reduce_sum( y*z, axis=-1, keepdims=True)
        xz = tf.reduce_sum( x*z, axis=-1, keepdims=True)
        x2 = tf.reduce_sum( x*x, axis=-1, keepdims=True)
        y2 = tf.reduce_sum( y*y, axis=-1, keepdims=True)
        z2 = tf.reduce_sum( z*z, axis=-1, keepdims=True)
        A = c*yz - c**2 * xz * y2 + 2 * c**2 * xy * yz
        B = c**2 * yz * x2 + c * xz
        C = 1 + 2 * c* xy + c**2 * x2 * y2
        return 2*(A * x - B * y)/tf.maximum(C, self.min_norm)+ z

    def parallel_transport(self, x, y, v, c=1.0):
        """
        The parallel transport of the tangent vector v from the tangent space at x
        to the tangent space at y
        """
        return self.lambda_x(x,c)/ self.lambda_x(y,c) * self.gyr(y,-x,v)

    def dist(self, x, y, c=1.0):
        sqrt_c = sqrt(c)
        norm = tf.norm(self.mobius_add(-x,y,c) + self.eps[x.dtype], 
                        axis=1, keepdims=True)
        return 2./sqrt_c * atanh_( sqrt_c * norm)
