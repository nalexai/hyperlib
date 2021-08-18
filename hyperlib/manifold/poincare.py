import tensorflow as tf
import numpy as np
from math import sqrt
from ..utils.math import tanh, atanh_

class Poincare:
    """
    This class can be used for mathematical functions on the Poincare ball.
    The Poincare n-ball with curvature -c< 0 is the set of points n-dim Euclidean space such that |x|^2 < 1/c
    with a Riemannian metric given by 
    math::
        \lambda_c g_E = \frac{1}{1-c \|x\|^2} g_E
    where g_E is the standard Euclidean metric.
    """

    def __init__(self, c=1.0):
        super(Poincare, self).__init__()
        self.name = "PoincareBall"
        assert c>0
        self._c = c
        self._sqrt_c = sqrt(c)
        self.min_norm = 1e-15
        self.eps = {tf.float32: 4e-3, tf.float64: 1e-5,
                    np.dtype('float32'): 4e-3, np.dtype('float64'): 1e-5}

    @property
    def curvature(self):
        return self._c

    @curvature.setter
    def curvature(self, c):
        assert c>0
        self._c = c
        self._sqrt_c = sqrt(c)

    def mobius_matvec(self, m, x):
        """
        Generalization for matrix-vector multiplication to hyperbolic space defined as
        math::
            M \otimes_c x = (1/\sqrt{c}) \tanh\left(
                \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
            \right)\frac{Mx}{\|Mx\|_2}
        Args:
            m : Tensor for multiplication
            x : Tensor point on poincare ball
        Returns
            Mobius matvec result
        """

        x_norm = self.clipped_norm(x)
        mx = x @ m
        mx_norm = self.clipped_norm(x)
        res_c = (
            tanh(mx_norm / x_norm * atanh_(self._sqrt_c * x_norm)) * mx / (mx_norm * self._sqrt_c)
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

    def expmap(self, u, p):
        u_norm = self.clipped_norm(u) 
        second_term = (
            tanh(self._sqrt_c / 2 * self.lambda_x(p) * u_norm) * u / (self._sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term)
        return gamma_1

    def lambda_x(self, x):
        x_norm2 = tf.square(self.clipped_norm(x))
        return 2.0 / (1.0 - self._c*x_norm2)

    def expmap0(self, u):
        """
        Hyperbolic exponential map at zero in the Poincare ball model.
          Args:
            u: tensor of size B x dimension representing tangent vectors.
          Returns:
            Tensor of shape B x dimension.
          """
        max_num = tf.math.reduce_max(u)
        u_norm = self.clipped_norm(u)
        gamma_1 = tf.math.tanh(self._sqrt_c * u_norm) * u / (self._sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p):
        """
        Hyperbolic logarithmic map at zero in the Poincare ball model.
        Args:
          p: tensor of size B x dimension representing hyperbolic points.
        Returns:
          Tensor of shape B x dimension.
        """
        p_norm = self.clipped_norm(p)
        scale = 1.0 / self._sqrt_c * artanh(self._sqrt_c * p_norm) / p_norm
        return scale * p

    def proj(self, x):
        """
        Safe projection on the manifold for numerical stability. This was mentioned in [1]

        Args:
            x : Tensor point on the Poincare ball
        Returns:
            Projected vector on the manifold

        References:
            [1] Hyperbolic Neural Networks, NIPS2018
            https://arxiv.org/abs/1805.09112
        """
        norm = self.clipped_norm(x)
        maxnorm = (1 - self.eps[x.dtype]) / self._sqrt_c  # tf.math.reduce_max(x)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return tf.where(cond, projected, x)

    def mobius_add(self, x, y):
        """Element-wise Mobius addition.
      Args:
        x: Tensor of size B x dimension representing hyperbolic points.
        y: Tensor of size B x dimension representing hyperbolic points.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.
      Returns:
        Tensor of shape B x dimension representing the element-wise Mobius addition
        of x and y.
      """
        cx2 = self._c * tf.reduce_sum(x * x, axis=-1, keepdims=True)
        cy2 = self._c * tf.reduce_sum(y * y, axis=-1, keepdims=True)
        cxy = self._c * tf.reduce_sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
        denom = 1 + 2 * cxy + cx2 * cy2
        return self.proj(num / tf.maximum(denom, self.min_norm))
    
    def gyr(x, y, z):
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
        A = self._c*yz - self._c**2 * xz * y2 + 2 * self._c**2 * xy * yz
        B = self._c**2 * yz * x2 + self._c * xz
        C = 1 + 2 * self._c* xy + self._c**2 * x2 * y2
        return 2*(A * x - B * y)/tf.maximum(C, self.min_norm)+ z

    def parallel_transport(self, x, y, v):
        """
        The parallel transport of the tangent vector v from the tangent space at x
        to the tangent space at y
        """
        return self.lambda_x(x)/ self.lambda_x(y) * self.gyr(y,-x,v)

    def dist(self, x, y):
        """ Hyperbolic distance between points 
        Args:
            x, y: Tensors of size B x dim of points in the Poincare ball
        """
        norm = tf.norm(self.mobius_add(-x,y) + self.eps[x.dtype], 
                        axis=1, 
                        keepdims=True
                        )
        return 2./self._sqrt_c * atanh_( self._sqrt_c * norm)

    def reflect(self, a, x):
        """ Hyperbolic reflection with center at a, i.e. sphere inversion
        about the sphere centered at a orthogonal to Poincare ball.

        math::
            \frac{r^2}{\|x-a\|^2} (x-a) + a,    where r^2 + \frac{1}{c} = \|a\|^2
        Args:
            a: Tensor representing center of reflection 
            x: Tensor of size B x dim in the Poincare ball
        Returns:
            size B x dim Tensor of reflected points
        """
        a_norm = tf.norm(a, ord=2)
        r2 = tf.square(a_norm) - 1/self._c
        return r2/tf.square(self.clipped_norm(x-a)) * (x-a) + a

    def reflect0(self, z, x):
        """ Hyperbolic reflection that maps z to 0 and 0 to z. 

        Args:
            z: point in the Poincare ball that maps to the origin 
            x: Tensor of size B x dim representing B points in the Poincare ball to reflect
        Returns:
            size B x dim Tensor of reflected points
        """
        z_norm2 = tf.square(self.clipped_norm(z))
        a = self._c*z/z_norm2
        return self.reflect(a,x)
