import numpy as np
import tensorflow as tf

from .base import Manifold
from ..utils.functional import cosh, sinh, arcosh

class Lorentz(Manifold):
    """
    Implementation of the Lorentz/Hyperboloid manifold defined by
    :math: `L = \{ x \in R^d | -x_0^2 + x_1^2 + ... + x_d^2 = -K \}`, 
    where c = 1 / K is the hyperbolic curvature and d is the manifold dimension.

    The point :math: `( \sqrt{K}, 0, \dots, 0 )` is referred to as "zero".
    """

    def __init__(self):
        super(Lorentz).__init__()
        self.name = 'Lorentz'
        self.eps = {tf.float32: 1e-7, tf.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = tf.math.reduce_sum(x * y, axis=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = tf.reshape(res, res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        t = tf.clip_by_value(
            dot, clip_value_min=self.eps[u.dtype], clip_value_max=self.max_norm)
        return tf.math.sqrt(t)

    def dist_squared(self, x, y, c):
        """Squared hyperbolic distance between x, y"""
        K = 1. / c
        theta = tf.clip_by_value( -self.minkowski_dot(x, y) / K, 
            clip_value_min=1.0 + self.eps[x.dtype], clip_value_max=self.max_norm)
        return K * arcosh(theta)**2

    def proj(self, x, c):
        """Projects point (d+1)-dimensional point x to the manifold"""
        K = 1. / c
        d1 = x.shape[-1]
        y = x[:,1:d1]
        y_sqnorm = tf.math.square( 
            tf.norm(y, ord=2, axis=1, keepdims=True))
        t = tf.clip_by_value(K + y_sqnorm, 
            clip_value_min=self.eps[x.dtype],
            clip_value_max=self.max_norm
        )
        return tf.concat([tf.math.sqrt(t), y], axis=1)

    def proj_tan(self, u, x, c):
        """Projects vector u onto the tangent space at x.
        Note: this is not the orthogonal projection"""
        d1 = x.shape[-1]
        ud = u[:,1:d1]
        ux = tf.math.reduce_sum( x[:,1:d1]*ud, axis=1, keepdims=True)
        x0 = tf.clip_by_value(x[:,0:1], clip_value_min=self.eps[x.dtype], clip_value_max=1e5)
        return tf.concat( [ux/x0, ud], axis=1 )

    def proj_tan0(self, u, c):
        """Projects vector u onto the tangent space at zero.
        See also: Lorentz.proj_tan"""
        b, d1 = u.shape
        z = tf.zeros((b,1), dtype=u.dtype)
        ud = u[:,1:d1]
        return tf.concat([z, ud], axis=1)

    def expmap(self, u, x, c):
        """Maps vector u in the tangent space at x onto the manifold""" 
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = self.clip_norm(normu)
        theta = normu / sqrtK
        theta = self.clip_norm(theta)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, y, x, c):
        """Maps point y in the manifold to the tangent space at x"""
        K = 1. / c
        xy = tf.clip_by_value(self.minkowski_dot(x, y) + K, 
            clip_value_min=-self.max_norm, clip_value_max=-self.eps[x.dtype]) 
        xy -= K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = self.clip_norm(normu)
        dist = tf.math.sqrt(self.dist_squared(x, y, c))
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def hyp_act(self, act, x, c_in, c_out):
        """Apply an activation function to a tensor in the hyperbolic space"""
        xt = act(self.logmap0(x, c=c_in))
        return self.proj(self.expmap0(xt, c=c_out), c=c_out)

    def expmap0(self, u, c):
        """Maps vector u in the tangent space at zero onto the manifold""" 
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.shape[-1]
        x = tf.reshape(u[:,1:d], [-1, d-1])
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = self.clip_norm(x_norm)
        theta = x_norm / sqrtK
        res = tf.ones_like(u)
        b, d = res.shape
        res1 = tf.ones((b,1), dtype=res.dtype)
        res1 *= sqrtK * cosh(theta)
        res2 = tf.ones((b,d-1), dtype=res.dtype)
        res2 *= sqrtK * sinh(theta) * (x / x_norm)
        res = tf.concat([res1,res2], axis=1)
        return self.proj(res, c)

    def logmap0(self, x, c):
        """Maps point y in the manifold to the tangent space at zero.
        See also: Lorentz.logmap"""
        K = 1. / c
        sqrtK = K ** 0.5
        b, d = x.shape
        y = tf.reshape(x[:,1:], [-1, d-1])
        y_norm = tf.norm(y, ord=2, axis=1, keepdims=True)
        y_norm = self.clip_norm(y_norm)
        theta = tf.clip_by_value(x[:, 0:1] / sqrtK, 
            clip_value_min=1.0+self.eps[x.dtype], clip_value_max=self.max_norm)
        res = sqrtK * arcosh(theta) * y / y_norm
        zeros = tf.zeros((b,1), dtype=res.dtype)
        return tf.concat([zeros, res], axis=1)

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m 
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        """Parallel transport a vector u in the tangent space at x
        to the tangent space at y"""
        log_xy = self.logmap(y, x, c)
        log_yx= self.logmap(x, y, c)
        dist_squared = self.clip_norm(self.dist_squared(x, y, c))
        alpha = self.minkowski_dot(log_xy, u) / dist_squared
        res = u - alpha * (log_xy + log_yx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        """Parallel transport a vector u in the tangent space at zero
        to the tangent space at x.
        See also: Lorentz.ptransp"""
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x[:,:1]
        d1 = x.shape[-1]
        y = x[:,1:d1]

        y_norm = tf.norm(y, ord=2, axis=1, keepdims=True)
        y_norm = self.clip_norm(y_norm)
        y_unit = y / y_norm
        v = tf.concat([y_norm, (sqrtK - x0)*y_unit], axis=1)

        alpha = tf.math.reduce_sum(y_unit * u[:, 1:], axis=1, keepdims=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d1 = x.shape[-1]
        return sqrtK * x[:,1:d1] / (x[:, 0:1] + sqrtK)

    def clip_norm(self, x):
        return tf.clip_by_value(x, clip_value_min=self.min_norm, clip_value_max=self.max_norm)
