import numpy as np
import tensorflow as tf

from .base import Manifold
from ..utils.math import cosh, sinh


class Lorentz(Manifold):
    """
    Implementation of the Lorentz/Hyperboloid manifold defined by
    :math: `L = \{ x \in R^d | -x_0^2 + x_1^2 + ... + x_d^2 = -K \}`, 
    where c = 1 / K is the hyperbolic curvature and d is the manifold dimension.
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

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = tf.clip_by_value(-prod / K, 
            clip_value_min=1.0 + self.eps[x.dtype], clip_value_max=self.max_norm)
        sqdist = K * tf.math.arcosh(theta) ** 2
        return sqdist

    def proj(self, x, c):
        """Projects point (d+1)-dimensional point x to the manifold"""
        K = 1. / c
        d = x.shape[-1]
        y = x[:,1:d]
        y_sqnorm = tf.norm(y, ord=2, axis=1, keepdims=True) ** 2
        max_num = tf.math.reduce_max(K + y_sqnorm)
        t = tf.clip_by_value(
            K + y_sqnorm, clip_value_min=self.eps[x.dtype], clip_value_max=max_num
        )
        return tf.concat([t, y], axis=1)

    def proj_tan(self, u, x, c):
        """Projects vector u onto the tangent space at x.
        Note: this is not the orthogonal projection"""
        d = x.shape[-1]
        ud = u[:,1:d]
        ux = tf.math.reduce_sum( x[:,1:d]*ud, axis=1, keepdims=True)
        x0 = tf.clip_by_value(x[:,0:1], clip_value_min=self.eps[x.dtype], clip_value_max=1e5)
        return tf.concat( [ux/x0, ud], axis=1 )

    def proj_tan0(self, u, c):
        narrowed = u[:,:1]
        vals = tf.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = self.clip_norm(normu)
        theta = normu / sqrtK
        theta = self.clip_norm(theta)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = tf.clip_by_value(self.minkowski_dot(x, y), 
            clip_value_min=-self.max_norm, clip_value_max=-self.eps[x.dtype]) 
        xy -= K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = self.clip_norm(normu)
        dist = tf.math.sqrt(self.sqdist(x, y, c))
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def hyp_act(self, act, x, c_in, c_out):
        """Apply an activation function to a tensor in the hyperbolic space"""
        xt = act(self.logmap0(x, c=c_in))
        return self.proj(self.expmap0(xt, c=c_out), c=c_out)

    def expmap0(self, u, c):
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
        K = 1. / c
        sqrtK = K ** 0.5
        b, d = x.shape
        y = tf.reshape(x[:,1:], [-1, d-1])
        y_norm = tf.norm(y, ord=2, axis=1, keepdims=True)
        y_norm = self.clip_norm(y_norm)
        theta = tf.clip_by_value(x[:, 0:1] / sqrtK, 
            clip_value_min=1.0+self.eps[x.dtype], clip_value_max=self.max_norm)
        res = sqrtK * tf.math.acosh(theta) * y / y_norm
        zeros = tf.zeros((b,1), dtype=res.dtype)
        return tf.concat([zeros, res], axis=1)

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        #print('m',m.shape)
        #print('x',x.shape)
        u = self.logmap0(x, c)
        mu = u @ m #hgcn: #mu = u @ tf.transpose(m)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = self.clip_norm(self.sqdist(x, y, c))
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def clip_norm(self, x):
        return tf.clip_by_value(x, clip_value_min=self.min_norm, clip_value_max=self.max_norm)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x[:,:1]
        d = x.shape[-1] - 1
        y = x[:,1:d]

        y_norm = tf.norm(y, ord=2, axis=1, keepdims=True)
        y_norm = self.clip_norm(y_norm)
        y_normalized = y / y_norm
        v = tf.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = tf.math.reduce_sum(y_normalized * u[:, 1:], axis=1, keepdims=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.shape[-1] - 1
        return sqrtK * x[:,1:d] / (x[:, 0:1] + sqrtK)
