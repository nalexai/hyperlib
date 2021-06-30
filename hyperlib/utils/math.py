"""
Implementation of matrix, vector, and scalar functions in the Poincare ball model
of n-dimensional hyperbolic space with curvature -c (c > 0) i.e. the set 
math::
        \{ \|x\|^2 < 1/c \}  with metric
        g = \frac{4}{ (1 - \|x\|^2)^2 } g_{Euclidean} 
"""

import tensorflow as tf
from math import sqrt

MIN_NORM = 1e-15
EPS = {tf.float32: 4e-3, tf.float64: 1e-5}

# Hyperbolic trig functions 
def cosh(x, clamp=15):
    return tf.math.cosh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

def sinh(x, clamp=15):
    return tf.math.sinh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

def tanh(x, clamp=15):
    return tf.math.tanh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

def arcosh(x):
    return Arcosh.apply(x)

def arsinh(x):
    return Arsinh.apply(x)

def atanh_(x):
    x = tf.clip_by_value(x, clip_value_min=-1 + 1e-15, clip_value_max=1 - 1e-15)
    return tf.math.atanh(x)

@tf.custom_gradient
def custom_artanh_cg(x):
    x = tf.clip_by_value(x, clip_value_min=-1 + 1e-15, clip_value_max=1 - 1e-15)
    z = tf.cast(x, tf.float64, name=None)
    temp = tf.math.subtract(tf.math.log(1 + z), (tf.math.log(1 - z)))
    def artanh_grad(grad):
        return grad/ (1 - x ** 2)

    return tf.cast(tf.math.multiply(temp, 0.5), x.dtype, name=None), artanh_grad


# Hyperbolic mat/vec functions

def hyp_dist(x, y, c=1.0):
    """ The hyperbolic distance between x and y """ 
    sqrt_c = sqrt(c) 
    return 2*atanh_( sqrt_c * clipped_norm( mobius_add(-x, y, c) ) ) / sqrt_c

def mobius_matvec(m, x, c=1.0):
    """
    Generalization for matrix-vector multiplication to hyperbolic space defined as
    math::
        M \otimes_c x = (1/\sqrt{c}) \tanh\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}
    Args:
        m : Tensor for multiplication
        x : Tensor point on Poincare ball
        c : Tensor of size 1 representing the hyperbolic curvature.
    Returns
        Mobius matvec result
    """
    x_norm = clipped_norm(x)  
    mx = x @ m
    mx_norm = clipped_norm(mx) 
    sqrt_c = sqrt(c)
    res_c = (
        tanh(mx_norm / x_norm * atanh_(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    )
    cond = tf.reduce_prod(
        tf.cast((mx == 0), tf.uint8, name=None), axis=-1, keepdims=True
    )
    res_0 = tf.zeros(1, dtype=res_c.dtype)
    res = tf.where(tf.cast(cond, tf.bool), res_0, res_c)
    return res

def clipped_norm(x, max_norm = None):
    """ Clipped Euclidean norm of x """ 
    x_norm = tf.norm(x, axis=-1, ord=2, keepdims=True)
    if max_norm is None:
        max_norm= tf.math.reduce_max(x_norm)
    return tf.clip_by_value(
        x_norm, 
        clip_value_min=MIN_NORM,
        clip_value_max=max_norm,
    )

def lambda_x(x, c=1.0):
    """ Poincare conformal factor at point x """ 
    cx2 = c * tf.reduce_sum(x * x, axis=-1, keepdims=True)
    return 2.0 / (1.0 - cx2)

def expmap(u, x, c=1.0):
    """ Exponential map of u at p in the Poincare ball """ 
    u += 1e-15 #avoid u=0
    sqrt_c = sqrt(c) 
    u_norm = clipped_norm(u)
    second_term = (
        tanh(sqrt_c / 2 * lambda_x(x, c) * u_norm) * u / (sqrt_c * u_norm)
    )
    return mobius_add(x, second_term, c)

def expmap0(u, c=1.0):
    """
    Hyperbolic exponential map at zero in the Poincare ball model.
      Args:
        u: tensor of size B x dimension representing tangent vectors.
        c: tensor of size 1 representing the hyperbolic curvature.
      Returns:
        Tensor of shape B x dimension.
    """
    sqrt_c = sqrt(c) 
    u_norm = clipped_norm(u)
    return tf.math.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

def logmap0(p, c=1.0):
    """
    Hyperbolic logarithmic map at zero in the Poincare ball model.
    Args:
        p: tensor of size B x dimension representing hyperbolic points.
        c: tensor of size 1 representing the hyperbolic curvature.
    Returns:
        Tensor of shape B x dimension.
    """
    sqrt_c = sqrt(c) 
    p_norm = clipped_norm(p) 
    scale = atanh_(sqrt_c * p_norm) / (p_norm * sqrt_c)
    return scale * p

def proj(x, c=1.0):
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
    x_norm = clipped_norm(x)
    maxnorm = (1 - EPS[x.dtype]) / sqrt(c)  
    cond = x_norm > maxnorm
    projected = x / x_norm * maxnorm
    return tf.where(cond, projected, x)

def mobius_add(x, y, c=1.0):
    """
    Element-wise Mobius addition.
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
    return proj(num / tf.maximum(denom, MIN_NORM), c)

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
    return tf.add(2*tf.divide(A * x - B * y, C), z)

def parallel_transport(x, y, v, c=1.0):
    """
    The parallel transport of the tangent vector v from the tangent space at x
    to the tangent space at y
    """
    return tf.divide(lambda_x(x,c), lambda_x(y,c)) * gyr(y,-x,v)
