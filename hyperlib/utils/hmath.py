import tensorflow as tf

"""
Hyperbolic trig functions
"""

MIN_NORM = 1e-15
EPS = {tf.float32: 4e-3, tf.float64: 1e-5}

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

"""
Hyperbolic matrix and vector functions in the Poincare ball  
"""

def mobius_matvec(m, x, c):
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

    sqrt_c = c ** 0.5
    x_norm = norm(x)  
    mx = x @ m
    mx_norm = norm(mx) 

    res_c = (
        tanh(mx_norm / x_norm * atanh_(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    )
    cond = tf.reduce_prod(
        tf.cast((mx == 0), tf.uint8, name=None), axis=-1, keepdims=True
    )
    res_0 = tf.zeros(1, dtype=res_c.dtype)
    res = tf.where(tf.cast(cond, tf.bool), res_0, res_c)
    return res

def norm(x):
    """ Clipped Euclidean norm of x """ 
    x_norm = tf.norm(x, axis=-1, ord=2, keepdims=True)
    max_num = tf.math.reduce_max(x_norm)
    return tf.clip_by_value(
        x_norm, 
        clip_value_min=MIN_NORM,
        clip_value_max=max_num,
    )

def lambda_x(x, c):
    """ Poincare conformal factor at x """ 
    return 2.0 / (1 - c*norm(x))

def expmap(u, p, c):
    """ Exponential map of u at p in the Poincare ball """ 
    u += MIN_NORM
    sqrt_c = c ** 0.5
    u_norm = norm(u)
    second_term = (
        tanh(sqrt_c / 2 * lambda_x(p, c) * u_norm) * u / (sqrt_c * u_norm)
    )
    gamma_1 = mobius_add(p, second_term, c)
    return gamma_1

def expmap0(u, c):
    """
    Hyperbolic exponential map at zero in the Poincare ball model.
      Args:
        u: tensor of size B x dimension representing tangent vectors.
        c: tensor of size 1 representing the hyperbolic curvature.
      Returns:
        Tensor of shape B x dimension.
      """
    sqrt_c = c ** 0.5
    max_num = tf.math.reduce_max(u)
    u_norm = norm(u)
    gamma_1 = tf.math.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1

def logmap0( p, c):
    """
    Hyperbolic logarithmic map at zero in the Poincare ball model.
    Args:
      p: tensor of size B x dimension representing hyperbolic points.
      c: tensor of size 1 representing the hyperbolic curvature.
    Returns:
      Tensor of shape B x dimension.
    """
    sqrt_c = c ** 0.5
    p_norm = norm(p) 
    scale = 1.0 / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
    return scale * p

def proj(x, c):
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

    x_norm = norm(x)
    maxnorm = (1 - EPS[x.dtype]) / (c ** 0.5)  # tf.math.reduce_max(x)
    cond = x_norm > maxnorm
    projected = x / x_norm * maxnorm
    return tf.where(cond, projected, x)

def mobius_add(x, y, c):
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
    return proj(num / tf.maximum(denom, MIN_NORM), c)

