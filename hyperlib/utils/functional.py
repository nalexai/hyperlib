import tensorflow as tf

"""
Tensorflow Math functions with clipping as required for hyperbolic functions.
"""

def cosh(x, clamp=15):
    return tf.math.cosh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

def sinh(x, clamp=15):
    return tf.math.sinh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

def tanh(x, clamp=15):
    return tf.math.tanh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

def arcosh(x):
    x = tf.clip_by_value(x, clip_value_min=1+1e-15, clip_value_max=tf.reduce_max(x))
    return tf.math.acosh(x)

def asinh(x):
    return tf.math.asinh(x)

def atanh(x):
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
