import pytest
import tensorflow as tf
from hyperlib.manifold import Lorentz

def test_proj_on_manifold():
    tol = 1e-4
    x = tf.constant([
        [0.5,1.0,-2.0,10.0],
        [1.0,-2.0,3.0,4.0],
        [2.4,-1.5,0.3,0.2]])

    M = Lorentz()
    K = 1.0
    y = M.proj(x, c=K)
    dot = M.minkowski_dot(y)
    targ = -tf.ones_like(dot)
    assert tf.reduce_max(tf.abs(dot - targ))


def test_proj_tan():
    tol = 1e-4
    x = tf.constant([[1.0,-3.0,0.2,-0.3],[0.0,0.3,-30.,1.]])
    M = Lorentz()
    K = 1.0
    y = M.proj(x, c=K)

    v = tf.constant([[4.0,4.0,-1.0,0.0],[20.,20.,0.0,-1.]])
    p = M.proj_tan(v, x, K)

    dot = M.minkowski_dot(p, x)
    assert tf.reduce_max(dot) < tol
