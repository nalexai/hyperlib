import pytest
import tensorflow as tf
from hyperlib.manifold.lorentz import Lorentz

def is_in_tangent(v, x, tol=1e-4):
    dot = Lorentz().minkowski_dot(v, x)
    return tf.reduce_max(dot) < tol

def test_proj_on_manifold():
    tol = 1e-4
    x = tf.constant([
        [0.5,1.0,-2.0,10.0],
        [1.0,-2.0,3.0,4.0],
        [2.4,-1.5,0.3,0.2]])

    M = Lorentz()
    K = 1.0
    y = M.proj(x, c=K)
    dot = M.minkowski_dot(y, y)
    targ = -tf.ones_like(dot)
    assert tf.reduce_max(tf.abs(dot - targ))

def test_proj_tan_orthogonal():
    x = tf.constant([[1.0,-3.0,0.2,-0.3],[0.0,0.3,-30.,1.]])
    M = Lorentz()
    K = 1.0
    y = M.proj(x, c=K)

    v = tf.constant([[4.0,4.0,-1.0,0.0],[20.,20.,0.0,-1.]])
    p = M.proj_tan(v, y, K)

    assert is_in_tangent(p, y)

def test_exp_log_inv():
    tol = 1e-4
    c = 1.0
    x = tf.constant([[4.690416,1.,-2.,4.],[5.477226,-2.,3.,4.]])
    y = tf.constant([[1.8384776,-1.5,0.3,0.2],[3.5,3.,-1.5,0.]])

    M = Lorentz()
    v = M.logmap(y, x, c)
    assert is_in_tangent(v, x)

    expv = M.expmap(v, x, c)
    print(expv)
    assert tf.reduce_max(tf.abs(expv - y)) < tol 
