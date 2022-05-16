import pytest
import tensorflow as tf
from hyperlib.manifold.lorentz import Lorentz

def is_in_tangent(v, x, tol=1e-5):
    dot = Lorentz().minkowski_dot(v, x)
    return tf.reduce_max(dot) < tol

@pytest.mark.parametrize("c", [0.5, 1.0, 1.5, 2.0, 4.0])
def test_proj_on_manifold(c):
    tol = 1e-5
    x = tf.constant([
        [0.5,1.0,-2.0,10.0],
        [1.0,-2.0,3.0,4.0],
        [2.4,-1.5,0.3,0.2]])

    M = Lorentz()
    y = M.proj(x, c)
    dot = M.minkowski_dot(y, y)
    targ = -tf.ones_like(dot)
    assert tf.reduce_max(tf.abs(dot - targ))

@pytest.mark.parametrize("c", [0.5, 1.0, 1.5, 2.0, 4.0])
def test_proj_tan_orthogonal(c):
    x = tf.constant([[1.0,-3.0,0.2,-0.3],[0.0,0.3,-30.,1.]])
    M = Lorentz()
    y = M.proj(x, c)

    v = tf.constant([[4.0,4.0,-1.0,0.0],[20.,20.,0.0,-1.]])
    p = M.proj_tan(v, y, c)

    assert is_in_tangent(p, y)

@pytest.mark.parametrize("c", [0.5, 1.0, 1.5, 2.0, 4.0])
def test_proj_tan_orthogonal_0(c):
    K = 1./c
    zero = tf.constant([[K**0.5,0.,0.,0.],[K**0.5,0.,0.,0.]])
    v = tf.constant([[4.0,2.0,-1.0,-0.4],[20.,20.,0.0,-1.]])
    v = Lorentz().proj_tan0(v, c)
    assert is_in_tangent(v, zero)

def test_exp_log_inv():
    tol = 1e-4
    c = 1.0
    x = tf.constant([[4.690416,1.,-2.,4.],[5.477226,-2.,3.,4.]])
    y = tf.constant([[1.8384776,-1.5,0.3,0.2],[3.5,3.,-1.5,0.]])

    M = Lorentz()
    v = M.logmap(y, x, c)
    assert is_in_tangent(v, x)

    expv = M.expmap(v, x, c)
    assert tf.reduce_max(tf.abs(expv - y)) < tol 

def test_exp_log_inv_0():
    tol = 1e-4
    c = 2.0

    x = tf.constant([[1.6970563,-1.5,0.3,0.2],[3.4278274,3.,-1.5,0.]])
    M = Lorentz()
    v = M.logmap0(x, c)
    expv = M.expmap0(v, c)
    assert tf.reduce_max(tf.abs(expv - x)) < tol

def test_parallel_transport():
    tol = 1e-4
    c = 1.0
    x = tf.constant([[4.690416,1.,-2.,4.],[5.477226,-2.,3.,4.]])
    y = tf.constant([[1.8384776,-1.5,0.3,0.2],[3.5,3.,-1.5,0.]])

    M = Lorentz()
    v = M.proj_tan(
        tf.constant([[-3.0,-2.0,1.0,0.0],[0.5,-3.4,4.0,1.0]]),
        x, c)
    assert is_in_tangent(v, x, tol=tol)

    tr_v = M.ptransp(x, y, v, c) 
    assert is_in_tangent(tr_v, y, tol=tol)

def test_parallel_transport_0():
    tol = 1e-4
    c = 1.0
    zero = tf.constant([[1.,0.,0,0.],[1.,0.,0.,0.]])
    x = tf.constant([[4.690416,1.,-2.,4.],[5.477226,-2.,3.,4.]])

    M = Lorentz()
    v = M.proj_tan0(tf.constant([[-3.0,-2.0,1.0,0.0],[0.5,-3.4,4.0,1.0]]), c)
    M = Lorentz()
    assert is_in_tangent(v, zero, tol=tol)

    tr_v = M.ptransp0(x, v, c)
    assert is_in_tangent(tr_v, x, tol=tol)
