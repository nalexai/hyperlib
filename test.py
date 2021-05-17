import tensorflow as tf
from .manifold import poincare
from .utils import math
from .nn.layers import lin_hyp
from .nn.optimizers import rsgd

def math_functions(x=tf.constant([1.0,2.0,3.0])):
    return math.cosh(x)

def poincare_functions():
    manifold = poincare.PoincareTF()
    return manifold

def create_layer(units=32, manifold=poincare.PoincareTF(), c=1):
    hyp_layer = lin_hyp.LinearHyperbolic(units, manifold, c)
    return hyp_layer

def create_optimizer():
    opt = rsgd.RSGD()
    return opt

if __name__ == "__main__":
    math_functions()
    poincare_functions()
    create_layer()
    create_optimizer()
