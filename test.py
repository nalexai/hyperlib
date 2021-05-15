import tensorflow as tf
from manifold import poincare
from utils import math
from nn.layers import lin_hyp
from nn.optimizers import rsgd

def math_functions(x=tf.constant([1.0,2.0,3.0])):
    return math.cosh(x)

def poincare_functions():
    manifold = poincare.PoincareTF()
    return manifold

def create_layer():
    hyp_layer = lin_hyp.LinearHyperbolic()
    return hyp_layer

def create_optimizer():
    opt = rsgd.RSGD()
    return opt
