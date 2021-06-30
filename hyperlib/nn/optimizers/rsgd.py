import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops import control_flow_ops, state_ops, math_ops
from ...utils import math as hmath

@keras_export("keras.optimizers.RSGD")
class RSGD(optimizer_v2.OptimizerV2):
    """
    Riemannian Stochastic Gradient Descent for the Poincare Ball model of
    hyperbolic space with curvature -c, (c > 0).
    Args:
        curvature (float): absolute value of the curvature
        retract (bool): if True use linear retraction instead of exp

    Update weights as 
        x <- exp_x( -lr * rgrad)
    where rgrad is the Riemannian gradient and exp_x is the exponential map at x.

    If retract is True then exp is approximated with retraction. The update becomes 
        x <- x - lr * rgrad

    """
    #TODO: Add momentum
    _HAS_AGGREGATE_GRAD = True
    def __init__(self, learning_rate=1e-3, 
                curvature = 1.0,
                retract=False,
                name="RSGD", 
                **kwargs):
        super(RSGD, self).__init__(name, **kwargs)
        self._set_hyper(
            "learning_rate", 
            kwargs.get("lr", tf.cast(learning_rate, tf.float64)) # handle lr=learning_rate
            )
        self._lr = learning_rate
        self.retract = retract
        self.c = curvature  

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = math_ops.cast(self._lr,var_dtype) 
        r_grad = self.rgrad(var, grad)
        r_grad = tf.math.multiply(r_grad, -lr_t)
        if self.retract:
            var_t = hmath.proj(tf.math.add(var, r_grad), self.c)
        else:
            var_t = hmath.expmap(r_grad, var, self.c)
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        return control_flow_ops.group(var_update)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = math_ops.cast(self._lr, var_dtype) 
        rgrad = self.rgrad(tf.gather(var,indices), grad)
        var_t = self._resource_scatter_add(var,indices, -lr_t * rgrad)
        
        # use retraction (exp needs the whole gradient)
        var_update = state_ops.assign(var, 
                            hmath.proj(var_t, self.c),
                            use_locking=self._use_locking)
        return control_flow_ops.group(var_update) 

    def rgrad(self, var, grads):
        """Transforms the gradients to hyperbolic space"""
        scalars = tf.math.divide(1.0, hmath.lambda_x(var, self.c))
        scalars = tf.broadcast_to(tf.math.square(scalars), tf.shape(grads))
        return tf.math.multiply(scalars,grads)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "curvature": self.c,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "retract": self.retract 
        }
