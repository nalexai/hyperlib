import tensorflow as tf
from tensorflow import keras

class RSGD(keras.optimizers.Optimizer):

    """
    Implmentation of a Riemannian Stochastic Gradient Descent. This class inherits form the keras Optimizer class.
    """

    def __init__(self, learning_rate=0.01, name="RSGOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper(
            "learning_rate", kwargs.get("lr", tf.cast(learning_rate, tf.float64))
        )  # handle lr=learning_rate
        self._is_first = True

    def _create_slots(self, var_list):
        """
        For each model variable, create the optimizer variable associated with it
        """
        for var in var_list:
            self.add_slot(var, "pv")  # previous variable i.e. weight or bias
        for var in var_list:
            self.add_slot(var, "pg")  # previous gradient

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """
        Update the slots and perform one optimization step for one model variable
        """
        r_grad = self.rgrad(var, grad)
        r_grad = tf.math.multiply(r_grad, -self.lr)
        new_var_m = self.expm(var, r_grad)

        # slots aren't currently used - they store previous weights and gradients
        pv_var = self.get_slot(var, "pv")
        pg_var = self.get_slot(var, "pg")
        pv_var.assign(var)
        pg_var.assign(grad)
        # assign new weights

        var.assign(new_var_m)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Not implemented")

    def rgrad(self, var, grads):
        """
        Transforms the gradients to Riemannian space
        """
        vars_sqnorm = tf.math.reduce_sum(var ** 2, axis=-1, keepdims=True)
        grads = grads * tf.broadcast_to(((1 - vars_sqnorm) ** 2 / 4), tf.shape(grads))
        return grads

    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        """
        Maps the variable values
        """
        if lr is not None:
            d_p = tf.math.multiply(d_p, -lr)
        if out is None:
            out = p
        p = tf.math.add(p, d_p)
        if normalize:
            self.normalize(p)
        return p

    def normalize(self, u):
        d = u.shape[-1]
        if self.max_norm:
            u = tf.reshape(u, [-1, d])
            u.renorm_(2, 0, self.max_norm)
        return u

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }
