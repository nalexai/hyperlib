import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, state_ops, math_ops, array_ops
from ...utils import math as hmath

@keras_export("keras.optimizers.RAdam")
class RAdam(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=1e-3, 
                beta_1=0.9, 
                beta_2=0.999,
                epsilon=1e-7,
                curvature=1.0,
                amsgrad=False,
                retract=True,
                name = 'RAdam',
                **kwargs):
        super(RAdam,self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get('lr', learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.retract = retract 
        self.c = curvature
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        for var in var_list:
            self.add_slot(var, 'tau')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'v_hat')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations+1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        tau = self.get_slot(var, 'tau')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        if self.lr_multiplier is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        rgrad = self.rgrad(grad, var)

        # m_t = beta1 * tau_t + (1 - beta1) * rgrad
        m_t = state_ops.assign(m, 
                beta_1_t * m  + (1.0 - beta_1_t) * rgrad,
                use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * |rgrad|^2
        rgrad2 = self.rgrad(math_ops.square(grad), var) # Poincare norm of rgrad is 1/lambda_x^2  * \|grad\|^2
        v_t = state_ops.assign(v, 
                beta_2_t * v + (1.0 - beta_2_t) * rgrad2,
                use_locking=self._use_locking)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                        math_ops.maximum(vhat, v_t),
                        use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self.retract:
            var_t = hmath.proj(math_ops.sub(var, lr_t * var_delta), self.c)
        else:
            var_t = hmath.expmap(-lr_t * var_delta, var, self.c)
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # tau_t = parallel transport of m_t to var_t 
        tau_t = state_ops.assign(tau,
                    hmath.parallel_transport(var, var_t, m_t, self.c), 
                    use_locking=self._use_locking)
        updates = [var_update, m_t, v_t, tau_t] 
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        # sparse update uses retraction
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations+1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        tau = self.get_slot(var, 'tau')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        rgrad = self.rgrad(tf.gather(var, indices), grad)
        rgrad2 = self.rgrad(math_ops.square(grad), 
                            tf.gather(var,indices))
        m_scaled_g_values = rgrad * (1 -  beta_1_t)
        m_t = state_ops.assign(m, m * tau, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_t = state_ops.assign( v, v * beta_2_t, use_locking=self._use_locking)
        v_scaled_g_values = rgrad2 * (1 - beta_2_t) 
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot( var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                    math_ops.maximum(vhat, v_t),
                                    use_lock = self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t) 

        if self.retract:
            var_t = hmath.proj(math_ops.sub(var, lr_t * var_delta), self.c)
        else:
            var_t = hmath.expmap(-lr_t * var_delta, var, self.c)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        tau_t = state_ops.assign(tau,
                    hmath.parallel_transport(var, var_t, m_t, self.c), 
                    use_locking=self._use_locking)
        updates = [var_update, m_t, v_t, tau_t] 
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def rgrad(self, var, grads):
        """Transforms the gradients to hyperbolic space"""
        scalars = tf.math.divide(1.0, hmath.lambda_x(var, self.c))
        scalars = tf.broadcast_to(math_ops.square(scalars), tf.shape(grads))
        return tf.math.multiply(scalars,grads)

    def get_config(self):
        config = super(RAdam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'decay': self._initial_decay,
            'amsgrad': self.amsgrad,
            'retract': self.retract,
            'curvature': self.c
        })
        return config

