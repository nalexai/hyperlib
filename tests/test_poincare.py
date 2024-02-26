import tensorflow as tf
from hyperlib.manifold import poincare
from hyperlib.utils import functional as F
from hyperlib.nn.layers import linear, dense_attention
from hyperlib.nn.optimizers import rsgd
import pytest

class TestClass:
    @classmethod
    def setup_class(self):
        self.test_tensor_shape_2_2_a = tf.constant(
            [[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64
        )
        self.test_tensor_shape_2_2_b = tf.constant(
            [[0.5, 0.5], [0.5, 0.5]], dtype=tf.float64
        )
        self.test_tensor_shape_2_1a = tf.constant([[1.0], [2.0]], dtype=tf.float64)
        self.test_tensor_shape_2_1b = tf.constant([[0.5], [0.5]], dtype=tf.float64)
        self.poincare_manifold = poincare.Poincare()
        self.curvature_tensor = tf.Variable([1], dtype="float64")

    def test_math_functions(self, x=tf.constant([1.0, 2.0, 3.0])):
        return F.cosh(x)

    def test_mobius_matvec_(self):
        result = self.poincare_manifold.mobius_matvec(
            self.test_tensor_shape_2_2_a,
            self.test_tensor_shape_2_2_b, c=self.curvature_tensor
        )
        with pytest.raises(tf.errors.InvalidArgumentError):
            self.poincare_manifold.mobius_matvec(
                self.test_tensor_shape_2_2_a,
                self.test_tensor_shape_2_1a, c=self.curvature_tensor
            )

    def test_expmap0(self):
        c = tf.Variable([1], dtype="float64")
        result = self.poincare_manifold.expmap0(
            self.test_tensor_shape_2_2_a, c=self.curvature_tensor
        )

    @pytest.mark.skip(reason="working on a test for this")
    def test_logmap0(self):
        result = self.poincare_manifold.logmap0(
            self.test_tensor_shape_2_2_b, c=self.curvature_tensor
        )

    def test_proj(self):
        result = self.poincare_manifold.proj(self.test_tensor_shape_2_2_a, c=self.curvature_tensor)

    def test_poincare_functions(self):
        manifold = poincare.Poincare()
        assert manifold.name == "PoincareBall"
        assert manifold.min_norm == 1e-15

    def test_create_layer(self, units=32):
        hyp_layer = linear.LinearHyperbolic(
            units, self.poincare_manifold, 1.0 
        )
        assert hyp_layer.units == units
        assert hyp_layer.manifold == self.poincare_manifold

    def test_attention_layer(self):
        sample_hidden = tf.Variable(tf.random.uniform([10, 1], 0, 100, dtype=tf.float64, seed=0))
        sample_output = tf.Variable(tf.random.uniform([10, 1], 0, 100, dtype=tf.float64, seed=0))

        attention_layer = dense_attention.HypLuongAttention(manifold=self.poincare_manifold, c=1, use_scale=False,
                                                            hyperbolic_input=False)
        query_value_attention_seq = attention_layer([sample_hidden, sample_output])

        assert query_value_attention_seq.shape == sample_output.shape
        assert attention_layer.manifold == self.poincare_manifold

    def test_layer_training(self, units=32):
        x_input = tf.zeros([units, 1])
        hyp_layer = linear.LinearHyperbolic(
            units, self.poincare_manifold, 1.0 
        )
        output = hyp_layer(x_input)

    def test_layer_training_with_bias(self, units=32):
        x_input = tf.zeros([units, 1])
        hyp_layer = linear.LinearHyperbolic(
            units, self.poincare_manifold, 1.0, use_bias=True
        )
        output = hyp_layer(x_input)

    def test_create_optimizer(self):
        opt = rsgd.RSGD()
