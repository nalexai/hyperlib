import tensorflow as tf
from tensorflow import keras

class HypLuongAttention(keras.layers.Attention):
  """Dot-product attention layer, a.k.a. Luong-style attention.
  Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
  shape `[batch_size, Tv, dim]` and `key` tensor of shape

  Call Args:
    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
  Output:
    Attention outputs of shape `[batch_size, Tq, dim]`.
  # ...
  ```
  """

  def __init__(self, manifold, c=1, use_scale=False, hyperbolic_input=False, **kwargs):
    super(HypLuongAttention, self).__init__(**kwargs)
    self.use_scale = use_scale
    self.hyperbolc_input = hyperbolic_input
    self.manifold = manifold
    self.c = tf.Variable([c], dtype="float64")

  def build(self, input_shape):
    """Creates scale variable if use_scale==True."""
    if self.use_scale:
      self.scale = self.add_weight(
          name='scale',
          shape=(),
          initializer='ones',
          dtype=self.dtype,
          trainable=True)
    else:
      self.scale = None
    super(HypLuongAttention, self).build(input_shape)

  def _calculate_scores(self, query, key):
    """Calculates attention scores as a query-key dot product.
    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    if self.hyperbolc_input:
        scores = self.manifold.single_query_attn_scores(query, key, self.c)
    else:
        scores = tf.linalg.matmul(query, key, transpose_b=True)

        if self.scale is not None:
          scores *= self.scale

    return scores

  def get_config(self):
    config = {'use_scale': self.use_scale}
    base_config = super(keras.layers.Attention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
