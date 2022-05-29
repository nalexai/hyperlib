import tensorflow as tf
from tensorflow import keras


class HierarchicalEmbeddings(tf.keras.Model):
    """
    Hierarchical Embeddings model from Poincaré Embeddings for
    Learning Hierarchical Representations by Nickel and Keila

    Please find an example of how to use this model in hyperlib/examples/wordnet_embedding.py
    """

  def __init__(self, embedding_length, embedding_dim):
    super().__init__()

    initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=None)

    self.string_lookup = keras.layers.StringLookup(vocabulary=embedding_length, name="string_lookup")
    self.embedding =  keras.layers.Embedding(
        len(embedding_length)+1,
        embedding_dim,
        embeddings_initializer=initializer,
        name="embeddings",
    )

  def call(self, inputs):
    indices = self.string_lookup(inputs)
    return self.embedding(indices)
