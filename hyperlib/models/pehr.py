import logging

import tensorflow as tf
from tensorflow import keras

from hyperlib.manifold.lorentz import Lorentz
from hyperlib.manifold.poincare import Poincare
from hyperlib.loss.constrastive_loss import contrastive_loss


log = logging.getLogger(__name__)


class HierarchicalEmbeddings(tf.keras.Model):
    """
    Hierarchical Embeddings model from Poincaré Embeddings for
    Learning Hierarchical Representations by Nickel and Keila

    Please find an example of how to use this model in hyperlib/examples/wordnet_embedding.py
    """

    def __init__(self, vocab, embedding_dim=2, manifold=Poincare, c=1.0, clip_value=0.9):
        super().__init__()

        initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=None)
        self.string_lookup = keras.layers.StringLookup(vocabulary=vocab, name="string_lookup")
        self.embedding =  keras.layers.Embedding(
            len(vocab)+1,
            embedding_dim,
            embeddings_initializer=initializer,
            name="embeddings",
        )
        self.vocab = vocab
        self.manifold = manifold()
        self.c = c
        self.clip_value = clip_value

    def call(self, inputs):
        indices = self.string_lookup(inputs)
        return self.embedding(indices)

    def get_embeddings(self):
        embeddings = self.embedding (tf.constant([i for i in range(len(self.vocab))]))
        embeddings_copy = tf.identity(embeddings)
        embeddings_hyperbolic = self.manifold.expmap0(embeddings_copy, c=self.c)
        return embeddings_hyperbolic

    def get_vocabulary(self):
        return self.vocab

    @staticmethod
    def get_model(vocab, embedding_dim=2):
        embedding_dim=2
        initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=None)
        string_lookup_layer = keras.layers.StringLookup(vocabulary=vocab)

        emb_layer = keras.layers.Embedding(
            len(vocab)+1,
            embedding_dim,
            embeddings_initializer=initializer,
            name="embeddings",
        )

        model = keras.Sequential([string_lookup_layer, emb_layer])
        return model

    def fit(self, train_dataset, optimizer, epochs=100):

        for epoch in range(epochs):
            log.info("Epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    pos_embs = self.embedding(self.string_lookup(x_batch_train))
                    neg_embs = self.embedding(self.string_lookup(y_batch_train))
                    loss_value = contrastive_loss(
                            pos_embs, neg_embs, self.manifold, c=self.c, clip_value=self.clip_value)

                grads = tape.gradient(loss_value, self.embedding.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.embedding.trainable_weights))

                if step % 100 == 0:
                    log.info("Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value)))
