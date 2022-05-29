from random import choice
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from hyperlib.manifold.lorentz import Lorentz
from hyperlib.manifold.poincare import Poincare


def load_wordnet_data(file, negative_samples=10):
    '''Load wordnet nouns transitive closure dataset 
    and compute negative samples'''
    noun_closure = pd.read_csv(file)
    noun_closure_np = noun_closure[["id1","id2"]].values

    # edges represent positive samples
    edges = set()
    for i, j in noun_closure_np:
        edges.add((i,j))

    unique_nouns = list(set(
        noun_closure["id1"].tolist()+noun_closure["id2"].tolist()
    ))

    # for each noun find negative samples
    neg_samples = {}
    for noun in unique_nouns:
        neg_list = []
        while len(neg_list) < negative_samples:
            neg_noun = choice(unique_nouns)
            if neg_noun != noun \
            and neg_noun not in neg_list \
            and (noun, neg_noun) not in edges:
                neg_list.append(neg_noun)
        neg_samples[noun] = neg_list

    noun_closure["neg_pairs"] = noun_closure["id1"].apply(lambda x: neg_samples[x])
    return noun_closure, unique_nouns

def contrastive_loss(pos_embs, neg_embs, M, c=1.0, clip_value=0.9):
    '''
    The contrastive loss for embeddings used by Nickel & Kiela
    math::
        -\log( e^{-d(x,y)} / \sum_{n \in N(x)} e^{-d(x,n)})

    where (x,y) is a positive example,
    N(x) is the set of negative samples for x,
    and d(.,.) is the hyperbolic distance
    '''
    # clip embedding norms before expmap
    pos_embs = tf.clip_by_norm(pos_embs, clip_value, axes=2)
    neg_embs = tf.clip_by_norm(neg_embs, clip_value, axes=2)
    
    x_pos = M.expmap0(pos_embs, c)
    x_neg = M.expmap0(neg_embs, c)
    
    batch_loss = M.dist(x_pos[:,0,:], x_pos[:,1,:], c)
    
    x = x_pos[:,0,:]
    x = tf.expand_dims(x, 1)
    x = tf.broadcast_to(x, x_neg.shape)

    neg_loss = tf.reduce_sum(tf.exp(-M.dist(x, x_neg, c)), axis=1)
    # clip to avoid log(0)
    neg_loss = tf.clip_by_value(neg_loss, clip_value_min=1e-15, clip_value_max=1e10)
    batch_loss += tf.math.log(neg_loss)
    return tf.reduce_sum(batch_loss, axis=0)


def train(model, train_data, **kwargs):
    epochs = kwargs.get("epochs", 10)
    c = kwargs.get("c", 1.0) 
    lr = kwargs.get("lr", 1e-2)
    clip_value = kwargs.get("clip_value", 0.9)
    momentum = kwargs.get("momentum", 0)
    M = Poincare()

    sgd = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

    for epoch in range(epochs):
        print("Epoch %d" % (epoch,))
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pos_embs = model(x_batch_train, training=True)
                neg_embs = model(y_batch_train)
                loss_value = contrastive_loss(
                        pos_embs, neg_embs, M, c=c, clip_value=clip_value)

            grads = tape.gradient(loss_value, model.trainable_weights)
            sgd.apply_gradients(zip(grads, model.trainable_weights))

            if step % 100 == 0:
                print("Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value)))


# Make training dataset
noun_closure, unique_nouns = load_wordnet_data("data/mammal_closure.csv")
noun_closure_dataset = noun_closure[["id1","id2"]].values

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices(
        (noun_closure_dataset, noun_closure["neg_pairs"].tolist()))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


# define embedding layer
embedding_dim = 5
initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=None)
string_lookup_layer = keras.layers.StringLookup(vocabulary=unique_nouns)

emb_layer = keras.layers.Embedding(
    len(unique_nouns)+1,
    embedding_dim,
    input_length=2,
    embeddings_initializer=initializer,
    name="embeddings",
)

model = keras.Sequential([string_lookup_layer, emb_layer])

train(
    model, 
    train_dataset, 
    c=1.0, 
    epochs=10,
    clip_value=0.9,
    lr=1e-2,
    momentum=0.9,
)
