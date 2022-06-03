from random import choice
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from hyperlib.manifold.lorentz import Lorentz
from hyperlib.manifold.poincare import Poincare
from hyperlib.models.pehr import HierarchicalEmbeddings


def load_wordnet_data(file, negatives=20):
    noun_closure = pd.read_csv(file)
    noun_closure_np = noun_closure[["id1","id2"]].values

    edges = set()
    for i, j in noun_closure_np:
        edges.add((i,j))

    unique_nouns = list(set(
        noun_closure["id1"].tolist()+noun_closure["id2"].tolist()
    ))

    noun_closure["neg_pairs"] = noun_closure["id1"].apply(get_neg_pairs, args=(edges, unique_nouns, 20,))
    return noun_closure, unique_nouns

def get_neg_pairs(noun, edges, unique_nouns, negatives=20):
    neg_list = []
    while len(neg_list) < negatives:
        neg_noun = choice(unique_nouns)
        if neg_noun != noun \
        and neg_noun not in neg_list \
        and (noun, neg_noun) not in edges:
            neg_list.append(neg_noun)
    return neg_list


# Make training dataset
noun_closure, unique_nouns = load_wordnet_data("data/mammal_closure.csv")
noun_closure_dataset = noun_closure[["id1","id2"]].values

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices(
        (noun_closure_dataset, noun_closure["neg_pairs"].tolist()))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Create model
model = HierarchicalEmbeddings(vocab=unique_nouns, embedding_dim=5)
sgd = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

# Run custom training loop
model.fit(train_dataset, sgd, epochs=5)
print(model.get_embeddings())
