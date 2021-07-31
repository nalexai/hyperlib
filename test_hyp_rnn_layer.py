from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import RNN
from hyperlib.nn.layers import lin_hyp, rnn
import tensorflow.keras.layers
from tensorflow.keras.datasets import imdb
from hyperlib.manifold import poincare
from hyperlib.nn.optimizers import rsgd
"""
This code was adapted from:
https://slundberg.github.io/shap/notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.html

A euclidean rnn achieved the following metrics:
Test score: 1.0867270610725879
Test accuracy: 0.80912

"""
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
y_train = tf.cast(y_train, tf.float64)
y_test = tf.cast(y_test, tf.float64)
x_train = tf.cast(sequence.pad_sequences(x_train, maxlen=maxlen), tf.float64)
x_test = tf.cast(sequence.pad_sequences(x_test, maxlen=maxlen), tf.float64)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print("\n\n y test")
print(y_test)

print('Build model...')
hyp_rnn_cell = rnn.MinimalHYPRNNCell(units=128, activation='relu', c=1, manifold=poincare.Poincare())
rsgd = rsgd.RSGD(learning_rate=1)


# init_state = hyp_rnn_cell.zero_state(128, tf.float64)
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(RNN(hyp_rnn_cell), )
model.add(lin_hyp.LinearHyperbolic(1, poincare.Poincare(), 1, activation='sigmoid', use_bias=True))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=35,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)