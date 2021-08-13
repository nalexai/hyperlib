from tensorflow import keras
import tensorflow as tf
from hyperlib.nn.layers import lin_hyp
from hyperlib.nn.optimizers import rsgd
from hyperlib.manifold import poincare
import numpy as np
import tensorflow_datasets as tfds
import datetime

def get_mnist_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_test, ds_info
    
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float64) / 255., label
  
## get mnist daa
ds_train, ds_test, ds_info = get_mnist_data()
# normalise mnist training images
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
# normalise mnist test images
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #LinearEuclidean(1128, activation="relu"),
  #tf.keras.layers.Dense(10)
  lin_hyp.LinearHyperbolic(50, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(100, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(100, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(100, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(100, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(100, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(100, poincare.Poincare(), 1, activation='relu', use_bias=True),
  lin_hyp.LinearHyperbolic(10, poincare.Poincare(), 1)
])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
sgd = keras.optimizers.SGD(learning_rate=0.1)
rsgd = rsgd.RSGD(learning_rate=1)

model.compile(
    optimizer=rsgd,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_test,
)
