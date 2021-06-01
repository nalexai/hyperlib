# HyperLib: Deep learning in the Hyperbolic space

[![PyPI version](https://badge.fury.io/py/hyperlib.svg)](https://badge.fury.io/py/hyperlib)

## Background
This library implements common Neural Network components in the hypberbolic space (using the Poincare model). The implementation of this library uses Tensorflow as a backend and can easily be used with Keras and is meant to help Data Scientists, Machine Learning Engineers, Researchers and others to implement hyperbolic neural networks.

You can also use this library for uses other than neural networks by using the mathematical functions avaialbe in the Poincare class. In the future we may implement components that can be used in models other than neural networks.

## Example Usage

Install the library
```
pip install hyperlib
```

Creating a hyperbolic neural network using Keras:
```
import tensorflow as tf
from tensorflow import keras
from hyperlib.nn.layers.lin_hyp import LinearHyperbolic
from hyperlib.nn.optimizers.rsgd import RSGD
from hyperlib.manifold.poincare import Poincare

# Create layers
hyperbolic_layer_1 = LinearHyperbolic(32, Poincare(), 1)
hyperbolic_layer_2 = LinearHyperbolic(32, Poincare(), 1)
output_layer = LinearHyperbolic(10, Poincare(), 1)

# Create optimizer
optimizer = RSGD(learning_rate=0.1)

# Create model architecture
model = tf.keras.models.Sequential([
  hyperbolic_layer_1,
  hyperbolic_layer_2,
  output_layer
])

# Compile the model with the Riemannian optimizer            
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
```

Using math functions on the Poincare ball:
```
import tensorflow as tf
from hyperlib.manifold.poincare import Poincare

p = Poincare()

# Create two matrices
a = tf.constant([[5.0,9.4,3.0],[2.0,5.2,8.9],[4.0,7.2,8.9]])
b = tf.constant([[4.8,1.0,2.3]])

# Matrix multiplication on the Poincare ball
curvature = 1
p.mobius_matvec(a, b, curvature)
```

## TODO:
- Implement an Attention Mechanism
- Implement a Riemannian Adam Optimizer
- Remove casting of layer variables to tf.float64

## References
[1] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)

[2] [Nickel, M. and Kiela, D. Poincaré embeddings for learning hierarchical representations. NIPS 2017.](https://papers.nips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html)

[3] [Khrulkov, Mirvakhabova, Ustinova, Oseledets, Lempitsky. Hyperbolic Image Embeddings.](https://arxiv.org/pdf/1904.02239.pdf)

[4] [Wei Peng, Varanka, Mostafa, Shi, Zhao. Hyperbolic Deep Neural Networks: A Survey.](https://arxiv.org/pdf/2101.04562.pdf)
