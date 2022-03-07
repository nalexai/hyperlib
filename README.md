# HyperLib: Deep learning in the Hyperbolic space

[![PyPI version](https://badge.fury.io/py/hyperlib.svg)](https://badge.fury.io/py/hyperlib)

## Background
This library implements common Neural Network components in the hyperbolic space (using the Poincare model). The implementation of this library uses Tensorflow as a backend and can easily be used with Keras and is meant to help Data Scientists, Machine Learning Engineers, Researchers and others to implement hyperbolic neural networks.

You can also use this library for uses other than neural networks by using the mathematical functions available in the Poincare class. In the future we may implement components that can be used in models other than neural networks. You can learn more about Hyperbolic networks [here](https://www.nalex.ai/post/hyperlib-deep-learning-in-the-hyperbolic-space), and in the references[^1] [^2] [^3] [^4].

## Install
The recommended way to install is with pip
```
pip install hyperlib
```

To build from source, you need to compile the pybind11 extensions.   
For example to build on linux:
```shell
conda -n hyperlib python=3.8 gxx_linux-64 pybind11
python setup.py install
```

Hyperlib works with python>=3.8 and tensorflow>=2.0.

## Example Usage

Creating a hyperbolic neural network using Keras:
```python
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
```python
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

### Embeddings 
A big advantage of hyperbolic space is its ability to represent hierarchical data. There are several techniques for embedding data in hyperbolic space; the most common is gradient methods [^6].

If your data has a natural metric you can also use TreeRep[^5].
Input a symmetric distance matrix, or a compressed distance matrix 
```python
import numpy as np
from hyperlib.embedding.treerep import treerep
from hyperlib.embedding.sarkar import sarkar_embedding 

# Example: immunological distances between 8 mammals by Sarich
compressed_metric = np.array([ 
        32.,  48.,  51.,  50.,  48.,  98., 148.,  
        26.,  34.,  29.,  33., 84., 136.,  
        42.,  44.,  44.,  92., 152.,  
        44.,  38.,  86., 142.,
        42.,  89., 142.,  
        90., 142., 
        148.
    ])

# outputs a weighted networkx Graph
tree = treerep(compressed_metric, return_networkx=True)

# embed the tree in 2D hyperbolic space
root = 0
embedding = sarkar_embedding(tree, root, tau=0.5)
```

Please see the [examples directory](https://github.com/nalexai/hyperlib/tree/main/examples) for complete examples.

## References
[^1]: [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)

[^2]: [Nickel, M. and Kiela, D. Poincaré embeddings for learning hierarchical representations. NIPS 2017.](https://papers.nips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html)

[^3]: [Khrulkov, Mirvakhabova, Ustinova, Oseledets, Lempitsky. Hyperbolic Image Embeddings.](https://arxiv.org/pdf/1904.02239.pdf)

[^4]: [Wei Peng, Varanka, Mostafa, Shi, Zhao. Hyperbolic Deep Neural Networks: A Survey.](https://arxiv.org/pdf/2101.04562.pdf)

[^5]: [Rishi Sonthalia and Anna Gilbert. Tree! I am no Tree! I am a Low Dimensional Hyperbolic Embedding](https://arxiv.org/abs/2005.03847)  
[^6]: [De Sa et. al. Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/abs/1804.03329)
