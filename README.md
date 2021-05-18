# Hyperbolic Library

## Background
This library implements common Neural Network components in the hypberbolic space; this imlementation uses the Poincare model. The implementation of this library uses Tensorflow as a backend and can therefore easily be used with Keras. This library is meant to help Data Scientists, Machine Learning Engineer, Researchers and others to implement hyperbolic neural networks.

You can also use this library for uses other than neural networks by using the mathematical functions avaialbe in the Poincare class. In the future we may implement components that can be used in models other than neural networks.

## Example Usage

Creating a hyperbolic neural network using Keras:
```
from tensorflow import keras
from hyperbolic.nn.layers import lin_hyp
from hyperbolic.nn.optimizers import rsgd

# Create layers
hyperbolic_layer_1 = lin_hyp.LinearHyperbolic(32, poincare.PoincareTF(), 1)
hyperbolic_layer_2 = lin_hyp.LinearHyperbolic(32, poincare.PoincareTF(), 1)
output_layer = lin_hyp.LinearHyperbolic(10, poincare.PoincareTF(), 1)

# Create optimizer
optimizer = rsgd.RSGD(learning_rate=0.1)

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
```

Using math functions on the Poincare ball:
```
import tensorflowa s tf
from hyperbolic.manifold import poincare

p = poincare.Poincare()

# Create two matrices
a = tf.constant([[5.0,9.4,3.0],[2.0,5.2,8.9])
b = tf.constant([[4.8,1.0,2.3],[5.1,3.5,7.5])

# Matrix multiplication on the Poincare ball
curvature = 1
p.mobius_matvec(a, b, curvature)
```

## TODO:
- Implement use of a bias in hyperbolic l;ayer
- Make curvature in the hyperbolic layer a trainable variable
- Implement an Attention Mechanism
- Implement a Riemannian Adam Optimizer
- Remove casting of layer variables to tf.float64
