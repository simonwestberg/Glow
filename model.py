# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import numpy as np

### LAYERS

## https://www.tensorflow.org/guide/keras/custom_layers_and_models

## Skeleton for custom layer
class MyLayer(Layer):

    def __init__(self):
        super(MyLayer, self).__init__()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        return inputs

## ActNorm
class ActNorm(Layer):

    def __init__(self):
        super(ActNorm, self).__init__()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        return inputs

## Permutation (1x1 convolution, reverse, or shuffle)
class Permutation(Layer):

    def __init__(self):
        super(Permutation, self).__init__()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        return inputs

## Affine coupling
class AffineCoupling(Layer):

    def __init__(self):
        super(AffineCoupling, self).__init__()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        return inputs

## Squeeze, no trainable parameters?
class Squeeze(Layer):

    def __init__(self):
        super(Squeeze, self).__init__()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        return inputs

## Split, no trainable parameters?
class Split(Layer):

    def __init__(self):
        super(Split, self).__init__()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        print(inputs.shape)
        return inputs, inputs

## Step of flow
class FlowStep(Layer):

    def __init__(self):
        super(FlowStep, self).__init__()

        self.actnorm = ActNorm()
        self.perm = Permutation()
        self.coupling = AffineCoupling()

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        """
        inputs: input tensor
        returns: output tensor
        """
        x = self.actnorm(inputs)
        x = self.perm(x)
        x = self.coupling(x)

        return x
        
### MODEL

class Glow(keras.Model):

    def __init__(self, steps, levels):
        super(Glow, self).__init__()

        self.steps = steps  # Number of steps in each flow, K in the paper
        self.levels = levels    # Number of levels, L in the paper

        self.squeeze = Squeeze()
        self.split = Split()
        self.flow_layers = []

        for l in range(levels):
            flows = []

            for k in range(steps):
                flows.append([ActNorm(), Permutation(), AffineCoupling()])

            self.flow_layers.append(flows)

    def call(self, inputs):
        x = inputs
        latent_variables = []
        
        for l in range(self.levels - 1):
            
            x = self.squeeze(x)

            # K steps of flow
            for k in range(self.steps):
                # ActNorm
                x = self.flow_layers[l][k][0](x)
                # Permutation
                x = self.flow_layers[l][k][1](x)
                # Affine coupling
                x = self.flow_layers[l][k][2](x)
            
            x, z = self.split(x)

            latent_variables.append(z)
        
        # Last squeeze
        x = self.squeeze(x)
        
        # Last steps of flow
        for k in range(self.steps):
            # ActNorm
            x = self.flow_layers[-1][k][0](x)
            # Permutation
            x = self.flow_layers[-1][k][1](x)
            # Affine coupling
            x = self.flow_layers[-1][k][2](x)
        
        latent_variables.append(x)         

        return latent_variables
