# TensorFlow
import tensorflow as tf
import tensorflow_probability as tf_prob
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

    # Create the state of the layer. ActNorm initialization is done in call()
    def build(self, input_shape):
        b, h, w, c = input_shape    # Batch size, height, width, channels

        # Scale parameters per channel, called 's' in Glow
        self.scale = self.add_weight(
            shape=(1, 1, 1, c),
            trainable=True
        )

        # Bias parameter per channel, called 'b' in Glow
        self.bias = self.add_weight(
            shape=(1, 1, 1, c),
            trainable=True
        )

        # Used to check if scale and bias have been initialized
        self.initialized = self.add_weight(
            trainable=False,
            dtype=tf.bool
        )

        self.initialized.assign(False)

    def call(self, inputs, forward=True):
        """
        inputs: input tensor
        returns: output tensor
        """
        b, h, w, c = inputs.shape    # Batch size, height, width, channels

        if not self.initialized:
            """
            Given an initial batch X=inputs, setting 
            scale = 1 / mean(X) and
            bias = -mean(X) / std(X)
            where mean and std is calculated per channel in X,
            results in post-actnorm activations X = X*s + b having zero mean 
            and unit variance per channel. 
            """

            assert(len(inputs.shape) == 4)

            # Calculate mean per channel
            mean = tf.math.reduce_mean(inputs, axis=[0, 1, 2], keepdims=True)

            # Calculate standard deviation per channel
            std = tf.math.reduce_std(inputs, axis=[0, 1, 2], keepdims=True)

            # Add small value to std to avoid division by zero
            eps = tf.constant(1e-6, shape=std.shape, dtype=std.dtype)
            std = tf.math.add(std, eps)

            self.scale.assign(tf.math.divide(1.0, std))
            self.bias.assign(-tf.math.divide(mean, std))

            self.initialized.assign(True)

        if forward:
            
            outputs = tf.math.multiply(inputs, self.scale) + self.bias

            # log-determinant of ActNorm layer
            log_s = tf.math.log(tf.math.abs(self.scale))
            logdet = h * w * tf.math.reduce_sum(log_s)

            # Loss for this layer is negative log-determinant
            self.add_loss(-logdet)

        else:
            # Reverse operation
            outputs = (inputs - self.bias) / self.scale 
        
        return outputs

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

    def __init__(self, hidden_channels):
        """
        :param hidden_channels: Number of filters used for the hidden layers of the NN() function, see GLOW paper
        """
        super(AffineCoupling, self).__init__()

        self.NN = tf.keras.Sequential()
        self.hidden_channels = hidden_channels

    # Create the state of the layer (weights)
    def build(self, input_shape):
        b, h, w, c = input_shape

        self.NN.add(tf.keras.layers.Conv2D(self.hidden_channels, kernel_size=3,
                                           activation='relu', strides=(1, 1), 
                                           padding='same'))
        self.NN.add(tf.keras.layers.Conv2D(self.hidden_channels, kernel_size=1,
                                           activation='relu', strides=(1, 1), 
                                           padding='same'))
        self.NN.add(tf.keras.layers.Conv2D(c, kernel_size=3,
                                           activation='relu', strides=(1, 1), 
                                           padding='same',
                                           kernel_initializer=
                                           tf.keras.initializers.Zeros))

    # Defines the computation from inputs to outputs
    def call(self, inputs, forward=True):
        """
        Computes the forward/reverse calculations of the affine coupling layer
        inputs: A tensor with assumed dimensions: (batch_size x height x width x channels)
        returns: A tensor, same dimensions as input tensor, for next step of flow and the scalar log determinant
        """

        if forward:
            # split along the channels, which is axis=3
            x_a, x_b = tf.split(inputs, num_or_size_splits=2, axis=3)

            # split along the channels again? to get log_s and t
            log_s, t = tf.split(self.NN(x_b), num_or_size_splits=2, axis=3)
            s = tf.math.exp(log_s)
            y_a = tf.math.multiply(s, x_a) + t

            y_b = x_b
            output = tf.concat((y_a, y_b), axis=3)

            log_det = tf.math.log(tf.math.abs(s))
            log_det = tf.math.reduce_sum(log_det)

            self.add_loss(-log_det)

            return output

        # the reverse calculations, if forward is False
        else:
            y_a, y_b = tf.split(inputs, num_or_size_splits=2, axis=3)
            log_s, t = tf.split(self.NN(y_b), num_or_size_splits=2, axis=3)
            s = tf.math.exp(log_s)

            x_a = tf.math.divide((y_a - t), s)
            x_b = y_b
            output = tf.concat((x_a, x_b), axis=3)

            return output

## Squeeze, no trainable parameters
class Squeeze(Layer):

    def __init__(self):
        super(Squeeze, self).__init__()

    # Defines the computation from inputs to outputs
    def call(self, inputs, forward=True):
        """
        inputs: input tensor
        returns: output tensor
        """
        if forward:
            outputs = tf.nn.space_to_depth(inputs, block_size=2)
        else:
            outputs = tf.nn.depth_to_space(inputs, block_size=2)

        return outputs

## Split, what the hell do they do in https://github.com/openai/glow ???
class Split(Layer):

    def __init__(self):
        super(Split, self).__init__()

    # Defines the computation from inputs to outputs
    def call(self, inputs, forward=True, last_layer=False):
        """
        inputs: input tensor
        returns: output tensor
        """
        b, h, w, c = inputs.shape   # Batch size, height, width, channels

        if forward:
            z = inputs[:, :, :, 0:c // 2]
            x = inputs[:, :, :, c // 2:]

            return x, z

        else:
            # should just sample with one colour channel, right?
            # definitely not sure about the reshaping part
            if last_layer:
                sample_shape = b, 1
                sample = Glow.latent_distribution.sample(sample_shape=sample_shape, seed=None, name='sample')
                reverse_input = tf.reshape(sample, shape=(b, h, w, 1))

                return reverse_input

            # then for the other layers, it should sample the same number of channels as input channels... I think
            else:
                sample_shape = b, c
                samples = Glow.latent_distribution.sample(sample_shape=sample_shape, seed=None, name='sample')
                reverse_z = tf.reshape(samples, shape=(b, h, w, c))

                reverse_input = tf.concat((reverse_z, inputs), axis=3)

                return reverse_input


## Step of flow
class FlowStep(Layer):

    def __init__(self, hidden_channels):
        super(FlowStep, self).__init__()
        self.hidden_channels = hidden_channels

        self.actnorm = ActNorm()
        self.perm = Permutation()
        self.coupling = AffineCoupling(hidden_channels)

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

def log(x, base):
    """
    x: tensor
    b: int
    returns log_base(x)
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

### MODEL

class Glow(keras.Model):

    def __init__(self, steps, levels, dimension, hidden_channels):
        super(Glow, self).__init__()

        self.steps = steps  # Number of steps in each flow, K in the paper
        self.levels = levels    # Number of levels, L in the paper
        self.dimension = dimension  # Dimension of input/latent space
        self.hidden_channels = hidden_channels

        # Normal distribution with 0 mean and std=1, defined over R^dimension
        self.latent_distribution = tf_prob.distributions.MultivariateNormalDiag(
            loc=[0.0]*dimension)

        self.squeeze = Squeeze()
        self.split = Split()
        self.flow_layers = []

        for l in range(levels):
            flows = []

            for k in range(steps):
                flows.append([ActNorm(), Permutation(), AffineCoupling(hidden_channels)])

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

            latent_dim = np.prod(z.shape[1:])   # Dimension of extracted z
            latent_variables.append(tf.reshape(z, [-1, latent_dim])) 
        
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
        
        latent_dim = np.prod(x.shape[1:])   # Dimension of last latent variable
        latent_variables.append(tf.reshape(x, [-1, latent_dim]))         

        # Concatenate latent variables
        latent_variables = tf.concat(latent_variables, axis=1)

        latent_logprob = self.latent_distribution.log_prob(latent_variables)
        latent_logprob = tf.reduce_mean(latent_logprob)

        self.add_loss(-latent_logprob)

        return latent_variables
