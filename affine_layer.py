import tensorflow as tf
from tensorflow.keras.layers import Layer


class AffineCoupling(Layer):

    def __init__(self, in_channels, out_channels):
        """

        :param in_channels: Number of filters used for the hidden layers of the NN() function, see GLOW paper
        :param out_channels: Number of filters/output channels from the last convolutional layer

        """
        super(AffineCoupling, self).__init__()

        self.NN = tf.keras.Sequential()
        self.NN.add(tf.keras.layers.Conv2D(in_channels, kernel_size=3,
                                           activation='relu', strides=(1, 1), padding='same'))
        self.NN.add(tf.keras.layers.Conv2D(in_channels, kernel_size=1,
                                           activation='relu', strides=(1, 1), padding='same'))
        self.NN.add(tf.keras.layers.Conv2D(out_channels, kernel_size=3,
                                           activation='relu', strides=(1, 1), padding='same',
                                           kernel_initializer=tf.keras.initializers.Zeros))

    def build(self, input_shape):

        pass

    # Defines the computation from inputs to outputs
    def forward_call(self, inputs):
        """
        Computes the forward calculations of the affine coupling layer

        inputs: A tensor with assumed dimensions: (batch_size x height x width x channels)
        returns: A tensor, same dimensions as input tensor, for next step of flow and the scalar log determinant
        """

        # split along the channels, which is axis=3
        x_a, x_b = tf.split(inputs, num_or_size_splits=2, axis=3)

        # split along the channels again? to get log_s and t
        log_s, t = tf.split(self.NN(x_b), num_or_size_splits=2, axis=3)
        log_s = tf.math.log(log_s)
        s = tf.math.exp(log_s)
        y_a = tf.math.multiply(s, x_a) + t

        y_b = x_b
        output = tf.concat((y_a, y_b), axis=3)

        log_det = tf.math.reduce_sum(log_s)

        return output, log_det

    def reverse_call(self, inputs):
        """
        Computes the reverse calculations of the affine coupling layer

        :param inputs: A tensor with assumed dimensions: (batch_size x height x width x channels)
        :return: A tensor, same dimensions as input tensor, for a reverse step of flow
        """

        y_a, y_b = tf.split(inputs, num_or_size_splits=2, axis=3)
        log_s, t = tf.split(self.NN(y_b), num_or_size_splits=2, axis=3)
        log_s = tf.math.log(log_s)
        s = tf.math.exp(log_s)

        x_a = tf.math.divide((y_a - t), s)
        x_b = y_b
        output = tf.concat((x_a, x_b), axis=3)

        return output


if __name__ == '__main__':

    tf.random.set_seed(10)
    tensor_shape = (4, 28, 28, 4)
    test_tensor = tf.random.normal(tensor_shape)

    aff_coup_layer = AffineCoupling(in_channels=512, out_channels=test_tensor.shape[-1] / 2)
    forward_transf = aff_coup_layer.forward_call(test_tensor)
    print(forward_transf[0].shape)
    reverse_transf = aff_coup_layer.reverse_call(test_tensor)
    print(reverse_transf.shape)







