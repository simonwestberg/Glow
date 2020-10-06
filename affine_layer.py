
import tensorflow as tf
import numpy as np

class NN:

    def __init__(self, in_channels, mid_channels, input_shape, testing=True):

        if testing:
            self.first_conv = tf.keras.layers.Conv2D(in_channels, kernel_size=3,
                                                     activation='relu', strides=(1, 1), padding='same',
                                                     input_shape=input_shape)
        else:
            self.first_conv = tf.keras.layers.Conv2D(in_channels, kernel_size=3,
                                                     activation='relu', strides=(1, 1), padding='same')

        self.second_conv = tf.keras.layers.Conv2D(mid_channels, kernel_size=1,
                                                  activation='relu', strides=(1, 1), padding='same')

        self.third_conv = tf.keras.layers.Conv2D(2*mid_channels, kernel_size=3,
                                                 activation='relu', strides=(1, 1), padding='same')

    def forward(self, x, actnorm=False):
        """

        :param actnorm:
        :param x: split of the data set
        :return: returns the concatened output

        """
        # when actnorm is implemented
        if actnorm:
            pass
        # possibly implement some other function as well?
        else:
            pass

        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)

        return x


if __name__ == '__main__':
    tensor_shape = (4, 28, 28, 3)
    test_tensor = tf.random.normal(tensor_shape)
    nn = NN(512, 512, tensor_shape[1:], testing=True)
    output = nn.forward(test_tensor, actnorm=True)
    print(test_tensor.shape)
    print(output.shape)



