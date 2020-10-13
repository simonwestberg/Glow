import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

from model import Glow

# Data set
from keras.datasets import mnist

def preprocess(X_train, X_test):
    # Rescale inputs [0, 255] -> [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Subtract training mean divide my training std
    mean_tr = np.mean(X_train, axis=0)
    std_tr = np.std(X_train, axis=0, ddof=1)
    
    # Replace zeros with ones?
    std_tr[std_tr == 0] = 1

    X_train = (X_train - mean_tr) / std_tr
    X_test = (X_test - mean_tr) / std_tr

    return X_train, X_test


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocess?
X_train, X_test = preprocess(X_train, X_test)

# Add channel dimension
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Add zero-padding to get 32x32 images
X_train = np.pad(X_train, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))

# Train
model = Glow(steps=3, levels=2, img_shape=(32, 32, 1), hidden_channels=128, perm_type="1x1")
adam = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=adam)

model.fit(X_train, epochs=2, batch_size=128)

# Encoding and sampling

# Encode x
x = X_train[0, :, :, :].reshape((1, 32, 32, 1))
z, log_det = model(x, forward=True)

# Sample z and decode to generate image
z = model.latent_distribution.sample()

x, log_det = model(z, forward=False)
