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

# Train
model = Glow(steps=3, levels=2, dimension=28*28*1, hidden_channels=512)
adam = keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer=adam)   # The loss to optimize is defined inside Glow model

model.fit(X_train, epochs=10, batch_size=128)
