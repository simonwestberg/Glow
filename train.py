### MNIST

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

from model import Glow

n_samples = 60000

# Data set
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Add channel dimension
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

X_train = X_train[0 : n_samples, :, :, :]
X_test = X_test[0 : n_samples, :, :, :]

# Add zero-padding to get 32x32 images
X_train = np.pad(X_train, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Just train on zeros
X_zeros = np.zeros((5923, 32, 32, 1))
idx = 0
for i, x in enumerate(X_train):
    if Y_train[i] == 0:
        X_zeros[idx] = x
        idx += 1
assert(idx == X_zeros.shape[0])

import time

model = Glow(steps=32, levels=3, img_shape=(32, 32, 1), hidden_channels=512, perm_type="1x1")
adam = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=adam)


model.fit(X_zeros, epochs=10, batch_size=128)

# Plot loss
loss = model.history.history["loss"]

epochs = np.arange(len(loss))

plt.plot(epochs, loss)

# Sample image
x, _ = model.sample_image()

plt.imshow(x[0, :, :, 0], cmap='gray')

# Linear interpolation
x1 = X_zeros[0:1]
x2 = X_zeros[5700:5701]

# Encode
z1, _ = model(x1, forward=True)
z1 = z1[0]
z2, _ = model(x2, forward=True)
z2 = z2[0]

# Interpolate
num = 10

latents = [None for i in range(num)]

alpha = 0

for i in range(num):
    latents[i] = (1 - alpha) * z1 + alpha * z2
    alpha += 1 / (num - 1.0)

# Decode
imgs = [None for i in range(num)]

for i, z in enumerate(latents):
    output, _ = model(z, forward=False)
    imgs[i] = output

# Plot
# Plot
fig=plt.figure(figsize=(8, 8))

for i in range(num):
    fig.add_subplot(1, num, i+1)
    plt.imshow(imgs[i][0, :, :, 0], cmap="gray")

plt.show()
