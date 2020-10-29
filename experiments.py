from matplotlib import pyplot as plt
from extra_keras_datasets import emnist
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

from model import *


def preprocess(data_set, type=None):
    if type is not None:
        (X_train, Y_train), (X_test, Y_test) = data_set.load_data(type=type)
    else:
        (X_train, Y_train), (X_test, Y_test) = data_set.load_data()
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    X_train = X_train[0:, :, :, :]
    X_test = X_test[0:, :, :, :]

    # Add zero-padding to get 32x32 images
    X_train = np.pad(X_train, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
    X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, Y_train, X_test, Y_test


# Load data sets
X_train, Y_train, X_test, Y_test = preprocess(mnist)
X_train_letters, Y_train_letters, X_test_letters, Y_test_letters = preprocess(emnist, type='letters')
X_train_fashion, _, X_test_fashion, _ = preprocess(fashion_mnist)

np.random.shuffle(X_test)
np.random.shuffle(X_test_letters)
np.random.shuffle(X_test_fashion)

# Create balanced MNIST training set

num = 4000  # samples per class

X_balanced = np.zeros((num * 10, 32, 32, 1))
count = [0 for i in range(10)]

idx = 0
for i, x in enumerate(X_train):
    if count[Y_train[i]] < num:
        X_balanced[idx] = x
        count[Y_train[i]] += 1
        idx += 1

assert (idx == X_balanced.shape[0])

np.random.shuffle(X_balanced)

# Train the model
lr = 1e-4

model = Glow(steps=32, levels=3, img_shape=(32, 32, 1), hidden_channels=512, perm_type="1x1")
adam = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=adam)

model.fit(X_balanced, epochs=70, batch_size=128)

# Plot loss
loss = model.history.history["loss"]
epochs = np.arange(len(loss))
plt.plot(epochs, loss, color="r")

# LINEAR INTERPOLATION

images = [[X_test[10:11], X_test[192:193]],
          [X_test[2:3], X_test[46:47]],
          [X_test[1:2], X_test[47:48]],
          [X_test[30:31], X_test[76:77]],
          [X_test[4:5], X_test[24:25]],
          [X_test[15:16], X_test[23:24]],
          [X_test[21:22], X_test[100:101]],
          [X_test[0:1], X_test[64:65]],
          [X_test[61:62], X_test[84:85]],
          [X_test[12:13], X_test[78:79]]
          ]

zs = []

for pair in images:
    x1, x2 = pair[0], pair[1]

    # Encode
    z1, _ = model(x1, forward=True)
    z1 = z1[0]
    z2, _ = model(x2, forward=True)
    z2 = z2[0]

    zs.append([z1, z2])

# Interpolate
num = 10
latents_full = []

for pair in zs:
    z1, z2 = pair[0], pair[1]
    lis = []
    alpha = 0
    for i in range(num):
        lis.append((1 - alpha) * z1 + alpha * z2)
        alpha += 1 / (num - 1.0)
    latents_full.append(lis)

# Decode
decodings = []

for latents in latents_full:
    imgs = []
    for i, z in enumerate(latents):
        output, _ = model(z, forward=False, reconstruct=True)
        imgs.append(output)
    decodings.append(imgs)

# Plot
fig = plt.figure(figsize=(8, 8))

for row in range(10):
    for col in range(num):
        fig.add_subplot(10, num, row * num + col + 1)
        plt.imshow(decodings[row][col][0, :, :, 0], cmap="gray")
        plt.axis('off')

# SAMPLE NEW IMAGES

rows = 10
cols = 10

images = []

for r in range(rows):
    for c in range(cols):
        x = model.sample_image(temperature=0.7)
        images.append(x)

# Plot
fig = plt.figure(figsize=(8, 8))

for r in range(rows):
    for c in range(cols):
        fig.add_subplot(rows, cols, r * cols + c + 1)
        plt.imshow(images[r * cols + c][0, :, :, 0], cmap="gray")
        plt.axis('off')

######################################################
# Typicality experiments for OOD-detection


def compute_nlls(model, data_set, batch_size=128):
    n_batches = int(data_set.shape[0] / batch_size)
    nlls = []

    for batch_index in range(n_batches):
        _, nll = model(data_set[batch_index * batch_size:(batch_index + 1) * batch_size], forward=True)
        nlls.append(nll)

    return nlls


def get_epsilon(model, X_val, nll_trained_model, K, M, alpha=0.99):
    """
    K: number of bootstrap sampled data sets
    M: size of each K data sets
    alpha: confidence level
    """

    N = X_val.shape[0]
    epsilons = []

    # Sample K M-sized data-sets from X_val

    for k in range(K):
        sample_inds = np.random.choice(N, size=M, replace=False)
        e_k = np.abs(tf.reduce_mean(compute_nlls(model, X_val[sample_inds], batch_size=M)) - nll_trained_model)
        epsilons.append(e_k)

    epsilon_M_alpha = np.quantile(epsilons, alpha)

    return epsilons, epsilon_M_alpha, M


def OOD_detection(model, X_test, nll_trained_model, epsilon, M):
    nlls = compute_nlls(model, X_test, batch_size=M)

    ood_count = 0.0
    count = 0.0

    for i in range(len(nlls)):
        nll = nlls[i]
        if np.abs(nll - nll_trained_model) > epsilon:
            ood_count += 1.0
        count += 1.0

    ood_fraction = ood_count / count
    id_fraction = 1.0 - ood_fraction

    return ood_fraction, id_fraction


# Extract validation data for MNIST
X_val = X_test[0:5000]
X_test = X_test[5000:]

nll_trained_model = tf.reduce_mean(compute_nlls(model, X_balanced, batch_size=64))

epsilons, epsilon, M = get_epsilon(model, X_val, nll_trained_model, K=50, M=2, alpha=0.99)

ood_mnist, id_mnist = OOD_detection(model, X_test, nll_trained_model, epsilon, M)
ood_fashion, id_fashion = OOD_detection(model, X_test_fashion[0:5000], nll_trained_model, epsilon, M)
ood_letters, id_letters = OOD_detection(model, X_test_letters[0:5000], nll_trained_model, epsilon, M)

print(ood_mnist, id_mnist)
print(ood_fashion, id_fashion)
print(ood_letters, id_letters)
