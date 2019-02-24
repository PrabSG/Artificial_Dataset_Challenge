import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

data = np.load("bank-full-balanced-shuffled-onehot-unknowns.npz")
X = data['training_data']
y = data['labels']

NAME = "DNN-relu-128-256-128-softmax-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#Normalise the data
#X = keras.utils.normalize(X, axis=-1, order=2)

model = keras.Sequential()

model.add(keras.layers.Dense(128, kernel_initializer="glorot_normal", bias_initializer="RandomNormal"))
model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("softmax"))

model.compile(keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])
