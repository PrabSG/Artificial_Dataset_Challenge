import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

data = np.load("bank-full-shuffled-onehot.npz")
X = data['training_data']
y = data['labels']

#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
