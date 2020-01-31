import os, sys 
from sklearn.preprocessing import OneHotEncoder, Binarizer, StandardScaler
import librosa as ro
import soundfile as sf
import random as rd
import requests, tqdm

import tensorflow as tf 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from CNNVocSep import CNNVocSep
from DataHelper import Data

# from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ROOT_DIR = '../LibriSpeech/dev_clean'
BS = 32
EPOCHS = 100
train_test_split = 0.8

data = Data(ROOT_DIR)
x, y = data.generate_data()

x_train, y_train, x_test, y_test = x[:len(x) * train_test_split], y[:len(x) * train_test_split], x[len(x) * train_test_split:], y[len(x) * train_test_split:]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BS)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BS)

model = CNNVocSep(input_size, 3, num_freq_bins, 2)

loss_object = CategoricalCrossentropy()
optimizer = Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()