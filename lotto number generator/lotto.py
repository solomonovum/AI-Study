import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('c:/lotto.csv', header=None)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

BATCH_SIZE = 128
BUFFER_SIZE = 10000

train = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
val = val.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=train),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 1000

simple_lstm_model.fit(traine, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val, validation_steps=50)
