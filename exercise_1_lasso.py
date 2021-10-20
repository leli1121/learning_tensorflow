import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

p = 50
n = 100
p_start = 10
bias = 1.0
EPOCHS = 2000  # should be a large epoch value to converge
BATCH_SIZE = 32


# construct dataset
theta_start = np.zeros(p)
np.random.seed(10)
idx = np.random.choice(range(p), p_start)
theta_start[idx] = np.random.normal(loc=10, size=10)
x = np.random.normal(loc=3, size=[n, p])
y = np.dot(x, theta_start) + bias + np.random.normal(size=n)


# 利用TF中已有模型构造
def build_model():
    model = keras.Sequential([
            layers.Dense(1,
                         kernel_regularizer=tf.keras.regularizers.l1(10),
                         input_shape=[p])])

    optimizer = tf.keras.optimizers.SGD(lr=0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model


model = build_model()
model.summary()


history = model.fit(x, y, epochs=EPOCHS, verbose=0, batch_size=BATCH_SIZE)

parameters = model.trainable_variables[0].numpy()

print(parameters[idx])
print(np.delete(parameters, idx))