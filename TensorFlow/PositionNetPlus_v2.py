### PositionNet+ for Support Recovery of Sparse Signals (Version 2) ###
### Massive MIMO-OTFS Wireless Systems ###

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

Nt = 16            # Number of transmit antennas
Nfft = 1024        # FFT size of base modem
M  = 600           # OTFS frame length (delay)
N  = 10            # OTFS frame length (Doppler)
eta = 0.10         # Pilot overhead
Mt = 64            # Number of pilots along delay dimension
Nv = 10            # Number of pilots along Doppler dimension
Nc = 32            # Number of filters (feature maps)
alpha = 0.8        # Binary Cross-Entropy weight

### Loading Datasets ###

yDD_real_train = sio.loadmat('yDD_Real_Nt64_0p20.mat', mat_dtype=True)['yDD_real']    # Delay-Doppler domain Observation Vector (Real part)
yDD_imag_train = sio.loadmat('yDD_Imag_Nt64_0p20.mat', mat_dtype=True)['yDD_imag']    # Delay-Doppler domain Observation Vector (Imaginary part)
Position_train = sio.loadmat('Position_Nt64_0p20.mat', mat_dtype=True)['Position']    # Delay-Doppler domain Support

### Learning Rate Schedule ###

def learning_rate_scheduler(epoch):
    if epoch < 12:
        return 1e-3
    elif epoch < 14:
        return 1e-4
    else:
        return 1e-5

learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

### TensorFlow/Keras Custom Functions and Layers ###

def Weighted_Binary_CrossEntropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    WBCE   = -tf.math.reduce_mean(alpha * y_true * tf.math.log(y_pred) + (1.0 - alpha) * (1 - y_true) * tf.math.log(1 - y_pred))
    return WBCE

class Permute_and_Negate_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Permute_and_Negate_Layer, self).__init__(**kwargs)
    def call(self, inputs):
        permuted = tf.reverse(inputs, axis=[1])
        y = tf.concat([permuted[:, 0:1, :] * -1, permuted[:, 1:, :]], axis=1)
        return y

### PositionNet+ ###

# Model's Inputs
yDD_real = tf.keras.Input(shape=(Mt*Nv,))           # Delay-Doppler domain Observation Vector (Real part)
yDD_imag = tf.keras.Input(shape=(Mt*Nv,))           # Delay-Doppler domain Observation Vector (Imaginary part)

yDDr = tf.keras.layers.Reshape((1, Mt*Nv))(yDD_real)
yDDi = tf.keras.layers.Reshape((1, Mt*Nv))(yDD_imag)

yDD1 = tf.keras.layers.Concatenate(axis=1)([yDDr,  yDDi])
yDD2 = tf.keras.layers.Concatenate(axis=1)([-yDDi, yDDr])

F11 = tf.keras.layers.Dense(Mt*Nv*Nt)(yDD1)                       # Complex Dense layer
F11 = tf.keras.layers.Dropout(0.2)(F11)
F21 = tf.keras.layers.Dense(Mt*Nv*Nt)(yDD2)                       # Complex Dense layer
F21 = tf.keras.layers.Dropout(0.2)(F21)
F1  = tf.keras.layers.Add()([F11, F21])
Feature  = tf.keras.layers.Lambda(lambda x: x ** 2, output_shape=(2, Mt*Nv*Nt))(F1)     # Absolute Value layer (Squared)
Feature  = tf.keras.layers.AveragePooling1D(pool_size=(2))(Feature)
Feature  = tf.keras.layers.Reshape((Mt, Nv, Nt, 1))(Feature)
Feature  = tf.keras.layers.Conv3D(Nc, (1, 3, 3), padding='same')(Feature)
Feature  = tf.keras.layers.LayerNormalization()(Feature)
Feature  = tf.keras.layers.BatchNormalization(axis=1)(Feature)
Feature  = tf.keras.layers.Rescaling(2.5, offset=0)(Feature)

Position = tf.keras.layers.Softmax(axis=1)(Feature)               # Softmax layer
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1)(Position)
Position = tf.keras.layers.Reshape((Mt, Nv, Nt))(Position)
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1)(Position)
Position = tf.keras.layers.Reshape((Mt, Nv))(Position)
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1, activation="sigmoid")(Position)
Position = tf.keras.layers.Reshape((Mt,))(Position)

PositionNetPlus = tf.keras.models.Model([yDD_real, yDD_imag], Position)

PositionNetPlus.summary()

PositionNetPlus.compile(optimizer=tf.keras.optimizers.AdamW(),
              loss=[tf.keras.losses.CosineSimilarity()],       # Weighted_Binary_CrossEntropy
              metrics=['mse'])

PN_history = PositionNetPlus.fit([yDD_real_train, yDD_imag_train], Position_train, validation_split=0.20, epochs=15, callbacks=[learning_rate_schedule])

PositionNetPlus.save("PositionNetPlus.keras")

plt.plot(PN_history.history['loss'], label='Train Loss')
plt.plot(PN_history.history['val_loss'], label='Validation Loss')
plt.title('Training Result of PositionNet+') 
plt.ylabel('Negated Cosine Similarity')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.75)
plt.tight_layout()
plt.show()
