### PositionNet+ for Support Recovery of Sparse Signals ###
### Channel Estimation of Massive MIMO-OTFS Wireless Systems ###

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

Nt = 16            # Number of transmit antennas
Nfft = 1024        # FFT size of base modem
M = 600            # OTFS frame length (delay)
N = 10             # OTFS frame length (Doppler)
eta = 0.20         # Pilot overhead
Mt = 128           # Number of pilots along delay dimension
Nv = 10            # Number of pilots along Doppler dimension
Nc = 64            # Number of filters (feature maps)
alpha = 0.9        # Binary Cross-Entropy weight

### Loading Datasets ###

yDD_train = sio.loadmat('yDD_Nt64_0p20.mat', mat_dtype=True)['yDD']       # Delay-Doppler domain Observation Vector
Position_train = sio.loadmat('Position_Nt64_0p20.mat', mat_dtype=True)['Position']  # Delay-Doppler domain Support

### Learning Rate Schedule ###

def learning_rate_scheduler(epoch):
    if epoch < 8:
        return 1e-3
    elif epoch < 10:
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

# Model's Input
yDD = tf.keras.Input(shape=(2, Mt*Nv))         # Delay-Doppler domain Observation Vector

yDDr = tf.keras.layers.Reshape((2, Mt*Nv, 1))(yDD)
yDDr = tf.keras.layers.Conv2D(Nc, (3, 3), padding='same')(yDDr)
yDDr = tf.keras.layers.LeakyReLU(negative_slope=0.1)(yDDr)
yDDr = tf.keras.layers.LayerNormalization()(yDDr)
yDDr = tf.keras.layers.Conv2D(Nc, (3, 3), padding='same')(yDDr)
yDDr = tf.keras.layers.LeakyReLU(negative_slope=0.1)(yDDr)
yDDr = tf.keras.layers.LayerNormalization()(yDDr)
yDDr = tf.keras.layers.Conv2D(Nc, (3, 5), padding='same')(yDDr)
yDDr = tf.keras.layers.LeakyReLU(negative_slope=0.1)(yDDr)
yDDr = tf.keras.layers.LayerNormalization()(yDDr)
yDDr = tf.keras.layers.Dense(Nc)(yDDr)
yDDr = tf.keras.layers.BatchNormalization()(yDDr)
yDDr = tf.keras.layers.Dense(1)(yDDr)
yDDr = tf.keras.layers.Reshape((2, Mt*Nv))(yDDr)

yDDi = Permute_and_Negate_Layer()(yDD)
yDDi = tf.keras.layers.Reshape((2, Mt*Nv, 1))(yDDi)
yDDi = tf.keras.layers.Conv2D(Nc, (3, 3), padding='same')(yDDi)
yDDi = tf.keras.layers.LeakyReLU(negative_slope=0.1)(yDDi)
yDDi = tf.keras.layers.LayerNormalization()(yDDi)
yDDi = tf.keras.layers.Conv2D(Nc, (3, 3), padding='same')(yDDi)
yDDi = tf.keras.layers.LeakyReLU(negative_slope=0.1)(yDDi)
yDDi = tf.keras.layers.LayerNormalization()(yDDi)
yDDi = tf.keras.layers.Conv2D(Nc, (3, 5), padding='same')(yDDi)
yDDi = tf.keras.layers.LeakyReLU(negative_slope=0.1)(yDDi)
yDDi = tf.keras.layers.LayerNormalization()(yDDi)
yDDi = tf.keras.layers.Dense(Nc)(yDDi)
yDDi = tf.keras.layers.BatchNormalization()(yDDi)
yDDi = tf.keras.layers.Dense(1)(yDDi)
yDDi = tf.keras.layers.Reshape((2, Mt*Nv))(yDDi)

# Feature Extraction (Support Recovery)
F11 = tf.keras.layers.Dense(Mt*Nv*Nt)(yDDr)           # Complex Dense layer
F11 = tf.keras.layers.Dropout(0.2)(F11)
F21 = tf.keras.layers.Dense(Mt*Nv*Nt)(yDDi)
F21 = tf.keras.layers.Dropout(0.2)(F21)
F1  = tf.keras.layers.Add()([F11, F21])
Feature = tf.keras.layers.Lambda(lambda x: x ** 2, output_shape=(2, Mt*Nv*Nt))(F1)     # Absolute Value layer (Squared)
Feature = tf.keras.layers.AveragePooling1D(pool_size=(2))(Feature)
Feature = tf.keras.layers.Reshape((Mt, Nv, Nt, 1))(Feature)
Feature = tf.keras.layers.Conv3D(Nc, (3, 3, 3), padding='same')(Feature)
Feature = tf.keras.layers.Conv3D(Nc, (3, 3, 3), padding='same')(Feature)
Feature = tf.keras.layers.Conv3D(Nc, (3, 3, 3), padding='same')(Feature)
Feature = tf.keras.layers.LayerNormalization()(Feature)
Feature = tf.keras.layers.BatchNormalization(axis=1)(Feature)
Feature = tf.keras.layers.Rescaling(2.5, offset=0)(Feature)

Position = tf.keras.layers.Softmax(axis=1)(Feature)    # Softmax layer
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1)(Position)
Position = tf.keras.layers.Reshape((Mt, Nv, Nt))(Position)
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1)(Position)
Position = tf.keras.layers.Reshape((Mt, Nv))(Position)
Position = tf.keras.layers.Dense(32)(Position)
Position = tf.keras.layers.Dense(1, activation="sigmoid")(Position)
Position = tf.keras.layers.Reshape((Mt,))(Position)

PositionNetPlus = tf.keras.models.Model(yDD, Position)

PositionNetPlus.summary()

PositionNetPlus.compile(optimizer=tf.keras.optimizers.AdamW(),
              loss=[tf.keras.losses.CosineSimilarity()],        # Weighted_Binary_CrossEntropy
              metrics=['mse'])

PN_history = PositionNetPlus.fit(yDD_train, Position_train, validation_split=0.20, epochs=12, callbacks=[learning_rate_schedule])

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

