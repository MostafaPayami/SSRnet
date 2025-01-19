### PositionNet+ for Support Recovery of Sparse Signals ###
### Channel Estimation of Massive MIMO-OTFS Wireless Systems ###

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

Nt = 32            # Number of transmit antennas
Nfft = 1024        # FFT size of base modem
M  = 600           # OTFS frame length (delay)
N  = 10            # OTFS frame length (Doppler)
eta = 0.15         # Pilot overhead
K  = 6             # Sparsity
Mt = 96            # Number of pilots along delay dimension
Nv = 10            # Number of pilots along Doppler dimension
Nc = 32            # Number of filters (feature maps)
Nf = 32            # Number of filters (feature maps)
kappa = 0.8        # Binary Cross-Entropy weight
# Binary Cross-Entropy Weights based on Probabilities of Occurrence of Nonzero Indices in 3GPP Spatial Channel Model
# kappa = tf.constant(tf.concat([[0.5], 0.254 * tf.math.pow(0.952, tf.range(1, Mt, dtype=tf.float32))], axis=0))

### Load Dataset ###

yDD_train = sio.loadmat('yDD_observation_vector_Nt32_0p15.mat')['yDD']           # Delay-Doppler domain Observation Vector
Position_train = sio.loadmat('Position_Nt32_0p15.mat', mat_dtype=True)['Position']    # Delay domain Support: Position = sign(|H_ADD|)

### Learning Rate Schedule ###

def learning_rate_scheduler(epoch):
    if epoch < 16:
        return 1e-3
    elif epoch < 18:
        return 1e-4
    else:
        return 1e-5

learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

### TensorFlow/Keras Custom Functions and Layers ###

def Weighted_Binary_CrossEntropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    # WBCE   = (1.0 / kappa - 1.0) * y_true * tf.math.log(y_pred) + 1.0 * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    WBCE   = kappa * y_true * tf.math.log(y_pred) + (1.0 - kappa) * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    WBCE   = -tf.math.reduce_mean(WBCE)
    return WBCE

class Permute_and_Negate_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Permute_and_Negate_Layer, self).__init__(**kwargs)
    def call(self, inputs):
        p = tf.reverse(inputs, axis=[1])
        y = tf.concat([p[:, 0:1, :] * -1, p[:, 1:, :]], axis=1)
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
F11 = tf.keras.layers.Dense(Mt*Nv*Nt)(yDDr)    # Complex Dense layer
F11 = tf.keras.layers.Dropout(0.25)(F11)
F21 = tf.keras.layers.Dense(Mt*Nv*Nt)(yDDi)
F21 = tf.keras.layers.Dropout(0.25)(F21)
F1  = tf.keras.layers.Add()([F11, F21])
# Feature = tf.keras.ops.abs(F1)          # Absolute Value layer
Feature = tf.keras.ops.square(F1)       # Squared Absolute Value layer
Feature = tf.keras.layers.AveragePooling1D(pool_size=(2))(Feature)
Feature = tf.keras.layers.Reshape((Mt, Nv, Nt, 1))(Feature)
Feature = tf.keras.layers.Conv3D(Nc, (3, 3, 3), padding='same')(Feature)
Feature = tf.keras.layers.Conv3D(Nc, (3, 3, 3), padding='same')(Feature)
Feature = tf.keras.layers.Conv3D(Nc, (3, 3, 3), padding='same')(Feature)
Feature = tf.keras.layers.LayerNormalization()(Feature)
Feature = tf.keras.layers.BatchNormalization(axis=1)(Feature)
Feature = tf.keras.layers.Rescaling(2.5, offset=0.0)(Feature)

Position = tf.keras.layers.Softmax(axis=1)(Feature)    # Softmax layer

Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1)(Position)
Position = tf.keras.layers.Reshape((Mt, Nv, Nt))(Position)
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1)(Position)
Position = tf.keras.layers.Reshape((Mt, Nv))(Position)
Position = tf.keras.layers.Dense(Nc)(Position)
Position = tf.keras.layers.Dense(1, activation="sigmoid")(Position)
Position = tf.keras.layers.Reshape((Mt,))(Position)

PositionNetPlus = tf.keras.models.Model(yDD, Position)

PositionNetPlus.summary()

PositionNetPlus.compile(optimizer=tf.keras.optimizers.AdamW(),
              loss=[tf.keras.losses.CosineSimilarity()],        # Weighted_Binary_CrossEntropy
              metrics=['mse', 'binary_crossentropy'])           # 'cosine_similarity'

PN_history = PositionNetPlus.fit(yDD_train, Position_train, validation_split=0.25, epochs=20, callbacks=[learning_rate_schedule])

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
