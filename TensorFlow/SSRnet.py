### SSRnet: Sparse Signal Recovery Neural Network ###
### Channel Estimation of Massive MIMO-OTFS Wireless Systems ###

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

Nt = 16            # Number of transmit antennas
Nfft = 1024        # FFT size of base modem
M  = 600           # OTFS frame length (delay)
N  = 10            # OTFS frame length (Doppler)
eta = 0.15         # Pilot overhead
K  = 6             # Sparsity
Mt = 96            # Number of pilots along delay dimension
Nv = 10            # Number of pilots along Doppler dimension
Nc = 32            # Number of filters (feature maps)
Nf = 32            # Number of filters (feature maps)
NL = 12            # Number of Learned ISP-SL0 layers
alpha = 0.8        # Binary Cross-Entropy weight

### Loading Dataset ###

yDD = sio.loadmat('yDD_Nt64_0p20.mat')['yDD']                  # Delay-Doppler domain Observation Vector
Phi = sio.loadmat('Phi_Nt64_0p20.mat')['Phi']                  # Sensing matrix
CH_ADD_train = sio.loadmat('CH_ADD_Nt64_0p20.mat')['CH_ADD']   # Delay-Doppler domain 3D sparse channel matrix

A        = tf.constant(Phi, dtype=tf.complex64)
A_AH     = tf.linalg.matmul(A, A, adjoint_b=True) + (0.0001 + 0.0j) * tf.eye(Mt*Nv, dtype=tf.complex64)
A_AH_inv = tf.linalg.inv(A_AH)
A_pinv   = tf.linalg.matmul(A, A_AH_inv, adjoint_a=True)
# A_pinv   = tf.constant(np.linalg.pinv(A), dtype=tf.complex64)

Ar = tf.math.real(A)
Ai = tf.math.imag(A)
A  = tf.concat([tf.concat([Ar, -Ai], axis=-1),
                tf.concat([Ai,  Ar], axis=-1)], axis=0)    # Equivalent real-valued processing
Apinv_r = tf.math.real(A_pinv)
Apinv_i = tf.math.imag(A_pinv)
A_pinv  = tf.concat([tf.concat([Apinv_r, -Apinv_i], axis=-1),
                     tf.concat([Apinv_i,  Apinv_r], axis=-1)], axis=0)

### Learning Rate Schedule ###

def learning_rate_scheduler(epoch):
    if epoch < 12:
        return 1e-3
    elif epoch < 14:
        return 1e-4
    else:
        return 1e-5

learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

### Custom Functions and Layers ###

def log_NMSE(y_true, y_pred):
    Dy  = tf.math.square(y_true - y_pred)
    mse = tf.math.reduce_sum(Dy, axis=(1, 2, 3, 4))
    L2norm = tf.math.reduce_sum(tf.math.square(y_true), axis=(1, 2, 3, 4))
    nmse = mse / L2norm
    log_nmse = tf.math.log(nmse + 1e-7)
    return log_nmse

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

### PositionNet+ with Learned ISP-SL0 ###

PositionNetPlus = tf.keras.models.load_model("PositionNetPlus.keras",
                                         custom_objects={"Permute_and_Negate_Layer": Permute_and_Negate_Layer},
                                         compile=False,
                                         safe_mode=False)
PositionNetPlus.trainable = False

### Learned ISP-SL0

class Initial_Sprase_H_ADD_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Initial_Sprase_H_ADD_Layer, self).__init__(**kwargs)
      
    def call(self, inputs):
        I, y = inputs
        I  = tf.math.round(I)                  # Sparse zero-one vector (Delay dimension)
        I  = tf.tile(I, [1, Nv*Nt*2, 1])       # Nonzero indices along Delay-Doppler-Spatial dimensions
        x0 = tf.linalg.matmul(A_pinv, y) * I   # Initial sparse vector h_ADD
        return x0

# Learned Iterative Sparsification-Projection with Smoothed L0 Norm
class Learned_ISP_SL0_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Learned_ISP_SL0_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a1w   = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Sparsification parameter
        # self.a2w   = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Sparsification parameter
        self.a3w   = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Sparsification parameter
        self.beta  = self.add_weight(shape=(), initializer="zeros", trainable=True)   # Nesterov's Acceleration
        self.alpha = self.add_weight(shape=(), initializer="zeros", trainable=True)   # Step size
        super(Learned_ISP_SL0_Layer, self).build(input_shape)

    def Exponential_Shrinkage(self, z):
        a1 = tf.nn.relu(self.a1w)
        # a2 = tf.nn.relu(self.a2w)
        a3 = tf.nn.relu(self.a3w) * 0.001 + 1e-6
        z  = tf.reshape(z, shape=(-1, 2, Mt*Nv*Nt))
        r  = tf.math.reduce_sum(tf.math.square(z), axis=1, keepdims=True)
        fz = z * (a1 - a1 * tf.math.exp(-r / (a3 ** 2)))
        fz = tf.reshape(fz, shape=(-1, Mt*Nv*Nt*2, 1))
        return fz

    def call(self, inputs):
        x, x0, y = inputs
        w  = tf.math.sigmoid(self.beta)  * 2            # Nesterov's Acceleration
        m  = tf.math.sigmoid(self.alpha) * 2            # Step size
        s  = x + w * (x - x0)                           # Momentum
        x0 = x
        x  = self.Exponential_Shrinkage(s)              # Sparsification
        Ax = tf.linalg.matmul(A, x)
        x  = x - m * tf.linalg.matmul(A_pinv, Ax - y)   # Projection
        return [x, x0]


def Conv3d_Block(x, filters):
    y = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding='same')(x)
    y = tf.keras.layers.LeakyReLU(negative_slope=0.1)(y)
    y = tf.keras.layers.LayerNormalization()(y)
    return y

Support = PositionNetPlus.output
Ydd0    = PositionNetPlus.input
Ydd     = tf.keras.layers.Reshape((Mt*Nv*2, 1))(Ydd0)

h_ADD  = Initial_Sprase_H_ADD_Layer()([Support, Ydd])           # Initial h_ADD = (A_pinv * yDD) .* Support
h_ADD0 = tf.keras.layers.Rescaling(1.0, offset=0.0)(h_ADD)

for i in range(NL):
    h_ADD, h_ADD0 = Learned_ISP_SL0_Layer()([h_ADD, h_ADD0, Ydd])

H_ADD = tf.keras.layers.Reshape((2, Nt, Nv, Mt))(h_ADD)
H_ADD = tf.keras.layers.Permute((4, 3, 2, 1))(H_ADD)
HDD_real, HDD_imag = tf.keras.ops.split(H_ADD, 2, axis=-1)
# HDD_real, HDD_imag = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(H_ADD)

# Model's Inputs
# HDD_real = tf.keras.Input(shape=(Mt, Nv, Nt, 1))
# HDD_imag = tf.keras.Input(shape=(Mt, Nv, Nt, 1))

# Real part of 3D channel matrix
HDD_OTFSr = Conv3d_Block(HDD_real,  Nf)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)

HDD_OTFSr = tf.keras.layers.Subtract()([HDD_real, HDD_OTFSr])

HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nf)
HDD_OTFSr = tf.keras.layers.Conv3D(1, (3, 3, 3), padding='same')(HDD_OTFSr)
CH_ADD_real = tf.keras.layers.Subtract()([HDD_real, HDD_OTFSr])

# Imaginary part of 3D channel matrix
HDD_OTFSi = Conv3d_Block(HDD_imag,  Nf)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)

HDD_OTFSi = tf.keras.layers.Subtract()([HDD_imag, HDD_OTFSi])

HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nf)
HDD_OTFSi = tf.keras.layers.Conv3D(1, (3, 3, 3), padding='same')(HDD_OTFSi)
CH_ADD_imag = tf.keras.layers.Subtract()([HDD_imag, HDD_OTFSi])

# 3D sparse channel matrix
CH_ADD = tf.keras.layers.Concatenate(axis=-1)([CH_ADD_real, CH_ADD_imag])
# CH_ADD = tf.keras.layers.Concatenate(axis=-1)([HDD_real, HDD_imag])

SSRnet = tf.keras.models.Model(PositionNetPlus.input, CH_ADD)

SSRnet.summary()

SSRnet.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
               loss=[log_NMSE],
               metrics=['mse'])

SSRnet_history = SSRnet.fit(yDD_train, CH_ADD_train, validation_split=0.20, epochs=25) # , callbacks=[learning_rate_schedule])

SSRnet.save("SSRnet.keras")

plt.plot(SSRnet_history.history['loss'], label='Train Loss')
plt.plot(SSRnet_history.history['val_loss'], label='Validation Loss')
plt.title('Training Result of SSRnet')
plt.ylabel('log(NMSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.75)
plt.tight_layout()
plt.show()
