### SSRnet: Sparse Signal Recovery Neural Network ###
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
NL = 10            # Number of Learned ISP-SL0 layers
kappa = 0.8        # Binary Cross-Entropy weight
# Binary Cross-Entropy Weights based on Probabilities of Occurrence of Nonzero Indices
# kappa = tf.constant(tf.concat([[0.5], 0.254 * tf.math.pow(0.952, tf.range(1, Mt, dtype=tf.float32))], axis=0))

### Load Dataset ### 

yDD_train = sio.loadmat('yDD_observation_vector.mat')['yDD']           # Delay-Doppler domain Observation Vector
CH_ADD_train = sio.loadmat('CH_ADD_3D_sparse_channel.mat')['CH_ADD']   # Delay-Doppler domain 3D sparse channel matrix
Phi = sio.loadmat('Sensing_matrix.mat')['Phi']                         # Sensing matrix

A        = tf.constant(Phi, dtype=tf.complex64)
A_AH     = tf.linalg.matmul(A, A, adjoint_b=True) + (0.001 + 0.0j) * tf.eye(Mt*Nv, dtype=tf.complex64)
A_AH_inv = tf.linalg.inv(A_AH)
A_pinv   = tf.linalg.matmul(A, A_AH_inv, adjoint_a=True)

Ar = tf.math.real(A)
Ai = tf.math.imag(A)
A  = tf.constant(tf.concat([tf.concat([Ar, -Ai], axis=-1),
                            tf.concat([Ai,  Ar], axis=-1)], axis=0), dtype=tf.float32)    # Equivalent real-valued processing

Apinv_r = tf.math.real(A_pinv)
Apinv_i = tf.math.imag(A_pinv)
A_pinv  = tf.constant(tf.concat([tf.concat([Apinv_r, -Apinv_i], axis=-1),
                                 tf.concat([Apinv_i,  Apinv_r], axis=-1)], axis=0), dtype=tf.float32)

### Learning Rate Schedule ###

def learning_rate_scheduler(epoch):
    if epoch < 12:
        return 1e-3
    elif epoch < 15:
        return 5e-4
    elif epoch < 18:
        return 1e-4
    else:
        return 1e-5

learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

### TensorFlow/Keras Custom Functions and Layers ###

def log_NMSE(y_true, y_pred):
    Dy2 = tf.math.square(y_true - y_pred)
    mse = tf.math.reduce_sum(Dy2, axis=(1, 2, 3, 4))
    L2norm = tf.math.reduce_sum(tf.math.square(y_true), axis=(1, 2, 3, 4))
    nmse = (mse + 1e-12) / (L2norm + 1e-12)
    log_nmse = tf.math.log(nmse)
    return log_nmse

def NMSE(y_true, y_pred):
    Dy  = tf.math.square(y_true - y_pred)
    mse = tf.math.reduce_sum(Dy, axis=(1, 2, 3, 4))
    L2norm = tf.math.reduce_sum(tf.math.square(y_true), axis=(1, 2, 3, 4))
    nmse = (mse + 1e-12) / (L2norm + 1e-12)
    return nmse

def Weighted_Binary_CrossEntropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    # WBCE   = (1.0 / kappa - 1.0) * y_true * tf.math.log(y_pred) + 1.0 * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    WBCE   = kappa * y_true * tf.math.log(y_pred) + (1.0 - kappa) * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    WBCE   = -tf.math.reduce_mean(WBCE)
    return WBCE

def SSRnet_Conv3D_Block(x, filters):
    y = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding='same', use_bias=False)(x)
    y = tf.keras.layers.LeakyReLU(negative_slope=0.1)(y)
    y = tf.keras.layers.LayerNormalization()(y)
    return y

class Permute_and_Negate_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Permute_and_Negate_Layer, self).__init__(**kwargs)

    def call(self, inputs):
        p = tf.reverse(inputs, axis=[1])
        y = tf.concat([p[:, 0:1, :] * -1, p[:, 1:, :]], axis=1)
        return y

# Learned Iterative Sparsification-Projection with Smoothed L0 Norm
@tf.keras.utils.register_keras_serializable()
class Learned_ISP_SL0_Layer(tf.keras.layers.Layer):
    def __init__(self, sigma0=1.0, **kwargs):
        super(Learned_ISP_SL0_Layer, self).__init__(**kwargs)
        self.sigma0 = sigma0

    def build(self, input_shape):
        self.a1    = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Sparsification parameter
        self.a2    = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Sparsification parameter
        self.alpha = self.add_weight(shape=(), initializer="zeros", trainable=True)   # Sparsification parameter
        self.sigma = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(self.sigma0), trainable=True)   # Sparsification parameter
        self.w     = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Nesterov's Acceleration
        self.m1    = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Step size
        self.m2    = self.add_weight(shape=(), initializer="ones",  trainable=True)   # Step size
        super(Learned_ISP_SL0_Layer, self).build(input_shape)

    def Exponential_Shrinkage(self, z, I):
        sigmaPN = tf.nn.relu(self.sigma) * (1.0 - tf.math.sigmoid(self.alpha) * I) + 1e-7    # Lower threshold for elements at known nonzero indices
        z  = tf.reshape(z, shape=(-1, 2, Mt*Nv*Nt))
        r  = tf.math.reduce_sum(tf.math.square(z), axis=1, keepdims=True)
        fz = z * (tf.nn.relu(self.a1) - tf.nn.relu(self.a2) * tf.math.exp(-r / (sigmaPN ** 2)))
        fz = tf.reshape(fz, shape=(-1, Mt*Nv*Nt*2, 1))
        return fz

    def call(self, inputs):
        x, x0, y, I2 = inputs
        s  = x + tf.nn.relu(self.w) * (x - x0)                    # Momentum
        x0 = x
        x  = self.Exponential_Shrinkage(s, I2)                    # Sparsification
        Ax = tf.linalg.matmul(A, x)
        x  = tf.nn.relu(self.m1) * x - tf.nn.relu(self.m2) * tf.linalg.matmul(A_pinv, Ax - y)   # Projection
        return [x, x0]

    def get_config(self):
        config = super(Learned_ISP_SL0_Layer, self).get_config()
        config.update({"sigma0": self.sigma0})
        return config

### PositionNet+ ###

PositionNetPlus = tf.keras.models.load_model("PositionNetPlus.keras",
                                         custom_objects={"Permute_and_Negate_Layer": Permute_and_Negate_Layer},
                                         compile=False,
                                         safe_mode=False)
PositionNetPlus.trainable = False

### Learned ISP ###

Ydd = tf.keras.layers.Reshape((2*Mt*Nv, 1))(PositionNetPlus.input)   # Delay-Doppler domain Observation Vector
I   = tf.keras.layers.Reshape((Mt, 1))(PositionNetPlus.output)       # Delay domain Support

I_3D   = tf.keras.ops.tile(I, [1, Nv*Nt*2, 1])                       # Angular-Delay-Doppler domain Support
h_ADD  = tf.keras.ops.matmul(A_pinv, Ydd)
h_ADD  = tf.keras.layers.Multiply()([h_ADD, I_3D])                   # Initial Sparse Channel Vector h_ADD = (A_pinv * yDD) .* Support
h_ADD0 = tf.keras.layers.Rescaling(0.0, offset=0.0)(h_ADD)

I2 = tf.keras.ops.tile(I, [1, Nv*Nt, 1])                             # Support for Sparsification
I2 = tf.keras.layers.Permute((2, 1))(I2)

for k in range(NL):
    h_ADD, h_ADD0 = Learned_ISP_SL0_Layer(sigma0=3.0 * (2e-5 ** (k / (NL - 1))))([h_ADD, h_ADD0, Ydd, I2])    # Initial sigma of SL0

H_ADD = tf.keras.layers.Reshape((2, Nt, Nv, Mt))(h_ADD)
H_ADD = tf.keras.layers.Permute((4, 3, 2, 1))(H_ADD)

Learned_ISP = tf.keras.models.Model([PositionNetPlus.input], H_ADD)

Learned_ISP.summary()

Learned_ISP.compile(optimizer=tf.keras.optimizers.AdamW(),
               loss=[log_NMSE],
               metrics=['mse', NMSE])

Learned_ISP_history = Learned_ISP.fit(yDD_train, CH_ADD_train, validation_split=0.25, epochs=20, callbacks=[learning_rate_schedule])

plt.plot(Learned_ISP_history.history['loss'], label='Train Loss')
plt.plot(Learned_ISP_history.history['val_loss'], label='Validation Loss')
plt.title('Training Result of Learned ISP')
plt.ylabel('log(NMSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.75)
plt.tight_layout()
plt.show()

Learned_ISP.trainable = False

### SSRnet ###

HDD_real, HDD_imag = tf.keras.ops.split(Learned_ISP.output, 2, axis=-1)

# Real part of 3D channel matrix
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_real,  Nf)
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)

HDD_OTFSr = tf.keras.layers.Subtract()([HDD_real, HDD_OTFSr])

HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)
HDD_OTFSr = SSRnet_Conv3D_Block(HDD_OTFSr, Nf)
HDD_OTFSr = tf.keras.layers.Conv3D(1, (3, 3, 3), padding='same', use_bias=False, 
                                   activity_regularizer=tf.keras.regularizers.L1(1e-4))(HDD_OTFSr)
CH_ADD_real = tf.keras.layers.Subtract()([HDD_real, HDD_OTFSr])

# Imaginary part of 3D channel matrix
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_imag,  Nf)
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)

HDD_OTFSi = tf.keras.layers.Subtract()([HDD_imag, HDD_OTFSi])

HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)
HDD_OTFSi = SSRnet_Conv3D_Block(HDD_OTFSi, Nf)
HDD_OTFSi = tf.keras.layers.Conv3D(1, (3, 3, 3), padding='same', use_bias=False, 
                                   activity_regularizer=tf.keras.regularizers.L1(1e-4))(HDD_OTFSi)
CH_ADD_imag = tf.keras.layers.Subtract()([HDD_imag, HDD_OTFSi])

# Estimated 3D Sparse Channel matrix H_ADD
CH_ADD = tf.keras.layers.Concatenate(axis=-1)([CH_ADD_real, CH_ADD_imag])

SSRnet = tf.keras.models.Model(PositionNetPlus.input, CH_ADD)

SSRnet.summary()

SSRnet.compile(optimizer=tf.keras.optimizers.AdamW(),
               loss=[log_NMSE],
               metrics=['mse', NMSE])

SSRnet_history = SSRnet.fit(yDD_train, CH_ADD_train, validation_split=0.25, epochs=20, callbacks=[learning_rate_schedule])

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
