### SSRnet: Sparse Signal Recovery Neural Network based on LS Solution ###
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
# Binary Cross-Entropy Weights based on Probabilities of Occurrence of Nonzero Indices
# kappa = tf.constant(tf.concat([[0.5], 0.254 * tf.math.pow(0.952, tf.range(1, Mt, dtype=tf.float32))], axis=0))

### Load Dataset ###

yDD_train = sio.loadmat('yDD_observation_vector.mat')['yDD']           # Delay-Doppler domain Observation Vector
CH_ADD_train = sio.loadmat('CH_ADD_3D_sparse_channel.mat')['CH_ADD']   # Delay-Doppler domain 3D sparse channel matrix
Phi = sio.loadmat('Sensing_matrix.mat')['Phi']                         # Sensing matrix

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

# Non-Trainable Computational Layer Obtaining LS Solution using Support Derived by PositionNet+ (Refined by 3D-OMP)
class Computational_LS_Solution_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Computational_LS_Solution_Layer, self).__init__(**kwargs)

    def call(self, inputs):
        Support, yDD = inputs
        I = tf.math.round(Support)                                      # Sparse zero-one vector (Delay dimension)
        M_Delay, index_Delay = tf.math.top_k(I, k=K)                    # Nonzero indices along the delay dimension
        index_expan = tf.range(Nv * Nt, dtype=tf.int32) * Mt
        index_expan = tf.repeat(index_expan, [K])
        index = tf.tile(index_Delay, [1, Nv*Nt]) + index_expan
        M = tf.tile(M_Delay, [1, Nv*Nt])
        M = tf.expand_dims(M, axis=1)
        Phi_I = tf.gather(Phi, index, axis=1)                           # Collect culomns of sensing matrix according to the support
        Phi_I = tf.transpose(Phi_I, perm=[1, 0, 2])
        Phi_I = tf.math.multiply(Phi_I, tf.complex(M, tf.zeros_like(M)))
        yDDx = tf.expand_dims(tf.complex(yDD[:, 0, :], yDD[:, 1, :]), axis=-1)
        N_batch = tf.shape(Support)[0]

        h_ADD = tf.linalg.lstsq(Phi_I, yDDx, l2_regularizer=1e-3)       # Least Squares (LS) solution
        h_ADD = tf.reshape(h_ADD, [N_batch, Nt, Nv, K])                 # Reshape to a 3D dense tensor
        h_ADD = tf.transpose(h_ADD, perm=[0, 3, 2, 1])

        index_3D = tf.expand_dims(index_Delay, axis=-1)
        index_batch = tf.range(tf.shape(index_3D)[0])[:, tf.newaxis, tf.newaxis]
        index_batch = tf.tile(index_batch, [1, K, 1])
        index_3D = tf.stack([index_batch, index_3D], axis=-1)
        index_3D = tf.reshape(index_3D, [N_batch, K, 2])                # Nonzero indices of 3D sparse channel matrix

        HDD_real = tf.scatter_nd(index_3D, tf.math.real(h_ADD), [N_batch, Mt, Nv, Nt])   # Scatter into a 3D sparse channel tensor
        HDD_imag = tf.scatter_nd(index_3D, tf.math.imag(h_ADD), [N_batch, Mt, Nv, Nt])
        return HDD_real, HDD_imag

### PositionNet+ and Computational LS Solution Layer ###

PositionNetPlus = tf.keras.models.load_model("PositionNetPlus_3D_OMP.keras",
                                         custom_objects={"Permute_and_Negate_Layer": Permute_and_Negate_Layer},
                                         compile=False,
                                         safe_mode=False)
PositionNetPlus.trainable = False

HDD_real, HDD_imag = Computational_LS_Solution_Layer()([PositionNet.output, PositionNet.input])    # Computational LS Solution Layer
PositionNet_LS = tf.keras.models.Model(PositionNet.input, [HDD_real, HDD_imag])
PositionNet_LS.summary()

### SSRnet ###

HDD_real, HDD_imag = tf.keras.ops.split(PositionNet_LS.output, 2, axis=-1)

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
