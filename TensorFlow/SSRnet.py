### SSRnet: Sparse Signal Recovery Neural Network ###
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
K = 6              # Sparsity
Mt = 120           # Number of pilots along delay dimension
Nv = 10            # Number of pilots along Doppler dimension
Nc = 64            # Number of filters (feature maps)
alpha = 0.9        # Binary Cross-Entropy weight

### Loading Dataset ###

yDD = sio.loadmat('yDD_Nt64_0p20.mat')['yDD']                  # Delay-Doppler domain Observation Vector
Phi = sio.loadmat('Phi_Nt64_0p20.mat')['Phi']                  # Sensing matrix
CH_ADD_train = sio.loadmat('CH_ADD_Nt64_0p20.mat')['CH_ADD']   # Delay-Doppler domain 3D sparse channel matrix

### Learning Rate Schedule ###

def learning_rate_scheduler(epoch):
    if epoch < 10:
        return 1e-3
    elif epoch < 12:
        return 1e-4
    else:
        return 1e-5

learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

### TensorFlow/Keras Custom Functions and Layers ###

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

def pseudo_inverse(A, rcond=1e-6):
    # Moore-Penrose pseudo-inverse based on Singular Value Decomposition (SVD)
    S, U, V = tf.linalg.svd(A)
    S_pinv  = tf.where(S > rcond * Mt * Nv , 1.0 / S, 0.0)   # Reciprocal of non-zero singular values
    S_pinv  = tf.linalg.diag(tf.cast(S_pinv, tf.complex64))
    A_pinv  = tf.linalg.matmul(V, tf.linalg.matmul(S_pinv, U, adjoint_b=True))
    return A_pinv

class Permute_and_Negate_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Permute_and_Negate_Layer, self).__init__(**kwargs)
    def call(self, inputs):
        permuted = tf.reverse(inputs, axis=[1])
        y = tf.concat([permuted[:, 0:1, :] * -1, permuted[:, 1:, :]], axis=1)
        return y

### PositionNet+ Post-Processsing ###

PositionNet = tf.keras.models.load_model("PositionNetPlus.keras",
                                         custom_objects={"Permute_and_Negate_Layer": Permute_and_Negate_Layer,
                                                         "Weighted_Binary_CrossEntropy": Weighted_Binary_CrossEntropy},
                                         compile=False,
                                         safe_mode=False)
PositionNet.trainable = False

class Post_Processing_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Post_Processing_Layer, self).__init__(**kwargs)

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

        # Phi_pinv = pseudo_inverse(Phi_I)                                # Moore-Penrose pseudo-inverse
        # h_ADD = tf.linalg.matmul(Phi_pinv, yDDx)                        # Least Squares (LS) solution        
        h_ADD = tf.linalg.lstsq(Phi_I, yDDx, l2_regularizer=1e-6)       # Least Squares (LS) solution
        h_ADD = tf.reshape(h_ADD, [N_batch, Nt, Nv, K])                 # Reshape to a 3D dense tensor
        h_ADD = tf.transpose(h_ADD, perm=[0, 3, 2, 1])
      
        index_3D = tf.expand_dims(index_Delay, axis=-1)                 
        index_batch = tf.range(tf.shape(index_3D)[0])[:, tf.newaxis, tf.newaxis]
        index_batch = tf.tile(index_batch, [1, K, 1])
        index_3D = tf.stack([index_batch, index_3D], axis=-1)
        index_3D = tf.reshape(index_3D, [N_batch, K, 2])                # Nonzero indices of 3D sparse channel matrix  
        
        HDD_real = tf.scatter_nd(index_3D, tf.math.real(h_ADD), [N_batch, Mt, Nv, Nt])   # Scatter into the 3D sparse channel tensor
        HDD_imag = tf.scatter_nd(index_3D, tf.math.imag(h_ADD), [N_batch, Mt, Nv, Nt])
        return HDD_real, HDD_imag

HDD_real, HDD_imag = Post_Processing_Layer()([PositionNet.output, PositionNet.input])    # Post-Processing Layer

# PositionNet+ Model with Post-Processing
PositionNet_Post_Processing = tf.keras.models.Model(PositionNet.input, [HDD_real, HDD_imag])
PositionNet_Post_Processing.summary()

HDD_real_train, HDD_imag_train = PositionNet_Post_Processing.predict(yDD_train)          # Inference

### SSRnet ###

def Conv3d_Block(x, filters):
    y = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding='same')(x)
    y = tf.keras.layers.LeakyReLU(negative_slope=0.10)(y)
    y = tf.keras.layers.LayerNormalization()(y)
    return y

# Model's Inputs
HDD_real = tf.keras.Input(shape=(Mt, Nv, Nt, 1))
HDD_imag = tf.keras.Input(shape=(Mt, Nv, Nt, 1))

# Real part of 3D channel matrix
HDD_OTFSr = Conv3d_Block(HDD_real,  Nc)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)

HDD_OTFSr = tf.keras.layers.Subtract()([HDD_real, HDD_OTFSr])

HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)
HDD_OTFSr = Conv3d_Block(HDD_OTFSr, Nc)
HDD_OTFSr = tf.keras.layers.Conv3D(1, (3, 3, 3), padding='same')(HDD_OTFSr)
CH_ADD_real = tf.keras.layers.Subtract()([HDD_real, HDD_OTFSr])

# Imaginary part of 3D channel matrix
HDD_OTFSi = Conv3d_Block(HDD_imag,  Nc)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)

HDD_OTFSi = tf.keras.layers.Subtract()([HDD_imag, HDD_OTFSi])

HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)
HDD_OTFSi = Conv3d_Block(HDD_OTFSi, Nc)
HDD_OTFSi = tf.keras.layers.Conv3D(1, (3, 3, 3), padding='same')(HDD_OTFSi)
CH_ADD_imag = tf.keras.layers.Subtract()([HDD_imag, HDD_OTFSi])

# 3D sparse channel matrix
CH_ADD = tf.keras.layers.Concatenate(axis=-1)([CH_ADD_real, CH_ADD_imag])

SSRnet = tf.keras.models.Model([HDD_real, HDD_imag], CH_ADD)

SSRnet.summary()

SSRnet.compile(optimizer=tf.keras.optimizers.AdamW(),
              loss=[log_NMSE],
              metrics=['mse'])

SSRnet_history = SSRnet.fit([HDD_real_train, HDD_imag_train], CH_ADD_train, validation_split=0.20, epochs=15, callbacks=[learning_rate_schedule])

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
