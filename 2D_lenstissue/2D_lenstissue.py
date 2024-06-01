# -*- coding: utf-8 -*-
"""
Created on Tue May 28 23:43:49 2024

@author: Administrator
"""

# %% 1
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import keras
# from keras import layers
from keras import backend as K
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
# latent_dim=16
# height=64
# width=64
# channels=3
from tensorflow.keras import regularizers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# %%
def ssim(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis = -1)))) / 2.303


# %% 4
from tensorflow.keras.layers import Layer
class Hadamard_i(Layer):
    def __init__(self, **kwargs):
        super(Hadamard_i, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',dtype='float32',
                                      shape=(2,108,720,720),# + input_shape[1:],
                                        # initializer='uniform',
                                        # initializer=keras.initializers.RandomUniform(minval=0., maxval=1.),
                                        initializer=keras.initializers.lecun_normal(seed=42),
                                        # initializer=keras.initializers.Ones(),
                                        # regularizer=linf_reg,
                                        # regularizer=tf.keras.regularizers.l1(1e-6),
                                        # activation='elu',#tf.keras.activations.elu(x, alpha=1.0),#'relu',
                                      trainable=True)
        super(Hadamard_i, self).build(input_shape)
    def call(self, x):
        x=tf.cast(x,tf.complex64)
        # xhadam=tf.Variable(tf.zeros([1,640,640,108],tf.float32))
        x=tf.transpose(x,[0,3,1,2])
        x2=tf.signal.fft2d(x[0,:,:,:])
        
        kernel=tf.complex(self.kernel[0,:,:,:],self.kernel[1,:,:,:])
        kernel=tf.cast(kernel,tf.complex64)
        xhadaf1=x2*tf.math.conj(kernel)
        xhada1=tf.signal.ifft2d(xhadaf1)
        xhadam=tf.abs(xhada1)
        xhadam=tf.transpose(xhadam,[1,2,0])
        xhadam=tf.expand_dims(xhadam,0)
        
        xhadam=tf.math.divide(xhadam,tf.math.reduce_max(xhadam))
        return tf.stack(xhadam)
    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return 
# %%
from tensorflow.keras.layers import Layer
class Hadamard_x(Layer):
    def __init__(self, **kwargs):
        super(Hadamard_x, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',#dtype='complex64',
                                      shape=(2,108,180,180),# + input_shape[1:],
                                        # initializer='uniform',
                                        # initializer=keras.initializers.RandomUniform(minval=0., maxval=1.),
                                        initializer=keras.initializers.Ones(),
                                        # initializer=keras.initializers.lecun_normal(seed=None),
                                        # activation='elu',#tf.keras.activations.elu(x, alpha=1.0),#'relu',
                                      trainable=True)
        self.pho = self.add_weight(name='pho',#dtype='complex64',
                                        # initializer='uniform',
                                        initializer=keras.initializers.RandomUniform(minval=0., maxval=0.1),
                                        # initializer=keras.initializers.lecun_normal(seed=None),
                                        # activation='elu',#tf.keras.activations.elu(x, alpha=1.0),#'relu',
                                      trainable=True)
        super(Hadamard_x, self).build(input_shape)
    def call(self, z,u,xi):
        zu=tf.math.subtract(z,u)
        zu=tf.math.multiply(zu,self.pho)
        # xi2=tf.image.resize(xi,)
        # print(xi.shape)
        xu=tf.math.add(xi,zu)
        
        x=tf.cast(xu,tf.complex64)
        
        # x=tf.cast(x,tf.complex64)
        # xhadam=tf.Variable(tf.zeros([1,640,640,108],tf.float32))
        x=tf.transpose(x,[0,3,1,2])
        x2=tf.signal.fft2d(x[0,:,:,:])
        
        kernel=tf.complex(self.kernel[0,:,:,:],self.kernel[1,:,:,:])
        kernel=tf.cast(kernel,tf.complex64)
        xhadaf1=x2*tf.math.conj(kernel)
        xhada1=tf.signal.ifft2d(xhadaf1)
        xhadam=tf.abs(xhada1)
        xhadam=tf.transpose(xhadam,[1,2,0])
        xhadam=tf.expand_dims(xhadam,0)
        
        xhadam=tf.math.divide(xhadam,tf.math.reduce_max(xhadam))
        
        return xhadam
    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return input_shape
# %%
reconM=keras.models.load_model('reconM_0308_z5', custom_objects={'Hadamard_i': Hadamard_i,'Hadamard_x': Hadamard_x,'ssim':ssim})

# %%
from scipy.io import loadmat
import mat73
datav1 = loadmat('data_2d_lenstissue.mat')
Xt=datav1['Xt']
Xt=Xt.astype('float32')
Yt=datav1['Yt']
Yt=Yt.astype('float32')
# %%
Y=np.zeros((15,132,132,108))
# %%
import scipy
rcof=0
vid=50
rmin=rcof
rmax=Y.shape[1]-rcof
cmin=rcof
cmax=Y.shape[2]-rcof
pid=0
temp=Xt[pid]
# temp=np.rot90(np.rot90(temp))
temp=np.expand_dims(temp,0)
generated_images=reconM.predict(temp)
# generated_images=generated_images[0]
plt.imshow(generated_images[0,rmin:rmax,cmin:cmax,vid],clim=(0,9e-1))
# %%
from scipy.io import savemat
savemat('gen_lenstissue.mat', {"generated_images": generated_images})



