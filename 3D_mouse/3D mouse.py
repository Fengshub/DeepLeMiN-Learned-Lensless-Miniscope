# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:24:03 2024

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
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# import layers.Lambda as Lambda
of=24
# %%
def rnorm(ip):
    # x[0]=x[0]/K.max(x[1])
    x=ip[0]
    y=ip[1]
    # print(K.max(y))
    # res=x/K.max(y)
    # res=K.abs(x)/K.max(x)
    res=(x-K.min(x))/K.max(K.abs(x))
    # res=(x-K.mean(x)+K.std(x))/K.max(K.abs(x))
    return res
# %%
"""
metric functions PNSR and SSIM
"""
def ssimmfov(y_true, y_pred):
    y_true=y_true[0]
    y_pred=y_pred[0]
    y_true=tf.expand_dims(y_true,3)
    y_pred=tf.expand_dims(y_pred,3)
    # return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    # return 1-tf.reduce_mean(tf.image.ssim(tf.expand_dims(y_true,3), tf.expand_dims(y_pred,3), 1.0))
    return 1-tf.image.ssim(y_true,y_pred,1.0)
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis = -1)))) / 2.303
# %%
lidn=6
# %% 4
from tensorflow.keras.layers import Layer
class Hadamard_i(Layer):
    def __init__(self, **kwargs):
        super(Hadamard_i, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',#dtype='complex64',
                                      shape=(2,lidn,1408,1408),
                                      # shape=(2,) + input_shape[1:],
                                       # initializer='uniform',
                                       # initializer=keras.initializers.RandomUniform(minval=0., maxval=1.),
                                       initializer=keras.initializers.lecun_normal(seed=None),
                                       # activation='elu',#tf.keras.activations.elu(x, alpha=1.0),#'relu',
                                      trainable=True)
        super(Hadamard_i, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        xin=tf.cast(x,tf.complex64)
        x2=tf.signal.fft2d(xin[0,:,:,:])
        kernelc=tf.complex(self.kernel[0,:,:,:],self.kernel[1,:,:,:])
        kernel=tf.cast(kernelc,tf.complex64)
        xhadaf=x2*tf.math.conj(kernel)
        
        xhada=tf.signal.ifft2d(xhadaf)
        # xhada=tf.signal.fftshift(xhada)
        xhada2=tf.abs(xhada)
        # xhada2=tf.math.divide(xhada2,tf.math.reduce_max(xhada2))
        
        
        xhada3=tf.cast(xhada2,tf.complex64)
        xhada3f=tf.signal.fft2d(xhada3)
        xrawf=xhada3f*kernel
        xraw=tf.abs(tf.signal.ifft2d(xrawf))
        
        xhada2e=tf.expand_dims(xhada2,0)
        xhada2ee=tf.expand_dims(xhada2e,4)
        # xhada2=tf.math.divide(xhada2,tf.math.reduce_max(xhada2))
        
        xrawe=tf.expand_dims(xraw,0)
        xrawee=tf.expand_dims(xrawe,4)
        # xraw=tf.math.divide(xraw,tf.math.reduce_max(xraw))
        
        return [xhada2ee,xrawee]
        # return xhada2
    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return input_shape
# %%
from tensorflow.keras.layers import Layer
class Hadamard_x(Layer):
    def __init__(self, **kwargs):
        super(Hadamard_x, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',#dtype='complex64',
                                      shape=(2,lidn,13,416,416),# + input_shape[1:],
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
        super(Hadamard_x, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, z,u,xi):
        zus=tf.math.subtract(z,u)
        zu=tf.math.multiply(zus,self.pho)
        # xi2=tf.image.resize(xi,)
        # print(xi.shape)
        xu=tf.math.add(xi,zu)
        
        xc=tf.cast(xu,tf.complex64)
        
        # x=tf.cast(x,tf.complex64)
        # xhadam=tf.Variable(tf.zeros([1,640,640,108],tf.float32))
        x=tf.transpose(xc,[0,1,4,2,3])
        x2=tf.signal.fft2d(x[0,:,:,:,:])
        
        kernelc=tf.complex(self.kernel[0],self.kernel[1])
        kernel=tf.cast(kernelc,tf.complex64)
        xhadaf1=x2*tf.math.conj(kernel)
        xhada1=tf.signal.ifft2d(xhadaf1)
        xhadam=tf.abs(xhada1)
        xhadamt=tf.transpose(xhadam,[0,2,3,1])
        xhadamte=tf.expand_dims(xhadamt,0)
        
        xhadamted=tf.math.divide(xhadamte,tf.math.reduce_max(xhadamte))
        
        return xhadamted
    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return input_shape
# %%
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def c_o4(y_pred,y_true):
    y_preds=tf.squeeze(y_pred)
    y_predsn=rnorm([y_preds,y_preds])
    y_trues=tf.squeeze(y_true)
    y_truesn=rnorm([y_trues,y_trues])
    
    y_prede=tf.expand_dims(y_predsn,4)
    y_truee=tf.expand_dims(y_truesn,4)


    y_true1=tf.reduce_sum(y_truee, 1)
    y_true1d=y_true1/tf.reduce_max(y_true1)
    # y_true1=tf.transpose(y_true1,[0,2,1,3])
    # y_true1dr=tf.reshape(y_true1d,[y_true1d.shape[1]*lidn,y_true1d.shape[2],1])
    y_pred1=tf.reduce_sum(y_prede, 1)
    y_pred1d=y_pred1/tf.reduce_max(y_pred1)
    # y_pred1=tf.transpose(y_pred1,[0,2,1,3])
    # y_pred1dr=tf.reshape(y_pred1d,[y_pred1d.shape[1]*lidn,y_pred1d.shape[2],1])
    loss1=1-tf.image.ssim(y_true1d,y_pred1d,1)
    # loss1=bce(y_true1,y_pred1)
    # loss1=K.mean(K.square(y_true1 - y_pred1))
    # loss1=loss1/tf.reduce_max(loss1)

    y_true2=tf.reduce_sum(y_truee, 2)
    y_true2d=y_true2/tf.reduce_max(y_true2)
    # y_true2=tf.transpose(y_true2,[0,2,1,3])
    # y_true2dr=tf.reshape(y_true2d,[y_true2d.shape[1]*lidn,y_true2d.shape[2],1])
    y_pred2=tf.reduce_sum(y_prede, 2)
    y_pred2d=y_pred2/tf.reduce_max(y_pred2)
    # y_pred2=tf.transpose(y_pred2,[0,2,1,3])
    # y_pred2dr=tf.reshape(y_pred2d,[y_pred2d.shape[1]*lidn,y_pred2d.shape[2],1])
    loss2=1-tf.image.ssim(y_true2d,y_pred2d,1)
    # loss2=bce(y_true2,y_pred2)
    # loss2=K.mean(K.square(y_true2 - y_pred2))
    # loss2=loss2/tf.reduce_max(loss2)

    y_true3=tf.reduce_sum(y_truee, 3)
    y_true3d=y_true3/tf.reduce_max(y_true3)
    # y_true3=tf.transpose(y_true3,[0,2,1,3])
    # y_true3dr=tf.reshape(y_true3d,[y_true3d.shape[1]*lidn,y_true3d.shape[2],1])
    y_pred3=tf.reduce_sum(y_prede, 3)
    y_pred3d=y_pred3/tf.reduce_max(y_pred3)
    # y_pred3=tf.transpose(y_pred3,[0,2,1,3])
    # y_pred3dr=tf.reshape(y_pred3d,[y_pred3d.shape[1]*lidn,y_pred3d.shape[2],1])
    loss3=1-tf.image.ssim(y_true3d,y_pred3d,1)
    loss=loss1+loss2+loss3
    loss=tf.math.reduce_mean(loss)
    return loss#+loss4#+lossbce#loss1+loss2
# %%
reconM_g704_z5=keras.models.load_model('reconM_g704_z5_v4', custom_objects={'Hadamard_i': Hadamard_i,'Hadamard_x': Hadamard_x,'c_o4': c_o4})




# %% 7 datasets
from scipy.io import loadmat
import mat73
datav1=mat73.loadmat("t_img_recd_video0003 24-04-04 18-31-11_abetterrecordlong_03560_1_290.mat")
Xt=datav1['Xts']
Xt=Xt.astype('float32')
# Xt=Xt[0:1,:,:,72:73]
Xt=Xt/255
# Xt=Xt-0.45;
# Xt[Xt<0]=0;
# %%
vfn=290
# %%
Xt=np.swapaxes(Xt,1,3)
Xt=np.swapaxes(Xt,2,3)
Xt=np.expand_dims(Xt,4)
# Y=np.swapaxes(Y,1,3)
# Y=np.swapaxes(Y,2,3)
Xt=Xt[:,0:lidn,:,:,:]
# Y=Y[:,0:3,:,:,:]
# X=tf.transpose(X,[0,3,1,2])
# Y=tf.transpose(Y,[0,3,1,2,4])
# %%
Y=np.zeros((10,6,416,416,13))
# %%
'''
reconstruction of mouse in vivo imaging video, batch size = 1 (frame-by-frame)
'''
import scipy
pid=0
std1=1
std2=1
rcof=0
rmin=rcof
rmax=Y.shape[2]-rcof
cmin=rcof
cmax=Y.shape[3]-rcof
gamma=1
vid=4

xflag=1
yflag=1
gimid=1

generated_images_f=np.zeros((vfn,6,416,416,13))
for pid in range(0,vfn):
    print(pid)
    temp=Xt[pid]**gamma
    temp=temp/np.max(temp)
    temp=np.expand_dims(temp,0)

    generated_images=reconM_g704_z5.predict([temp])
    generated_images=generated_images[1]
    generated_images_f[pid]=generated_images

# %%
# results = reconM_g704_z5.predict(Xt[1:10],batch_size=9)
fig=plt.figure(figsize=(4*2,3*2))
vid=0
# generated_images=reconM_g704_z5.predict([temp])
# if gimid==1:
    # generated_images=generated_images[1]
for idx in range(0,12):
    # plt.figure(idx)
    ax=plt.subplot(3,4,idx+1)
    tempr=generated_images_f[100,vid,rmin:rmax,cmin:cmax,idx]/np.max(generated_images[0,:,rmin:rmax,cmin:cmax,:])
    # temp=(generated_images[0,rmin:rmax,cmin:cmax,:]/np.max(generated_images[0,rmin:rmax,cmin:cmax,:]))**gamma
    # plt.imshow(tempr,clim=(np.mean(temp)-std1*np.std(temp),np.mean(temp)+std2*np.std(temp)))
    plt.imshow(tempr,clim=(0,1))
# %%
generated_images_fu=(generated_images_f*255).astype('uint8')
# %% save reconstructed 3D video
from scipy.io import savemat
# savemat('weights73icp2.mat', {"weights": weights})
# savemat('gen_img_recd_video0003 24-04-04 18-31-11_abetterrecordlong_03560_1_290_v4.mat', {"generated_images_fu": generated_images_fu})



