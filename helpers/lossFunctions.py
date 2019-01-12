import os
import pandas as pd
import numpy as np
import PIL
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.models import load_model,save_model
from keras.callbacks import ModelCheckpoint
from matplotlib.patches import Rectangle
import os
from scipy.misc import imsave
from tqdm import tqdm
import tensorflow as tf
import keras
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam,Adagrad
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from math import sqrt
from keras.callbacks import History 
from keras.optimizers import Adam, SGD
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, merge, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.layers import BatchNormalization
from keras.models import Model
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ELU
import keras.backend as K
from math import sqrt
from keras.callbacks import History 
import gc
from sklearn import metrics as mt

def Cross_Entropy_Loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #intersection = y_true_f * y_pred_f
    #
    return K.mean(-(2.*y_true_f*K.log(y_pred_f + K.epsilon()) + (1 - y_true_f)*K.log(1. - y_pred_f + K.epsilon())))
#     else:
#         return -K.log(1. - y_pred)

def f1(y_true, y_pred):
    y_trueF = K.flatten(y_true)
    y_predF = K.flatten(y_pred)
    y_pred = K.round(y_predF)
    tp = K.sum(K.cast(y_trueF*y_predF, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_trueF)*(1-y_predF), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_trueF)*y_predF, 'float'), axis=0)
    fn = K.sum(K.cast(y_trueF*(1-y_predF), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    y_trueF = K.flatten(y_true)
    y_predF = K.flatten(y_pred)
    
    tp = K.sum(K.cast(y_trueF*y_predF, 'float'))
    #print('tp', K.eval(tp))
    tn = K.sum(K.cast((1-y_trueF)*(1-y_predF), 'float'))
    fp = K.sum(K.cast((1-y_trueF)*y_predF, 'float'))
    fn = K.sum(K.cast(y_trueF*(1-y_predF), 'float'))

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2.*p*r / (p+r+K.epsilon())
    #print('f1', K.eval(f1))
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    #print('f1_after', K.eval(f1))
    return 1. - K.mean(f1)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    y_trueF = K.flatten(y_true)
    y_predF = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_trueF * y_predF), axis = 0)
    sum_ = K.sum(K.abs(y_trueF) + K.abs(y_predF), axis = 0)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
