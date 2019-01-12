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

def evaluateImage(generator, model):
    
    img,mask = generator.generate(1).__next__()
    
    pred = model.predict(img)
    scores = model.evaluate(img,mask)
    print("Model scores : {}".format(scores))
    #print("Predicted entropy : {}".format(getCrossEntropy(mask=mask,pred=pred)))
    #print("Predicted mask : {}".format(getDice(mask,pred)))
    
    plt.figure(figsize=(20,10))
    plt.subplot(131)
    plt.imshow(pred.reshape(256,256),cmap='gray')
    
    plt.subplot(132)
    plt.imshow(mask.reshape(256,256),cmap='gray')
    
    plt.subplot(133)
    plt.imshow(img[:,:,:,2].reshape(256,256))
    
    return(pred,mask,img)
    