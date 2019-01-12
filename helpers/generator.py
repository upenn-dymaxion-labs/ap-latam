import os
import pandas as pd
import numpy as np
import PIL
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
from helpers.util import *
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
from keras.preprocessing import image


class Generator(keras.utils.Sequence):

	
    def __init__(self,path,folderList,imgSize):
        
        self.path = path
        self.folderList = folderList
        self.imgSize = imgSize
        
        fileList = []
        
        for i in range(len(folderList)):
            files = os.listdir(os.path.join(path,folderList[i]))
            fileList += [os.path.join(path,folderList[i],x) for x in files]
            
        self.imgList = [x for x in fileList if "img" in x]  #Keeping only the RGB map images
        
        
    def __len__(self):
        
        return 30
    
    def __getitem__(self,index):
        
        pairs, targets = self.getBatch(batch_size)
        
        return pairs, targets
        

     
        

    def getBatch(self,batchSize):
              
        imgs=[]
        masks=[]
        
        selections = np.random.choice(len(self.imgList),batchSize,replace=False)
        
        for i in range(batchSize):
            
            img = read_img(self.imgList[selections[i]],self.imgSize)
            
            maskFileName = self.imgList[selections[i]].replace("img","mask")
            
            mask = read_img(maskFileName,self.imgSize,grayscale=True)
            
            imgs.append(img/255.0)
            mask = invertMask(mask/255.0)
            masks.append(mask.reshape(mask.shape[0],mask.shape[1],1))
        
        imgs = np.array(imgs)
        masks = np.array(masks)
            
          

        return( imgs, masks)
    
    def on_epoch_end(self):
        'Updates to be done after each epoch'
        a = 5
        
        
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.getBatch(batch_size)
            yield (pairs, targets)




class GeneratorWithAug(keras.utils.Sequence):

	def __init__(self,path,folderList,imgSize):
		
		self.path = path
		self.folderList = folderList
		self.imgSize = imgSize
		
		
		
		fileList = []
		
		for i in range(len(folderList)):
			files = os.listdir(os.path.join(path,folderList[i]))
			fileList += [os.path.join(path,folderList[i],x) for x in files]
			
		self.imgList = [x for x in fileList if "img" in x]  #Keeping only the RGB map images
		
		
	def __len__(self):
		
		return 30
	
	def __getitem__(self,index):
		
		pairs, targets = self.getBatch(batch_size)
		pair, target = self.Augument(pairs, targets)
		
		return pair, target
	
	def Augument(self, images, masks, Addk, Multiplyk, addNoise, invertPixel, addBlur, contrast, convolve, flip, transform, resize):
		Prob = np.random.random_sample()
		lenIndex = np.round(Prob*images.shape[0]).astype(int)
		index = np.unique(np.random.randint(images.shape[0], size = lenIndex))
		if Addk != None:
			images[index] = AddConstant(images[index], k = Addk)
		if Multiplyk != None:
			images[index] = MultiplyConstant(images[index], m = Multiplyk)
		if addNoise != None:
			images[index] = AddNoisy(images[index], noise_typ = addNoise)
		if invertPixel != None:
			images[index] = InvertPixel(images[index], P = invertPixel)
		if addBlur != None:
			images[index] = AddBlur(images[index], BlurType = addBlur)
		if contrast != None:
			images[index] = ImageContrastCorrect(images[index], cor = contrast)
		if convolve != None:
			images[index], masks[index] = ImageConv(images[index], masks[index], convType=convolve)
		if flip != None:
			images[index], masks[index] = imageFlip(images[index], masks[index], flipType = flip)
		if transform != None:
			images[index], masks[index] = imageTransform(images[index], masks[index], transType=transform)
		if resize != None:
			ima = []
			mas = []
			indexInv = [ i for i in range(images.shape[0]) if i not in index]
			for i in indexInv:
				ima.append(images[i])
				mas.append(masks[i])
			imag, maks = imageResize(images[index], masks[index], HW=resize)
			for i in range(len(imag)):
				ima.append(imag[i])
				mas.append(maks[i])
			return (ima, mas)
		
		
		
		return (images, masks)
			
			
			

	 
		

	def getBatch(self,batchSize):
			  
		imgs=[]
		masks=[]
		
		selections = np.random.choice(len(self.imgList),batchSize,replace=False)
		
		for i in range(batchSize):
			
			img = read_img(self.imgList[selections[i]],self.imgSize)
			
			maskFileName = self.imgList[selections[i]].replace("img","mask")
			
			mask = read_img(maskFileName,self.imgSize,grayscale=True)
			
			imgs.append(img/255.0)
			mask = invertMask(mask/255.0)
			masks.append(mask.reshape(mask.shape[0],mask.shape[1],1))
		
		imgs = np.array(imgs)
		masks = np.array(masks)
			
		  

		return( imgs, masks)
	
	def on_epoch_end(self):
		'Updates to be done after each epoch'
		a = 5
		
		
	def generate(self, batch_size, s="train", AddK = None, MultiplyK = None, AddNoise = None, InvertPixel = None, AddBlur = None, Contrast = None, Convolve = None, Flip = None, Transform = None, Resize = None):
		"""a generator for batches, so model.fit_generator can be used. """
		while True:
			pairs, targets = self.getBatch(batch_size)
			pair, target = self.Augument(pairs, targets, AddK, MultiplyK, AddNoise, InvertPixel, AddBlur, Contrast, Convolve, Flip, Transform, Resize)
			yield (pair, target)