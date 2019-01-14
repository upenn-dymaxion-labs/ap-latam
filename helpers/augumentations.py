## Add Constant value 
import numpy as np
import os
import pandas as pd
import numpy as np
import PIL
import matplotlib
from skimage import exposure
from scipy.ndimage import zoom
import scipy
import matplotlib.pyplot as plt
from PIL import Image
#%matplotlib inline
from keras.models import load_model,save_model
from keras.callbacks import ModelCheckpoint
from matplotlib.patches import Rectangle
import os
from skimage.transform import warp, AffineTransform
from scipy.misc import imsave
from tqdm import tqdm
import tensorflow as tf
import keras
import scipy.signal as ss
import cv2
from scipy import ndimage
from keras.metrics import binary_crossentropy
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
from skimage.transform import resize
from numpy import array
from keras import metrics
from skimage import transform as tf


def AddConstant(image, k = None):
	if k==None:
		k = 0
	image = image + k/255.0
	return image

def MultiplyConstant(image, m = None): 
	if m == None:
		m = 0
	image = image*(k/255.0)
	return image


# Add Noise)
def AddNoisy(image, noise_typ = None):
	if noise_typ == None:
		return image
	
	if noise_typ == "gauss":
		
		row,col,ch= image[0].shape
		mean = 0
		var = 0.05
		sigma = np.sqrt(var)
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = image[0].shape
		print('image shape', image.shape)
		s_vs_p = 0.5
		amount = 0.004
		out = image.copy()
		print('out shape', out.shape)

	  # Salt mode
		num_salt = np.ceil(amount * image[0].size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image[0].shape[:2]]
		out[:,coords[0], coords[1], :] = 1

	  # Pepper mode
		num_pepper = np.ceil(amount* image[0].size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
			  for i in image[0].shape[:2]]
		out[:,coords[0], coords[1], :] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col,ch = image[0].shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy
	
## Invert Random Pixel
def InvertPixel(image, P = None):
	if P == None:
		return image
	np.random.seed()
	imageC = image.copy()
	randP = np.random.random(imageC[0].shape[:2])
	imageC[:, randP>P] = 1.0 - imageC[:, randP>P]  
	return imageC

## apply blur to images
def BlurImage(image,k = None, BlurType = None):
	if BlurType == None:
		return image
	
	Blur = np.zeros(image.shape)
	if BlurType == 'Guass' and k == None:
		k = np.array([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
		k = k.reshape((1, k.shape[0], k.shape[1]))
		#k = np.tile(k, (3,1,1))
		k = k*(1.0/273)
		#k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
		for i in range(image.shape[3]):
			Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
		return np.clip(Blur, 0, 1)
	if BlurType == 'Average' and k == None:
		k = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
		k = k.reshape((1, k.shape[0], k.shape[1]))
		#k = np.tile(k, (3,1,1))
		#print('k', k.shape)
		k = k*(1.0/(k.shape[1]*k.shape[2]))
		for i in range(image.shape[3]):
			Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
		return np.clip(Blur, 0, 1)
		
	if BlurType == 'Median' and k == None:
		for i in range(image.shape[3]):
			Blur[:, :, :, i] = ss.medfilt(image[:, :, : , i], kernel_size = (1,5,5))
		return np.clip(Blur, 0, 1)
	if BlurType == 'MotionBlur' and k == None:
		size = 10
		k = np.zeros((size, size))
		k[int((size-1)/2), :] = np.ones(size)
		k = k / size
		k = k.reshape((1, k.shape[0], k.shape[1]))
		for i in range(image.shape[3]):
			Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
		return np.clip(Blur, 0, 1)
	elif k != None:
		k = k.reshape((1, k.shape[0], k.shape[1]))
		for i in range(image.shape[3]):
			Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
		return np.clip(Blur, 0, 1)

## Image Constrast Correction
def ImageContrastCorrect(image, cor = None):
	if cor == None:
		return image
	if cor == 'Gamma':
		corrected = exposure.adjust_gamma(image, 2)
	elif cor == 'Log':
		corrected = exposure.adjust_log(image, 1)
	return corrected

# Convolutions
def ImageConv(image, masks, convType = None, k = None):
	if convType == None:
		return image
	Convo = np.zeros(image.shape)
	ConvoM = np.zeros(masks.shape)
	if convType == 'Sharp' and k == None:
		k = np.array([[-0.00391, -0.01563, -0.02344, -0.01563, -0.00391],[-0.01563, -0.06250, -0.09375, -0.06250, -0.01563],[-0.02344, -0.09375, 1.85980, -0.09375, -0.02344], [-0.01563, -0.06250, -0.09375, -0.06250, -0.01563], [-0.00391, -0.01563, -0.02344, -0.01563, -0.00391]])
		k = k.reshape((1, k.shape[0], k.shape[1]))
		#k = np.tile(k, (3,1,1)
		#k = k*(1.0/273)
		#k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
		for i in range(image.shape[3]):
			Convo[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
			
		ConvoM[:, :, :, 0] = ndimage.convolve(image[:, :, : , 0], k, mode = 'constant', cval = 0.0)
		return np.clip(Convo, 0, 1), np.clip(ConvoM, 0, 1)
	if convType == 'Emboss' and k == None:
		k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
		k = k.reshape((1, k.shape[0], k.shape[1]))
		#k = np.tile(k, (3,1,1)
		#k = k*(1.0/273)
		#k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
		for i in range(image.shape[3]):
			Convo[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
		ConvoM[:, :, :, 0] = ndimage.convolve(image[:, :, : , 0], k, mode = 'constant', cval = 0.0)
		return np.clip(Convo, 0, 1), np.clip(ConvoM, 0, 1)
	if convType == 'Edge' and k == None:
		k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		k = k.reshape((1, k.shape[0], k.shape[1]))
		#k = np.tile(k, (3,1,1)
		#k = k*(1.0/273)
		#k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
		for i in range(image.shape[3]):
			Convo[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
		ConvoM[:, :, :, 0] = ndimage.convolve(image[:, :, : , 0], k, mode = 'constant', cval = 0.0)
		return np.clip(Convo, 0, 1), np.clip(ConvoM, 0, 1)

## flip images
def imageFlip(images, masks, flipType = None):
	if flipType == None:
		return images, masks
	
	if flipType == 'lr':
		images = images[:, :, ::-1, :]
		masks = masks[:, :, ::-1]
	if flipType == 'ud':
		images = images[:, ::-1, :, :]
		masks = masks[:, ::-1, :, :]
	return images, masks


## apply transform
def imageTransform(images, masks, transType = None):
	if transType == None:
		return images, masks
	zerI = np.zeros(images.shape)
	zerM = np.zeros(masks.shape)
	if transType == 'Affine':
		tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
						translation=(210, 50))
		xx, yy = np.meshgrid(np.arange(0, images[0].shape[1] - 1), np.arange(0, images[0].shape[0] - 1))
		coords = np.stack((xx.flatten(), yy.flatten()), axis = -1).transpose(1,0)
		coords = np.vstack((coords, np.ones(coords.shape[1]))).reshape(1, 3, -1)
		coords = coords.transpose(2,1,0)
		tformI = tform.inverse
		tformP = tform.params
		InvC = np.matmul(tformP, coords)

		X = np.clip(InvC[:, 0, :].astype(int), 0, 255)
		Y = np.clip(InvC[:, 1, :].astype(int), 0, 255)
		img1 = img[0]

		zerI = np.zeros(images.shape)
		zerI[:,xx.flatten(), yy.flatten(), :] = images[:,X.flatten(), Y.flatten(), :]
		zerM[:,xx.flatten(), yy.flatten(), :] = masks[:,X.flatten(), Y.flatten(), :]

#         for i in range(images.shape[0]):
#             zerI[i] = warp(images[i], tform.inverse, output_shape=(images.shape[1], images.shape[2]))
#             zerM[i] = warp(masks[i], tform.inverse, output_shape=(masks.shape[1], masks.shape[2]))
			
		return zerI, zerM
			
#Image resize and Crop
# def imageResize(images, masks, HW = None):

#     if HW == None:
#         H, W = 256, 256
#     if HW != None:
#         (H, W) = HW
#         for i in range(len(images)):
			
#             zerI = resize(images, (len(images), H, W), anti_aliasing=True)
#             zerM = resize(masks, (len(images), H, W), anti_aliasing=True)
#     return zerI, zerM
	
def zoomImage(img, masks, zoom_factor = None, **kwargs):

	s, h, w = img.shape[:3]
	chooseProb = np.random.random_sample()
	if chooseProb >= 0.5:
		zoom_factor = zoom_factor[1]
	else:
		zoom_factor = zoom_factor[0]

	# For multichannel images we don't want to apply the zoom factor to the RGB
	# dimension, so instead we create a tuple of zoom factors, one per array
	# dimension, with 1's for any trailing dimensions after the width and height.
	zoom_tuple = []
	for i in range(len(img.shape)):
		if i == 1 or i == 2:
			zoom_tuple.append(zoom_factor)
		else:
			zoom_tuple.append(1)
	
#     zoom_tuple = list(zoom_tuple)
#     zoom_tuple[0] = 1
	zoom_tuple= tuple(zoom_tuple)
	#print(zoom_tuple)

	# Zooming out
	if zoom_factor < 1:

		# Bounding box of the zoomed-out image within the output array
		zh = int(np.round(h * zoom_factor))
		zw = int(np.round(w * zoom_factor))
		top = (h - zh) // 2
		left = (w - zw) // 2

		# Zero-padding
		outI = np.zeros_like(img)
		outI[:, top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
		outM = np.zeros_like(masks)
		outM[:, top:top+zh, left:left+zw] = zoom(masks, zoom_tuple, **kwargs)
		

	# Zooming in
	elif zoom_factor > 1:

		# Bounding box of the zoomed-in region within the input array
		zh = int(np.round(h / zoom_factor))
		zw = int(np.round(w / zoom_factor))
		top = (h - zh) // 2
		left = (w - zw) // 2

		outI = zoom(img[:,top:top+zh, left:left+zw], zoom_tuple, **kwargs)
		outM = zoom(masks[:,top:top+zh, left:left+zw], zoom_tuple, **kwargs)

		# `out` might still be slightly larger than `img` due to rounding, so
		# trim off any extra pixels at the edges
		trim_top = ((outI.shape[1] - h) // 2)
		trim_left = ((outI.shape[2] - w) // 2)
		outI = outI[:,trim_top:trim_top+h, trim_left:trim_left+w]
		outM = outM[:,trim_top:trim_top+h, trim_left:trim_left+w]
		
		

	# If zoom_factor == 1, just return the input array
	else:
		outI = img
		outM = masks
	return outI, outM
	
	