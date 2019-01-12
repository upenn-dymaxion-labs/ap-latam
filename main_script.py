from shapely.geometry import box, shape, mapping
import rasterio
import rasterio.mask
import fiona
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from helpers.io_management import all_raster_files
from helpers.transformations import *
from helpers.plotting import getShapeCoords, plotShapesOnRaster
from helpers.masking import getBinaryMask, createMask
from helpers.generator import Generator, GeneratorWithAug
from helpers.lossFunctions import *
from helpers.evaluateImage import evaluateImage
from helpers.augumentations import *
from helpers.util import *
from models.model import unet

import keras
import os
import pandas as pd
import numpy as np
import PIL
import matplotlib
from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline
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
from keras.metrics import binary_crossentropy
import keras.backend as K
from math import sqrt
from keras.callbacks import History 
import gc
from skimage.transform import resize
from numpy import array
from keras import metrics
from skimage import transform as tf
from keras.preprocessing import image
# Main Script

#e = all_raster_files(dirname = '/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/')
#dataset = rasterio.open(e[0])

#vector = fiona.open('/Users/dishajindal/Documents/ap-latam/aplatam_data/montevideo_2018_valid.shp')

#north_up = dataset.transform.e <= 0
#rotated = dataset.transform.b != 0 or dataset.transform.d != 0
#matchingShapes = getOverlappingShapes(dataset, vector)
#len(matchingShapes)
#for i in range(len(matchingShapes)):
#    poly = matchingShapes[i][2]
#    fig = plt.figure() 
#    ax = fig.gca() 
#    ax.add_patch(PolygonPatch(poly, fc='#6699cc', ec='#6699cc', alpha=0.5, zorder=2 ))
#    ax.axis('scaled')
#    plt.show()

#generateWindowsWithMasksForRasters("/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/", "Patches", vector)

# create Actual Binary master for a raster
#raster = rasterio.open("/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/17MAY17134609_P006_R3C2.tif")
#matchingShapes = getOverlappingShapes(raster, vector, returnCoord = True)
#b_mask = getBinaryMask(raster, matchingShapes)
#imsave("17MAY17134609_P006_R3C2_actual_mask.png", b_mask.astype('float64'))

# Generate Masks
#generateWindowsWithMasks(rasterio.open("/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/17MAY17134609_P006_R3C2.tif"), vector, "Patches")

# Join Masks to create one output
#createOutputForRaster("Patches", "/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/", "17MAY17134609_P006_R3C2.tif")

dataPath = '/media/chetan/de104728-db4d-4432-a0d9-ddcfc771ae84/DymaxionLabs/ap-latam/aplatam/console/Patches'

imgSize = (256,256)
batchSize = 6

trainFolders = ['17JUN10140042_P003_R2C2','16AUG22134016_P004_R1C1','17JAN07134545_P002_R2C1']
testFolders = ['17JUN10140042_P003_R1C1']

gen = Generator(dataPath,trainFolders,imgSize)
testGen  = Generator(dataPath,testFolders, imgSize)

# img,mask = gen.generate(1).__next__()

# plt.imshow(img[:,:,:,2].reshape(256,256))

# plt.show()

epochs=3

model = unet(lossFunc = jaccard_distance_loss)
model.fit_generator(gen.generate(4),epochs=epochs,verbose=1,validation_data=testGen.generate(5),validation_steps = 5,steps_per_epoch = 100)

pred,mask,img = evaluateImage(gen, model)

plt.show()