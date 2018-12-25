from model import *
from IPython import embed
from keras.preprocessing.image import ImageDataGenerator

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')



# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()
img_gen = image_datagen.flow_from_directory('aplatam_data',class_mode=None)
mask_gen = mask_datagen.flow_from_directory('aplatam_data',class_mode=None)
train_gen = zip(img_gen, mask_gen)


model = unet()
preds = model.predict_generator(train_gen,steps=1)

model.fit_generator(train_gen, steps_per_epoch=int(1019/32), epochs = 5)


