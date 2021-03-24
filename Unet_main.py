# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:15:18 2021

@author: Erik
"""
#%%

# Commented out IPython magic to ensure Python compatibility.
import os
os.environ['TF_KERAS'] = '1'

import random
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import cv2 as cv
import shutil, sys
import time

from tqdm import tqdm_notebook, tnrange
import skimage
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.config.list_physical_devices("GPU")


from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import layers, models, losses
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras
from tensorflow.keras import backend as K

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tensorflow.keras.__version__ )
print("Python", sys.version)

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import folders_masks
from folders_masks import make_folders, rmv_folders, build_mask

import data_generators
from data_generators import my_traingenerator, my_validgenerator, my_testgenerator

import unet_rocks
from unet_rocks import get_unet_rocks

import unet_bnsreenu
from unet_bnsreenu import multi_unet_model

import tensor_test
from tensor_test import get_tensor, get_tensor_orig

import metrics
from metrics import dice_coeff, jacard_coeff

import color_plot
from color_plot import mask_color_img


#%% READ DATASET

im_width = 256
im_height = 256

Path_images = 'C:/Users/Erik/Google Drive/Images/Rocks_CT/Images/'
Path_mask_1 = 'C:/Users/Erik/Google Drive/Images/Rocks_CT/Macro/'
Path_mask_2 = 'C:/Users/Erik/Google Drive/Images/Rocks_CT/Micro/'

# Path_images = 'C:/Users/Erik/Google Drive/Images/MICCAI MRI/MICCAI MRI/Imagenes/'
# Path_mask_1 = 'C:/Users/Erik/Google Drive/Images/MICCAI MRI/MICCAI MRI/VI/'
# Path_mask_2 = 'C:/Users/Erik/Google Drive/Images/MICCAI MRI/MICCAI MRI/VD/'

# Path_images = 'C:/Users/Erik/Google Drive/Images/Mascaras ventriculares/images/'
# Path_mask_1 = 'C:/Users/Erik/Google Drive/Images/Mascaras ventriculares/Anotacion ventriculo izquierdo 2/'
# Path_mask_2 = 'C:/Users/Erik/Google Drive/Images/Mascaras ventriculares/Anotacion ventriculo derecho 2/'

ids_images = sorted(os.listdir(Path_images))
ids_mask_1 = sorted(os.listdir(Path_mask_1))
ids_mask_2 = sorted(os.listdir(Path_mask_2))

print("Total of images :", len(ids_images))
print("Total of masks 1:", len(ids_mask_1))
print("Total of masks 2:", len(ids_mask_2))

Sample_image = cv.imread(Path_images + ids_images[0])
Sample_mask_1 = cv.imread(Path_mask_1 + ids_mask_1[0])
Sample_mask_2 = cv.imread(Path_mask_2 + ids_mask_2[0])

print('Sample image:',Sample_image.shape, Sample_image.min(), Sample_image.max(), 
      Sample_image.dtype)
print('Sample mask 1:',Sample_mask_1.shape, Sample_mask_1.min(), Sample_mask_1.max(),
      Sample_mask_1.dtype)
print('Sample mask 2:',Sample_mask_2.shape, Sample_mask_2.min(), Sample_mask_2.max(),
      Sample_mask_2.dtype)

print('Files images:',ids_images[0:5])
print('Files mask 1:',ids_mask_1[0:5])
print('Files mask 2:',ids_mask_2[0:5])


#%% SEPARATE DATABASE IN [TRAIN,VALID,TEST]

if os.path.isdir('/tmp_dataset/train_imgs/'):
  rmv_folders()

make_folders()

seed = 0

X = list(range(0, len(ids_images)))
Y = X

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, shuffle=True, 
                                                      random_state = seed)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size = 0.2, shuffle=True, 
                                                    random_state = seed*2)

data_train = []
for i in X_train:
  I = cv.imread(Path_images + ids_images[i])
  md = cv.imread(Path_mask_1 + ids_mask_1[i])
  mi = cv.imread(Path_mask_2 + ids_mask_2[i])
  mask = build_mask(md,mi)
  skimage.io.imsave('/tmp_dataset/train_masks/data/' + ids_images[i], mask, check_contrast=False)
  skimage.io.imsave('/tmp_dataset/train_imgs/data/' + ids_images[i], I, check_contrast=False)
  data_train.append([ ids_images[i] , ids_mask_1[i] , ids_mask_2[i] ])
  
data_train = pd.DataFrame(data_train , columns = ["Image","Mask 1","Mask 2"])

data_valid = []
for i in X_valid:
  I = cv.imread(Path_images + ids_images[i])
  md = cv.imread(Path_mask_1 + ids_mask_1[i])
  mi = cv.imread(Path_mask_2 + ids_mask_2[i])
  mask = build_mask(md,mi)
  skimage.io.imsave('/tmp_dataset/valid_masks/data/' + ids_images[i], mask, check_contrast=False)
  skimage.io.imsave('/tmp_dataset/valid_imgs/data/' + ids_images[i], I, check_contrast=False)
  data_valid.append([ ids_images[i] , ids_mask_1[i] , ids_mask_2[i] ])

data_valid = pd.DataFrame(data_valid , columns = ["Image","Mask 1","Mask 2"])

data_test = []
for i in X_test:
  I = cv.imread(Path_images + ids_images[i])
  md = cv.imread(Path_mask_1 + ids_mask_1[i])
  mi = cv.imread(Path_mask_2 + ids_mask_2[i])
  mask = build_mask(md,mi)
  skimage.io.imsave('/tmp_dataset/test_masks/data/' + ids_images[i], mask, check_contrast=False)
  skimage.io.imsave('/tmp_dataset/test_imgs/data/' + ids_images[i], I, check_contrast=False)
  data_test.append([ ids_images[i] , ids_mask_1[i] , ids_mask_2[i] ])
  
data_test = pd.DataFrame(data_test , columns = ["Image","Mask 1","Mask 2"])


#%% PRINTING FILE SIZE

print("Training images:", len(os.listdir('/tmp_dataset/train_imgs/data/')))
print("Training masks:", len(os.listdir('/tmp_dataset/train_masks/data/')))

print("Validation images:", len(os.listdir('/tmp_dataset/valid_imgs/data/')))
print("Validation masks:", len(os.listdir('/tmp_dataset/valid_masks/data/')))

print("Testing images:", len(os.listdir('/tmp_dataset/test_imgs/data/')))
print("Testing masks:", len(os.listdir('/tmp_dataset/test_masks/data/')))

ids = next(os.walk('/tmp_dataset/train_imgs/data/'))[2]

Sample_image = cv.imread('/tmp_dataset/train_imgs/data/' + ids[0] )
Sample_mask = cv.imread('/tmp_dataset/train_masks/data/' + ids[0] )

print('Sample training image:',Sample_image.shape, Sample_image.min(), Sample_image.max(), 
      Sample_image.dtype)
print('Sample training mask:',Sample_mask.shape, Sample_mask.min(), Sample_mask.max(),
      Sample_mask.dtype)

#%% PLOTING SAMPLE FROM DATA GENERATORS

batch_size = 4
target_size = 256
image_batch, mask_batch = next(my_traingenerator(target_size,batch_size))

print('Image batch size type: ', image_batch.shape, image_batch.dtype, image_batch.min(), image_batch.max())
print('Mask batch size type: ',mask_batch.shape, mask_batch.dtype, mask_batch.min(), mask_batch.max())
plt.figure(figsize=(15,15))

label = 0
plt.subplot(1,2,1)
b = image_batch[label,:,:,0]
print('Image type: ',b.shape, b.dtype, b.min(), b.max())
plt.imshow(b, cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
a = mask_batch[label,:,:,0]
print('Mask type: ',a.shape, a.dtype, a.min(), a.max())
plt.imshow(np.uint8(a), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.show()

#%% GET UNET

def get_model():
    return get_unet_rocks()

model = get_model()

model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.005), 
              loss = losses.sparse_categorical_crossentropy,
              metrics = ['sparse_categorical_accuracy']
              )

#%% TRAIN UNET

epochs = 2
target_size = 256

callbacks = [
    #EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    ModelCheckpoint('model-Unet_multiclass_1.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

start = time.time()
historial = model.fit_generator(my_traingenerator(target_size , batch_size),
                      epochs = epochs,
                      steps_per_epoch = int(np.fix(len(os.listdir('/tmp_dataset/train_imgs/data')) / float(batch_size)) ),
                      validation_data = my_validgenerator(target_size , batch_size), 
                      validation_steps = int(np.fix(len(os.listdir('/tmp_dataset/valid_imgs/data')) / float(batch_size)) ),
                      callbacks = callbacks)

end = time.time()
tiempo = end - start

print(tiempo)

#%% PLOTTING CURVES AND SAVING HISTORY

plt.figure(figsize=(7,5))
plt.title("Learning curve")
plt.plot(historial.history["loss"], label="train_loss")
plt.plot(historial.history["val_loss"], label="val_loss")
plt.plot( np.argmin(historial.history["val_loss"]), np.min(historial.history["val_loss"]), 
         marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();

plt.figure(figsize=(7,5))
plt.title("Learning curve")
plt.plot(historial.history['sparse_categorical_accuracy'], label='train_accuracy')
plt.plot(historial.history['val_sparse_categorical_accuracy'], label= 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend();

hist_df = pd.DataFrame(historial.history) 
hist_csv_file = 'Results.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


#%% DO INFERENCE ON TEST DATA

x_test, y_test = get_tensor('/tmp_dataset/test_imgs/data/','/tmp_dataset/test_masks/data/', 256 , 256)
print('x_test:',x_test.shape, x_test.dtype, x_test.min(), x_test.max())
print('y_test:',y_test.shape, y_test.dtype, y_test.min(), y_test.max())

# Batch normalization x_eval
x_test_norm = (x_test - np.mean(x_test))/(np.std(x_test))

print('x_test_norm:',x_test_norm.shape, x_test_norm.dtype, x_test_norm.min(), x_test_norm.max())
print('y_test:',y_test.shape, y_test.dtype, y_test.min(), y_test.max())

# Evaluate on validation set (this must be equals to the best log_loss)
#model.evaluate(my_validgenerator(), verbose=1)

predicted_label = model.predict(x_test_norm)
print('Predicted label shape: ',predicted_label.shape, predicted_label.dtype, predicted_label.min(), predicted_label.max())
predicted_mask = to_categorical(np.argmax(predicted_label, -1), dtype='uint8')
print('Predicted mask shape: ',predicted_mask.shape, predicted_mask.dtype, predicted_mask.min(), predicted_mask.max())
y_true = to_categorical(y_test)
print('Y_true shape: ',y_true.shape, y_true.dtype, y_true.min(), y_true.max())

#%% PLOTTING RESULTS 1

plt.figure(figsize=(15,15))

label = 0
plt.subplot(1,3,1)
a = x_test[label,:,:,0]
print('Imagen: ',a.shape, a.dtype, a.min(), a.max())
plt.imshow(a, cmap='gray')
plt.title("Image")
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
b = y_test[label,:,:,0]
print('Mask: ',b.shape, b.dtype, b.min(), b.max())
plt.imshow(b, cmap='gray')
plt.title("Annotations")
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
c = predicted_mask[label,:,:,1] + 2.*(predicted_mask[label,:,:,2])
print('Predicted mask: ',c.shape, c.dtype, c.min(), c.max())
plt.imshow(c, cmap='gray')
plt.title("Segmentation")
plt.xticks([])
plt.yticks([])

plt.show()

#%% COMPUTING METRICS

Dice_class_1, Dice_class_2 = dice_coeff(y_true,predicted_mask)

print('Shape of true labels:',y_true.shape,y_true.min(),y_true.max())
print('Shape of predicted labels:',predicted_mask.shape,predicted_mask.min(),predicted_mask.max())

print('Dice mean class 1:',Dice_class_1.mean())
print('Dice per image_class_1:',Dice_class_1)

print('Dice mean class 2:',Dice_class_2.mean())
print('Dice per image_class_2:',Dice_class_2)

Jacard_class_1 , Jacard_class_2 = jacard_coeff(y_true,predicted_mask)

print('Shape of true labels:',y_true.shape,y_true.min(),y_true.max())
print('Shape of predicted labels:',predicted_mask.shape,predicted_mask.min(),predicted_mask.max())

print('Jacard mean class 1:',Jacard_class_1.mean())
print('Jacard per image_class_1:',Jacard_class_1)

print('Jacard mean class 2:',Jacard_class_2.mean())
print('Jacard per image_class_2:',Jacard_class_2)


#%% COLOR PLOT

label = 4
a = x_test[label,:,:,0]
a = ( a- a.min())/(a.max() - a.min())
Image_rgb = np.repeat(a[:, :, np.newaxis], 3, axis=2)

true_class_1 = y_test[label,:,:,0]==1
true_class_2 = y_test[label,:,:,0]==2

True_mask = mask_color_img(Image_rgb,true_class_1,[0,255,255],
                           true_class_2,[100,0,200],alpha = 0.001)


predicted_class_1 = predicted_mask[label,:,:,1]==1
predicted_class_2 = predicted_mask[label,:,:,2]==1

Predicted_mask = mask_color_img(Image_rgb,predicted_class_1,[0,255,255],
                                predicted_class_2,[100,0,200],alpha = 0.001)


plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
plt.imshow(True_mask)
plt.title("True mask")
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(Predicted_mask)
plt.title("Predicted mask")
plt.xticks([])
plt.yticks([])












