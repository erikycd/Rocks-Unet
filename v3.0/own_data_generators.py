# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:30:59 2021

@author: Erik
"""
import os
import numpy as np
import cv2 as cv

def own_data_gen2(img_dir, label_dir, batch_size, input_size):
  list_images = os.listdir(img_dir)
  #random.shuffle(list_images) #Randomize the choice of batches
  ids_train_split = range(len(list_images))

  while True:

    for start in range(0, len(ids_train_split), batch_size):
      x_batch = []
      y_batch = []
      end = min(start + batch_size, len(ids_train_split))
      ids_train_batch = ids_train_split[start:end]

      for id in ids_train_batch:
        # img = cv.imread(os.path.join(img_dir, list_images[id]))
        # img = cv.resize(img, (input_size[0], input_size[1]))

        img = cv.imread(img_dir + list_images[id], cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (input_size[0], input_size[1]), interpolation = cv.INTER_NEAREST) # Resize
        img = (img - np.mean(img))/(np.std(img)) # Normalization

        #mask = cv.imread(os.path.join(label_dir, list_images[id].replace('jpg', 'png')), 0)
        mask = cv.imread(label_dir + list_images[id])[:,:,0]
        mask = cv.resize(mask, (input_size[0], input_size[1]), interpolation = cv.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=2)
        x_batch.append(img)
        y_batch.append(mask)

      x_batch = np.array(x_batch, np.float32)
      y_batch = np.array(y_batch, np.float32)

      yield x_batch, y_batch