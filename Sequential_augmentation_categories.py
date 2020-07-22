# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:09:59 2020

@author: Spyros
"""

import numpy as np
import keras,glob,os
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img
import time

img_path = r"C:\path\to\img_data"
outpath = r"C:\path\to\aug_data"

filenames = glob.glob(img_path + "/*/*.jpg",recursive=True)
filenames = filenames+glob.glob(img_path + "/*/*.png",recursive=True)



for img in filenames:
    print("processing img No {} out of {}".format(filenames.index(img), len(filenames)))
    start=time.time()
    src_fname, ext = os.path.splitext(img)


    datagen = ImageDataGenerator(rotation_range=90,
                                    brightness_range=[0.6,1.4],
                                    vertical_flip=True,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest',
                                    channel_shift_range = 20)


    img = load_img(img)

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    img_name = src_fname.split('\\')[-1]
    new_dir = os.path.join(outpath, src_fname.split('\\')[-2])




    save_fname = new_dir

    i = 0
    for batch in datagen.flow (x, batch_size=32, save_to_dir = save_fname,
                               save_prefix = img_name, save_format='jpg'):
        i+=1
        if i>15:
            break

    end=time.time()
    print("Time needed    :",round(end-start,2))
