# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:23:48 2020

@author: Spyros
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import random
import pickle
model = tf.keras.models.load_model(r"C:\path\to\model")

from PIL import Image
img = Image.open(test_file_path)
img.show()

#%%

#file to be tested

test_file_path=r"C:\path\to\testfile.jpg"

def prepare(file):
    IMG_SIZE = 360
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array=img_array/255
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

CATEGORIES =["1_Purchase_Rcpt", "2_Service_Rcpt", "3_Police_Rprt", "4_IBAN", "5_Ins_Contract", "6_Claim_Stmnt"]

#%%

image=prepare(test_file_path)

prediction = model.predict([image])

proba=model.predict_proba([image])

prediction = list(prediction[0])

print("Τhis is a : ", CATEGORIES[prediction.index(max(prediction))])

print("With probability", round((proba[0][prediction.index(max(prediction))]*100),3), "%")



