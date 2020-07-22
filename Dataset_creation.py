# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:57:39 2020

@author: Spyros
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:50:30 2020

@author: Spyros
"""
import winsound
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle


file_list = []
class_list = []

DATADIR = r"C:\path\to\img_data"

# All the categories you want your neural network to detect
CATEGORIES = ["1_Purchase_Rcpt", "2_Service_Rcpt", "3_Police_Rprt", "4_IBAN", "5_Ins_Contract", "6_Claim_Stmnt"]

# The size of the images that your neural network will use
IMG_SIZE = 50
# =============================================================================
#
# # Checking or all images in the data folder
# for category in CATEGORIES :
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         print(os.listdir(path).index(img), img)
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#
#
# =============================================================================
training_data = []

def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                print(os.listdir(path).index(img), img)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(os.listdir(path).index(img), img, message)
                pass

create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Creating the files containing all the information about your model
print("Saving X")
pickle_out = open("C:\Capstone\XsYs\X50.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
print("Saving y")
pickle_out = open("C:\Capstone\XsYs\y50.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# =============================================================================
# pickle_in = open("Xaug356.pickle", "rb")
# X = pickle.load(pickle_in)
#
# =============================================================================


winsound.Beep(700, 1000)
