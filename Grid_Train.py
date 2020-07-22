# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:21 2020

@author: Spyros
"""

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

import winsound
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau
import time
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.python.keras import backend as K
# ================



pickle_in = open("C:\path\to\X50.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("C:\path\to\y50.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [ 1, 2,3,4,5]
c_layer_sizes = [32, 64, 128]
h_layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]
dropout=[ 0.25, 0.5, 0.75 ]
optimizers=['adam',"RMSprop" ,'SGD']
activations=[ 'relu',"sigmoid"]


reduce_lr = ReduceLROnPlateau(
    monitor='val_sparse_categorical_accuracy', factor=0.95, patience=3, verbose=2, mode='auto',
    min_delta=0.01, cooldown=0, min_lr=0)

early_stop=tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', min_delta=0.01, patience=6, verbose=2, mode='auto',
    baseline=None)

c_w={0: y.count(0), 1:y.count(1), 2:y.count(2),3:y.count(3),4:y.count(4),5:y.count(5)}


for dense_layer in dense_layers:
    for c_size in c_layer_sizes:
        for conv_layer in conv_layers:
            for drop in dropout:
                for opt in optimizers:
                    for act in activations:
                        for h_size in h_layer_sizes:
                            NAME = "{}-conv--{}-Cnodes-{}-dense--{}-Hnodes--{}-drop--{}--{}-{}".format(conv_layer, c_size, dense_layer, h_size,drop, opt,act, int(time.time()))
                            print(NAME)

                            model = Sequential()

                            model.add(Conv2D(c_size, (3, 3), input_shape=X.shape[1:]))
                            model.add(Activation(act))
                            model.add(MaxPooling2D(pool_size=(2, 2)))

                            for l in range(conv_layer-1):
                                model.add(Conv2D(c_size, (3, 3)))
                                model.add(Activation(act))
                                model.add(MaxPooling2D(pool_size=(2, 2)))
                                model.add(Dropout(drop))

                            model.add(Flatten())

                            for _ in range(dense_layer):
                                model.add(Dense(h_size))
                                model.add(Activation(act))

                            model.add(Dense(6))
                            model.add(Activation('softmax'))

                            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                            model.compile(loss='sparse_categorical_crossentropy',
                                          optimizer=opt,
                                          metrics=['sparse_categorical_accuracy'],
                                          )

                            model.fit(X, y,
                                      batch_size=32,
                                      class_weight=c_w,
                                      epochs=50,
                                      validation_split=0.2,
                                      callbacks=[tensorboard,reduce_lr,early_stop])

                            model_json = model.to_json()
                            with open("model.json", "w") as json_file :
                                json_file.write(model_json)

                            model.save_weights(os.path.join(r"C:\path\to\models",(NAME+".h5")))
                            print("Saved model to disk")

                            model.save(os.path.join(r"C:\path\to\models",NAME))
                            K.clear_session()
                            tf.reset_default_graph()








