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

ep=50
opt = "adam"
pool=(3,3)
convWin=(3,3)
drop=0.25
categories=["Service_Rec","IBAN","Claims_st", "Insurance_Contract","Police_Report","Purchase_Rec"]



for cat in categories:


    NAME="Binary_{},{}".format(cat,int(time.time()))
    tensorboard=TensorBoard(log_dir="logs3/{}".format(NAME))
    print(NAME)
    # Opening the files about data
    X = pickle.load(open(r"C:\path\to\X_360_Bin_{}.pickle".format(cat), "rb"))
    y = pickle.load(open(r"C:\path\to\y_360_Bin_{}.pickle".format(cat), "rb"))
    c_w={0: y.count(0), 1:y.count(1)}

    print ("IS_{}: {}, is_NOT_{}: {}".format(cat, y.count(0),cat,y.count(1)))

    # normalizing data (a pixel goes from 0 to 255)
    X = X/255.0

    # Building the model
    model = Sequential()
    # 3 convolutional layers
    model.add(Conv2D(64, convWin, input_shape = X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool))


    model.add(Conv2D(128,convWin))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool))

    model.add(Conv2D(256,convWin))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool))
    model.add(Dropout(drop))




    # 4 hidden layers
    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(32))
    model.add(Activation("relu"))

    # The output layer with 2 neurons, for 2 classes positive/negative
    model.add(Dense(2))
    model.add(Activation("softmax"))

    opt = "adam" #tf.keras.optimizers.Adam()

    reduce_lr = ReduceLROnPlateau(
        monitor='val_sparse_categorical_accuracy', factor=0.90, patience=2, verbose=2, mode='auto',
        min_delta=0.01, cooldown=0, min_lr=0)

    early_stop=tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy', min_delta=0.01, patience=6, verbose=2, mode='auto',
        baseline=None)


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

    # Saving the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file :
    	json_file.write(model_json)

    model.save_weights(os.path.join(r"C:\path\to\models",(NAME+".h5")))


    model.save(os.path.join(r"C:\path\to\models",NAME))
    print("Saved model to disk :",NAME)
    K.clear_session()
    tf.reset_default_graph()


winsound.Beep(700, 3000)




