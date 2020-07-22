# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:04:41 2020

@author: Spyros
"""

import os
import cv2
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import pickle
import numpy as np

directory=r"C:\Capstone\Validation_dataset"

def prepare(file):
    IMG_SIZE = 360
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array=img_array/255
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model_Multi = tf.keras.models.load_model(r"C:\path\to\models\multi model")
model_Pur = tf.keras.models.load_model(r"C:\path\to\models\Binary_Purchase_Rec")
model_Serv = tf.keras.models.load_model(r"C:\path\to\models\Binary_Service_Rec")
model_Pol = tf.keras.models.load_model(r"C:\path\to\models\Binary_Police_Report")
model_IBAN = tf.keras.models.load_model(r"C:\path\to\models\Binary_IBAN")
model_InsCont = tf.keras.models.load_model(r"C:\path\to\models\Binary_Insurance_Contract")
model_Claims_Stmnt = tf.keras.models.load_model(r"C:\path\to\models\Binary_Claims_st")


CATEGORIES_Multi =["1_Purchase_Rcpt", "2_Service_Rcpt", "3_Police_Rprt", "4_IBAN", "5_Ins_Contract", "6_Claim_Stmnt"]
CATEGORIES_Pur = ["1_is_Purchase_Rcpt", "2_is_Not_Purchase_Rcpt"]
CATEGORIES_Serv = ["1_is_Service_Rcpt", "2_is_Not_Service_Rcpt"]
CATEGORIES_Pol = ["1_is_Police_Stmnt", "2_is_Not_Police_Stmnt"]
CATEGORIES_IBAN = ["1_is_IBAN", "2_is_Not_IBAN"]
CATEGORIES_Insurance_Contr = ["1_is_Insurance_Contr", "2_is_Not_Insurance_Contr"]
CATEGORIES_Claims_Stmnt = ["1_is_Claims_Stmnt", "2_is_Not_Claims_Stmnt"]


df=pd.DataFrame()
for filename in os.listdir(directory):
    print(filename.split(".")[0])

    try:
        image=prepare(os.path.join(directory, filename))
        prediction = model_Multi.predict([image])
        proba=model_Multi.predict_proba([image])
        prediction = list(prediction[0])
        catigoria=CATEGORIES_Multi[prediction.index(max(prediction))]
        print(catigoria)
        catigoria_bool=""

        if "Purchase" in catigoria:
            prediction2 = model_Pur.predict([image])
            proba_bool=model_Pur.predict_proba([image])
            prediction2 = list(prediction2[0])
            catigoria_bool=CATEGORIES_Pur[prediction2.index(max(prediction2))]

        elif "Service" in catigoria:
            prediction2 = model_Serv.predict([image])
            proba_bool=model_Serv.predict_proba([image])
            prediction2 = list(prediction2[0])
            catigoria_bool=CATEGORIES_Serv[prediction2.index(max(prediction2))]

        elif "Police" in catigoria:
            prediction2 = model_Pol.predict([image])
            proba_bool=model_Pol.predict_proba([image])
            prediction2 = list(prediction2[0])
            catigoria_bool=CATEGORIES_Pol[prediction2.index(max(prediction2))]

        elif "IBAN" in catigoria:
            prediction2 = model_IBAN.predict([image])
            proba_bool=model_IBAN.predict_proba([image])
            prediction2 = list(prediction2[0])
            catigoria_bool=CATEGORIES_IBAN[prediction2.index(max(prediction2))]

        elif "Ins_Contract" in catigoria:
            prediction2 = model_InsCont.predict([image])
            proba_bool=model_InsCont.predict_proba([image])
            prediction2 = list(prediction2[0])
            catigoria_bool=CATEGORIES_Insurance_Contr[prediction2.index(max(prediction2))]

        elif "Claim_Stmnt" in catigoria:
            prediction2 = model_Claims_Stmnt.predict([image])
            proba_bool=model_Claims_Stmnt.predict_proba([image])
            prediction2 = list(prediction2[0])
            catigoria_bool=  CATEGORIES_Claims_Stmnt[prediction2.index(max(prediction2))]

        df = df.append({'File': filename,
                        'prediction_multi': catigoria,
                        'proba_multi': round((proba[0][prediction.index(max(prediction))]*100),3),
                        "category_bool": catigoria_bool,
                        "proba_bool":round((proba_bool[0][prediction2.index(max(prediction2))]*100),3) }
                        , ignore_index=True)

    except:

        df = df.append({'File': filename, 'prediction_multi': "Not Possible"},ignore_index=True)

df2=df.copy()


df = df.dropna(subset=['prediction_multi'])
df['Real_Category'] = df.File.apply(lambda x: x[:5])
df['prediction_multi'] = df.prediction_multi.apply(lambda x: x[:5])
df['Result_Multi'] = np.where(df['Real_Category'].eq(df['prediction_multi']), 'Correct_Multi_Prediction', 'False_Multi_Prediction')

df.to_excel("combined.xlsx")


