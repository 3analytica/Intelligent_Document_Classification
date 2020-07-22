# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'thirdscreen_final2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QVBoxLayout, QListWidgetItem 

from PyQt5.QtWidgets import QApplication,QWidget,QInputDialog,QLineEdit,QFileDialog,QMessageBox
from PyQt5.QtGui import QIcon
import cv2
import glob
import sys

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
import glob
import pandas as pd
import numpy as np
import matplotlib
#from secondscreen import Ui_secondScreen


import cv2 
import sys
import pandas as pd
import pytesseract
import numpy as np
import os
import glob
from wand.image import Image as wi
import pickle

sys.path.append('G:/Capstone_Files')
from custom_spell_checker import correction
from enchant_autocorrect import enchant_correction
from doc_rotation import rotate_image
from preprocess_function import preprocess

#%matplotlib qt



#---------------------------------------------------------------------
#Load the two models


#TEXT BASED CLASSIFIER

filename = 'text_classification_model.sav'
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
loaded_model = pickle.load(open(filename, 'rb'))



#IMAGE BASED CLASSIFIER

#Set the path where 1595022683 model is saved
model = tf.keras.models.load_model("G:/Capstone_Files/1595022683")


#--------- Image Preprocess for Image Based Classifier

def prepare(file):
    IMG_SIZE = 360
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array=img_array/255
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)





class Ui_ThirdOutput(object):
    
    def ReturnBack(self):
        self.window=QtWidgets.QMainWindow()
        self.ui= Ui_ThirdOutput()
        self.ui.setupUi(self.window)
        #Ui_ThirdOutput.hide()
        self.window.show()

    
    
    def setupUi(self, ThirdOutput):
        ThirdOutput.setObjectName("ThirdOutput")
        ThirdOutput.resize(1140, 860)
        self.centralwidget = QtWidgets.QWidget(ThirdOutput)
        self.centralwidget.setObjectName("centralwidget")
        self.BackgroundImage = QtWidgets.QLabel(self.centralwidget)
        self.BackgroundImage.setGeometry(QtCore.QRect(10, 10, 1111, 811))
        self.BackgroundImage.setText("")
 
        #Seting backgorund image. Set the path where third.pic.png is saved.
        self.BackgroundImage.setPixmap(QtGui.QPixmap("G:/Nick/MSC Data Science/6th Semester/Capstone Project/Practice/New folder/third_pic.jpg"))
        self.BackgroundImage.setScaledContents(True)
        self.BackgroundImage.setObjectName("BackgroundImage")
        self.Third_Label = QtWidgets.QLabel(self.centralwidget)
        self.Third_Label.setGeometry(QtCore.QRect(320, 40, 531, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Third_Label.setFont(font)
        self.Third_Label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Third_Label.setAlignment(QtCore.Qt.AlignCenter)
        self.Third_Label.setObjectName("Third_Label")
        self.backButton = QtWidgets.QPushButton(self.centralwidget)
        self.backButton.setGeometry(QtCore.QRect(470, 680, 241, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.backButton.setFont(font)
        self.backButton.setObjectName("backButton")
        
        
        #Option to go back linked with click button
        self.backButton.clicked.connect(self.ReturnBack)
        
        
        ThirdOutput.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ThirdOutput)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1140, 26))
        self.menubar.setObjectName("menubar")
        ThirdOutput.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ThirdOutput)
        self.statusbar.setObjectName("statusbar")
        ThirdOutput.setStatusBar(self.statusbar)

        self.retranslateUi(ThirdOutput)
        QtCore.QMetaObject.connectSlotsByName(ThirdOutput)

    def retranslateUi(self, ThirdOutput):
        _translate = QtCore.QCoreApplication.translate
        ThirdOutput.setWindowTitle(_translate("ThirdOutput", "MainWindow"))
        #Text in Title
        self.Third_Label.setText(_translate("ThirdOutput", "Classification Results for Selected Files"))
        #Text included in button
        self.backButton.setText(_translate("ThirdOutput", "Back to File Selection"))

        filename=QFileDialog.getOpenFileNames()
  #      print(filename)
        path=filename[0]
        
        
        print(path)
        print(len(path))
        
        ##
        if len(path)==0: 
            msg = QMessageBox()
            msg.setWindowTitle("Received Documents Info")
            msg.setText("Please insert at least 1 document")
            msg.setIcon(QMessageBox.Critical)
            x = msg.exec_()  # this will show our messagebox
        
        elif len(path)>0: 
            msg = QMessageBox()
            msg.setWindowTitle("Received Documents Info")
            msg.setText("Correct Documents Input")
            #Below set path where tick.png is saved
            msg.setIconPixmap(QtGui.QPixmap("G:/Nick/MSC Data Science/6th Semester/Capstone Project/Practice/New folder/tick.PNG"))
            x = msg.exec_()  # this will show our messagebox

        array2=[]
        data2=pd.DataFrame()


        counter = 0
       
        for i in path:
            j=prepare(i)
            proba=model.predict_proba([j])
            array2.append(proba[0])


            ######
            pdf = wi(filename = i, resolution = 300) # fn replaced with i 
            pdfimage = pdf.convert('tiff')
            pg = 1
            for img in pdfimage.sequence:
                counter = counter + 1
                pgn = str(pg)
                page = wi(image = img)
                page.save(filename=str(i+pgn)+'.tiff') # fn replaced with i
                pg +=1
                print(">> done ",counter,">>")
            ######

        
######
        root = os.listdir('G:/Capstone_Files/images')
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        images = [cv2.imread(file,0) for file in glob.glob('G:/Capstone_Files/images/*1.tiff')]

        NoneType = type(None)
######

#Preprocces for any new inserted document in order to be prepared for text based classifier 

        array = []
        data = pd.DataFrame([])
        total_info = []
        count = 0
        
        for i in images:
            image2 = preprocess(i) 
            rotate = rotate_image(image2)
            if type(rotate) == NoneType:
                total_df = pytesseract.image_to_data(image2, lang='ell', output_type='data.frame')
            else:
                total_df = pytesseract.image_to_data(rotate, lang='ell', output_type='data.frame')    
            total_df = total_df[total_df.conf > 59]
            total_df = total_df[total_df.text != '']
            total_df = total_df[['text']]                  
            total_df['document'] = count
            total_df['lowercase'] = total_df['text'].str.lower()
            total_df[['our_corrected']] = total_df[['lowercase']].applymap(correction)
            total_df[['enchant_corrected']] = total_df[['lowercase']].applymap(enchant_correction)              
            total_df['final_text'] = total_df.apply(lambda r: (r['our_corrected'] + r['enchant_corrected']) if r['our_corrected'] == "" else r['our_corrected'], axis=1)
            final_series = total_df.groupby('document')['final_text'].apply(list)
            final_df = pd.DataFrame(final_series)    
            final_df['final_text_string'] = [' '.join(map(str, l)) for l in final_df['final_text']]
        
            doc_info = final_df['final_text_string'].tolist()
            total_info.extend(doc_info)
            
            doc_df_transformed = loaded_vectorizer.transform([final_df['final_text_string']])
            conf_table2 = loaded_model.predict_proba(doc_df_transformed)    
            array.append(conf_table2[0])
            data_final = data.append(pd.DataFrame(array), ignore_index=True)
            data_final=data_final.round(2)*100
            count = count + 1
        
        
        
# EXTRACTED WORDS FROM EVERY DOCUMENT

        a=[]
        for i in range (1,len(total_info)+1):
            a.append(str(i))
        
        b=["Extracted Words from Document " + i for i in a]
        c=[i + ": " for i in b]    
        total_info = [c[i] + total_info[i] for i in range(len(total_info))] 
            

        self.listWidget = QListWidget()
        self.listWidget.addItems(total_info)
        self.listWidget.show()            
        

                 


        
#===========================================================================================================

# BARPLOTS PRESENTING THE PROBABILITY FOR EVERY DOCUMENT FOR THE TWO MODELS

                 
        data2=data2.append(array2)
        data2.columns=["Purchase Receipt", "Service Receipt", "Police Report", "ΙΒΑΝ", "Contract", "Claim Statement"] 
        data2=data2.rename(index = lambda x: x+1 ) 
        data2.index = data2.index.map(str)
        data2=data2.rename(index = lambda x: "Document " + x ) 
        
                
        
        data2=data2.round(2)*100
    
    
        fig, axarr = plt.subplots(2, 1, figsize=(22, 10),sharex=True)
        matplotlib.style.use('fivethirtyeight') 
    
    
    
    
        data_final.plot.bar(ax=axarr[0],width=1.5)
        axarr[0].set_title("Text Based Classifier", fontsize=20,fontweight="bold",color="brown")
        axarr[0].get_legend().remove()
    
    
        
        data2.plot.bar(ax=axarr[1],width=1.5)
        axarr[1].set_title("Image Based Classifier", fontsize=20,fontweight="bold",color="brown")
        axarr[1].get_legend().remove()
        
        
            
        fig.legend(data2,     # The line objects
                           #labels=line_labels,   # The labels for each line
        loc="center right",   # Position of legend
        borderaxespad=2.2,    # Small spacing around legend box
        facecolor="white",
        title="Legend",  # Title for the legend
        title_fontsize=20,
        frameon=True,
        edgecolor="black")
                
        fig.text(0.04, 0.5, 'Probabilityy for each class', va='center', rotation='vertical',fontsize=20)
                
        plt.setp(plt.xticks()[1], rotation=0)
                
        plt.suptitle('Class Probability for each Document for both methods',fontsize=28,fontweight="bold")
        plt.subplots_adjust(right=0.85)
        
        
        
        
        
        
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ThirdOutput = QtWidgets.QMainWindow()
    ui = Ui_ThirdOutput()
    ui.setupUi(ThirdOutput)
    ThirdOutput.show()
    sys.exit(app.exec_())

