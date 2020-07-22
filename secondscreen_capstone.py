# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'secondscreen.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtWidgets import QApplication,QWidget,QInputDialog,QLineEdit,QFileDialog,QMessageBox
#from PyQt5.QtGui import QIcon
#import cv2
#import glob
#from thirdscreen_final2_Copy import Ui_ThirdOutput
from thirdscreen_capstone import Ui_ThirdOutput


class Ui_secondScreen(object):

#Connect second screen with 3rd screen
    def openThirdWindow(self):
        self.window=QtWidgets.QMainWindow()
        self.ui= Ui_ThirdOutput()
        self.ui.setupUi(self.window)
        # Ui_secondScreen.hide()
        self.window.show()

    
    
        
    
    def setupUi(self, secondScreen):
        secondScreen.setObjectName("secondScreen")
        secondScreen.resize(1160, 872)
        self.centralwidget = QtWidgets.QWidget(secondScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundImage_second = QtWidgets.QLabel(self.centralwidget)
        self.backgroundImage_second.setGeometry(QtCore.QRect(0, -20, 1161, 871))
        self.backgroundImage_second.setText("")
        
        #Seting backgorund image. Set the path where second.pic.jpg is saved.

        self.backgroundImage_second.setPixmap(QtGui.QPixmap("G:/Nick/MSC Data Science/6th Semester/Capstone Project/Practice/New folder/second_pic.jpg"))
        self.backgroundImage_second.setScaledContents(True)
        self.backgroundImage_second.setObjectName("backgroundImage_second")
        self.BrowseButton = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseButton.setGeometry(QtCore.QRect(400, 450, 341, 111))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.BrowseButton.setFont(font)
        self.BrowseButton.setObjectName("BrowseButton")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(230, 60, 701, 111))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setAutoFillBackground(True)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        secondScreen.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(secondScreen)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1160, 26))
        self.menubar.setObjectName("menubar")
        secondScreen.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(secondScreen)
        self.statusbar.setObjectName("statusbar")
        secondScreen.setStatusBar(self.statusbar)

        self.retranslateUi(secondScreen)
        QtCore.QMetaObject.connectSlotsByName(secondScreen)

    def retranslateUi(self, secondScreen):
        _translate = QtCore.QCoreApplication.translate
        secondScreen.setWindowTitle(_translate("secondScreen", "MainWindow"))
        self.BrowseButton.setText(_translate("secondScreen", "Press here to browse files \n and \n wait few minutes for the result")) #set text to button
        self.title.setText(_translate("secondScreen", "Insert Documents for Classification")) #set text to button
       
        self.BrowseButton.clicked.connect(self.BrowseButton_handler)
        
        
        
    def BrowseButton_handler(self):
        #print("Button pressed")
        self.openThirdWindow()
        
         


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    secondScreen = QtWidgets.QMainWindow()
    ui = Ui_secondScreen()
    ui.setupUi(secondScreen)
    secondScreen.show()
    sys.exit(app.exec_())

