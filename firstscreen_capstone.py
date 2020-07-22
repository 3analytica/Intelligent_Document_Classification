# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'firstscreen_final3.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QWidget,QInputDialog,QLineEdit,QFileDialog,QMessageBox
from secondscreen_capstone import Ui_secondScreen

## Uncomment out the below command and execute it alone ,in order the graphs to be presented in seperate window! 
#Then comment it out again!

#%matplotlib qt




class Ui_DocumentClassification(object):
    
  ###Verifying the credentials, present notification messages and connect with second screen and then hide 1st screen window  
    
    def openWindow(self):
        msg=QMessageBox()
        
        if self.UserText.text() =="Master" and self.PasswordText.text() =="Is over":
            msg.setText("Correct Credentials")
            msg.exec_()
            self.window=QtWidgets.QMainWindow()
            self.ui= Ui_secondScreen()
            self.ui.setupUi(self.window)
            DocumentClassification.hide()
            self.window.show()
        else:
            msg.setText("Incorrect Credentials")
            msg.exec_()

    
    #Set names,font,font size and geometry for every element in the UI. Read and present background pictures from path   
    
    def setupUi(self, DocumentClassification):
        DocumentClassification.setObjectName("DocumentClassification")
        DocumentClassification.resize(1087, 815)
        self.centralwidget = QtWidgets.QWidget(DocumentClassification)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundPic = QtWidgets.QLabel(self.centralwidget)
        self.backgroundPic.setGeometry(QtCore.QRect(10, 20, 1071, 761))
        self.backgroundPic.setAutoFillBackground(False)
        self.backgroundPic.setText("")
        
        #Insert the according path that you have stored first2.PNG
        
        self.backgroundPic.setPixmap(QtGui.QPixmap("G:/Nick/MSC Data Science/6th Semester/Capstone Project/Practice/New folder/New folder/first2.PNG"))
        self.backgroundPic.setScaledContents(True)
        self.backgroundPic.setObjectName("backgroundPic")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 0, 1071, 61))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.title.setFont(font)
        self.title.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.title.setAutoFillBackground(True)
        self.title.setScaledContents(True)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.LoginButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoginButton.setGeometry(QtCore.QRect(390, 730, 361, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(170, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        self.LoginButton.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.LoginButton.setFont(font)
        self.LoginButton.setObjectName("LoginButton")
        
  ### Connect with second screen through click         
        self.LoginButton.clicked.connect(self.openWindow)

        
        self.Userlabel = QtWidgets.QLabel(self.centralwidget)
        self.Userlabel.setGeometry(QtCore.QRect(390, 630, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.Userlabel.setFont(font)
        self.Userlabel.setAutoFillBackground(True)
        self.Userlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.Userlabel.setObjectName("Userlabel")
        self.PasswordLebel = QtWidgets.QLabel(self.centralwidget)
        self.PasswordLebel.setGeometry(QtCore.QRect(390, 680, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PasswordLebel.setFont(font)
        self.PasswordLebel.setAutoFillBackground(True)
        self.PasswordLebel.setAlignment(QtCore.Qt.AlignCenter)
        self.PasswordLebel.setObjectName("PasswordLebel")
        self.AskCredentials = QtWidgets.QLabel(self.centralwidget)
        self.AskCredentials.setGeometry(QtCore.QRect(390, 580, 361, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.AskCredentials.setFont(font)
        self.AskCredentials.setAutoFillBackground(True)
        self.AskCredentials.setAlignment(QtCore.Qt.AlignCenter)
        self.AskCredentials.setObjectName("AskCredentials")
        self.UserText = QtWidgets.QLineEdit(self.centralwidget)
        self.UserText.setGeometry(QtCore.QRect(560, 630, 181, 41))
        self.UserText.setObjectName("UserText")
        self.PasswordText = QtWidgets.QLineEdit(self.centralwidget)
        self.PasswordText.setGeometry(QtCore.QRect(560, 680, 181, 41))
        self.PasswordText.setObjectName("PasswordText")
       

        #Set the Credentials
        self.UserText.text()== "Master"
        self.PasswordText.text()== "Is Over"
        
        
                
        DocumentClassification.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(DocumentClassification)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1087, 26))
        self.menubar.setObjectName("menubar")
        DocumentClassification.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(DocumentClassification)
        self.statusbar.setObjectName("statusbar")
        DocumentClassification.setStatusBar(self.statusbar)

        self.retranslateUi(DocumentClassification)
        QtCore.QMetaObject.connectSlotsByName(DocumentClassification)

    def retranslateUi(self, DocumentClassification):
        _translate = QtCore.QCoreApplication.translate
        DocumentClassification.setWindowTitle(_translate("DocumentClassification", "Document Classification Application"))
        self.title.setText(_translate("DocumentClassification", "Welcome to the Document Classification Application"))
        self.LoginButton.setText(_translate("DocumentClassification", "Login"))
        self.Userlabel.setText(_translate("DocumentClassification", "Username"))
        self.PasswordLebel.setText(_translate("DocumentClassification", "Password"))
        self.AskCredentials.setText(_translate("DocumentClassification", "Please enter your credentials"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DocumentClassification = QtWidgets.QMainWindow()
    ui = Ui_DocumentClassification()
    ui.setupUi(DocumentClassification)
    DocumentClassification.show()
    sys.exit(app.exec_())

