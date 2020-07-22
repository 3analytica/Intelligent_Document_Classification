# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:28:42 2020

@author: Spyros
"""
#This method reads a pdf and converts it into a sequence of images


from pdf2image.exceptions import (    PDFInfoNotInstalledError,    PDFPageCountError,    PDFSyntaxError)
from pdf2image import convert_from_path, convert_from_bytes
import os
import pdf2image
from PIL import Image
import time

#PDF_PATH sets the path to the PDF file
#dpi parameter assists in adjusting the resolution of the image
#output_folder parameter sets the path to the folder to which the PIL images can be stored (optional)
#first_page parameter allows you to set a first page to be processed by pdftoppm
#last_page parameter allows you to set a last page to be processed by pdftoppm
#fmt parameter allows to set the format of pdftoppm conversion (PpmImageFile, TIFF)
#thread_count parameter allows you to set how many thread will be used for conversion.
#userpw parameter allows you to set a password to unlock the converted PDF
#use_cropbox parameter allows you to use the crop box instead of the media box when converting
#strict parameter allows you to catch pdftoppm syntax error with a custom type PDFSyntaxError



src = r"C:\Users\Spyros\OneDrive - The American College of Greece\Capstone-2020-Fournogerakis-Petsakos-Poulos\Spyros\Images me lathos gia niko"

DPI = 300
OUTPUT_FOLDER =None
FIRST_PAGE = None
LAST_PAGE = None
FORMAT = 'jpg'
THREAD_COUNT = 12
USERPWD = None
USE_CROPBOX = False
STRICT = False
SAVE_PATH=r"C:\Users\Spyros\OneDrive - The American College of Greece\Capstone-2020-Fournogerakis-Petsakos-Poulos\Spyros\New folder"
error_count=0




for root, subdirs, files in os.walk(src):
    for file in files:
        try:
            path = os.path.join(root, file)
            print(os.path.basename((path)))
            if "pdf" in os.path.basename((path)) or "PDF" in os.path.basename((path)):
                print("processing : ",os.path.basename((path)))
                start=time.time()
                pil_images = pdf2image.convert_from_path(path, dpi=DPI,
                                                         output_folder=OUTPUT_FOLDER,
                                                         first_page=FIRST_PAGE, last_page=LAST_PAGE,
                                                         fmt=FORMAT, thread_count=THREAD_COUNT,
                                                         userpw=USERPWD, use_cropbox=USE_CROPBOX,
                                                         strict=STRICT)
                #save_path=os.path.join(SAVE_PATH,(os.path.basename((path)) +".jpg"))
                for i, image in enumerate(pil_images):
                    fname = os.path.basename((path))+"image" + str(i) + ".jpg"
                    image.save(os.path.join(SAVE_PATH,fname), "jpeg")
                end=time.time()
                print("elapsed time : ", str(round((end-start),3)))
        except:
            error_count+=1
            print("δεν επεξεργαστηξε : ", os.path.basename((path)))

import winsound
#this makes a sound to notify us at the end of the process
winsound.Beep(540, 2000)





