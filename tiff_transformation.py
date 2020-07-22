import cv2 
import sys
import pandas as pd
import numpy as np
import os
import glob
from wand.image import Image as wi
from wand.display import display as wd

#TRANSFORMATION LOOP

counter = 0

for fn in glob.glob('C:/Users/Thanos/Desktop/images/Dataset/*'):
    pdf = wi(filename = fn, resolution = 300)
    pdfimage = pdf.convert('tiff')
    pg = 1
    for img in pdfimage.sequence:
        counter = counter + 1
        pgn = str(pg)
        page = wi(image = img)
        page.save(filename=str(fn+pgn)+'.tiff')
        pg +=1
        print(">> done ",counter,">>")


