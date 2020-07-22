import cv2 
import sys
import pandas as pd
import pytesseract
import numpy as np
import os
import glob
from wand.image import Image as wi
import pickle

sys.path.append('C:/Users/Thanos/Desktop/images/Functions')
from custom_spell_checker import correction
from enchant_autocorrect import enchant_correction
from doc_rotation import rotate_image
from preprocess_function import preprocess

filename = 'text_classification_model.sav'
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
loaded_model = pickle.load(open(filename, 'rb'))

#===========================================================================================================

# document transformation loop

counter = 0

for fn in glob.glob('C:/Users/Thanos/Desktop/images/Doc_Pool/*'):
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
        
#===========================================================================================================

# loop for filename indicator

root = os.listdir('C:/Users/Thanos/Desktop/images/Doc_Pool')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
images = [cv2.imread(file,0) for file in glob.glob('C:/Users/Thanos/Desktop/images/Doc_Pool/*1.tiff')]
NoneType = type(None)

#===========================================================================================================

# loop for rotation, text extraction, correction, dataframe creation, transformation and classification

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
    total_df = total_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
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
    count = count + 1

#############################################################################################################
