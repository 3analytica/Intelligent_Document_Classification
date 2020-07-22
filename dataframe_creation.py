import cv2
import sys
import pytesseract
import pandas as pd
import os
import glob
import time

sys.path.append('C:/Users/Thanos/Desktop/images/Functions')
from custom_spell_checker import correction
from enchant_autocorrect import enchant_correction
from doc_rotation import rotate_image
from preprocess_function import preprocess
from nlp_features_function import punct_percent, num_percent, words_len

root = os.listdir('C:/Users/Thanos/Desktop/images/Dataset(TIFF)')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
images = [cv2.imread(file,0) for file in glob.glob('C:/Users/Thanos/Desktop/images/Dataset(TIFF)/*1.tiff')]
NoneType = type(None)
    
#----------------------------------------------------------------------------------------------------------#

categories = []
filenames = []
cat_indicator = 0
filename_indicator = 0

for file in root:
    if file.endswith('1.tiff'):
        cat = file.split("_")[0]
        categories.append(cat)
        filename = file.split(".")[0]
        filenames.append(filename)
      
for i in images:
    start_time = time.time()
    image2 = preprocess(i) 
    rotate = rotate_image(image2)
    cat_indicator = cat_indicator
    filename_indicator = filename_indicator            
    if type(rotate) == NoneType:
        total_df = pytesseract.image_to_data(image2, lang='ell', output_type='data.frame')
    else:
        total_df = pytesseract.image_to_data(rotate, lang='ell', output_type='data.frame')    
    total_df = total_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    total_df = total_df[total_df.conf > 59]
    total_df = total_df[total_df.text != '']
    total_df = total_df[['text']]
    total_df['document'] = (filenames[filename_indicator])                      
    total_df['lowercase'] = total_df['text'].str.lower()
    total_df[['our_corrected']] = total_df[['lowercase']].applymap(correction)
    total_df[['enchant_corrected']] = total_df[['lowercase']].applymap(enchant_correction)              
    total_df['final_text'] = total_df.apply(lambda r: (r['our_corrected'] + r['enchant_corrected']) if r['our_corrected'] == "" else r['our_corrected'], axis=1)
    final_series = total_df.groupby('document')['final_text'].apply(list)
    final_df = pd.DataFrame(final_series)    
    final_df['final_text_string'] = [' '.join(map(str, l)) for l in final_df['final_text']]
    final_df["category"] = categories[cat_indicator]                    
    final_df["length"] = words_len(final_df.final_text_string)
    final_df["punct_%"] = punct_percent(final_df.final_text_string)
    final_df["num_%"] = num_percent(final_df.final_text_string)
    final_df.to_csv('C:\\Users\\Thanos\\Desktop\\images\\Validation_CSVs\\'+filenames[filename_indicator]+'.csv',index = True,encoding='utf_8_sig')
    print("done >>",filenames[filename_indicator])
    print("took ", round(time.time() - start_time,2), "seconds to run")
    cat_indicator = cat_indicator + 1
    filename_indicator = filename_indicator + 1            


#################################################################################################################################
