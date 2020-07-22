import pandas as pd
import glob

path = 'C:/Users/Thanos/Desktop/images/All_CSVs'
all_files = glob.glob(path + "/*.csv")

list1 = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    list1.append(df)

final_df = pd.concat(list1, axis=0, ignore_index=True)

final_df.to_csv(r'C:/Users/Thanos/Desktop/images\Final_Dataset.csv', index = False, encoding='utf_8_sig')

#################################################################################################################################