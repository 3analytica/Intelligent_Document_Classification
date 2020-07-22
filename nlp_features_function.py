import pandas as pd
import string

punct = string.punctuation
numbers = string.digits

def listToString(s):   
    string = ""  
    for token in s:  
        string += token
    return string 

def words_len(string):
    length = string.str.split().str.len()
    return length[0]
        
def punct_percent(df):
    x = df.tolist()
    y = listToString(x)
    count = sum([1 for char in y if char in punct])
    return round(count/(len(y))*100,3)

def num_percent(df):    
    x = df.tolist()
    y = listToString(x)
    count = sum([1 for char in y if char in numbers])
    return round(count/(len(y))*100,3)


