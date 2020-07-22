import re
import string
import nltk
import nltk.tag
from nltk.corpus import brown
from collections import defaultdict
from nltk.corpus import names, stopwords, words

# define stopwords
stopword = nltk.corpus.stopwords.words('greek')

# add a few more (insignifiant words)
new_stop_words = ['ο','δ','υ','ε','e','δου','φπα','καει','καεί','©','€','αφης']

# extend the stopword list
stopword.extend(new_stop_words)

# function to remove stopwords
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

# function to remove punctuations and all digits (including words following digits)
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('\w*[0-9]\w*', '', text)
    return text

# function to clean text
def clean_text(text):
    text_lc  = "".join([word.lower() for word in text if word not in string.punctuation])
    text_rc = re.sub('\w*[0-9]\w*', '', text_lc)  
    tokens = nltk.word_tokenize(text_rc)
    stop = [word for word in tokens if word not in stopword]
    return stop


