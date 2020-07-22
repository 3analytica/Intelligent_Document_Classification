import nltk
import nltk.tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from string import punctuation
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import pickle

sys.path.append('C:/Users/Thanos/Desktop/images/Functions')
from nlp_functions import remove_punct, remove_stopwords, clean_text

#%matplotlib qt # output in separate window

###########################################################################################################

# import dataset csv into df (2535 documents!)
all_data = pd.read_csv('Final_Dataset.csv')

###########################################################################################################

#SOME CLEANING UP (Using Functions!)

# create new column in df for docs without punctuation  (with function)
all_data['No_Punct'] = all_data['final_text'].apply(lambda x: remove_punct(x.lower()))

# add tokenized column
all_data['Tokenized'] = all_data.apply(lambda row: nltk.word_tokenize(row['No_Punct']), axis=1)

# create new column in df for docs without stopwords (with function)
all_data['Nonstop'] = all_data['Tokenized'].apply(lambda x: remove_stopwords(x))

# make nonstop column into a string
all_data['final_string'] = [' '.join(map(str, l)) for l in all_data['Nonstop']]

# keep only importnant columns
all_data = all_data.drop(['final_text','final_text_string','Nonstop','No_Punct','Tokenized'], axis = 1) 


###########################################################################################################

# QUICK PLOTS

# quick plot
sns.countplot(x = 'category', data = all_data)

# plot Char_count for each of the three final ratings 
g = sns.FacetGrid(data = all_data, col = 'category')
g.map(plt.hist,'length',bins=50)

# plot word_count for each of the three final ratings 
g = sns.FacetGrid(data = all_data, col = 'category')
g.map(plt.hist,'punct_%',bins=50)

# plot word_count for each of the three final ratings 
g = sns.FacetGrid(data = all_data, col = 'category')
g.map(plt.hist,'num_%',bins=50)

# boxplots
sns.boxplot(x = 'category', y = 'length', data = all_data)
sns.boxplot(x = 'category', y = 'punct_%', data = all_data)
sns.boxplot(x = 'category', y = 'num_%', data = all_data)

# distribution of categories
sns.distplot(all_data.category)
plt.show

###########################################################################################################

# QUICK SUMMARY

# split df into 6 separate dfs 
purchase = all_data.loc[all_data['category'] == 1]
service = all_data.loc[all_data['category'] == 2]
police = all_data.loc[all_data['category'] == 3]
bank = all_data.loc[all_data['category'] == 4]
insurance = all_data.loc[all_data['category'] == 5]
claim = all_data.loc[all_data['category'] == 6]

# WORDCLOUD VISUALIZATION

pur_words = " ".join(review for review in purchase.final_string)
ser_words = " ".join(review for review in service.final_string)
pol_words = " ".join(review for review in police.final_string)
bnk_words = " ".join(review for review in bank.final_string)
ins_words = " ".join(review for review in insurance.final_string)
clm_words = " ".join(review for review in claim.final_string)

# Create and generate a word cloud image & Display the generated image
wordcloud = WordCloud().generate(clm_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

###########################################################################################################

# CLASSIFICATION - USING COUNT VECORIZER ON DATAFRAME 

shuffled_df = all_data.sample(frac=1,random_state=4)

one_df = shuffled_df.loc[shuffled_df.category == 1].sample(replace=False, n=450,random_state=123)
two_df = shuffled_df.loc[shuffled_df.category == 2].sample(replace=False, n=446,random_state=123)
three_df = shuffled_df.loc[shuffled_df.category == 3].sample(replace=False, n=334,random_state=123)
four_df = shuffled_df.loc[shuffled_df.category == 4].sample(replace=False, n=450,random_state=123)
five_df = shuffled_df.loc[shuffled_df.category == 5].sample(replace=False, n=500,random_state=123)
six_df = shuffled_df.loc[shuffled_df.category == 6].sample(replace=False, n=406,random_state=123)

# combine the 6 dfs
df_trimmed = pd.concat([one_df, two_df, three_df, four_df, five_df, six_df])

# shuffle (again) the df_trimmed df
df_trimmed = shuffle(df_trimmed)

# quick plot (lets see what we got)
sns.countplot(x = 'category', data = df_trimmed)

###########################################################################################################

# define y variable from now
y = df_trimmed['category']
    
# define count vectorizer
countVectorizer = CountVectorizer(analyzer = clean_text) 

# fit transform count vectorizer - use the df column that you want to!
countVector = countVectorizer.fit_transform(df_trimmed['final_string'])

# save the vectorizer to disk
vec_file = 'vectorizer.pickle'
pickle.dump(countVectorizer, open(vec_file, 'wb'))

###########################################################################################################

# print number of document and number of words (total)
print('{} Documents have {} words'.format(countVector.shape[0], countVector.shape[1]))

# create count vect dataframe - get feature names 
count_vect_df = pd.DataFrame(countVector.toarray(), columns = countVectorizer.get_feature_names())

# head of new vectorized dataframe
count_vect_df.head()

###########################################################################################################

# Split in to test and train set
X_train, X_test, y_train, y_test = train_test_split(countVector, y, test_size=0.2, random_state=101)

###########################################################################################################

# Logistic Regression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)

# Let’s evaluate our predictions against the actual ratings
print(accuracy_score(y_test, preds_lr)*100,'%')
print('\n')
print(confusion_matrix(y_test, preds_lr))
print('\n')
print(classification_report(y_test, preds_lr))

###########################################################################################################

# Neural Network
NN = MLPClassifier(activation='relu', hidden_layer_sizes=(9,), max_iter=600, alpha=0.0001,
                   solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

NN.fit(X_train, y_train)
preds_NN = NN.predict(X_test)

# Let’s evaluate our predictions against the actual ratings
print(accuracy_score(y_test, preds_NN)*100,'%')
print('\n')
print(confusion_matrix(y_test, preds_NN))
print('\n')
print(classification_report(y_test, preds_NN))

#####################################################################################################################

# TESTING WITH A SINGLE DOCUMENT

# Select a single document
random_doc = df_trimmed['final_string'][13]

# print it out so we can see what the text is
random_doc

# transform it using the count vectorizer
random_doc_transformed = countVectorizer.transform([random_doc])

prediction_lr = lr.predict(random_doc_transformed)[0]
prediction_NN = NN.predict(random_doc_transformed)[0]

# predict the category of the document
print("This document has been classified as category:",prediction_lr)
print("This document has been classified as category:",prediction_NN)

#####################################################################################################################

# TESTING WITH NEW TEXT

# Select a single document
text = ['λογαριασμος']

# print it out so we can see what the text is
text

# transform it using the count vectorizer
text_transformed = countVectorizer.transform([text])

prediction_lr2 = lr.predict(text_transformed)[0]
prediction_NN2 = NN.predict(text_transformed)[0]

# predict the category of the document
print("This document has been classified as category:",prediction_lr2)
print("This document has been classified as category:",prediction_NN2)

#####################################################################################################################

# Make random document confidence table (df) 

conf_table = NN.predict_proba(random_doc_transformed)

conf_df = pd.DataFrame (conf_table, columns = ['Purchase_Receipt','Service_Receipt',
                                               'Police_Report','Bank_IBAN',
                                               'Insurance_Contract','Claim_Statement'])

#####################################################################################################################

# TESTING MODEL WITH VALIDATION DATASET

# import documents csv into df (300+ documents!)
second_data = pd.read_csv('Validation_Dataset.csv')

# define y variable from now
y2 = second_data['category']

# define count vectorizer
countVectorizer = CountVectorizer(analyzer = clean_text) 

# fit transform count vecotrizer
countVector = countVectorizer.fit_transform(second_data['final_text_string'])

# Split in to test and train set
X_train2, X_test2, y_train2, y_test2 = train_test_split(countVector, y2, test_size=0.2, random_state=101)

# Let’s build a model and fit it to our training set
lr.fit(X_train2, y_train2)
NN.fit(X_train2, y_train2)

# Our model has now been trained
preds_test1 = lr.predict(X_test2)
preds_test2 = NN.predict(X_test2)

# Let’s evaluate our predictions against the actual ratings
print("accuracy with LR:",accuracy_score(y_test2, preds_test1)*100,'%')
print("accuracy with NN:",accuracy_score(y_test2, preds_test2)*100,'%')

#####################################################################################################################

# GridSearch Paramerters (for report)

#param_grid = [{'activation' : ['relu'],
#            'max_iter' : [600], 
#            'solver' : ['sgd'],
#            'hidden_layer_sizes': [
#             (5,),(6,),(7,),(8,),(9,),(10,)
#             ]}]

#NN = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy')
#print("Best parameters found:")
#print(NN.best_params_)

#####################################################################################################################

# Voting Classifier (equal weight)

score_LR = lr.fit(X_train,y_train).score(X_test, y_test)
score_NN = NN.fit(X_train,y_train).score(X_test, y_test)

Ensemble = VotingClassifier(estimators=[('LR', lr), ('NN', NN)],
                        voting='soft',
                        weights=[1, 1])

score_Ensemble = Ensemble.fit(X_train,y_train).score(X_test, y_test)

print(score_Ensemble)

#####################################################################################################################

# Save Model Using Pickle

# save the model to disk
filename = 'text_classification_model.sav'
pickle.dump(Ensemble, open(filename, 'wb'))

#####################################################################################################################
