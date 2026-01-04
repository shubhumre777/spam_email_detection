import pandas as pd

# importing the dataset : 
df = pd.read_csv('spam.csv', encoding="latin1")

# Removing waste columns
df.drop(columns=['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4' ] , inplace=True)

# renaming remaining columns
df.rename(columns={'v1' : 'Target' , 'v2' : 'Text'} , inplace=True)

# Applying LabelEncoding on the Target column

from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()

df['Target'] = encode.fit_transform(df['Target'])

# print(df.isnull().sum())

df = df.drop_duplicates(keep='first')

# here :
'''
0 -> NOT SPAM (HAM)
1 -> SPAM 
'''


'''                                                                              PERFORMING EDA (Exploratory Data Analysis)                                                   '''

# print(df['Target'].value_counts())

# plotting the value_counts data on PIE chart

import matplotlib.pyplot as plt

plt.pie(df['Target'].value_counts() , labels=['Ham' , 'Spam'] , autopct='%0.2f')
# plt.show()

# by the pie chart we get to know that data is imballenced because spam mails are just 12.63 % and not spam mails are 87.37 %

# now doing some text processing
'''                                                             PERFORMING NLP (Natural language processing)                                                                   '''

import nltk 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# This object is used to do stemming of the words , basically to bring the words to the root words .
ps = PorterStemmer()


# These commands downloads all the necesary things to run NLTK library .
'''
 nltk.download('punkt')
 nltk.download('punkt_tab')
 nltk.download('stopwords')
 '''

def text_pre_process(text) :
    # This will make the text to lower case .
    text = text.lower()
    # This will break the text to tokens . 
    text = nltk.word_tokenize(text)
    
    # This helps to remove the special characters like (( , @ # $ % ^ , etc...) , newlines and spaces also .
    y = []
    for i in text :
        if i.isalnum() :
            y.append(i)

    # This removes Stopwords and Punctuations from the text .
    text = y[:] # here we are copying the y and then clearing it in next step , hence the complete data of y is stored in text .
    y.clear()
    for i in text :
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)

    # This loop does stemming of the words in the text (STEMMINg means converting words to their root words .)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    # This join methods make a complete sentence from the list of words like : ["Hello" , "World"] --> Hello World. 
    return " ".join(y)


# i am add a new columns to my dataset after performing the NLP on text column . 
df['Processed_text']= df['Text'].apply(text_pre_process)

# This will give all the spam messages from the Processed_text column . 
spam_messages = df[df['Target']==1]['Processed_text']


# This is one of way to find out words that are comming most frequently in spam messages or ham messages.
'''
# Import CountVectorizer from sklearn
# CountVectorizer will break text into words and count how many times each word appears
from sklearn.feature_extraction.text import CountVectorizer

# Create an object (tool) of CountVectorizer
cv = CountVectorizer()

# Use CountVectorizer on our list of text messages (spam_messages)
# This will change text messages into numbers (word counts)
X = cv.fit_transform(spam_messages)


# Step 4: Total count of each word
# Import pandas library to make tables (DataFrames)
import pandas as pd

# Change the result X into an array (table of numbers)
# Then sum all rows to get total count of each word across all messages
word_counts = X.toarray().sum(axis=0)

# Get the list of all words (features) that CountVectorizer found
words = cv.get_feature_names_out()

# Make a DataFrame (table) that shows each word and its total count
freq_df = pd.DataFrame({'word': words, 'count': word_counts})

# Sort the table from highest count to lowest count
freq_df = freq_df.sort_values(by='count', ascending=False)

# Print the first 20 words with the highest counts
print(freq_df.head(20))


'''

# This is  the another way and easiest way to perform the same upper thing :

'''df[df['Target']==0]['Processed_text'].tolist() : By this logic we are astracting all the spam text in
'Processed_text' column and then converting that column-series to list . 
Then applying for loop on that list '''

'''There are nested for loops , the first for loop iterates on each (end to end complete sentence) , then the 
2nd for loop iterates on each letter of each sentence , but after adding .split() all the words in each sentence
form a list of words by splitting them according to the white spaces between them. (This is the reason why we used .split()
method in the 2nd for loop . )
'''

# Create an empty list to store all the words we will collect
spam_corpus = []  # currently it is an empty list.

# Go through each sentence of the 'Processed_text' column where 'Target' value is 1 (SPAM) .
for text in df[df['Target'] == 1]['Processed_text'].tolist():
    
    # Split each text into individual words and go through each word
    for word in text.split():
        
        # Add (append) each word into the word_corpus list
        spam_corpus.append(word)

# print(len(spam_corpus))

# Import the Counter class from the collections library
# Counter helps to count how many times each word appears
from collections import Counter

# Count how many times each word appears in word_corpus
# Then get the 40 most common words and turn them into a DataFrame table
spam_word_count = pd.DataFrame(Counter(spam_corpus).most_common(40))
# print(spam_word_count)

# APPLYING SAME UPPER LOGIC FOR NON_SPAM (HAM) messages : 


ham_corpus = [] # currently it is an empty list .

# Go through each sentence of the 'Processed_text' column where 'Target' value is 0 (NON_SPAM or ham) .
for text in df[df['Target']==0]['Processed_text'].tolist() : 
    for word in text.split():
         ham_corpus.append(word) 

# print(len(ham_corpus))
         
from collections import Counter
ham_word_count = pd.DataFrame(Counter(ham_corpus).most_common(40)) 
# print(ham_word_count)



# '''                                                                                 MODEL BUILDING :                                                                                       '''
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler , StandardScaler , OneHotEncoder , LabelEncoder , FunctionTransformer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB , BernoulliNB , GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score , precision_score , classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# print(df.info())

# Separatin the Freatures and Target : 
target = 'Target'   # To predict spam or Ham
text_col = 'Processed_text' # Input  text

''' No need for this pipeline because we have just one int64 column but that is our target .'''
# num_pipeline = Pipeline([
#     (SimpleImputer(strategy='most_frequent')) ,
#     (FunctionTransformer(np.log1p , validate=True))
# ])

# This is the text pipeline which will run and transform text automatically .
text_pipeline = Pipeline([
    ('tf-idf' , TfidfVectorizer(max_features=3000))
])

# Combining pipeline through column Transformer : 
transformer = ColumnTransformer([
    ('txt_process' , text_pipeline , text_col)
])

# Pipelien for model we are using in this project : 
model_pipeline = Pipeline([
    ('preprocessor', transformer),
    ('classifier', MultinomialNB(fit_prior=True))
])


# Storing features in X : 
X = df.drop(['Target' ,'Text'] , axis=1)
# Storing target in y
y = df['Target']

# print(y.head())
# print('......................................................................')
# print(y.ndim)
# print('......................................................................')
# print(X.ndim)

# Splitting the data for training and testing : 
X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size=0.2 ,random_state=42)

# Training the model with training data : 
model_pipeline.fit(X_train , Y_train)

# Making Prediction on Testing data : 
Y_pred = model_pipeline.predict(X_test)

# Calculating metrics  : 

# print("Classification Report :\n ",classification_report(Y_test , Y_pred))
# print("Accuracy Score : ",accuracy_score(Y_test , Y_pred))
# print("Precision score : ",precision_score(Y_test , Y_pred , average='weighted' , zero_division=0))



'''                                                    DEPLOYMENT                                                          '''

import joblib
import streamlit as st

# Save the trained model
joblib.dump(model_pipeline, "spam_model.joblib")
print("✅ Model saved as spam_model.joblib")



'''                                    DEPLOYED VERSION IS IN :   app.py                               '''
