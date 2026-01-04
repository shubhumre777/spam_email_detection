import joblib
import streamlit as st
import nltk 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd

# This object is used to do stemming of the words , basically to bring the words to their root words .
ps = PorterStemmer()

# stop_words = set(stopwords.words('english'))
import os

# Create a local folder for NLTK data in your app
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add this path to NLTK data search path
nltk.data.path.append(nltk_data_dir)

# Download required resources into that folder
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)


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

model = joblib.load("spam_model.joblib")
# vectorizer = joblib.load("vectorizer.joblib")


st.title("Spam Emial Classifier")
user_text = st.text_area("Enter your email/message text:")

if st.button("Check"):
    if user_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed = text_pre_process(user_text)
        # Wrap into a DataFrame because pipeline expects 'Processed_text' column
        input_df = pd.DataFrame({"Processed_text": [processed]})
        
        pred = model.predict(input_df)[0]
        
        if pred == 1:
            st.warning("Alert!! This message is SPAM")
        else:
            st.success("This message is NOT_SPAM")


