import joblib
import streamlit as st
import nltk 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd

# ---------------------------------------------------
# NLTK FIX FOR STREAMLIT CLOUD (DO NOT REMOVE)
# ---------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        if resource in ["punkt", "punkt_tab"]:
            nltk.data.find(f"tokenizers/{resource}")
        else:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
# ---------------------------------------------------

# This object is used to do stemming of the words
ps = PorterStemmer()

def text_pre_process(text):
    # This will make the text to lower case
    text = text.lower()

    # This will break the text to tokens
    text = nltk.word_tokenize(text)
    
    # Remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords and punctuations
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load trained model
model = joblib.load("spam_model.joblib")

# Streamlit UI
st.title("Spam Email Classifier")
user_text = st.text_area("Enter your email/message text:")

if st.button("Check"):
    if user_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed = text_pre_process(user_text)
        input_df = pd.DataFrame({"Processed_text": [processed]})
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.warning("Alert!! This message is SPAM")
        else:
            st.success("This message is NOT_SPAM")
