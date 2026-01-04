import joblib
import streamlit as st
import nltk 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd

# ---------------------------
# Ensure NLTK data is available at runtime
# ---------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Initialize stemmer
ps = PorterStemmer()

# ---------------------------
# Text preprocessing function
# ---------------------------
def text_pre_process(text):
    text = text.lower()                     # lowercase
    text = nltk.word_tokenize(text)         # tokenize

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
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

# ---------------------------
# Load the model
# ---------------------------
model = joblib.load("spam_model.joblib")

# ---------------------------
# Streamlit interface
# ---------------------------
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
