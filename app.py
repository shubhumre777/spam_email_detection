import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
import os

# ---------------- NLTK FIX (DO NOT REMOVE) ----------------
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.insert(0, NLTK_DATA_DIR)

# Force punkt_tab compatibility (NLTK ≥3.9)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
# ----------------------------------------------------------

ps = PorterStemmer()

def text_pre_process(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model
model = joblib.load("spam_model.joblib")

# ---------------- STREAMLIT UI ----------------
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
            st.warning("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")
