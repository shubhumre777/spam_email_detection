import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
import os

# ---- Set a local path for NLTK data ----
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# ---- Download required NLTK resources ----
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# ---- Initialize stemmer ----
ps = PorterStemmer()

# ---- Text preprocessing function ----
def text_pre_process(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

# ---- Load model ----
model = joblib.load("spam_model.joblib")
# vectorizer = joblib.load("vectorizer.joblib")  # Uncomment if needed

# ---- Streamlit App ----
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
            st.success("This message is NOT SPAM")
