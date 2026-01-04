import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd

# ---- Cache NLTK downloads to avoid repeated downloads ----
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_resources()

# ---- Initialize stemmer ----
ps = PorterStemmer()

# ---- Text preprocessing function ----
def text_pre_process(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    tokens = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    tokens = [i for i in tokens if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Stemming
    tokens = [ps.stem(i) for i in tokens]
    
    # Join tokens back to string
    return " ".join(tokens)

# ---- Load model ----
model = joblib.load("spam_model.joblib")
# vectorizer = joblib.load("vectorizer.joblib")  # Uncomment if your pipeline needs it

# ---- Streamlit App ----
st.title("Spam Email Classifier")

user_text = st.text_area("Enter your email/message text:")

if st.button("Check"):
    if user_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed = text_pre_process(user_text)
        # Make sure pipeline input matches expected column
        input_df = pd.DataFrame({"Processed_text": [processed]})
        
        pred = model.predict(input_df)[0]
        
        if pred == 1:
            st.warning("Alert!! This message is SPAM")
        else:
            st.success("This message is NOT SPAM")
