# SMS Spam Classifier

import joblib
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# 1. LOAD AND CLEAN DATA

df = pd.read_csv("spam.csv", encoding="latin1")

# Remove unused columns
df.drop(
    columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    inplace=True
)

# Rename columns
df.rename(
    columns={"v1": "Target", "v2": "Text"},
    inplace=True
)

# Encode: 0 = Ham, 1 = Spam
encoder = LabelEncoder()
df["Target"] = encoder.fit_transform(df["Target"])

# Remove duplicate messages
df.drop_duplicates(inplace=True)


# 2. EXPLORATORY DATA ANALYSIS

print(df["Target"].value_counts())

plt.pie(
    df["Target"].value_counts(),
    labels=["Ham", "Spam"],
    autopct="%0.2f"
)

plt.title("Class Distribution")
# plt.show()


# 3. TEXT PREPROCESSING

# Run these once if NLTK resources are not installed:
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")

ps = PorterStemmer()


def text_pre_process(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove special characters
    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # Remove stopwords and punctuation
    tokens = [
        word for word in tokens
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]

    # Stemming
    tokens = [
        ps.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


df["Processed_text"] = df["Text"].apply(text_pre_process)


# 4. WORD FREQUENCY ANALYSIS

spam_corpus = []

for text in df.loc[df["Target"] == 1, "Processed_text"]:
    spam_corpus.extend(text.split())

spam_word_count = pd.DataFrame(
    Counter(spam_corpus).most_common(40),
    columns=["word", "count"]
)


ham_corpus = []

for text in df.loc[df["Target"] == 0, "Processed_text"]:
    ham_corpus.extend(text.split())

ham_word_count = pd.DataFrame(
    Counter(ham_corpus).most_common(40),
    columns=["word", "count"]
)

# print(spam_word_count)
# print(ham_word_count)


# 5. MODEL BUILDING

text_col = "Processed_text"

X = df.drop(
    columns=["Target", "Text"]
)

y = df["Target"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. TF-IDF + NAIVE BAYES PIPELINE

text_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(max_features=3000)
    )
])


preprocessor = ColumnTransformer([
    (
        "text",
        text_pipeline,
        text_col
    )
])


model_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# 7. TRAIN MODEL

model_pipeline.fit(
    X_train,
    y_train
)


# 8. MAKE PREDICTIONS

y_pred = model_pipeline.predict(
    X_test
)


# 9. MODEL EVALUATION

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


# 10. SAVE MODEL

joblib.dump(
    model_pipeline,
    "spam_model.joblib"
)

print("Model saved as spam_model.joblib")


# DEPLOYMENT
# Streamlit deployment code is in app.py
