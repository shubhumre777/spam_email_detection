import joblib
import streamlit as st
import streamlit.components.v1 as components
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
import os


# Configure the Streamlit page
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Configure the local directory for NLTK resources
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.insert(0, NLTK_DATA_DIR)


# Download required NLTK resources if they are not available
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}"
            if "punkt" in resource
            else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(
            resource,
            download_dir=NLTK_DATA_DIR,
            quiet=True
        )


# Initialize the Porter Stemmer
ps = PorterStemmer()


# Preprocess the input text
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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the trained machine learning model
model = joblib.load("spam_model.joblib")


# Add custom styling to the Streamlit application
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;

        background: linear-gradient(
            90deg,
            #4338ca,
            #6366f1,
            #0ea5e9
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-description {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        font-size: 22px;
        font-weight: 700;
        animation: resultIn 0.5s ease-out;
    }

    .spam-card {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #b91c1c;
    }

    .safe-card {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #15803d;
    }

    .rainbow-line {
        height: 4px;
        width: 100%;
        border-radius: 10px;

        background: linear-gradient(
            90deg,
            #ff0000,
            #ff7f00,
            #ffff00,
            #00ff00,
            #00ffff,
            #0000ff,
            #8b00ff,
            #ff0000
        );

        background-size: 200% 100%;
        animation: rainbowMove 4s linear infinite;

        margin: 30px 0;
    }

    @keyframes rainbowMove {
        0% {
            background-position: 0% 50%;
        }

        100% {
            background-position: 200% 50%;
        }
    }

    @keyframes resultIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Display the animated welcome section
welcome_animation = """
<!DOCTYPE html>
<html>

<head>

<style>

body {
    margin: 0;
    background: transparent;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.wrapper {
    height: 500px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.card {
    width: 850px;
    padding: 60px 45px;
    text-align: center;
    position: relative;
    overflow: hidden;

    background: rgba(255, 255, 255, 0.75);

    border: 1px solid rgba(0, 0, 0, 0.06);

    border-radius: 28px;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.08);

    backdrop-filter: blur(18px);

    animation: cardIn 1s ease-out;
}

.glow {
    position: absolute;

    width: 320px;
    height: 320px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.16),
            transparent 70%
        );

    top: -170px;
    left: -100px;

    animation: glowMove 7s ease-in-out infinite;
}

.glow2 {
    position: absolute;

    width: 280px;
    height: 280px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(14, 165, 233, 0.13),
            transparent 70%
        );

    bottom: -150px;
    right: -100px;

    animation: glowMove2 8s ease-in-out infinite;
}

.email {
    width: 90px;
    height: 65px;

    margin: 0 auto 30px;

    border: 2px solid #6366f1;

    border-radius: 14px;

    position: relative;

    animation: floating 3s ease-in-out infinite;
}

.email::before,
.email::after {
    content: "";

    position: absolute;

    width: 58px;
    height: 2px;

    background: #6366f1;

    top: 28px;
}

.email::before {
    left: 3px;
    transform: rotate(32deg);
}

.email::after {
    right: 3px;
    transform: rotate(-32deg);
}

.scan {
    position: absolute;

    width: 100%;
    height: 2px;

    left: 0;
    top: 0;

    background:
        linear-gradient(
            90deg,
            transparent,
            #6366f1,
            transparent
        );

    animation: scanLine 4s linear infinite;
}

.title {
    font-size: 52px;

    font-weight: 800;

    margin: 0;

    letter-spacing: -1.5px;

    background:
        linear-gradient(
            90deg,
            #4338ca,
            #6366f1,
            #0ea5e9
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation: titleIn 1.2s ease-out;
}

.description {
    max-width: 680px;

    margin: 20px auto 0;

    font-size: 18px;

    line-height: 1.7;

    color: #64748b;

    animation: textIn 1.5s ease-out;
}

.note {
    margin-top: 28px;

    font-size: 15px;

    color: #475569;

    opacity: 0;

    animation:
        noteIn 1s ease forwards;

    animation-delay: 1.2s;
}

@keyframes cardIn {
    from {
        opacity: 0;
        transform:
            translateY(30px)
            scale(0.97);
    }

    to {
        opacity: 1;
        transform:
            translateY(0)
            scale(1);
    }
}

@keyframes titleIn {
    from {
        opacity: 0;
        transform:
            translateY(25px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}

@keyframes textIn {
    from {
        opacity: 0;
        transform:
            translateY(15px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}

@keyframes noteIn {
    to {
        opacity: 1;
    }
}

@keyframes floating {
    0%, 100% {
        transform:
            translateY(0);
    }

    50% {
        transform:
            translateY(-10px);
    }
}

@keyframes scanLine {
    0% {
        top: 0%;
        opacity: 0;
    }

    20% {
        opacity: 1;
    }

    80% {
        opacity: 1;
    }

    100% {
        top: 100%;
        opacity: 0;
    }
}

@keyframes glowMove {
    0%, 100% {
        transform:
            translate(0, 0);
    }

    50% {
        transform:
            translate(80px, 60px);
    }
}

@keyframes glowMove2 {
    0%, 100% {
        transform:
            translate(0, 0);
    }

    50% {
        transform:
            translate(-70px, -50px);
    }
}

</style>

</head>

<body>

<div class="wrapper">

    <div class="card">

        <div class="glow"></div>

        <div class="glow2"></div>

        <div class="scan"></div>

        <div class="email"></div>

        <h1 class="title">
            Welcome to Spam Email Classifier
        </h1>

        <p class="description">
            A machine learning system designed to identify
            unwanted and potentially harmful messages,
            helping keep your inbox cleaner and safer.
        </p>

        <p class="note">
            Paste a message below and let the model analyze
            its content and classify it as Spam or Not Spam.
        </p>

    </div>

</div>

</body>

</html>
"""


components.html(
    welcome_animation,
    height=530
)


# Display the message analysis section
st.markdown(
    '<div class="main-title">Analyze Your Message</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'Paste an email or message below to check its classification.'
    '</div>',
    unsafe_allow_html=True
)


# Collect the email or message from the user
user_text = st.text_area(
    "Message",
    placeholder="Paste your email or message here...",
    height=180,
    label_visibility="collapsed"
)


# Run the classification when the user clicks the button
if st.button(
    "🔍 Check Message",
    use_container_width=True
):

    if user_text.strip() == "":
        st.warning("Please enter a message before checking.")

    else:
        processed = text_pre_process(user_text)

        input_df = pd.DataFrame(
            {
                "Processed_text": [processed]
            }
        )

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.markdown(
                """
                <div class="result-card spam-card">
                    🚨 This message is classified as SPAM
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                """
                <div class="result-card safe-card">
                    ✅ This message is classified as NOT SPAM
                </div>
                """,
                unsafe_allow_html=True
            )


# Add the animated rainbow separator
st.markdown(
    """
    <div class="rainbow-line"></div>
    """,
    unsafe_allow_html=True
)


# Explain the basic workflow of the classifier
st.subheader("How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### 01 · Enter

        Paste or type an email or message
        into the input box.
        """
    )

with col2:
    st.markdown(
        """
        ### 02 · Process

        The text is cleaned and transformed
        before being passed to the model.
        """
    )

with col3:
    st.markdown(
        """
        ### 03 · Classify

        The machine learning model predicts
        whether the message is Spam or Not Spam.
        """
    )
