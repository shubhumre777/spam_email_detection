# 📧 Spam Email Classifier Web App

A modern **Machine Learning-powered web application** that classifies messages as:

- 🚨 **Spam** → Unwanted / promotional / malicious messages  
- ✅ **Not Spam (Ham)** → Legitimate messages  

Built using **NLP + Scikit-learn + Streamlit**, this app provides **real-time predictions** with a clean UI.

---

## 🚀 Overview

This project uses **Natural Language Processing (NLP)** techniques and a **Multinomial Naive Bayes model** to classify text messages.

Users can input any message, and the app instantly predicts whether it is spam or not.

---

## 🧠 How It Works

### 🔹 1. Text Preprocessing
- Convert text to lowercase  
- Tokenization using NLTK  
- Remove:
  - Stopwords  
  - Punctuation  
  - Non-alphanumeric characters  
- Apply **stemming** using PorterStemmer  

Example:
"Congratulations!!! You've won $1000"
→ "congratul you 1000"

---

### 🔹 2. Feature Extraction
- Uses **TF-IDF Vectorization**
- Converts text into numerical features
- Limits features to top **3000 words**

---

### 🔹 3. Model
- **Multinomial Naive Bayes**
- Integrated with preprocessing using **Scikit-learn Pipeline**

---

### 🔹 4. Prediction Output

| Value | Meaning |
|------|--------|
| 1 | 🚨 Spam |
| 0 | ✅ Not Spam |

---

## 🏗️ Project Structure

Spam-Classifier/
│
├── app.py
├── spam_model.joblib
├── spam.csv
├── nltk_data/
├── requirements.txt
└── README.md

---

## 📊 Dataset

- SMS Spam Collection Dataset  
- Columns:
  - `Text` → Message content  
  - `Target` → Label  

Distribution:
- ✅ Ham: ~87%  
- 🚨 Spam: ~13%  

---

## ⚙️ Tech Stack

### 🔹 Backend / ML
- Python  
- Pandas  
- NumPy  
- Scikit-learn (1.6.1)  
- NLTK  
- Joblib  

### 🔹 Frontend
- Streamlit  

---

## 🧪 ML Pipeline

Raw Text  
↓  
Text Preprocessing (NLP)  
↓  
TF-IDF Vectorizer  
↓  
Multinomial Naive Bayes  
↓  
Prediction (Spam / Ham)  

---

## 🖥️ Run Locally

1. Clone Repository  
git clone https://github.com/your-username/spam-classifier.git  
cd spam-classifier  

2. Install Dependencies  
pip install -r requirements.txt  

3. Run App  
streamlit run app.py  

4. Open in Browser  
http://localhost:8501  

---

## 📦 Requirements

streamlit  
pandas  
numpy  
scikit-learn==1.6.1  
nltk  
joblib  

---

## ✨ Features

- ⚡ Real-time spam detection  
- 🧠 NLP-based preprocessing  
- 📊 TF-IDF feature engineering  
- 🎯 High precision classification  
- 💡 Clean Streamlit UI  
- 📁 Lightweight deployment  

---

## 🔍 Example

Input:
"Congratulations! You have won a free prize!"

Output:
🚨 This message is SPAM

Input:
"Hey, are we meeting today?"

Output:
✅ This message is NOT SPAM

---

## 📈 Future Improvements

### 🔹 Scenario 1: Advanced ML Models
- Logistic Regression  
- SVM  
- Random Forest  

### 🔹 Scenario 2: Deep Learning
- LSTM / GRU  
- BERT / Transformers  

### 🔹 Scenario 3: Production Deployment
- Streamlit Cloud  
- AWS / GCP  
- Docker  

### 🔹 Scenario 4: Real-world Features
- Gmail integration  
- Spam confidence score  
- Multi-language support  

---

## ⚠️ Limitations

- Dataset is SMS-based (not full email data)  
- Limited vocabulary (3000 features)  
- No deep contextual understanding  

---

## 👨‍💻 Author

Built for learning and real-world ML deployment using NLP.

---

## ⭐ Support

If you like this project:

- ⭐ Star the repository  
- 🍴 Fork it  
- 🚀 Improve it  
