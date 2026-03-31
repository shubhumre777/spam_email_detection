<h1 align="center">🚀 Spam Email Classifier Web App</h1>

<p align="center">
  <b>Machine Learning + NLP powered Spam Detection System</b><br>
  🚨 Detect Spam | ✅ Identify Legit Messages | ⚡ Real-time Prediction
</p>

<p align="center">
  <!-- Typing Animation -->
  <img src="https://readme-typing-svg.herokuapp.com?size=22&duration=3000&color=00FFAA&center=true&vCenter=true&width=600&lines=Spam+Detection+using+Machine+Learning;NLP+%2B+TF-IDF+%2B+Naive+Bayes;Real-time+Prediction+Web+App" />
</p>

<p align="center">
  <a href="https://spamemaildetection-a5ve47mcinjnq5vnsuakel.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20App-Click%20Here-success?style=for-the-badge&logo=streamlit">
  </a>
</p>

---

<h2>✨ Features</h2>

<ul>
  <li>⚡ Real-time spam detection</li>
  <li>🧠 NLP-based preprocessing</li>
  <li>📊 TF-IDF feature extraction</li>
  <li>🎯 High accuracy classification</li>
  <li>🎨 Clean Streamlit UI</li>
  <li>🚀 Fast & lightweight deployment</li>
</ul>

---

<h2>🧠 How It Works</h2>

<h3>🔹 1. Text Preprocessing</h3>

<p align="center">
  <img src="[https://raw.githubusercontent.com/ageron/handson-ml2/master/images/unsupervised_learning/nlp_pipeline.png](https://www.facebook.com/photo/?fbid=1174237634718784&set=a.466580212151200&__cft__[0]=AZbyx7mdEhBwxVqdkk9krDni030MMg9cWTLrTK4PcY2D7CXfx8ninMLm3qlRwjlPGQikYxrQGkhjSEZ9wcEF9pP-Oh32t0D6xNsDICH6RpIhrgWvg3bhgvKoZ6mDNn7Qlh1i8088NTi3V6dcGSRs2Jhu5rrcmdxytlyaJ0ZUIvTISQ&__tn__=EH-R)" width="600"/>
</p>

<ul>
  <li>Lowercasing</li>
  <li>Tokenization (NLTK)</li>
  <li>Stopword removal</li>
  <li>Punctuation removal</li>
  <li>Stemming (PorterStemmer)</li>
</ul>

<p><b>Example:</b><br>
"Congratulations!!! You've won $1000"<br>
➡️ <code>congratul you 1000</code>
</p>

---

<h3>🔹 2. Feature Extraction (TF-IDF)</h3>

<p align="center">
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_tfidf_001.png" width="600"/>
</p>

<ul>
  <li>Text → Numerical vectors</li>
  <li>Top 3000 features</li>
</ul>

---

<h3>🔹 3. Model (Naive Bayes)</h3>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Naive_Bayes_classifier.svg/800px-Naive_Bayes_classifier.svg.png" width="600"/>
</p>

<ul>
  <li>Multinomial Naive Bayes</li>
  <li>Fast & efficient for text classification</li>
</ul>

---

<h2>🔄 ML Pipeline</h2>

<p align="center">
  <img src="https://raw.githubusercontent.com/ageron/handson-ml2/master/images/unsupervised_learning/pipeline.png" width="700"/>
</p>

<pre>
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorizer
   ↓
Naive Bayes Model
   ↓
Prediction (Spam / Ham)
</pre>

---

<h2>📊 Dataset</h2>

<ul>
  <li>SMS Spam Collection Dataset</li>
</ul>

<table align="center">
  <tr>
    <th>Class</th>
    <th>Percentage</th>
  </tr>
  <tr>
    <td>✅ Ham</td>
    <td>~87%</td>
  </tr>
  <tr>
    <td>🚨 Spam</td>
    <td>~13%</td>
  </tr>
</table>

---

<h2>⚙️ Tech Stack</h2>

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,sklearn,pandas,numpy,streamlit,git" />
</p>

---

<h2>🧪 Example</h2>

<p><b>Input:</b><br>
<code>Congratulations! You have won a free prize!</code>
</p>

<p>➡️ 🚨 <b>SPAM</b></p>

<p><b>Input:</b><br>
<code>Hey, are we meeting today?</code>
</p>

<p>➡️ ✅ <b>NOT SPAM</b></p>

---

<h2>🚀 Run Locally</h2>

<pre>
git clone https://github.com/shubhumre777/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
streamlit run app.py
</pre>

---

<h2>🚀 Future Improvements</h2>

<ul>
  <li>🤖 Advanced ML models (SVM, Random Forest)</li>
  <li>🧠 Deep Learning (LSTM, BERT)</li>
  <li>☁️ Cloud deployment (AWS/GCP)</li>
  <li>📧 Gmail integration</li>
  <li>🌍 Multi-language support</li>
</ul>

---

<h2>⚠️ Limitations</h2>

<ul>
  <li>SMS dataset only</li>
  <li>Limited vocabulary</li>
  <li>No deep contextual understanding</li>
</ul>

---

<h2 align="center">👨‍💻 Author</h2>

<p align="center">
  Built with ❤️ using Machine Learning & NLP
</p>

---

<h2 align="center">⭐ Support</h2>

<p align="center">
  ⭐ Star the repo &nbsp;&nbsp; 🍴 Fork it &nbsp;&nbsp; 🚀 Contribute
</p>
