<h1 align="center">Spam Email Classification System</h1>

<p align="center">
  <b>A Machine Learning-based NLP system for real-time spam detection</b><br>
  Designed using TF-IDF feature engineering and Multinomial Naive Bayes
</p>

<hr>

<h2>Overview</h2>

<p>
This project presents a robust Natural Language Processing (NLP) pipeline that classifies textual messages into:
</p>

<ul>
  <li><b>Spam</b> — unsolicited, promotional, or malicious content</li>
  <li><b>Ham</b> — legitimate communication</li>
</ul>

<p>
The system is implemented using Scikit-learn and deployed via Streamlit, enabling real-time predictions.
</p>

<hr>

<h2>System Architecture</h2>

<p align="center">
<svg width="700" height="200">
  <defs>
    <linearGradient id="grad1">
      <stop offset="0%" stop-color="#00c6ff"/>
      <stop offset="100%" stop-color="#0072ff"/>
    </linearGradient>
  </defs>

  <!-- Boxes -->
  <rect x="10" y="60" width="120" height="60" rx="10" fill="url(#grad1)" opacity="0.8"/>
  <rect x="150" y="60" width="140" height="60" rx="10" fill="url(#grad1)" opacity="0.8"/>
  <rect x="310" y="60" width="140" height="60" rx="10" fill="url(#grad1)" opacity="0.8"/>
  <rect x="470" y="60" width="140" height="60" rx="10" fill="url(#grad1)" opacity="0.8"/>

  <!-- Text -->
  <text x="25" y="95" fill="white">Raw Text</text>
  <text x="165" y="95" fill="white">Preprocessing</text>
  <text x="325" y="95" fill="white">TF-IDF</text>
  <text x="490" y="95" fill="white">Classifier</text>

  <!-- Arrows -->
  <line x1="130" y1="90" x2="150" y2="90" stroke="black"/>
  <line x1="290" y1="90" x2="310" y2="90" stroke="black"/>
  <line x1="450" y1="90" x2="470" y2="90" stroke="black"/>
</svg>
</p>

<hr>

<h2>Methodology</h2>

<h3>1. Text Preprocessing</h3>

<ul>
  <li>Lowercasing</li>
  <li>Tokenization</li>
  <li>Stopword removal</li>
  <li>Punctuation filtering</li>
  <li>Stemming using Porter Stemmer</li>
</ul>

<p><b>Example Transformation:</b></p>

<pre>
Input:  "Congratulations!!! You've won $1000"
Output: "congratul you 1000"
</pre>

---

<h3>2. Feature Engineering</h3>

<p>
Text data is transformed into numerical representations using TF-IDF vectorization.
</p>

<p align="center">
<svg width="400" height="200">
  <rect x="20" y="50" width="80" height="80" fill="#00c6ff" opacity="0.7"/>
  <rect x="120" y="30" width="80" height="100" fill="#0072ff" opacity="0.7"/>
  <rect x="220" y="70" width="80" height="60" fill="#00c6ff" opacity="0.7"/>

  <text x="35" y="140">Word1</text>
  <text x="135" y="140">Word2</text>
  <text x="235" y="140">Word3</text>
</svg>
</p>

---

<h3>3. Classification Model</h3>

<p>
A Multinomial Naive Bayes classifier is used due to its efficiency and strong performance in text classification tasks.
</p>

<p align="center">
<svg width="300" height="200">
  <circle cx="150" cy="100" r="70" fill="url(#grad1)" opacity="0.8"/>
  <text x="110" y="105" fill="white">Naive Bayes</text>
</svg>
</p>

---

<h2>Prediction Output</h2>

<table>
  <tr>
    <th>Value</th>
    <th>Interpretation</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Spam</td>
  </tr>
  <tr>
    <td>0</td>
    <td>Not Spam</td>
  </tr>
</table>

---

<h2>Dataset</h2>

<p>
The model is trained on the SMS Spam Collection dataset.
</p>

<ul>
  <li>Ham: ~87%</li>
  <li>Spam: ~13%</li>
</ul>

---

<h2>Technology Stack</h2>

<ul>
  <li>Python</li>
  <li>Pandas, NumPy</li>
  <li>Scikit-learn</li>
  <li>NLTK</li>
  <li>Streamlit</li>
</ul>

---

<h2>Example Predictions</h2>

<pre>
Input: "Congratulations! You have won a free prize!"
Output: Spam

Input: "Hey, are we meeting today?"
Output: Not Spam
</pre>

---

<h2>Future Enhancements</h2>

<ul>
  <li>Integration with advanced classifiers (SVM, Ensemble methods)</li>
  <li>Transformer-based models (BERT)</li>
  <li>Multi-language support</li>
  <li>Email client integration</li>
</ul>

---

<h2>Limitations</h2>

<ul>
  <li>Dataset limited to SMS format</li>
  <li>Restricted vocabulary size</li>
  <li>Lack of contextual semantic understanding</li>
</ul>

---

<h2 align="center">Author</h2>

<p align="center">
Developed as a practical implementation of NLP and Machine Learning techniques for real-world applications.
</p>

---

<h2 align="center">Support</h2>

<p align="center">
If you find this project useful, consider starring the repository and contributing.
</p>
