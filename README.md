# N-Gram Language Model with Smoothing (Streamlit App)

This project implements a Bigram (2-gram) Language Model using Natural Language Processing. It supports two smoothing techniques—**Laplace Smoothing** and **Good-Turing Smoothing**—to estimate probabilities for seen and unseen word pairs. The model is interactive and runs in a Streamlit web app.

🚀 Features

- 📁 Upload your own `.txt` corpus
- 🧠 Generate bigrams and count their frequencies
- 📊 Apply Laplace and Good-Turing smoothing
- 🔍 Enter custom bigrams to check their probabilities
- 📈 Compare probabilities side-by-side

---

## 🛠️ Requirements

Install the required Python packages:
pip install -r requirements.txt


## ▶️ How to Run

streamlit run app.py

## 📂 File Structure

ngram_language_model/
│
├── app.py                # Main Streamlit application
├── ngram_utils.py        # Utility functions for preprocessing and probability calculation
├── requirements.txt      # Python dependencies
└── long_sample_corpus.txt# Example text corpus


## 📚 Smoothing Techniques
Laplace Smoothing: Adds 1 to every bigram count to avoid zero probabilities.
Good-Turing Smoothing: Recalculates probability based on the frequency of frequency counts.

## 🧪 Example Test Bigrams
("language", "models")
("speech", "recognition")
("deep", "learning") (Unseen bigram)

## 🧠 Educational Goals
This project is great for:
Understanding the mechanics of N-gram models
Seeing the effects of smoothing
Hands-on exploration of NLP probability models

