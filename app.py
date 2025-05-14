# app.py
import streamlit as st
from ngram_utils import *
from collections import Counter

st.title("\U0001F524 N-gram Language Model with Smoothing")

# Step 1: Upload or enter corpus
st.header("Step 1: Upload or Enter Corpus")
input_method = st.radio("Choose input method", ("Type or paste text", "Upload .txt file"))

corpus = ""
if input_method == "Type or paste text":
    corpus = st.text_area("Enter corpus text here", height=200)
else:
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    if uploaded_file:
        corpus = uploaded_file.read().decode("utf-8")

if corpus:
    st.header("Step 2: Configure N-gram Model")
    n = st.slider("Select N for N-grams", 1, 5, 2)

    tokens = preprocess_text(corpus)
    ngrams_list = generate_ngrams(tokens, n)
    ngram_counts = Counter(ngrams_list)
    unigram_counts = Counter(tokens)
    V = len(unigram_counts)

    st.subheader("Top N-grams")
    top_n = st.slider("How many top N-grams to show?", 5, 50, 10)
    top_ngrams = ngram_counts.most_common(top_n)
    st.table(top_ngrams)

    st.subheader("Probability Estimation")
    bigram_input = st.text_input("Enter a bigram (e.g., 'language models')")

    if bigram_input and len(bigram_input.split()) == 2:
        bigram = tuple(bigram_input.lower().split())
        laplace_p = laplace_probability(bigram, ngram_counts, unigram_counts, V)
        gt_counts = good_turing_counts(ngram_counts)
        gt_p = good_turing_probability(bigram, gt_counts, unigram_counts)

        st.write(f"**Laplace Probability**: {laplace_p:.6f}")
        st.write(f"**Good-Turing Probability**: {gt_p:.6f}")
    elif bigram_input:
        st.warning("Please enter exactly two words for a bigram.")
