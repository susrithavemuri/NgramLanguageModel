# ngram_utils.py
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
nltk.download('punkt')

def preprocess_text(text):
    return word_tokenize(text.lower())

def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def laplace_probability(bigram, bigram_counts, unigram_counts, V):
    first_word = bigram[0]
    return (bigram_counts[bigram] + 1) / (unigram_counts[first_word] + V)

def good_turing_counts(bigram_counts):
    count_of_counts = Counter(bigram_counts.values())
    max_count = max(count_of_counts)
    gt_counts = {}
    for bigram in bigram_counts:
        c = bigram_counts[bigram]
        if c < max_count:
            gt_counts[bigram] = (c + 1) * (count_of_counts[c + 1] / count_of_counts[c])
        else:
            gt_counts[bigram] = c
    return gt_counts

def good_turing_probability(bigram, gt_bigram_counts, unigram_counts):
    first_word = bigram[0]
    total_count = sum(gt_bigram_counts[bg] for bg in gt_bigram_counts if bg[0] == first_word)
    if total_count == 0:
        return 1 / len(unigram_counts)
    return gt_bigram_counts.get(bigram, 0) / total_count
