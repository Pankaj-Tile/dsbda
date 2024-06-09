# Text Analytics
# 1. Extract Sample document and apply following document preprocessing methods:
# Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
# 2. Create representation of document by calculating Term Frequency and Inverse Document
# Frequency.

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Sample document
sample_document = """Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans using natural language. It enables computers to understand, interpret, and generate human-like text. NLP involves various tasks, such as tokenization, part-of-speech tagging, stop words removal, stemming, and lemmatization."""

# Tokenization
tokens = word_tokenize(sample_document)

# POS Tagging
pos_tags = pos_tag(tokens)

# Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming
porter_stemmer = PorterStemmer()
stemmed_tokens = [porter_stemmer.stem(word) for word in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# TF-IDF Representation
documents = [sample_document]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Display Results
print("Original Document:\n", sample_document, "\n")
print("Tokenization:\n", tokens, "\n")
print("POS Tagging:\n", pos_tags, "\n")
print("Stop Words Removal:\n", filtered_tokens, "\n")
print("Stemming:\n", stemmed_tokens, "\n")
print("Lemmatization:\n", lemmatized_tokens, "\n")
print("TF-IDF Representation:\n", tfidf_matrix.toarray(), "\n")
print("Feature Names:\n", feature_names)


