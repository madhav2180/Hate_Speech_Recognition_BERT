
import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the trained model
model_clf = joblib.load('trained_model1.pkl')

# Load pre-trained GloVe embeddings
def load_glove_embeddings(embeddings_file):
    embeddings_index = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings_file = "glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_embeddings_file)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Clean input text
def clean_tweet(tweet):
    tweet = re.sub("#", "", tweet)
    tweet = re.sub("[^a-zA-Z#]", " ", tweet)
    tweet = re.sub(r'http[s]?://\\S+', "<URL>", tweet)
    tweet = re.sub('http', '', tweet)
    tweet = re.sub(" +", " ", tweet)
    tweet = tweet.lower()
    tweet = word_tokenize(tweet)
    return [word for word in tweet if word not in stop_words]

# Extract features from tweet using GloVe embeddings
def get_features(tweet):
    features = [glove_embeddings[word] for word in tweet if word in glove_embeddings]
    return np.mean(features, axis=0) if features else np.zeros(100)

# Predict class label for input text
def predict_class(input_tweet, class_labels):
    preprocessed_input_tweet = clean_tweet(input_tweet)
    input_features = get_features(preprocessed_input_tweet)
    class_probabilities = model_clf.predict_proba(input_features.reshape(1, -1))
    return class_labels[np.argmax(class_probabilities)]

# Streamlit UI
st.title('Hate Speech Classification')
input_text = st.text_area("Enter a comment:")

if st.button("Predict"):
    if input_text:
        class_labels = ["Hate Speech", "Offensive Language", "Normal Language"]
        prediction = predict_class(input_text, class_labels)
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text to classify.")
