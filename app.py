from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import gdown
import os

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define model path and Google Drive URL
model_url = "https://drive.google.com/uc?id=1eISm1ObAs8-icIS-qhDptuMgwYP8PUl-"
model_path = "lstm100.pkl"

# Download the model from Google Drive if not already downloaded
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the model
mod = pickle.load(open(model_path, 'rb'))

# Load Word2Vec model from TensorFlow Hub
word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

# Preprocess text
def preprocess_text(sentence):
    sentence = sentence.lower()
    for punc in string.punctuation:
        sentence = sentence.replace(punc, '')
    tok = sentence.split()
    lemmatizer = WordNetLemmatizer()
    tok = [lemmatizer.lemmatize(word) for word in tok if word not in stopwords.words('english')]
    return tok

# Make prediction
def make_prediction(text):
    tokens = preprocess_text(text)
    vectors = [np.array(word2vec([word])) for word in tokens]
    vectors = np.array(vectors)
    results = mod.predict(vectors)
    result_str = "Fake" if results[0][0] >= 0.5 else "Real"
    confidence = round(results[0][0] * 100, 3)
    return {"result": result_str, "confidence": f"{confidence}%"}

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    response = make_prediction(text)
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
