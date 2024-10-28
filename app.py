from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import gdown
import os
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize FastAPI
app = FastAPI()

# Google Drive Model URL and local path
model_url = "https://drive.google.com/uc?id=1eISm1ObAs8-icIS-qhDptuMgwYP8PUl-"
model_path = "lstm100.pkl"

# Download the model if it's not already downloaded
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the model
with open(model_path, 'rb') as file:
    mod = pickle.load(file)

# Load Word2Vec model
word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

# Text preprocessing
def preprocess_text(sentence):
    sentence = sentence.lower()
    for punc in string.punctuation:
        sentence = sentence.replace(punc, '')
    tok = sentence.split()
    lemmatizer = WordNetLemmatizer()
    tok = [lemmatizer.lemmatize(word) for word in tok if word not in stopwords.words('english')]
    return tok

# Define input format
class TextInput(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
async def predict(input: TextInput):
    tokens = preprocess_text(input.text)
    vectors = [np.array(word2vec([word])) for word in tokens]
    vectors = np.array(vectors)
    results = mod.predict(vectors)
    
    result_str = "Fake" if results[0][0] >= 0.5 else "Real"
    confidence = round(results[0][0] * 100, 3)
    
    return {"result": result_str, "confidence": f"{confidence}%"}

# Error handling example (if needed)
@app.get("/")
async def root():
    return {"message": "Welcome to the Fake News Detection API"}
