import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model and data once
model = load_model('model/chat_model.h5')
data = pickle.load(open("model/training_data.pkl", "rb"))
words = data['words']
classes = data['classes']

with open('intents.json') as file:
    intents = json.load(file)

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array([bag])

def get_response(msg):
    bow = bag_of_words(msg)
    res = model.predict(bow, verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I didn't understand that. Could you please rephrase?"