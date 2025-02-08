import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return tokens

# Load intents
with open('intents.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = clean_text(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump({'words': words, 'classes': classes}, open("model/training_data.pkl", "wb"))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Create and train model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model/chat_model.h5')
print("Training complete. Model saved.")