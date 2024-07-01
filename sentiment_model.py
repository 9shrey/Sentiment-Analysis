from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load or define your TensorFlow sentiment analysis model here
# For the purpose of this example, we'll assume it's already trained and saved
model = tf.keras.models.load_model('sentiment_model.h5')

# Example tokenizer setup - you should load your own tokenizer
tokenizer = Tokenizer(num_words=5000)
# Assuming tokenizer is already fitted on your dataset

def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded)
    return np.argmax(prediction)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    sentiment = predict_sentiment(text)
    emoji = ''
    if sentiment == 0:
        emoji = '😢'
    elif sentiment == 1:
        emoji = '😐'
    elif sentiment == 2:
        emoji = '😊'
    return jsonify({'sentiment': int(sentiment), 'emoji': emoji})  # Convert sentiment to int explicitly

if __name__ == '__main__':
    app.run(debug=True)
