from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipeline from Hugging Face Transformers
classifier = pipeline("sentiment-analysis")

# Define emoji mapping for sentiment labels
emoji_mapping = {
    'LABEL_1': '😞',  # Negative
    'LABEL_2': '😐',  # Neutral
    'LABEL_3': '😊'   # Positive
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    results = classifier(text)

    # Extract sentiment label and score from the result
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']

    # Map sentiment label to emoji
    emoji = emoji_mapping.get(sentiment_label, 'Unknown')

    return jsonify({'sentiment': sentiment_label, 'emoji': emoji, 'confidence': sentiment_score})

if __name__ == '__main__':
    app.run(debug=True)
