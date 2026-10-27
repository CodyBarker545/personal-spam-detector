from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
from NLPClassifierSpamDetection import TextPreprocessor, FeatureEngineer, stemming_remove_url
import joblib


app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Load the trained pipeline
model = joblib.load('spam_detector_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    prediction = model.predict(pd.Series([comment]))[0]
    label = 'Spam' if prediction == 1 else 'Not Spam'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
