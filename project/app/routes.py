from flask import request, jsonify
from app import app
from app.model_loader import load_cnn_model, preprocess_data

@app.route('/')
def home():
    return 'Welcome to the Bank CNN Model API!'

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    preprocessed_data = preprocess_data(input_data)
    cnn_model = load_cnn_model()
    predictions = cnn_model.predict(preprocessed_data)
    response = {'predictions': predictions.tolist()}
    return jsonify(response)
