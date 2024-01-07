from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('complete_model.h5')
model.load_weights('complete_model.h5')

# Sample function for processing input data and making predictions
def predict_churn(features):
    # Preprocess the input features if needed
    processed_features = preprocess_data(features)
    
    # Make predictions
    predictions = model.predict(np.array([processed_features]))
    
    # Assuming it's a binary classification, you may need to adjust accordingly
    churn_probability = predictions[0][0]
    churn_label = 'Attrited' if churn_probability > 0.5 else 'Existing'
    
    return churn_label, churn_probability

# Sample preprocessing function, replace with your actual preprocessing logic
def preprocess_data(features):
    # Implement your preprocessing steps here
    # Make sure to format the input data in the same way it was during training
    processed_features = features  # Replace with your actual preprocessing logic
    return processed_features

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # Get features from the request data
        features = request.json['features']
        
        # Make prediction
        churn_label, churn_probability = predict_churn(features)
        
        # Prepare the response
        response = {
            'prediction': churn_label,
            'probability': float(churn_probability)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
