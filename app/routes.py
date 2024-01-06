# app/routes.py
from flask import Blueprint, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Create a Blueprint
bp = Blueprint('routes', __name__)

# Load the pre-trained model and weights
model = load_model('complete_model.h5')
model.load_weights('model_weights.h5')

# Define routes
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming your model expects some input data in JSON format
        data = request.get_json()

        # Preprocess the input data as needed (convert to numpy array, scale, etc.)
        # Make predictions using your loaded model
        prediction = model.predict(np.array(data['input_data']))

        # Assuming binary classification
        result = "Attrited" if prediction > 0.5 else "Existing"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})
