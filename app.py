# app.py
import numpy as np
from tensorflow.keras.models import load_model
from flask import render_template, request, jsonify

# Load the pre-trained model and weights
model = load_model('/workspaces/model-deployment/complete_model.h5')
model.load_weights('/workspaces/model-deployment/model_weights.h5')

from app import create_app  # Import create_app function

app = create_app()  # Create Flask app instance

if __name__ == '__main__':
    app.run(debug=True)

# app/routes.py
from flask import Blueprint

bp = Blueprint('routes', __name__)

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
