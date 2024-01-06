from flask import Blueprint, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

bp = Blueprint('routes', __name__)

model = load_model('complete_model.h5')
model.load_weights('model_weights.h5')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction = model.predict(np.array(data['input_data']))
        result = "Attrited" if prediction > 0.5 else "Existing"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
