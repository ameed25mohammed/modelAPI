from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# تحميل النموذج
model = load_model('drug_addiction_model.h5')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API is running on Railway!',
        'status': 'success'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Missing input data'}), 400
            
        input_data = np.array(data['input']).reshape(1, -1)
        prediction_prob = model.predict(input_data)[0][0]
        prediction = int(prediction_prob > 0.5)
        
        return jsonify({
            'prediction': prediction,
            'probability': float(prediction_prob),
            'platform': 'Railway'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
