from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# تحميل النموذج
model = load_model('drug_addiction_model.h5')

# إعداد تطبيق Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API is running!',
        'status': 'success',
        'endpoints': {
            'predict': '/predict (POST)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات بصيغة JSON
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        if 'input' not in data:
            return jsonify({'error': 'Missing "input" key in request'}), 400
        
        input_data = np.array(data['input'])  # قائمة من القيم
        
        # التحقق من عدد المعاملات (حدد العدد الصحيح حسب نموذجك)
        expected_features = len(input_data)  # سيقبل أي عدد الآن
        
        # التأكد من الأبعاد
        input_data = input_data.reshape(1, -1)  # تحويلها إلى صف واحد
        
        # إجراء التنبؤ
        prediction_prob = model.predict(input_data)[0][0]
        prediction = int(prediction_prob > 0.5)
        
        return jsonify({
            'prediction': prediction,
            'probability': float(prediction_prob),
            'interpretation': 'High Risk' if prediction == 1 else 'Low Risk',
            'features_received': len(data['input'])
        })
        
    except ValueError as ve:
        return jsonify({
            'error': 'Invalid input data',
            'details': str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        # اختبار بسيط للنموذج
        test_input = np.zeros((1, 11))  # جرب مع 11 معامل
        _ = model.predict(test_input)
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 500

# تشغيل السيرفر
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
