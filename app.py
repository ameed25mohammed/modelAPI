from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# تحميل النموذج
model = load_model('falin_ann_model.h5')

# إعداد تطبيق Flask
app = Flask(__name__)

# نقطة النهاية للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات بصيغة JSON
        data = request.get_json()
        input_data = np.array(data['input'])  # قائمة من القيم
        
        # التأكد من الأبعاد
        input_data = input_data.reshape(1, -1)  # تحويلها إلى صف واحد
        
        # إجراء التنبؤ
        prediction_prob = model.predict(input_data)[0][0]
        prediction = int(prediction_prob > 0.5)
        
        return jsonify({
            'prediction': prediction,
            'probability': float(prediction_prob)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# تشغيل السيرفر
if __name__ == '__main__':
    app.run(debug=True)
