from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

# تحميل النموذج
try:
    model = joblib.load('drug_addiction_random_forest_model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# أسماء الأعمدة الصحيحة من الداتاسيت الفعلي (28 عمود)
feature_names = [
    'Gender',
    'Education', 
    'Family relationship',
    'Financials of family',
    'Addicted person in family',
    'no. of friends',
    'Friends influence',
    "friends' houses at night",
    'Live_with_Alone',
    'Live_with_With Family/Relatives',
    'Live_with_Hostel/Hall',
    'Spend_most_time_Alone',
    'Spend_most_time_Family/ Relatives',
    'Spend_most_time_Friends',
    'Spend_most_time_Hostel',
    'Spend_most_time_Business',
    'Spend_most_time_Job/Work place',
    'Satisfied with workplace',
    'Living with drug user',
    'Smoking',
    'Enjoyable with',
    'If chance given to taste drugs',
    'Easy to control use of drug',
    'Withdrawal symptoms',
    'Conflict with law',
    'Failure in life',
    'Suicidal thoughts'
]

app = Flask(__name__)

# إضافة CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/predict', methods=['OPTIONS'])
@app.route('/health', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API v3.0 - Real Dataset',
        'status': 'running',
        'model_loaded': model is not None,
        'total_features': len(feature_names),
        'dataset': 'Drug_Addiction_transformed with Satisfied with workplace without age and spend school.csv',
        'cors_enabled': True
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'feature_count': len(feature_names),
        'expected_features': 28,
        'api_version': '3.0'
    })

@app.route('/features', methods=['GET'])
def get_features():
    """عرض قائمة بجميع المتغيرات المطلوبة"""
    features_list = []
    for i, feature in enumerate(feature_names):
        features_list.append({
            'index': i,
            'name': feature,
            'position': i + 1
        })
    
    return jsonify({
        'features': features_list,
        'total_count': len(feature_names),
        'note': 'Send exactly 28 values in this exact order'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التحقق من تحميل النموذج
        if model is None:
            return jsonify({
                'error': 'Model not loaded properly',
                'status': 'error'
            }), 500
            
        # التحقق من وجود البيانات
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error',
                'example': {
                    'input': [0] * 28
                }
            }), 400
            
        if 'input' not in data:
            return jsonify({
                'error': 'Missing "input" field in JSON',
                'status': 'error',
                'expected_format': {
                    'input': 'array of 28 numbers'
                }
            }), 400
        
        input_values = data['input']
        
        # التحقق من عدد المتغيرات
        if len(input_values) != len(feature_names):
            return jsonify({
                'error': f'Expected exactly {len(feature_names)} features, but got {len(input_values)}',
                'expected_count': len(feature_names),
                'received_count': len(input_values),
                'status': 'error',
                'tip': 'Use GET /features to see all required features'
            }), 400

        # التحقق من أن جميع القيم رقمية
        try:
            # تحويل إلى أرقام وتأكد من صحتها
            numeric_values = [float(val) for val in input_values]
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'All input values must be numeric',
                'status': 'error',
                'details': str(e)
            }), 400

        # تحويل البيانات إلى DataFrame مع الأعمدة الصحيحة
        try:
            input_df = pd.DataFrame([numeric_values], columns=feature_names)
            print(f"✅ DataFrame created successfully with shape: {input_df.shape}")
            print(f"📊 Sample values: {input_df.iloc[0].head(5).to_dict()}")
        except Exception as e:
            return jsonify({
                'error': 'Failed to create DataFrame',
                'details': str(e),
                'status': 'error'
            }), 400
        
        # التنبؤ
        try:
            prediction = model.predict(input_df)[0]
            print(f"🎯 Prediction: {prediction}")
        except Exception as e:
            return jsonify({
                'error': 'Model prediction failed',
                'details': str(e),
                'status': 'error'
            }), 500
        
        # حساب الاحتمالية
        probability = None
        try:
            prediction_prob = model.predict_proba(input_df)[0]
            if len(prediction_prob) == 2:
                probability = float(prediction_prob[1])  # احتمالية الفئة الإيجابية
                print(f"📈 Probability: {probability}")
            else:
                probability = float(max(prediction_prob))
        except Exception as prob_error:
            print(f"⚠️ Probability calculation failed: {prob_error}")
        
        # بناء النتيجة
        result = {
            'prediction': int(prediction),
            'prediction_label': 'High Risk' if int(prediction) == 1 else 'Low Risk',
            'status': 'success',
            'model_type': 'Random Forest',
            'dataset_features': len(feature_names),
            'api_version': '3.0'
        }

        if probability is not None:
            result.update({
                'probability': round(probability, 4),
                'confidence_percentage': f"{probability:.2%}",
                'confidence_score': round(probability * 100, 1)
            })

        print(f"✅ Prediction successful: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/test', methods=['GET'])
def test_prediction():
    """اختبار سريع للنموذج"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # بيانات اختبار (28 قيمة صفر)
    test_data = [0] * 28
    
    try:
        test_df = pd.DataFrame([test_data], columns=feature_names)
        prediction = model.predict(test_df)[0]
        
        try:
            probability = model.predict_proba(test_df)[0][1]
        except:
            probability = None
            
        return jsonify({
            'test_result': 'success',
            'prediction': int(prediction),
            'probability': probability,
            'test_data_length': len(test_data),
            'message': 'Model is working correctly'
        })
    except Exception as e:
        return jsonify({
            'test_result': 'failed',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print(f"🚀 Starting API with {len(feature_names)} features")
    print(f"📊 Dataset: Drug_Addiction_transformed")
    print(f"🔧 Model loaded: {model is not None}")
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
