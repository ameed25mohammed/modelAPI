from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

# تحميل النماذج
try:
    # النموذج الأول - تحديد وجود الإدمان
    addiction_model = joblib.load('drug_addiction_random_forest_model.pkl')
    print("✅ Addiction Model loaded successfully")
except Exception as e:
    print(f"❌ Addiction Model loading failed: {e}")
    addiction_model = None

try:
    # النموذج الثاني - تحديد شدة الإدمان
    severity_model = joblib.load('random_forest_drug_usage_model_metadata.pkl')
    print("✅ Severity Model loaded successfully")
except Exception as e:
    print(f"❌ Severity Model loading failed: {e}")
    severity_model = None

# أسماء الأعمدة الصحيحة من الداتاسيت الفعلي (27 عمود)
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
@app.route('/predict-severity', methods=['OPTIONS'])
@app.route('/health', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API v4.0 - Dual Model System',
        'status': 'running',
        'addiction_model_loaded': addiction_model is not None,
        'severity_model_loaded': severity_model is not None,
        'total_features': len(feature_names),
        'endpoints': {
            '/predict': 'Primary addiction detection',
            '/predict-severity': 'Addiction severity classification (for addicted individuals)',
            '/health': 'System health check',
            '/features': 'List all required features'
        },
        'cors_enabled': True
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if (addiction_model is not None and severity_model is not None) else 'partial',
        'addiction_model_loaded': addiction_model is not None,
        'severity_model_loaded': severity_model is not None,
        'feature_count': len(feature_names),
        'expected_features': 27,
        'api_version': '4.0',
        'system_status': {
            'primary_model': 'online' if addiction_model is not None else 'offline',
            'severity_model': 'online' if severity_model is not None else 'offline'
        }
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
        'note': f'Send exactly {len(feature_names)} values in this exact order'
    })

def validate_input_data(data):
    """التحقق من صحة البيانات المدخلة"""
    if not data:
        return False, 'No JSON data provided', {
            'input': [0] * len(feature_names)
        }
        
    if 'input' not in data:
        return False, 'Missing "input" field in JSON', {
            'input': 'array of numbers'
        }
    
    input_values = data['input']
    
    # التحقق من عدد المتغيرات
    if len(input_values) != len(feature_names):
        return False, f'Expected exactly {len(feature_names)} features, but got {len(input_values)}', {
            'expected_count': len(feature_names),
            'received_count': len(input_values),
            'tip': 'Use GET /features to see all required features'
        }

    # التحقق من أن جميع القيم رقمية
    try:
        numeric_values = [float(val) for val in input_values]
        return True, numeric_values, None
    except (ValueError, TypeError) as e:
        return False, 'All input values must be numeric', str(e)

@app.route('/predict', methods=['POST'])
def predict_addiction():
    """النموذج الأول - تحديد وجود الإدمان"""
    try:
        # التحقق من تحميل النموذج
        if addiction_model is None:
            return jsonify({
                'error': 'Addiction model not loaded properly',
                'status': 'error'
            }), 500
            
        # التحقق من صحة البيانات
        data = request.get_json()
        is_valid, result, extra_info = validate_input_data(data)
        
        if not is_valid:
            return jsonify({
                'error': result,
                'status': 'error',
                'details': extra_info
            }), 400
        
        numeric_values = result

        # تحويل البيانات إلى DataFrame
        try:
            input_df = pd.DataFrame([numeric_values], columns=feature_names)
            print(f"✅ DataFrame created for addiction prediction: {input_df.shape}")
        except Exception as e:
            return jsonify({
                'error': 'Failed to create DataFrame for addiction model',
                'details': str(e),
                'status': 'error'
            }), 400
        
        # التنبؤ
        try:
            prediction = addiction_model.predict(input_df)[0]
            print(f"🎯 Addiction Prediction: {prediction}")
        except Exception as e:
            return jsonify({
                'error': 'Addiction model prediction failed',
                'details': str(e),
                'status': 'error'
            }), 500
        
        # حساب الاحتمالية
        probability = None
        try:
            prediction_prob = addiction_model.predict_proba(input_df)[0]
            if len(prediction_prob) == 2:
                probability = float(prediction_prob[1])  # احتمالية الإدمان
                print(f"📈 Addiction Probability: {probability}")
            else:
                probability = float(max(prediction_prob))
        except Exception as prob_error:
            print(f"⚠️ Addiction probability calculation failed: {prob_error}")
        
        # بناء النتيجة
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Addicted' if int(prediction) == 1 else 'Not Addicted',
            'is_addicted': bool(int(prediction) == 1),
            'status': 'success',
            'model_type': 'Random Forest - Addiction Detection',
            'dataset_features': len(feature_names),
            'api_version': '4.0'
        }

        if probability is not None:
            result.update({
                'probability': round(probability, 4),
                'confidence_percentage': f"{probability:.2%}",
                'confidence_score': round(probability * 100, 1)
            })

        print(f"✅ Addiction prediction successful: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"💥 Unexpected error in addiction prediction: {e}")
        return jsonify({
            'error': 'Internal server error in addiction prediction',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/predict-severity', methods=['POST'])
def predict_severity():
    """النموذج الثاني - تحديد شدة الإدمان (يُستخدم فقط للأشخاص المدمنين)"""
    try:
        # التحقق من تحميل النموذج
        if severity_model is None:
            return jsonify({
                'error': 'Severity model not loaded properly',
                'status': 'error'
            }), 500
            
        # التحقق من صحة البيانات
        data = request.get_json()
        is_valid, result, extra_info = validate_input_data(data)
        
        if not is_valid:
            return jsonify({
                'error': result,
                'status': 'error',
                'details': extra_info
            }), 400
        
        numeric_values = result

        # تحويل البيانات إلى DataFrame
        try:
            input_df = pd.DataFrame([numeric_values], columns=feature_names)
            print(f"✅ DataFrame created for severity prediction: {input_df.shape}")
        except Exception as e:
            return jsonify({
                'error': 'Failed to create DataFrame for severity model',
                'details': str(e),
                'status': 'error'
            }), 400
        
        # التنبؤ بشدة الإدمان
        try:
            severity_prediction = severity_model.predict(input_df)[0]
            print(f"🎯 Severity Prediction: {severity_prediction}")
        except Exception as e:
            return jsonify({
                'error': 'Severity model prediction failed',
                'details': str(e),
                'status': 'error'
            }), 500
        
        # حساب الاحتمالية
        severity_probability = None
        try:
            severity_prob = severity_model.predict_proba(input_df)[0]
            severity_probability = float(max(severity_prob))
            print(f"📈 Severity Probability: {severity_probability}")
        except Exception as prob_error:
            print(f"⚠️ Severity probability calculation failed: {prob_error}")
        
        # تحديد تسميات شدة الإدمان (قم بتخصيصها حسب نموذجك)
        severity_labels = {
            0: 'Mild Addiction',
            1: 'Moderate Addiction',
            2: 'Severe Addiction',
            3: 'Critical Addiction'
        }
        
        severity_label = severity_labels.get(int(severity_prediction), f'Level {int(severity_prediction)}')
        
        # بناء النتيجة
        result = {
            'prediction': int(severity_prediction),
            'severity_label': severity_label,
            'severity_level': int(severity_prediction),
            'status': 'success',
            'model_type': 'Random Forest - Severity Classification',
            'dataset_features': len(feature_names),
            'api_version': '4.0',
            'severity_levels': severity_labels
        }

        if severity_probability is not None:
            result.update({
                'probability': round(severity_probability, 4),
                'confidence_percentage': f"{severity_probability:.2%}",
                'confidence_score': round(severity_probability * 100, 1)
            })

        print(f"✅ Severity prediction successful: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"💥 Unexpected error in severity prediction: {e}")
        return jsonify({
            'error': 'Internal server error in severity prediction',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/predict-complete', methods=['POST'])
def predict_complete():
    """تنبؤ كامل - يستخدم النموذجين معاً"""
    try:
        # التحقق من تحميل النماذج
        if addiction_model is None:
            return jsonify({
                'error': 'Addiction model not loaded',
                'status': 'error'
            }), 500
            
        # التحقق من صحة البيانات
        data = request.get_json()
        is_valid, result, extra_info = validate_input_data(data)
        
        if not is_valid:
            return jsonify({
                'error': result,
                'status': 'error',
                'details': extra_info
            }), 400
        
        numeric_values = result
        input_df = pd.DataFrame([numeric_values], columns=feature_names)

        # الخطوة 1: تحديد وجود الإدمان
        addiction_prediction = addiction_model.predict(input_df)[0]
        addiction_prob = None
        try:
            addiction_prob_array = addiction_model.predict_proba(input_df)[0]
            addiction_prob = float(addiction_prob_array[1]) if len(addiction_prob_array) == 2 else float(max(addiction_prob_array))
        except:
            pass

        # بناء النتيجة الأساسية
        complete_result = {
            'addiction_prediction': int(addiction_prediction),
            'is_addicted': bool(int(addiction_prediction) == 1),
            'addiction_probability': addiction_prob,
            'status': 'success',
            'api_version': '4.0'
        }

        # الخطوة 2: إذا كان مدمن وهناك نموذج شدة، احسب شدة الإدمان
        if int(addiction_prediction) == 1 and severity_model is not None:
            try:
                severity_prediction = severity_model.predict(input_df)[0]
                severity_prob = None
                try:
                    severity_prob_array = severity_model.predict_proba(input_df)[0]
                    severity_prob = float(max(severity_prob_array))
                except:
                    pass

                severity_labels = {
                    0: 'Mild Addiction',
                    1: 'Moderate Addiction',
                    2: 'Severe Addiction',
                    3: 'Critical Addiction'
                }

                complete_result.update({
                    'severity_prediction': int(severity_prediction),
                    'severity_label': severity_labels.get(int(severity_prediction), f'Level {int(severity_prediction)}'),
                    'severity_probability': severity_prob,
                    'has_severity_analysis': True
                })
                
                print(f"✅ Complete prediction with severity: Addiction={addiction_prediction}, Severity={severity_prediction}")
                
            except Exception as sev_error:
                print(f"⚠️ Severity prediction failed: {sev_error}")
                complete_result.update({
                    'has_severity_analysis': False,
                    'severity_error': 'Severity analysis failed'
                })
        else:
            complete_result.update({
                'has_severity_analysis': False,
                'reason': 'Not addicted' if int(addiction_prediction) == 0 else 'Severity model not available'
            })

        return jsonify(complete_result)
    
    except Exception as e:
        print(f"💥 Unexpected error in complete prediction: {e}")
        return jsonify({
            'error': 'Internal server error in complete prediction',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/test', methods=['GET'])
def test_prediction():
    """اختبار سريع للنماذج"""
    # بيانات اختبار
    test_data = [0] * len(feature_names)
    
    results = {
        'test_data_length': len(test_data),
        'feature_count': len(feature_names)
    }
    
    # اختبار النموذج الأول
    if addiction_model is not None:
        try:
            test_df = pd.DataFrame([test_data], columns=feature_names)
            addiction_pred = addiction_model.predict(test_df)[0]
            results['addiction_model'] = {
                'status': 'working',
                'prediction': int(addiction_pred)
            }
        except Exception as e:
            results['addiction_model'] = {
                'status': 'failed',
                'error': str(e)
            }
    else:
        results['addiction_model'] = {'status': 'not_loaded'}
    
    # اختبار النموذج الثاني
    if severity_model is not None:
        try:
            test_df = pd.DataFrame([test_data], columns=feature_names)
            severity_pred = severity_model.predict(test_df)[0]
            results['severity_model'] = {
                'status': 'working',
                'prediction': int(severity_pred)
            }
        except Exception as e:
            results['severity_model'] = {
                'status': 'failed',
                'error': str(e)
            }
    else:
        results['severity_model'] = {'status': 'not_loaded'}
    
    return jsonify(results)

if __name__ == '__main__':
    print(f"🚀 Starting Dual Model API with {len(feature_names)} features")
    print(f"📊 Addiction Model loaded: {addiction_model is not None}")
    print(f"📊 Severity Model loaded: {severity_model is not None}")
    print(f"🔧 Total endpoints: 6")
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
