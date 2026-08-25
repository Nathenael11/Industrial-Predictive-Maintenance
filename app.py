import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, session, send_file, redirect
import io

app = Flask(__name__)
app.secret_key = 'elevvo_predictive_maintenance_secret_key_2026'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')

# Load Artifacts
model = None
scaler = None
optimal_threshold = 0.76

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
            optimal_threshold = metrics_data.get('best_threshold', 0.76)
    except Exception:
        pass

FEATURE_COLS = [
    'Air_temperature_K',
    'Process_temperature_K',
    'Rotational_speed_rpm',
    'Torque_Nm',
    'Tool_wear_min',
    'Type_H',
    'Type_L',
    'Type_M'
]

def preprocess_input_df(raw_df):
    """
    Transforms raw user input dataframe into feature vector matching model schema.
    Accepts column names in various formats (e.g. 'Air temperature [K]' or 'air_temp')
    """
    df = raw_df.copy()
    
    col_mapping = {
        'Air temperature [K]': 'Air_temperature_K',
        'air_temp': 'Air_temperature_K',
        'Air_temperature_K': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'process_temp': 'Process_temperature_K',
        'Process_temperature_K': 'Process_temperature_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'rotational_speed': 'Rotational_speed_rpm',
        'Rotational_speed_rpm': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'torque': 'Torque_Nm',
        'Torque_Nm': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min',
        'tool_wear': 'Tool_wear_min',
        'Tool_wear_min': 'Tool_wear_min',
        'Type': 'Type',
        'product_type': 'Type'
    }
    
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Handle One-Hot Encoding for Type
    if 'Type' in df.columns:
        type_series = df['Type'].astype(str).str.upper()
        df['Type_L'] = (type_series == 'L').astype(int)
        df['Type_M'] = (type_series == 'M').astype(int)
        df['Type_H'] = (type_series == 'H').astype(int)
        df = df.drop(columns=['Type'])
    else:
        for t in ['Type_L', 'Type_M', 'Type_H']:
            if t not in df.columns:
                df[t] = 0

    # Ensure numeric columns are float/int
    num_cols = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0

    # Reorder exactly to FEATURE_COLS
    df_features = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df_features

def generate_recommendation(row_dict, is_failure, proba):
    """
    Generates domain-specific maintenance action recommendations.
    """
    air_temp = float(row_dict.get('Air_temperature_K', 300))
    proc_temp = float(row_dict.get('Process_temperature_K', 310))
    torque = float(row_dict.get('Torque_Nm', 40))
    speed = float(row_dict.get('Rotational_speed_rpm', 1500))
    tool_wear = float(row_dict.get('Tool_wear_min', 0))
    temp_diff = proc_temp - air_temp

    recs = []
    if is_failure:
        if tool_wear >= 200:
            recs.append("CRITICAL: Replace worn cutting tool immediately (Tool wear >= 200 min).")
        if torque >= 60 or speed <= 1200:
            recs.append("HIGH RISK: Inspect spindle and gear drive for overstrain and mechanical friction.")
        if temp_diff < 8.6 or proc_temp > 313:
            recs.append("THERMAL WARNING: Check heat dissipation, coolant flow, and thermal ventilation.")
        if not recs:
            recs.append("IMMEDIATE ACTION: Schedule emergency diagnostic inspection before severe failure.")
    else:
        if tool_wear >= 180:
            recs.append("ATTENTION: Tool wear approaching threshold limit (>= 180 min). Schedule replacement during next shift.")
        elif torque >= 55:
            recs.append("MONITOR: High torque detected. Keep under close observation.")
        else:
            recs.append("NORMAL: Machine telemetry operating within standard nominal parameters.")
            
    return " | ".join(recs)

@app.route('/')
def index():
    history = session.get('prediction_history', [])
    return render_template('index.html', history=history, threshold=optimal_threshold)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not initialized. Run train_model.py first.'}), 500

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Parse & Validate input values
        air_temp = float(data.get('air_temp', data.get('Air_temperature_K', 300.0)))
        proc_temp = float(data.get('process_temp', data.get('Process_temperature_K', 310.0)))
        rot_speed = float(data.get('rotational_speed', data.get('Rotational_speed_rpm', 1500.0)))
        torque = float(data.get('torque', data.get('Torque_Nm', 40.0)))
        tool_wear = float(data.get('tool_wear', data.get('Tool_wear_min', 0.0)))
        prod_type = str(data.get('product_type', data.get('Type', 'L'))).upper()

        if prod_type not in ['L', 'M', 'H']:
            prod_type = 'L'

        raw_dict = {
            'Air_temperature_K': air_temp,
            'Process_temperature_K': proc_temp,
            'Rotational_speed_rpm': rot_speed,
            'Torque_Nm': torque,
            'Tool_wear_min': tool_wear,
            'Type': prod_type
        }

        input_df = pd.DataFrame([raw_dict])
        features_df = preprocess_input_df(input_df)

        # Prediction probability
        proba = float(model.predict_proba(features_df)[0, 1])
        risk_score = round(proba * 100, 2)
        is_failure = proba >= optimal_threshold
        prediction = "Failure" if is_failure else "No Failure"

        # Confidence metric calculation
        confidence = round((abs(proba - optimal_threshold) / (1.0 - optimal_threshold if proba >= optimal_threshold else optimal_threshold) * 50 + 50), 2)
        confidence = min(max(confidence, 50.0), 99.9)

        rec_action = generate_recommendation(raw_dict, is_failure, proba)

        result = {
            'risk_score': risk_score,
            'prediction': prediction,
            'is_failure': bool(is_failure),
            'confidence': confidence,
            'threshold_used': optimal_threshold,
            'recommended_action': rec_action,
            'inputs': raw_dict
        }

        # Store in session history (last 10 predictions)
        history = session.get('prediction_history', [])
        history.insert(0, result)
        session['prediction_history'] = history[:10]
        session.modified = True

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'GET':
        return render_template('batch.html')

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({'error': 'Uploaded CSV is empty'}), 400

        features_df = preprocess_input_df(df)
        probas = model.predict_proba(features_df)[:, 1]

        results = []
        failures_count = 0

        for i, proba in enumerate(probas):
            r_score = round(float(proba) * 100, 2)
            is_fail = bool(proba >= optimal_threshold)
            if is_fail:
                failures_count += 1

            row_raw = df.iloc[i].to_dict()
            rec = generate_recommendation(row_raw, is_fail, float(proba))

            results.append({
                'row_id': i + 1,
                'type': str(row_raw.get('Type', row_raw.get('product_type', 'L'))),
                'air_temp': float(row_raw.get('Air temperature [K]', row_raw.get('air_temp', 300))),
                'proc_temp': float(row_raw.get('Process temperature [K]', row_raw.get('process_temp', 310))),
                'speed': float(row_raw.get('Rotational speed [rpm]', row_raw.get('rotational_speed', 1500))),
                'torque': float(row_raw.get('Torque [Nm]', row_raw.get('torque', 40))),
                'tool_wear': float(row_raw.get('Tool wear [min]', row_raw.get('tool_wear', 0))),
                'risk_score': r_score,
                'prediction': 'Failure' if is_fail else 'No Failure',
                'is_failure': is_fail,
                'action': rec
            })

        total = len(results)
        failure_rate = round((failures_count / total) * 100, 2) if total > 0 else 0
        avg_risk = round(sum(r['risk_score'] for r in results) / total, 2) if total > 0 else 0

        return jsonify({
            'total_records': total,
            'failure_count': failures_count,
            'failure_rate': failure_rate,
            'avg_risk_score': avg_risk,
            'predictions': results
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process CSV file: {str(e)}'}), 400

@app.route('/download_template')
def download_template():
    sample_df = pd.DataFrame([
        {'Type': 'L', 'Air temperature [K]': 298.1, 'Process temperature [K]': 308.6, 'Rotational speed [rpm]': 1551, 'Torque [Nm]': 42.8, 'Tool wear [min]': 0},
        {'Type': 'M', 'Air temperature [K]': 301.2, 'Process temperature [K]': 311.5, 'Rotational speed [rpm]': 1400, 'Torque [Nm]': 65.2, 'Tool wear [min]': 210},
        {'Type': 'H', 'Air temperature [K]': 299.0, 'Process temperature [K]': 309.2, 'Rotational speed [rpm]': 2860, 'Torque [Nm]': 11.5, 'Tool wear [min]': 120},
        {'Type': 'L', 'Air temperature [K]': 302.5, 'Process temperature [K]': 312.1, 'Rotational speed [rpm]': 1320, 'Torque [Nm]': 58.4, 'Tool wear [min]': 195},
        {'Type': 'M', 'Air temperature [K]': 300.0, 'Process temperature [K]': 310.0, 'Rotational speed [rpm]': 1500, 'Torque [Nm]': 40.0, 'Tool wear [min]': 30}
    ])
    output = io.BytesIO()
    sample_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='sample_predictive_maintenance_batch.csv')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'optimal_threshold': optimal_threshold,
        'service': 'Elevvo Predictive Maintenance API',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
