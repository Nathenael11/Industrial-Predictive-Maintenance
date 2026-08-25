import os
import sys
import json
import pytest
import pandas as pd
import io

# Add parent directory to path to import app and train_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, preprocess_input_df, generate_recommendation, optimal_threshold
from train_model import feature_engineering

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test_secret_key'
    with app.test_client() as client:
        yield client

def test_feature_engineering_data_leakage():
    """Verify that feature engineering drops identifier & leakage columns"""
    raw_df = pd.DataFrame([{
        'UDI': 1,
        'Product ID': 'M14860',
        'Type': 'M',
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 0,
        'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0,
        'Machine failure': 0
    }])
    
    proc_df = feature_engineering(raw_df)
    
    assert 'UDI' not in proc_df.columns
    assert 'Product ID' not in proc_df.columns
    assert 'TWF' not in proc_df.columns
    assert 'HDF' not in proc_df.columns
    assert 'Type_M' in proc_df.columns
    assert 'Air_temperature_K' in proc_df.columns

def test_preprocess_input_df():
    """Test preprocessing input for single and batch dictionary payloads"""
    raw = pd.DataFrame([{
        'air_temp': 301.0,
        'process_temp': 311.0,
        'rotational_speed': 1400,
        'torque': 55.0,
        'tool_wear': 180,
        'product_type': 'H'
    }])
    
    features = preprocess_input_df(raw)
    
    assert features.shape == (1, 8)
    assert 'Air_temperature_K' in features.columns
    assert features['Type_H'].iloc[0] == 1
    assert features['Type_L'].iloc[0] == 0

def test_generate_recommendation():
    """Test domain logic recommendation text generator"""
    normal_row = {'Air_temperature_K': 300, 'Process_temperature_K': 310, 'Torque_Nm': 40, 'Tool_wear_min': 50}
    rec_normal = generate_recommendation(normal_row, is_failure=False, proba=0.1)
    assert 'NORMAL' in rec_normal or 'nominal' in rec_normal

    fail_row = {'Air_temperature_K': 300, 'Process_temperature_K': 310, 'Torque_Nm': 70, 'Tool_wear_min': 220}
    rec_fail = generate_recommendation(fail_row, is_failure=True, proba=0.9)
    assert 'CRITICAL' in rec_fail or 'HIGH RISK' in rec_fail

def test_health_endpoint(client):
    """Test /health API endpoint"""
    res = client.get('/health')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['status'] == 'healthy'
    assert 'optimal_threshold' in data

def test_index_route(client):
    """Test GET / route"""
    res = client.get('/')
    assert res.status_code == 200
    assert b'Elevvo Maintenance AI' in res.data

def test_predict_endpoint_json(client):
    """Test POST /predict with JSON payload"""
    payload = {
        'air_temp': 300.0,
        'process_temp': 310.0,
        'rotational_speed': 1500,
        'torque': 40.0,
        'tool_wear': 30,
        'product_type': 'L'
    }
    
    res = client.post('/predict', json=payload)
    assert res.status_code == 200
    data = json.loads(res.data)
    
    assert 'risk_score' in data
    assert 'prediction' in data
    assert 'recommended_action' in data
    assert data['prediction'] in ['Failure', 'No Failure']

def test_batch_endpoint(client):
    """Test POST /batch with CSV file upload"""
    csv_data = "Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]\n" \
               "L,300.0,310.0,1500,40.0,20\n" \
               "M,302.0,312.0,1300,65.0,210\n"
               
    data = {
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'test_batch.csv')
    }
    
    res = client.post('/batch', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    resp_json = json.loads(res.data)
    
    assert resp_json['total_records'] == 2
    assert 'failure_count' in resp_json
    assert len(resp_json['predictions']) == 2

def test_download_template_endpoint(client):
    """Test GET /download_template endpoint"""
    res = client.get('/download_template')
    assert res.status_code == 200
    assert res.content_type == 'text/csv; charset=utf-8'
    assert b'Air temperature [K]' in res.data
