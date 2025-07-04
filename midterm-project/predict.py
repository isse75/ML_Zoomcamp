# Load the model

import pickle
from flask import Flask
from flask import request
from flask import jsonify

import warnings
warnings.filterwarnings('ignore')

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in: #f_in = file input, opens file for reading 'rb'
    dv, model = pickle.load(f_in)

app = Flask('heart_disease')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return 'Send a POST request with customer data to get a prediction.'
    
    patient = request.get_json()
    
    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0,1]
    diagnosis = y_pred >= 0.39
    
    result = {
        'diagnosis_probability': float(y_pred),
        'heart_disease': bool(diagnosis)
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) #app running locally

@app.route('/')
def health_check():
    return 'OK', 200
