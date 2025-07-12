# Load the model

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

import warnings
warnings.filterwarnings('ignore')

model_file = 'XGBoost_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Deposit Prediction API")

# Pydantic model for request validation
class Customer(BaseModel):
    class Config:
        extra = "allow"  # Allow any fields for flexibility

@app.get("/")
def health_check():
    return {"status": "OK"}

@app.get("/predict")
def predict_get():
    return {"message": "Send a POST request with customer data to get a prediction."}

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    deposit_decision = y_pred >= 0.61
    
    result = {
        'deposit_probability': float(y_pred),
        'will_deposit': bool(deposit_decision)
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)