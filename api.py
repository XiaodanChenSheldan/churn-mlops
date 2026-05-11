# api.py - Fixed with consistent naming
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
from typing import List
import json
from src.monitor_drift import calculate_drift
from datetime import datetime
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# =========================
# Configuration
# =========================
MODEL_NAME = "churn-prediction"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

# =========================
# Global state (single source of truth)
# =========================
model = None           # ← SINGLE global variable for the model
model_version = None   # ← Track which version is loaded

# ----------------------------
# Pydantic Models
# ----------------------------
class Customer(BaseModel):
    gender: int = Field(..., description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., description="0=No, 1=Yes")
    Partner: int = Field(..., description="0=No, 1=Yes")
    Dependents: int = Field(..., description="0=No, 1=Yes")
    tenure: int = Field(..., ge=0, le=72, description="Months with company")
    PhoneService: int = Field(..., description="0=No, 1=Yes")
    MultipleLines: int = Field(..., description="0=No, 1=Yes")
    InternetService: int = Field(..., description="0=DSL, 1=Fiber optic, 2=No")
    OnlineSecurity: int = Field(..., description="0=No, 1=Yes, 2=No internet")
    OnlineBackup: int = Field(..., description="0=No, 1=Yes, 2=No internet")
    DeviceProtection: int = Field(..., description="0=No, 1=Yes, 2=No internet")
    TechSupport: int = Field(..., description="0=No, 1=Yes, 2=No internet")
    StreamingTV: int = Field(..., description="0=No, 1=Yes, 2=No internet")
    StreamingMovies: int = Field(..., description="0=No, 1=Yes, 2=No internet")
    Contract: int = Field(..., description="0=Month-to-month, 1=One year, 2=Two year")
    PaperlessBilling: int = Field(..., description="0=No, 1=Yes")
    PaymentMethod: int = Field(..., description="0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges in USD")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
                "tenure": 12, "PhoneService": 1, "MultipleLines": 1, "InternetService": 1,
                "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0,
                "TechSupport": 0, "StreamingTV": 1, "StreamingMovies": 1, "Contract": 0,
                "PaperlessBilling": 1, "PaymentMethod": 0, "MonthlyCharges": 70.0,
                "TotalCharges": 840.0
            }
        }
    )

# ----------------------------
# Helper function to load model
# ----------------------------
def load_model_from_registry(stage="production"):
    """Load model from MLflow registry by stage"""
    global model, model_version
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get latest version in specified stage
        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        
        if not latest_versions:
            print(f"No model found in stage '{stage}'")
            return False
        
        version = latest_versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{version}"
        
        model = mlflow.sklearn.load_model(model_uri)
        model_version = version
        
        print(f"Loaded model version {model_version} (stage: {stage})")
        return True
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
        model_version = None
        return False

# ----------------------------
# Lifespan context manager
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup"""
    global model, model_version
    
    print("🚀 Starting up API...")
    
    # Retry logic for model loading
    max_retries = 5
    for i in range(max_retries):
        if load_model_from_registry("production"):
            break
        elif i < max_retries - 1:
            print(f"Retry {i+1}/{max_retries} in 5 seconds...")
            import time
            time.sleep(5)
        else:
            print("Could not load model after all retries")
    
    yield
    
    # Shutdown
    print(f"🛑 Shutting down API. Model version {model_version} was active.")
    model = None
    model_version = None

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict which customers are likely to churn - MLOps Demo",
    version="1.0.0",
    lifespan=lifespan
)

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "service": "Churn Prediction API"
    }

@app.post("/predict")
async def predict(customer: Customer):
    """Predict churn risk for a single customer"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first."
        )
    
    customer_df = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]
    
    return {
        "churn_risk": int(prediction),
        "churn_probability": float(probability),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
    }

@app.post("/predict_batch")
async def predict_batch(customers: List[Customer]):
    """Predict churn risk for multiple customers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    customers_df = pd.DataFrame([c.model_dump() for c in customers])
    predictions = model.predict(customers_df)
    probabilities = model.predict_proba(customers_df)[:, 1]
    
    return {
        "predictions": [
            {"index": i, "churn_risk": int(pred), "churn_probability": float(prob)}
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ],
        "total": len(predictions)
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "no_model_loaded"}
    
    return {
        "status": "loaded",
        "version": model_version,
        "type": type(model).__name__,
        "parameters": model.get_params()
    }

@app.get("/monitor/drift")
async def check_drift():
    """Check if data drift has occurred"""
    drift_detected = calculate_drift()
    return {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "message": "Retraining recommended" if drift_detected else "Model is healthy"
    }

@app.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the latest production model"""
    background_tasks.add_task(refresh_model)
    return {"status": "reloading", "message": "Model reload scheduled"}

async def refresh_model():
    """Background task to refresh model from registry"""
    global model, model_version
    
    print("Reloading model...")
    success = load_model_from_registry("production")
    
    if success:
        print(f"Model refreshed to version {model_version}")
    else:
        print("Model refresh failed")

@app.post("/reload/stage/{stage}")
async def reload_from_stage(stage: str, background_tasks: BackgroundTasks):
    """Reload model from specific stage (staging, production, archived)"""
    if stage not in ["staging", "production", "archived"]:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
    
    background_tasks.add_task(lambda: load_model_from_registry(stage))
    return {"status": "reloading", "message": f"Loading model from {stage}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )