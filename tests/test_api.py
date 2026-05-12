"""Unit tests for FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

client = TestClient(app)

class TestAPI:
    """Test API endpoints"""
    
    def test_health_check(self):
        """Test root endpoint returns healthy status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "service" in data
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        
        # Should have status field
        assert "status" in data
    
    def test_predict_valid_input(self):
        """Test prediction with valid customer data"""
        test_customer = {
            "gender": 1,
            "SeniorCitizen": 0,
            "Partner": 1,
            "Dependents": 0,
            "tenure": 12,
            "PhoneService": 1,
            "MultipleLines": 1,
            "InternetService": 1,
            "OnlineSecurity": 0,
            "OnlineBackup": 1,
            "DeviceProtection": 0,
            "TechSupport": 0,
            "StreamingTV": 1,
            "StreamingMovies": 1,
            "Contract": 0,
            "PaperlessBilling": 1,
            "PaymentMethod": 0,
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0
        }
        
        response = client.post("/predict", json=test_customer)
        
        # Even if model not loaded, should return proper error
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_risk" in data
            assert "churn_probability" in data
            assert "risk_level" in data
            assert data["churn_risk"] in [0, 1]
    
    def test_predict_batch(self):
        """Test batch prediction endpoint"""
        test_customers = [
            {
                "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
                "tenure": 12, "PhoneService": 1, "MultipleLines": 1, "InternetService": 1,
                "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0,
                "TechSupport": 0, "StreamingTV": 1, "StreamingMovies": 1,
                "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 0,
                "MonthlyCharges": 70.0, "TotalCharges": 840.0
            },
            {
                "gender": 0, "SeniorCitizen": 1, "Partner": 0, "Dependents": 0,
                "tenure": 48, "PhoneService": 1, "MultipleLines": 0, "InternetService": 0,
                "OnlineSecurity": 2, "OnlineBackup": 2, "DeviceProtection": 2,
                "TechSupport": 2, "StreamingTV": 2, "StreamingMovies": 2,
                "Contract": 2, "PaperlessBilling": 0, "PaymentMethod": 2,
                "MonthlyCharges": 95.0, "TotalCharges": 4560.0
            }
        ]
        
        response = client.post("/predict_batch", json=test_customers)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total" in data
            assert data["total"] == len(test_customers)
    
    def test_reload_endpoint(self):
        """Test model reload endpoint"""
        response = client.post("/reload")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "reloading"
    
    # def test_drift_monitor(self):
    #     """Test drift monitoring endpoint"""
    #     response = client.get("/monitor/drift")
    #     assert response.status_code == 200
    #     data = response.json()
        
    #     assert "drift_detected" in data
    #     assert "timestamp" in data
    #     assert "message" in data