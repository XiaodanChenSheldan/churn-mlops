"""Pytest configuration and shared fixtures"""

import pandas as pd
import pytest

from src.preprocess import load_and_preprocess


@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    X, y, _ = load_and_preprocess()
    return X, y


@pytest.fixture
def sample_customer():
    """Provide a sample customer for API tests"""
    return {
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
        "TotalCharges": 840.0,
    }
