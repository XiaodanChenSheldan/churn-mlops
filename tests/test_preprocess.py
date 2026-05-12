"""Unit tests for data preprocessing"""
import pytest
import pandas as pd
import numpy as np
from src.preprocess import load_and_preprocess
import os

class TestPreprocessing:
    """Test data preprocessing functions"""
    
    def test_load_data_exists(self):
        """Test that data file exists"""
        assert os.path.exists("data/raw/churn.csv"), "Data file not found"
    
    def test_load_and_preprocess_shape(self):
        """Test preprocessing returns correct shapes"""
        X, y, encoders = load_and_preprocess()
        
        # Check shapes
        assert X.shape[0] == y.shape[0], "X and y row count mismatch"
        assert X.shape[0] > 0, "No data loaded"
        assert len(X.shape) == 2, "X should be 2D array"
    
    def test_no_missing_values(self):
        """Test that preprocessing handles missing values"""
        X, y, _ = load_and_preprocess()
        
        # Check for NaN values
        assert not X.isnull().any().any(), "Missing values found in X"
        assert not y.isnull().any(), "Missing values found in y"
    
    def test_churn_rate_reasonable(self):
        """Test churn rate is between 0 and 1"""
        _, y, _ = load_and_preprocess()
        churn_rate = y.mean()
        
        assert 0 <= churn_rate <= 1, f"Invalid churn rate: {churn_rate}"
        assert churn_rate > 0, "Churn rate should be > 0"
        assert churn_rate < 0.5, f"Churn rate {churn_rate} > 50% seems high"
    
    def test_encoders_created(self):
        """Test that encoders are created for categorical columns"""
        _, _, encoders = load_and_preprocess()
        
        # Should have at least one encoder
        assert len(encoders) > 0, "No encoders created"