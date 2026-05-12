"""Unit tests for model training and prediction"""
import pytest
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import load_and_preprocess

class TestModel:
    """Test model functionality"""
    
    def test_model_trains(self):
        """Test that model trains without errors"""
        X, y, _ = load_and_preprocess()
        
        model = RandomForestClassifier(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        assert model is not None
        assert hasattr(model, "predict")
    
    def test_model_accuracy_reasonable(self):
        """Test model achieves reasonable accuracy"""
        from sklearn.model_selection import train_test_split
        
        X, y, _ = load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        # Should be better than random (50%)
        assert accuracy > 0.7, f"Accuracy too low: {accuracy}"
        assert accuracy < 0.95, f"Accuracy impossibly high: {accuracy}"
    
    def test_prediction_output_format(self):
        """Test prediction output format"""
        X, y, _ = load_and_preprocess()
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Single prediction
        pred = model.predict(X.iloc[[0]])
        assert pred.shape == (1,)
        assert pred[0] in [0, 1]
        
        # Probability prediction
        proba = model.predict_proba(X.iloc[[0]])
        assert proba.shape == (1, 2)
        assert abs(proba.sum() - 1.0) < 0.01
    
    def test_feature_importance(self):
        """Test model produces feature importance"""
        X, y, _ = load_and_preprocess()
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]
        assert sum(model.feature_importances_) > 0