# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(data_path="data/raw/churn.csv"):
    """Load raw data and prepare for ML training"""
    
    # Load
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    
    # Drop customerID (useless for prediction)
    df = df.drop("customerID", axis=1)
    
    # Convert TotalCharges from string to number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Fill missing TotalCharges with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    
    # Encode target variable: Churn (Yes=1, No=0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    
    # Encode categorical features (convert text to numbers)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Save encoder for later use
    
    # Separate features (X) and target (y)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    print(f"Processed shape: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    
    return X, y, encoders

if __name__ == "__main__":
    X, y, _ = load_and_preprocess()
    print("Preprocessing complete!")
    print(f"Features: {list(X.columns)[:5]}...")