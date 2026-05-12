# src/train.py - Complete version with registration
import warnings
import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import load_and_preprocess
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn-prediction"  # This is the REGISTERED model name
RUN_NAME = "random_forest_v1"

# =========================
# Setup MLflow
# =========================
# Use the MLflow server (not local file)
# Use the MLflow server (not local file)
if os.getenv("CI"):
    # Set MLflow to use a local directory for tracking
    mlflow.set_tracking_uri("file:///tmp/mlruns")  # Use /tmp/ (writable in GitHub Actions)
    os.makedirs("/tmp/mlruns", exist_ok=True)  # Ensure directory exists
else:
    mlflow.set_tracking_uri("http://mlflow-server:5000")

mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# Load and prepare data
# =========================
print("Loading data...")
X, y, _ = load_and_preprocess()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Churn rate in test: {y_test.mean():.2%}")

# =========================
# Train and register model
# =========================
with mlflow.start_run(run_name=RUN_NAME) as run:
    
    # Model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    # Train
    print("\nTraining model...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    # Log to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    
    print("\nPerformance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # =========================
    # CRITICAL: REGISTER THE MODEL
    # =========================
    print("\nRegistering model to MLflow Model Registry...")
    model_uri = f"runs:/{run.info.run_id}/model"
    
    try:
        # Register the model
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        
        # Add version description
        client = MlflowClient()
        client.update_model_version(
            name=MODEL_NAME,
            version=registered_model.version,
            description=f"Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}"
        )
        
        # Transition to "Production" stage (optional but good)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Production"
        )
        
        print(f"Model registered as '{MODEL_NAME}' version {registered_model.version}")
        print("Stage: Production")
        
    except Exception as e:
        print(f"Registration error: {e}")
        print("Model may already exist. Creating new version...")
        
        # If model exists, it will create a new version automatically
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Created version {registered_model.version}")

print("\nDone! Model registered in MLflow Model Registry")
print(f"View at: http://localhost:5000/#/models/{MODEL_NAME}")
