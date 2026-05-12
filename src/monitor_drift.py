# src/monitor_drift.py
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.preprocess import load_and_preprocess
import json
from datetime import datetime

def load_production_data():
    """Simulate getting new data from production"""
    # In real world, this would query your database or data warehouse
    X, y, _ = load_and_preprocess()
    
    # Simulate "new" data (last 20% of records)
    split_point = int(len(X) * 0.8)
    new_data = X.iloc[split_point:]
    return new_data

def calculate_drift():
    """Compare reference (training) data vs current (production) data"""
    
    # 1. Load reference data (what model was trained on)
    X_ref, y_ref, _ = load_and_preprocess()
    
    # 2. Load current production data
    X_current = load_production_data()
    
    # 3. Create drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),  # Checks if input features changed
    ])
    
    # 4. Run the comparison
    drift_report.run(reference_data=X_ref, current_data=X_current)
    
    # 5. Extract results
    result = drift_report.as_dict()

    
    # 6. Check if drift detected
    drift_detected = result['metrics'][1]['result']['drift_by_columns']
    num_drifted = sum(1 for col in drift_detected.values() if col['drift_detected'])
    
    print("Drift Analysis Results:")
    print(f"   Total features: {len(drift_detected)}")
    print(f"   Features with drift: {num_drifted}")
    
    # 7. Log to MLflow
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_metric("num_drifted_features", num_drifted)
        mlflow.log_metric("drift_percentage", (num_drifted / len(drift_detected)) * 100)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"drift_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        mlflow.log_artifact(report_file)
    
    # 8. Alert if drift is severe
    if num_drifted > len(drift_detected) * 0.3:  # >30% features drifted
        print("ALERT: Significant data drift detected! Consider retraining.")
        return True
    else:
        print("No significant drift detected.")
        return False

if __name__ == "__main__":
    calculate_drift()
    