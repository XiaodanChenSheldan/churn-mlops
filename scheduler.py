# scheduler.py - Complete working scheduler
import schedule
import time
import requests
import subprocess
import logging
from datetime import datetime
from src.monitor_drift import calculate_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_performance():
    """Check model metrics from MLflow"""
    try:
        response = requests.get("http://mlflow-server:5000/api/2.0/mlflow/experiments/search")
        logger.info(f"MLflow check: {response.status_code}")
    except Exception as e:
        logger.error(f"MLflow connection error: {e}")

def trigger_retraining():
    """Trigger model retraining pipeline"""
    logger.info("🔄 Starting retraining pipeline...")
    
    try:
        # Run training script (which now handles registration internally)
        result = subprocess.run(
            ["python", "src/train.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Retraining and registration successful!")
            
            # Optional: Transition to staging/production
            subprocess.run(["python", "promote_model.py", "--stage", "production"])
        else:
            logger.error(f"❌ Retraining failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Retraining error: {e}")

def monitor_and_retrain():
    """Main monitoring function"""
    logger.info(f"🔍 Running check at {datetime.now()}")
    
    # Check for drift
    drift_detected = calculate_drift()
    
    # Check last model performance
    check_model_performance()
    
    if drift_detected:
        logger.warning("🚨 Drift detected! Triggering retraining...")
        trigger_retraining()
    else:
        logger.info("✅ No drift detected")

# Schedule monitoring every 6 hours
schedule.every(6).hours.do(monitor_and_retrain)

# Also check at 2 AM daily
schedule.every().day.at("02:00").do(monitor_and_retrain)

logger.info("🚀 Scheduler started. Monitoring every 6 hours...")

while True:
    schedule.run_pending()
    time.sleep(60)
