# register_model.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow-server:5000")
client = MlflowClient()

# Get the latest run from your experiment
experiment = client.get_experiment_by_name("churn-prediction")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if len(runs) > 0:
        best_run_id = runs.iloc[0].run_id
        print(f"Registering model from run: {best_run_id}")

        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "churn-prediction")
        print("Model registered as 'churn-prediction'")
    else:
        print("No runs found. Run 'python src/train.py' first")
else:
    print("Experiment 'churn-prediction' not found. Run 'python src/train.py' first")
