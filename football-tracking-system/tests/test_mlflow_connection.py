"""
Test script to verify MLflow tracking server connection and metric logging.
"""

import os
import mlflow

def run_test():
    # If MLFLOW_TRACKING_URI is set, use it. Otherwise, use local sqlite DB for test.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"Connecting to MLflow Tracking Server at: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Relative sqlite database from system execution context
        local_db = "sqlite:///experiments/mlflow/mlflow.db"
        print(f"No MLFLOW_TRACKING_URI environment variable detected.")
        print(f"Falling back to local SQLite DB: {local_db}")
        os.makedirs("experiments/mlflow", exist_ok=True)
        mlflow.set_tracking_uri(local_db)
        
    experiment_name = "mlflow-connection-test"
    mlflow.set_experiment(experiment_name)
    
    print(f"Starting test MLflow run under experiment '{experiment_name}'...")
    with mlflow.start_run(run_name="connection_test_run") as run:
        # Log dummy params and metrics
        mlflow.log_param("test_status", "success")
        mlflow.log_metric("dummy_metric", 0.99)
        mlflow.log_metric("epoch", 1)
        
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        print("\n[SUCCESS] Successfully logged dummy data to MLflow!")
        print(f"Run ID: {run_id}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Logged parameter: test_status = success")
        print(f"Logged metric: dummy_metric = 0.99")

if __name__ == "__main__":
    run_test()
