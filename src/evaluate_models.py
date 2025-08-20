import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import json

def get_best_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name("sales_prediction")
    if not experiment:
        print("No experiment found")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found")
        return None
    
    best_run = runs[0]
    
    print(f"Best model run ID: {best_run.info.run_id}")
    print(f"Best model R2 score: {best_run.data.metrics.get('r2', 'N/A')}")
    print(f"Best model RMSE: {best_run.data.metrics.get('rmse', 'N/A')}")
    print(f"Model type: {best_run.data.params.get('model_type', 'N/A')}")
    
    # Save best model info for deployment
    best_model_info = {
        'run_id': best_run.info.run_id,
        'r2_score': best_run.data.metrics.get('r2'),
        'rmse': best_run.data.metrics.get('rmse'),
        'model_type': best_run.data.params.get('model_type')
    }
    
    with open('../models/best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    return best_model_info

if __name__ == "__main__":
    best_model = get_best_model()
    if best_model:
        print("Model evaluation completed successfully!")
    else:
        print("No models found for evaluation")