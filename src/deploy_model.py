import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import json
import os

def deploy_best_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    # Load best model info
    best_model_path = '../models/best_model_info.json'
    if not os.path.exists(best_model_path):
        print("No best model info found. Run evaluate_models.py first.")
        return False
    
    with open(best_model_path, 'r') as f:
        best_model_info = json.load(f)
    
    run_id = best_model_info['run_id']
    r2_score = best_model_info['r2_score']
    
    # Quality gate: only deploy if R2 > 0.7
    if r2_score < 0.7:
        print(f"Model R2 score {r2_score} is below threshold 0.7. Not deploying.")
        return False
    
    # Register model in MLflow Model Registry
    model_uri = f"runs:/{run_id}/model"
    model_name = "sales_prediction_model"
    
    try:
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        print(f"Model registered as {model_name} version {model_version.version}")
        
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"Model version {model_version.version} transitioned to Production")
        
        # Create deployment info
        deployment_info = {
            'model_name': model_name,
            'model_version': model_version.version,
            'run_id': run_id,
            'r2_score': r2_score,
            'deployment_status': 'SUCCESS'
        }
        
        with open('../models/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        
        deployment_info = {
            'deployment_status': 'FAILED',
            'error': str(e)
        }
        
        with open('../models/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        return False

if __name__ == "__main__":
    success = deploy_best_model()
    if success:
        print("Model deployment completed successfully!")
    else:
        print("Model deployment failed!")