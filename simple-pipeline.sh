#!/bin/bash

echo "=== Running ML Pipeline ==="

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

cd /home/hedi/Desktop/mlflow/src

echo "Step 1: Generating data..."
python3 generate_data.py

echo "Step 2: Training Random Forest model..."
python3 train_model.py --model-type random_forest --n-estimators 100

echo "Step 3: Training Linear Regression model..."  
python3 train_model.py --model-type linear_regression

echo "Step 4: Evaluating models..."
python3 evaluate_models.py

echo "Step 5: Deploying best model..."
python3 deploy_model.py

echo "=== Pipeline Complete ==="
echo "Check MLflow at http://localhost:5000 for results"