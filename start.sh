#!/bin/bash

echo "Starting ML Pipeline Infrastructure..."

# Start infrastructure services
docker-compose up -d mlflow-db mlflow-server jenkins

echo "Waiting for services to start..."
sleep 30

# Generate initial data
echo "Generating initial dataset..."
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python generate_data.py

echo "Setup complete!"
echo "Access Jenkins at: http://localhost:8080"
echo "Access MLflow at: http://localhost:5000"
echo ""
echo "To run training manually:"
echo "cd src && source venv/bin/activate && python train_model.py"