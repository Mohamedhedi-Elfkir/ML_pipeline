# Automated ML Pipeline with Jenkins & MLflow

This project demonstrates an automated machine learning pipeline that combines Jenkins CI/CD with MLflow for model tracking and deployment.

## Architecture

- **Jenkins**: Orchestrates the ML pipeline (data generation, training, evaluation, deployment)
- **MLflow**: Tracks experiments, manages models, and provides model registry
- **PostgreSQL**: Backend store for MLflow metadata
- **Docker**: Containerizes the entire infrastructure

## Quick Start

1. **Start the infrastructure:**
   ```bash
   ./start.sh
   ```

2. **Access the services:**
   - Jenkins: http://localhost:8080
   - MLflow: http://localhost:5000

3. **Configure Jenkins:**
   - Create a new Pipeline job
   - Point to `jenkins/Jenkinsfile`
   - Set up polling or webhooks for automatic triggers

## Project Structure

```
├── data/              # Generated datasets
├── src/               # ML training scripts
│   ├── generate_data.py    # Data generation
│   ├── train_model.py      # Model training with MLflow
│   ├── evaluate_models.py  # Model evaluation
│   └── deploy_model.py     # Model deployment
├── models/            # Saved models and deployment info
├── jenkins/           # Jenkins pipeline configuration
├── docker/            # Docker configuration
└── requirements.txt   # Python dependencies
```

## Pipeline Workflow

1. **Data Generation**: Creates synthetic sales data
2. **Model Training**: Trains Random Forest and Linear Regression models in parallel
3. **Model Evaluation**: Compares models and selects the best one
4. **Model Deployment**: Deploys models with R² > 0.7 to production

## MLflow Features

- **Experiment Tracking**: All training runs are logged with parameters and metrics
- **Model Registry**: Best models are registered and versioned
- **Model Deployment**: Automatic deployment with quality gates

## Customization

- Modify `src/generate_data.py` to simulate your data
- Add new model types in `src/train_model.py`
- Adjust quality gates in `src/deploy_model.py`
- Update `jenkins/Jenkinsfile` for custom pipeline logic