import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import argparse
import os
import joblib
from datetime import datetime

def load_data(data_path):
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_features(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    feature_cols = ['price', 'marketing_spend', 'competitor_price', 
                   'day_of_week', 'is_weekend', 'month', 'day_of_year']
    
    X = df[feature_cols]
    y = df['sales']
    
    return X, y

def train_model(model_type, X_train, y_train, **params):
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42
        )
    elif model_type == 'linear_regression':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='../data/sales_data.csv')
    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'linear_regression'])
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--test-size', type=float, default=0.2)
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sales_prediction")
    
    with mlflow.start_run():
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("test_size", args.test_size)
        
        if args.model_type == 'random_forest':
            mlflow.log_param("n_estimators", args.n_estimators)
            if args.max_depth:
                mlflow.log_param("max_depth", args.max_depth)
        
        df = load_data(args.data_path)
        print(f"Loaded {len(df)} rows of data")
        
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
        
        model_params = {}
        if args.model_type == 'random_forest':
            model_params['n_estimators'] = args.n_estimators
            if args.max_depth:
                model_params['max_depth'] = args.max_depth
        
        model = train_model(args.model_type, X_train, y_train, **model_params)
        
        metrics = evaluate_model(model, X_test, y_test)
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.sklearn.log_model(model, "model")
        
        model_dir = "../models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model, model_path)
        
        print(f"Model trained successfully!")
        print(f"Metrics: {metrics}")
        print(f"Model saved to: {model_path}")
        
        return metrics

if __name__ == "__main__":
    main()