import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sales_data(n_samples=1000, start_date="2024-01-01"):
    np.random.seed(42)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(n_samples)]
    
    base_price = 100
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    trend = 0.0001 * np.arange(n_samples)
    noise = np.random.normal(0, 0.1, n_samples)
    
    price = base_price * (seasonal_factor + trend + noise)
    
    marketing_spend = np.random.uniform(1000, 5000, n_samples)
    competitor_price = price + np.random.normal(0, 5, n_samples)
    day_of_week = np.array([d.weekday() for d in dates])
    is_weekend = (day_of_week >= 5).astype(int)
    
    base_sales = 1000
    price_elasticity = -2.0
    marketing_effect = 0.1
    weekend_boost = 50
    
    sales = (base_sales + 
             price_elasticity * (price - base_price) +
             marketing_effect * marketing_spend +
             weekend_boost * is_weekend +
             np.random.normal(0, 50, n_samples))
    
    sales = np.maximum(sales, 0)
    
    data = pd.DataFrame({
        'date': dates,
        'price': price,
        'marketing_spend': marketing_spend,
        'competitor_price': competitor_price,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'sales': sales
    })
    
    return data

if __name__ == "__main__":
    data = generate_sales_data(1000)
    
    os.makedirs('../data', exist_ok=True)
    data.to_csv('../data/sales_data.csv', index=False)
    print(f"Generated {len(data)} rows of sales data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Sales range: {data['sales'].min():.2f} to {data['sales'].max():.2f}")