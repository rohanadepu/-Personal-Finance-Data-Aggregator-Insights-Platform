import mlflow
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta

def prepare_forecast_data(df):
    # Aggregate daily cash flow
    daily_flow = df.groupby(pd.Grouper(key='timestamp', freq='D'))['amount'].sum().reset_index()
    daily_flow.columns = ['ds', 'y']  # Prophet requires these column names
    return daily_flow

def train_forecast_model(df, forecast_periods=30):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("forecast_periods", forecast_periods)
        mlflow.log_param("seasonality_mode", "multiplicative")
        
        # Fit model
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        # Calculate metrics on training data
        train_metrics = calculate_metrics(df['y'], forecast['yhat'][:len(df)])
        
        # Log metrics
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        
        # Log model
        mlflow.prophet.log_model(model, "prophet_model")
        
        return model, forecast

def calculate_metrics(actual, predicted):
    metrics = {
        'mape': mean_absolute_percentage_error(actual, predicted),
        'rmse': np.sqrt(np.mean((actual - predicted) ** 2)),
        'mae': np.mean(np.abs(actual - predicted))
    }
    return metrics

def generate_forecast(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    results = pd.DataFrame({
        'timestamp': forecast['ds'],
        'predicted_amount': forecast['yhat'],
        'lower_bound': forecast['yhat_lower'],
        'upper_bound': forecast['yhat_upper']
    })
    
    return results

if __name__ == "__main__":
    import duckdb
    
    # Load transactions
    conn = duckdb.connect(database=':memory:', read_only=False)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    
    # Prepare data
    forecast_data = prepare_forecast_data(df)
    
    # Train model and generate forecast
    model, forecast = train_forecast_model(forecast_data)
    
    # Generate future predictions
    future_forecast = generate_forecast(model)
    print("Forecast generated for next 30 days")