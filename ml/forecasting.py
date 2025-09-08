import mlflow
import pandas as pd
import numpy as np
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        print("⚠️  Prophet not available. Please install with: pip install prophet")
        Prophet = None
        PROPHET_AVAILABLE = False

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("transaction_forecasting")

class FinancialForecaster:
    def __init__(self, algorithm='prophet'):
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.lookback_window = 30
        
    def prepare_forecast_data(self, df, freq='D', aggregation='sum'):
        """Prepare data for time series forecasting"""
        # Ensure timestamp is datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Aggregate by specified frequency
        if aggregation == 'sum':
            daily_data = df.groupby(pd.Grouper(key='timestamp', freq=freq))['amount'].sum().reset_index()
        elif aggregation == 'count':
            daily_data = df.groupby(pd.Grouper(key='timestamp', freq=freq))['amount'].count().reset_index()
        elif aggregation == 'mean':
            daily_data = df.groupby(pd.Grouper(key='timestamp', freq=freq))['amount'].mean().reset_index()
        
        # Fill missing dates
        date_range = pd.date_range(start=daily_data['timestamp'].min(), 
                                 end=daily_data['timestamp'].max(), 
                                 freq=freq)
        full_range = pd.DataFrame({'timestamp': date_range})
        daily_data = full_range.merge(daily_data, on='timestamp', how='left')
        daily_data['amount'] = daily_data['amount'].fillna(0)
        
        return daily_data
        
    def train_prophet(self, df, forecast_periods=30, seasonality_mode='additive'):
        """Train Prophet model for forecasting with enhanced MLflow tracking"""
        
        if not PROPHET_AVAILABLE or Prophet is None:
            print("❌ Prophet not available. Skipping Prophet training.")
            return None, None
            
        with mlflow.start_run(run_name=f"prophet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_data = df.copy()
            prophet_data.columns = ['ds', 'y']
            
            # Log parameters
            mlflow.log_param("algorithm", "prophet")
            mlflow.log_param("forecast_periods", forecast_periods)
            mlflow.log_param("seasonality_mode", seasonality_mode)
            mlflow.log_param("data_points", len(prophet_data))
            mlflow.log_param("data_start_date", prophet_data['ds'].min().isoformat())
            mlflow.log_param("data_end_date", prophet_data['ds'].max().isoformat())
            mlflow.log_param("data_span_days", (prophet_data['ds'].max() - prophet_data['ds'].min()).days)
            
            # Log data statistics
            mlflow.log_metric("mean_value", prophet_data['y'].mean())
            mlflow.log_metric("std_value", prophet_data['y'].std())
            mlflow.log_metric("min_value", prophet_data['y'].min())
            mlflow.log_metric("max_value", prophet_data['y'].max())
            
            # Initialize and configure Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=0.05,
                interval_width=0.8
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            mlflow.log_param("yearly_seasonality", True)
            mlflow.log_param("weekly_seasonality", True)
            mlflow.log_param("daily_seasonality", False)
            mlflow.log_param("changepoint_prior_scale", 0.05)
            mlflow.log_param("monthly_seasonality_fourier_order", 5)
            
            # Fit model
            model.fit(prophet_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)
            
            # Calculate metrics on training data
            train_metrics = self.calculate_metrics(prophet_data['y'], 
                                                 forecast['yhat'][:len(prophet_data)])
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            # Log forecast statistics
            future_forecast = forecast['yhat'][-forecast_periods:]
            mlflow.log_metric("forecast_mean", future_forecast.mean())
            mlflow.log_metric("forecast_std", future_forecast.std())
            mlflow.log_metric("forecast_min", future_forecast.min())
            mlflow.log_metric("forecast_max", future_forecast.max())
            
            # Create and log visualizations
            try:
                import matplotlib.pyplot as plt
                
                # Create forecast plot
                fig, axes = plt.subplots(2, 2, figsize=(20, 15))
                
                # Main forecast plot
                model.plot(forecast, ax=axes[0, 0])
                axes[0, 0].set_title('Prophet Forecast')
                axes[0, 0].set_xlabel('Date')
                axes[0, 0].set_ylabel('Value')
                
                # Components plot - simplified
                try:
                    # Residuals plot
                    residuals = prophet_data['y'] - forecast['yhat'][:len(prophet_data)]
                    axes[0, 1].plot(prophet_data['ds'], residuals)
                    axes[0, 1].set_title('Residuals')
                    axes[0, 1].set_xlabel('Date')
                    axes[0, 1].set_ylabel('Residual')
                    axes[0, 1].axhline(y=0, color='r', linestyle='--')
                    
                    # Residuals histogram
                    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Residuals Distribution')
                    axes[1, 0].set_xlabel('Residual')
                    axes[1, 0].set_ylabel('Frequency')
                    
                    # Forecast uncertainty
                    future_forecast_df = forecast.tail(forecast_periods)
                    axes[1, 1].plot(future_forecast_df['ds'], future_forecast_df['yhat'], 
                                   label='Forecast', color='blue')
                    axes[1, 1].fill_between(future_forecast_df['ds'], 
                                           future_forecast_df['yhat_lower'], 
                                           future_forecast_df['yhat_upper'], 
                                           alpha=0.3, color='blue', label='Uncertainty')
                    axes[1, 1].set_title('Forecast with Uncertainty')
                    axes[1, 1].set_xlabel('Date')
                    axes[1, 1].set_ylabel('Value')
                    axes[1, 1].legend()
                    
                    plt.tight_layout()
                    plt.savefig("prophet_analysis.png", dpi=150, bbox_inches='tight')
                    mlflow.log_artifact("prophet_analysis.png")
                    plt.close()
                    
                    # Clean up
                    os.remove("prophet_analysis.png")
                    
                except Exception as plot_error:
                    print(f"Warning: Could not create detailed plots: {plot_error}")
                    plt.close('all')
                
            except Exception as e:
                print(f"Warning: Could not create Prophet visualizations: {e}")
            
            # Log model (simplified)
            try:
                # Save model manually as artifact
                import joblib
                joblib.dump(model, "prophet_model.pkl")
                mlflow.log_artifact("prophet_model.pkl")
                os.remove("prophet_model.pkl")
                print("✅ Prophet model logged successfully")
            except Exception as e:
                print(f"⚠️  Prophet model logging warning: {e}")
            
            # Log forecast data
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
            forecast_df.to_csv("forecast_results.csv", index=False)
            mlflow.log_artifact("forecast_results.csv")
            os.remove("forecast_results.csv")
            
            self.model = model
            print(f"✅ Prophet model trained successfully for {forecast_periods} days")
            return model, forecast
            return model, forecast
            
    def train_arima(self, df, order=(1,1,1), forecast_periods=30):
        """Train ARIMA model for forecasting"""
        with mlflow.start_run(run_name=f"arima_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("algorithm", "arima")
            mlflow.log_param("order", str(order))
            mlflow.log_param("forecast_periods", forecast_periods)
            mlflow.log_param("data_points", len(df))
            
            # Prepare time series
            ts_data = df.set_index('timestamp')['amount']
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            forecast_result = fitted_model.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(start=ts_data.index[-1] + timedelta(days=1), 
                                         periods=forecast_periods, freq='D')
            
            # Calculate training metrics
            fitted_values = fitted_model.fittedvalues
            train_metrics = self.calculate_metrics(ts_data, fitted_values)
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            mlflow.log_metric("aic", fitted_model.aic)
            mlflow.log_metric("bic", fitted_model.bic)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'timestamp': forecast_index,
                'predicted_amount': forecast_result,
                'model': 'arima'
            })
            
            # Save model
            joblib.dump(fitted_model, "arima_model.pkl")
            mlflow.log_artifact("arima_model.pkl")
            os.remove("arima_model.pkl")
            
            self.model = fitted_model
            return fitted_model, forecast_df
            
    def prepare_lstm_data(self, df, lookback_window=30):
        """Prepare data for LSTM training"""
        # Scale the data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df[['amount']])
        
        # Create sequences
        X, y = [], []
        for i in range(lookback_window, len(scaled_data)):
            X.append(scaled_data[i-lookback_window:i])
            y.append(scaled_data[i])
        
        return np.array(X), np.array(y)
        
    def build_lstm_model(self, input_shape, units=[50, 25], dropout_rate=0.2):
        """Build LSTM architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for unit in units[1:]:
            model.add(LSTM(unit, return_sequences=True))
            model.add(Dropout(dropout_rate))
            
        # Final LSTM layer
        model.add(LSTM(units[-1]))
        model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def train_lstm(self, df, lookback_window=30, forecast_periods=30, epochs=100, 
                   batch_size=32, validation_split=0.2):
        """Train LSTM model for forecasting"""
        with mlflow.start_run(run_name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("algorithm", "lstm")
            mlflow.log_param("lookback_window", lookback_window)
            mlflow.log_param("forecast_periods", forecast_periods)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("data_points", len(df))
            
            self.lookback_window = lookback_window
            
            # Prepare data
            X, y = self.prepare_lstm_data(df, lookback_window)
            
            # Build model
            model = self.build_lstm_model((lookback_window, 1))
            
            # Train model
            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0
            )
            
            # Generate forecasts
            last_sequence = X[-1].reshape(1, lookback_window, 1)
            forecasts = []
            
            for _ in range(forecast_periods):
                pred = model.predict(last_sequence, verbose=0)
                forecasts.append(pred[0, 0])
                
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred[0, 0]
            
            # Inverse transform forecasts
            forecasts_scaled = np.array(forecasts).reshape(-1, 1)
            forecasts_original = self.scaler.inverse_transform(forecasts_scaled).flatten()
            
            # Create forecast dataframe
            forecast_index = pd.date_range(start=df['timestamp'].iloc[-1] + timedelta(days=1),
                                         periods=forecast_periods, freq='D')
            forecast_df = pd.DataFrame({
                'timestamp': forecast_index,
                'predicted_amount': forecasts_original,
                'model': 'lstm'
            })
            
            # Calculate training metrics
            train_pred = model.predict(X, verbose=0)
            train_pred_original = self.scaler.inverse_transform(train_pred).flatten()
            y_original = self.scaler.inverse_transform(y).flatten()
            
            train_metrics = self.calculate_metrics(y_original, train_pred_original)
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            mlflow.log_metric("final_loss", history.history['loss'][-1])
            mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
            
            # Save artifacts
            model.save("lstm_model.h5")
            joblib.dump(self.scaler, "lstm_scaler.pkl")
            
            mlflow.log_artifact("lstm_model.h5")
            mlflow.log_artifact("lstm_scaler.pkl")
            
            # Clean up
            os.remove("lstm_model.h5")
            os.remove("lstm_scaler.pkl")
            
            self.model = model
            return model, forecast_df
            
    def calculate_metrics(self, actual, predicted):
        """Calculate forecasting metrics"""
        # Handle NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {"mape": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
        
        metrics = {
            'mape': mean_absolute_percentage_error(actual_clean, predicted_clean) * 100,
            'rmse': np.sqrt(mean_squared_error(actual_clean, predicted_clean)),
            'mae': np.mean(np.abs(actual_clean - predicted_clean)),
            'r2': 1 - np.sum((actual_clean - predicted_clean) ** 2) / np.sum((actual_clean - np.mean(actual_clean)) ** 2)
        }
        return metrics
        
    def train(self, df, **kwargs):
        """Train forecasting model"""
        # Prepare data
        forecast_data = self.prepare_forecast_data(df)
        
        if self.algorithm == 'prophet':
            return self.train_prophet(forecast_data, **kwargs)
        elif self.algorithm == 'arima':
            return self.train_arima(forecast_data, **kwargs)
        elif self.algorithm == 'lstm':
            return self.train_lstm(forecast_data, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
    def predict(self, periods=30):
        """Generate predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() method first.")
            
        # Implementation depends on the algorithm
        if self.algorithm == 'prophet':
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        else:
            raise NotImplementedError(f"Prediction not implemented for {self.algorithm}")

# Backward compatibility functions
def prepare_forecast_data(df):
    """Legacy function for backward compatibility"""
    forecaster = FinancialForecaster()
    return forecaster.prepare_forecast_data(df)

def train_forecast_model(df, forecast_periods=30):
    """Legacy function for backward compatibility"""
    forecaster = FinancialForecaster('prophet')
    return forecaster.train_prophet(df, forecast_periods)

def calculate_metrics(actual, predicted):
    """Legacy function for backward compatibility"""
    forecaster = FinancialForecaster()
    return forecaster.calculate_metrics(actual, predicted)

def generate_forecast(model, periods=30):
    """Legacy function for backward compatibility"""
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
    # Example usage
    sample_data = pd.read_csv("../data/sample_transactions.csv")
    sample_data['timestamp'] = pd.to_datetime(sample_data['timestamp'])
    
    print("Testing Prophet...")
    forecaster_prophet = FinancialForecaster('prophet')
    model_prophet, forecast_prophet = forecaster_prophet.train(sample_data, forecast_periods=30)
    print("Prophet model trained successfully")
    
    print("\nTesting ARIMA...")
    forecaster_arima = FinancialForecaster('arima')
    model_arima, forecast_arima = forecaster_arima.train(sample_data, order=(1,1,1), forecast_periods=30)
    print("ARIMA model trained successfully")
    
    print("\nTesting LSTM...")
    forecaster_lstm = FinancialForecaster('lstm')
    model_lstm, forecast_lstm = forecaster_lstm.train(sample_data, epochs=50, forecast_periods=30)
    print("LSTM model trained successfully")