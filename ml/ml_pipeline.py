"""
Comprehensive ML Pipeline for Personal Finance Platform
Supports anomaly detection and forecasting with multiple algorithms
"""

import mlflow
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://localhost:5000")

class MLPipeline:
    def __init__(self, data_path=None):
        self.data_path = data_path or "../data/sample_transactions.csv"
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and preprocess transaction data"""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Add derived features
            df['category'] = df.get('mcc', 5912).map({
                5912: 'grocery',
                5411: 'grocery', 
                4121: 'transport',
                5812: 'restaurant'
            }).fillna('other')
            
            logger.info(f"Loaded {len(df)} transactions spanning {df['timestamp'].dt.date.nunique()} days")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def run_anomaly_detection(self, df):
        """Run anomaly detection experiments"""
        logger.info("Starting anomaly detection experiments...")
        
        try:
            from ml.anomaly_detection import AnomalyDetector
            
            # Test Isolation Forest
            mlflow.set_experiment("anomaly_detection_isolation_forest")
            detector_if = AnomalyDetector('isolation_forest')
            model_if, scaler_if = detector_if.train(df, contamination=0.1)
            features_df, _ = detector_if.prepare_features(df)
            anomalies_if = detector_if.detect_anomalies(features_df)
            
            self.models['isolation_forest'] = {'model': model_if, 'scaler': scaler_if, 'detector': detector_if}
            self.results['isolation_forest'] = {
                'anomalies_found': int(anomalies_if['is_anomaly'].sum()),
                'anomaly_rate': float(anomalies_if['is_anomaly'].mean()),
                'mean_score': float(anomalies_if['anomaly_score'].mean())
            }
            
            logger.info(f"Isolation Forest: {anomalies_if['is_anomaly'].sum()} anomalies detected")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            
    def run_forecasting(self, df):
        """Run forecasting experiments"""
        logger.info("Starting forecasting experiments...")
        
        try:
            from ml.forecasting import FinancialForecaster
            
            # Test Prophet
            mlflow.set_experiment("forecasting_prophet")
            forecaster_prophet = FinancialForecaster('prophet')
            model_prophet, forecast_prophet = forecaster_prophet.train(df, forecast_periods=30)
            
            self.models['prophet'] = {'model': model_prophet, 'forecaster': forecaster_prophet}
            self.results['prophet'] = {
                'forecast_periods': 30,
                'model_type': 'prophet'
            }
            
            logger.info("Prophet forecasting completed")
            
            # Test ARIMA
            mlflow.set_experiment("forecasting_arima") 
            forecaster_arima = FinancialForecaster('arima')
            model_arima, forecast_arima = forecaster_arima.train(df, order=(1,1,1), forecast_periods=30)
            
            self.models['arima'] = {'model': model_arima, 'forecaster': forecaster_arima}
            self.results['arima'] = {
                'forecast_periods': 30,
                'model_type': 'arima'
            }
            
            logger.info("ARIMA forecasting completed")
            
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            
    def generate_model_comparison_report(self):
        """Generate comprehensive model comparison report"""
        logger.info("Generating model comparison report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'results': self.results,
            'recommendations': self.get_recommendations()
        }
        
        # Save report
        report_path = f"ml_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Report saved to {report_path}")
        return report
        
    def get_recommendations(self):
        """Generate model recommendations based on results"""
        recommendations = []
        
        if 'isolation_forest' in self.results:
            if_results = self.results['isolation_forest']
            if if_results['anomaly_rate'] > 0.2:
                recommendations.append("High anomaly rate detected - consider investigating data quality")
            else:
                recommendations.append("Isolation Forest performing well for anomaly detection")
                
        if 'prophet' in self.results and 'arima' in self.results:
            recommendations.append("Both Prophet and ARIMA models trained - compare forecasting accuracy")
            
        return recommendations
        
    def run_full_pipeline(self):
        """Run the complete ML pipeline"""
        logger.info("Starting full ML pipeline...")
        
        try:
            # Load data
            df = self.load_data()
            
            # Run anomaly detection
            self.run_anomaly_detection(df)
            
            # Run forecasting
            self.run_forecasting(df)
            
            # Generate report
            report = self.generate_model_comparison_report()
            
            logger.info("ML pipeline completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

class ModelServer:
    """Simple model serving utility"""
    
    def __init__(self):
        self.models = {}
        
    def load_model(self, model_name, model_uri):
        """Load model from MLflow"""
        try:
            if 'isolation_forest' in model_name:
                model = mlflow.sklearn.load_model(model_uri)
            elif 'prophet' in model_name:
                model = mlflow.prophet.load_model(model_uri)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
            self.models[model_name] = model
            logger.info(f"Loaded model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            
    def predict_anomalies(self, data, model_name='isolation_forest'):
        """Predict anomalies using loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        # This would need the same feature engineering as training
        # Simplified for demonstration
        predictions = self.models[model_name].predict(data)
        return predictions
        
    def predict_forecast(self, periods=30, model_name='prophet'):
        """Generate forecast using loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model = self.models[model_name]
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast

if __name__ == "__main__":
    # Run the ML pipeline
    pipeline = MLPipeline()
    report = pipeline.run_full_pipeline()
    
    print("\n" + "="*50)
    print("ML PIPELINE SUMMARY")
    print("="*50)
    print(f"Models trained: {', '.join(report['models_trained'])}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    print("="*50)
