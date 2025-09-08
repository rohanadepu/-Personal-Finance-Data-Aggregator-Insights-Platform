"""
MLflow Integration Script for Personal Finance ML Platform

This script provides comprehensive MLflow integration including:
- Model tracking and versioning
- Experiment management
- Model deployment utilities
- Performance monitoring
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.prophet
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowManager:
    """Comprehensive MLflow manager for the personal finance platform"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """Initialize MLflow manager with tracking URI"""
        self.tracking_uri = tracking_uri
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking and experiments"""
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Configure S3/MinIO for artifact storage
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
        
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        
    def create_experiments(self):
        """Create all necessary experiments"""
        experiments = [
            "anomaly_detection_isolation_forest",
            "anomaly_detection_autoencoder", 
            "forecasting_prophet",
            "forecasting_arima",
            "forecasting_lstm",
            "model_comparison",
            "production_models"
        ]
        
        created_experiments = []
        for exp_name in experiments:
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(exp_name)
                    logger.info(f"Created experiment: {exp_name} (ID: {experiment_id})")
                    created_experiments.append(exp_name)
                else:
                    logger.info(f"Experiment already exists: {exp_name}")
            except Exception as e:
                logger.error(f"Error creating experiment {exp_name}: {e}")
                
        return created_experiments
        
    def log_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Log comprehensive dataset information"""
        dataset_info = {
            "dataset_name": dataset_name,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "date_range_start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
            "date_range_end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None,
            "date_span_days": (df['timestamp'].max() - df['timestamp'].min()).days if 'timestamp' in df.columns else None
        }
        
        # Log basic dataset metrics
        for col in df.select_dtypes(include=[np.number]).columns:
            dataset_info[f"{col}_mean"] = float(df[col].mean())
            dataset_info[f"{col}_std"] = float(df[col].std())
            dataset_info[f"{col}_min"] = float(df[col].min())
            dataset_info[f"{col}_max"] = float(df[col].max())
            dataset_info[f"{col}_null_count"] = int(df[col].isnull().sum())
            
        return dataset_info
        
    def run_anomaly_detection_experiments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive anomaly detection experiments"""
        logger.info("Starting anomaly detection experiments...")
        
        from ml.anomaly_detection import AnomalyDetector
        
        results = {}
        
        # Test different contamination levels for Isolation Forest
        contamination_levels = [0.05, 0.1, 0.15, 0.2]
        
        for contamination in contamination_levels:
            mlflow.set_experiment("anomaly_detection_isolation_forest")
            
            with mlflow.start_run(run_name=f"isolation_forest_contamination_{contamination}"):
                try:
                    # Log dataset info
                    dataset_info = self.log_dataset_info(df, "transaction_data")
                    for key, value in dataset_info.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"dataset_{key}", value)
                        else:
                            mlflow.log_param(f"dataset_{key}", str(value))
                    
                    # Train model
                    detector = AnomalyDetector('isolation_forest')
                    model, scaler = detector.train(df, contamination=contamination)
                    
                    # Detect anomalies
                    features_df, _ = detector.prepare_features(df)
                    anomalies = detector.detect_anomalies(features_df)
                    
                    # Store results
                    results[f"isolation_forest_{contamination}"] = {
                        "contamination": contamination,
                        "anomalies_detected": int(anomalies['is_anomaly'].sum()),
                        "anomaly_rate": float(anomalies['is_anomaly'].mean()),
                        "mean_anomaly_score": float(anomalies['anomaly_score'].mean()),
                        "run_id": mlflow.active_run().info.run_id
                    }
                    
                    logger.info(f"Isolation Forest (contamination={contamination}): "
                              f"{anomalies['is_anomaly'].sum()} anomalies detected")
                    
                except Exception as e:
                    logger.error(f"Error in Isolation Forest experiment (contamination={contamination}): {e}")
        
        # Test Autoencoder with different architectures
        mlflow.set_experiment("anomaly_detection_autoencoder")
        
        encoding_dims = [8, 16, 32]
        
        for encoding_dim in encoding_dims:
            with mlflow.start_run(run_name=f"autoencoder_encoding_dim_{encoding_dim}"):
                try:
                    # Log dataset info
                    dataset_info = self.log_dataset_info(df, "transaction_data")
                    for key, value in dataset_info.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"dataset_{key}", value)
                        else:
                            mlflow.log_param(f"dataset_{key}", str(value))
                    
                    # Train model
                    detector = AnomalyDetector('autoencoder')
                    model, scaler, threshold = detector.train(df, encoding_dim=encoding_dim, epochs=50)
                    
                    # Detect anomalies
                    features_df, _ = detector.prepare_features(df)
                    anomalies = detector.detect_anomalies(features_df)
                    
                    # Store results
                    results[f"autoencoder_{encoding_dim}"] = {
                        "encoding_dim": encoding_dim,
                        "anomalies_detected": int(anomalies['is_anomaly'].sum()),
                        "anomaly_rate": float(anomalies['is_anomaly'].mean()),
                        "mean_reconstruction_error": float(anomalies['anomaly_score'].mean()),
                        "threshold": float(threshold),
                        "run_id": mlflow.active_run().info.run_id
                    }
                    
                    logger.info(f"Autoencoder (encoding_dim={encoding_dim}): "
                              f"{anomalies['is_anomaly'].sum()} anomalies detected")
                    
                except Exception as e:
                    logger.error(f"Error in Autoencoder experiment (encoding_dim={encoding_dim}): {e}")
        
        return results
        
    def run_forecasting_experiments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive forecasting experiments"""
        logger.info("Starting forecasting experiments...")
        
        from ml.forecasting import FinancialForecaster
        
        results = {}
        forecast_periods = [7, 14, 30]
        
        # Test Prophet with different seasonality modes
        seasonality_modes = ['additive', 'multiplicative']
        
        for periods in forecast_periods:
            for seasonality_mode in seasonality_modes:
                mlflow.set_experiment("forecasting_prophet")
                
                with mlflow.start_run(run_name=f"prophet_{periods}d_{seasonality_mode}"):
                    try:
                        # Log dataset info
                        dataset_info = self.log_dataset_info(df, "transaction_data")
                        for key, value in dataset_info.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"dataset_{key}", value)
                            else:
                                mlflow.log_param(f"dataset_{key}", str(value))
                        
                        # Prepare data
                        forecaster = FinancialForecaster('prophet')
                        forecast_data = forecaster.prepare_forecast_data(df)
                        
                        # Train model
                        model, forecast = forecaster.train_prophet(
                            forecast_data, 
                            forecast_periods=periods,
                            seasonality_mode=seasonality_mode
                        )
                        
                        # Store results
                        future_forecast = forecast['yhat'][-periods:]
                        results[f"prophet_{periods}d_{seasonality_mode}"] = {
                            "algorithm": "prophet",
                            "forecast_periods": periods,
                            "seasonality_mode": seasonality_mode,
                            "forecast_mean": float(future_forecast.mean()),
                            "forecast_std": float(future_forecast.std()),
                            "run_id": mlflow.active_run().info.run_id
                        }
                        
                        logger.info(f"Prophet forecast completed: {periods} days, {seasonality_mode} seasonality")
                        
                    except Exception as e:
                        logger.error(f"Error in Prophet experiment ({periods}d, {seasonality_mode}): {e}")
        
        # Test ARIMA with different orders
        arima_orders = [(1,1,1), (2,1,2), (1,1,2)]
        
        for periods in [7, 14, 30]:
            for order in arima_orders:
                mlflow.set_experiment("forecasting_arima")
                
                with mlflow.start_run(run_name=f"arima_{periods}d_order{order}"):
                    try:
                        # Log dataset info
                        dataset_info = self.log_dataset_info(df, "transaction_data")
                        for key, value in dataset_info.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"dataset_{key}", value)
                            else:
                                mlflow.log_param(f"dataset_{key}", str(value))
                        
                        # Prepare data
                        forecaster = FinancialForecaster('arima')
                        forecast_data = forecaster.prepare_forecast_data(df)
                        
                        # Train model
                        model, forecast_df = forecaster.train_arima(
                            forecast_data, 
                            order=order,
                            forecast_periods=periods
                        )
                        
                        # Store results
                        results[f"arima_{periods}d_{order}"] = {
                            "algorithm": "arima",
                            "forecast_periods": periods,
                            "order": order,
                            "forecast_mean": float(forecast_df['predicted_amount'].mean()),
                            "forecast_std": float(forecast_df['predicted_amount'].std()),
                            "run_id": mlflow.active_run().info.run_id
                        }
                        
                        logger.info(f"ARIMA forecast completed: {periods} days, order {order}")
                        
                    except Exception as e:
                        logger.error(f"Error in ARIMA experiment ({periods}d, order {order}): {e}")
        
        return results
        
    def compare_models(self, anomaly_results: Dict, forecasting_results: Dict) -> Dict[str, Any]:
        """Compare model performance and generate recommendations"""
        mlflow.set_experiment("model_comparison")
        
        with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            comparison_results = {
                "timestamp": datetime.now().isoformat(),
                "anomaly_detection": anomaly_results,
                "forecasting": forecasting_results,
                "recommendations": []
            }
            
            # Analyze anomaly detection results
            if anomaly_results:
                # Find best contamination level for Isolation Forest
                if_results = {k: v for k, v in anomaly_results.items() if k.startswith('isolation_forest')}
                if if_results:
                    best_if = min(if_results.items(), key=lambda x: abs(x[1]['anomaly_rate'] - 0.1))
                    comparison_results["recommendations"].append(
                        f"Best Isolation Forest contamination: {best_if[1]['contamination']} "
                        f"(anomaly rate: {best_if[1]['anomaly_rate']:.2%})"
                    )
                    mlflow.log_metric("best_if_contamination", best_if[1]['contamination'])
                    mlflow.log_metric("best_if_anomaly_rate", best_if[1]['anomaly_rate'])
                
                # Find best autoencoder architecture
                ae_results = {k: v for k, v in anomaly_results.items() if k.startswith('autoencoder')}
                if ae_results:
                    best_ae = min(ae_results.items(), key=lambda x: x[1]['mean_reconstruction_error'])
                    comparison_results["recommendations"].append(
                        f"Best Autoencoder encoding dimension: {best_ae[1]['encoding_dim']} "
                        f"(mean reconstruction error: {best_ae[1]['mean_reconstruction_error']:.4f})"
                    )
                    mlflow.log_metric("best_ae_encoding_dim", best_ae[1]['encoding_dim'])
                    mlflow.log_metric("best_ae_reconstruction_error", best_ae[1]['mean_reconstruction_error'])
            
            # Analyze forecasting results  
            if forecasting_results:
                prophet_results = {k: v for k, v in forecasting_results.items() if k.startswith('prophet')}
                arima_results = {k: v for k, v in forecasting_results.items() if k.startswith('arima')}
                
                if prophet_results:
                    # Find most stable Prophet configuration (lowest std)
                    best_prophet = min(prophet_results.items(), key=lambda x: x[1]['forecast_std'])
                    comparison_results["recommendations"].append(
                        f"Most stable Prophet config: {best_prophet[1]['forecast_periods']} days, "
                        f"{best_prophet[1]['seasonality_mode']} seasonality "
                        f"(forecast std: {best_prophet[1]['forecast_std']:.2f})"
                    )
                    mlflow.log_metric("best_prophet_periods", best_prophet[1]['forecast_periods'])
                    mlflow.log_param("best_prophet_seasonality", best_prophet[1]['seasonality_mode'])
                
                if arima_results:
                    # Find most stable ARIMA configuration
                    best_arima = min(arima_results.items(), key=lambda x: x[1]['forecast_std'])
                    comparison_results["recommendations"].append(
                        f"Most stable ARIMA config: {best_arima[1]['forecast_periods']} days, "
                        f"order {best_arima[1]['order']} "
                        f"(forecast std: {best_arima[1]['forecast_std']:.2f})"
                    )
                    mlflow.log_metric("best_arima_periods", best_arima[1]['forecast_periods'])
                    mlflow.log_param("best_arima_order", str(best_arima[1]['order']))
            
            # Log comparison results
            mlflow.log_param("total_models_tested", len(anomaly_results) + len(forecasting_results))
            mlflow.log_param("anomaly_models_tested", len(anomaly_results))
            mlflow.log_param("forecasting_models_tested", len(forecasting_results))
            
            # Save detailed comparison report
            comparison_json = json.dumps(comparison_results, indent=2)
            with open("model_comparison_report.json", "w") as f:
                f.write(comparison_json)
            mlflow.log_artifact("model_comparison_report.json")
            os.remove("model_comparison_report.json")
            
            return comparison_results
    
    def promote_best_models(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Promote best performing models to production"""
        mlflow.set_experiment("production_models")
        
        promoted_models = []
        
        # Promote best anomaly detection model
        try:
            anomaly_results = comparison_results.get("anomaly_detection", {})
            if anomaly_results:
                # Get the best performing models
                if_results = {k: v for k, v in anomaly_results.items() if k.startswith('isolation_forest')}
                ae_results = {k: v for k, v in anomaly_results.items() if k.startswith('autoencoder')}
                
                # Promote best Isolation Forest
                if if_results:
                    best_if = min(if_results.items(), key=lambda x: abs(x[1]['anomaly_rate'] - 0.1))
                    best_if_run_id = best_if[1]['run_id']
                    
                    # Register model
                    model_uri = f"runs:/{best_if_run_id}/isolation_forest_model"
                    model_name = "isolation_forest_production"
                    
                    try:
                        mv = mlflow.register_model(model_uri, model_name)
                        # Transition to production
                        client = mlflow.tracking.MlflowClient()
                        client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Production"
                        )
                        promoted_models.append(f"{model_name}_v{mv.version}")
                        logger.info(f"Promoted Isolation Forest model to production: {model_name}_v{mv.version}")
                    except Exception as e:
                        logger.error(f"Error promoting Isolation Forest model: {e}")
                
                # Promote best Autoencoder
                if ae_results:
                    best_ae = min(ae_results.items(), key=lambda x: x[1]['mean_reconstruction_error'])
                    best_ae_run_id = best_ae[1]['run_id']
                    
                    # Register model
                    model_uri = f"runs:/{best_ae_run_id}/autoencoder_model"
                    model_name = "autoencoder_production"
                    
                    try:
                        mv = mlflow.register_model(model_uri, model_name)
                        # Transition to production
                        client = mlflow.tracking.MlflowClient()
                        client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Production"
                        )
                        promoted_models.append(f"{model_name}_v{mv.version}")
                        logger.info(f"Promoted Autoencoder model to production: {model_name}_v{mv.version}")
                    except Exception as e:
                        logger.error(f"Error promoting Autoencoder model: {e}")
        
        except Exception as e:
            logger.error(f"Error promoting anomaly detection models: {e}")
        
        # Promote best forecasting model
        try:
            forecasting_results = comparison_results.get("forecasting", {})
            if forecasting_results:
                prophet_results = {k: v for k, v in forecasting_results.items() if k.startswith('prophet')}
                
                if prophet_results:
                    best_prophet = min(prophet_results.items(), key=lambda x: x[1]['forecast_std'])
                    best_prophet_run_id = best_prophet[1]['run_id']
                    
                    # Register model
                    model_uri = f"runs:/{best_prophet_run_id}/prophet_model"
                    model_name = "prophet_production"
                    
                    try:
                        mv = mlflow.register_model(model_uri, model_name)
                        # Transition to production
                        client = mlflow.tracking.MlflowClient()
                        client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Production"
                        )
                        promoted_models.append(f"{model_name}_v{mv.version}")
                        logger.info(f"Promoted Prophet model to production: {model_name}_v{mv.version}")
                    except Exception as e:
                        logger.error(f"Error promoting Prophet model: {e}")
        
        except Exception as e:
            logger.error(f"Error promoting forecasting models: {e}")
        
        return promoted_models

def run_complete_ml_pipeline(data_path: str = "../data/sample_transactions.csv"):
    """Run the complete ML pipeline with MLflow integration"""
    logger.info("Starting complete ML pipeline with MLflow integration...")
    
    try:
        # Initialize MLflow manager
        mlflow_manager = MLflowManager()
        
        # Create experiments
        created_experiments = mlflow_manager.create_experiments()
        logger.info(f"Created/verified {len(created_experiments)} experiments")
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df)} transactions spanning {df['timestamp'].dt.date.nunique()} days")
        
        # Run anomaly detection experiments
        anomaly_results = mlflow_manager.run_anomaly_detection_experiments(df)
        logger.info(f"Completed {len(anomaly_results)} anomaly detection experiments")
        
        # Run forecasting experiments
        forecasting_results = mlflow_manager.run_forecasting_experiments(df)
        logger.info(f"Completed {len(forecasting_results)} forecasting experiments")
        
        # Compare models and generate recommendations
        comparison_results = mlflow_manager.compare_models(anomaly_results, forecasting_results)
        logger.info("Model comparison completed")
        
        # Promote best models to production
        promoted_models = mlflow_manager.promote_best_models(comparison_results)
        logger.info(f"Promoted {len(promoted_models)} models to production: {promoted_models}")
        
        # Print summary
        print("\n" + "="*80)
        print("ML PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"📊 Data processed: {len(df)} transactions")
        print(f"🔍 Anomaly detection models tested: {len(anomaly_results)}")
        print(f"📈 Forecasting models tested: {len(forecasting_results)}")
        print(f"🚀 Models promoted to production: {len(promoted_models)}")
        print("\n📋 RECOMMENDATIONS:")
        for rec in comparison_results["recommendations"]:
            print(f"  • {rec}")
        print("\n🏭 PRODUCTION MODELS:")
        for model in promoted_models:
            print(f"  • {model}")
        print("="*80)
        print("✅ Pipeline completed successfully!")
        print(f"🌐 MLflow UI: http://localhost:5000")
        print("="*80)
        
        return {
            "anomaly_results": anomaly_results,
            "forecasting_results": forecasting_results,
            "comparison_results": comparison_results,
            "promoted_models": promoted_models
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_ml_pipeline()
