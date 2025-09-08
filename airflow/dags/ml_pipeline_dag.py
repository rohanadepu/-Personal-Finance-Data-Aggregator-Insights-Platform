"""
Airflow DAG for ML Pipeline - Anomaly Detection and Forecasting
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
import os
import sys

# Add ml module to path
sys.path.append('/opt/airflow/ml')

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline_finance',
    default_args=default_args,
    description='ML Pipeline for Personal Finance Analytics',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'finance', 'anomaly-detection', 'forecasting'],
)

def train_anomaly_models(**context):
    """Train anomaly detection models"""
    import pandas as pd
    from ml.anomaly_detection import AnomalyDetector
    import mlflow
    
    # Configure MLflow
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Load data
    df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add category mapping
    df['category'] = df.get('mcc', 5912).map({
        5912: 'grocery',
        5411: 'grocery', 
        4121: 'transport',
        5812: 'restaurant'
    }).fillna('other')
    
    # Train Isolation Forest
    detector_if = AnomalyDetector('isolation_forest')
    model_if, scaler_if = detector_if.train(df, contamination=0.1)
    
    # Log results to XCom
    features_df, _ = detector_if.prepare_features(df)
    anomalies_if = detector_if.detect_anomalies(features_df)
    
    results = {
        'anomalies_found': int(anomalies_if['is_anomaly'].sum()),
        'total_records': len(features_df),
        'anomaly_rate': float(anomalies_if['is_anomaly'].mean())
    }
    
    return results

def train_forecasting_models(**context):
    """Train forecasting models"""
    import pandas as pd
    from ml.forecasting import FinancialForecaster
    import mlflow
    
    # Configure MLflow
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Load data
    df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    results = {}
    
    # Train Prophet
    try:
        forecaster_prophet = FinancialForecaster('prophet')
        model_prophet, forecast_prophet = forecaster_prophet.train(df, forecast_periods=30)
        results['prophet'] = {'status': 'success', 'forecast_periods': 30}
    except Exception as e:
        results['prophet'] = {'status': 'failed', 'error': str(e)}
    
    # Train ARIMA
    try:
        forecaster_arima = FinancialForecaster('arima')
        model_arima, forecast_arima = forecaster_arima.train(df, order=(1,1,1), forecast_periods=30)
        results['arima'] = {'status': 'success', 'forecast_periods': 30}
    except Exception as e:
        results['arima'] = {'status': 'failed', 'error': str(e)}
    
    return results

def evaluate_models(**context):
    """Evaluate and compare trained models"""
    import json
    
    # Get results from previous tasks
    anomaly_results = context['ti'].xcom_pull(task_ids='train_anomaly_detection')
    forecast_results = context['ti'].xcom_pull(task_ids='train_forecasting')
    
    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'anomaly_detection': anomaly_results,
        'forecasting': forecast_results,
        'recommendations': []
    }
    
    # Add recommendations based on results
    if anomaly_results and anomaly_results['anomaly_rate'] > 0.2:
        evaluation_report['recommendations'].append(
            "High anomaly rate detected - investigate data quality"
        )
    
    if forecast_results:
        successful_models = [k for k, v in forecast_results.items() if v.get('status') == 'success']
        evaluation_report['recommendations'].append(
            f"Successfully trained forecasting models: {', '.join(successful_models)}"
        )
    
    # Save evaluation report
    report_path = f"/opt/airflow/data/ml_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print("Model Evaluation Complete!")
    print(f"Report saved to: {report_path}")
    
    return evaluation_report

def cleanup_old_models(**context):
    """Cleanup old model artifacts and logs"""
    import os
    import glob
    from datetime import datetime, timedelta
    
    # Clean up old evaluation reports (keep last 7 days)
    cutoff_date = datetime.now() - timedelta(days=7)
    cutoff_str = cutoff_date.strftime('%Y%m%d')
    
    report_pattern = "/opt/airflow/data/ml_evaluation_*.json"
    for report_file in glob.glob(report_pattern):
        # Extract date from filename
        try:
            file_date_str = report_file.split('_')[-2]  # Extract YYYYMMDD
            if file_date_str < cutoff_str:
                os.remove(report_file)
                print(f"Removed old report: {report_file}")
        except:
            pass  # Skip files that don't match expected pattern
    
    return "Cleanup completed"

# Define tasks
wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='/opt/airflow/data/sample_transactions.csv',
    fs_conn_id='fs_default',
    poke_interval=30,
    timeout=300,
    dag=dag,
)

train_anomaly_detection = PythonOperator(
    task_id='train_anomaly_detection',
    python_callable=train_anomaly_models,
    dag=dag,
)

train_forecasting = PythonOperator(
    task_id='train_forecasting',
    python_callable=train_forecasting_models,
    dag=dag,
)

evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_models',
    python_callable=cleanup_old_models,
    dag=dag,
)

# Define task dependencies
wait_for_data >> [train_anomaly_detection, train_forecasting]
[train_anomaly_detection, train_forecasting] >> evaluate_models_task
evaluate_models_task >> cleanup_task
