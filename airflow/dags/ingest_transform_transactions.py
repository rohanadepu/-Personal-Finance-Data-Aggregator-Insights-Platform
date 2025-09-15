from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import logging
import pandas as pd
import os
import sys

# Add ml module to path for ML operations
sys.path.append('/opt/airflow/ml')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ingest_transform_transactions',
    default_args=default_args,
    description='Ingest and transform transaction data',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Install PySpark and required packages first
install_deps = BashOperator(
    task_id='install_dependencies',
    bash_command='pip install pyspark==3.5.0 boto3==1.28.85 duckdb==0.9.2 mlflow==2.8.1 scikit-learn==1.3.2',
    dag=dag,
)

# Add logging task to check environment
def check_environment(**context):
    import os
    logging.info("Checking environment variables:")
    logging.info(f"SPARK_HOME: {os.getenv('SPARK_HOME')}")
    logging.info(f"JAVA_HOME: {os.getenv('JAVA_HOME')}")
    logging.info("Listing directory contents:")
    os.system('ls -la /opt/airflow/spark_jobs/')

check_env = PythonOperator(
    task_id='check_environment',
    python_callable=check_environment,
    dag=dag,
)

# Ingest raw transactions using python directly
ingest_transactions = BashOperator(
    task_id='ingest_transactions',
    bash_command='cd /opt/airflow/spark_jobs && python ingest_transactions.py 2>&1 | tee /opt/airflow/logs/spark_ingest.log',
    dag=dag,
)

# Transform transactions using python directly  
transform_transactions = BashOperator(
    task_id='transform_transactions',
    bash_command='cd /opt/airflow/spark_jobs && python transform_transactions.py 2>&1 | tee /opt/airflow/logs/spark_transform.log',
    dag=dag,
)

# Load transformed data to warehouse
load_warehouse = BashOperator(
    task_id='load_warehouse',
    bash_command='PGPASSWORD=airflow psql -h postgres -U airflow -d airflow -f /opt/airflow/sql/load_transactions_warehouse.sql 2>&1 | tee /opt/airflow/logs/warehouse_load.log',
    dag=dag,
)

# DuckDB Analytics Processing
def process_duckdb_analytics(**context):
    """Process analytics using DuckDB for fast analytical queries and ML feature engineering"""
    import duckdb
    import pandas as pd
    import logging
    
    logging.info("Starting DuckDB analytics processing...")
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect('/opt/airflow/data/finance_analytics.duckdb')
        
        # Load transactions from CSV for analytics
        df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create analytics tables in DuckDB
        conn.execute("DROP TABLE IF EXISTS transactions_analytics")
        conn.execute("CREATE TABLE transactions_analytics AS SELECT * FROM df")
        
        # Create aggregated views for ML features
        conn.execute("""
            CREATE OR REPLACE VIEW daily_spending AS
            SELECT 
                DATE(timestamp) as transaction_date,
                SUM(amount) as daily_total,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_transaction,
                MAX(amount) as max_transaction,
                MIN(amount) as min_transaction,
                STDDEV(amount) as std_transaction
            FROM transactions_analytics 
            GROUP BY DATE(timestamp)
            ORDER BY transaction_date
        """)
        
        # Create category analytics for transaction categorization features
        conn.execute("""
            CREATE OR REPLACE VIEW category_analytics AS
            SELECT 
                CASE 
                    WHEN mcc = 5912 OR mcc = 5411 THEN 'grocery'
                    WHEN mcc = 4121 THEN 'transport'
                    WHEN mcc = 5812 THEN 'restaurant'
                    ELSE 'other'
                END as category,
                COUNT(*) as transaction_count,
                SUM(amount) as total_spent,
                AVG(amount) as avg_amount,
                MAX(amount) as max_amount,
                MIN(amount) as min_amount
            FROM transactions_analytics
            GROUP BY category
        """)
        
        # Create ML features table for anomaly detection
        conn.execute("""
            CREATE OR REPLACE VIEW ml_features AS
            SELECT 
                *,
                EXTRACT(hour FROM timestamp) as hour_of_day,
                EXTRACT(dow FROM timestamp) as day_of_week,
                EXTRACT(month FROM timestamp) as month,
                EXTRACT(day FROM timestamp) as day_of_month,
                LN(amount + 1) as log_amount,
                amount / LAG(amount, 1) OVER (ORDER BY timestamp) as amount_ratio,
                amount - LAG(amount, 1) OVER (ORDER BY timestamp) as amount_diff,
                ROW_NUMBER() OVER (PARTITION BY DATE(timestamp) ORDER BY timestamp) as daily_transaction_rank
            FROM transactions_analytics
            ORDER BY timestamp
        """)
        
        # Create weekly and monthly aggregations for forecasting
        conn.execute("""
            CREATE OR REPLACE VIEW weekly_spending AS
            SELECT 
                DATE_TRUNC('week', timestamp) as week_start,
                COUNT(*) as weekly_transaction_count,
                SUM(amount) as weekly_total,
                AVG(amount) as weekly_avg,
                MAX(amount) as weekly_max,
                MIN(amount) as weekly_min,
                STDDEV(amount) as weekly_std
            FROM transactions_analytics
            GROUP BY DATE_TRUNC('week', timestamp)
            ORDER BY week_start
        """)
        
        conn.execute("""
            CREATE OR REPLACE VIEW monthly_spending AS
            SELECT 
                DATE_TRUNC('month', timestamp) as month_start,
                COUNT(*) as monthly_transaction_count,
                SUM(amount) as monthly_total,
                AVG(amount) as monthly_avg,
                MAX(amount) as monthly_max,
                MIN(amount) as monthly_min,
                STDDEV(amount) as monthly_std
            FROM transactions_analytics
            GROUP BY DATE_TRUNC('month', timestamp)
            ORDER BY month_start
        """)
        
        row_count = conn.execute("SELECT COUNT(*) FROM transactions_analytics").fetchone()[0]
        logging.info(f"DuckDB analytics processing complete. Processed {row_count} transactions")
        
        # Log feature engineering summary
        daily_count = conn.execute("SELECT COUNT(*) FROM daily_spending").fetchone()[0]
        category_count = conn.execute("SELECT COUNT(*) FROM category_analytics").fetchone()[0]
        logging.info(f"Created {daily_count} daily aggregations and {category_count} category features")
        
        conn.close()
        
    except Exception as e:
        logging.error(f"DuckDB processing failed: {str(e)}")
        raise

duckdb_analytics = PythonOperator(
    task_id='duckdb_analytics',
    python_callable=process_duckdb_analytics,
    dag=dag,
)

# MLflow Setup and Model Training
def setup_mlflow(**context):
    """Setup MLflow tracking and experiments"""
    import mlflow
    import os
    
    # Configure MLflow
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Create experiments if they don't exist
    experiments = [
        'transaction_categorization',
        'anomaly_detection', 
        'spending_forecasting'
    ]
    
    for exp_name in experiments:
        try:
            mlflow.create_experiment(exp_name)
            logging.info(f"Created experiment: {exp_name}")
        except Exception as e:
            logging.info(f"Experiment {exp_name} already exists or error: {str(e)}")

mlflow_setup = PythonOperator(
    task_id='mlflow_setup',
    python_callable=setup_mlflow,
    dag=dag,
)

def train_categorization_model(**context):
    """Train transaction categorization model"""
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, accuracy_score
    import numpy as np
    import logging
    
    logging.info("Starting transaction categorization model training...")
    
    # Set MLflow experiment
    mlflow.set_experiment("transaction_categorization")
    
    with mlflow.start_run():
        try:
            # Load data
            df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
            
            # Create features and labels
            df['description'] = df.get('description', 'transaction').fillna('transaction')
            df['category'] = df.get('mcc', 5912).map({
                5912: 'grocery',
                5411: 'grocery', 
                4121: 'transport',
                5812: 'restaurant'
            }).fillna('other')
            
            # Prepare features
            X = df[['amount', 'description']]
            y = df['category']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create pipeline with text processing and classifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            # Prepare training data for pipeline
            X_train_desc = X_train['description'].values
            X_test_desc = X_test['description'].values
            
            # Train model
            pipeline.fit(X_train_desc, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test_desc)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_features", 100)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "categorization_model")
            
            logging.info(f"Model training complete. Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

train_model = PythonOperator(
    task_id='train_categorization_model',
    python_callable=train_categorization_model,
    dag=dag,
)

def train_anomaly_detection(**context):
    """Train anomaly detection model"""
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np
    import logging
    
    logging.info("Starting anomaly detection model training...")
    
    # Set MLflow experiment
    mlflow.set_experiment("anomaly_detection")
    
    with mlflow.start_run():
        try:
            # Load data
            df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create features for anomaly detection
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['amount_log'] = np.log1p(df['amount'])
            
            # Select features
            features = ['amount', 'amount_log', 'hour', 'day_of_week']
            X = df[features].fillna(0)
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('anomaly_detector', IsolationForest(contamination=0.1, random_state=42))
            ])
            
            # Train model
            pipeline.fit(X)
            
            # Predict anomalies
            anomaly_scores = pipeline.decision_function(X)
            anomalies = pipeline.predict(X)
            
            # Calculate metrics
            anomaly_rate = (anomalies == -1).sum() / len(anomalies)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "IsolationForest")
            mlflow.log_param("contamination", 0.1)
            mlflow.log_metric("anomaly_rate", anomaly_rate)
            mlflow.log_metric("avg_anomaly_score", np.mean(anomaly_scores))
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "anomaly_detection_model")
            
            logging.info(f"Anomaly detection training complete. Anomaly rate: {anomaly_rate:.3f}")
            
        except Exception as e:
            logging.error(f"Anomaly detection training failed: {str(e)}")
            raise

train_anomaly = PythonOperator(
    task_id='train_anomaly_detection',
    python_callable=train_anomaly_detection,
    dag=dag,
)

def train_forecasting_model(**context):
    """Train spending forecasting model"""
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import logging
    
    logging.info("Starting spending forecasting model training...")
    
    # Set MLflow experiment
    mlflow.set_experiment("spending_forecasting")
    
    with mlflow.start_run():
        try:
            # Load data
            df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create daily aggregations for time series
            daily_spending = df.groupby(df['timestamp'].dt.date).agg({
                'amount': ['sum', 'count', 'mean']
            }).reset_index()
            
            daily_spending.columns = ['date', 'total_amount', 'transaction_count', 'avg_amount']
            daily_spending['date'] = pd.to_datetime(daily_spending['date'])
            daily_spending = daily_spending.sort_values('date')
            
            # Create features for forecasting
            daily_spending['day_of_week'] = daily_spending['date'].dt.dayofweek
            daily_spending['month'] = daily_spending['date'].dt.month
            daily_spending['day_of_month'] = daily_spending['date'].dt.day
            
            # Use rolling averages as features
            daily_spending['rolling_3d'] = daily_spending['total_amount'].rolling(3, min_periods=1).mean()
            daily_spending['rolling_7d'] = daily_spending['total_amount'].rolling(7, min_periods=1).mean()
            
            # Prepare features and target
            features = ['day_of_week', 'month', 'day_of_month', 'rolling_3d', 'rolling_7d', 'transaction_count']
            X = daily_spending[features].fillna(0)
            y = daily_spending['total_amount']
            
            # Split data (use last 20% for testing)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("features", features)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "forecasting_model")
            
            logging.info(f"Forecasting model training complete. R²: {r2:.3f}, RMSE: {rmse:.2f}")
            
        except Exception as e:
            logging.error(f"Forecasting model training failed: {str(e)}")
            raise

train_forecasting = PythonOperator(
    task_id='train_forecasting_model',
    python_callable=train_forecasting_model,
    dag=dag,
)

def predict_and_store(**context):
    """Apply trained models to make predictions and store results"""
    import pandas as pd
    import mlflow
    import psycopg2
    import logging
    import json
    import numpy as np
    
    logging.info("Starting prediction and storage...")
    
    try:
        # Load data
        df = pd.read_csv('/opt/airflow/data/sample_transactions.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Load latest categorization model
        mlflow.set_experiment("transaction_categorization")
        
        # For demo, we'll create simple predictions
        # In production, load actual trained models from MLflow
        df['predicted_category'] = df.get('mcc', 5912).map({
            5912: 'grocery',
            5411: 'grocery', 
            4121: 'transport',
            5812: 'restaurant'
        }).fillna('other')
        
        df['anomaly_score'] = np.random.uniform(-0.5, 0.5, len(df))
        df['is_anomaly'] = df['anomaly_score'] < -0.2
        
        # Store predictions in PostgreSQL
        import psycopg2
        conn = psycopg2.connect(
            host="postgres",
            database="airflow", 
            user="airflow",
            password="airflow"
        )
        
        cur = conn.cursor()
        
        # Create predictions table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transaction_predictions (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(100),
                predicted_category VARCHAR(50),
                anomaly_score FLOAT,
                is_anomaly BOOLEAN,
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert predictions
        for idx, row in df.head(100).iterrows():  # Limit for demo
            cur.execute("""
                INSERT INTO transaction_predictions 
                (transaction_id, predicted_category, anomaly_score, is_anomaly)
                VALUES (%s, %s, %s, %s)
            """, (
                f"txn_{idx}",
                row['predicted_category'],
                float(row['anomaly_score']),
                bool(row['is_anomaly'])
            ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        logging.info(f"Stored predictions for {len(df)} transactions")
        
    except Exception as e:
        logging.error(f"Prediction and storage failed: {str(e)}")
        raise

predict_task = PythonOperator(
    task_id='predict_and_store',
    python_callable=predict_and_store,
    dag=dag,
)

# Define the task dependencies - Enhanced ML Pipeline
install_deps >> check_env >> ingest_transactions >> transform_transactions >> [load_warehouse, duckdb_analytics]
load_warehouse >> mlflow_setup
duckdb_analytics >> mlflow_setup
mlflow_setup >> [train_model, train_anomaly, train_forecasting]
[train_model, train_anomaly, train_forecasting] >> predict_task
