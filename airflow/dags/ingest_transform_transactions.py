from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import logging

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
    bash_command='pip install pyspark==3.5.0 boto3==1.28.85',
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

# Define the task dependencies
install_deps >> check_env >> ingest_transactions >> transform_transactions >> load_warehouse
