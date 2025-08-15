from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
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

# Ingest raw transactions
ingest_transactions = SparkSubmitOperator(
    task_id='ingest_transactions',
    application='/opt/airflow/spark_jobs/ingest_transactions.py',
    name='ingest_transactions',
    conn_id='spark_default',
    dag=dag,
)

# Run data quality checks on raw data
validate_raw_data = GreatExpectationsOperator(
    task_id='validate_raw_data',
    expectation_suite_name='transactions_raw_suite',
    batch_kwargs={
        'path': '/data/raw/transactions/',
        'datasource': 'raw_transactions'
    },
    dag=dag,
)

# Transform and categorize transactions
transform_transactions = SparkSubmitOperator(
    task_id='transform_transactions',
    application='/opt/airflow/spark_jobs/transform_transactions.py',
    name='transform_transactions',
    conn_id='spark_default',
    dag=dag,
)

# Load transformed data to warehouse
load_warehouse = PostgresOperator(
    task_id='load_warehouse',
    postgres_conn_id='warehouse_default',
    sql='sql/load_transactions_warehouse.sql',
    dag=dag,
)

# Run data quality checks on transformed data
validate_transformed_data = GreatExpectationsOperator(
    task_id='validate_transformed_data',
    expectation_suite_name='transactions_transformed_suite',
    batch_kwargs={
        'path': '/data/curated/transactions/',
        'datasource': 'curated_transactions'
    },
    dag=dag,
)

# Define the task dependencies
ingest_transactions >> validate_raw_data >> transform_transactions >> load_warehouse >> validate_transformed_data
