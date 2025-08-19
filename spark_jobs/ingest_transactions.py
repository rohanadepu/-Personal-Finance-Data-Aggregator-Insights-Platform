from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, year, month, day
import os

def create_spark_session():
    return SparkSession.builder \
        .appName("TransactionIngestion") \
        .config("spark.sql.warehouse.dir", "/data/warehouse") \
        .getOrCreate()

def ingest_transactions():
    spark = create_spark_session()
    
    # Read from incoming directory
    input_path = "/opt/airflow/data/raw/transactions/incoming/*.csv"
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Add ingestion timestamp
    df = df.withColumn("ingestion_timestamp", current_timestamp())
    
    # Write to processed directory with date partitioning
    output_path = "/opt/airflow/data/raw/transactions/processed"
    df.write \
        .partitionBy(year("timestamp"), month("timestamp"), day("timestamp")) \
        .mode("append") \
        .parquet(output_path)
    
    # Clean up processed files
    spark.stop()

if __name__ == "__main__":
    ingest_transactions()