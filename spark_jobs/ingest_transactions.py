from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, year, month, day
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():
    return SparkSession.builder \
        .appName("TransactionIngestion") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

def ingest_transactions():
    logger.info("Starting transaction ingestion process")
    spark = create_spark_session()
    
    try:
        # Read from incoming directory
        input_path = "s3a://raw/transactions/incoming/*.csv"
        logger.info(f"Reading transactions from {input_path}")
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        logger.info(f"Read {df.count()} transactions")
        
        # Add ingestion timestamp
        df = df.withColumn("ingestion_timestamp", current_timestamp())
        
        # Write to processed directory with date partitioning
        output_path = "s3a://raw/transactions/processed"
        logger.info(f"Writing processed transactions to {output_path}")
        df.write \
            .partitionBy(year("timestamp"), month("timestamp"), day("timestamp")) \
            .mode("append") \
            .parquet(output_path)
        logger.info("Successfully wrote transactions to processed directory")
        
    except Exception as e:
        logger.error(f"Error processing transactions: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    ingest_transactions()