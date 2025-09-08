from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, when, regexp_replace, to_timestamp, count, year, month
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():
    return SparkSession.builder \
        .appName("TransactionProcessing") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.11.1034") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def clean_transactions(df):
    """Clean and standardize transaction data"""
    return df \
        .withColumn("amount", regexp_replace(col("amount"), "[$,]", "").cast("double")) \
        .withColumn("timestamp", to_timestamp(col("timestamp"))) \
        .withColumn("normalized_merchant", regexp_replace(col("merchant").lower(), "[^a-z0-9 ]", "")) \
        .dropDuplicates(["tx_id"]) \
        .dropna(subset=["tx_id", "amount", "timestamp"])

def categorize_transactions(df):
    """Categorize transactions using rules and ML"""
    # Rule-based categorization
    df = df.withColumn("category",
        when(col("mcc").isin([5411, 5422, 5451]), "Groceries")
        .when(col("mcc").isin([5812, 5813, 5814]), "Dining")
        .when(col("mcc").isin([4111, 4121, 4131]), "Transportation")
        .otherwise("Other")
    )
    
    # ML-based categorization for "Other" transactions
    other_transactions = df.filter(col("category") == "Other")
    if other_transactions.count() > 0:
        # Prepare features
        indexer = StringIndexer(inputCol="normalized_merchant", outputCol="merchant_index")
        encoder = OneHotEncoder(inputCols=["merchant_index"], outputCols=["merchant_features"])
        assembler = VectorAssembler(inputCols=["merchant_features", "amount"], outputCol="features")
        
        # Train logistic regression
        lr = LogisticRegression(featuresCol="features", labelCol="category_index")
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
        model = pipeline.fit(other_transactions)
        
        # Make predictions
        predictions = model.transform(other_transactions)
        df = df.join(predictions.select("tx_id", "prediction_category"), "tx_id", "left_outer") \
            .withColumn("category", 
                when(col("category") == "Other", col("prediction_category"))
                .otherwise(col("category")))
    
    return df

def detect_recurring_transactions(df):
    """Identify recurring transactions based on patterns"""
    window_spec = Window.partitionBy("normalized_merchant", "amount") \
                       .orderBy("timestamp") \
                       .rangeBetween(-30, 30)  # 30-day window
    
    return df.withColumn("transaction_count", count("*").over(window_spec)) \
            .withColumn("is_recurring", col("transaction_count") >= 2)

def main():
    logger.info("Starting transaction transformation process")
    spark = create_spark_session()
    
    try:
        # Read raw transactions from MinIO raw bucket
        logger.info("Reading data from raw bucket")
        raw_df = spark.read.csv("s3a://raw/transaction/incoming/sample_transactions.csv", header=True, inferSchema=True)
        logger.info(f"Read {raw_df.count()} transactions from raw bucket")
        
        # Process transactions
        logger.info("Starting data transformation")
        processed_df = raw_df.transform(clean_transactions)
        logger.info("Completed data cleaning")
        
        processed_df = processed_df.transform(categorize_transactions)
        logger.info("Completed transaction categorization")
        
        processed_df = processed_df.transform(detect_recurring_transactions)
        logger.info("Completed recurring transaction detection")
        
        # Add partitioning columns
        processed_df = processed_df.withColumn("year", year("timestamp")) \
                                   .withColumn("month", month("timestamp"))
        
        # Write processed data to MinIO curated bucket
        logger.info("Writing processed data to curated bucket")
        processed_df.write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet("s3a://curated/transactions")
        logger.info("Successfully wrote processed data to curated bucket")

    except Exception as e:
        logger.error(f"Error processing transactions: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
