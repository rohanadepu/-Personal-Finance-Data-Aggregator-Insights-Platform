from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, to_timestamp
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def create_spark_session():
    return SparkSession.builder \
        .appName("TransactionProcessing") \
        .config("spark.sql.warehouse.dir", "/data/warehouse") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

def clean_transactions(df):
    """Clean and standardize transaction data"""
    return df \
        .withColumn("amount", regexp_replace(col("amount"), "[$,]", "").cast("double")) \
        .withColumn("timestamp", to_timestamp(col("date"))) \
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
    # Initialize Spark session
    spark = create_spark_session()
    
    # Read raw transactions
    raw_df = spark.read.csv(
        "/data/raw/transactions/",
        header=True,
        inferSchema=True
    )
    
    # Process transactions
    processed_df = raw_df.transform(clean_transactions) \
                        .transform(categorize_transactions) \
                        .transform(detect_recurring_transactions)
    
    # Write processed data
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("/data/curated/transactions/")

if __name__ == "__main__":
    main()
