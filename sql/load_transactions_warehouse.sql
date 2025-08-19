CREATE TABLE IF NOT EXISTS transactions (
    tx_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP,
    amount DECIMAL(15,2),
    currency VARCHAR(3),
    merchant VARCHAR(255),
    mcc VARCHAR(4),
    account_id VARCHAR(50),
    city VARCHAR(100),
    state VARCHAR(2),
    channel VARCHAR(20),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    is_recurring BOOLEAN,
    normalized_amount DECIMAL(15,2)
);

-- Create extension for S3 foreign data wrapper if not exists
CREATE EXTENSION IF NOT EXISTS aws_s3_fdw;

-- Create server for MinIO connection
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_foreign_server WHERE srvname = 'minio_server') THEN
        CREATE SERVER minio_server
            FOREIGN DATA WRAPPER aws_s3_fdw
            OPTIONS (
                endpoint 'http://minio:9000',
                aws_access_key_id 'minioadmin',
                aws_secret_access_key 'minioadmin'
            );
    END IF;
END
$$;

-- Import data from MinIO curated bucket
INSERT INTO transactions (
    tx_id, timestamp, amount, currency, merchant, 
    mcc, account_id, city, state, channel,
    category, subcategory, is_recurring, normalized_amount
)
SELECT * FROM aws_s3.query_parquet(
    'minio_server',
    's3://curated/transactions/*.parquet',
    table_columns := ARRAY[
        'tx_id VARCHAR',
        'timestamp TIMESTAMP',
        'amount DECIMAL(15,2)',
        'currency VARCHAR',
        'merchant VARCHAR',
        'mcc VARCHAR',
        'account_id VARCHAR',
        'city VARCHAR',
        'state VARCHAR',
        'channel VARCHAR',
        'category VARCHAR',
        'subcategory VARCHAR',
        'is_recurring BOOLEAN',
        'normalized_amount DECIMAL(15,2)'
    ]
)
ON CONFLICT (tx_id) 
DO UPDATE SET
    category = EXCLUDED.category,
    subcategory = EXCLUDED.subcategory,
    is_recurring = EXCLUDED.is_recurring,
    normalized_amount = EXCLUDED.normalized_amount;