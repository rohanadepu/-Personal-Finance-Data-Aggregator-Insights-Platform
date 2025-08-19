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

INSERT INTO transactions (
    tx_id, timestamp, amount, currency, merchant, 
    mcc, account_id, city, state, channel,
    category, subcategory, is_recurring, normalized_amount
)
SELECT * FROM read_parquet('/data/curated/transactions/*.parquet')
ON CONFLICT (tx_id) 
DO UPDATE SET
    category = EXCLUDED.category,
    subcategory = EXCLUDED.subcategory,
    is_recurring = EXCLUDED.is_recurring,
    normalized_amount = EXCLUDED.normalized_amount;