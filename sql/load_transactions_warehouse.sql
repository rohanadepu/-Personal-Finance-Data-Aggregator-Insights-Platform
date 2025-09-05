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

-- For now, we'll use a simpler approach without the S3 foreign data wrapper
-- Data will be loaded directly through the application