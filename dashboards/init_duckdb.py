#!/usr/bin/env python3
"""
Initialize DuckDB with data for Docker container
"""

import os
import sys
import pandas as pd
import duckdb
import psycopg2
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_duckdb():
    """Initialize DuckDB file with data from PostgreSQL"""
    
    duckdb_path = os.getenv('DUCKDB_PATH', '/shared_data/finance_analytics.duckdb')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
    
    logger.info(f"Initializing DuckDB at: {duckdb_path}")
    
    try:
        # Connect to PostgreSQL
        connection_string = "postgresql://airflow:airflow@postgres:5432/airflow"
        pg_engine = create_engine(connection_string)
        
        # Read data from PostgreSQL
        df = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp", pg_engine)
        logger.info(f"Read {len(df)} transactions from PostgreSQL")
        
        if df.empty:
            logger.warning("No data found in PostgreSQL, using sample data")
            # Use sample data as fallback
            df = pd.read_csv('/app/data/sample_transactions.csv') if os.path.exists('/app/data/sample_transactions.csv') else pd.DataFrame()
        
        if not df.empty:
            # Create DuckDB and populate with data
            duck_conn = duckdb.connect(database=duckdb_path, read_only=False)
            duck_conn.execute("DROP TABLE IF EXISTS transactions")
            duck_conn.execute("CREATE TABLE transactions AS SELECT * FROM df")
            
            # Create indexes for better performance
            duck_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions(timestamp)")
            duck_conn.execute("CREATE INDEX IF NOT EXISTS idx_merchant ON transactions(merchant)")
            duck_conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON transactions(category)")
            
            count = duck_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            logger.info(f"Successfully initialized DuckDB with {count} transactions")
            duck_conn.close()
        else:
            logger.error("No data available to initialize DuckDB")
            
    except Exception as e:
        logger.error(f"Error initializing DuckDB: {e}")
        # Create empty DuckDB structure at least
        try:
            duck_conn = duckdb.connect(database=duckdb_path, read_only=False)
            duck_conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_id VARCHAR,
                    timestamp TIMESTAMP,
                    amount DOUBLE,
                    currency VARCHAR,
                    merchant VARCHAR,
                    mcc VARCHAR,
                    account_id VARCHAR,
                    city VARCHAR,
                    state VARCHAR,
                    channel VARCHAR,
                    category VARCHAR,
                    subcategory VARCHAR,
                    is_recurring BOOLEAN,
                    normalized_amount DOUBLE
                )
            """)
            duck_conn.close()
            logger.info("Created empty DuckDB structure")
        except Exception as e2:
            logger.error(f"Failed to create empty DuckDB: {e2}")

if __name__ == "__main__":
    init_duckdb()
