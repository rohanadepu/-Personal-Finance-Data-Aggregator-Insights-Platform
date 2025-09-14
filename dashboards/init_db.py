#!/usr/bin/env python3
"""
Database initialization script for Personal Finance Dashboard
This script sets up PostgreSQL tables and loads sample data for testing
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_postgres_connection():
    """Create PostgreSQL connection"""
    try:
        connection_string = "postgresql://airflow:airflow@localhost:5432/airflow"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return None

def create_tables(engine):
    """Create the transactions table if it doesn't exist"""
    create_table_sql = """
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
        normalized_amount DECIMAL(15,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant);
    CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
    CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id);
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        logger.info("Tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

def load_sample_data(engine):
    """Load sample data from CSV file"""
    sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_transactions.csv')
    
    if not os.path.exists(sample_data_path):
        logger.error(f"Sample data file not found: {sample_data_path}")
        return False
    
    try:
        # Read the sample data
        df = pd.read_csv(sample_data_path)
        logger.info(f"Loaded {len(df)} sample transactions")
        
        # Add missing columns with default values
        df['city'] = 'Sample City'
        df['state'] = 'CA'
        df['channel'] = 'online'
        
        # Map merchants to categories
        merchant_categories = {
            'Walmart': 'retail',
            'Costco': 'retail', 
            'Amazon': 'retail',
            'Target': 'retail',
            'Whole Foods': 'grocery',
            'Starbucks': 'food',
            'McDonalds': 'food',
            'Shell': 'gas',
            'Exxon': 'gas',
            'Chase': 'banking'
        }
        
        df['category'] = df['merchant'].map(lambda x: merchant_categories.get(x, 'other'))
        df['subcategory'] = 'general'
        df['is_recurring'] = False
        df['normalized_amount'] = df['amount']
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Check if data already exists
        with engine.connect() as conn:
            existing_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
            
        if existing_count > 0:
            logger.info(f"Data already exists ({existing_count} records). Skipping load.")
            return True
        
        # Load data to PostgreSQL
        df.to_sql('transactions', engine, if_exists='append', index=False)
        logger.info(f"Successfully loaded {len(df)} transactions to PostgreSQL")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        return False

def verify_data(engine):
    """Verify that data was loaded correctly"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
            logger.info(f"Total transactions in database: {result}")
            
            # Show sample records
            sample_result = conn.execute(text("""
                SELECT tx_id, timestamp, merchant, amount, category 
                FROM transactions 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)).fetchall()
            
            logger.info("Sample transactions:")
            for row in sample_result:
                logger.info(f"  {row}")
                
        return True
    except Exception as e:
        logger.error(f"Failed to verify data: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    # Connect to PostgreSQL
    engine = get_postgres_connection()
    if not engine:
        logger.error("Cannot proceed without database connection")
        return False
    
    # Create tables
    if not create_tables(engine):
        logger.error("Failed to create tables")
        return False
    
    # Load sample data
    if not load_sample_data(engine):
        logger.error("Failed to load sample data")
        return False
    
    # Verify data
    if not verify_data(engine):
        logger.error("Failed to verify data")
        return False
    
    logger.info("Database initialization completed successfully!")
    return True

if __name__ == "__main__":
    main()
