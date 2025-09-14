#!/usr/bin/env python3
"""
Populate PostgreSQL with demo transaction data
Works with both Docker and local PostgreSQL instances
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from sqlalchemy import create_engine, text
import psycopg2

def generate_realistic_demo_data(num_transactions=500):
    """Generate realistic demo transaction data"""
    
    # Realistic merchant categories
    merchants = {
        'grocery': ['Whole Foods', 'Trader Joes', 'Kroger', 'Safeway', 'Walmart Grocery'],
        'retail': ['Amazon', 'Target', 'Walmart', 'Costco', 'Best Buy', 'Apple Store'],
        'food': ['Starbucks', 'McDonalds', 'Chipotle', 'Subway', 'Pizza Hut', 'KFC'],
        'gas': ['Shell', 'Exxon', 'Chevron', 'BP', 'Mobil', '76'],
        'entertainment': ['Netflix', 'Spotify', 'AMC Theaters', 'Dave & Busters'],
        'healthcare': ['CVS Pharmacy', 'Walgreens', 'Kaiser Permanente'],
        'transportation': ['Uber', 'Lyft', 'Metro Transit'],
        'utilities': ['PG&E', 'Comcast', 'Verizon', 'AT&T'],
        'fitness': ['24 Hour Fitness', 'Planet Fitness'],
        'other': ['Home Depot', 'Lowes', 'Bank Fee']
    }
    
    # Amount ranges by category
    amount_ranges = {
        'grocery': (15, 200),
        'retail': (25, 500), 
        'food': (8, 60),
        'gas': (30, 80),
        'entertainment': (12, 100),
        'healthcare': (20, 300),
        'transportation': (8, 45),
        'utilities': (50, 250),
        'fitness': (30, 150),
        'other': (10, 400)
    }
    
    # Cities and states
    locations = [
        ('San Francisco', 'CA'), ('New York', 'NY'), ('Seattle', 'WA'),
        ('Austin', 'TX'), ('Chicago', 'IL'), ('Boston', 'MA'),
        ('Los Angeles', 'CA'), ('Denver', 'CO'), ('Miami', 'FL')
    ]
    
    transactions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    for i in range(num_transactions):
        # Random category
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        
        # Random timestamp with realistic patterns
        days_back = random.randint(0, 90)
        base_date = start_date + timedelta(days=days_back)
        
        # Time patterns (more activity during business hours)
        if random.random() < 0.7:  # 70% during business hours
            hour = random.randint(9, 21)
        else:
            hour = random.randint(0, 23)
        
        timestamp = base_date.replace(
            hour=hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        # Amount based on category
        min_amt, max_amt = amount_ranges[category]
        amount = round(random.uniform(min_amt, max_amt), 2)
        
        # Weekend premium
        if timestamp.weekday() >= 5:  # Weekend
            amount *= random.uniform(1.1, 1.3)
        
        # Location
        city, state = random.choice(locations)
        
        transaction = {
            'tx_id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': 'USD',
            'merchant': merchant,
            'mcc': random.randint(5000, 5999),
            'account_id': f'acc_{random.randint(1000, 9999)}',
            'city': city,
            'state': state,
            'channel': random.choice(['online', 'in-store', 'mobile']),
            'category': category,
            'subcategory': 'general',
            'is_recurring': random.choice([True, False]) if category in ['utilities', 'entertainment'] else False,
            'normalized_amount': round(amount, 2)
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def create_transactions_table(engine):
    """Create transactions table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS transactions (
        tx_id VARCHAR(255) PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'USD',
        merchant VARCHAR(255) NOT NULL,
        mcc INTEGER,
        account_id VARCHAR(255),
        city VARCHAR(255),
        state VARCHAR(10),
        channel VARCHAR(50),
        category VARCHAR(100),
        subcategory VARCHAR(100),
        is_recurring BOOLEAN DEFAULT FALSE,
        normalized_amount DECIMAL(10,2)
    );
    
    CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
    CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant);
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()

def populate_database():
    """Populate PostgreSQL with demo data"""
    
    # Connection strings for Docker and local
    connection_strings = [
        "postgresql://airflow:airflow@postgres:5432/airflow",  # Docker
        "postgresql://airflow:airflow@localhost:5432/airflow"  # Local
    ]
    
    print("🎭 Generating demo transaction data...")
    df = generate_realistic_demo_data(500)
    
    print(f"📊 Generated {len(df)} transactions")
    print(f"💰 Total amount: ${df['amount'].sum():,.2f}")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    success = False
    
    for conn_str in connection_strings:
        try:
            print(f"\n🔌 Connecting to {conn_str}...")
            engine = create_engine(conn_str)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print("✅ Connection successful!")
            
            # Create table
            print("📋 Creating transactions table...")
            create_transactions_table(engine)
            
            # Clear existing data
            with engine.connect() as conn:
                result = conn.execute(text("DELETE FROM transactions"))
                conn.commit()
                print(f"🗑️ Cleared existing transactions")
            
            # Insert new data
            print("📥 Inserting demo data...")
            df.to_sql('transactions', engine, if_exists='append', index=False, method='multi')
            
            # Verify insertion
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM transactions")).fetchone()
                count = result[0]
                print(f"✅ Successfully inserted {count} transactions!")
            
            success = True
            break
            
        except Exception as e:
            print(f"❌ Failed to connect to {conn_str}: {e}")
            continue
    
    if not success:
        print("\n❌ Could not connect to any PostgreSQL instance")
        print("💡 Make sure PostgreSQL is running with the correct credentials")
        print("   Docker: docker compose up -d")
        print("   Local: Check your PostgreSQL service")
    else:
        print("\n🎉 Demo data population completed successfully!")
        print("🚀 You can now run the Streamlit dashboard")

if __name__ == "__main__":
    populate_database()
