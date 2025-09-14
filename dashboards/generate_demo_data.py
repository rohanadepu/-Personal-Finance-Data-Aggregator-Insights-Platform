#!/usr/bin/env python3
"""
Demo Data Generator for Personal Finance Dashboard
Creates realistic transaction data for impressive demos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import duckdb
import os

# Configuration
NUM_TRANSACTIONS = 1000
START_DATE = datetime.now() - timedelta(days=90)
END_DATE = datetime.now()

# Merchant categories and their typical spending patterns
MERCHANT_DATA = {
    'grocery': {
        'merchants': ['Whole Foods', 'Safeway', 'Trader Joes', 'Costco', 'Target Grocery'],
        'amount_range': (25, 150),
        'frequency_weight': 0.25,  # 25% of transactions
        'weekend_multiplier': 1.3
    },
    'food': {
        'merchants': ['Starbucks', 'McDonalds', 'Chipotle', 'Subway', 'Local Cafe', 'Pizza Hut'],
        'amount_range': (8, 45),
        'frequency_weight': 0.20,
        'weekend_multiplier': 1.5
    },
    'retail': {
        'merchants': ['Amazon', 'Target', 'Walmart', 'Best Buy', 'Macys'],
        'amount_range': (20, 300),
        'frequency_weight': 0.15,
        'weekend_multiplier': 1.8
    },
    'gas': {
        'merchants': ['Shell', 'Exxon', 'Chevron', 'BP'],
        'amount_range': (35, 85),
        'frequency_weight': 0.12,
        'weekend_multiplier': 0.8
    },
    'entertainment': {
        'merchants': ['Netflix', 'Spotify', 'AMC Theaters', 'Dave & Busters'],
        'amount_range': (15, 120),
        'frequency_weight': 0.10,
        'weekend_multiplier': 2.0
    },
    'utilities': {
        'merchants': ['PG&E', 'Comcast', 'Verizon', 'Water Company'],
        'amount_range': (50, 200),
        'frequency_weight': 0.08,
        'weekend_multiplier': 0.3
    },
    'transport': {
        'merchants': ['Uber', 'Lyft', 'BART', 'Parking Meter'],
        'amount_range': (8, 65),
        'frequency_weight': 0.10,
        'weekend_multiplier': 1.4
    }
}

CITIES = ['San Francisco', 'New York', 'Seattle', 'Austin', 'Boston']
STATES = ['CA', 'NY', 'WA', 'TX', 'MA']
CHANNELS = ['online', 'in_store', 'mobile']

def generate_realistic_transactions():
    """Generate realistic transaction data with patterns"""
    transactions = []
    
    # Calculate total weights for proper distribution
    total_weight = sum(cat['frequency_weight'] for cat in MERCHANT_DATA.values())
    
    for i in range(NUM_TRANSACTIONS):
        # Generate random timestamp with business hour preferences
        random_days = random.randint(0, (END_DATE - START_DATE).days)
        base_date = START_DATE + timedelta(days=random_days)
        
        # Add some business hour bias (more transactions during day)
        if random.random() < 0.7:  # 70% during business hours
            hour = random.randint(9, 18)
        else:
            hour = random.randint(6, 23)
        
        timestamp = base_date.replace(
            hour=hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        # Select category based on weights
        category_choice = random.choices(
            list(MERCHANT_DATA.keys()),
            weights=[cat['frequency_weight'] for cat in MERCHANT_DATA.values()]
        )[0]
        
        category_info = MERCHANT_DATA[category_choice]
        merchant = random.choice(category_info['merchants'])
        
        # Generate amount with weekend/weekday variation
        base_amount = random.uniform(*category_info['amount_range'])
        
        # Weekend spending modifier
        if timestamp.weekday() in [5, 6]:  # Weekend
            base_amount *= category_info['weekend_multiplier']
        
        # Add some random variation
        amount = round(base_amount * random.uniform(0.8, 1.4), 2)
        
        # Ensure minimum amount
        amount = max(amount, 5.0)
        
        # Add occasional large purchases
        if random.random() < 0.05:  # 5% chance of large purchase
            amount *= random.uniform(2.0, 5.0)
        
        transaction = {
            'tx_id': f"tx_{i+1:06d}",
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': 'USD',
            'merchant': merchant,
            'mcc': random.randint(1000, 9999),
            'account_id': f"acc_{random.randint(1, 3):03d}",  # 3 different accounts
            'city': random.choice(CITIES),
            'state': random.choice(STATES),
            'channel': random.choice(CHANNELS),
            'category': category_choice,
            'subcategory': 'general',
            'is_recurring': random.random() < 0.1,  # 10% recurring
            'normalized_amount': amount
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def add_seasonal_patterns(df):
    """Add seasonal spending patterns"""
    df = df.copy()
    
    # Holiday spending boost
    for _, row in df.iterrows():
        date = row['timestamp']
        month = date.month
        
        # Holiday months (November, December)
        if month in [11, 12]:
            if row['category'] in ['retail', 'food', 'entertainment']:
                df.loc[df['tx_id'] == row['tx_id'], 'amount'] *= random.uniform(1.2, 1.8)
        
        # Summer vacation spending
        elif month in [6, 7, 8]:
            if row['category'] in ['entertainment', 'transport', 'food']:
                df.loc[df['tx_id'] == row['tx_id'], 'amount'] *= random.uniform(1.1, 1.4)
    
    return df

def add_anomalies(df):
    """Add some anomalous transactions for anomaly detection demo"""
    df = df.copy()
    
    # Add 5 anomalous transactions
    anomaly_indices = random.sample(range(len(df)), 5)
    
    for idx in anomaly_indices:
        # Create unusually large transaction
        df.loc[idx, 'amount'] *= random.uniform(5.0, 10.0)
        df.loc[idx, 'merchant'] = 'UNUSUAL_MERCHANT_' + str(random.randint(1, 3))
    
    return df

def save_to_databases(df):
    """Save data to both CSV and DuckDB"""
    
    # Save to CSV
    csv_path = '../data/sample_transactions.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} transactions to {csv_path}")
    
    # Save to DuckDB
    try:
        duckdb_path = 'finance_analytics.duckdb'
        conn = duckdb.connect(duckdb_path)
        
        # Drop existing table if it exists
        conn.execute("DROP TABLE IF EXISTS transactions")
        
        # Create table from dataframe
        conn.execute("CREATE TABLE transactions AS SELECT * FROM df")
        
        # Verify data
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        print(f"✅ Saved {count} transactions to DuckDB: {duckdb_path}")
        
        # Show sample
        sample = conn.execute("SELECT merchant, amount, category FROM transactions LIMIT 5").fetchall()
        print("\n📊 Sample transactions:")
        for row in sample:
            print(f"  {row[0]}: ${row[1]:.2f} ({row[2]})")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error saving to DuckDB: {e}")

def generate_summary_stats(df):
    """Generate and display summary statistics"""
    print("\n📈 Dataset Summary:")
    print(f"Total Transactions: {len(df):,}")
    print(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Total Amount: ${df['amount'].sum():,.2f}")
    print(f"Average Transaction: ${df['amount'].mean():.2f}")
    print(f"Median Transaction: ${df['amount'].median():.2f}")
    
    print("\n🏷️ Category Breakdown:")
    category_stats = df.groupby('category').agg({
        'amount': ['sum', 'count', 'mean']
    }).round(2)
    category_stats.columns = ['Total', 'Count', 'Average']
    print(category_stats.sort_values('Total', ascending=False))
    
    print("\n🏪 Top Merchants:")
    merchant_stats = df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
    for merchant, amount in merchant_stats.items():
        print(f"  {merchant}: ${amount:,.2f}")

def main():
    """Main function to generate demo data"""
    print("🎯 Generating Demo Financial Data...")
    print(f"Target: {NUM_TRANSACTIONS} transactions over {(END_DATE - START_DATE).days} days")
    
    # Generate base transactions
    df = generate_realistic_transactions()
    
    # Add patterns and anomalies
    df = add_seasonal_patterns(df)
    df = add_anomalies(df)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Generate summary
    generate_summary_stats(df)
    
    # Save to databases
    save_to_databases(df)
    
    print("\n🎉 Demo data generation complete!")
    print("💡 Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
