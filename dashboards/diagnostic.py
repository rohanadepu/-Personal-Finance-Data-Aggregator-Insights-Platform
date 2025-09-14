#!/usr/bin/env python3
"""
Streamlit Diagnostic Script
"""

import streamlit as st
import pandas as pd
import duckdb
import os
from sqlalchemy import create_engine

st.title("🔍 Database Diagnostic Dashboard")

# Test 1: Sample Data
st.header("Test 1: Sample Data")
try:
    sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_transactions.csv')
    st.write(f"Sample data path: {sample_data_path}")
    st.write(f"Sample data exists: {os.path.exists(sample_data_path)}")
    
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        st.success(f"✅ Sample data loaded: {len(df)} rows")
        st.write("First 3 rows:")
        st.dataframe(df.head(3))
    else:
        st.error("❌ Sample data file not found")
except Exception as e:
    st.error(f"❌ Sample data error: {e}")

# Test 2: DuckDB Connection
st.header("Test 2: DuckDB Connection")
try:
    conn = duckdb.connect(':memory:')
    st.success("✅ DuckDB connection established")
    
    # Test creating table
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        conn.execute("CREATE TABLE transactions AS SELECT * FROM df")
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        st.success(f"✅ DuckDB table created with {count} rows")
        
        # Test querying
        result = conn.execute("SELECT * FROM transactions LIMIT 5").df()
        st.success(f"✅ DuckDB query successful: {len(result)} rows returned")
        st.dataframe(result)
        
    conn.close()
except Exception as e:
    st.error(f"❌ DuckDB error: {e}")

# Test 3: PostgreSQL Connection
st.header("Test 3: PostgreSQL Connection")
try:
    connection_string = "postgresql://airflow:airflow@localhost:5432/airflow"
    engine = create_engine(connection_string)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute("SELECT 1").fetchone()
        st.success("✅ PostgreSQL connection successful")
        
        # Test transactions table
        pg_result = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
        st.success(f"✅ PostgreSQL transactions table has {pg_result[0]} rows")
        
        # Sample data
        pg_sample = conn.execute("SELECT * FROM transactions LIMIT 3").fetchall()
        st.write("PostgreSQL sample data:")
        for row in pg_sample:
            st.write(row)
            
except Exception as e:
    st.error(f"❌ PostgreSQL error: {e}")

# Test 4: Full Data Loading Function
st.header("Test 4: Data Loading Test")

@st.cache_data
def test_load_data():
    try:
        # Try PostgreSQL first
        connection_string = "postgresql://airflow:airflow@localhost:5432/airflow"
        engine = create_engine(connection_string)
        df = pd.read_sql("SELECT * FROM transactions LIMIT 100", engine)
        
        if not df.empty:
            st.success(f"✅ Loaded {len(df)} rows from PostgreSQL")
            return df
            
    except Exception as e:
        st.warning(f"PostgreSQL failed: {e}")
        
    # Fallback to sample data
    try:
        sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_transactions.csv')
        df = pd.read_csv(sample_data_path)
        st.info(f"📁 Using sample data: {len(df)} rows")
        return df.head(100)  # Limit for testing
    except Exception as e:
        st.error(f"Sample data failed: {e}")
        return pd.DataFrame()

test_df = test_load_data()
if not test_df.empty:
    st.success(f"✅ Final data loaded: {len(test_df)} rows")
    st.dataframe(test_df.head())
else:
    st.error("❌ No data could be loaded")
