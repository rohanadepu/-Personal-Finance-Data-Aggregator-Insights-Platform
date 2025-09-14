#!/usr/bin/env python3
"""
Database Status Checker for Personal Finance Dashboard
"""

import streamlit as st
import duckdb
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime

st.set_page_config(
    page_title="Database Status",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Database Status Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("DuckDB Status (Analytics)")
    try:
        duck_conn = duckdb.connect(database='finance_analytics.duckdb', read_only=True)
        
        # Check tables
        tables = duck_conn.execute("SHOW TABLES").fetchall()
        st.success(f"✅ DuckDB Connected")
        st.write(f"Tables: {[t[0] for t in tables]}")
        
        if tables:
            # Check transactions table
            count = duck_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            st.metric("Total Transactions", count)
            
            # Date range
            date_range = duck_conn.execute("SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM transactions").fetchone()
            st.write(f"Date Range: {date_range[0]} to {date_range[1]}")
            
            # Sample data
            sample = duck_conn.execute("SELECT * FROM transactions LIMIT 3").df()
            st.subheader("Sample Data")
            st.dataframe(sample)
            
        duck_conn.close()
        
    except Exception as e:
        st.error(f"❌ DuckDB Error: {e}")

with col2:
    st.header("PostgreSQL Status (Source)")
    try:
        connection_string = "postgresql://airflow:airflow@localhost:5432/airflow"
        pg_engine = create_engine(connection_string)
        
        with pg_engine.connect() as conn:
            # Check if transactions table exists
            result = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
            st.success(f"✅ PostgreSQL Connected")
            st.metric("Total Transactions", result)
            
            # Date range
            date_range = conn.execute(text("SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM transactions")).fetchone()
            st.write(f"Date Range: {date_range[0]} to {date_range[1]}")
            
    except Exception as e:
        st.error(f"❌ PostgreSQL Error: {e}")

# Data sync status
st.header("Data Sync Status")
try:
    duck_conn = duckdb.connect(database='finance_analytics.duckdb', read_only=True)
    pg_engine = create_engine("postgresql://airflow:airflow@localhost:5432/airflow")
    
    duck_count = duck_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    with pg_engine.connect() as conn:
        pg_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
    
    if duck_count == pg_count:
        st.success(f"✅ Data in sync! {duck_count} records in both databases")
    else:
        st.warning(f"⚠️ Data out of sync! PostgreSQL: {pg_count}, DuckDB: {duck_count}")
        if st.button("🔄 Sync Data Now"):
            with st.spinner("Syncing data..."):
                try:
                    from data_sync import DataSyncManager
                    sync_manager = DataSyncManager()
                    if sync_manager.sync_to_duckdb():
                        st.success("✅ Data synced successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Data sync failed")
                except Exception as sync_e:
                    st.error(f"❌ Sync error: {sync_e}")
    
    duck_conn.close()
    
except Exception as e:
    st.error(f"❌ Sync check error: {e}")

# Performance comparison
st.header("Performance Comparison")
col1, col2 = st.columns(2)

with col1:
    st.subheader("DuckDB Query Performance")
    if st.button("Test DuckDB Query"):
        start_time = datetime.now()
        try:
            duck_conn = duckdb.connect(database='finance_analytics.duckdb', read_only=True)
            df = duck_conn.execute("SELECT category, COUNT(*) as count, SUM(amount) as total FROM transactions GROUP BY category ORDER BY count DESC").df()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            st.success(f"✅ Query completed in {duration:.3f} seconds")
            st.dataframe(df)
            duck_conn.close()
        except Exception as e:
            st.error(f"❌ Query failed: {e}")

with col2:
    st.subheader("PostgreSQL Query Performance")
    if st.button("Test PostgreSQL Query"):
        start_time = datetime.now()
        try:
            pg_engine = create_engine("postgresql://airflow:airflow@localhost:5432/airflow")
            df = pd.read_sql("SELECT category, COUNT(*) as count, SUM(amount) as total FROM transactions GROUP BY category ORDER BY count DESC", pg_engine)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            st.success(f"✅ Query completed in {duration:.3f} seconds")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
