#!/usr/bin/env python3
"""
Data Sync Utility for Personal Finance Dashboard
This script manages the hybrid PostgreSQL + DuckDB approach
"""

import pandas as pd
import psycopg2
import duckdb
from sqlalchemy import create_engine, text
import logging
import argparse
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSyncManager:
    def __init__(self):
        self.pg_engine = self._get_postgres_connection()
        # Use environment variable for DuckDB path, fallback to local path
        duckdb_path = os.getenv('DUCKDB_PATH', 'finance_analytics.duckdb')
        self.duck_conn = duckdb.connect(database=duckdb_path, read_only=False)
        
    def _get_postgres_connection(self):
        """Create PostgreSQL connection"""
        try:
            connection_string = "postgresql://airflow:airflow@localhost:5432/airflow"
            return create_engine(connection_string)
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return None
    
    def sync_to_duckdb(self):
        """Sync data from PostgreSQL to DuckDB for analytics"""
        if not self.pg_engine:
            logger.error("PostgreSQL connection not available")
            return False
            
        try:
            # Read data from PostgreSQL
            df = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp", self.pg_engine)
            logger.info(f"Read {len(df)} transactions from PostgreSQL")
            
            # Create/replace table in DuckDB
            self.duck_conn.execute("DROP TABLE IF EXISTS transactions")
            self.duck_conn.execute("""
                CREATE TABLE transactions AS 
                SELECT * FROM df
            """)
            
            # Create indexes for better performance
            self.duck_conn.execute("CREATE INDEX idx_timestamp ON transactions(timestamp)")
            self.duck_conn.execute("CREATE INDEX idx_merchant ON transactions(merchant)")
            self.duck_conn.execute("CREATE INDEX idx_category ON transactions(category)")
            
            logger.info(f"Successfully synced {len(df)} transactions to DuckDB")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing data: {e}")
            return False
    
    def validate_sync(self):
        """Validate that data is in sync between PostgreSQL and DuckDB"""
        try:
            # Count in PostgreSQL
            with self.pg_engine.connect() as conn:
                pg_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
            
            # Count in DuckDB
            duck_count = self.duck_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            
            logger.info(f"PostgreSQL: {pg_count} records")
            logger.info(f"DuckDB: {duck_count} records")
            
            if pg_count == duck_count:
                logger.info("✅ Data is in sync!")
                return True
            else:
                logger.warning("⚠️ Data is NOT in sync!")
                return False
                
        except Exception as e:
            logger.error(f"Error validating sync: {e}")
            return False
    
    def run_analytics_query(self, query):
        """Run analytics query on DuckDB"""
        try:
            result = self.duck_conn.execute(query).df()
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error running query: {e}")
            return None
    
    def get_summary_stats(self):
        """Get summary statistics from DuckDB"""
        queries = {
            "total_transactions": "SELECT COUNT(*) as count FROM transactions",
            "total_amount": "SELECT SUM(amount) as total FROM transactions",
            "date_range": "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM transactions",
            "top_categories": """
                SELECT category, COUNT(*) as transactions, SUM(amount) as total_amount 
                FROM transactions 
                GROUP BY category 
                ORDER BY total_amount DESC 
                LIMIT 5
            """,
            "monthly_trend": """
                SELECT 
                    strftime('%Y-%m', timestamp) as month,
                    COUNT(*) as transactions,
                    SUM(amount) as total_amount
                FROM transactions 
                GROUP BY month 
                ORDER BY month
            """
        }
        
        results = {}
        for name, query in queries.items():
            try:
                results[name] = self.duck_conn.execute(query).df()
                logger.info(f"✅ {name}: Success")
            except Exception as e:
                logger.error(f"❌ {name}: {e}")
                results[name] = None
        
        return results
    
    def close_connections(self):
        """Close database connections"""
        if self.duck_conn:
            self.duck_conn.close()
        if self.pg_engine:
            self.pg_engine.dispose()

def main():
    parser = argparse.ArgumentParser(description='Data Sync Utility for Personal Finance Dashboard')
    parser.add_argument('--sync', action='store_true', help='Sync data from PostgreSQL to DuckDB')
    parser.add_argument('--validate', action='store_true', help='Validate data sync')
    parser.add_argument('--stats', action='store_true', help='Show summary statistics')
    parser.add_argument('--query', type=str, help='Run custom analytics query')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # Default action: sync and validate
        args.sync = True
        args.validate = True
        args.stats = True
    
    sync_manager = DataSyncManager()
    
    try:
        if args.sync:
            logger.info("🔄 Syncing data from PostgreSQL to DuckDB...")
            sync_manager.sync_to_duckdb()
        
        if args.validate:
            logger.info("🔍 Validating data sync...")
            sync_manager.validate_sync()
        
        if args.stats:
            logger.info("📊 Generating summary statistics...")
            stats = sync_manager.get_summary_stats()
            
            print("\n=== SUMMARY STATISTICS ===")
            for name, df in stats.items():
                if df is not None:
                    print(f"\n{name.upper()}:")
                    print(df.to_string(index=False))
        
        if args.query:
            logger.info(f"🔍 Running custom query: {args.query}")
            result = sync_manager.run_analytics_query(args.query)
            if result is not None:
                print("\n=== QUERY RESULTS ===")
                print(result.to_string(index=False))
    
    finally:
        sync_manager.close_connections()

if __name__ == "__main__":
    main()
