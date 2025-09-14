import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
import psycopg2
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# Page config
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide"
)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_transactions():
    """Load transactions data with hybrid DuckDB + PostgreSQL strategy"""
    
    # Get DuckDB path from environment or use default
    duckdb_path = os.getenv('DUCKDB_PATH', 'finance_analytics.duckdb')
    
    # Strategy 1: Try DuckDB first (optimized for analytics)
    try:
        duck_conn = duckdb.connect(database=duckdb_path, read_only=True)
        df = duck_conn.execute("SELECT * FROM transactions ORDER BY timestamp DESC").df()
        duck_conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.success(f"✅ Loaded {len(df)} transactions from DuckDB (Analytics)")
            return df
    except Exception as e:
        st.warning(f"DuckDB failed: {e}")
        st.info(f"Attempted DuckDB path: {duckdb_path}")
    
    # Strategy 2: Try PostgreSQL as fallback
    try:
        connection_string = "postgresql://airflow:airflow@localhost:5432/airflow"
        pg_engine = create_engine(connection_string)
        df = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp DESC", pg_engine)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.info(f"📊 Loaded {len(df)} transactions from PostgreSQL (Source)")
            st.warning("Consider running data sync to populate DuckDB for better analytics performance")
            return df
    except Exception as e:
        st.warning(f"PostgreSQL failed: {e}")
    
    # Strategy 3: Try sample data
    try:
        sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_transactions.csv')
        if os.path.exists(sample_data_path):
            df = pd.read_csv(sample_data_path)
            
            # Add missing columns for the sample data
            df['city'] = 'Sample City'
            df['state'] = 'CA'
            df['channel'] = 'online'
            df['category'] = df.get('merchant', 'Unknown').map({
                'Walmart': 'retail',
                'Costco': 'retail', 
                'Amazon': 'retail',
                'Target': 'retail',
                'Whole Foods': 'grocery',
                'Starbucks': 'food',
                'McDonalds': 'food',
                'Shell': 'gas',
                'Exxon': 'gas'
            }).fillna('other')
            df['subcategory'] = 'general'
            df['is_recurring'] = False
            df['normalized_amount'] = df['amount']
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.info(f"📁 Loaded {len(df)} transactions from sample data")
            return df
    except Exception as e:
        st.error(f"Sample data failed: {e}")
    
    # Strategy 3: Create empty dataframe
    st.warning("No data available - creating empty dataset")
    return pd.DataFrame(columns=[
        'tx_id', 'timestamp', 'amount', 'currency', 'merchant', 
        'mcc', 'account_id', 'city', 'state', 'channel', 
        'category', 'subcategory', 'is_recurring', 'normalized_amount'
    ])

def main():
    st.title("Personal Finance Dashboard 💰")
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Data sync option in sidebar
    if st.sidebar.button("🔄 Sync Data (PostgreSQL → DuckDB)"):
        with st.spinner("Syncing data..."):
            try:
                from data_sync import DataSyncManager
                sync_manager = DataSyncManager()
                if sync_manager.sync_to_duckdb():
                    st.sidebar.success("✅ Data synced successfully!")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("❌ Data sync failed")
            except Exception as e:
                st.sidebar.error(f"❌ Sync error: {e}")
    
    # Load data
    try:
        df = load_transactions()
        
        if df.empty:
            st.warning("No transaction data found. Please run the data pipeline or check your database connection.")
            st.info("The system will try to:")
            st.info("1. Connect to PostgreSQL database for source data")
            st.info("2. Fall back to sample data if PostgreSQL is not available")
            st.info("3. Load data into DuckDB for fast analytics")
            
            # Show sample data structure
            st.subheader("Expected Data Structure")
            sample_columns = ['tx_id', 'timestamp', 'amount', 'currency', 'merchant', 
                            'mcc', 'account_id', 'city', 'state', 'channel', 
                            'category', 'subcategory', 'is_recurring', 'normalized_amount']
            st.write("Expected columns:", sample_columns)
            return
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Date filter
        if len(df) > 0:
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
        else:
            min_date = datetime.now().date()
            max_date = datetime.now().date()
            
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply filters
        if len(date_range) == 2:
            mask = (df['timestamp'].dt.date >= date_range[0]) & \
                   (df['timestamp'].dt.date <= date_range[1])
            filtered_df = df.loc[mask]
        else:
            filtered_df = df
        
        # Show data info
        st.sidebar.metric("Total Transactions", len(filtered_df))
        if len(filtered_df) > 0:
            st.sidebar.metric("Total Amount", f"${filtered_df['amount'].sum():,.2f}")
        
        # Dashboard layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Spend by Category")
            if len(filtered_df) > 0:
                monthly_spend = filtered_df.groupby(
                    [pd.Grouper(key='timestamp', freq='M'), 'category']
                )['amount'].sum().reset_index()
                
                if len(monthly_spend) > 0:
                    fig = px.bar(
                        monthly_spend,
                        x='timestamp',
                        y='amount',
                        color='category',
                        title="Monthly Spend by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for the selected date range")
            else:
                st.info("No transactions found")
        
        with col2:
            st.subheader("Top Merchants")
            if len(filtered_df) > 0:
                top_merchants = filtered_df.groupby('merchant')['amount'].sum()\
                    .sort_values(ascending=False).head(10)
                
                if len(top_merchants) > 0:
                    fig = px.pie(
                        values=top_merchants.values,
                        names=top_merchants.index,
                        title="Top 10 Merchants by Spend"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No merchant data available")
            else:
                st.info("No transactions found")
        
        # Transaction list
        st.subheader("Recent Transactions")
        if len(filtered_df) > 0:
            display_columns = ['timestamp', 'merchant', 'amount', 'category']
            # Only show columns that exist in the dataframe
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df.sort_values('timestamp', ascending=False)\
                    .head(10)[available_columns]
            )
        else:
            st.info("No transactions to display")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("This might be because:")
        st.info("1. PostgreSQL database is not running")
        st.info("2. The transactions table doesn't exist")
        st.info("3. Sample data file is not found")
        
        # Show debugging info
        with st.expander("Debug Information"):
            st.write("Error details:", str(e))
            st.write("Current working directory:", os.getcwd())
            sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_transactions.csv')
            st.write("Sample data path:", sample_path)
            st.write("Sample data exists:", os.path.exists(sample_path))

if __name__ == "__main__":
    main()
