import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
import os
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_db_connection():
    return duckdb.connect(database=':memory:', read_only=False)

# Data loading functions
@st.cache_data
def load_transactions():
    conn = get_db_connection()
    return pd.read_sql("SELECT * FROM transactions", conn)

def main():
    st.title("Personal Finance Dashboard 💰")
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Load data
    try:
        df = load_transactions()
        
        # Date filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(
                df['timestamp'].min(),
                df['timestamp'].max()
            )
        )
        
        # Apply filters
        mask = (df['timestamp'].dt.date >= date_range[0]) & \
               (df['timestamp'].dt.date <= date_range[1])
        filtered_df = df.loc[mask]
        
        # Dashboard layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Spend by Category")
            monthly_spend = filtered_df.groupby(
                [pd.Grouper(key='timestamp', freq='M'), 'category']
            )['amount'].sum().reset_index()
            
            fig = px.bar(
                monthly_spend,
                x='timestamp',
                y='amount',
                color='category',
                title="Monthly Spend by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Merchants")
            top_merchants = filtered_df.groupby('merchant')['amount'].sum()\
                .sort_values(ascending=False).head(10)
            
            fig = px.pie(
                values=top_merchants.values,
                names=top_merchants.index,
                title="Top 10 Merchants by Spend"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction list
        st.subheader("Recent Transactions")
        st.dataframe(
            filtered_df.sort_values('timestamp', ascending=False)\
                .head(10)[['timestamp', 'merchant', 'amount', 'category']]
        )
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
