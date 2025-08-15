import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide"
)

# Database connection (replace with your actual connection)
def get_db_connection():
    import psycopg2
    return psycopg2.connect(
        dbname="finance_db",
        user="finance_user",
        password="finance_password",
        host="postgres"
    )

# Load data functions
@st.cache_data(ttl=3600)
def load_monthly_spending():
    conn = get_db_connection()
    query = """
    SELECT date_trunc('month', timestamp) as month,
           category,
           sum(amount) as total_spend
    FROM transactions_curated
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return pd.read_sql(query, conn)

@st.cache_data(ttl=3600)
def load_anomalies():
    conn = get_db_connection()
    query = """
    SELECT *
    FROM transaction_anomalies
    ORDER BY timestamp DESC
    LIMIT 10
    """
    return pd.read_sql(query, conn)

@st.cache_data(ttl=3600)
def load_cashflow_forecast():
    conn = get_db_connection()
    query = """
    SELECT date_trunc('month', date) as month,
           predicted_cashflow,
           lower_bound,
           upper_bound
    FROM cashflow_forecasts
    ORDER BY month
    """
    return pd.read_sql(query, conn)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Spending Analysis", "Anomalies", "Forecasts"])

# Overview Page
if page == "Overview":
    st.title("Financial Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Monthly spending trend
    spending_data = load_monthly_spending()
    fig_spending = px.line(
        spending_data,
        x="month",
        y="total_spend",
        color="category",
        title="Monthly Spending by Category"
    )
    st.plotly_chart(fig_spending)
    
    # Key metrics
    with col1:
        st.metric(
            label="This Month's Spending",
            value="$2,345",
            delta="-15%"
        )
    
    with col2:
        st.metric(
            label="Savings Rate",
            value="25%",
            delta="5%"
        )
    
    with col3:
        st.metric(
            label="Investment Returns",
            value="8.5%",
            delta="2.3%"
        )

# Spending Analysis Page
elif page == "Spending Analysis":
    st.title("Spending Analysis")
    
    # Category breakdown
    spending_by_category = px.pie(
        spending_data,
        values="total_spend",
        names="category",
        title="Spending by Category"
    )
    st.plotly_chart(spending_by_category)
    
    # Monthly trend
    st.subheader("Monthly Trend by Category")
    category = st.selectbox("Select Category", spending_data["category"].unique())
    
    category_trend = px.line(
        spending_data[spending_data["category"] == category],
        x="month",
        y="total_spend",
        title=f"{category} - Monthly Trend"
    )
    st.plotly_chart(category_trend)

# Anomalies Page
elif page == "Anomalies":
    st.title("Spending Anomalies")
    
    anomalies = load_anomalies()
    
    for _, anomaly in anomalies.iterrows():
        with st.expander(f"{anomaly['merchant']} - ${anomaly['amount']:,.2f}"):
            st.write(f"Date: {anomaly['timestamp']}")
            st.write(f"Category: {anomaly['category']}")
            st.write(f"Reason: {anomaly['anomaly_reason']}")
            st.write(f"Confidence: {anomaly['anomaly_score']:.2%}")

# Forecasts Page
elif page == "Forecasts":
    st.title("Financial Forecasts")
    
    forecast_data = load_cashflow_forecast()
    
    # Cashflow forecast
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_data["month"],
            y=forecast_data["predicted_cashflow"],
            name="Forecast",
            line=dict(color="blue")
        )
    )
    
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_data["month"],
            y=forecast_data["upper_bound"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_data["month"],
            y=forecast_data["lower_bound"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="Confidence Interval"
        )
    )
    
    fig_forecast.update_layout(
        title="Cashflow Forecast",
        xaxis_title="Month",
        yaxis_title="Amount ($)"
    )
    
    st.plotly_chart(fig_forecast)
    
    # Forecast details
    st.subheader("Forecast Details")
    st.dataframe(forecast_data)
