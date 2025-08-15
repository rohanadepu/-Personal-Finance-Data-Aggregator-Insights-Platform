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

# Mock data functions
@st.cache_data(ttl=3600)
def load_monthly_spending():
    # Generate 12 months of mock data for different categories
    categories = ['Housing', 'Food', 'Transportation', 'Entertainment', 'Utilities']
    data = []
    
    for month in range(12):
        date = datetime.now() - timedelta(days=30*month)
        for category in categories:
            # Base amount plus some random variation
            base_amounts = {
                'Housing': 1500,
                'Food': 600,
                'Transportation': 400,
                'Entertainment': 300,
                'Utilities': 200
            }
            amount = base_amounts[category] * (1 + np.random.normal(0, 0.1))
            data.append({
                'month': date.replace(day=1),
                'category': category,
                'total_spend': amount
            })
    
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_anomalies():
    # Generate mock anomalies
    anomalies = [
        {
            'timestamp': datetime.now() - timedelta(days=2),
            'merchant': 'Amazon.com',
            'amount': 599.99,
            'category': 'Shopping',
            'anomaly_reason': 'Unusually large transaction',
            'anomaly_score': 0.92
        },
        {
            'timestamp': datetime.now() - timedelta(days=5),
            'merchant': 'Unknown Merchant',
            'amount': 299.99,
            'category': 'Entertainment',
            'anomaly_reason': 'Unusual merchant',
            'anomaly_score': 0.85
        },
        {
            'timestamp': datetime.now() - timedelta(days=7),
            'merchant': 'Gas Station',
            'amount': 150.00,
            'category': 'Transportation',
            'anomaly_reason': 'Multiple transactions same day',
            'anomaly_score': 0.78
        }
    ]
    return pd.DataFrame(anomalies)

@st.cache_data(ttl=3600)
def load_cashflow_forecast():
    # Generate 6 months of forecast data
    data = []
    base_cashflow = 5000
    
    for month in range(6):
        date = datetime.now() + timedelta(days=30*month)
        predicted = base_cashflow * (1 + month * 0.02)  # 2% growth per month
        data.append({
            'month': date.replace(day=1),
            'predicted_cashflow': predicted,
            'lower_bound': predicted * 0.9,
            'upper_bound': predicted * 1.1
        })
    
    return pd.DataFrame(data)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Spending Analysis", "Anomalies", "Forecasts"])

# Overview Page
if page == "Overview":
    st.title("Financial Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Monthly spending trend
    monthly_spending = load_monthly_spending()
    fig_spending = px.line(
        monthly_spending,
        x="month",
        y="total_spend",
        color="category",
        title="Monthly Spending by Category"
    )
    st.plotly_chart(fig_spending)
    
    # Calculate current month metrics
    current_month = monthly_spending[monthly_spending['month'] == monthly_spending['month'].max()]
    total_spending = current_month['total_spend'].sum()
    prev_month = monthly_spending[monthly_spending['month'] == monthly_spending['month'].unique()[-2]]
    prev_total = prev_month['total_spend'].sum()
    spending_change = (total_spending - prev_total) / prev_total
    
    # Key metrics
    with col1:
        st.metric(
            label="This Month's Spending",
            value=f"${total_spending:,.0f}",
            delta=f"{spending_change:.1%}"
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
    
    monthly_spending = load_monthly_spending()
    
    # Category breakdown
    current_month = monthly_spending[monthly_spending['month'] == monthly_spending['month'].max()]
    spending_by_category = px.pie(
        current_month,
        values="total_spend",
        names="category",
        title="Current Month Spending by Category"
    )
    st.plotly_chart(spending_by_category)
    
    # Monthly trend
    st.subheader("Monthly Trend by Category")
    category = st.selectbox("Select Category", monthly_spending["category"].unique())
    
    category_trend = px.line(
        monthly_spending[monthly_spending["category"] == category],
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
