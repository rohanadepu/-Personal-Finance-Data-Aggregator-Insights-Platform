"""
ML Model Dashboard for Personal Finance Platform
Streamlit app for monitoring anomaly detection and forecasting models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import requests

st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="📊",
    layout="wide"
)

# App configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Adjust for your setup

@st.cache_data
def load_sample_data():
    """Load sample transaction data"""
    try:
        df = pd.read_csv("../data/sample_transactions.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        return df
    except:
        # Create synthetic data if file not found
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = []
        for date in dates:
            n_transactions = np.random.poisson(3)
            for _ in range(n_transactions):
                data.append({
                    'timestamp': date + timedelta(hours=np.random.randint(0, 24)),
                    'amount': np.random.normal(-100, 50),
                    'merchant': np.random.choice(['Walmart', 'Target', 'Amazon', 'Costco']),
                    'category': np.random.choice(['grocery', 'transport', 'restaurant', 'other'])
                })
        df = pd.DataFrame(data)
        df['date'] = df['timestamp'].dt.date
        return df

def create_spending_overview(df):
    """Create spending overview visualizations"""
    st.subheader("📈 Spending Overview")
    
    # Daily spending
    daily_spending = df.groupby('date')['amount'].sum().reset_index()
    daily_spending['amount'] = -daily_spending['amount']  # Make expenses positive for display
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_daily = px.line(daily_spending, x='date', y='amount', 
                           title='Daily Spending Trend')
        fig_daily.update_traces(line_color='#FF6B6B')
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        category_spending = df.groupby('category')['amount'].sum().abs()
        fig_cat = px.pie(values=category_spending.values, names=category_spending.index,
                        title='Spending by Category')
        st.plotly_chart(fig_cat, use_container_width=True)

def create_anomaly_detection_dashboard(df):
    """Create anomaly detection monitoring dashboard"""
    st.subheader("🚨 Anomaly Detection")
    
    # Simulate anomaly detection results
    np.random.seed(42)
    n_days = len(df['date'].unique())
    
    # Create synthetic anomaly scores
    anomaly_scores = np.random.beta(2, 8, n_days)  # Most scores low, few high
    anomaly_threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.8, 0.05)
    
    daily_data = df.groupby('date').agg({
        'amount': ['sum', 'count'],
        'merchant': 'nunique'
    }).reset_index()
    daily_data.columns = ['date', 'total_amount', 'transaction_count', 'unique_merchants']
    daily_data['anomaly_score'] = anomaly_scores
    daily_data['is_anomaly'] = daily_data['anomaly_score'] > anomaly_threshold
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anomaly_count = daily_data['is_anomaly'].sum()
        st.metric("Anomalies Detected", anomaly_count)
    
    with col2:
        anomaly_rate = (anomaly_count / len(daily_data)) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    with col3:
        avg_score = daily_data['anomaly_score'].mean()
        st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
    
    # Anomaly timeline
    fig_anomaly = go.Figure()
    
    # Add normal points
    normal_points = daily_data[~daily_data['is_anomaly']]
    fig_anomaly.add_trace(go.Scatter(
        x=normal_points['date'],
        y=normal_points['anomaly_score'],
        mode='markers',
        name='Normal',
        marker=dict(color='green', size=6)
    ))
    
    # Add anomalous points
    anomaly_points = daily_data[daily_data['is_anomaly']]
    fig_anomaly.add_trace(go.Scatter(
        x=anomaly_points['date'],
        y=anomaly_points['anomaly_score'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10, symbol='diamond')
    ))
    
    # Add threshold line
    fig_anomaly.add_hline(y=anomaly_threshold, line_dash="dash", 
                         line_color="orange", annotation_text="Threshold")
    
    fig_anomaly.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Date",
        yaxis_title="Anomaly Score"
    )
    
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Show anomalous days
    if anomaly_count > 0:
        st.subheader("🔍 Anomalous Days Details")
        anomalous_days = daily_data[daily_data['is_anomaly']].copy()
        anomalous_days = anomalous_days.sort_values('anomaly_score', ascending=False)
        st.dataframe(anomalous_days[['date', 'total_amount', 'transaction_count', 
                                   'unique_merchants', 'anomaly_score']])

def create_forecasting_dashboard(df):
    """Create forecasting dashboard"""
    st.subheader("🔮 Cash Flow Forecasting")
    
    # Prepare historical data
    daily_flow = df.groupby('date')['amount'].sum().reset_index()
    daily_flow = daily_flow.sort_values('date')
    
    # Simulate forecasting results
    forecast_days = st.slider("Forecast Days", 7, 90, 30)
    
    # Simple trend-based forecast for demonstration
    last_30_days = daily_flow.tail(30)['amount'].values
    trend = np.polyfit(range(30), last_30_days, 1)[0]
    
    forecast_dates = pd.date_range(
        start=daily_flow['date'].max() + timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )
    
    # Generate forecast with trend and noise
    base_forecast = last_30_days[-1] + trend * np.arange(1, forecast_days + 1)
    noise = np.random.normal(0, np.std(last_30_days) * 0.3, forecast_days)
    forecast_values = base_forecast + noise
    
    # Calculate confidence intervals
    std_dev = np.std(last_30_days)
    upper_bound = forecast_values + 1.96 * std_dev
    lower_bound = forecast_values - 1.96 * std_dev
    
    # Create forecast visualization
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=daily_flow['date'],
        y=daily_flow['amount'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Confidence intervals
    fig_forecast.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    fig_forecast.update_layout(
        title=f"Cash Flow Forecast - Next {forecast_days} Days",
        xaxis_title="Date",
        yaxis_title="Daily Cash Flow"
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_forecast = forecast_values.sum()
        st.metric("Forecasted Total", f"${total_forecast:,.2f}")
    
    with col2:
        avg_daily = forecast_values.mean()
        st.metric("Avg Daily Flow", f"${avg_daily:.2f}")
    
    with col3:
        trend_direction = "↗️" if trend > 0 else "↘️" if trend < 0 else "➡️"
        st.metric("Trend", f"{trend_direction} ${trend:.2f}/day")

def create_model_performance_dashboard():
    """Create model performance monitoring dashboard"""
    st.subheader("🎯 Model Performance")
    
    # Simulate model performance metrics
    models = ['Isolation Forest', 'Autoencoder', 'Prophet', 'ARIMA', 'LSTM']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Anomaly Detection Models**")
        
        anomaly_metrics = {
            'Model': ['Isolation Forest', 'Autoencoder'],
            'Precision': [0.85, 0.78],
            'Recall': [0.72, 0.81],
            'F1-Score': [0.78, 0.79],
            'Last Updated': ['2024-01-15', '2024-01-15']
        }
        
        st.dataframe(pd.DataFrame(anomaly_metrics))
    
    with col2:
        st.write("**Forecasting Models**")
        
        forecast_metrics = {
            'Model': ['Prophet', 'ARIMA', 'LSTM'],
            'MAPE': [12.5, 15.2, 11.8],
            'RMSE': [45.2, 52.1, 43.7],
            'MAE': [38.9, 44.3, 37.2],
            'Last Updated': ['2024-01-15', '2024-01-15', '2024-01-15']
        }
        
        st.dataframe(pd.DataFrame(forecast_metrics))
    
    # Model training history
    st.write("**Model Training History**")
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    training_history = pd.DataFrame({
        'Date': dates,
        'Models Trained': np.random.poisson(2, len(dates)),
        'Avg Training Time (min)': np.random.normal(15, 3, len(dates)),
        'Success Rate': np.random.uniform(0.9, 1.0, len(dates))
    })
    
    fig_training = px.bar(training_history, x='Date', y='Models Trained',
                         title='Daily Model Training Activity')
    st.plotly_chart(fig_training, use_container_width=True)

def main():
    """Main dashboard application"""
    st.title("🏦 Personal Finance ML Dashboard")
    st.write("Monitor anomaly detection and forecasting models for financial data")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Overview",
        "Anomaly Detection", 
        "Forecasting",
        "Model Performance"
    ])
    
    # Load data
    df = load_sample_data()
    
    if page == "Overview":
        create_spending_overview(df)
        
    elif page == "Anomaly Detection":
        create_anomaly_detection_dashboard(df)
        
    elif page == "Forecasting":
        create_forecasting_dashboard(df)
        
    elif page == "Model Performance":
        create_model_performance_dashboard()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tips:**\n"
        "- Adjust thresholds to fine-tune anomaly detection\n"
        "- Monitor model performance regularly\n"
        "- Check forecasts for planning purposes"
    )

if __name__ == "__main__":
    main()
