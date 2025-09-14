"""
ML Dashboard Integration for Personal Finance Platform
Connects with existing ML pipeline and MLflow for real-time insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ml.anomaly_detection import AnomalyDetector
    from ml.forecasting import SpendingForecaster
    from ml.ml_pipeline import MLPipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ ML modules not available. Some features will be limited.")

class MLDashboardIntegration:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector() if ML_AVAILABLE else None
        self.forecaster = SpendingForecaster() if ML_AVAILABLE else None
        self.ml_pipeline = MLPipeline() if ML_AVAILABLE else None
    
    def detect_anomalies_advanced(self, df):
        """Advanced anomaly detection using ML pipeline"""
        if not ML_AVAILABLE or self.anomaly_detector is None:
            return self._fallback_anomaly_detection(df)
        
        try:
            # Use the actual ML anomaly detection
            anomalies = self.anomaly_detector.detect_anomalies(df)
            return anomalies
        except Exception as e:
            st.warning(f"ML anomaly detection failed: {e}")
            return self._fallback_anomaly_detection(df)
    
    def _fallback_anomaly_detection(self, df):
        """Fallback statistical anomaly detection"""
        if len(df) == 0:
            return pd.DataFrame()
        
        # Statistical Z-score method
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        
        if std_amount == 0:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy['z_score'] = (df_copy['amount'] - mean_amount) / std_amount
        anomalies = df_copy[abs(df_copy['z_score']) > 2.5]
        
        return anomalies
    
    def forecast_spending_advanced(self, df, days_ahead=30):
        """Advanced spending forecasting using ML models"""
        if not ML_AVAILABLE or self.forecaster is None:
            return self._fallback_forecasting(df, days_ahead)
        
        try:
            # Use the actual ML forecasting
            forecast = self.forecaster.forecast_spending(df, days_ahead)
            return forecast, None
        except Exception as e:
            st.warning(f"ML forecasting failed: {e}")
            return self._fallback_forecasting(df, days_ahead)
    
    def _fallback_forecasting(self, df, days_ahead):
        """Fallback linear trend forecasting"""
        if len(df) < 7:
            return None, "Need at least 7 days of data"
        
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare data
            daily_spend = df.groupby(df['timestamp'].dt.date)['amount'].sum().reset_index()
            daily_spend['days_since_start'] = (pd.to_datetime(daily_spend['timestamp']) - 
                                             pd.to_datetime(daily_spend['timestamp']).min()).dt.days
            
            # Train simple model
            X = daily_spend['days_since_start'].values.reshape(-1, 1)
            y = daily_spend['amount'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict
            future_days = np.arange(X[-1][0] + 1, X[-1][0] + days_ahead + 1).reshape(-1, 1)
            predictions = model.predict(future_days)
            predictions = np.maximum(predictions, 0)
            
            # Create forecast dataframe
            future_dates = pd.date_range(start=daily_spend['timestamp'].max() + pd.Timedelta(days=1), 
                                       periods=days_ahead)
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_amount': predictions
            })
            
            return forecast_df, None
            
        except ImportError:
            return None, "ML libraries not available"
    
    def get_ml_insights(self, df):
        """Get ML-powered insights"""
        insights = {}
        
        if len(df) == 0:
            return insights
        
        # Spending pattern analysis
        insights['patterns'] = self._analyze_spending_patterns(df)
        
        # Category predictions
        insights['category_trends'] = self._analyze_category_trends(df)
        
        # Risk assessment
        insights['risk_score'] = self._calculate_risk_score(df)
        
        # Seasonal effects
        insights['seasonality'] = self._detect_seasonality(df)
        
        return insights
    
    def _analyze_spending_patterns(self, df):
        """Analyze spending patterns using ML techniques"""
        patterns = {}
        
        # Day of week patterns
        dow_spending = df.groupby(df['timestamp'].dt.day_name())['amount'].mean()
        patterns['day_of_week'] = dow_spending.to_dict()
        
        # Hour patterns
        df_copy = df.copy()
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        hourly_spending = df_copy.groupby('hour')['amount'].mean()
        patterns['hourly'] = hourly_spending.to_dict()
        
        # Monthly patterns
        monthly_spending = df.groupby(df['timestamp'].dt.month)['amount'].sum()
        patterns['monthly'] = monthly_spending.to_dict()
        
        return patterns
    
    def _analyze_category_trends(self, df):
        """Analyze category spending trends"""
        if len(df) == 0:
            return {}
        
        # Calculate month-over-month growth for each category
        monthly_category = df.groupby([df['timestamp'].dt.to_period('M'), 'category'])['amount'].sum().unstack(fill_value=0)
        
        if len(monthly_category) < 2:
            return {}
        
        # Calculate growth rates
        growth_rates = monthly_category.pct_change().iloc[-1].to_dict()
        
        return {category: rate for category, rate in growth_rates.items() if not pd.isna(rate)}
    
    def _calculate_risk_score(self, df):
        """Calculate financial risk score based on spending patterns"""
        if len(df) == 0:
            return 50
        
        risk_factors = []
        
        # Volatility risk
        daily_spend = df.groupby(df['timestamp'].dt.date)['amount'].sum()
        volatility = daily_spend.std() / daily_spend.mean() if daily_spend.mean() > 0 else 0
        risk_factors.append(min(volatility * 100, 100))
        
        # Large transaction risk
        avg_amount = df['amount'].mean()
        large_tx_ratio = len(df[df['amount'] > avg_amount * 3]) / len(df)
        risk_factors.append(large_tx_ratio * 100)
        
        # Category concentration risk
        category_share = df.groupby('category')['amount'].sum() / df['amount'].sum()
        concentration = (category_share ** 2).sum()  # Herfindahl index
        risk_factors.append(concentration * 100)
        
        # Weekend overspending risk
        df_copy = df.copy()
        df_copy['is_weekend'] = df_copy['timestamp'].dt.dayofweek >= 5
        weekend_avg = df_copy[df_copy['is_weekend']]['amount'].mean()
        weekday_avg = df_copy[~df_copy['is_weekend']]['amount'].mean()
        
        if weekday_avg > 0:
            weekend_premium = ((weekend_avg - weekday_avg) / weekday_avg) * 100
            risk_factors.append(max(0, weekend_premium))
        
        return np.mean(risk_factors)
    
    def _detect_seasonality(self, df):
        """Detect seasonal patterns in spending"""
        if len(df) == 0:
            return {}
        
        # Monthly seasonality
        monthly_avg = df.groupby(df['timestamp'].dt.month)['amount'].mean()
        
        # Identify peak and low months
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        seasonality = {
            'peak_month': peak_month,
            'low_month': low_month,
            'seasonal_variation': (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100
        }
        
        return seasonality
    
    def create_ml_visualizations(self, df):
        """Create advanced ML-powered visualizations"""
        visualizations = {}
        
        if len(df) == 0:
            return visualizations
        
        # 1. Anomaly visualization
        anomalies = self.detect_anomalies_advanced(df)
        if len(anomalies) > 0:
            fig_anomaly = px.scatter(
                df, x='timestamp', y='amount', 
                title="Transaction Anomaly Detection",
                labels={'timestamp': 'Date', 'amount': 'Amount ($)'}
            )
            
            # Highlight anomalies
            fig_anomaly.add_scatter(
                x=anomalies['timestamp'], 
                y=anomalies['amount'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Anomalies'
            )
            
            visualizations['anomalies'] = fig_anomaly
        
        # 2. Spending pattern heatmap
        df_copy = df.copy()
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
        
        heatmap_data = df_copy.groupby(['day_of_week', 'hour'])['amount'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='amount').fillna(0)
        
        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot.index = [day_names[i] for i in heatmap_pivot.index]
        
        fig_heatmap = px.imshow(
            heatmap_pivot,
            title="Spending Patterns by Day and Hour",
            labels=dict(x="Hour of Day", y="Day of Week", color="Avg Amount ($)"),
            color_continuous_scale="Viridis"
        )
        
        visualizations['patterns'] = fig_heatmap
        
        # 3. Category trend analysis
        monthly_category = df.groupby([df['timestamp'].dt.to_period('M'), 'category'])['amount'].sum().unstack(fill_value=0)
        
        if len(monthly_category) > 1:
            fig_trends = go.Figure()
            
            for category in monthly_category.columns:
                fig_trends.add_trace(go.Scatter(
                    x=[str(period) for period in monthly_category.index],
                    y=monthly_category[category],
                    mode='lines+markers',
                    name=category.title()
                ))
            
            fig_trends.update_layout(
                title="Category Spending Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Amount ($)"
            )
            
            visualizations['trends'] = fig_trends
        
        return visualizations
    
    def generate_ml_recommendations(self, df):
        """Generate ML-powered recommendations"""
        if len(df) == 0:
            return []
        
        recommendations = []
        insights = self.get_ml_insights(df)
        
        # Risk-based recommendations
        risk_score = insights.get('risk_score', 50)
        
        if risk_score > 70:
            recommendations.append({
                'type': 'warning',
                'title': '⚠️ High Financial Risk Detected',
                'message': f'Your spending patterns indicate a risk score of {risk_score:.0f}/100. Consider implementing stricter budgets.',
                'priority': 'high'
            })
        
        # Anomaly-based recommendations
        anomalies = self.detect_anomalies_advanced(df)
        if len(anomalies) > len(df) * 0.05:  # More than 5% anomalies
            recommendations.append({
                'type': 'info',
                'title': '🔍 Unusual Spending Detected',
                'message': f'Found {len(anomalies)} unusual transactions. Review large purchases for accuracy.',
                'priority': 'medium'
            })
        
        # Seasonal recommendations
        seasonality = insights.get('seasonality', {})
        if seasonality.get('seasonal_variation', 0) > 50:
            recommendations.append({
                'type': 'info',
                'title': '📅 Seasonal Spending Pattern',
                'message': f'Peak spending in month {seasonality.get("peak_month", "N/A")}. Plan budget accordingly.',
                'priority': 'low'
            })
        
        # Category trend recommendations
        category_trends = insights.get('category_trends', {})
        growing_categories = [cat for cat, growth in category_trends.items() if growth > 0.2]
        
        if growing_categories:
            recommendations.append({
                'type': 'warning',
                'title': '📈 Rapid Category Growth',
                'message': f'High growth in: {", ".join(growing_categories)}. Monitor these categories closely.',
                'priority': 'medium'
            })
        
        return recommendations

# Streamlit integration functions
def display_ml_dashboard(df, ml_integration):
    """Display ML dashboard section"""
    
    st.header("🤖 AI-Powered Financial Intelligence")
    
    if len(df) == 0:
        st.warning("No data available for ML analysis")
        return
    
    # ML Insights Overview
    insights = ml_integration.get_ml_insights(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_score = insights.get('risk_score', 50)
        st.metric(
            "🎯 Financial Risk Score", 
            f"{risk_score:.0f}/100",
            delta="Lower is better" if risk_score < 50 else "Monitor closely"
        )
    
    with col2:
        anomalies = ml_integration.detect_anomalies_advanced(df)
        st.metric("🚨 Anomalies Detected", len(anomalies))
    
    with col3:
        seasonality = insights.get('seasonality', {})
        seasonal_var = seasonality.get('seasonal_variation', 0)
        st.metric("📅 Seasonal Variation", f"{seasonal_var:.1f}%")
    
    # ML Visualizations
    st.subheader("📊 Advanced Analytics")
    
    visualizations = ml_integration.create_ml_visualizations(df)
    
    if 'anomalies' in visualizations:
        st.plotly_chart(visualizations['anomalies'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'patterns' in visualizations:
            st.plotly_chart(visualizations['patterns'], use_container_width=True)
    
    with col2:
        if 'trends' in visualizations:
            st.plotly_chart(visualizations['trends'], use_container_width=True)
    
    # AI Recommendations
    st.subheader("🧠 AI Recommendations")
    
    recommendations = ml_integration.generate_ml_recommendations(df)
    
    if recommendations:
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**: {rec['message']}")
            elif rec['type'] == 'info':
                st.info(f"**{rec['title']}**: {rec['message']}")
            else:
                st.success(f"**{rec['title']}**: {rec['message']}")
    else:
        st.success("✅ No AI recommendations at this time. Your spending patterns look healthy!")
    
    # ML Model Performance (if available)
    if ML_AVAILABLE:
        with st.expander("🔧 Model Performance Details"):
            st.info("🎯 Anomaly Detection: Isolation Forest + Statistical Analysis")
            st.info("📈 Forecasting: Ensemble (Random Forest + Linear Regression)")
            st.info("🧠 Pattern Recognition: Time Series Analysis + Clustering")
            st.info("⚡ Real-time Processing: Enabled")

# Factory function to create ML integration
def create_ml_integration():
    """Create ML integration instance"""
    return MLDashboardIntegration()

if __name__ == "__main__":
    # Test the ML integration
    ml_integration = create_ml_integration()
    print("✅ ML Dashboard Integration initialized successfully")
