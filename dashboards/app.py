import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import psycopg2
import os
import numpy as np
import io
import base64
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #721c24;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        color: #856404;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_transactions():
    """Load transactions data with hybrid DuckDB + PostgreSQL strategy"""
    
    # Get DuckDB path from environment (Docker volume mount)
    duckdb_path = os.getenv('DUCKDB_PATH', '/shared_data/finance_analytics.duckdb')
    
    # For local development, fallback to local path
    if not os.path.exists(duckdb_path):
        duckdb_path = 'finance_analytics.duckdb'
    
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
    
    # Strategy 2: Try PostgreSQL as fallback (Docker vs local)
    postgres_hosts = [
        "postgresql://airflow:airflow@postgres:5432/airflow",  # Docker
        "postgresql://airflow:airflow@localhost:5432/airflow"  # Local
    ]
    
    for connection_string in postgres_hosts:
        try:
            pg_engine = create_engine(connection_string)
            df = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp DESC", pg_engine)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                host = "Docker PostgreSQL" if "postgres:5432" in connection_string else "Local PostgreSQL"
                st.info(f"📊 Loaded {len(df)} transactions from {host}")
                st.warning("Consider running data sync to populate DuckDB for better analytics performance")
                return df
        except Exception as e:
            st.warning(f"PostgreSQL connection failed for {connection_string}: {e}")
            continue
    
    # Strategy 3: Try sample data (multiple paths for Docker vs local)
    sample_data_paths = [
        '/app/data/sample_transactions.csv',  # Docker path
        '../data/sample_transactions.csv',    # Local relative path
        './data/sample_transactions.csv',     # Alternative local path
        '/shared_data/sample_transactions.csv'  # Shared volume
    ]
    
    for sample_data_path in sample_data_paths:
        try:
            if os.path.exists(sample_data_path):
                df = pd.read_csv(sample_data_path)
                
                # Add missing columns for the sample data
                df['city'] = df.get('city', 'Sample City')
                df['state'] = df.get('state', 'CA')
                df['channel'] = df.get('channel', 'online')
                
                # Enhanced category mapping
                category_mapping = {
                    'Walmart': 'retail', 'Costco': 'retail', 'Amazon': 'retail', 'Target': 'retail',
                    'Whole Foods': 'grocery', 'Trader Joes': 'grocery', 'Kroger': 'grocery',
                    'Starbucks': 'food', 'McDonalds': 'food', 'Chipotle': 'food', 'Subway': 'food',
                    'Shell': 'gas', 'Exxon': 'gas', 'Chevron': 'gas', 'BP': 'gas',
                    'Netflix': 'entertainment', 'Spotify': 'entertainment', 'AMC Theaters': 'entertainment',
                    'CVS Pharmacy': 'healthcare', 'Walgreens': 'healthcare',
                    'Uber': 'transportation', 'Lyft': 'transportation',
                    'PG&E': 'utilities', 'Comcast': 'utilities', 'Verizon': 'utilities',
                    '24 Hour Fitness': 'fitness', 'Planet Fitness': 'fitness'
                }
                
                df['category'] = df.get('category', df.get('merchant', 'Unknown').map(category_mapping).fillna('other'))
                df['subcategory'] = df.get('subcategory', 'general')
                df['is_recurring'] = df.get('is_recurring', False)
                df['normalized_amount'] = df.get('normalized_amount', df['amount'])
                
                # Ensure required columns exist
                required_columns = ['tx_id', 'timestamp', 'amount', 'merchant']
                for col in required_columns:
                    if col not in df.columns:
                        if col == 'tx_id':
                            df['tx_id'] = range(1, len(df) + 1)
                        else:
                            df[col] = 'Unknown'
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                st.info(f"📁 Loaded {len(df)} transactions from sample data ({sample_data_path})")
                return df
        except Exception as e:
            continue
    
    # Strategy 4: Create demo data if nothing else works
    st.warning("No data sources available - generating demo data")
    return generate_demo_data()

def generate_demo_data():
    """Generate demo data for demonstration purposes"""
    import random
    from datetime import datetime, timedelta
    
    # Demo merchants and categories
    demo_data = []
    merchants = {
        'grocery': ['Whole Foods', 'Trader Joes', 'Kroger'],
        'retail': ['Amazon', 'Target', 'Best Buy'],
        'food': ['Starbucks', 'McDonalds', 'Chipotle'],
        'gas': ['Shell', 'Exxon', 'Chevron'],
        'entertainment': ['Netflix', 'Spotify', 'AMC Theaters']
    }
    
    # Generate 200 transactions over last 60 days
    end_date = datetime.now()
    
    for i in range(200):
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        
        # Random date in last 60 days
        days_back = random.randint(0, 60)
        timestamp = end_date - timedelta(days=days_back)
        
        # Category-based amount ranges
        amount_ranges = {
            'grocery': (20, 150),
            'retail': (25, 300),
            'food': (8, 50),
            'gas': (30, 80),
            'entertainment': (10, 100)
        }
        
        amount = random.uniform(*amount_ranges[category])
        
        demo_data.append({
            'tx_id': f'demo_{i+1}',
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': 'USD',
            'merchant': merchant,
            'mcc': 5999,
            'account_id': 'demo_account',
            'city': 'Demo City',
            'state': 'CA',
            'channel': random.choice(['online', 'in-store']),
            'category': category,
            'subcategory': 'general',
            'is_recurring': False,
            'normalized_amount': round(amount, 2)
        })
    
    df = pd.DataFrame(demo_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    st.info("🎭 Generated demo data for demonstration")
    return df

# ========== ADVANCED ANALYTICS FUNCTIONS ==========

def create_spending_trends_chart(df):
    """Create daily/weekly spending trends with advanced analytics"""
    daily_spend = df.groupby(df['timestamp'].dt.date)['amount'].sum().reset_index()
    daily_spend['timestamp'] = pd.to_datetime(daily_spend['timestamp'])
    daily_spend['7_day_avg'] = daily_spend['amount'].rolling(window=7).mean()
    daily_spend['30_day_avg'] = daily_spend['amount'].rolling(window=30).mean()
    
    fig = go.Figure()
    
    # Add daily spending
    fig.add_trace(go.Scatter(
        x=daily_spend['timestamp'], 
        y=daily_spend['amount'],
        mode='lines+markers',
        name='Daily Spending',
        line=dict(color='lightblue', width=1),
        marker=dict(size=4)
    ))
    
    # Add 7-day average
    fig.add_trace(go.Scatter(
        x=daily_spend['timestamp'], 
        y=daily_spend['7_day_avg'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='orange', width=3)
    ))
    
    # Add 30-day average
    fig.add_trace(go.Scatter(
        x=daily_spend['timestamp'], 
        y=daily_spend['30_day_avg'],
        mode='lines',
        name='30-Day Average',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Daily Spending Trends with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        hovermode='x unified'
    )
    
    return fig

def create_category_comparison(df):
    """Compare current vs previous period spending with advanced metrics"""
    current_month = df[df['timestamp'] >= pd.Timestamp.now().replace(day=1)]
    prev_month = df[(df['timestamp'] >= pd.Timestamp.now().replace(day=1) - pd.DateOffset(months=1)) &
                    (df['timestamp'] < pd.Timestamp.now().replace(day=1))]
    
    current_by_cat = current_month.groupby('category')['amount'].agg(['sum', 'count', 'mean']).round(2)
    prev_by_cat = prev_month.groupby('category')['amount'].agg(['sum', 'count', 'mean']).round(2)
    
    comparison = pd.DataFrame({
        'Current_Amount': current_by_cat['sum'],
        'Previous_Amount': prev_by_cat['sum'],
        'Current_Transactions': current_by_cat['count'],
        'Previous_Transactions': prev_by_cat['count'],
        'Current_Avg': current_by_cat['mean'],
        'Previous_Avg': prev_by_cat['mean']
    }).fillna(0)
    
    comparison['Amount_Change_%'] = ((comparison['Current_Amount'] - comparison['Previous_Amount']) / 
                                   comparison['Previous_Amount'] * 100).fillna(0)
    comparison['Transaction_Change_%'] = ((comparison['Current_Transactions'] - comparison['Previous_Transactions']) / 
                                        comparison['Previous_Transactions'] * 100).fillna(0)
    
    return comparison

def calculate_spending_insights(df):
    """Calculate comprehensive spending insights and KPIs"""
    insights = {}
    
    # Basic metrics
    insights['total_amount'] = df['amount'].sum()
    insights['total_transactions'] = len(df)
    insights['avg_transaction'] = df['amount'].mean()
    insights['median_transaction'] = df['amount'].median()
    
    # Time-based insights
    unique_days = len(df['timestamp'].dt.date.unique())
    insights['avg_daily'] = df['amount'].sum() / unique_days if unique_days > 0 else 0
    insights['avg_transactions_per_day'] = len(df) / unique_days if unique_days > 0 else 0
    
    # Spending distribution
    insights['std_dev'] = df['amount'].std()
    insights['spending_volatility'] = insights['std_dev'] / insights['avg_transaction'] if insights['avg_transaction'] > 0 else 0
    
    # Peak spending analysis
    daily_totals = df.groupby(df['timestamp'].dt.date)['amount'].sum()
    if len(daily_totals) > 0:
        insights['max_day_amount'] = daily_totals.max()
        insights['max_day_date'] = daily_totals.idxmax()
        insights['min_day_amount'] = daily_totals.min()
        insights['min_day_date'] = daily_totals.idxmin()
    
    # Weekend vs weekday analysis
    df_copy = df.copy()
    df_copy['weekday'] = df_copy['timestamp'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['weekday'].isin([5, 6])
    
    weekend_stats = df_copy[df_copy['is_weekend']]['amount']
    weekday_stats = df_copy[~df_copy['is_weekend']]['amount']
    
    if len(weekend_stats) > 0 and len(weekday_stats) > 0:
        insights['weekend_avg'] = weekend_stats.mean()
        insights['weekday_avg'] = weekday_stats.mean()
        insights['weekend_premium'] = ((insights['weekend_avg'] - insights['weekday_avg']) / 
                                     insights['weekday_avg'] * 100)
    else:
        insights['weekend_avg'] = 0
        insights['weekday_avg'] = 0
        insights['weekend_premium'] = 0
    
    # Category concentration
    category_spend = df.groupby('category')['amount'].sum()
    if len(category_spend) > 0:
        insights['top_category'] = category_spend.idxmax()
        insights['top_category_amount'] = category_spend.max()
        insights['top_category_pct'] = (insights['top_category_amount'] / insights['total_amount'] * 100)
        
        # Calculate Herfindahl-Hirschman Index for spending concentration
        category_shares = category_spend / insights['total_amount']
        insights['spending_concentration_index'] = (category_shares ** 2).sum()
    
    return insights

def create_spending_heatmap(df):
    """Create advanced spending heatmap by day of week and hour"""
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_name'] = df_copy['timestamp'].dt.day_name()
    df_copy['day_order'] = df_copy['timestamp'].dt.dayofweek
    
    # Create heatmap data
    heatmap_data = df_copy.groupby(['day_name', 'hour'])['amount'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='amount').fillna(0)
    
    # Reorder days of week properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([day for day in day_order if day in heatmap_pivot.index])
    
    fig = px.imshow(heatmap_pivot, 
                    title="Spending Heatmap (Day vs Hour)",
                    labels=dict(x="Hour of Day", y="Day of Week", color="Amount ($)"),
                    color_continuous_scale="Viridis")
    
    fig.update_layout(
        xaxis=dict(side="bottom"),
        height=400
    )
    
    return fig

def detect_anomalies(df, threshold=2.5):
    """Detect spending anomalies using statistical methods"""
    # Calculate Z-scores for transaction amounts
    mean_amount = df['amount'].mean()
    std_amount = df['amount'].std()
    
    df_copy = df.copy()
    df_copy['z_score'] = (df_copy['amount'] - mean_amount) / std_amount
    
    # Identify anomalies
    anomalies = df_copy[abs(df_copy['z_score']) > threshold]
    
    # Daily spending anomalies
    daily_spend = df.groupby(df['timestamp'].dt.date)['amount'].sum()
    daily_mean = daily_spend.mean()
    daily_std = daily_spend.std()
    
    daily_anomalies = daily_spend[abs((daily_spend - daily_mean) / daily_std) > threshold]
    
    return anomalies, daily_anomalies

def predict_future_spending(df, days_ahead=30):
    """Advanced ML prediction with multiple models"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error
        import numpy as np
        
        # Prepare data
        daily_spend = df.groupby(df['timestamp'].dt.date)['amount'].sum().reset_index()
        daily_spend['timestamp'] = pd.to_datetime(daily_spend['timestamp'])
        daily_spend['days_since_start'] = (daily_spend['timestamp'] - daily_spend['timestamp'].min()).dt.days
        daily_spend['day_of_week'] = daily_spend['timestamp'].dt.dayofweek
        daily_spend['day_of_month'] = daily_spend['timestamp'].dt.day
        daily_spend['month'] = daily_spend['timestamp'].dt.month
        
        if len(daily_spend) < 14:
            return None, "Need at least 14 days of data for reliable prediction"
        
        # Features engineering
        X = daily_spend[['days_since_start', 'day_of_week', 'day_of_month', 'month']].values
        y = daily_spend['amount'].values
        
        # Train multiple models
        models = {}
        
        # Linear regression with polynomial features
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        lr_model = LinearRegression()
        lr_model.fit(X_poly, y)
        models['polynomial'] = (lr_model, poly_features)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        models['random_forest'] = rf_model
        
        # Generate future dates and features
        last_date = daily_spend['timestamp'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        future_features = pd.DataFrame({
            'timestamp': future_dates,
            'days_since_start': range(X[-1][0] + 1, X[-1][0] + days_ahead + 1),
            'day_of_week': future_dates.dayofweek,
            'day_of_month': future_dates.day,
            'month': future_dates.month
        })
        
        future_X = future_features[['days_since_start', 'day_of_week', 'day_of_month', 'month']].values
        
        # Make predictions with both models
        poly_pred = models['polynomial'][0].predict(models['polynomial'][1].transform(future_X))
        rf_pred = models['random_forest'].predict(future_X)
        
        # Ensemble prediction (average)
        ensemble_pred = (poly_pred + rf_pred) / 2
        ensemble_pred = np.maximum(ensemble_pred, 0)  # Ensure no negative predictions
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_amount': ensemble_pred,
            'polynomial_pred': np.maximum(poly_pred, 0),
            'rf_pred': np.maximum(rf_pred, 0)
        })
        
        # Calculate prediction confidence (based on historical accuracy)
        if len(daily_spend) > 20:
            # Use last 20% of data for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Retrain on training data
            rf_model_val = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model_val.fit(X_train, y_train)
            y_pred = rf_model_val.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            predictions_df['confidence_lower'] = predictions_df['predicted_amount'] - mae
            predictions_df['confidence_upper'] = predictions_df['predicted_amount'] + mae
        
        return predictions_df, None
        
    except ImportError:
        # Fallback to simple linear regression if sklearn not available
        daily_spend = df.groupby(df['timestamp'].dt.date)['amount'].sum().reset_index()
        daily_spend['days_since_start'] = (pd.to_datetime(daily_spend['timestamp']) - 
                                         pd.to_datetime(daily_spend['timestamp']).min()).dt.days
        
        if len(daily_spend) < 7:
            return None, "Need at least 7 days of data for prediction"
        
        # Simple linear trend
        X = daily_spend['days_since_start'].values.reshape(-1, 1)
        y = daily_spend['amount'].values
        
        # Calculate trend manually
        slope = np.polyfit(X.flatten(), y, 1)[0]
        intercept = np.polyfit(X.flatten(), y, 1)[1]
        
        # Predict future
        future_days = np.arange(X[-1][0] + 1, X[-1][0] + days_ahead + 1)
        future_predictions = slope * future_days + intercept
        future_predictions = np.maximum(future_predictions, 0)
        
        future_dates = pd.date_range(start=daily_spend['timestamp'].max() + pd.Timedelta(days=1), 
                                   periods=days_ahead)
        
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_amount': future_predictions
        })
        
        return predictions_df, None

def create_merchant_analysis(df):
    """Advanced merchant spending analysis"""
    merchant_stats = df.groupby('merchant').agg({
        'amount': ['sum', 'count', 'mean', 'std'],
        'timestamp': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    merchant_stats.columns = ['total_spent', 'transaction_count', 'avg_amount', 'amount_std', 'first_visit', 'last_visit']
    
    # Calculate loyalty metrics
    merchant_stats['spending_consistency'] = 1 / (1 + merchant_stats['amount_std'] / merchant_stats['avg_amount'])
    merchant_stats['days_active'] = (merchant_stats['last_visit'] - merchant_stats['first_visit']).dt.days + 1
    merchant_stats['frequency'] = merchant_stats['transaction_count'] / merchant_stats['days_active']
    
    return merchant_stats.sort_values('total_spent', ascending=False)

def create_financial_health_score(df):
    """Calculate a comprehensive financial health score"""
    insights = calculate_spending_insights(df)
    
    # Initialize score components
    score_components = {}
    
    # 1. Spending Consistency (25%) - Lower volatility is better
    volatility_score = max(0, 100 - (insights['spending_volatility'] * 50))
    score_components['consistency'] = min(100, volatility_score)
    
    # 2. Category Diversification (25%) - Less concentration is better
    concentration_score = max(0, 100 - (insights.get('spending_concentration_index', 0.5) * 200))
    score_components['diversification'] = min(100, concentration_score)
    
    # 3. Weekend Premium Control (25%) - Reasonable weekend spending is good
    weekend_premium = abs(insights['weekend_premium'])
    premium_score = max(0, 100 - (weekend_premium / 2))  # Penalize extreme differences
    score_components['weekend_control'] = min(100, premium_score)
    
    # 4. Transaction Size Distribution (25%) - Balanced transaction sizes
    if insights['median_transaction'] > 0:
        size_ratio = insights['avg_transaction'] / insights['median_transaction']
        size_score = max(0, 100 - abs(size_ratio - 1.5) * 50)  # Optimal ratio around 1.5
    else:
        size_score = 50
    score_components['transaction_balance'] = min(100, size_score)
    
    # Calculate overall score
    overall_score = np.mean(list(score_components.values()))
    
    return overall_score, score_components

def generate_spending_report(df, insights):
    """Generate a comprehensive spending report"""
    report = []
    
    # Overall summary
    report.append(f"📊 **Financial Overview**")
    report.append(f"- Total Spending: ${insights['total_amount']:,.2f}")
    report.append(f"- Total Transactions: {insights['total_transactions']:,}")
    report.append(f"- Average Transaction: ${insights['avg_transaction']:,.2f}")
    report.append(f"- Daily Average: ${insights['avg_daily']:,.2f}")
    report.append("")
    
    # Spending patterns
    report.append(f"🔍 **Spending Patterns**")
    report.append(f"- Top Category: {insights.get('top_category', 'N/A')} (${insights.get('top_category_amount', 0):,.2f})")
    report.append(f"- Weekend Premium: {insights['weekend_premium']:+.1f}%")
    report.append(f"- Spending Volatility: {insights['spending_volatility']:.2f}")
    report.append("")
    
    # Financial health score
    health_score, components = create_financial_health_score(df)
    report.append(f"💪 **Financial Health Score: {health_score:.0f}/100**")
    for component, score in components.items():
        report.append(f"  - {component.replace('_', ' ').title()}: {score:.0f}/100")
    report.append("")
    
    # Recommendations
    report.append(f"💡 **Recommendations**")
    if insights['weekend_premium'] > 50:
        report.append("- Consider reducing weekend spending to improve budget control")
    if insights.get('top_category_pct', 0) > 40:
        report.append("- Spending is heavily concentrated in one category - consider diversifying")
    if insights['spending_volatility'] > 1:
        report.append("- High spending volatility detected - consider more consistent spending habits")
    
    return "\n".join(report)

def sync_data_to_duckdb():
    """Sync data from PostgreSQL to DuckDB for analytics (Docker-aware)"""
    try:
        # Try Docker PostgreSQL first, then local
        postgres_hosts = [
            "postgresql://airflow:airflow@postgres:5432/airflow",  # Docker
            "postgresql://airflow:airflow@localhost:5432/airflow"  # Local
        ]
        
        df = None
        for connection_string in postgres_hosts:
            try:
                pg_engine = create_engine(connection_string)
                df = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp DESC", pg_engine)
                if not df.empty:
                    host = "Docker PostgreSQL" if "postgres:5432" in connection_string else "Local PostgreSQL"
                    break
            except Exception as e:
                continue
        
        if df is None or df.empty:
            return False, "No data found in PostgreSQL databases"
        
        # Get DuckDB path (Docker vs local)
        duckdb_path = os.getenv('DUCKDB_PATH', '/shared_data/finance_analytics.duckdb')
        if not os.path.exists(os.path.dirname(duckdb_path)):
            duckdb_path = 'finance_analytics.duckdb'
        
        # Connect to DuckDB and create/replace table
        duck_conn = duckdb.connect(duckdb_path)
        duck_conn.execute("DROP TABLE IF EXISTS transactions")
        duck_conn.execute("CREATE TABLE transactions AS SELECT * FROM df")
        duck_conn.close()
        
        return True, f"Successfully synced {len(df)} transactions to DuckDB at {duckdb_path}"
    except Exception as e:
        return False, f"Sync failed: {str(e)}"

def get_data_source_status():
    """Check status of all data sources"""
    status = {
        'postgres_docker': False,
        'postgres_local': False,
        'duckdb_shared': False,
        'duckdb_local': False,
        'sample_data': False
    }
    
    # Check PostgreSQL connections
    postgres_connections = {
        'postgres_docker': "postgresql://airflow:airflow@postgres:5432/airflow",
        'postgres_local': "postgresql://airflow:airflow@localhost:5432/airflow"
    }
    
    for key, conn_str in postgres_connections.items():
        try:
            pg_engine = create_engine(conn_str)
            result = pd.read_sql("SELECT COUNT(*) as count FROM transactions", pg_engine)
            if result.iloc[0]['count'] > 0:
                status[key] = result.iloc[0]['count']
        except:
            status[key] = False
    
    # Check DuckDB files
    duckdb_paths = {
        'duckdb_shared': os.getenv('DUCKDB_PATH', '/shared_data/finance_analytics.duckdb'),
        'duckdb_local': 'finance_analytics.duckdb'
    }
    
    for key, path in duckdb_paths.items():
        try:
            if os.path.exists(path):
                duck_conn = duckdb.connect(path, read_only=True)
                result = duck_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
                duck_conn.close()
                status[key] = result[0] if result[0] > 0 else False
        except:
            status[key] = False
    
    # Check sample data
    sample_paths = [
        '/app/data/sample_transactions.csv',
        '../data/sample_transactions.csv',
        './data/sample_transactions.csv'
    ]
    
    for path in sample_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                status['sample_data'] = len(df)
                break
            except:
                continue
    
    return status
    st.markdown('<h1 class="main-header">🏦 Personal Finance Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
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
            st.warning("📭 No transaction data found")
            st.info("💡 To get started:")
            st.info("1. 🐘 Run PostgreSQL with transaction data")
            st.info("2. 🦆 Sync to DuckDB for analytics")
            st.info("3. 📁 Use sample data for demo")
            return
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sidebar filters
        st.sidebar.subheader("🔍 Filters")
        
        # Date range filter
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Category filter
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.sidebar.selectbox("🏷️ Category", categories)
        
        # Amount range filter
        min_amount, max_amount = float(df['amount'].min()), float(df['amount'].max())
        amount_range = st.sidebar.slider(
            "💰 Amount Range",
            min_value=min_amount,
            max_value=max_amount,
            value=(min_amount, max_amount)
        )
        
        # Apply filters
        if len(date_range) == 2:
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            filtered_df = df.loc[mask]
        else:
            filtered_df = df
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        filtered_df = filtered_df[
            (filtered_df['amount'] >= amount_range[0]) & 
            (filtered_df['amount'] <= amount_range[1])
        ]
        
        # Sidebar metrics
        st.sidebar.subheader("📈 Quick Stats")
        st.sidebar.metric("Total Transactions", f"{len(filtered_df):,}")
        st.sidebar.metric("Total Amount", f"${filtered_df['amount'].sum():,.2f}")
        if len(filtered_df) > 0:
            st.sidebar.metric("Average Transaction", f"${filtered_df['amount'].mean():.2f}")
        
        # Main dashboard with tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🔍 Deep Analytics", 
            "🤖 ML Predictions", 
            "⚠️ Smart Alerts"
        ])
        
        with tab1:
            st.header("📊 Financial Overview")
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            if len(filtered_df) > 0:
                insights = calculate_spending_insights(filtered_df)
                
                with col1:
                    st.metric(
                        "💰 Total Spending", 
                        f"${insights['total_amount']:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "🛒 Transactions", 
                        f"{insights['total_transactions']:,}"
                    )
                
                with col3:
                    st.metric(
                        "📊 Avg Transaction", 
                        f"${insights['avg_transaction']:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "📅 Daily Average", 
                        f"${insights['avg_daily']:.2f}"
                    )
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💳 Spending by Category")
                if len(filtered_df) > 0:
                    category_spend = filtered_df.groupby('category')['amount'].sum().sort_values(ascending=True)
                    fig = px.bar(
                        x=category_spend.values, 
                        y=category_spend.index,
                        orientation='h',
                        title="Category Breakdown",
                        labels={'x': 'Amount ($)', 'y': 'Category'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected filters")
            
            with col2:
                st.subheader("🏪 Top Merchants")
                if len(filtered_df) > 0:
                    top_merchants = filtered_df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
                    fig = px.pie(
                        values=top_merchants.values,
                        names=top_merchants.index,
                        title="Top 10 Merchants"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected filters")
            
            # Spending trends
            st.subheader("📈 Spending Trends")
            if len(filtered_df) > 0:
                trends_fig = create_spending_trends_chart(filtered_df)
                st.plotly_chart(trends_fig, use_container_width=True)
            
            # Recent transactions
            st.subheader("🕒 Recent Transactions")
            if len(filtered_df) > 0:
                display_cols = ['timestamp', 'merchant', 'amount', 'category']
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                recent_transactions = filtered_df.sort_values('timestamp', ascending=False).head(10)
                st.dataframe(recent_transactions[available_cols], use_container_width=True)
            else:
                st.info("No transactions to display")
        
        with tab2:
            st.header("🔍 Deep Spending Analytics")
            
            if len(filtered_df) > 0:
                insights = calculate_spending_insights(filtered_df)
                
                # Advanced metrics grid
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Statistical Insights")
                    st.metric("Spending Volatility", f"{insights['spending_volatility']:.2f}")
                    st.metric("Weekend Premium", f"{insights['weekend_premium']:+.1f}%")
                    if 'max_day_amount' in insights:
                        st.metric("Highest Day", f"${insights['max_day_amount']:.2f}")
                        st.caption(f"Date: {insights['max_day_date']}")
                
                with col2:
                    st.subheader("🎯 Category Analysis")
                    if 'top_category' in insights:
                        st.metric("Top Category", insights['top_category'].title())
                        st.metric("Category Dominance", f"{insights['top_category_pct']:.1f}%")
                        st.metric("Spending Concentration", f"{insights.get('spending_concentration_index', 0):.3f}")
                
                with col3:
                    st.subheader("💪 Financial Health")
                    health_score, components = create_financial_health_score(filtered_df)
                    st.metric("Health Score", f"{health_score:.0f}/100")
                    
                    # Health score breakdown
                    for component, score in components.items():
                        st.progress(score/100, text=f"{component.replace('_', ' ').title()}: {score:.0f}/100")
                
                # Advanced visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 Spending Heatmap")
                    heatmap_fig = create_spending_heatmap(filtered_df)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Period Comparison")
                    comparison = create_category_comparison(filtered_df)
                    if not comparison.empty:
                        st.dataframe(comparison.round(2), use_container_width=True)
                    else:
                        st.info("Need more historical data for comparison")
        
        with tab3:
            st.header("🤖 AI-Powered Predictions")
            
            if len(filtered_df) > 0:
                # Prediction controls
                col1, col2 = st.columns(2)
                with col1:
                    prediction_days = st.slider("📅 Prediction Period (days)", 7, 90, 30)
                
                # Generate predictions
                with st.spinner("🧠 Training ML models..."):
                    predictions_df, error = predict_future_spending(filtered_df, prediction_days)
                
                if error:
                    st.warning(f"⚠️ {error}")
                    st.info("💡 Try with more historical data for better predictions")
                else:
                    # Prediction summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_predicted = predictions_df['predicted_amount'].sum()
                        st.metric("🔮 Predicted Total", f"${total_predicted:,.2f}")
                    
                    with col2:
                        daily_avg = predictions_df['predicted_amount'].mean()
                        st.metric("📊 Daily Average", f"${daily_avg:.2f}")
                    
                    with col3:
                        historical_avg = filtered_df['amount'].sum() / len(filtered_df['timestamp'].dt.date.unique())
                        change_pct = ((daily_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                        st.metric("📈 Trend Change", f"{change_pct:+.1f}%")
                    
                    with col4:
                        max_day = predictions_df['predicted_amount'].max()
                        st.metric("🎯 Peak Day", f"${max_day:.2f}")
                    
                    # Prediction visualization
                    st.subheader("📈 Forecast Visualization")
                    
                    # Combine historical and predicted data
                    historical = filtered_df.groupby(filtered_df['timestamp'].dt.date)['amount'].sum().reset_index()
                    historical['type'] = 'Historical'
                    historical['date'] = pd.to_datetime(historical['timestamp'])
                    historical['amount'] = historical['amount']
                    
                    predicted = predictions_df.copy()
                    predicted['amount'] = predicted['predicted_amount']
                    predicted['type'] = 'Predicted'
                    predicted['date'] = pd.to_datetime(predicted['date'])
                    
                    # Combine datasets
                    combined = pd.concat([
                        historical[['date', 'amount', 'type']],
                        predicted[['date', 'amount', 'type']]
                    ])
                    
                    fig = px.line(
                        combined, 
                        x='date', 
                        y='amount', 
                        color='type',
                        title="Historical vs Predicted Spending",
                        labels={'amount': 'Amount ($)', 'date': 'Date'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("⚠️ Smart Financial Alerts")
            
            if len(filtered_df) > 0:
                insights = calculate_spending_insights(filtered_df)
                
                # Budget tracking
                st.subheader("💰 Budget Management")
                
                col1, col2 = st.columns(2)
                with col1:
                    monthly_budget = st.number_input("🎯 Monthly Budget ($)", value=2000.0, step=100.0)
                with col2:
                    alert_threshold = st.slider("⚠️ Alert Threshold (%)", 50, 95, 80)
                
                # Calculate current month spending
                current_month = filtered_df[filtered_df['timestamp'] >= pd.Timestamp.now().replace(day=1)]
                monthly_spend = current_month['amount'].sum()
                days_in_month = pd.Timestamp.now().days_in_month
                days_passed = pd.Timestamp.now().day
                
                if days_passed > 0:
                    projected_monthly = (monthly_spend / days_passed) * days_in_month
                else:
                    projected_monthly = monthly_spend
                
                # Budget status
                budget_used_pct = (monthly_spend / monthly_budget) * 100 if monthly_budget > 0 else 0
                projection_pct = (projected_monthly / monthly_budget) * 100 if monthly_budget > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "💳 Current Spend", 
                        f"${monthly_spend:.2f}",
                        delta=f"{budget_used_pct:.1f}% of budget"
                    )
                
                with col2:
                    st.metric(
                        "📊 Projected Monthly", 
                        f"${projected_monthly:.2f}",
                        delta=f"{projection_pct:.1f}% of budget"
                    )
                
                with col3:
                    remaining = monthly_budget - monthly_spend
                    st.metric(
                        "💰 Remaining", 
                        f"${remaining:.2f}",
                        delta=f"{(remaining/monthly_budget)*100:.1f}% left" if monthly_budget > 0 else "No budget set"
                    )
                
                # Alert system
                st.subheader("🚨 Active Alerts")
                
                alerts = []
                
                # Budget alerts
                if budget_used_pct > alert_threshold:
                    alerts.append(('danger', f'🚨 Budget Alert: You\'ve used {budget_used_pct:.1f}% of your monthly budget!'))
                elif projection_pct > 100:
                    alerts.append(('warning', f'⚠️ Overspend Warning: Projected to exceed budget by ${projected_monthly - monthly_budget:.2f}'))
                
                # Spending pattern alerts
                if insights['weekend_premium'] > 50:
                    alerts.append(('warning', f'🎉 Weekend spending is {insights["weekend_premium"]:.1f}% higher than weekdays'))
                
                if insights['spending_volatility'] > 1.5:
                    alerts.append(('warning', '📈 High spending volatility detected - consider budgeting'))
                
                # Display alerts
                if alerts:
                    for alert_type, message in alerts:
                        if alert_type == 'danger':
                            st.error(message)
                        elif alert_type == 'warning':
                            st.warning(message)
                        else:
                            st.info(message)
                else:
                    st.success("✅ No active alerts - your spending is on track!")
                
                # Category breakdown
                st.subheader("📊 Category Breakdown")
                category_spend = filtered_df.groupby('category')['amount'].sum().sort_values(ascending=False)
                
                for category, amount in category_spend.head(5).items():
                    pct_of_total = (amount / filtered_df['amount'].sum()) * 100
                    if pct_of_total > 30:
                        st.error(f"🔴 {category.title()}: ${amount:.2f} ({pct_of_total:.1f}%)")
                    elif pct_of_total > 20:
                        st.warning(f"🟡 {category.title()}: ${amount:.2f} ({pct_of_total:.1f}%)")
                    else:
                        st.success(f"🟢 {category.title()}: ${amount:.2f} ({pct_of_total:.1f}%)")
    
    except Exception as e:
        st.error(f"❌ Error loading dashboard: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.write("Error details:", str(e))
            st.write("Current working directory:", os.getcwd())
            st.write("Python path:", os.path.dirname(__file__))

    return status

def main():
    st.title("🏦 Personal Finance Analytics Platform")
    
    # Data source diagnostics in sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Data source status
        with st.expander("📊 Data Sources Status"):
            status = get_data_source_status()
            
            st.write("**PostgreSQL:**")
            pg_docker = status.get('postgres_docker', False)
            pg_local = status.get('postgres_local', False)
            
            if pg_docker:
                st.success(f"✅ Docker: {pg_docker} transactions")
            else:
                st.error("❌ Docker: Not available")
                
            if pg_local:
                st.success(f"✅ Local: {pg_local} transactions")  
            else:
                st.error("❌ Local: Not available")
            
            st.write("**DuckDB:**")
            duck_shared = status.get('duckdb_shared', False)
            duck_local = status.get('duckdb_local', False)
            
            if duck_shared:
                st.success(f"✅ Shared Volume: {duck_shared} transactions")
            else:
                st.warning("⚠️ Shared Volume: Not available")
                
            if duck_local:
                st.success(f"✅ Local File: {duck_local} transactions")
            else:
                st.warning("⚠️ Local File: Not available")
            
            st.write("**Sample Data:**")
            sample = status.get('sample_data', False)
            if sample:
                st.info(f"📁 Available: {sample} transactions")
            else:
                st.warning("📁 Not found")
        
        # Data sync controls
        st.subheader("🔄 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh"):
                st.experimental_rerun()
        
        with col2:
            if st.button("🔗 Sync to DuckDB"):
                with st.spinner("Syncing..."):
                    success, message = sync_data_to_duckdb()
                    if success:
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)
    
    # Load and display data
    try:
        df = load_transactions()
        
        if df.empty:
            st.warning("📭 No transaction data found")
            st.info("💡 **Getting Started:**")
            st.info("1. 🐘 Ensure PostgreSQL is running with transaction data")
            st.info("2. 🦆 Sync data to DuckDB for analytics")
            st.info("3. 📁 Add sample data files")
            st.info("4. 🎭 Use generated demo data")
            return
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sidebar filters
        st.sidebar.subheader("🔍 Filters")
        
        # Date range filter
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Category filter
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.sidebar.selectbox("🏷️ Category", categories)
        
        # Amount range filter
        min_amount, max_amount = float(df['amount'].min()), float(df['amount'].max())
        amount_range = st.sidebar.slider(
            "💰 Amount Range",
            min_value=min_amount,
            max_value=max_amount,
            value=(min_amount, max_amount)
        )
        
        # Apply filters
        if len(date_range) == 2:
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            filtered_df = df.loc[mask]
        else:
            filtered_df = df
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        filtered_df = filtered_df[
            (filtered_df['amount'] >= amount_range[0]) & 
            (filtered_df['amount'] <= amount_range[1])
        ]
        
        # Sidebar metrics
        st.sidebar.subheader("📈 Quick Stats")
        st.sidebar.metric("Total Transactions", f"{len(filtered_df):,}")
        st.sidebar.metric("Total Amount", f"${filtered_df['amount'].sum():,.2f}")
        if len(filtered_df) > 0:
            st.sidebar.metric("Average Transaction", f"${filtered_df['amount'].mean():.2f}")
        
        # Main dashboard with enhanced tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🔍 Analytics", 
            "🤖 ML Insights", 
            "📋 Reports"
        ])
        
        with tab1:
            st.header("📊 Financial Overview")
            
            # Key metrics row
            if len(filtered_df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                total_amount = filtered_df['amount'].sum()
                total_transactions = len(filtered_df)
                avg_transaction = filtered_df['amount'].mean()
                
                # Calculate daily average
                unique_days = len(filtered_df['timestamp'].dt.date.unique())
                daily_avg = total_amount / unique_days if unique_days > 0 else 0
                
                with col1:
                    st.metric("💰 Total Spending", f"${total_amount:,.2f}")
                
                with col2:
                    st.metric("🛒 Transactions", f"{total_transactions:,}")
                
                with col3:
                    st.metric("📊 Avg Transaction", f"${avg_transaction:.2f}")
                
                with col4:
                    st.metric("📅 Daily Average", f"${daily_avg:.2f}")
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💳 Spending by Category")
                if len(filtered_df) > 0:
                    category_spend = filtered_df.groupby('category')['amount'].sum().sort_values(ascending=True)
                    fig = px.bar(
                        x=category_spend.values, 
                        y=category_spend.index,
                        orientation='h',
                        title="Category Breakdown",
                        labels={'x': 'Amount ($)', 'y': 'Category'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected filters")
            
            with col2:
                st.subheader("🏪 Top Merchants")
                if len(filtered_df) > 0:
                    top_merchants = filtered_df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
                    fig = px.pie(
                        values=top_merchants.values,
                        names=top_merchants.index,
                        title="Top 10 Merchants"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for selected filters")
            
            # Spending trends
            st.subheader("📈 Spending Trends")
            if len(filtered_df) > 0:
                daily_spend = filtered_df.groupby(filtered_df['timestamp'].dt.date)['amount'].sum().reset_index()
                daily_spend['timestamp'] = pd.to_datetime(daily_spend['timestamp'])
                
                fig = px.line(
                    daily_spend,
                    x='timestamp',
                    y='amount',
                    title="Daily Spending Trend",
                    labels={'timestamp': 'Date', 'amount': 'Amount ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent transactions
            st.subheader("🕒 Recent Transactions")
            if len(filtered_df) > 0:
                display_cols = ['timestamp', 'merchant', 'amount', 'category']
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                recent_transactions = filtered_df.sort_values('timestamp', ascending=False).head(10)
                st.dataframe(recent_transactions[available_cols], use_container_width=True)
            else:
                st.info("No transactions to display")
        
        with tab2:
            st.header("🔍 Deep Analytics")
            
            if len(filtered_df) > 0:
                # Weekend vs Weekday analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📅 Day of Week Analysis")
                    dow_spend = filtered_df.groupby(filtered_df['timestamp'].dt.day_name())['amount'].mean()
                    fig = px.bar(
                        x=dow_spend.index,
                        y=dow_spend.values,
                        title="Average Spending by Day",
                        labels={'x': 'Day of Week', 'y': 'Average Amount ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🕒 Hourly Patterns")
                    if 'timestamp' in filtered_df.columns:
                        filtered_df_copy = filtered_df.copy()
                        filtered_df_copy['hour'] = filtered_df_copy['timestamp'].dt.hour
                        hourly_spend = filtered_df_copy.groupby('hour')['amount'].mean()
                        fig = px.line(
                            x=hourly_spend.index,
                            y=hourly_spend.values,
                            title="Average Spending by Hour",
                            labels={'x': 'Hour', 'y': 'Average Amount ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Monthly comparison
                st.subheader("📊 Monthly Comparison")
                monthly_spend = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('M'))['amount'].sum()
                if len(monthly_spend) > 1:
                    fig = px.line(
                        x=[str(period) for period in monthly_spend.index],
                        y=monthly_spend.values,
                        title="Monthly Spending Trend",
                        labels={'x': 'Month', 'y': 'Total Amount ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                st.subheader("📊 Statistical Summary")
                stats_df = filtered_df.groupby('category')['amount'].agg([
                    'count', 'mean', 'std', 'min', 'max', 'sum'
                ]).round(2)
                stats_df.columns = ['Transactions', 'Mean', 'Std Dev', 'Min', 'Max', 'Total']
                st.dataframe(stats_df, use_container_width=True)
        
        with tab3:
            st.header("🤖 Machine Learning Insights")
            
            if len(filtered_df) > 0:
                # Simple anomaly detection
                st.subheader("🚨 Anomaly Detection")
                
                mean_amount = filtered_df['amount'].mean()
                std_amount = filtered_df['amount'].std()
                
                if std_amount > 0:
                    z_scores = (filtered_df['amount'] - mean_amount) / std_amount
                    anomalies = filtered_df[abs(z_scores) > 2.5]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Anomalies Detected", len(anomalies))
                    
                    with col2:
                        st.metric("Anomaly Rate", f"{len(anomalies)/len(filtered_df)*100:.1f}%")
                    
                    if len(anomalies) > 0:
                        st.dataframe(anomalies[['timestamp', 'merchant', 'amount', 'category']].head())
                
                # Simple forecasting
                st.subheader("🔮 Spending Forecast")
                
                if len(filtered_df) > 7:
                    try:
                        from sklearn.linear_model import LinearRegression
                        
                        # Prepare daily data
                        daily_data = filtered_df.groupby(filtered_df['timestamp'].dt.date)['amount'].sum().reset_index()
                        daily_data['days'] = (pd.to_datetime(daily_data['timestamp']) - 
                                            pd.to_datetime(daily_data['timestamp']).min()).dt.days
                        
                        # Train simple model
                        X = daily_data['days'].values.reshape(-1, 1)
                        y = daily_data['amount'].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Predict next 30 days
                        future_days = np.arange(X[-1][0] + 1, X[-1][0] + 31).reshape(-1, 1)
                        predictions = model.predict(future_days)
                        
                        # Create prediction dataframe
                        future_dates = pd.date_range(
                            start=daily_data['timestamp'].max() + pd.Timedelta(days=1), 
                            periods=30
                        )
                        
                        pred_df = pd.DataFrame({
                            'date': future_dates,
                            'predicted_amount': np.maximum(predictions, 0)
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("30-Day Forecast", f"${pred_df['predicted_amount'].sum():,.2f}")
                        
                        with col2:
                            st.metric("Daily Average Forecast", f"${pred_df['predicted_amount'].mean():.2f}")
                        
                        # Plot forecast
                        fig = px.line(
                            pred_df,
                            x='date',
                            y='predicted_amount',
                            title="30-Day Spending Forecast",
                            labels={'date': 'Date', 'predicted_amount': 'Predicted Amount ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ImportError:
                        st.info("📊 Install scikit-learn for advanced ML features")
                    except Exception as e:
                        st.warning(f"Forecasting error: {e}")
        
        with tab4:
            st.header("📋 Reports & Export")
            
            if len(filtered_df) > 0:
                # Summary report
                st.subheader("📊 Executive Summary")
                
                total_amount = filtered_df['amount'].sum()
                avg_transaction = filtered_df['amount'].mean()
                top_category = filtered_df.groupby('category')['amount'].sum().idxmax()
                top_merchant = filtered_df.groupby('merchant')['amount'].sum().idxmax()
                
                report = f"""
                **Financial Summary Report**
                
                - **Total Spending:** ${total_amount:,.2f}
                - **Total Transactions:** {len(filtered_df):,}
                - **Average Transaction:** ${avg_transaction:.2f}
                - **Top Category:** {top_category.title()}
                - **Top Merchant:** {top_merchant}
                - **Date Range:** {filtered_df['timestamp'].min().date()} to {filtered_df['timestamp'].max().date()}
                """
                
                st.markdown(report)
                
                # Export functionality
                st.subheader("📤 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Download CSV"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download Filtered Data",
                            data=csv,
                            file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("📈 Download Summary"):
                        summary_data = {
                            'Metric': ['Total Spending', 'Total Transactions', 'Average Transaction'],
                            'Value': [f"${total_amount:,.2f}", f"{len(filtered_df):,}", f"${avg_transaction:.2f}"]
                        }
                        summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
                        st.download_button(
                            label="Download Summary",
                            data=summary_csv,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
    
    except Exception as e:
        st.error(f"❌ Dashboard Error: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.write("Error details:", str(e))
            st.write("Environment variables:")
            st.write(f"- DUCKDB_PATH: {os.getenv('DUCKDB_PATH', 'Not set')}")
            st.write(f"- Current directory: {os.getcwd()}")
            st.write(f"- Python path: {os.path.dirname(__file__)}")

if __name__ == "__main__":
    main()
