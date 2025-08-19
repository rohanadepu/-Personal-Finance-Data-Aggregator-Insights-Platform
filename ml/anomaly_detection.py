import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def prepare_features(df):
    # Group by category and date to get daily spending
    daily_spend = df.groupby(['category', pd.Grouper(key='timestamp', freq='D')])['amount'].sum().reset_index()
    
    # Pivot to get categories as features
    features_df = daily_spend.pivot(index='timestamp', columns='category', values='amount').fillna(0)
    return features_df

def train_anomaly_detector(features_df, contamination=0.1):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    
    with mlflow.start_run():
        mlflow.log_param("contamination", contamination)
        model.fit(scaled_features)
        
        # Log the model
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        mlflow.log_artifact("scaler.pkl")
        
        # Calculate and log metrics
        predictions = model.predict(scaled_features)
        anomaly_ratio = np.sum(predictions == -1) / len(predictions)
        mlflow.log_metric("anomaly_ratio", anomaly_ratio)
    
    return model, scaler

def detect_anomalies(model, scaler, features_df):
    scaled_features = scaler.transform(features_df)
    predictions = model.predict(scaled_features)
    scores = model.score_samples(scaled_features)
    
    results = pd.DataFrame({
        'timestamp': features_df.index,
        'is_anomaly': predictions == -1,
        'anomaly_score': scores
    })
    
    return results

if __name__ == "__main__":
    # Example usage
    import duckdb
    
    # Load transactions
    conn = duckdb.connect(database=':memory:', read_only=False)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    
    # Prepare features
    features_df = prepare_features(df)
    
    # Train model
    model, scaler = train_anomaly_detector(features_df)
    
    # Detect anomalies
    anomalies = detect_anomalies(model, scaler, features_df)
    print(f"Found {anomalies['is_anomaly'].sum()} anomalies")