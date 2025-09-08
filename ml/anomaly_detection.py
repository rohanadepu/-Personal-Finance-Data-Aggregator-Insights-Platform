import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("transaction_anomaly_detection")

class AnomalyDetector:
    def __init__(self, algorithm='isolation_forest'):
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Enhanced feature engineering for anomaly detection"""
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create spending behavior features
        df['amount_abs'] = np.abs(df['amount'])
        df['is_expense'] = (df['amount'] < 0).astype(int)
        
        # Rolling statistics (last 7 days)
        df['amount_7d_mean'] = df['amount'].rolling(window=7, min_periods=1).mean()
        df['amount_7d_std'] = df['amount'].rolling(window=7, min_periods=1).std().fillna(0)
        df['amount_7d_max'] = df['amount'].rolling(window=7, min_periods=1).max()
        df['amount_7d_min'] = df['amount'].rolling(window=7, min_periods=1).min()
        
        # Merchant frequency features
        merchant_counts = df.groupby('merchant')['amount'].count()
        df['merchant_frequency'] = df['merchant'].map(merchant_counts)
        
        # Daily aggregation features
        daily_features = df.groupby(df['timestamp'].dt.date).agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'merchant': 'nunique',
            'amount_abs': 'sum'
        }).fillna(0)
        
        # Flatten column names
        daily_features.columns = ['_'.join(col).strip() for col in daily_features.columns]
        daily_features.reset_index(inplace=True)
        daily_features['timestamp'] = pd.to_datetime(daily_features['timestamp'])
        
        # Add category features if available
        if 'category' in df.columns:
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
            
        # Select numerical features for modeling
        feature_cols = [
            'amount', 'amount_abs', 'hour', 'day_of_week', 'day_of_month', 
            'month', 'is_weekend', 'is_expense', 'amount_7d_mean', 
            'amount_7d_std', 'amount_7d_max', 'amount_7d_min', 'merchant_frequency'
        ]
        
        # Add category features if they exist
        category_cols = [col for col in df.columns if col.startswith('category_')]
        feature_cols.extend(category_cols)
        
        self.feature_columns = feature_cols
        return df[feature_cols].fillna(0), daily_features
        
    def train_isolation_forest(self, features_df, contamination=0.1):
        """Train Isolation Forest for anomaly detection with enhanced MLflow tracking"""
        with mlflow.start_run(run_name=f"isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("algorithm", "isolation_forest")
            mlflow.log_param("contamination", contamination)
            mlflow.log_param("n_features", features_df.shape[1])
            mlflow.log_param("n_samples", features_df.shape[0])
            mlflow.log_param("feature_names", list(self.feature_columns))
            
            # Scale features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features_df)
            
            # Train model
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(scaled_features)
            
            # Calculate and log metrics
            predictions = self.model.predict(scaled_features)
            scores = self.model.score_samples(scaled_features)
            anomaly_ratio = np.sum(predictions == -1) / len(predictions)
            
            mlflow.log_metric("anomaly_ratio", anomaly_ratio)
            mlflow.log_metric("anomalies_detected", int(np.sum(predictions == -1)))
            mlflow.log_metric("mean_anomaly_score", np.mean(scores))
            mlflow.log_metric("std_anomaly_score", np.std(scores))
            mlflow.log_metric("min_anomaly_score", np.min(scores))
            mlflow.log_metric("max_anomaly_score", np.max(scores))
            
            # Create feature importance visualization
            try:
                import matplotlib.pyplot as plt
                
                # Create anomaly score distribution plot
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
                plt.title('Distribution of Anomaly Scores')
                plt.xlabel('Anomaly Score')
                plt.ylabel('Frequency')
                
                plt.subplot(1, 2, 2)
                anomaly_scores = scores[predictions == -1]
                normal_scores = scores[predictions == 1]
                plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green')
                plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
                plt.title('Anomaly vs Normal Score Distribution')
                plt.xlabel('Anomaly Score')
                plt.ylabel('Frequency')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig("anomaly_analysis.png", dpi=150, bbox_inches='tight')
                mlflow.log_artifact("anomaly_analysis.png")
                plt.close()
                
                # Clean up
                os.remove("anomaly_analysis.png")
                
            except Exception as e:
                print(f"Warning: Could not create visualization: {e}")
            
            # Log model with MLflow (simplified)
            try:
                mlflow.sklearn.log_model(
                    self.model, 
                    "isolation_forest_model"
                )
                print("✅ Model logged successfully")
            except Exception as e:
                print(f"⚠️  Model logging warning: {e}")
                # Continue anyway - the model is still trained
            
            # Save and log scaler
            joblib.dump(self.scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl")
            
            # Clean up
            os.remove("scaler.pkl")
            
        return self.model, self.scaler
        
    def build_autoencoder(self, input_dim, encoding_dim=10, hidden_layers=[20, 15]):
        """Build autoencoder architecture"""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in hidden_layers:
            encoded = Dense(units, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(hidden_layers):
            decoded = Dense(units, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder, encoder
        
    def train_autoencoder(self, features_df, encoding_dim=10, epochs=100, batch_size=32, 
                         validation_split=0.2, anomaly_threshold_percentile=95):
        """Train Autoencoder for anomaly detection with enhanced MLflow tracking"""
        with mlflow.start_run(run_name=f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("algorithm", "autoencoder")
            mlflow.log_param("encoding_dim", encoding_dim)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("validation_split", validation_split)
            mlflow.log_param("anomaly_threshold_percentile", anomaly_threshold_percentile)
            mlflow.log_param("n_features", features_df.shape[1])
            mlflow.log_param("n_samples", features_df.shape[0])
            mlflow.log_param("feature_names", list(self.feature_columns))
            
            # Scale features
            self.scaler = MinMaxScaler()
            scaled_features = self.scaler.fit_transform(features_df)
            
            # Build autoencoder
            autoencoder, encoder = self.build_autoencoder(
                input_dim=scaled_features.shape[1],
                encoding_dim=encoding_dim
            )
            
            # Train autoencoder
            history = autoencoder.fit(
                scaled_features, scaled_features,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0
            )
            
            # Log training metrics over time
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                if 'val_loss' in history.history:
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            
            # Calculate reconstruction errors
            reconstructions = autoencoder.predict(scaled_features)
            reconstruction_errors = np.mean(np.square(scaled_features - reconstructions), axis=1)
            
            # Determine threshold
            threshold = np.percentile(reconstruction_errors, anomaly_threshold_percentile)
            predictions = (reconstruction_errors > threshold).astype(int)
            
            # Log metrics
            mlflow.log_metric("final_loss", history.history['loss'][-1])
            if 'val_loss' in history.history:
                mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
            mlflow.log_metric("anomaly_threshold", threshold)
            mlflow.log_metric("anomaly_ratio", np.mean(predictions))
            mlflow.log_metric("anomalies_detected", int(np.sum(predictions)))
            mlflow.log_metric("mean_reconstruction_error", np.mean(reconstruction_errors))
            mlflow.log_metric("std_reconstruction_error", np.std(reconstruction_errors))
            mlflow.log_metric("min_reconstruction_error", np.min(reconstruction_errors))
            mlflow.log_metric("max_reconstruction_error", np.max(reconstruction_errors))
            
            # Create visualizations
            try:
                import matplotlib.pyplot as plt
                
                # Create training history and reconstruction error plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Training loss plot
                axes[0, 0].plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Training History')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                
                # Reconstruction error distribution
                axes[0, 1].hist(reconstruction_errors, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(threshold, color='red', linestyle='--', 
                                 label=f'Threshold: {threshold:.4f}')
                axes[0, 1].set_title('Reconstruction Error Distribution')
                axes[0, 1].set_xlabel('Reconstruction Error')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                
                # Normal vs Anomaly reconstruction errors
                normal_errors = reconstruction_errors[predictions == 0]
                anomaly_errors = reconstruction_errors[predictions == 1]
                
                axes[1, 0].hist(normal_errors, bins=30, alpha=0.7, label='Normal', color='green')
                axes[1, 0].hist(anomaly_errors, bins=30, alpha=0.7, label='Anomaly', color='red')
                axes[1, 0].set_title('Normal vs Anomaly Reconstruction Errors')
                axes[1, 0].set_xlabel('Reconstruction Error')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
                
                # Feature reconstruction comparison (first few features)
                n_features_plot = min(5, scaled_features.shape[1])
                feature_indices = np.random.choice(scaled_features.shape[1], n_features_plot, replace=False)
                sample_indices = np.random.choice(len(scaled_features), 100, replace=False)
                
                for i, feat_idx in enumerate(feature_indices):
                    axes[1, 1].scatter(scaled_features[sample_indices, feat_idx], 
                                     reconstructions[sample_indices, feat_idx], 
                                     alpha=0.5, s=20, label=f'Feature {feat_idx}')
                
                axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
                axes[1, 1].set_title('Original vs Reconstructed Features (Sample)')
                axes[1, 1].set_xlabel('Original Value')
                axes[1, 1].set_ylabel('Reconstructed Value')
                axes[1, 1].legend()
                
                plt.tight_layout()
                plt.savefig("autoencoder_analysis.png", dpi=150, bbox_inches='tight')
                mlflow.log_artifact("autoencoder_analysis.png")
                plt.close()
                
                # Clean up
                os.remove("autoencoder_analysis.png")
                
            except Exception as e:
                print(f"Warning: Could not create visualization: {e}")
            
            # Log model with MLflow
            try:
                # Save model manually and log as artifact
                self.model.save("autoencoder_model.h5")
                mlflow.log_artifact("autoencoder_model.h5")
                os.remove("autoencoder_model.h5")
                print("✅ Autoencoder model logged successfully")
            except Exception as e:
                print(f"⚠️  Autoencoder model logging warning: {e}")
                # Continue anyway - the model is still trained
            
            # Save and log other artifacts
            joblib.dump(self.scaler, "scaler.pkl")
            np.save("anomaly_threshold.npy", threshold)
            
            mlflow.log_artifact("scaler.pkl")
            mlflow.log_artifact("anomaly_threshold.npy")
            
            # Store model and threshold
            self.model = autoencoder
            self.anomaly_threshold = threshold
            
            # Clean up
            os.remove("scaler.pkl")
            os.remove("anomaly_threshold.npy")
            
        return autoencoder, self.scaler, threshold
        
    def detect_anomalies(self, features_df):
        """Detect anomalies using trained model"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_* method first.")
            
        scaled_features = self.scaler.transform(features_df)
        
        if self.algorithm == 'isolation_forest':
            predictions = self.model.predict(scaled_features)
            scores = self.model.score_samples(scaled_features)
            is_anomaly = predictions == -1
            anomaly_scores = -scores  # Convert to positive scores
            
        elif self.algorithm == 'autoencoder':
            reconstructions = self.model.predict(scaled_features)
            reconstruction_errors = np.mean(np.square(scaled_features - reconstructions), axis=1)
            is_anomaly = reconstruction_errors > self.anomaly_threshold
            anomaly_scores = reconstruction_errors
            
        results = pd.DataFrame({
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_scores
        })
        
        return results
        
    def train(self, df, **kwargs):
        """Train anomaly detection model"""
        features_df, daily_features = self.prepare_features(df)
        
        if self.algorithm == 'isolation_forest':
            return self.train_isolation_forest(features_df, **kwargs)
        elif self.algorithm == 'autoencoder':
            return self.train_autoencoder(features_df, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

# Backward compatibility functions
def prepare_features(df):
    """Legacy function for backward compatibility"""
    detector = AnomalyDetector()
    features_df, _ = detector.prepare_features(df)
    return features_df

def train_anomaly_detector(features_df, contamination=0.1):
    """Legacy function for backward compatibility"""
    detector = AnomalyDetector('isolation_forest')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    
    with mlflow.start_run():
        mlflow.log_param("contamination", contamination)
        model.fit(scaled_features)
        
        # Calculate and log metrics
        predictions = model.predict(scaled_features)
        anomaly_ratio = np.sum(predictions == -1) / len(predictions)
        mlflow.log_metric("anomaly_ratio", anomaly_ratio)
        
        # Save artifacts
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        os.remove("scaler.pkl")
    
    return model, scaler

def detect_anomalies(model, scaler, features_df):
    """Legacy function for backward compatibility"""
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
    
    # Load sample data for testing
    sample_data = pd.read_csv("../data/sample_transactions.csv")
    sample_data['timestamp'] = pd.to_datetime(sample_data['timestamp'])
    
    print("Testing Isolation Forest...")
    detector_if = AnomalyDetector('isolation_forest')
    model_if, scaler_if = detector_if.train(sample_data, contamination=0.1)
    anomalies_if = detector_if.detect_anomalies(detector_if.prepare_features(sample_data)[0])
    print(f"Isolation Forest found {anomalies_if['is_anomaly'].sum()} anomalies")
    
    print("\nTesting Autoencoder...")
    detector_ae = AnomalyDetector('autoencoder')
    model_ae, scaler_ae, threshold = detector_ae.train(sample_data, epochs=50)
    anomalies_ae = detector_ae.detect_anomalies(detector_ae.prepare_features(sample_data)[0])
    print(f"Autoencoder found {anomalies_ae['is_anomaly'].sum()} anomalies")