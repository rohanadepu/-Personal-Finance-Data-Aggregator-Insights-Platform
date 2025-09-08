"""
Setup and Run ML Pipeline with MLflow

This script helps you get started with MLflow integration for your personal finance ML platform.
Run this script to:
1. Start the MLflow infrastructure
2. Run comprehensive ML experiments
3. Track and compare model performance
4. Deploy best models to production
"""

import os
import sys
import subprocess
import time
import requests
import pandas as pd
from pathlib import Path

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def start_infrastructure():
    """Start the MLflow infrastructure using Docker Compose"""
    print("🚀 Starting MLflow infrastructure...")
    
    # Change to the compose directory
    compose_dir = Path("infra/compose")
    if not compose_dir.exists():
        print("❌ Error: infra/compose directory not found!")
        print("Make sure you're running this from the project root directory.")
        return False
    
    try:
        # Start the services
        cmd = ["docker-compose", "up", "-d", "postgres", "minio", "mlflow"]
        result = subprocess.run(cmd, cwd=compose_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error starting infrastructure: {result.stderr}")
            return False
        
        print("✅ Infrastructure services started successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error starting infrastructure: {e}")
        return False

def wait_for_services():
    """Wait for services to be ready"""
    print("⏳ Waiting for services to be ready...")
    
    services = {
        "MLflow": "http://localhost:5000/health",
        "MinIO": "http://localhost:9000/minio/health/live"
    }
    
    max_retries = 30
    for service_name, health_url in services.items():
        for attempt in range(max_retries):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name} is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_retries - 1:
                print(f"⏳ Waiting for {service_name}... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"❌ {service_name} is not responding after {max_retries} attempts")
                return False
    
    return True

def setup_minio_bucket():
    """Setup MinIO bucket for MLflow artifacts"""
    try:
        # Install minio client if not available
        try:
            import minio
        except ImportError:
            print("📦 Installing minio client...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "minio"])
            import minio
        
        # Create MinIO client
        client = minio.Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        
        # Create bucket if it doesn't exist
        bucket_name = "mlflow"
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✅ Created MinIO bucket: {bucket_name}")
        else:
            print(f"✅ MinIO bucket already exists: {bucket_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up MinIO bucket: {e}")
        return False

def run_ml_pipeline():
    """Run the ML pipeline with MLflow integration"""
    print("🤖 Running ML pipeline with MLflow integration...")
    
    try:
        # Add the project root to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root))
        
        # Import and run the MLflow integration
        from ml.mlflow_integration import run_complete_ml_pipeline
        
        # Check if sample data exists
        data_path = "data/sample_transactions.csv"
        if not Path(data_path).exists():
            print(f"❌ Sample data not found at {data_path}")
            print("Please ensure the sample data file exists.")
            return False
        
        # Run the pipeline
        results = run_complete_ml_pipeline(data_path)
        
        print("✅ ML pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running ML pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary():
    """Print summary and next steps"""
    print("\n" + "="*80)
    print("🎉 MLFLOW SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 SERVICES RUNNING:")
    print("  • MLflow UI: http://localhost:5000")
    print("  • MinIO Console: http://localhost:9001 (admin/admin123)")
    print("  • PostgreSQL: localhost:5432 (airflow/airflow)")
    print("\n🔍 WHAT'S BEEN SET UP:")
    print("  • MLflow tracking server with PostgreSQL backend")
    print("  • MinIO S3-compatible storage for model artifacts")
    print("  • Comprehensive ML experiments for anomaly detection")
    print("  • Time series forecasting experiments")
    print("  • Model comparison and performance tracking")
    print("  • Production model registration and versioning")
    print("\n🚀 NEXT STEPS:")
    print("  1. Open MLflow UI: http://localhost:5000")
    print("  2. Explore experiments and model performance")
    print("  3. Compare different model configurations")
    print("  4. Review promoted production models")
    print("  5. Set up model serving for inference")
    print("\n📖 USEFUL COMMANDS:")
    print("  • View running containers: docker ps")
    print("  • Stop services: docker-compose down (in infra/compose/)")
    print("  • View logs: docker-compose logs [service_name]")
    print("  • Restart MLflow: docker-compose restart mlflow")
    print("="*80)

def main():
    """Main function to set up and run MLflow integration"""
    print("🏗️  Setting up MLflow integration for Personal Finance ML Platform")
    print("="*80)
    
    # Check Docker
    if not check_docker():
        print("❌ Docker is not running or not installed!")
        print("Please start Docker Desktop and try again.")
        return False
    
    # Start infrastructure
    if not start_infrastructure():
        print("❌ Failed to start infrastructure")
        return False
    
    # Wait for services
    if not wait_for_services():
        print("❌ Services are not ready")
        return False
    
    # Setup MinIO bucket
    if not setup_minio_bucket():
        print("❌ Failed to setup MinIO bucket")
        return False
    
    # Run ML pipeline
    if not run_ml_pipeline():
        print("❌ Failed to run ML pipeline")
        return False
    
    # Print summary
    print_summary()
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
