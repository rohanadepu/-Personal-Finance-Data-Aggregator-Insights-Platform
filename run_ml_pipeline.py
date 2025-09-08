"""
Simple MLflow Runner for Personal Finance ML Platform

This script runs your existing ML pipeline with proper MLflow configuration.
"""

import os
import sys
from pathlib import Path

def setup_mlflow_environment():
    """Setup MLflow environment variables for local development"""
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
    
    print("✅ MLflow environment configured for localhost")

def run_ml_pipeline():
    """Run the existing ML pipeline"""
    print("🤖 Running ML Pipeline with MLflow integration...")
    
    try:
        # Setup environment
        setup_mlflow_environment()
        
        # Add project root to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root))
        
        # Import and run the pipeline
        from ml.ml_pipeline import MLPipeline
        
        # Check if sample data exists
        data_path = "data/sample_transactions.csv"
        if not Path(data_path).exists():
            print(f"❌ Sample data not found at {data_path}")
            print("Please ensure the sample data file exists.")
            return False
        
        # Run the pipeline
        pipeline = MLPipeline(data_path=data_path)
        report = pipeline.run_full_pipeline()
        
        print("\n" + "="*50)
        print("✅ ML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Models trained: {', '.join(report['models_trained'])}")
        print("\n🔍 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        print("\n🌐 View results at: http://localhost:5000")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error running ML pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Simple MLflow ML Pipeline Runner")
    print("="*40)
    
    success = run_ml_pipeline()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print("📊 Open MLflow UI at: http://localhost:5000")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")
