#!/bin/bash
set -e

echo "Initializing DuckDB..."
python init_duckdb.py

echo "Starting Streamlit..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
