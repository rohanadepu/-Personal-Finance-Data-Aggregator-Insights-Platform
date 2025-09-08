FROM ghcr.io/mlflow/mlflow:v2.7.1

# Install PostgreSQL drivers and additional dependencies
RUN pip install psycopg2-binary boto3 pymysql

# Ensure MLflow uses the correct artifact root
ENV MLFLOW_BACKEND_STORE_URI=""
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=""

# Set the working directory
WORKDIR /mlflow

# Expose port 5000
EXPOSE 5000

# Default command - will be overridden by docker-compose
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
