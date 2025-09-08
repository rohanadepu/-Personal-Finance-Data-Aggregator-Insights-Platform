-- Create MLflow database if it doesn't exist
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Grant necessary permissions
\c mlflow;

-- MLflow will create its own tables, so we just need to ensure the database exists
-- and that the user has proper permissions

GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
