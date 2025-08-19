FROM apache/airflow:2.7.1

USER root

# Fix debian frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Java 11 specifically
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates-java \
        openjdk-11-jdk-headless \
        postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java

# Set JAVA_HOME and add to PATH
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Create necessary directories and set permissions
RUN mkdir -p /opt/airflow/spark_jobs/jars && \
    chown -R airflow:root /opt/airflow/spark_jobs

USER airflow

# Install Python dependencies
COPY --chown=airflow:root requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up Spark environment variables
ENV SPARK_HOME=/home/airflow/.local/lib/python3.8/site-packages/pyspark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip:${PYTHONPATH}"

# Create log directory for Spark
USER root
RUN mkdir -p /opt/airflow/logs/spark && \
    chown -R airflow:root /opt/airflow/logs

# Verify installations
USER root
RUN java -version
USER airflow
RUN spark-submit --version
