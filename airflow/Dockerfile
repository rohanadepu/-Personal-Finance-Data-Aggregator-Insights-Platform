FROM apache/airflow:2.7.1
USER root
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean

USER airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-apache-spark==4.1.1 \
    apache-airflow-providers-postgres==5.6.1 \
    apache-airflow-providers-amazon==8.4.0 \
    apache-airflow-provider-great-expectations==0.2.8 \
    great-expectations==0.17.19
