FROM postgres:14

RUN apt-get update \
    && apt-get install -y postgresql-14-aws-s3-fdw \
    && rm -rf /var/lib/apt/lists/*
