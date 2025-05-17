FROM apache/airflow:2.8.1-python3.11

USER root

# Install additional system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy requirements file
COPY requirements-airflow.txt /requirements-airflow.txt

# Switch to airflow user for pip installation
USER airflow

# Install additional Python packages
RUN pip install --no-cache-dir -r /requirements-airflow.txt

USER airflow
