FROM python:3.12-slim

WORKDIR /app

RUN pip install mlflow psycopg2-binary boto3
