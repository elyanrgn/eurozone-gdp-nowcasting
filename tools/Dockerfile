FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy competition programs
COPY ingestion_program/ ingestion_program/
COPY scoring_program/   scoring_program/

ENV PYTHONPATH=/app/ingestion_program
