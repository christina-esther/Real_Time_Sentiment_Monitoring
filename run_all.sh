#!/usr/bin/env bash
# One-shot launcher for the viva demo.
set -e
docker compose up -d                      # Kafka + Zookeeper + MongoDB
sleep 15
uvicorn Backend.api:app --port 8000 &     # backend
streamlit run Dashboard/app.py &          # dashboard
python Kafka/producer.py --source reddit & # live data
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 Spark/streaming.py
