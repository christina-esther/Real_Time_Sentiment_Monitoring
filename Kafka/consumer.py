"""
Lightweight Kafka consumer.

Two modes:
  --mode inspect  : print raw incoming posts (used to verify the producer in viva)
  --mode predict  : score each post with the Prediction Engine and persist to MongoDB.
                    This is the non-Spark fallback path (useful on a laptop where
                    spark-submit with the Kafka connector is heavy).
"""
import argparse, json, os, sys
from kafka import KafkaConsumer
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "social-posts")


def main(a):
    consumer = KafkaConsumer(
        a.topic,
        bootstrap_servers=BOOTSTRAP.split(","),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=a.group,
        value_deserializer=lambda v: json.loads(v.decode()))

    if a.mode == "predict":
        from Backend.prediction_engine import PredictionEngine
        from Database.mongo_client import SentimentStore
        engine, store = PredictionEngine(), SentimentStore()

    print(f"Consuming {a.topic} ...")
    for rec in consumer:
        msg = rec.value
        if a.mode == "inspect":
            print(json.dumps(msg, ensure_ascii=False)[:300])
            continue
        pred = engine.predict(msg["text"])
        doc = {**msg, **pred}
        store.insert(doc)
        print(f"{doc['sentiment']:<8} {doc['confidence']:.2f}  {doc['text'][:70]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inspect", "predict"], default="inspect")
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--group", default="sentiment-consumer")
    main(ap.parse_args())
