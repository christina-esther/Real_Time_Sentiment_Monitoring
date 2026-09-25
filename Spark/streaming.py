"""
Spark Structured Streaming consumer:
Kafka topic -> clean -> Spark MLlib model -> MongoDB + predictions topic.

Binary sentiment version:
    0 = negative
    1 = positive

Usage:
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,\
org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 Spark/streaming.py
"""

import os
import json

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel

from preprocessing import clean_text_col


# ============================================================
# CONFIGURATION
# ============================================================

BOOTSTRAP = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "social-posts"
)

PRED_TOPIC = os.getenv(
    "KAFKA_PRED_TOPIC",
    "sentiment-predictions"
)

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017"
)

MONGO_DB = os.getenv(
    "MONGO_DB",
    "sentiment_monitor"
)

MONGO_COLL = os.getenv(
    "MONGO_COLLECTION",
    "sentiment_results"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "Models/best_model"
)


# ============================================================
# BINARY SENTIMENT LABELS
# ============================================================

LABELS = ["negative", "positive"]


# ============================================================
# KAFKA MESSAGE SCHEMA
# ============================================================

schema = T.StructType([
    T.StructField("message_id", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("timestamp", T.StringType()),
    T.StructField("source", T.StringType()),
    T.StructField("language", T.StringType()),
    T.StructField(
        "hashtags",
        T.ArrayType(T.StringType())
    ),
])


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Start Spark
    # --------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("sentiment-streaming")
        .config(
            "spark.mongodb.write.connection.uri",
            MONGO_URI
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # --------------------------------------------------------
    # Load trained binary Logistic Regression model
    # --------------------------------------------------------

    print("Loading binary sentiment model...")

    model = PipelineModel.load(MODEL_PATH)

    print("Model loaded successfully.")
    print("Sentiment classes: negative, positive")

    # --------------------------------------------------------
    # Read messages from Kafka
    # --------------------------------------------------------

    raw = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            BOOTSTRAP
        )
        .option(
            "subscribe",
            TOPIC
        )
        .option(
            "startingOffsets",
            "latest"
        )
        .load()
    )

    # --------------------------------------------------------
    # Parse and clean incoming posts
    # --------------------------------------------------------

    posts = (
        raw
        .select(
            F.from_json(
                F.col("value").cast("string"),
                schema
            ).alias("d")
        )
        .select("d.*")
        .filter(
            F.col("text").isNotNull()
        )
        .withColumn(
            "clean_text",
            clean_text_col(F.col("text"))
        )
    )

    # --------------------------------------------------------
    # Run sentiment prediction
    # --------------------------------------------------------

    scored = model.transform(posts)

    # --------------------------------------------------------
    # Get prediction confidence
    # --------------------------------------------------------

    max_prob = F.udf(
        lambda v: float(max(v))
        if v is not None
        else 0.0,
        T.DoubleType()
    )

    # --------------------------------------------------------
    # Convert prediction number to sentiment
    #
    # prediction 0 -> negative
    # prediction 1 -> positive
    # --------------------------------------------------------

    final = (
        scored

        .withColumn(
            "sentiment",
            F.element_at(
                F.array(
                    F.lit("negative"),
                    F.lit("positive")
                ),
                (F.col("prediction") + 1).cast("int")
            )
        )

        .withColumn(
            "confidence",
            F.round(
                max_prob(F.col("probability")),
                4
            )
        )

        .select(
            "message_id",
            "text",
            "clean_text",
            "sentiment",
            "confidence",
            "timestamp",
            "source",
            "language",
            "hashtags"
        )
    )

    # ========================================================
    # SINK 1 -> MONGODB
    # ========================================================

    mongo_q = (
        final
        .writeStream
        .format("mongodb")
        .option(
            "checkpointLocation",
            "/tmp/chk_mongo"
        )
        .option(
            "spark.mongodb.write.database",
            MONGO_DB
        )
        .option(
            "spark.mongodb.write.collection",
            MONGO_COLL
        )
        .outputMode("append")
        .start()
    )

    # ========================================================
    # SINK 2 -> KAFKA
    # Feeds the live dashboard
    # ========================================================

    kafka_q = (
        final
        .select(
            F.to_json(
                F.struct("*")
            ).alias("value")
        )
        .writeStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            BOOTSTRAP
        )
        .option(
            "topic",
            PRED_TOPIC
        )
        .option(
            "checkpointLocation",
            "/tmp/chk_kafka"
        )
        .outputMode("append")
        .start()
    )

    # ========================================================
    # SINK 3 -> CONSOLE
    # Useful for viva/demo
    # ========================================================

    (
        final
        .writeStream
        .format("console")
        .option(
            "truncate",
            False
        )
        .outputMode("append")
        .start()
    )

    # --------------------------------------------------------
    # Keep streaming application running
    # --------------------------------------------------------

    spark.streams.awaitAnyTermination()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()