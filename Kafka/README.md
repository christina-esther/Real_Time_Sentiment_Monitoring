# Kafka setup, configuration and commands

## Option A — Docker (recommended)

`docker-compose.yml` at the project root starts Zookeeper, Kafka and MongoDB:

```bash
docker compose up -d
docker compose ps
```

## Option B — Local binaries

```bash
wget https://downloads.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz
tar -xzf kafka_2.13-3.7.0.tgz && cd kafka_2.13-3.7.0

bin/zookeeper-server-start.sh config/zookeeper.properties      # terminal 1
bin/kafka-server-start.sh config/server.properties             # terminal 2
```

## Create the topics

```bash
bin/kafka-topics.sh --create --topic social-posts \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

bin/kafka-topics.sh --create --topic sentiment-predictions \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

bin/kafka-topics.sh --list --bootstrap-server localhost:9092
bin/kafka-topics.sh --describe --topic social-posts --bootstrap-server localhost:9092
```

## Run the pipeline

```bash
# 1. live posts -> Kafka
python Kafka/producer.py --source reddit --subs technology+india+movies

# 2. Kafka -> Spark Structured Streaming -> MongoDB + predictions topic
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,\
org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 Spark/streaming.py

# 3. verify raw messages
python Kafka/consumer.py --mode inspect --topic social-posts

# 4. verify predictions
bin/kafka-console-consumer.sh --topic sentiment-predictions \
  --bootstrap-server localhost:9092 --from-beginning
```

## Key producer configuration used

| Property | Value | Why |
|---|---|---|
| `acks` | `all` | no post is lost if a broker dies |
| `retries` | 5 | transient broker errors are retried |
| `linger.ms` | 50 | small batching -> higher throughput |
| `key` | `message_id` | same post always lands on the same partition |
| partitions | 3 | allows 3 parallel Spark stream tasks |
