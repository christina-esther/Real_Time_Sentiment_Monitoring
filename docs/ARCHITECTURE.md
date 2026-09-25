# System Architecture

## Layers

### 1. Data layer
* **Batch:** Sentiment140 (1.6M tweets) + Amazon Electronics reviews. Loaded as Spark
  DataFrames with an explicit schema (no `inferSchema` — avoids a full extra pass).
* **Streaming:** Reddit API comment stream (PRAW) or Twitter/X filtered stream (Tweepy).

### 2. Ingestion layer — Apache Kafka
* Topic `social-posts` (3 partitions, key = `message_id`) carries raw live posts.
* Topic `sentiment-predictions` carries scored posts for any downstream consumer.
* Producer uses `acks=all`, `retries=5`, `linger.ms=50`.

### 3. Processing layer — Apache Spark
* **Batch path:** `preprocessing.py` → `clean.parquet` → `training.py`.
* **Streaming path:** `streaming.py` reads Kafka with Structured Streaming, applies the same
  cleaning expressions, transforms with the saved `PipelineModel`, and writes to three sinks
  (MongoDB, Kafka, console).
* Checkpointing (`/tmp/chk_*`) gives replay and at-least-once delivery; the unique
  `message_id` index in MongoDB makes the effect idempotent (exactly-once semantics).

### 4. ML layer — Spark MLlib
`Tokenizer → HashingTF(2^18) → IDF(minDocFreq=3) → classifier`, with Logistic Regression,
Naive Bayes and Random Forest compared on accuracy / precision / recall / F1. The best model
is persisted with `PipelineModel.save()` so that training and serving use identical
transformations — no train/serve skew.

### 5. Storage layer — MongoDB
Collection `sentiment_results`; schema and indexes in `Database/schema.md`. Aggregation
pipelines power the analytics, timeline and trending-hashtag endpoints.

### 6. Service layer — FastAPI
Loads the model once at startup, exposes `/predict`, `/analytics`, `/trends`, `/live`,
`/alerts`, `/model/performance`; CORS-enabled so any frontend can consume it.

### 7. Presentation layer — Streamlit + Plotly
Four dashboard pages polling the API, with an auto-refreshing live feed.

## Data flow (end-to-end latency budget)

| Stage | Typical latency |
|---|---|
| Reddit API → producer | 0.3–1 s |
| Producer → Kafka | < 50 ms |
| Kafka → Spark micro-batch | 1–3 s (trigger default) |
| Model inference (batch of N) | < 200 ms |
| Mongo write | < 20 ms |
| Dashboard refresh | 5 s poll |
| **End-to-end** | **≈ 3–6 s** |

## Fault tolerance
* Kafka retains messages (default 7 days) → a Spark restart replays from checkpoint.
* MongoDB unique index → duplicate replays do not double-count.
* Producer retries → transient broker failures do not lose posts.
