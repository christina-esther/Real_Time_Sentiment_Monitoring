# Real-Time Social Media Sentiment Monitoring
### Using Apache Spark, Apache Kafka, Spark MLlib, MongoDB, FastAPI and Streamlit

An end-to-end **Big Data Analytics** system that ingests live social-media posts, classifies
their sentiment with a distributed Spark MLlib model, stores the results in MongoDB and
visualises them on a real-time dashboard.

---

## Architecture

```
         REAL SOCIAL MEDIA DATASET  (Sentiment140 + Amazon Reviews)
                        |
                        v
        Data Cleaning & NLP Processing   (Spark SQL regex pipeline, NLTK)
                        |
                        v
        Apache Spark Distributed Processing   (DataFrames, ML Pipeline)
                        |
                        v
        Spark MLlib Sentiment Classification  (LR / Naive Bayes / Random Forest)
                        |
                        v
        Kafka Real-Time Streaming Pipeline    (Reddit/Twitter API -> topic)
                        |
                        v
        Prediction Engine  (Spark Structured Streaming + FastAPI)
                        |
                        v
        MongoDB Database   (sentiment_results)
                        |
                        v
        Web Dashboard Visualization  (Streamlit + Plotly)
```

---

## Project structure

```
Real_Time_Sentiment_Monitoring/
├── Dataset/               # download instructions + real datasets (not committed)
│   └── README.md
├── Notebook/
│   └── sentiment_analysis_training.ipynb   # Colab-ready, 10 steps
├── Spark/
│   ├── preprocessing.py   # dataset loading, exploration, distributed NLP cleaning
│   ├── training.py        # TF-IDF + 3 MLlib models, comparison, model export
│   ├── streaming.py       # Structured Streaming: Kafka -> model -> MongoDB
│   └── benchmark.py       # pandas vs Spark performance comparison
├── Kafka/
│   ├── producer.py        # live Reddit / Twitter API -> Kafka topic
│   ├── consumer.py        # inspect / non-Spark prediction fallback
│   └── README.md          # setup, configuration, commands
├── Backend/
│   ├── api.py             # FastAPI: /predict /analytics /trends /live /alerts
│   ├── prediction_engine.py
│   ├── topics.py          # trending topics & keyword sentiment
│   └── alerts.py          # negative-sentiment spike alerts
├── Dashboard/
│   └── app.py             # 4-page Streamlit dashboard
├── Database/
│   ├── mongo_client.py    # collection, indexes, aggregations
│   └── schema.md
├── Models/                # best_model/, metrics.json (generated)
├── docs/ARCHITECTURE.md
├── docker-compose.yml     # Kafka + Zookeeper + MongoDB
├── requirements.txt
└── .env.example
```

---

## Quick start

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # fill in Reddit API credentials

# 1. infrastructure (Kafka + Zookeeper + MongoDB)
docker compose up -d

# 2. dataset (see Dataset/README.md)
kaggle datasets download -d kazanova/sentiment140 -p Dataset/ && unzip Dataset/sentiment140.zip -d Dataset/
mv "Dataset/training.1600000.processed.noemoticon.csv" Dataset/twitter_sentiment.csv

# 3. explore + clean with Spark
spark-submit Spark/preprocessing.py --explore
spark-submit Spark/preprocessing.py --build --out Dataset/clean.parquet

# 4. train the Spark MLlib models
spark-submit Spark/training.py --data Dataset/clean.parquet --sample 0.3

# 5. start the real-time pipeline (3 terminals)
python Kafka/producer.py --source reddit --subs technology+india+movies
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 Spark/streaming.py
uvicorn Backend.api:app --port 8000

# 6. dashboard
streamlit run Dashboard/app.py    # http://localhost:8501
```

---

## API reference

| Method | Endpoint | Returns |
|---|---|---|
| POST | `/predict` | sentiment, confidence, language, hashtags, timestamp |
| POST | `/predict/batch` | list of predictions |
| GET | `/analytics` | total messages, positive/negative/neutral % |
| GET | `/trends` | trending topics, hashtags, keyword sentiment, timeline |
| GET | `/live` | latest N scored posts |
| GET | `/alerts` | negative-sentiment spike alert state |
| GET | `/model/performance` | accuracy / precision / recall / F1 / confusion matrix |

**Example**

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" \
     -d '{"text":"The product quality is amazing"}'
```

```json
{
  "text": "The product quality is amazing",
  "sentiment": "positive",
  "confidence": 0.94,
  "language": "en",
  "hashtags": [],
  "timestamp": "2026-09-25T10:14:02"
}
```

---

## Dashboard pages

1. **Overview** — total posts, positive/negative/neutral %, pie + bar, live try-it box
2. **Live Monitoring** — streaming feed of incoming posts with sentiment + confidence
3. **Analytics** — sentiment pie, sentiment timeline, trending topics, hashtags, word cloud
4. **Model Performance** — accuracy, precision, recall, F1, model comparison, confusion matrix

---

## Advanced features

* **Multi-language support** — English, Hindi, Marathi. Devanagari is preserved through the
  cleaning regex; `langdetect` plus a Marathi marker-word check assigns `en` / `hi` / `mr`.
* **Topic analysis** — trending keywords and hashtags with per-keyword sentiment split.
* **Alert system** — `/alerts` raises `warning` / `critical` when the negative share of the
  last 10 minutes crosses a configurable threshold; the dashboard sidebar shows it live.

---

## Why Apache Spark

| Aspect | Plain Python (pandas) | Apache Spark |
|---|---|---|
| Parallelism | 1 core, 1 process | all cores / all cluster nodes |
| Memory | whole dataset in RAM | partitioned, spills to disk |
| 1.6M-row cleaning | minutes, often OOM | seconds, linear scale-out |
| Streaming | manual threads | Structured Streaming, exactly-once |
| ML | single-node scikit-learn | distributed MLlib pipelines |

Run `python Spark/benchmark.py --rows 400000` to reproduce the measured comparison; it writes
`Report/benchmark.json`.

---

## License

MIT — free to use for academic submission and portfolio.
