
FastAPI backend.

Run:  uvicorn Backend.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json
import os
import datetime as dt
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Backend.prediction_engine import PredictionEngine
from Backend import alerts, topics
from Database.mongo_client import SentimentStore


app = FastAPI(
    title="Real-Time Social Media Sentiment Monitoring API",
    version="1.0.0",
    description="Spark MLlib + Kafka + MongoDB sentiment monitoring backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


engine: Optional[PredictionEngine] = None
store: Optional[SentimentStore] = None


@app.on_event("startup")
def _startup():
    global engine, store
    engine = PredictionEngine()
    store = SentimentStore()


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        examples=["The product quality is amazing"]
    )
    source: str = "api"
    persist: bool = True


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    language: str
    hashtags: List[str]
    timestamp: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_backend": engine.backend if engine else None
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    result = engine.predict(req.text)

    if req.persist:
        import hashlib

        store.insert({
            **result,
            "message_id": hashlib.md5(
                req.text.encode()
            ).hexdigest(),
            "source": req.source
        })

    return result


@app.post("/predict/batch")
def predict_batch(texts: List[str]):
    return engine.predict_batch(texts)


@app.get("/analytics")
def analytics():
    counts = store.counts()

    total = sum(counts.values())

    pct = lambda k: (
        round(100 * counts.get(k, 0) / total, 2)
        if total else 0.0
    )

    return {
        "total_messages": total,
        "counts": counts,
        "positive_percentage": pct("positive"),
        "negative_percentage": pct("negative"),
        "generated_at": dt.datetime.utcnow().isoformat(),
    }


@app.get("/trends")
def trends(minutes: int = 120, limit: int = 15):
    docs = store.latest(2000)

    kw = topics.trending_keywords(docs, limit)

    return {
        "trending_topics": kw,

        "hashtags": [
            {
                "hashtag": h["_id"],
                "count": h["n"]
            }
            for h in store.top_hashtags(limit)
        ],

        "keyword_sentiment": topics.keyword_sentiment(
            docs,
            [k["keyword"] for k in kw[:8]]
        ),

        "sentiment_timeline": [
            {
                "minute": r["_id"]["m"],
                "sentiment": r["_id"]["s"],
                "count": r["n"]
            }
            for r in store.timeline(minutes)
        ],
    }


@app.get("/live")
def live(limit: int = 50):
    return {
        "posts": store.latest(limit)
    }


@app.get("/alerts")
def get_alerts():
    return alerts.evaluate(store)


@app.get("/model/performance")
def model_performance():
    path = os.getenv(
        "METRICS_PATH",
        "Models/metrics.json"
    )

    if not os.path.exists(path):
        raise HTTPException(
            404,
            "metrics.json not found - run Spark/training.py first"
        )

    return json.load(open(path))
