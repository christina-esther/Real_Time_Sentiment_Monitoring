"""MongoDB layer for the `sentiment_results` collection."""
import os, datetime as dt
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

SCHEMA_FIELDS = ["message_id", "text", "sentiment", "confidence",
                 "timestamp", "source", "language", "hashtags"]


class SentimentStore:
    def __init__(self, uri=None, db=None, coll=None):
        self.client = MongoClient(uri or os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.client[db or os.getenv("MONGO_DB", "sentiment_monitor")]
        self.coll = self.db[coll or os.getenv("MONGO_COLLECTION", "sentiment_results")]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.coll.create_index([("message_id", ASCENDING)], unique=True)
        self.coll.create_index([("timestamp", DESCENDING)])
        self.coll.create_index([("sentiment", ASCENDING)])
        self.coll.create_index([("hashtags", ASCENDING)])

    # ---------- writes ----------
    def insert(self, doc):
        doc = {k: doc.get(k) for k in SCHEMA_FIELDS} | {
            "ingested_at": dt.datetime.utcnow()}
        return self.coll.update_one({"message_id": doc["message_id"]},
                                    {"$setOnInsert": doc}, upsert=True)

    # ---------- reads ----------
    def latest(self, limit=50):
        return list(self.coll.find({}, {"_id": 0}).sort("ingested_at", -1).limit(limit))

    def counts(self):
        rows = self.coll.aggregate([{"$group": {"_id": "$sentiment", "n": {"$sum": 1}}}])
        return {r["_id"]: r["n"] for r in rows if r["_id"]}

    def timeline(self, minutes=120):
        since = dt.datetime.utcnow() - dt.timedelta(minutes=minutes)
        return list(self.coll.aggregate([
            {"$match": {"ingested_at": {"$gte": since}}},
            {"$group": {"_id": {"m": {"$dateToString": {
                "format": "%Y-%m-%dT%H:%M", "date": "$ingested_at"}},
                "s": "$sentiment"}, "n": {"$sum": 1}}},
            {"$sort": {"_id.m": 1}}]))

    def top_hashtags(self, limit=15):
        return list(self.coll.aggregate([
            {"$unwind": "$hashtags"},
            {"$group": {"_id": "$hashtags", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}}, {"$limit": limit}]))

    def recent_negative_ratio(self, minutes=10):
        since = dt.datetime.utcnow() - dt.timedelta(minutes=minutes)
        rows = list(self.coll.aggregate([
            {"$match": {"ingested_at": {"$gte": since}}},
            {"$group": {"_id": "$sentiment", "n": {"$sum": 1}}}]))
        total = sum(r["n"] for r in rows) or 1
        neg = sum(r["n"] for r in rows if r["_id"] == "negative")
        return neg / total, total
