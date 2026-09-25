"""
Prediction Engine.

Loads the trained Spark MLlib PipelineModel once and serves single-text
predictions. Falls back to a scikit-learn model when Spark is unavailable.

This version uses only two sentiment classes:
    0 = negative
    1 = positive
"""

import datetime as dt
import os
import re

# Binary sentiment labels only
LABELS = ["negative", "positive"]

MODEL_PATH = os.getenv("MODEL_PATH", "Models/best_model")
SK_PATH = os.getenv("SK_MODEL_PATH", "Models/sklearn_model.joblib")

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
PUNCT_RE = re.compile(r"[^a-z\u0900-\u097F\s]")


def clean(text: str) -> str:
    """Clean input text before prediction."""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        from langdetect import detect
        lang = detect(text)
    except Exception:
        lang = "unknown"

    # langdetect may identify Marathi as Hindi.
    if lang == "hi" and re.search(
        r"(आहे|नाही|मला|तुम्ही|आणि)",
        text
    ):
        return "mr"

    return lang


class PredictionEngine:
    def __init__(self):
        self.backend = None
        self._spark = None
        self._model = None
        self._load()

    def _load(self):
        """Load the trained Spark model."""
        try:
            from pyspark.sql import SparkSession
            from pyspark.ml import PipelineModel

            self._spark = (
                SparkSession.builder
                .appName("prediction-engine")
                .master(os.getenv("SPARK_MASTER", "local[2]"))
                .getOrCreate()
            )

            self._spark.sparkContext.setLogLevel("ERROR")

            self._model = PipelineModel.load(MODEL_PATH)

            self.backend = "spark-mllib"

            print("[engine] Spark MLlib binary model loaded successfully.")
            print("[engine] Labels: negative, positive")

            return

        except Exception as e:
            print(
                f"[engine] Spark model unavailable ({e}); "
                "trying sklearn export"
            )

        # Fallback to scikit-learn model
        try:
            import joblib

            self._model = joblib.load(SK_PATH)
            self.backend = "sklearn"

            print("[engine] Scikit-learn model loaded successfully.")

        except Exception as e:
            raise RuntimeError(
                "No model found. Train first: "
                "python Spark/training.py"
            ) from e

    # ---------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """Predict Positive or Negative sentiment for one text."""

        ct = clean(text)

        if self.backend == "spark-mllib":

            df = self._spark.createDataFrame(
                [(ct,)],
                ["clean_text"]
            )

            row = (
                self._model
                .transform(df)
                .select("prediction", "probability")
                .head()
            )

            idx = int(row["prediction"])
            conf = float(max(row["probability"]))

        else:

            proba = self._model.predict_proba([ct])[0]

            idx = int(proba.argmax())
            conf = float(proba[idx])

        # Safety check: binary model must only return 0 or 1
        if idx not in (0, 1):
            idx = 0

        return {
            "text": text,
            "clean_text": ct,
            "sentiment": LABELS[idx],
            "confidence": round(conf, 4),
            "language": detect_language(text),
            "hashtags": HASHTAG_RE.findall(text),
            "model": self.backend,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }

    def predict_batch(self, texts):
        """Predict sentiment for multiple texts."""
        return [self.predict(t) for t in texts]