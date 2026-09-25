"""
Performance comparison: single-node pandas/Python vs distributed Apache Spark
on the same real dataset and the same cleaning steps.

Usage: python Spark/benchmark.py --rows 400000
"""
import argparse, re, time, json, os
import pandas as pd

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
PUNCT_RE = re.compile(r"[^a-z\s]")


def python_clean(s):
    s = s.lower()
    s = URL_RE.sub(" ", s); s = MENTION_RE.sub(" ", s); s = HASHTAG_RE.sub(" ", s)
    return " ".join(PUNCT_RE.sub(" ", s).split())


def run_python(path, rows):
    t0 = time.time()
    df = pd.read_csv(path, encoding="latin-1", header=None, nrows=rows,
                     names=["target", "ids", "date", "flag", "user", "text"])
    df["clean"] = df["text"].astype(str).map(python_clean)
    df = df.drop_duplicates("clean")
    n = len(df)
    dist = df["target"].value_counts().to_dict()
    return time.time() - t0, n, dist


def run_spark(path, rows):
    from pyspark.sql import SparkSession, functions as F
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import clean_text_col
    spark = SparkSession.builder.appName("bench").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    t0 = time.time()
    df = (spark.read.csv(path, header=False, inferSchema=False)
          .limit(rows)
          .withColumnRenamed("_c0", "target").withColumnRenamed("_c5", "text"))
    df = df.withColumn("clean", clean_text_col(F.col("text"))).dropDuplicates(["clean"])
    n = df.count()
    dist = {r["target"]: r["count"] for r in df.groupBy("target").count().collect()}
    el = time.time() - t0
    spark.stop()
    return el, n, dist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Dataset/twitter_sentiment.csv")
    ap.add_argument("--rows", type=int, default=400000)
    a = ap.parse_args()
    py_t, py_n, _ = run_python(a.csv, a.rows)
    sp_t, sp_n, _ = run_spark(a.csv, a.rows)
    report = {"rows": a.rows, "python_seconds": round(py_t, 2),
              "spark_seconds": round(sp_t, 2),
              "speedup": round(py_t / sp_t, 2),
              "python_rows_out": py_n, "spark_rows_out": sp_n}
    print(json.dumps(report, indent=2))
    os.makedirs("Report", exist_ok=True)
    json.dump(report, open("Report/benchmark.json", "w"), indent=2)
