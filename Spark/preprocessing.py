"""
Spark preprocessing + NLP cleaning for the real datasets.

Usage:
    spark-submit Spark/preprocessing.py --explore
    spark-submit Spark/preprocessing.py --build --out Dataset/clean.parquet
"""
import argparse, os, re, sys, time

from pyspark.sql import SparkSession, functions as F, types as T

TWITTER_CSV = os.getenv("TWITTER_CSV", "Dataset/twitter_sentiment.csv")
AMAZON_JSON = os.getenv("AMAZON_JSON", "Dataset/reviews_Electronics_5.json")

STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he her here hers herself him himself
his how i i'm i've if in into is isn't it it's its itself let's me more most my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she should shouldn't so some such than that the
their theirs them themselves then there these they this those through to too under until up very was wasn't we
were weren't what when where which while who whom why with won't would wouldn't you your yours yourself""".split())

URL_RE = r"http\S+|www\.\S+"
MENTION_RE = r"@\w+"
HASHTAG_RE = r"#\w+"
PUNCT_RE = r"[^a-z\u0900-\u097F\s]"   # keep latin + devanagari (Hindi/Marathi)


def get_spark(app="sentiment-preprocessing"):
    return (SparkSession.builder
            .appName(app)
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.driver.memory", "4g")
            .getOrCreate())


def clean_text_col(col):
    """Pure-Spark NLP cleaning (no UDF -> stays distributed and fast)."""
    c = F.lower(col)
    c = F.regexp_replace(c, URL_RE, " ")
    c = F.regexp_replace(c, MENTION_RE, " ")
    c = F.regexp_replace(c, HASHTAG_RE, " ")
    c = F.regexp_replace(c, PUNCT_RE, " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def load_twitter(spark):
    schema = T.StructType([
        T.StructField("target", T.StringType()),
        T.StructField("ids", T.StringType()),
        T.StructField("date", T.StringType()),
        T.StructField("flag", T.StringType()),
        T.StructField("user", T.StringType()),
        T.StructField("text", T.StringType()),
    ])
    df = spark.read.csv(TWITTER_CSV, schema=schema, header=False, quote='"', escape='"')
    return (df.withColumn("label_str",
                          F.when(F.col("target") == "0", "negative")
                           .when(F.col("target") == "4", "positive")
                           .otherwise("neutral"))
              .withColumn("source", F.lit("twitter"))
              .select("text", "label_str", "source",
                      F.col("date").alias("timestamp")))


def load_amazon(spark):
    if not os.path.exists(AMAZON_JSON):
        return None
    df = spark.read.json(AMAZON_JSON)
    txt = F.col("reviewText") if "reviewText" in df.columns else F.col("text")
    return (df.withColumn("label_str",
                          F.when(F.col("overall") <= 2, "negative")
                           .when(F.col("overall") == 3, "neutral")
                           .otherwise("positive"))
              .withColumn("source", F.lit("amazon"))
              .select(txt.alias("text"), "label_str", "source",
                      F.col("unixReviewTime").cast("string").alias("timestamp")))


def union_all(spark):
    dfs = [load_twitter(spark)]
    amz = load_amazon(spark)
    if amz is not None:
        dfs.append(amz)
    out = dfs[0]
    for d in dfs[1:]:
        out = out.unionByName(d)
    return out


def explore(df, path_hint=TWITTER_CSV):
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)
    n = df.count()
    print(f"Total records          : {n:,}")
    if os.path.exists(path_hint):
        print(f"On-disk size (primary) : {os.path.getsize(path_hint)/1024/1024:.1f} MB")
    print("\nSentiment distribution:")
    df.groupBy("label_str").count().orderBy("count", ascending=False).show()
    print("Missing / empty text:",
          df.filter(F.col("text").isNull() | (F.trim(F.col("text")) == "")).count())
    print("Duplicate texts     :", n - df.select("text").distinct().count())
    print("\nSample records:")
    df.show(5, truncate=80)


def build(spark, out):
    raw = union_all(spark)
    t0 = time.time()
    clean = (raw
             .filter(F.col("text").isNotNull())
             .withColumn("clean_text", clean_text_col(F.col("text")))
             .withColumn("tokens", F.split(F.col("clean_text"), " "))
             .withColumn("tokens", F.expr(
                 "filter(tokens, x -> length(x) > 2 AND NOT array_contains(array({}), x))".format(
                     ",".join("'%s'" % w.replace("'", "\\'") for w in sorted(STOPWORDS) if "'" not in w))))
             .withColumn("clean_text", F.concat_ws(" ", F.col("tokens")))
             .filter(F.size("tokens") >= 2)
             .dropDuplicates(["clean_text"])
             .withColumn("label", F.when(F.col("label_str") == "negative", 0.0)
                                   .when(F.col("label_str") == "neutral", 1.0)
                                   .otherwise(2.0)))
    clean.write.mode("overwrite").parquet(out)
    print(f"Wrote cleaned dataset to {out} in {time.time()-t0:.1f}s")
    clean.groupBy("label_str").count().show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default="Dataset/clean.parquet")
    a = ap.parse_args()
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    if a.explore:
        explore(union_all(spark))
    if a.build:
        build(spark, a.out)
    spark.stop()
