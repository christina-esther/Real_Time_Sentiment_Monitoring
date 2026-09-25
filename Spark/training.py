
import argparse
import json
import os
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import (
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ============================================================
# BINARY SENTIMENT ONLY
# 0 = negative
# 1 = positive
# ============================================================

LABELS = ["negative", "positive"]


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(clf):
    return Pipeline(
        stages=[
            Tokenizer(
                inputCol="clean_text",
                outputCol="words"
            ),

            HashingTF(
                inputCol="words",
                outputCol="tf",
                numFeatures=1 << 16
            ),

            IDF(
                inputCol="tf",
                outputCol="features",
                minDocFreq=3
            ),

            clf
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(pred):
    results = {}

    for metric in [
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "f1"
    ]:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName=metric
        )

        results[metric] = evaluator.evaluate(pred)

    confusion = (
        pred
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .collect()
    )

    results["confusion_matrix"] = [
        [
            int(row["label"]),
            int(row["prediction"]),
            int(row["count"])
        ]
        for row in confusion
    ]

    return results


# ============================================================
# CLASS WEIGHTS
# ============================================================

def add_class_weights(train):

    counts = {
        int(row["label"]): int(row["count"])
        for row in train.groupBy("label").count().collect()
    }

    print("\nTraining class counts:")

    for label_id, label_name in enumerate(LABELS):
        print(
            f"{label_name}: "
            f"{counts.get(label_id, 0):,}"
        )

    negative_count = counts.get(0, 1)
    positive_count = counts.get(1, 1)

    # Give slightly more weight to the smaller class
    negative_weight = (
        positive_count / negative_count
    ) ** 0.5

    positive_weight = 1.0

    print("\nClass weights:")
    print(f"negative: {negative_weight:.3f}")
    print(f"positive: {positive_weight:.3f}")

    weighted = train.withColumn(
        "classWeight",

        F.when(
            F.col("label") == 0.0,
            F.lit(float(negative_weight))
        )

        .otherwise(
            F.lit(float(positive_weight))
        )
    )

    return weighted


# ============================================================
# MAIN
# ============================================================

def main(args):

    spark = (
        SparkSession.builder
        .appName("sentiment-training-binary")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("\nLoading cleaned dataset...")

    df = spark.read.parquet(args.data)

    # ========================================================
    # OPTIONAL SAMPLING
    # ========================================================

    if args.sample < 1.0:

        print(
            f"Using {args.sample * 100:.1f}% "
            f"of the cleaned dataset..."
        )

        df = df.sample(
            withReplacement=False,
            fraction=args.sample,
            seed=42
        )

    # ========================================================
    # SELECT REQUIRED COLUMNS
    # ========================================================

    df = (
        df
        .select("clean_text", "label")
        .cache()
    )

    print("Dataset loaded.")

    # ========================================================
    # REMOVE NEUTRAL CLASS
    #
    # Original dataset:
    # 0 = negative
    # 1 = neutral
    # 2 = positive
    #
    # We keep only:
    # 0 = negative
    # 2 = positive
    # ========================================================

    print("\nOriginal class distribution:")

    original_counts = (
        df
        .groupBy("label")
        .count()
        .orderBy("label")
        .collect()
    )

    for row in original_counts:
        print(
            f"Label {int(row['label'])}: "
            f"{int(row['count']):,}"
        )

    print("\nRemoving neutral samples...")

    df = (
        df
        .filter(F.col("label") != 1)
        .withColumn(
            "label",
            F.when(
                F.col("label") == 0,
                F.lit(0.0)
            )
            .when(
                F.col("label") == 2,
                F.lit(1.0)
            )
        )
        .cache()
    )

    # ========================================================
    # SHOW FINAL BINARY DATASET
    # ========================================================

    print("\nFinal binary class distribution:")

    final_counts = (
        df
        .groupBy("label")
        .count()
        .orderBy("label")
        .collect()
    )

    for row in final_counts:

        label_id = int(row["label"])

        print(
            f"{label_id} = {LABELS[label_id]}: "
            f"{int(row['count']):,}"
        )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    train, test = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_count = train.count()
    test_count = test.count()

    print(f"\nTraining size: {train_count:,}")
    print(f"Test size: {test_count:,}")

    # ========================================================
    # ADD CLASS WEIGHTS
    # ========================================================

    train = add_class_weights(train).cache()

    print(
        f"\nFinal training size: "
        f"{train.count():,}"
    )

    print(
        f"Final test size: "
        f"{test.count():,}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "LogisticRegression":
            LogisticRegression(
                maxIter=25,
                regParam=0.01,
                elasticNetParam=0.0,
                weightCol="classWeight"
            ),

        "NaiveBayes":
            NaiveBayes(
                smoothing=1.0,
                modelType="multinomial",
                weightCol="classWeight"
            ),

        "RandomForest":
            RandomForestClassifier(
                numTrees=20,
                maxDepth=8,
                maxBins=16,
                seed=42,
                weightCol="classWeight"
            )
    }

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    results = {}

    best_model = None
    best_f1 = -1.0
    best_name = None

    for name, classifier in models.items():

        print("\n" + "=" * 60)
        print(f"Training {name}...")
        print("=" * 60)

        start_time = time.time()

        pipeline = build_pipeline(classifier)

        model = pipeline.fit(train)

        predictions = model.transform(test)

        metrics = evaluate(predictions)

        metrics["train_seconds"] = round(
            time.time() - start_time,
            1
        )

        results[name] = metrics

        print(
            json.dumps(
                {
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                indent=2
            )
        )

        # ====================================================
        # SELECT BEST MODEL USING F1
        # ====================================================

        if metrics["f1"] > best_f1:

            best_model = model
            best_f1 = metrics["f1"]
            best_name = name

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        "Models",
        exist_ok=True
    )

    best_model.write().overwrite().save(
        args.out
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_payload = {

        "best_model": best_name,

        "labels": LABELS,

        "num_classes": 2,

        "models": results,

        "trained_at":
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
    }

    with open(
        "Models/metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_payload,
            file,
            indent=2
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 60)

    print(
        f"Best model: "
        f"{best_name} "
        f"(F1={best_f1:.4f})"
    )

    print(
        f"Saved to: {args.out}"
    )

    print(
        "\nSentiment classes:"
    )

    print("0 = negative")
    print("1 = positive")

    print(
        "\nNeutral sentiment has been removed."
    )

    print("=" * 60)

    spark.stop()


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="Dataset/clean.parquet"
    )

    parser.add_argument(
        "--out",
        default="Models/best_model"
    )

    parser.add_argument(
        "--sample",
        type=float,
        default=1.0
    )

    main(
        parser.parse_args()
    )
