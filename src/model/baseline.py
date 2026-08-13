"""Phase 2: TF-IDF + Logistic Regression baseline for financial sentiment.

Run from the repo root:

    python -m src.model.baseline

Trains an unweighted and a class-balanced variant, compares both against a
most-frequent dummy, and evaluates everything on the VAL split. The test split
is deliberately never read here - it stays untouched until final evaluation.

Writes fitted pipelines to models/ and metrics to results/baseline_metrics.json.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src import config

TOP_N_FEATURES = 15


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val. test.csv is intentionally not read in this module."""
    train = pd.read_csv(config.DATA_PROCESSED / "train.csv")
    val = pd.read_csv(config.DATA_PROCESSED / "val.csv")
    print(f"train: {len(train):,} rows")
    print(f"val:   {len(val):,} rows")
    return train, val


def build_pipeline(class_weight: str | None) -> Pipeline:
    """TF-IDF (uni+bigram) -> multinomial logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=config.RANDOM_SEED,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def evaluate(name: str, y_true, y_pred) -> dict:
    """Print a full report for one model and return its metrics."""
    labels = list(range(len(config.LABEL_NAMES)))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )

    _rule(f"VAL RESULTS: {name}")
    print(f"accuracy    : {accuracy:.4f}")
    print(f"macro F1    : {macro_f1:.4f}   <-- HEADLINE METRIC")
    print(f"weighted F1 : {weighted_f1:.4f}")

    print("\nclassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=config.LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in config.LABEL_NAMES],
        columns=[f"pred_{name}" for name in config.LABEL_NAMES],
    )
    print("confusion matrix (rows = truth, cols = prediction):")
    print(frame.to_string())

    per_class = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=config.LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {
            name: {
                key: round(float(value), 4)
                for key, value in per_class[name].items()
            }
            for name in config.LABEL_NAMES
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(config.LABEL_NAMES),
    }


def top_features(pipeline: Pipeline, top_n: int = TOP_N_FEATURES) -> dict[str, list]:
    """Highest-coefficient n-grams per class, straight off the linear model."""
    _rule(f"TOP {top_n} PREDICTIVE N-GRAMS PER CLASS (class_weight='balanced')")

    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefficients = pipeline.named_steps["clf"].coef_
    result: dict[str, list] = {}

    for index, class_name in enumerate(config.LABEL_NAMES):
        weights = coefficients[index]
        ranked = weights.argsort()[::-1][:top_n]
        rows = [(vocabulary[i], round(float(weights[i]), 4)) for i in ranked]
        result[class_name] = [{"ngram": gram, "coef": coef} for gram, coef in rows]

        print(f"\n{class_name}:")
        for rank, (gram, coef) in enumerate(rows, start=1):
            print(f"  {rank:>2}. {gram:<28} {coef:+.4f}")

    return result


def summary_table(metrics: dict[str, dict]) -> None:
    _rule("SUMMARY (val split)")
    table = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": values["accuracy"],
                "macro_F1": values["macro_f1"],
                "weighted_F1": values["weighted_f1"],
            }
            for name, values in metrics.items()
        ]
    )
    print(table.to_string(index=False))
    print("\nmacro_F1 is the headline metric: it weights the small negative")
    print("class equally, so it does not reward simply predicting 'neutral'.")


def main() -> None:
    _rule("LOADING SPLITS (train + val only; test.csv untouched)")
    train, val = load_splits()

    x_train, y_train = train["text"], train["label"]
    x_val, y_val = val["text"], val["label"]

    _rule("TRAINING")
    dummy = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_SEED)
    dummy.fit(x_train, y_train)
    print("fitted: dummy (most_frequent)")

    unweighted = build_pipeline(class_weight=None)
    unweighted.fit(x_train, y_train)
    print("fitted: tfidf_logreg (class_weight=None)")

    balanced = build_pipeline(class_weight="balanced")
    balanced.fit(x_train, y_train)
    print("fitted: tfidf_logreg (class_weight='balanced')")

    vocabulary_size = len(unweighted.named_steps["tfidf"].get_feature_names_out())
    print(f"\nTF-IDF vocabulary (unigrams + bigrams, min_df=2): {vocabulary_size:,}")

    metrics = {
        "dummy_most_frequent": evaluate("dummy (most_frequent)", y_val, dummy.predict(x_val)),
        "tfidf_logreg_unweighted": evaluate(
            "TF-IDF + LogReg (class_weight=None)", y_val, unweighted.predict(x_val)
        ),
        "tfidf_logreg_balanced": evaluate(
            "TF-IDF + LogReg (class_weight='balanced')", y_val, balanced.predict(x_val)
        ),
    }

    features = top_features(balanced)
    summary_table(metrics)

    _rule("SAVING")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, pipeline in [
        ("baseline_unweighted.joblib", unweighted),
        ("baseline_balanced.joblib", balanced),
    ]:
        path = config.MODELS_DIR / filename
        joblib.dump(pipeline, path)
        print(f"{path}")

    payload = {
        "phase": "2-baseline",
        "evaluated_on": "val",
        "random_seed": config.RANDOM_SEED,
        "label_map": dict(enumerate(config.LABEL_NAMES)),
        "train_size": len(train),
        "val_size": len(val),
        "tfidf_vocabulary_size": vocabulary_size,
        "vectorizer_params": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "lowercase": True,
        },
        "classifier_params": {"max_iter": 2000, "random_state": config.RANDOM_SEED},
        "metrics": metrics,
        "top_features_balanced": features,
    }
    results_path = config.RESULTS_DIR / "baseline_metrics.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{results_path}")

    _rule("DONE")


if __name__ == "__main__":
    main()
