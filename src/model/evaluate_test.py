"""Final held-out test evaluation for all four models.

*** THIS IS THE ONLY SCRIPT IN THE PROJECT PERMITTED TO READ test.csv. ***

Every other module works from train.csv and val.csv. The test split exists to
be read once, at the end, to produce the numbers that get reported. Reading it
during development - to compare variants, pick a threshold, or choose a
checkpoint - silently converts it into a second validation set and inflates
every figure this project claims.

Evaluates: the most-frequent dummy, the Phase 2 TF-IDF + LogReg baseline, and
both fine-tuned DistilBERT variants. Writes results/test_metrics.json.

Run from the repo root:

    python -m src.model.evaluate_test
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src import config
from src.model.distilbert import load_finetuned, predict

NEGATIVE_INDEX = config.LABEL_NAMES.index("negative")


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def score(y_true, y_pred) -> dict:
    """Metrics for one model on the test split."""
    labels = list(range(len(config.LABEL_NAMES)))
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=config.LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
            4,
        ),
        "weighted_f1": round(
            float(
                f1_score(
                    y_true, y_pred, average="weighted", labels=labels, zero_division=0
                )
            ),
            4,
        ),
        "negative_f1": round(float(per_class_f1[NEGATIVE_INDEX]), 4),
        "negative_recall": round(float(per_class_recall[NEGATIVE_INDEX]), 4),
        "per_class": {
            name: {key: round(float(value), 4) for key, value in report[name].items()}
            for name in config.LABEL_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": list(config.LABEL_NAMES),
    }


def report(name: str, y_true, y_pred) -> dict:
    """Print the full report for one model and return its metrics."""
    _rule(f"TEST RESULTS: {name}")
    labels = list(range(len(config.LABEL_NAMES)))

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

    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{n}" for n in config.LABEL_NAMES],
        columns=[f"pred_{n}" for n in config.LABEL_NAMES],
    )
    print("confusion matrix (rows = truth, cols = prediction):")
    print(matrix.to_string())

    return score(y_true, y_pred)


def distilbert_predictions(checkpoint_dir, texts) -> np.ndarray | None:
    """Predict with a fine-tuned checkpoint, or return None if it is absent."""
    if not checkpoint_dir.exists():
        print(f"  SKIP {checkpoint_dir.name}: directory not found.")
        return None
    try:
        tokenizer, model = load_finetuned(checkpoint_dir)
    except Exception as exc:
        print(f"  SKIP {checkpoint_dir.name}: could not load ({type(exc).__name__}: {exc})")
        return None
    print(f"  loaded {checkpoint_dir.name}")
    return predict(texts, tokenizer, model)


def main() -> None:
    _rule("FINAL TEST EVALUATION (test.csv read here and nowhere else)")

    train = pd.read_csv(config.DATA_PROCESSED / "train.csv")
    test = pd.read_csv(config.DATA_PROCESSED / "test.csv")
    x_test, y_test = test["text"], test["label"]
    print(f"train: {len(train):,} rows (used only to fit the dummy)")
    print(f"test : {len(test):,} rows")

    predictions: dict[str, np.ndarray] = {}

    dummy = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_SEED)
    dummy.fit(train["text"], train["label"])
    predictions["dummy_most_frequent"] = dummy.predict(x_test)

    baseline_path = config.MODELS_DIR / "baseline_balanced.joblib"
    if baseline_path.exists():
        predictions["tfidf_logreg_balanced"] = joblib.load(baseline_path).predict(x_test)
        print(f"  loaded {baseline_path.name}")
    else:
        print(f"  SKIP {baseline_path.name}: not found. Run: python -m src.model.baseline")

    for key, directory in [
        ("distilbert_standard", config.DISTILBERT_STANDARD_DIR),
        ("distilbert_weighted", config.DISTILBERT_WEIGHTED_DIR),
    ]:
        result = distilbert_predictions(directory, x_test)
        if result is not None:
            predictions[key] = result

    metrics = {name: report(name, y_test, pred) for name, pred in predictions.items()}

    _rule("FOUR-MODEL COMPARISON (test split)")
    table = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": values["accuracy"],
                "macro_F1": values["macro_f1"],
                "weighted_F1": values["weighted_f1"],
                "negative_F1": values["negative_f1"],
                "negative_recall": values["negative_recall"],
            }
            for name, values in metrics.items()
        ]
    )
    print(table.to_string(index=False))

    missing = {"distilbert_standard", "distilbert_weighted"} - set(metrics)
    if missing:
        print(f"\nNOT EVALUATED (checkpoints absent): {', '.join(sorted(missing))}")
        print("Train them on Colab, then copy the folders into models/.")

    _rule("VERDICT")
    floor = None
    stats_path = config.DATA_PROCESSED / "dataset_stats.json"
    if stats_path.exists():
        floor = json.loads(stats_path.read_text(encoding="utf-8")).get(
            "majority_class_baseline_accuracy_test_split"
        )

    winner = max(metrics, key=lambda name: metrics[name]["macro_f1"])
    best = metrics[winner]
    print(f"Best macro F1: {winner} at {best['macro_f1']:.4f}")

    dummy_macro = metrics["dummy_most_frequent"]["macro_f1"]
    print(
        f"  vs majority-class dummy   : {best['macro_f1'] - dummy_macro:+.4f} macro F1 "
        f"(dummy = {dummy_macro:.4f})"
    )
    if floor is not None:
        print(
            f"  vs majority-class floor   : {best['accuracy'] - floor:+.4f} accuracy "
            f"(floor = {floor:.4f}, i.e. always predicting 'neutral')"
        )
    if "tfidf_logreg_balanced" in metrics:
        baseline_macro = metrics["tfidf_logreg_balanced"]["macro_f1"]
        delta = best["macro_f1"] - baseline_macro
        print(
            f"  vs TF-IDF + LogReg baseline: {delta:+.4f} macro F1 "
            f"(baseline = {baseline_macro:.4f})"
        )
        if winner != "tfidf_logreg_balanced" and delta <= 0:
            print("  NOTE: the transformer did not beat the classical baseline.")

    _rule("SAVING")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "3-test-evaluation",
        "evaluated_on": "test",
        "test_size": len(test),
        "random_seed": config.RANDOM_SEED,
        "label_map": dict(enumerate(config.LABEL_NAMES)),
        "majority_class_baseline_accuracy_test_split": floor,
        "models_evaluated": sorted(metrics),
        "models_missing": sorted(missing),
        "best_by_macro_f1": winner,
        "metrics": metrics,
    }
    path = config.RESULTS_DIR / "test_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{path}")


if __name__ == "__main__":
    main()
