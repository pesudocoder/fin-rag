"""Phase 1: acquire, inspect, clean and split the Financial PhraseBank dataset.

Run from the repo root:

    python -m src.data.load_dataset

Writes train.csv / val.csv / test.csv and dataset_stats.json to data/processed/.
"""

from __future__ import annotations

import json
import re
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# Maps a HF config name to its file inside FinancialPhraseBank-v1.0.zip.
# Only needed by the archive fallback below.
CONFIG_TO_ARCHIVE_FILE = {
    "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
}

ARCHIVE_PATH_IN_REPO = "data/FinancialPhraseBank-v1.0.zip"
LOADER_SCRIPT_IN_REPO = "financial_phrasebank.py"

# The .txt files ship as Latin-1, per the dataset's own loader script.
ARCHIVE_ENCODING = "iso-8859-1"


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _label_names_from_loader_script() -> list[str]:
    """Read the ClassLabel order from the dataset's own loader script.

    The script is the authoritative definition of the int -> name mapping, so we
    parse it rather than hardcoding an assumed order. Falls back to the order
    recorded in config if the script can't be fetched or parsed.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            config.DATASET_NAME, LOADER_SCRIPT_IN_REPO, repo_type="dataset"
        )
        source = open(path, encoding="utf-8").read()
        block = re.search(r"ClassLabel\(\s*names=\[(.*?)\]", source, re.DOTALL)
        names = re.findall(r'"([^"]+)"', block.group(1))
        if names:
            print(f"Label order read from {LOADER_SCRIPT_IN_REPO}: {names}")
            return names
    except Exception as exc:
        print(f"Could not read label order from loader script ({exc!r}).")

    print(f"Falling back to config.LABEL_NAMES: {config.LABEL_NAMES}")
    return list(config.LABEL_NAMES)


def _load_via_datasets() -> pd.DataFrame:
    """Preferred path: the datasets library.

    Only works on datasets < 4.0. This dataset is script-based and has no
    parquet conversion branch, and datasets >= 4.0 removed script execution
    (along with the trust_remote_code flag), so on a current install this
    raises and the caller falls back to the archive.
    """
    from datasets import load_dataset

    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    split = dataset["train"]

    # Verify the mapping against the dataset's own features rather than assuming.
    label_names = list(split.features["label"].names)
    print(f"Label order read from dataset features: {label_names}")

    frame = split.to_pandas().rename(columns={"sentence": "text"})
    frame["label_name"] = frame["label"].map(dict(enumerate(label_names)))
    return frame[["text", "label", "label_name"]]


def _load_via_archive() -> pd.DataFrame:
    """Fallback: parse FinancialPhraseBank-v1.0.zip from the same HF repo.

    Same source and same bytes as the datasets path, just read directly. Each
    line is `sentence@label`, where label is already the string name, so no int
    mapping is assumed here; ints are assigned from the loader script's order.
    """
    from huggingface_hub import hf_hub_download

    member = CONFIG_TO_ARCHIVE_FILE[config.DATASET_CONFIG]
    archive = hf_hub_download(
        config.DATASET_NAME, ARCHIVE_PATH_IN_REPO, repo_type="dataset"
    )
    print(f"Reading {member}")

    with zipfile.ZipFile(archive) as bundle:
        raw = bundle.read(member).decode(ARCHIVE_ENCODING)

    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sentence, label_name = line.rsplit("@", 1)
        records.append((sentence, label_name.strip()))

    label_names = _label_names_from_loader_script()
    name_to_id = {name: index for index, name in enumerate(label_names)}

    frame = pd.DataFrame(records, columns=["text", "label_name"])
    unexpected = set(frame["label_name"]) - set(name_to_id)
    if unexpected:
        raise ValueError(f"Unexpected label names in archive: {sorted(unexpected)}")

    frame["label"] = frame["label_name"].map(name_to_id)
    return frame[["text", "label", "label_name"]]


def load_raw() -> pd.DataFrame:
    """Load the raw dataset as a DataFrame of text / label / label_name."""
    _rule(f"LOADING {config.DATASET_NAME} [{config.DATASET_CONFIG}]")
    try:
        return _load_via_datasets()
    except Exception as exc:
        print(f"datasets loader unavailable: {type(exc).__name__}: {exc}")
        print("Falling back to the raw archive in the same HF repo.\n")
        return _load_via_archive()


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and percentages, ordered by label id."""
    counts = frame["label_name"].value_counts()
    table = pd.DataFrame(
        {"count": counts, "percent": (counts / len(frame) * 100).round(2)}
    )
    order = [name for name in config.LABEL_NAMES if name in table.index]
    return table.loc[order]


def majority_baseline_accuracy(frame: pd.DataFrame) -> tuple[str, float]:
    """Accuracy of always predicting the most common class."""
    counts = frame["label_name"].value_counts()
    return counts.index[0], counts.iloc[0] / len(frame)


def run_eda(frame: pd.DataFrame) -> None:
    _rule("EDA REPORT (raw)")

    print(f"Total samples: {len(frame):,}\n")

    print("Class distribution:")
    print(class_distribution(frame).to_string())

    majority_class, baseline = majority_baseline_accuracy(frame)
    print(
        f"\nMAJORITY-CLASS BASELINE ACCURACY: {baseline:.4f} "
        f"({baseline * 100:.2f}%) - always predicting '{majority_class}'"
    )
    print("Any classifier must beat this number to be worth anything.\n")

    text = frame["text"]
    print(f"Duplicate texts (rows beyond the first occurrence): {text.duplicated().sum():,}")
    print(f"Unique texts: {text.nunique():,}")

    nulls = text.isna().sum()
    empties = (text.fillna("").str.strip() == "").sum()
    print(f"Null texts: {nulls:,}")
    print(f"Empty / whitespace-only texts: {empties:,}\n")

    chars = text.fillna("").str.len()
    tokens = text.fillna("").str.split().str.len()
    stats = pd.DataFrame(
        {
            "characters": [chars.min(), chars.max(), chars.mean(), chars.median()],
            "whitespace_tokens": [
                tokens.min(),
                tokens.max(),
                tokens.mean(),
                tokens.median(),
            ],
        },
        index=["min", "max", "mean", "median"],
    ).round(2)
    print("Text length:")
    print(stats.to_string())

    print("\n5 random samples:")
    sample = frame.sample(5, random_state=config.RANDOM_SEED)
    for _, row in sample.iterrows():
        print(f"  [{row['label']} {row['label_name']:>8}] {row['text']}")


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, drop nulls/empties, drop exact duplicate texts.

    Stripping runs first so that texts differing only by surrounding whitespace
    are caught by the duplicate and empty checks.
    """
    _rule("CLEANING")
    start = len(frame)
    cleaned = frame.copy()

    stripped = cleaned["text"].fillna("").str.strip()
    n_whitespace_changed = int((stripped != cleaned["text"].fillna("")).sum())
    cleaned["text"] = stripped.where(cleaned["text"].notna())
    print(f"1. Stripped leading/trailing whitespace: {n_whitespace_changed:,} rows changed")

    before = len(cleaned)
    cleaned = cleaned[cleaned["text"].notna() & (cleaned["text"] != "")]
    print(f"2. Dropped null / empty texts: {before - len(cleaned):,} rows removed")

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset="text", keep="first")
    print(f"3. Dropped exact duplicate texts: {before - len(cleaned):,} rows removed")

    cleaned = cleaned.reset_index(drop=True)
    removed = start - len(cleaned)
    print(f"\nTotal: {start:,} -> {len(cleaned):,} rows ({removed:,} removed)")

    print("\nClass distribution after cleaning:")
    print(class_distribution(cleaned).to_string())
    majority_class, baseline = majority_baseline_accuracy(cleaned)
    print(
        f"\nMajority-class baseline after cleaning: {baseline:.4f} "
        f"({baseline * 100:.2f}%) - '{majority_class}'"
    )
    return cleaned


def split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split."""
    _rule("SPLITTING (70/15/15, stratified on label)")

    holdout_ratio = config.VAL_RATIO + config.TEST_RATIO
    train_df, holdout_df = train_test_split(
        frame,
        test_size=holdout_ratio,
        stratify=frame["label"],
        random_state=config.RANDOM_SEED,
    )
    # Halve the holdout into val/test; expressed as a fraction of the holdout.
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=config.TEST_RATIO / holdout_ratio,
        stratify=holdout_df["label"],
        random_state=config.RANDOM_SEED,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    total = len(frame)
    for name, part in splits.items():
        print(f"\n{name}: {len(part):,} rows ({len(part) / total * 100:.2f}% of total)")
        print(class_distribution(part).to_string())

    print("\nPercentages should match across splits if stratification worked.")
    return splits


def save(splits: dict[str, pd.DataFrame], raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    _rule("SAVING")
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    for name, part in splits.items():
        path = config.DATA_PROCESSED / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8")
        print(f"{path}  ({len(part):,} rows)")

    majority_class, baseline = majority_baseline_accuracy(cleaned)
    _, test_baseline = majority_baseline_accuracy(splits["test"])
    distribution = class_distribution(cleaned)

    chars = cleaned["text"].str.len()
    tokens = cleaned["text"].str.split().str.len()

    # The upstream Sentences_*.txt files carry pre-existing corruption: some
    # Nordic characters are stored as '+' followed by a Latin-1 byte
    # (e.g. 'L+\xf1nnen' for 'Lannen'). Counted here so the scale is on record.
    non_ascii = cleaned["text"].str.contains(r"[^\x00-\x7F]", regex=True)
    mojibake = cleaned["text"].str.contains(r"\+[^\x00-\x7F]", regex=True)

    stats = {
        "dataset": config.DATASET_NAME,
        "config": config.DATASET_CONFIG,
        "random_seed": config.RANDOM_SEED,
        "label_map": dict(enumerate(config.LABEL_NAMES)),
        "total_samples_raw": len(raw),
        "total_samples_clean": len(cleaned),
        "rows_removed_by_cleaning": len(raw) - len(cleaned),
        "class_counts": distribution["count"].astype(int).to_dict(),
        "class_percentages": distribution["percent"].to_dict(),
        "majority_class": majority_class,
        "majority_class_baseline_accuracy": round(baseline, 4),
        "majority_class_baseline_accuracy_test_split": round(test_baseline, 4),
        "split_ratios": {
            "train": config.TRAIN_RATIO,
            "val": config.VAL_RATIO,
            "test": config.TEST_RATIO,
        },
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "split_class_counts": {
            name: class_distribution(part)["count"].astype(int).to_dict()
            for name, part in splits.items()
        },
        "text_length_clean": {
            "characters": {
                "min": int(chars.min()),
                "max": int(chars.max()),
                "mean": round(float(chars.mean()), 2),
                "median": float(chars.median()),
            },
            "whitespace_tokens": {
                "min": int(tokens.min()),
                "max": int(tokens.max()),
                "mean": round(float(tokens.mean()), 2),
                "median": float(tokens.median()),
            },
        },
        "data_quality": {
            "rows_with_non_ascii": int(non_ascii.sum()),
            "rows_with_mojibake_pattern": int(mojibake.sum()),
            "mojibake_note": (
                "Upstream corruption in Sentences_66Agree.txt: some Nordic "
                "characters stored as '+' plus a Latin-1 byte. Not fixed."
            ),
        },
    }

    stats_path = config.DATA_PROCESSED / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"{stats_path}")


def main() -> None:
    raw = load_raw()
    run_eda(raw)
    cleaned = clean(raw)
    splits = split(cleaned)
    save(splits, raw, cleaned)
    _rule("DONE")


if __name__ == "__main__":
    main()
