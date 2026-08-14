"""Phase 3: shared logic for fine-tuning DistilBERT on financial sentiment.

Holds the pieces used by both the Colab training notebook and the local
evaluation scripts: split loading, tokenization, class weights, the weighted
Trainer subclass, and metric computation.

Written against transformers 5.15.0. Differences from the 4.x examples that
dominate online tutorials, all verified against the installed version:

  * ``evaluation_strategy`` was removed; the argument is ``eval_strategy``.
  * ``warmup_ratio`` was removed; ``warmup_steps`` now accepts a float in
    [0, 1) and interprets it as a ratio of total steps.
  * ``save_safetensors`` was removed (safetensors is now unconditional).
  * ``no_cuda`` was removed in favour of ``use_cpu``.
  * ``Trainer(tokenizer=...)`` was removed in favour of ``processing_class=``.
  * ``Trainer.compute_loss`` gained a ``num_items_in_batch`` parameter, so any
    override must accept it or Trainer calls will fail.
  * ``report_to`` now defaults to "none" rather than "all".

Trainer imports are deliberately lazy. Trainer requires ``accelerate``, which
is not installed locally, and the local machine is CPU-only and never trains -
keeping those imports inside the training functions lets this module be
imported for inference without accelerate present.

Local usage (evaluates a saved checkpoint on the VAL split):

    python -m src.model.distilbert --checkpoint models/distilbert_weighted
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from src import config

NUM_LABELS = len(config.LABEL_NAMES)


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------


def set_all_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Seed every RNG this pipeline touches.

    Note: this does NOT make GPU training fully deterministic. cuDNN kernel
    selection and non-deterministic reduction order on CUDA mean Colab runs
    will vary slightly between executions even with identical seeds. Full
    determinism would additionally require torch.use_deterministic_algorithms
    and CUBLAS_WORKSPACE_CONFIG, at a meaningful speed cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from transformers import set_seed

    set_seed(seed)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_split(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one split by name. 'test' is rejected here on purpose."""
    if name == "test":
        raise ValueError(
            "test.csv must only be read by src/model/evaluate_test.py. "
            "Use 'val' for model development."
        )
    directory = data_dir or config.DATA_PROCESSED
    return pd.read_csv(directory / f"{name}.csv")


def build_tokenizer(checkpoint: str = config.MODEL_CHECKPOINT):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint)


def tokenize_frame(frame: pd.DataFrame, tokenizer, max_length: int = config.MAX_LENGTH):
    """DataFrame -> tokenized HF Dataset.

    No padding here: DataCollatorWithPadding pads per batch at collation time,
    which is faster than padding everything to max_length when the median
    sentence is ~21 tokens against a 128 ceiling.
    """
    from datasets import Dataset

    dataset = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=max_length
        ),
        batched=True,
        remove_columns=["text"],
    )


def class_weights_from_train(train: pd.DataFrame) -> np.ndarray:
    """Balanced class weights, computed on the TRAIN split only.

    Computing these on val or on the full corpus would leak information about
    the evaluation distribution into training.
    """
    classes = np.arange(NUM_LABELS)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train["label"].to_numpy()
    )
    return weights.astype(np.float32)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def compute_metrics(eval_pred) -> dict:
    """Accuracy, macro F1, weighted F1, per-class F1, and per-class recall.

    macro_f1 is the model-selection metric: it weights the 77-row negative
    class equally with the 379-row neutral class, so it cannot be gamed by
    defaulting to the majority class. Per-class recall is included because
    negative-class recall is the failure mode that matters most here.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    labels_range = list(range(NUM_LABELS))

    per_class = f1_score(
        labels, predictions, average=None, labels=labels_range, zero_division=0
    )
    per_class_recall = recall_score(
        labels, predictions, average=None, labels=labels_range, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                labels=labels_range,
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                labels=labels_range,
                zero_division=0,
            )
        ),
    }
    for index, name in enumerate(config.LABEL_NAMES):
        metrics[f"f1_{name}"] = float(per_class[index])
        metrics[f"recall_{name}"] = float(per_class_recall[index])
    return metrics


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def build_weighted_trainer_class(weights: np.ndarray):
    """Build a Trainer subclass applying class-weighted cross-entropy.

    Returned as a factory rather than a module-level class so that importing
    this module does not require accelerate.
    """
    from transformers import Trainer

    class WeightedLossTrainer(Trainer):
        """Trainer with class-weighted cross-entropy.

        num_items_in_batch is new in transformers 5.x. It must be accepted
        even though plain cross-entropy does not use it, otherwise Trainer's
        internal call raises TypeError.
        """

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight_tensor = torch.tensor(
                weights, dtype=logits.dtype, device=logits.device
            )
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, NUM_LABELS), labels.view(-1), weight=weight_tensor
            )
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def build_training_arguments(output_dir: str | Path, **overrides):
    """TrainingArguments for transformers 5.x.

    Argument names here were verified against the installed 5.15.0 dataclass
    fields; see the module docstring for what changed from 4.x.
    """
    from transformers import TrainingArguments

    settings = dict(
        output_dir=str(output_dir),
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        eval_strategy="epoch",       # 4.x name was evaluation_strategy
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",   # not accuracy
        greater_is_better=True,
        save_total_limit=1,
        weight_decay=0.01,
        warmup_steps=0.1,            # 5.x reads a float as a ratio of total steps
        seed=config.RANDOM_SEED,
        data_seed=config.RANDOM_SEED,
        report_to="none",
    )
    settings.update(overrides)
    return TrainingArguments(**settings)


# --------------------------------------------------------------------------
# inference (plain torch - no Trainer, no accelerate)
# --------------------------------------------------------------------------


def load_finetuned(checkpoint_dir: str | Path):
    """Load a saved tokenizer + model pair for inference."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = str(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(
    texts,
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = config.MAX_LENGTH,
) -> np.ndarray:
    """Predicted label ids for a sequence of texts, batched on CPU or GPU."""
    device = next(model.parameters()).device
    texts = list(texts)
    predictions = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        predictions.append(logits.argmax(dim=-1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


# --------------------------------------------------------------------------
# local entrypoint
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned DistilBERT checkpoint on the val split."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(config.DISTILBERT_WEIGHTED_DIR),
        help="Directory containing the saved tokenizer + model.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    arguments = parser.parse_args()

    checkpoint = Path(arguments.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}.\n"
            "Train on Colab first (notebooks/train_distilbert.ipynb) and copy the "
            "downloaded folder into models/."
        )

    set_all_seeds()
    frame = load_split(arguments.split)
    tokenizer, model = load_finetuned(checkpoint)

    predictions = predict(frame["text"], tokenizer, model)
    metrics = compute_metrics((np.eye(NUM_LABELS)[predictions], frame["label"].to_numpy()))

    print(f"checkpoint : {checkpoint}")
    print(f"split      : {arguments.split} ({len(frame):,} rows)")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
