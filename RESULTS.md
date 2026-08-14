# Results Log

Running record of every measured number this project produces. Each figure cites
the file it came from. Appended to at the end of each phase.

---

## Project summary

A financial NLP system with two components sharing one repository. The first is a
sentiment/risk classifier for financial sentences: a DistilBERT model fine-tuned on
the Financial PhraseBank, benchmarked against a TF-IDF + logistic regression baseline
so the transformer's gain over classical methods is quantified rather than assumed.
The second is a retrieval-augmented generation pipeline over SEC filings, which
chunks source documents, embeds them locally with sentence-transformers, indexes them
in FAISS, retrieves against a user question, and passes the retrieved context to a
Gemini LLM to produce an answer with cited sources. Both components are served
through a FastAPI application. The RAG pipeline is evaluated on a hand-labeled
question set that measures hallucination rate with retrieval against hallucination
rate without it.

---

## Environment

| Component | Version / value | Note |
|---|---|---|
| Python | 3.13.2 | |
| torch | 2.13.0+cpu | CPU-only build; `torch.cuda.is_available()` is `False`. DistilBERT fine-tuning runs on Google Colab free GPU in Phase 3, not locally. |
| transformers | 5.15.0 | **Major version 5, not 4.x.** Most Trainer API examples online target 4.x; API signatures must be checked against the installed version. |
| scikit-learn | 1.9.0 | |
| pandas | 3.0.5 | |
| datasets | 5.0.1 | Script-based dataset loading removed in 4.0 — see Phase 1. |
| LLM provider | Google Gemini (free tier) | No OpenAI key used. |
| LLM model (pinned) | `gemini-3.5-flash` | `src/config.py` → `LLM_MODEL` |
| LLM fallback | `gemini-3.5-flash-lite` | Higher free-tier request caps; for use if rate-limited during evaluation. |
| Embedding model | `all-MiniLM-L6-v2`, 384 dimensions | Runs locally via sentence-transformers. **No API calls, no quota cost** — chosen deliberately so the corpus can be re-embedded freely while tuning chunk size. |

Source: `requirements.txt`, `src/config.py`.

---

## Phase 1 — Dataset

Source of all figures below: `data/processed/dataset_stats.json`, produced by
`src/data/load_dataset.py`.

### Dataset and config

- **Financial PhraseBank**, HF repo `takala/financial_phrasebank`, config `sentences_66agree`.
- Financial news sentences annotated by 5–8 finance-background annotators. The config
  selects the subset where **at least 66% of annotators agreed** on the label. This is
  the middle of four available agreement thresholds (50 / 66 / 75 / all). It was chosen
  as the balance point: `sentences_50agree` admits sentences that are close to
  coin-flips between annotators, while `sentences_allagree` cuts the corpus to only the
  unambiguous cases, which both shrinks the training set and makes reported scores
  flattering by removing exactly the hard examples a classifier should be judged on.
- Labels: `0 = negative`, `1 = neutral`, `2 = positive`.

### Loading workaround

The documented call `load_dataset("takala/financial_phrasebank", "sentences_66agree")`
**fails on this environment**:

```
RuntimeError: Dataset scripts are no longer supported, but found financial_phrasebank.py
```

This dataset is script-based. `datasets` 4.0 removed script execution entirely and
deleted the `trust_remote_code` flag in the same release, so no flag re-enables it. The
repo also has no auto-converted parquet branch (`refs/convert/parquet` returns 404).

`src/data/load_dataset.py` therefore attempts `load_dataset(...)` first — so it still
works on `datasets < 4.0` — and on failure downloads `data/FinancialPhraseBank-v1.0.zip`
from **the same HF repo** and parses `Sentences_66Agree.txt` directly. Same source, same
bytes; no third-party mirror was used, which preserves provenance.

The integer→name label order was **verified, not assumed**: the loader parses the
`ClassLabel(names=[...])` block out of the repo's own `financial_phrasebank.py`, which
confirms `["negative", "neutral", "positive"]`. Archive lines carry string labels
natively (`sentence@neutral`), and any label name outside the verified set raises.

### Sample counts

| | value |
|---|---|
| Total samples, raw | 4,217 |
| Total samples, after cleaning | 4,211 |
| Rows removed | 6 |

All 6 removed rows were exact duplicate texts. Zero nulls, zero empty/whitespace-only
texts, and zero rows changed by whitespace stripping — the source is clean apart from
the duplicates and the encoding issue noted below.

### Class distribution (after cleaning)

| class | count | percent |
|---|---|---|
| negative | 514 | 12.21% |
| neutral | 2,529 | 60.06% |
| positive | 1,168 | 27.74% |

The corpus is substantially imbalanced: `neutral` is ~4.9× `negative`.

### Majority-class baseline accuracy — reference floor

| measured on | accuracy |
|---|---|
| raw (4,217 rows) | **0.6011** (60.11%) |
| clean (4,211 rows) | **0.6006** (60.06%) |
| test split (632 rows) | **0.6013** (60.13%) |

This is the accuracy obtained by always predicting `neutral` and learning nothing.
**No accuracy figure anywhere in this project is meaningful without this number beside
it.** A model reporting "76% accuracy" on this dataset is 16 points above a constant
function, not 76 points above random. The test-split figure (0.6013) is the one to
quote against final test results.

### Splits

Stratified on label, 70/15/15, `random_state=42` (`src/config.py` → `RANDOM_SEED`).

| split | rows | % of total | negative | neutral | positive |
|---|---|---|---|---|---|
| train | 2,947 | 69.98% | 360 | 1,770 | 817 |
| val | 632 | 15.01% | 77 | 379 | 176 |
| test | 632 | 15.01% | 77 | 380 | 175 |

- **Stratification tolerance:** class percentages agree across all three splits to
  within **0.16 percentage points**.
- **Zero text overlap** confirmed between train/val and train/test — verified by set
  intersection on the text column of the written CSVs.
- `negative` has only **360 training examples**, the binding constraint on how well the
  minority class can be learned.

### Text length (after cleaning)

| statistic | characters | whitespace tokens |
|---|---|---|
| min | 9 | 2 |
| max | 315 | 81 |
| mean | 127.13 | 23.01 |
| median | 119.0 | 21.0 |

**DistilBERT's 512-token limit is not a constraint here.** The longest sentence in the
corpus is 81 whitespace tokens; even after subword tokenization expands that figure,
the maximum stays far below 512. No truncation strategy is needed, and `max_length` can
be set well below 512 in Phase 3 to save compute.

### Known data issue — mojibake (unfixed)

| | value |
|---|---|
| Rows containing non-ASCII characters | 102 of 4,211 (2.42%) |
| Rows matching the mojibake pattern | 86 of 4,211 (2.04%) |

Some Nordic characters in the source file are stored as `+` followed by a Latin-1 byte,
so `Lännen` appears as `L+ñnnen` and `Lemminkäinen` as `Lemmink+ñinen`.

This is **pre-existing corruption in the upstream distribution, not a decoding error
introduced by this project.** Verified at the byte level: the raw zip literally contains
`4C 65 6D 6D 69 6E 6B 2B F1` (`Lemmink+\xf1inen`), and `Sentences_66Agree.txt` is not
valid UTF-8, so the `iso-8859-1` decoding used by the dataset's own loader script is the
correct choice. The corruption is in the bytes as published.

Affected sentences are mostly Finnish company and place names (Lemminkäinen, Riihimäki,
Ålands). Impact is limited: the mangled tokens are proper nouns rather than
sentiment-bearing words. **Not yet fixed. Belongs in the README Limitations section.**

---

## Phase 2 — Baseline

Source of all figures below: `results/baseline_metrics.json`, produced by
`src/model/baseline.py`. **All metrics are on the validation split (632 rows). The test
split was not read by this script.**

### Pipeline

sklearn `Pipeline`: `TfidfVectorizer` → `LogisticRegression`.

| component | parameters |
|---|---|
| TfidfVectorizer | `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`, `strip_accents='unicode'`, `lowercase=True` |
| LogisticRegression | `max_iter=2000`, `random_state=42` |
| Vocabulary size | **9,717** unigram + bigram features |

Three variants were trained and all three are reported: a `DummyClassifier(strategy='most_frequent')`
control, and the pipeline with `class_weight=None` and `class_weight='balanced'`.

### Comparison (validation split, n=632)

| model | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| dummy (most_frequent) | 0.5997 | 0.2499 | 0.4496 |
| TF-IDF + LogReg, unweighted | 0.7753 | 0.6515 | 0.7507 |
| **TF-IDF + LogReg, balanced** | **0.7911** | **0.7326** | **0.7905** |

**Macro F1 is the headline metric.** It weights the 77-sample `negative` class equally
with the 379-sample `neutral` class, so it cannot be inflated by defaulting to the
majority class. The dummy demonstrates the gap precisely: **59.97% accuracy at 0.2499
macro F1** — near-60% "accurate" while being entirely useless.

**Class balancing won on every metric, which is the non-obvious result.** Class
weighting normally trades a little accuracy for macro-F1 gain; here it improved
accuracy (+1.58 pts), macro F1 (+8.11 pts) and weighted F1 (+3.98 pts) simultaneously.

### Confusion matrices (rows = truth, columns = prediction)

Unweighted:

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 24 | 33 | 20 |
| **true neutral** | 3 | 370 | 6 |
| **true positive** | 3 | 77 | 96 |

Balanced:

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 50 | 15 | 12 |
| **true neutral** | 20 | 330 | 29 |
| **true positive** | 13 | 43 | 120 |

The unweighted matrix shows the collapse toward `neutral`: of 632 validation rows it
predicts `neutral` 480 times. Balancing redistributes those predictions.

### Per-class metrics — balanced model

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| negative | 0.6024 | 0.6494 | 0.6250 | 77 |
| neutral | 0.8505 | 0.8707 | 0.8605 | 379 |
| positive | 0.7453 | 0.6818 | 0.7122 | 176 |

### Negative-class recall — the callout

| model | negative recall |
|---|---|
| unweighted | **0.3117** |
| balanced | **0.6494** |

The unweighted model detects fewer than one in three negative sentences, missing 53 of
77. **For a financial risk classifier this is the single worst available failure mode:**
the system's purpose is flagging adverse signal, and a model that silently reclassifies
risk as `neutral` fails precisely where it is relied upon. A missed negative is a
warning that never reaches the user; a false positive is only a wasted review. The costs
are not symmetric, so the metric must not treat them as if they were.

Balancing more than doubles negative recall (0.3117 → 0.6494) at the cost of negative
precision falling from 0.8000 to 0.6024. That is the correct direction of trade for this
application, and it is why `class_weight='balanced'` is the variant carried forward.

### Top predictive n-grams — balanced model

Highest-coefficient features per class. Read directly off the linear model, which is the
interpretability advantage the baseline holds over the transformer.

| rank | negative | positive | neutral |
|---|---|---|---|
| 1 | down (+3.6146) | increased (+2.2955) | is (+2.4492) |
| 2 | decreased (+3.1993) | rose (+2.2324) | and (+1.1183) |
| 3 | fell (+2.6654) | up (+2.1649) | includes (+1.0847) |
| 4 | lower (+1.9887) | increase (+2.0874) | approximately (+1.0492) |
| 5 | down from (+1.8220) | up from (+1.8798) | be (+1.0284) |
| 6 | decreased to (+1.7719) | rose to (+1.7694) | include (+0.9369) |
| 7 | staff (+1.7449) | grew (+1.7501) | of the (+0.9047) |
| 8 | off (+1.6742) | signed (+1.7366) | will be (+0.8422) |
| 9 | dropped (+1.5769) | our (+1.7226) | stake (+0.7672) |
| 10 | declined (+1.5465) | to (+1.5262) | shares (+0.7556) |
| 11 | lay (+1.4048) | improved (+1.4690) | not (+0.7508) |
| 12 | loss (+1.3790) | growth (+1.4199) | sell (+0.7505) |
| 13 | jobs (+1.3515) | year (+1.3847) | no (+0.7494) |
| 14 | than (+1.3445) | profit rose (+1.2720) | or (+0.7341) |
| 15 | mn in (+1.3407) | awarded (+1.2679) | value of (+0.7325) |

The model recovered directional financial vocabulary without any lexicon: a
decrease cluster (`down`, `decreased`, `fell`, `dropped`, `declined`) and a distinct
layoffs cluster (`staff`, `lay`, `jobs`, `off`) for negative; an increase cluster
(`increased`, `rose`, `grew`, `improved`, `growth`) plus contract-win language
(`signed`, `awarded`) for positive. Neutral is characterized by function words and
structural phrasing — the *absence* of directional signal rather than any signal.

**Bigrams justify `ngram_range=(1,2)` empirically rather than by convention:**
`down from` (+1.8220), `up from` (+1.8798), `rose to` (+1.7694), `decreased to`
(+1.7719) and `profit rose` (+1.2720) all rank in their class's top 15. These encode
direction-of-change relations that the unigrams alone cannot — `from` and `to` are
uninformative in isolation.

### Target for Phase 3

**The fine-tuned DistilBERT must beat macro F1 = 0.7326 on the same validation split
to justify its cost.** Secondary targets: accuracy 0.7911 and negative-class recall
0.6494. A transformer that beats accuracy while losing negative recall has not improved
on this baseline for this application.

### Stated limitations

- Single-run validation numbers. **No cross-validation**, so no error bars — the
  reported differences have no confidence interval attached.
- **No hyperparameter search.** The regularization strength `C` is at the sklearn
  default; no grid search over `C`, `min_df`, or `ngram_range` was run. The baseline is
  therefore an untuned reference point, not a best-effort classical model, and the
  DistilBERT comparison should be described that way.
- Metrics are validation-split only. The test split remains unread and is reserved for
  final evaluation.

---

## Phase 3 — Fine-tuned DistilBERT classifier

Sources: `results/distilbert_history.json` (per-epoch val history, training config),
`results/val_comparison.json` (val comparison), `results/test_metrics.json` (final
held-out test evaluation, produced by `src/model/evaluate_test.py`).

Trained on Google Colab, **Tesla T4**, via `notebooks/train_distilbert.ipynb`.

### Training configuration

| setting | value |
|---|---|
| base model | `distilbert-base-uncased` |
| max_length | 128 |
| learning_rate | 2e-05 |
| epochs | 4 |
| batch size | 16 |
| weight_decay | 0.01 |
| warmup | 0.1 (ratio of total steps) |
| fp16 | true |
| seed | 42 |
| model selection | `macro_f1`, `greater_is_better=True` |

### Computed class weights (train split only)

| class | weight |
|---|---|
| negative | 2.7287 |
| neutral | 0.5550 |
| positive | 1.2024 |

Computed with `compute_class_weight('balanced', ...)` on the 2,947 training rows
only. Deriving them from val or the full corpus would leak the evaluation
distribution into training.

### Per-epoch validation history — standard cross-entropy

| epoch | val loss | accuracy | macro F1 | negative F1 | negative recall |
|---|---|---|---|---|---|
| 1 | 0.3860 | 0.8703 | 0.8295 | 0.7612 | 0.6623 |
| **2** | **0.2919** | 0.8861 | **0.8654** | 0.8415 | 0.8961 |
| 3 | 0.3418 | 0.8892 | 0.8631 | 0.8182 | 0.8182 |
| 4 | 0.3634 | 0.8813 | 0.8589 | 0.8228 | 0.8442 |

**Selected: epoch 2** (macro F1 0.8654).

### Per-epoch validation history — class-weighted cross-entropy

| epoch | val loss | accuracy | macro F1 | negative F1 | negative recall |
|---|---|---|---|---|---|
| 1 | 0.5372 | 0.8244 | 0.7975 | 0.7591 | 0.6753 |
| 2 | **0.3136** | 0.8797 | 0.8561 | 0.8166 | 0.8961 |
| 3 | 0.3690 | 0.8861 | 0.8669 | 0.8375 | 0.8701 |
| **4** | 0.3718 | 0.8908 | **0.8739** | 0.8535 | 0.8701 |

**Selected: epoch 4** (macro F1 0.8739).

### Overfitting evidence — standard variant

Validation loss bottoms at **epoch 2 (0.2919)** and then rises monotonically to
**0.3634 by epoch 4**, while training loss falls from roughly 0.70 to 0.10. That
divergence — training loss still dropping while validation loss climbs — is
textbook overfitting, beginning after epoch 2 on a 2,947-row training set.

> Traceability note: the training-loss figures (0.70 → 0.10) are the one pair of
> numbers in this document **not** recoverable from a committed file. The
> notebook's `epoch_rows()` filter kept only log entries containing
> `eval_macro_f1`, which silently discarded every training-loss entry. They come
> from the Colab console output. The notebook now also captures a `train_logs`
> block, so a future re-run records them.

Best validation macro F1 was also epoch 2 (0.8654), and `load_best_model_at_end`
**restored epoch 2 rather than keeping epoch 4**. This is verifiable in the
history file rather than merely asserted: the final post-training evaluation row
(logged at epoch 4.0) reports `eval_macro_f1 = 0.8654119513631464` and
`eval_loss = 0.29186752438545227`, values identical to epoch 2 and different from
epoch 4's. Had selection silently fallen back to the last epoch, that row would
read 0.8589 / 0.3634.

### Divergence in the weighted variant — why selecting on loss would have been wrong

The weighted variant's validation loss also bottoms at epoch 2 (0.3136) and rises
through epochs 3 and 4 (0.3690, 0.3718). But its **macro F1 keeps climbing the
whole time**: 0.7975 → 0.8561 → 0.8669 → **0.8739**. Loss and the metric of
interest move in opposite directions over the second half of training.

These two quantities measure different things. Cross-entropy is a function of the
full predicted probability distribution, so it penalizes **miscalibrated
confidence** — a model that is right but has drifted from 0.95 to 0.75 confidence
on its correct answers accrues more loss while getting no answer wrong. Macro F1
depends only on `argmax`, so it is invisible to confidence and changes only when a
prediction actually crosses a decision boundary. In epochs 3 and 4 this model was
becoming less well-calibrated while continuing to move borderline cases onto the
correct side of the boundary.

**Had `metric_for_best_model` been left at its loss default, epoch 2 would have
been selected and 0.0178 macro F1 discarded** (0.8561 vs 0.8739). Choosing the
selection metric to match the metric that is actually reported is not a
formality.

### Final test evaluation (n = 632)

All four models, `results/test_metrics.json`. **Majority-class floor on test:
0.6013.**

| model | accuracy | macro F1 | weighted F1 | negative F1 | negative recall |
|---|---|---|---|---|---|
| dummy (most_frequent) | 0.6013 | 0.2503 | 0.4515 | 0.0000 | 0.0000 |
| TF-IDF + LogReg, balanced | 0.7975 | 0.7458 | 0.7948 | 0.6667 | 0.6234 |
| DistilBERT, standard | 0.8940 | 0.8699 | 0.8936 | 0.8313 | 0.8961 |
| **DistilBERT, weighted** | **0.8987** | **0.8787** | **0.8998** | **0.8383** | **0.9091** |

Per-class precision / recall / F1 on test:

| model | class | precision | recall | F1 | support |
|---|---|---|---|---|---|
| TF-IDF balanced | negative | 0.7164 | 0.6234 | 0.6667 | 77 |
| | neutral | 0.8375 | 0.8816 | 0.8590 | 380 |
| | positive | 0.7333 | 0.6914 | 0.7118 | 175 |
| DistilBERT standard | negative | 0.7753 | 0.8961 | 0.8313 | 77 |
| | neutral | 0.9128 | 0.9368 | 0.9247 | 380 |
| | positive | 0.9150 | 0.8000 | 0.8537 | 175 |
| DistilBERT weighted | negative | 0.7778 | 0.9091 | 0.8383 | 77 |
| | neutral | 0.9475 | 0.9026 | 0.9245 | 380 |
| | positive | 0.8611 | 0.8857 | 0.8732 | 175 |

### Confusion matrices (test, rows = truth, columns = prediction)

DistilBERT standard:

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 69 | 5 | 3 |
| **true neutral** | 14 | 356 | 10 |
| **true positive** | 6 | 29 | 140 |

DistilBERT weighted:

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 70 | 3 | 4 |
| **true neutral** | 16 | 343 | 21 |
| **true positive** | 4 | 16 | 155 |

**Directional errors — confusing positive with negative, the sign-flip failure —
total 8 of 632 for the weighted DistilBERT against 18 of 632 for TF-IDF.** The
standard variant sits at 9.

The remaining errors are overwhelmingly `neutral` confusions: of the weighted
model's 64 total errors, **56 (87.5%) involve `neutral`** on one side or the
other, and only 8 are outright sign flips. That distinction matters for a risk
system. Calling a negative sentence "neutral" understates risk and is recoverable
by a human reading the flagged document; calling a negative sentence "positive"
inverts the signal. The model almost never does the latter.

### Validation → test movement

| variant | val macro F1 | test macro F1 | change |
|---|---|---|---|
| standard | 0.8654 | 0.8699 | +0.0045 |
| weighted | 0.8739 | 0.8787 | +0.0048 |

Both variants scored **slightly higher on test than on the validation split used
for model selection**. Since selection pressure was applied to val and not to
test, a large drop would have indicated overfitting to val through checkpoint
selection. A small gain in both directions indicates the opposite: the selection
procedure did not overfit, and the two splits are drawn from the same
distribution. (The TF-IDF baseline moved the same way, 0.7326 → 0.7458.)

### Key finding — imbalance is largely a model-capacity problem

Class weighting had opposite importance for the two model families:

| model family | negative recall, unweighted | negative recall, weighted | change |
|---|---|---|---|
| TF-IDF + LogReg (val) | 0.3117 | 0.6494 | **+0.3377** |
| DistilBERT (test) | 0.8961 | 0.9091 | +0.0130 |

For the linear model, class weighting was transformative — it more than doubled
negative recall and was the difference between a model that detected fewer than
one in three negative sentences and one that detected roughly two in three. For
DistilBERT, the same intervention moved negative recall by 0.0130, because the
*unweighted* transformer already reached 0.8961 without any reweighting at all.

The conclusion is that the class imbalance was never purely a data problem to be
corrected by reweighting; it was substantially a **capacity** problem. The
bag-of-n-grams model lacked the representational power to separate the minority
class, and reweighting the loss was a way of forcing a weak model to spend its
limited capacity on the rare class — necessarily trading precision to do so
(negative precision 0.80 → 0.60 on val). A model that can actually represent the
distinction does not need the crutch: DistilBERT reaches 0.8961 negative recall
*and* 0.7753 negative precision unweighted, dominating the reweighted linear model
on both axes simultaneously.

### Caveat — weighting was not decisive for DistilBERT

The weighted variant beats the standard variant by **+0.0088 macro F1** (0.8787 vs
0.8699) on test. This is within the range of run-to-run variation from GPU
non-determinism, which the history file itself flags: cuDNN kernel selection and
non-deterministic CUDA reduction order produce differences in the third decimal
place even with all seeds fixed.

**No claim should be made that class weighting was decisive for the transformer.**
A defensible statement is that the two variants performed equivalently within
noise, with the weighted one marginally ahead on the minority class. Establishing
a real difference would require multiple seeds per variant and a comparison of
their distributions, which was not run.

### Limitations

- **The weighted variant peaked at its final epoch** (epoch 4, the last one
  trained). Its macro F1 was still rising at the point training stopped, so it may
  not have converged. More epochs might improve it, and the 4-epoch budget was
  chosen a priori rather than by observing convergence. The standard variant, by
  contrast, clearly peaked at epoch 2 and was overfitting thereafter.
- **The +0.0088 gap between variants is within GPU non-determinism** (see caveat
  above). Single run per variant, no seed replication, no confidence intervals.
- **No hyperparameter search.** Learning rate, batch size, epochs and weight decay
  were fixed a priori. As with the Phase 2 baseline, this is a reasonable single
  configuration, not a tuned optimum.
- **Checkpoint zips extracted one level nested.** `distilbert_standard.zip`
  unpacked to `distilbert_standard/distilbert_standard/`, requiring manual
  flattening before `evaluate_test.py` could load the checkpoints. Documented in
  `notebooks/README.md`.
- **The Colab upload cell's assertion was wrong.** It checked the dict returned by
  `files.upload()` rather than the filesystem. Since that return value reflects
  only the current invocation, uploading the three files across more than one call
  made the cell fail on files that were in fact present. Fixed to check
  `os.path.exists` instead.
- **Training loss is not in the committed history file** (see the traceability
  note above); the notebook has been fixed to capture it on future runs.

## Phase 4 — Document ingestion, chunking and embedding

Sources: `results/chunking_stats.json` (extraction, chunking and embedding
statistics), `data/raw/filings/manifest.json` (corpus provenance). Produced by
`src/rag/fetch_filings.py`, `src/rag/chunk.py`, `src/rag/embed.py`.

No LLM API calls anywhere in this phase. Embedding runs locally.

### Corpus

8 filings, 4 companies, FY2023 and FY2024 10-Ks, 51.54 MB raw.

| ticker | company | form | FY | filed | accession | size |
|---|---|---|---|---|---|---|
| AAPL | Apple Inc. | 10-K | 2023 | 2023-11-03 | 0000320193-23-000106 | 1.49 MB |
| AAPL | Apple Inc. | 10-K | 2024 | 2024-11-01 | 0000320193-24-000123 | 1.43 MB |
| JPM | JPMORGAN CHASE & CO | 10-K | 2023 | 2024-02-16 | 0000019617-24-000225 | 12.60 MB |
| JPM | JPMORGAN CHASE & CO | 10-K | 2024 | 2025-02-14 | 0000019617-25-000270 | 12.25 MB |
| KO | COCA COLA CO | 10-K | 2023 | 2024-02-20 | 0000021344-24-000009 | 3.97 MB |
| KO | COCA COLA CO | 10-K | 2024 | 2025-02-20 | 0000021344-25-000011 | 3.75 MB |
| MSFT | MICROSOFT CORP | 10-K | 2023 | 2023-07-27 | 0000950170-23-035122 | 9.50 MB |
| MSFT | MICROSOFT CORP | 10-K | 2024 | 2024-07-30 | 0000950170-24-087843 | 6.54 MB |

`data/raw/filings/manifest.json` is the provenance record and is committed;
the filings themselves are gitignored. It carries ticker, company, form type,
fiscal year, filing date, report date, accession number, CIK, source URL, local
filename and byte size for every document, so the corpus is reconstructable from
the repository without shipping 51 MB of HTML.

SEC access rules were honoured: a User-Agent declaring a real contact address
(EDGAR returns 403 without one) and a 0.5 s inter-request delay — 2 requests/second
against their 10/second fair-access cap.

### Two acquisition problems that had to be solved

**JPM returned zero filings on the first run.** The EDGAR submissions endpoint
caps its inline `filings.recent` block at roughly the last 1,000 submissions and
pages everything older into `filings.files`. For most companies 1,000 filings
covers many years. JPMorgan files hundreds of prospectuses and 8-Ks a year, so
`recent` reached back only a few months and did not contain the annual report at
all. The fix was to follow the `filings.files` pages and concatenate them before
filtering. **This fails silently** — the API returns HTTP 200 with a well-formed
response that simply lacks the 10-K, so the only symptom was an empty result for
one company. Any high-volume filer added to `FILING_TARGETS` later would have hit
the same wall.

**Fiscal year is keyed on `reportDate`, not `filingDate`.** A 10-K is filed after
the period it covers, sometimes months after. KO's FY2024 10-K was filed
**2025-02-20**, and JPM's FY2024 10-K on **2025-02-14**. Keying on the filing date
would have labelled both as FY2025 — producing files named `KO_10-K_2025.htm`
containing FY2024 data, and mismatching the requested `fiscal_years` so they would
not have been fetched at all. Every citation in Phase 6 inherits this field, so the
error would have propagated into user-facing output.

### Text extraction

| filing | raw | extracted chars | retained | chunks |
|---|---|---|---|---|
| AAPL_10-K_2023 | 1,558,924 B | 199,102 | 12.77% | 248 |
| AAPL_10-K_2024 | 1,503,780 B | 202,839 | 13.49% | 252 |
| JPM_10-K_2023 | 13,211,658 B | 1,214,696 | 9.19% | 1,519 |
| JPM_10-K_2024 | 12,849,180 B | 1,190,388 | 9.26% | 1,494 |
| KO_10-K_2023 | 4,161,309 B | 591,328 | 14.21% | 707 |
| KO_10-K_2024 | 3,930,907 B | 594,410 | 15.12% | 724 |
| MSFT_10-K_2023 | 9,963,591 B | 333,566 | 3.35% | 394 |
| MSFT_10-K_2024 | 6,860,911 B | 350,817 | 5.11% | 413 |

**Overall: 54,040,260 bytes → 4,677,146 characters, 8.65% retained.**

The discarded 91% is HTML markup, inline-XBRL tags and boilerplate, not lost
prose. 10-K primary documents are inline XBRL: every financial fact is wrapped in
`<ix:...>` elements, and the document opens with an `<ix:header>` block inside a
hidden `<div>` holding thousands of machine-readable facts with no readable
content. Hidden elements and XBRL headers are dropped before text extraction,
which is where the bulk of the reduction comes from.

**MSFT's 3.35% is XBRL fact density, not extraction failure.** Verified by reading
extracted chunks and by chunk count: MSFT yields **807 chunks** (394 + 413) against
AAPL's **500** (248 + 252), despite AAPL retaining four times the percentage. The
narrative content survived intact; Microsoft's filings simply carry proportionally
far more fact markup per unit of prose.

### Caught bug — CHUNK_SIZE exceeded the embedding model's hard limit

The chunk size was initially specified at **400 tokens**. `all-MiniLM-L6-v2`
truncates input at **256 word pieces** — verified directly against the loaded
model (`max_seq_length == 256`), not assumed from documentation.

Measured consequence at 400 tokens:

| | value |
|---|---|
| chunks produced | 2,932 |
| chunks over the 256-token limit | **2,743 (93.6%)** |
| corpus tokens silently discarded | **307,811 (29.4%)** |

Text past token 256 in an oversized chunk is dropped before encoding. It is
represented in **no vector** and is therefore unretrievable — while still sitting
in `chunks.json` looking perfectly intact.

**The failure mode is silence.** `sentence-transformers` raises nothing, logs
nothing, and returns a normal 384-dimensional vector for every chunk. All
downstream shapes are correct, the sanity check still passed, and a FAISS index
built on it would have worked. The symptom would have surfaced in Phase 6 or 7 as
"the RAG pipeline sometimes can't find things that are definitely in the corpus" —
an extremely expensive bug to trace backwards, and one easily mistaken for a
retrieval-tuning problem.

Corrected to **CHUNK_SIZE = 230**, leaving headroom for the `[CLS]` and `[SEP]`
tokens the tokenizer adds and the `length_function` does not count.

| after correction | value |
|---|---|
| chunks over the 256-token limit | **0 (0.0%)** |
| corpus tokens discarded | **0** |

Retrieval quality improved as a side effect — the on-topic probe mean rose from
0.6961 to 0.7397 while the random-pair noise floor fell from 0.3053 to 0.2819, so
signal-to-noise widened at both ends. Both `chunk.py` and `embed.py` now print the
over-limit count on every run.

### Chunking configuration

| setting | value |
|---|---|
| splitter | LangChain `RecursiveCharacterTextSplitter` |
| chunk_size | 230 tokens |
| chunk_overlap | 50 tokens |
| length_function | `all-MiniLM-L6-v2` tokenizer (word pieces) |
| separators | `\n\n`, `\n`, `. `, `? `, `! `, `; `, `, `, ` `, `` |

`RecursiveCharacterTextSplitter` measures **characters** by default. A token-based
`length_function` backed by the embedding model's own tokenizer was supplied
instead, so the budget is denominated in the same units as the model's 256-token
ceiling. This matters more for financial prose than for ordinary English: filings
are dense with figures, tickers, currency symbols and defined terms that tokenize
far less efficiently than plain text, so a fixed character budget produces widely
varying token counts and no reliable way to stay under the limit.

### Final chunk statistics

| statistic | tokens |
|---|---|
| total chunks | 5,751 |
| min | 2 |
| max | 230 |
| mean | 192.38 |
| median | 210 |
| p95 | 230 |

Every chunk carries **12 metadata fields**: `text`, `source_filename`, `ticker`,
`company`, `form_type`, `fiscal_year`, `accession_number`, `source_url`,
`chunk_index`, `char_start`, `char_end`, `token_count`. This is what makes Phase 6
citation possible — a retrieved chunk resolves to an exact character span in a
named EDGAR document, e.g. *JPMORGAN CHASE & CO 10-K FY2024, chunk 471, characters
752069–753495*, with the accession number and source URL attached. Without those
fields a retrieved chunk is unattributable and the "grounded answer with cited
sources" requirement cannot be met.

### Embeddings

| property | value |
|---|---|
| model | `all-MiniLM-L6-v2` (local, no API) |
| vectors | 5,751 × 384 |
| dtype | float32 |
| size on disk | 8,833,536 bytes (8.42 MB) |
| L2-normalised | yes — dot product is cosine |
| encoding time | 203.04 s (28.3 chunks/s, CPU) |
| truncated at embed time | 0 |

Embedding locally rather than through an API is a deliberate design choice: it is
what made the CHUNK_SIZE correction above cheap to act on. Re-embedding the entire
corpus cost 203 seconds and no quota, so discovering the truncation bug led
immediately to a fix rather than to a decision about whether the fix was worth
paying for.

### Sanity check — do the embeddings carry signal?

| comparison | mean cosine |
|---|---|
| random chunk pairs (noise floor) | 0.2819 |
| adjacent chunks, same document | 0.6508 |
| on-topic probe queries, top-1 | 0.7397 |
| off-topic contrast query, best match anywhere | **0.2156** |

All three checks passed: `adjacent_above_random`, `probes_above_random`,
`probes_above_contrast`.

**The strongest single result is the off-topic contrast score of 0.2156, which is
below the 0.2819 random-pair floor.** The contrast query ("recipes for baking
sourdough bread at home") shares no subject matter with any 10-K. Its best match
anywhere in 5,751 chunks scoring *below* the average similarity of two randomly
chosen chunks means the model places genuinely unrelated content further apart
than arbitrary financial text — the space discriminates by meaning rather than
assigning everything a high score.

That distinction matters because the common failure of a broken embedding setup is
a **collapsed space**: wrong pooling, an unnormalised output, or a mismatched
tokenizer produces vectors clustered so tightly that everything scores 0.9 against
everything else. Such a space passes a naive "are similar things similar?" test and
fails completely at retrieval, because ranking becomes meaningless. A high on-topic
score alone would not have ruled this out. The gap between 0.7397 on-topic and
0.2156 off-topic — a spread of 0.52 — does.

### Verified row-level alignment

Row *i* of `embeddings.npy` must correspond to row *i* of `metadata.parquet`.
Equal array lengths do not establish this: re-running `src.rag.chunk` without
re-running `src.rag.embed` leaves both files with plausible shapes and silently
wrong correspondence.

The stored text of metadata rows **0, 1500, 3000 and 5750** was re-embedded and
compared against the stored vector at the same index:

| row | cosine(stored vector, re-embedded text) |
|---|---|
| 0 | 1.000000 |
| 1500 | 1.000000 |
| 3000 | 1.000000 |
| 5750 | 1.000000 |

Exact agreement at every index proves row-level correspondence, not merely equal
counts. This is now enforced as a regression test in
`tests/test_embeddings_alignment.py` (6 tests, all passing), which additionally
includes a **negative control**: it verifies that comparing row *i*'s text against
row *i+1*'s vector scores clearly below 1.0. Without that control the alignment
assertion could pass on a shifted array if neighbouring chunks happened to embed
near-identically, and the test would provide false assurance.

### Limitations

- **Financial tables extract as unstructured token sequences.** Column structure is
  destroyed by tag stripping, so a figure ends up decoupled from its row label and
  period heading — a chunk may contain `121,649` and `62,087` and `2023` with no
  recoverable relationship between them. Chunks containing tables therefore cannot
  reliably answer figure-specific questions, and an LLM reading one could pair a
  number with the wrong year while sounding entirely confident. **Phase 7 evaluation
  questions must target qualitative and risk-factor content, not table figures.**
  Correct handling would require layout-aware parsing and is out of scope.
- **Near-identical boilerplate across fiscal years produces duplicate retrievals.**
  Risk factors are frequently copied verbatim between annual reports, so both years
  of the same filing surface together with effectively identical scores. Measured
  examples: KO FY2023 chunk 90 and KO FY2024 chunk 95 both score **0.8017** on
  "risks related to supply chain disruption and manufacturing" — occupying the top
  two slots; KO FY2024 chunk 173 and FY2023 chunk 162 both score **0.7433** on
  cybersecurity; AAPL FY2023 chunk 132 and FY2024 chunk 139 both score **0.6103** on
  revenue recognition. Top-k retrieval therefore spends part of its context budget
  on redundant text, reducing the effective diversity of evidence reaching the LLM.
- **Retrieval cannot discriminate by fiscal year.** The year exists only in
  metadata; it is not present in the embedded text. A question about FY2024
  specifically will match FY2023 and FY2024 chunks equally well, and the ranking
  between them is arbitrary with respect to the year asked about. Phase 5 or 6 will
  need metadata filtering, or the year prepended to chunk text before embedding.
- **32 chunks (0.56%) are shorter than 20 whitespace tokens**, 6 of them under 10.
  Twelve begin with a stray `.` where the splitter cut on the `". "` separator and
  left the delimiter leading the next chunk. Others are bare risk-factor headings
  such as *"Product safety and quality concerns could negatively affect our
  business."* — these match topical queries well precisely because they are clean
  topic statements, while carrying no substantive content to ground an answer in.
  They are wasted retrieval slots and worth filtering with a minimum-token threshold
  before building the index.

## Phase 5 — FAISS vector store and retrieval

*Not started.*

## Phase 6 — Retrieval and RAG answer generation

*Not started.*

## Phase 7 — RAG evaluation (hallucination rate, RAG vs no-RAG)

*Not started.*

## Phase 8 — FastAPI service

*Not started.*

## Phase 9 — Documentation and packaging

*Not started.*

---

## Resume-ready numbers

Figures suitable for a resume bullet or an interview answer. Each carries the context
required to keep it honest.

**Dataset**

- Financial PhraseBank, `sentences_66agree` config: **4,211 sentences** after
  deduplication, 3-class (negative / neutral / positive), split 70/15/15 stratified
  with zero text overlap between splits. Source: `data/processed/dataset_stats.json`.
- Class imbalance **12.21% / 60.06% / 27.74%**, with only **360 negative training
  examples** — the constraint that drives every modeling decision in the project.

**Baseline (Phase 2, validation split, n=632)**

- TF-IDF (unigram+bigram, 9,717 features) + logistic regression achieved
  **macro F1 0.7326 and accuracy 0.7911, against a majority-class floor of 0.6006** —
  i.e. **+19.1 accuracy points over always predicting `neutral`**, and 0.7326 macro F1
  against the floor's 0.2499. Source: `results/baseline_metrics.json`.
- Diagnosing class imbalance and applying `class_weight='balanced'` **raised
  negative-class recall from 0.3117 to 0.6494 — more than doubling detection of the
  minority risk class** — while also improving accuracy and macro F1, at a cost of
  negative-class precision (0.8000 → 0.6024).
- Chose macro F1 over accuracy as the headline metric after establishing that a
  most-frequent dummy classifier scores **59.97% accuracy but 0.2499 macro F1** on this
  corpus.

**Fine-tuned DistilBERT (Phase 3, held-out test split, n=632)**

- Fine-tuned DistilBERT reached **macro F1 0.8787 and accuracy 0.8987 on a
  held-out test split, against a majority-class floor of 0.6013** — i.e. **+29.7
  accuracy points over always predicting `neutral`**, and **+0.1329 macro F1 over
  the TF-IDF + logistic regression baseline** (0.7458). Source:
  `results/test_metrics.json`.
- **Negative-class (risk) recall 0.9091**, against 0.6234 for the tuned linear
  baseline and 0.0 for a majority-class dummy — a **+0.2857 improvement in
  detection of the minority risk class** over the classical model.
- **Sign-flip errors — predicting `positive` for a `negative` sentence or the
  reverse — reduced from 18/632 to 8/632** versus the TF-IDF baseline. 87.5% of
  the transformer's remaining errors involve `neutral` rather than inverting the
  signal.
- Selected checkpoints on **macro F1 rather than validation loss**; on the
  weighted variant the two diverged after epoch 2 (loss rising while macro F1
  climbed to 0.8739), and selecting on loss would have discarded 0.0178 macro F1.
- Caught overfitting on the standard variant via train/val loss divergence after
  epoch 2, and used `load_best_model_at_end` to restore epoch 2 (0.8654) rather
  than the final epoch (0.8589).
- Test scores came in **slightly above** validation for both variants (0.8739 →
  0.8787 weighted; 0.8654 → 0.8699 standard), evidence that checkpoint selection
  on val did not overfit val.
- **Framing point for interviews:** class weighting raised negative recall by
  +0.3377 for the linear model but only +0.0130 for DistilBERT, because the
  unweighted transformer already reached 0.8961. The imbalance was substantially a
  model-capacity problem, not purely a data problem. Do **not** claim weighting was
  decisive for the transformer — the +0.0088 macro F1 gap between its two variants
  is within GPU non-determinism.

**RAG pipeline (Phase 7)**

- Hallucination rate without retrieval: *TBD*.
- Hallucination rate with retrieval: *TBD*.
- Reduction in hallucination rate attributable to retrieval: *TBD*.
- Hand-labeled evaluation set size: *TBD*.
