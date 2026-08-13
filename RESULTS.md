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

*Not started.*

## Phase 4 — Document ingestion and chunking

*Not started.*

## Phase 5 — Embeddings and FAISS vector store

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

**Fine-tuned DistilBERT (Phase 3)**

- Macro F1: *TBD* — to be compared against baseline 0.7326 and floor 0.6006.
- Accuracy: *TBD* — to be compared against baseline 0.7911 and floor 0.6006.
- Negative-class recall: *TBD* — to be compared against baseline 0.6494.
- Improvement over classical baseline: *TBD*.

**RAG pipeline (Phase 7)**

- Hallucination rate without retrieval: *TBD*.
- Hallucination rate with retrieval: *TBD*.
- Reduction in hallucination rate attributable to retrieval: *TBD*.
- Hand-labeled evaluation set size: *TBD*.
