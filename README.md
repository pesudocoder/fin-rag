# Financial Document RAG + Risk Classification

Two NLP systems over financial text: a fine-tuned risk/sentiment classifier
benchmarked against a classical baseline, and a retrieval-augmented QA pipeline
over SEC filings measured against a no-retrieval control.

**The problem.** Language models answer financial questions fluently whether or
not they have a source, and a fabricated figure is indistinguishable from a real
one to anyone who has not read the filing. This project quantifies both halves:
how much a fine-tuned transformer beats a classical baseline at flagging negative
financial sentiment, and how much retrieval actually reduces fabrication when the
answer is not in the corpus.

Full measurement log, phase by phase: **[RESULTS.md](RESULTS.md)**.

---

## Results

### Risk classifier — held-out test split (n = 632)

| model | accuracy | macro F1 | negative recall |
|---|---|---|---|
| majority-class dummy | 0.6013 | 0.2503 | 0.0000 |
| TF-IDF + logistic regression | 0.7975 | 0.7458 | 0.6234 |
| DistilBERT, standard | 0.8940 | 0.8699 | 0.8961 |
| **DistilBERT, class-weighted** | **0.8987** | **0.8787** | **0.9091** |

The **majority-class floor is 0.6013** — always predicting `neutral`. No accuracy
figure here means anything without it. Macro F1 is the headline metric because it
weights the 12%-of-corpus `negative` class equally; negative recall is reported
separately because missing adverse signal is the failure mode that matters for a
risk classifier.

### RAG — fabrication rate, with retrieval vs without

Same model, same temperature, same question wording. The only difference is
whether retrieved filing text is supplied.

| arm | fabrications | absent questions refused |
|---|---|---|
| no retrieval | **11/16** | 1/8 |
| with retrieval | **3/16** | **8/8** |

Measured on 16 questions whose answers are genuinely absent from the corpus or
which are designed to elicit invention. All three surviving RAG fabrications are
requests for figures from financial tables — a documented limitation, below.

---

## Architecture

```
  CLASSIFIER                             RAG PIPELINE

  Financial PhraseBank                   SEC EDGAR: 8 x 10-K, 4 companies
  4,211 sentences                                │
        │                                        ▼  HTML + inline-XBRL strip
  stratified 70/15/15                     8.65% of bytes survive as prose
  2947 / 632 / 632                               │
        │                                        ▼  recursive split,
        ├─► TF-IDF + LogReg                 230 tokens / 50 overlap
        │   9,717 n-grams                          │
        │                                          ▼  all-MiniLM-L6-v2
        └─► DistilBERT                        5,751 x 384, local, no API
            Colab T4, 4 epochs, lr 2e-5            │
                    │                              ▼
                    ▼                       FAISS IndexFlatIP (exact cosine)
          macro F1 on held-out                     │
          test split                               ▼  top-k + filter + dedupe
                                            Gemini + numbered excerpts
                                                   │
                                                   ▼
                                            answer + resolved citations
                                            (company, FY, accession, char span)

  shared: src/config.py, src/utils.py, FastAPI service (Phase 8)
```

Embeddings run locally by design — the corpus was re-embedded from scratch when a
chunk-size bug was found, at zero API cost.

---

## How it was evaluated

**Baselines before models.** The dummy and TF-IDF baselines were built and scored
before DistilBERT was trained, so the transformer's gain is measured against
something real. The dummy makes the point concrete: 0.6013 accuracy at 0.2503
macro F1 — near-60% "accurate" while being useless.

**The test split was read once**, by the one script permitted to open `test.csv`;
every model-selection decision was made on validation data.

**Fabrication is defined by category rules fixed before any model output existed.**
Each evaluation question was labelled with its expected behaviour when the question
set was written — an absent-topic answer that is not a refusal is a fabrication, a
confidently-stated table figure is a fabrication. The headline number therefore
depends on no generated ground truth and on no model's opinion of answer quality.

**The refusal judge was validated against blind human labels.** Keyword matching
missed real refusals ("there are no disclosures regarding…", "please provide the
specific filings…"), so an LLM judge replaced it — then was validated on 12
hand-labelled rows with the judge's verdicts stripped from the labelling template:
**11/12 agreement, Cohen's kappa 0.8333** on three classes, **12/12 and kappa
1.0** on the binary axis used for scoring.

**Refusal is semantic, not a similarity threshold.** The highest out-of-corpus
retrieval probe scored 0.4630; the lowest legitimate answerable question scored
0.5750. No threshold separates them, so scores are logged but never gated on.

---

## What this does not do

Measured limitations, each reproducible from `results/`.

- **Financial table figures are unusable.** HTML extraction destroys column
  structure, so numbers reach the model stripped of row and period labels.
  Retrieval cut fabrication on figure questions only from 4/4 to 3/4, and those RAG
  answers carry citations resolving to real chunks — making a wrong number look
  *better* sourced than an uncited one. **Do not ask this system for a figure.**
- **Cross-company retrieval collapses to a single filer.** Mean unique companies in
  the top-5 is **1.625**; one probe returned five Coca-Cola chunks and nothing else.
  Comparative questions across companies cannot be answered reliably, so every
  evaluation question is scoped to one company.
- **RAG falsely refuses 25% of answerable questions** — 3 of 12, where the
  no-retrieval arm refused none. The system that fabricates less also answers fewer
  real questions; the fabrication number is misleading without this beside it.
- **Retrieval cannot filter by fiscal year semantically.** The year lives in
  metadata, not in the embedded text, so FY2024 questions match FY2023 equally.
- **The evaluation is small and single-author.** 28 questions written by the person
  who built the system. One or two fabrications move the rate by over six
  percentage points. No repeated trials, so no variance estimate.
- **Ground truths for the answerable arm were auto-generated**, not hand-written,
  so coverage is a secondary metric. It failed visibly on one question, producing
  zero points despite relevant material being retrieved.
- **Phase 7 scoring is not human-validated.** `results/eval_spotcheck.csv` is a
  blind 10-row sample provided for exactly that, and is unlabelled.
- **An LLM judged an LLM.** Gemini grading Gemini shares training data and failure
  modes, so errors are not independent of what is being measured.
- **Output is not deterministic.** The pinned model ignores `temperature`, accepted
  because mixed-model comparison is a worse problem than sampling noise.
- **Corpus is 8 filings from 4 companies.** Nothing generalises to unseen filings.

---

## Setup

Python 3.13 and a Google Gemini API key (free tier suffices).

```bash
git clone <repo-url> fin-rag      # keep the path short on Windows: MAX_PATH
cd fin-rag

python -m venv venv
venv\Scripts\activate             # Windows
# source venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
cp .env.example .env              # then set GOOGLE_API_KEY=your_key
```

## Running

Each step is independent and re-runnable. Everything before `ask` makes **no LLM
API calls**.

```bash
# classifier: dataset -> baseline -> (Colab fine-tune) -> held-out test
python -m src.data.load_dataset
python -m src.model.baseline
python -m src.model.evaluate_test        # after copying Colab checkpoints into models/

# RAG corpus: fetch -> extract+chunk -> embed -> index -> retrieval quality
python -m src.rag.fetch_filings
python -m src.rag.chunk
python -m src.rag.embed
python -m src.rag.index
python -m src.rag.eval_retrieval

# ask a grounded question (calls the LLM)
python -m src.rag.ask "What risks does Coca-Cola disclose about packaging waste?"
python -m src.rag.ask "..." --ticker KO --year 2024 --no-context

# full evaluation: 28 questions x 2 arms, then scoring
python -m src.rag.eval_populate
python -m src.rag.evaluate_rag

pytest tests/ -v                         # 86 tests, no API calls, no network
```

DistilBERT fine-tuning runs on Colab (local torch is CPU-only) — see
[notebooks/README.md](notebooks/README.md).

## Repository layout

`src/data/` splits · `src/model/` baseline, DistilBERT, test eval · `src/rag/`
EDGAR fetch, chunk, embed, FAISS, retrieval, generation, eval · `src/api/` FastAPI
(Phase 8) · `tests/` 86 tests · `results/` every metric, committed as evidence ·
`data/eval/` the 28-question set · `notebooks/` Colab fine-tuning.

Checkpoints, raw filings and embeddings are gitignored; the provenance manifest and
all metrics JSON are committed, so results are reproducible without shipping 51 MB
of HTML.
