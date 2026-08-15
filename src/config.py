import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Models
LLM_MODEL = "gemini-3.5-flash"
LLM_FALLBACK_MODEL = "gemini-3.5-flash-lite"   # higher free-tier caps, use if 429s during eval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"           # local via sentence-transformers

# Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths (absolute, so scripts work regardless of the cwd they're launched from)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"      # gitignored: binaries
RESULTS_DIR = PROJECT_ROOT / "results"    # committed: metrics JSON
FILINGS_DIR = DATA_RAW / "filings"        # gitignored except manifest.json
EMBEDDINGS_DIR = DATA_PROCESSED / "embeddings"
CHUNKS_FILE = DATA_PROCESSED / "chunks.json"
FAISS_DIR = DATA_PROCESSED / "faiss"

# Dataset
DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_66agree"

# Label order comes from the dataset's own ClassLabel definition in
# financial_phrasebank.py: names=["negative", "neutral", "positive"].
LABEL_NAMES = ["negative", "neutral", "positive"]

# DistilBERT fine-tuning (Phase 3)
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128          # corpus max is 81 whitespace tokens; 512 is unnecessary
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
BATCH_SIZE = 16
DISTILBERT_STANDARD_DIR = MODELS_DIR / "distilbert_standard"
DISTILBERT_WEIGHTED_DIR = MODELS_DIR / "distilbert_weighted"

# SEC EDGAR (Phase 4)
# SEC requires a declared User-Agent with a real contact address for programmatic
# access; requests without one are refused. Their fair-access limit is 10 req/s -
# the delay below is deliberately well under that.
SEC_USER_AGENT = "fin-rag research project pereirakevi7@gmail.com"
SEC_REQUEST_DELAY = 0.5   # seconds between requests (2 req/s, 5x under the cap)

# Edit freely: which filings to build the RAG corpus from.
FILING_TARGETS = [
    {"ticker": "AAPL", "form": "10-K", "fiscal_years": [2023, 2024]},
    {"ticker": "MSFT", "form": "10-K", "fiscal_years": [2023, 2024]},
    {"ticker": "JPM", "form": "10-K", "fiscal_years": [2023, 2024]},
    {"ticker": "KO", "form": "10-K", "fiscal_years": [2023, 2024]},
]

# Chunking (Phase 4)
# Measured in TOKENS, not characters. RecursiveCharacterTextSplitter counts
# characters by default; here a length_function backed by the all-MiniLM-L6-v2
# tokenizer is supplied instead, so these numbers are true word-piece counts
# against the model that actually embeds the text.
#
# all-MiniLM-L6-v2 truncates at 256 word pieces. CHUNK_SIZE must stay under that
# or the tail of every oversized chunk is silently discarded at embed time --
# at 400 this cost 29.4% of the corpus. 230 leaves headroom for the [CLS] and
# [SEP] tokens the tokenizer adds, which the length_function does not count.
# src/rag/chunk.py reports how many chunks exceed the limit; it should be 0.
CHUNK_SIZE = 230
CHUNK_OVERLAP = 50
EMBEDDING_MAX_TOKENS = 256   # all-MiniLM-L6-v2 hard limit
EMBEDDING_DIM = 384

# Retrieval (Phase 5)
TOP_K = 5
# Cosine above which two retrieved chunks are treated as near-duplicates. 10-K
# risk factors are frequently copied verbatim between fiscal years, so both years
# of the same filing otherwise surface together and waste the context budget.
DEDUPE_THRESHOLD = 0.95
# Fetch this multiple of k from FAISS before filtering/deduping, so there are
# spare candidates to backfill with.
OVERFETCH_MULTIPLIER = 4

# RAG generation (Phase 6)
# temperature=0 removes sampling variance. It does NOT guarantee bit-identical
# output across calls: providers batch requests nondeterministically and
# floating-point reduction order varies, so wording can still drift slightly.
LLM_TEMPERATURE = 0
MAX_RETRIES = 4

# Free-tier quota is 15 requests/minute PER MODEL (observed in the 429 payload:
# quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier, quotaValue 15).
# Batch runs must self-pace or they exhaust it: a 24-call smoke test with no
# pacing failed 4 generation calls outright.
#
# 60/15 = 4.0s is the theoretical floor; 4.5s leaves margin for clock skew and
# for the request itself counting against the window it started in.
LLM_MIN_REQUEST_INTERVAL = 4.5

# When a 429 arrives it carries a RetryInfo retryDelay (typically ~55s, since
# the quota window is per minute). Honour it rather than using exponential
# backoff: 1s/2s/4s cannot clear a per-minute quota no matter how many retries
# are allowed, which is why the first pinned run failed.
LLM_RATE_LIMIT_MAX_WAIT = 75.0

# Model pinned for BATCH / EXPERIMENT runs (smoke test, Phase 7 evaluation).
#
# Interactive use (src/rag/ask.py) may still fall back per call. Experiments may
# not: the Phase 6 smoke test resolved the model per call, and free-tier rate
# limiting on gemini-3.5-flash caused 9 of 12 calls to fall back at different
# points, so 3 of 6 question pairs compared a RAG answer from one model against a
# baseline answer from another. Any measured difference was then partly a model
# difference. Experiments resolve the model once and assert every call used it.
#
# KNOWN TRADE-OFF: gemini-3.5-flash-lite ignores temperature entirely ("uses
# fixed sampling defaults"), verified against the installed client. Pinning it
# buys rate-limit headroom - which is what makes pinning viable at all - at the
# cost of LLM_TEMPERATURE having no effect, so experiment output is NOT
# deterministic. gemini-3.5-flash does honour temperature=0 but is rate-limited
# too hard to complete a batch run without falling back, which is the defect
# being fixed. Sampling variance is the lesser of the two problems: it is random
# and can be quantified by repeated runs, whereas mixed-model comparison is a
# systematic confound that cannot be corrected after the fact.
EXPERIMENT_MODEL = "gemini-3.5-flash-lite"

# Refusal judging (Phase 6 defect fix)
JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_PROMPT_VERSION = "v1"

JUDGE_SYSTEM_PROMPT = """\
You classify how an AI assistant responded to a question. You are NOT judging
whether the answer is factually correct - only whether the assistant answered,
declined, or partially answered.

Return exactly one verdict:

REFUSED  - The response declines to answer. It states the information is not
           available, not in the provided documents, cannot be determined, or
           otherwise conveys that it will not or cannot answer. Any phrasing
           counts, including "there are no disclosures regarding X" or "this is
           not addressed". A refusal that adds no substantive answer is REFUSED
           even if it restates the question or explains why.

ANSWERED - The response provides substantive content addressing the question:
           facts, claims, descriptions, figures, or explanations.

PARTIAL  - The response answers part of the question but explicitly declines
           another part, or provides some substance while stating the source
           material is incomplete for the rest.

Judge only the response text. Do not consider whether the claims are true.
Provide a one-sentence justification quoting or paraphrasing the decisive part
of the response.\
"""

# Bump when either prompt below changes, so Phase 7 can record which prompt
# version produced which results.
RAG_PROMPT_VERSION = "v1"

RAG_SYSTEM_PROMPT = """\
You are a financial research assistant answering questions about SEC filings.

You will be given numbered excerpts from SEC filings, then a question.

RULES

1. Answer ONLY from the provided excerpts. They are your sole source of truth.

2. Cite the excerpt numbers supporting each claim, in square brackets, like [1]
   or [2][4]. Cite the specific excerpts you actually used. Do not cite an
   excerpt number that was not provided to you.

3. If the excerpts do not contain the answer, say so plainly, for example:
   "The filings provided do not contain this information."
   This is a correct and expected outcome, not a failure. A question the
   excerpts cannot answer SHOULD be refused. Never pad a refusal with a guess.

4. Do NOT use outside knowledge about these companies. You may have been trained
   on their filings, news coverage, or financial data. Ignore all of it. If a
   fact is not in the excerpts, it is not available to you, even if you are
   confident it is true.

5. Do not speculate, estimate, or infer beyond what the excerpts state. Do not
   combine figures to compute values the text does not state. If the excerpts
   are partial or ambiguous, say what they do and do not establish.

6. Excerpts are extracted from HTML filings and table structure is lost, so
   numbers may appear without their row or column labels. If you cannot
   determine what a figure refers to, or which period it covers, say so rather
   than guessing.

Be concise and factual. Do not add a preamble.\
"""

# Deliberately neutral: the Phase 7 comparison is only valid if the sole
# difference between the two arms is the presence of retrieved context. This
# prompt must not add or subtract caution relative to the RAG prompt.
NO_CONTEXT_SYSTEM_PROMPT = """\
You are a financial research assistant answering questions about SEC filings.

Answer the question as accurately as you can. Be concise and factual. Do not add
a preamble.\
"""

# Splits
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15