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

# Splits
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15