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

# Splits
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15