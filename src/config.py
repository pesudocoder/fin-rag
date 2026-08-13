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

# Dataset
DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_66agree"

# Label order comes from the dataset's own ClassLabel definition in
# financial_phrasebank.py: names=["negative", "neutral", "positive"].
LABEL_NAMES = ["negative", "neutral", "positive"]

# Splits
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15