import os
from dotenv import load_dotenv

load_dotenv()

# Models
LLM_MODEL = "gemini-3.5-flash"
LLM_FALLBACK_MODEL = "gemini-3.5-flash-lite"   # higher free-tier caps, use if 429s during eval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"           # local via sentence-transformers

# Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"