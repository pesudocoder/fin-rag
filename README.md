# Financial Document RAG + Risk Classification

Retrieval-augmented question answering over financial filings, paired with a risk classifier trained on the same corpus.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY`.
