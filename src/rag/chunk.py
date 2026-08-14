"""Phase 4 Part B: extract text from SEC filings and split it into chunks.

Run from the repo root (after src.rag.fetch_filings):

    python -m src.rag.chunk

Reads data/raw/filings/manifest.json, extracts readable text from each filing,
splits it with LangChain's RecursiveCharacterTextSplitter, and writes the chunks
plus their citation metadata to data/processed/chunks.json.

No LLM API calls.

Chunk sizing
------------
CHUNK_SIZE / CHUNK_OVERLAP in src/config.py are expressed in TOKENS, not
characters. RecursiveCharacterTextSplitter measures characters by default, so a
length_function backed by the all-MiniLM-L6-v2 tokenizer is supplied instead.
That ties the chunk budget to the tokenizer of the model that actually embeds
the text, rather than to a character count that only loosely correlates with it
(financial prose is dense with numbers and tickers, which tokenize far less
efficiently than ordinary English - a fixed character budget would produce
wildly varying token counts).

The tokenizer used for counting has its model_max_length raised, otherwise
transformers emits a "sequence length is longer than the specified maximum"
warning for every long span it measures.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import warnings

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

# 10-K primary documents are XHTML, so bs4's html parser warns on every one.
# Parsing them as HTML is intentional and works; the warning is noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Filings contain characters outside cp1252 (checkbox glyphs on the cover page,
# typographic dashes). The Windows console defaults to cp1252 and raises on
# them, so replace rather than crash when printing example chunks.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Paragraph, then line, then sentence, then word, then character.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Lines that are only a page number, or a bare roman numeral, or a form marker.
PAGE_NOISE = re.compile(
    r"^(page\s*)?\d{1,4}$|^[ivxlcdm]{1,7}$|^table of contents$|^form\s+10-k$",
    re.IGNORECASE,
)


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def build_counter():
    """Token-count function using the embedding model's own tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{config.EMBEDDING_MODEL}"
    )
    # Counting only - suppress the length warning by lifting the cap.
    tokenizer.model_max_length = int(1e9)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count, tokenizer


def extract_text(raw: bytes) -> str:
    """HTML (often inline-XBRL) -> readable plain text.

    10-K primary documents are inline XBRL: the financial data is marked up with
    <ix:...> tags, and the document opens with an <ix:header> block wrapped in a
    hidden div holding thousands of machine-readable facts. That block carries no
    prose and would otherwise dominate the extracted text, so hidden elements are
    dropped before extraction.
    """
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    # Inline-XBRL header and any display:none container.
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        tag.decompose()
    for name in ["ix:header", "ix:hidden"]:
        for tag in soup.find_all(name):
            tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalise unicode spaces that HTML entities leave behind.
    text = text.replace("\xa0", " ").replace("​", "")

    lines = []
    previous = None
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or PAGE_NOISE.match(line):
            continue
        # Collapse immediately repeated lines (running headers/footers).
        if line == previous:
            continue
        lines.append(line)
        previous = line

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def chunk_filing(entry: dict, text: str, splitter) -> list[dict]:
    """Split one filing's text and attach citation metadata to every chunk."""
    documents = splitter.create_documents([text])

    chunks = []
    for index, document in enumerate(documents):
        start = document.metadata.get("start_index", -1)
        content = document.page_content
        chunks.append(
            {
                "text": content,
                "source_filename": entry["local_filename"],
                "ticker": entry["ticker"],
                "company": entry["company"],
                "form_type": entry["form_type"],
                "fiscal_year": entry["fiscal_year"],
                "accession_number": entry["accession_number"],
                "source_url": entry["source_url"],
                "chunk_index": index,
                "char_start": start,
                "char_end": start + len(content) if start >= 0 else -1,
            }
        )
    return chunks


def main() -> None:
    _rule("EXTRACT + CHUNK SEC FILINGS")

    manifest_path = config.FILINGS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path}. Run: python -m src.rag.fetch_filings"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["filings"]

    count_tokens, _ = build_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=count_tokens,   # token-based, not character-based
        separators=SEPARATORS,
        add_start_index=True,           # gives char_start for citation spans
    )
    print(f"chunk_size={config.CHUNK_SIZE} tokens, overlap={config.CHUNK_OVERLAP} tokens")
    print(f"length_function: {config.EMBEDDING_MODEL} tokenizer\n")

    all_chunks: list[dict] = []
    extraction_rows = []

    for entry in manifest:
        path = config.FILINGS_DIR / entry["local_filename"]
        if not path.exists():
            print(f"  MISSING {entry['local_filename']} - skipping")
            continue

        raw = path.read_bytes()
        text = extract_text(raw)
        chunks = chunk_filing(entry, text, splitter)
        all_chunks.extend(chunks)

        extraction_rows.append(
            {
                "filing": entry["local_filename"],
                "raw_bytes": len(raw),
                "extracted_chars": len(text),
                "retained_pct": round(len(text) / len(raw) * 100, 2),
                "chunks": len(chunks),
            }
        )
        print(
            f"  {entry['local_filename']:<24} "
            f"{len(raw) / 1_048_576:6.2f} MB raw -> {len(text):>9,} chars "
            f"({len(text) / len(raw) * 100:5.2f}% retained), {len(chunks):>4} chunks"
        )

    if not all_chunks:
        raise SystemExit("No chunks produced.")

    _rule("EXTRACTION: RAW SIZE vs EXTRACTED TEXT")
    extraction = pd.DataFrame(extraction_rows)
    print(extraction.to_string(index=False))
    total_raw = extraction["raw_bytes"].sum()
    total_chars = extraction["extracted_chars"].sum()
    print(
        f"\ntotal: {total_raw / 1_048_576:.2f} MB raw -> {total_chars:,} chars "
        f"({total_chars / total_raw * 100:.2f}% retained). "
        f"The remainder is HTML markup, inline-XBRL tags and boilerplate."
    )

    _rule("CHUNK STATISTICS")
    token_lengths = [count_tokens(chunk["text"]) for chunk in all_chunks]
    for chunk, length in zip(all_chunks, token_lengths):
        chunk["token_count"] = length

    ordered = sorted(token_lengths)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    stats = {
        "total_chunks": len(all_chunks),
        "min": min(ordered),
        "max": max(ordered),
        "mean": round(statistics.mean(ordered), 2),
        "median": statistics.median(ordered),
        "p95": p95,
    }
    print(f"total chunks: {stats['total_chunks']:,}\n")
    print("chunks per filing:")
    print(
        pd.DataFrame(extraction_rows)[["filing", "chunks"]].to_string(index=False)
    )
    print("\ntoken-length distribution:")
    for key in ["min", "max", "mean", "median", "p95"]:
        print(f"  {key:>6}: {stats[key]}")

    over_limit = sum(1 for length in token_lengths if length > config.EMBEDDING_MAX_TOKENS)
    share = over_limit / len(token_lengths) * 100
    lost = sum(
        length - config.EMBEDDING_MAX_TOKENS
        for length in token_lengths
        if length > config.EMBEDDING_MAX_TOKENS
    )
    print(
        f"\nchunks exceeding the {config.EMBEDDING_MAX_TOKENS}-token embedding limit: "
        f"{over_limit:,} / {len(token_lengths):,} ({share:.1f}%)"
    )
    if over_limit:
        print(
            f"  !! {lost:,} tokens ({lost / sum(token_lengths) * 100:.1f}% of all corpus "
            f"tokens) will be SILENTLY TRUNCATED by {config.EMBEDDING_MODEL},\n"
            f"     which caps input at {config.EMBEDDING_MAX_TOKENS} word pieces.\n"
            f"     Text beyond that point is not represented in any vector and is\n"
            f"     therefore unretrievable. Lower CHUNK_SIZE in src/config.py to fix."
        )

    _rule("EXAMPLE CHUNKS")
    step = max(1, len(all_chunks) // 3)
    for chunk in [all_chunks[0], all_chunks[step], all_chunks[step * 2]]:
        print(
            f"\n--- {chunk['source_filename']} chunk {chunk['chunk_index']} "
            f"| chars {chunk['char_start']}-{chunk['char_end']} "
            f"| {chunk['token_count']} tokens ---"
        )
        print(chunk["text"])

    _rule("SAVING")
    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"{config.CHUNKS_FILE}  ({len(all_chunks):,} chunks)")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "4-chunking",
        "chunk_size_tokens": config.CHUNK_SIZE,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP,
        "length_function": f"{config.EMBEDDING_MODEL} tokenizer (word pieces)",
        "separators": SEPARATORS,
        "embedding_max_tokens": config.EMBEDDING_MAX_TOKENS,
        "filings": extraction_rows,
        "totals": {
            "filings": len(extraction_rows),
            "raw_bytes": int(total_raw),
            "extracted_chars": int(total_chars),
            "retained_pct": round(total_chars / total_raw * 100, 2),
        },
        "token_length": stats,
        "chunks_over_embedding_limit": over_limit,
        "chunks_over_embedding_limit_pct": round(share, 2),
        "tokens_lost_to_truncation": int(lost),
    }
    stats_path = config.RESULTS_DIR / "chunking_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{stats_path}")


if __name__ == "__main__":
    main()
