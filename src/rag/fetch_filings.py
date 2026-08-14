"""Phase 4 Part A: download SEC filings from EDGAR to build the RAG corpus.

Run from the repo root:

    python -m src.rag.fetch_filings

Reads the wanted filings from config.FILING_TARGETS, resolves each ticker to a
CIK, finds the matching filings via the EDGAR submissions API, and downloads the
primary document of each.

SEC access rules honoured here:
  * A User-Agent naming a real contact address is mandatory; EDGAR returns 403
    without one. Set in config.SEC_USER_AGENT.
  * Fair-access limit is 10 requests/second. config.SEC_REQUEST_DELAY throttles
    to 2/second, five times under the cap.

Writes files to data/raw/filings/ and a provenance manifest to
data/raw/filings/manifest.json. The filings themselves are gitignored; the
manifest is committed so the corpus is reproducible.

No LLM API calls.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from src import config

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _get(url: str) -> requests.Response:
    """Throttled GET against SEC endpoints."""
    time.sleep(config.SEC_REQUEST_DELAY)
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response


def ticker_to_cik() -> dict[str, dict]:
    """Map upper-case ticker -> {cik, title} using SEC's own registry."""
    payload = _get(TICKER_MAP_URL).json()
    return {
        entry["ticker"].upper(): {"cik": int(entry["cik_str"]), "title": entry["title"]}
        for entry in payload.values()
    }


def find_filings(cik: int, form: str, fiscal_years: list[int]) -> list[dict]:
    """Locate filings of a given form whose report period falls in fiscal_years.

    Fiscal year is taken from reportDate (the period covered), not filingDate -
    a 10-K for FY2024 is typically filed in late 2024 or early 2025, so keying
    on the filing date would mislabel it.

    The submissions endpoint caps its inline "recent" block at roughly 1000
    filings and pages the rest into filings.files. For a heavy filer such as
    JPM - which files hundreds of prospectuses a year - "recent" can cover only
    a few months, so the annual report is not in it. Those extra pages are
    fetched too, otherwise large banks silently yield zero results.
    """
    payload = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]
    frames = [pd.DataFrame(payload["recent"])]

    for page in payload.get("files", []):
        older = _get(f"https://data.sec.gov/submissions/{page['name']}").json()
        frames.append(pd.DataFrame(older))

    frame = pd.concat(frames, ignore_index=True)

    matches = frame[frame["form"] == form].copy()
    matches["fiscal_year"] = pd.to_datetime(
        matches["reportDate"], errors="coerce"
    ).dt.year

    wanted = matches[matches["fiscal_year"].isin(fiscal_years)]
    return wanted.to_dict("records")


def download_filing(cik: int, record: dict, ticker: str, company: str) -> dict | None:
    """Download one filing's primary document and return its manifest entry."""
    accession_plain = record["accessionNumber"].replace("-", "")
    document = record["primaryDocument"]
    url = ARCHIVE_URL.format(cik=cik, accession=accession_plain, document=document)

    extension = document.rsplit(".", 1)[-1].lower()
    if extension not in {"htm", "html", "txt"}:
        extension = "htm"

    filename = f"{ticker}_{record['form'].replace('/', '-')}_{record['fiscal_year']}.{extension}"
    destination = config.FILINGS_DIR / filename

    try:
        response = _get(url)
    except requests.HTTPError as exc:
        print(f"  FAILED {filename}: {exc}")
        return None

    destination.write_bytes(response.content)

    return {
        "ticker": ticker,
        "company": company,
        "form_type": record["form"],
        "fiscal_year": int(record["fiscal_year"]),
        "filing_date": record["filingDate"],
        "report_date": record["reportDate"],
        "accession_number": record["accessionNumber"],
        "cik": cik,
        "source_url": url,
        "local_filename": filename,
        "file_size_bytes": len(response.content),
    }


def main() -> None:
    _rule("SEC EDGAR FILING DOWNLOAD")
    print(f"User-Agent: {config.SEC_USER_AGENT}")
    print(f"Throttle  : {config.SEC_REQUEST_DELAY}s between requests "
          f"({1 / config.SEC_REQUEST_DELAY:.0f} req/s vs SEC's 10 req/s cap)")

    config.FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nResolving tickers to CIKs...")
    registry = ticker_to_cik()

    manifest: list[dict] = []
    for target in config.FILING_TARGETS:
        ticker = target["ticker"].upper()
        entry = registry.get(ticker)
        if entry is None:
            print(f"\n{ticker}: not found in SEC ticker registry - skipping")
            continue

        cik, company = entry["cik"], entry["title"]
        print(f"\n{ticker} ({company}, CIK {cik})")

        found = find_filings(cik, target["form"], target["fiscal_years"])
        if not found:
            print(f"  no {target['form']} filings for {target['fiscal_years']}")
            continue

        for record in found:
            result = download_filing(cik, record, ticker, company)
            if result:
                manifest.append(result)
                print(
                    f"  {result['local_filename']:<28} "
                    f"{result['file_size_bytes'] / 1_048_576:6.2f} MB  "
                    f"filed {result['filing_date']}"
                )

    if not manifest:
        raise SystemExit("No filings downloaded. Check network access and FILING_TARGETS.")

    manifest.sort(key=lambda row: (row["ticker"], row["fiscal_year"]))
    manifest_path = config.FILINGS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "phase": "4-corpus",
                "user_agent": config.SEC_USER_AGENT,
                "filing_count": len(manifest),
                "total_bytes": sum(row["file_size_bytes"] for row in manifest),
                "filings": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _rule("SUMMARY")
    table = pd.DataFrame(manifest)[
        ["ticker", "company", "form_type", "fiscal_year", "filing_date",
         "accession_number", "file_size_bytes"]
    ].copy()
    table["size_MB"] = (table.pop("file_size_bytes") / 1_048_576).round(2)
    table["company"] = table["company"].str.slice(0, 22)
    print(table.to_string(index=False))

    total = sum(row["file_size_bytes"] for row in manifest)
    print(f"\n{len(manifest)} filings, {total / 1_048_576:.2f} MB total")
    print(f"files    : {config.FILINGS_DIR}")
    print(f"manifest : {manifest_path}")


if __name__ == "__main__":
    main()
