"""
Live connectors to PubMed (via NCBI E-utilities) and ClinicalTrials.gov (API v2),
so the demo app can pull comparable published models into context alongside a
user's own AUROC/calibration numbers.

These are free, public, keyless APIs. NCBI asks that automated tools identify
themselves and stay under ~3 requests/second without an API key — this module
does small, on-demand, user-triggered lookups only (no polling/scraping).

Note: PubMed doesn't expose "reported AUROC" as structured data — that number
lives inside the abstract's free text. `extract_performance_snippets` does a
best-effort regex scan for common performance-reporting phrasing (AUC, AUROC,
c-statistic, c-index) and returns a short snippet around each match. This is a
heuristic, not a validated extraction: always check the original abstract before
relying on a number it surfaces.
"""

import re
import xml.etree.ElementTree as ET

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
REQUEST_TIMEOUT = 10  # seconds
TOOL_NAME = "clineval-demo-app"

_PERFORMANCE_PATTERN = re.compile(
    r"(AUCs?|AUROCs?|c[- ]statistics?|c[- ]indexe?s?|area under the (?:receiver operating characteristic )?curve)"
    r"[^.\n]{0,20}?(0\.\d{2,3}|1\.0{1,3})",
    re.IGNORECASE,
)


class LiteratureLookupError(Exception):
    """Raised when a PubMed or ClinicalTrials.gov request fails."""
    pass


def search_pubmed(query: str, max_results: int = 5) -> list:
    """
    Search PubMed for articles matching `query`. Returns a list of dicts with
    pmid, title, journal, year, authors (first 3 + 'et al.' if more), and url.
    """
    try:
        search_resp = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "tool": TOOL_NAME},
            timeout=REQUEST_TIMEOUT,
        )
        search_resp.raise_for_status()
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        summary_resp = requests.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json", "tool": TOOL_NAME},
            timeout=REQUEST_TIMEOUT,
        )
        summary_resp.raise_for_status()
        result = summary_resp.json().get("result", {})

        articles = []
        for pmid in id_list:
            info = result.get(pmid)
            if not info:
                continue
            authors = [a.get("name", "") for a in info.get("authors", [])]
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            year = (info.get("pubdate", "") or "").split(" ")[0][:4]
            articles.append({
                "pmid": pmid,
                "title": info.get("title", "").rstrip("."),
                "journal": info.get("source", ""),
                "year": year,
                "authors": author_str,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return articles

    except requests.exceptions.RequestException as e:
        raise LiteratureLookupError(f"PubMed search failed: {e}") from e


def fetch_pubmed_abstract(pmid: str) -> str:
    """Fetch the plain-text abstract for a single PMID."""
    try:
        resp = requests.get(
            f"{EUTILS_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text", "tool": TOOL_NAME},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        raise LiteratureLookupError(f"PubMed abstract fetch failed for PMID {pmid}: {e}") from e


def extract_performance_snippets(text: str, context_chars: int = 45) -> list:
    """
    Best-effort scan of free text for AUC/AUROC/c-statistic/c-index mentions.

    Returns a list of dicts: {metric, value, snippet}. `snippet` is a short window
    around the match (not the full sentence) — treat these as leads to verify in
    the original source, not as ground truth.
    """
    snippets = []
    for match in _PERFORMANCE_PATTERN.finditer(text):
        metric_label = match.group(1)
        value = float(match.group(2))
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        snippets.append({"metric": metric_label, "value": value, "snippet": snippet})
    return snippets


def search_clinical_trials(condition: str, max_results: int = 5) -> list:
    """
    Search ClinicalTrials.gov (API v2) for studies related to `condition`.
    Returns a list of dicts with nct_id, title, status, phase, and url.
    """
    try:
        resp = requests.get(
            CTGOV_BASE,
            params={"query.cond": condition, "pageSize": max_results, "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        studies = resp.json().get("studies", [])

        results = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            nct_id = ident.get("nctId", "")
            phases = design.get("phases", [])
            results.append({
                "nct_id": nct_id,
                "title": ident.get("briefTitle", ""),
                "status": status.get("overallStatus", ""),
                "phase": ", ".join(phases) if phases else "N/A",
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            })
        return results

    except requests.exceptions.RequestException as e:
        raise LiteratureLookupError(f"ClinicalTrials.gov search failed: {e}") from e
