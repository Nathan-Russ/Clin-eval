"""
Tests for the literature connector module.

These mock requests.get with realistic canned API responses rather than hitting
the live NCBI/ClinicalTrials.gov endpoints — keeps the test suite fast, offline,
and not subject to third-party rate limits, while still exercising the real
parsing logic against the documented response shapes.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo_app"))
import literature as lit  # noqa: E402


# --- extract_performance_snippets: pure function, no mocking needed ---

def test_extract_performance_snippets_finds_auroc():
    text = "The model achieved an AUROC of 0.82 in the validation cohort."
    results = lit.extract_performance_snippets(text)
    assert len(results) == 1
    assert results[0]["value"] == 0.82
    assert "auroc" in results[0]["metric"].lower()


def test_extract_performance_snippets_finds_c_statistic():
    text = "Discrimination was good, with a c-statistic of 0.76 overall."
    results = lit.extract_performance_snippets(text)
    assert len(results) == 1
    assert results[0]["value"] == 0.76


def test_extract_performance_snippets_finds_multiple_mentions():
    text = "Model A had an AUC of 0.71. Model B had an AUC of 0.85 in the same cohort."
    results = lit.extract_performance_snippets(text)
    assert len(results) == 2
    assert {r["value"] for r in results} == {0.71, 0.85}


def test_extract_performance_snippets_returns_empty_for_no_mentions():
    text = "This study describes a retrospective cohort of patients with diabetes."
    assert lit.extract_performance_snippets(text) == []


def test_extract_performance_snippets_snippet_is_short():
    text = "A " * 200 + "the model's AUC of 0.90 was notable" + " b" * 200
    results = lit.extract_performance_snippets(text, context_chars=45)
    assert len(results[0]["snippet"]) < 150  # well short of the full ~800-char text


# --- search_pubmed: mock esearch + esummary ---

FAKE_ESEARCH_RESPONSE = {"esearchresult": {"idlist": ["11111111", "22222222"]}}

FAKE_ESUMMARY_RESPONSE = {
    "result": {
        "uids": ["11111111", "22222222"],
        "11111111": {
            "title": "A risk prediction model for 30-day readmission.",
            "source": "J Clin Epidemiol",
            "pubdate": "2023 Jan",
            "authors": [{"name": "Smith J"}, {"name": "Lee K"}, {"name": "Patel R"}, {"name": "Nguyen T"}],
        },
        "22222222": {
            "title": "External validation of a sepsis early warning score.",
            "source": "Crit Care Med",
            "pubdate": "2022",
            "authors": [{"name": "Garcia M"}],
        },
    }
}


def _mock_get_for_pubmed_search(url, params=None, timeout=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if "esearch" in url:
        resp.json.return_value = FAKE_ESEARCH_RESPONSE
    elif "esummary" in url:
        resp.json.return_value = FAKE_ESUMMARY_RESPONSE
    return resp


def test_search_pubmed_parses_articles_correctly():
    with patch("literature.requests.get", side_effect=_mock_get_for_pubmed_search):
        articles = lit.search_pubmed("readmission risk prediction", max_results=2)

    assert len(articles) == 2
    assert articles[0]["pmid"] == "11111111"
    assert articles[0]["title"] == "A risk prediction model for 30-day readmission"
    assert articles[0]["year"] == "2023"
    assert "et al." in articles[0]["authors"]
    assert articles[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/11111111/"

    assert articles[1]["authors"] == "Garcia M"  # single author, no "et al."


def test_search_pubmed_returns_empty_list_when_no_hits():
    def mock_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"esearchresult": {"idlist": []}}
        return resp

    with patch("literature.requests.get", side_effect=mock_get):
        articles = lit.search_pubmed("an extremely specific nonsense query xyz123")
    assert articles == []


def test_search_pubmed_raises_lookup_error_on_network_failure():
    import requests

    def mock_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated network failure")

    with patch("literature.requests.get", side_effect=mock_get):
        with pytest.raises(lit.LiteratureLookupError):
            lit.search_pubmed("anything")


# --- search_clinical_trials: mock ClinicalTrials.gov API v2 ---

FAKE_CTGOV_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT01234567", "briefTitle": "A Trial of Early Sepsis Detection"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE3"]},
            }
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT07654321", "briefTitle": "Readmission Prevention Program"},
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {},
            }
        },
    ]
}


def test_search_clinical_trials_parses_studies_correctly():
    def mock_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = FAKE_CTGOV_RESPONSE
        return resp

    with patch("literature.requests.get", side_effect=mock_get):
        trials = lit.search_clinical_trials("sepsis")

    assert len(trials) == 2
    assert trials[0]["nct_id"] == "NCT01234567"
    assert trials[0]["phase"] == "PHASE3"
    assert trials[0]["url"] == "https://clinicaltrials.gov/study/NCT01234567"
    assert trials[1]["phase"] == "N/A"  # no phases listed in the fake response


def test_search_clinical_trials_raises_lookup_error_on_network_failure():
    import requests

    def mock_get(*args, **kwargs):
        raise requests.exceptions.Timeout("simulated timeout")

    with patch("literature.requests.get", side_effect=mock_get):
        with pytest.raises(lit.LiteratureLookupError):
            lit.search_clinical_trials("anything")
