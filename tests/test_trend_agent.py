"""
Unit tests for agents/trend_agent.py

Tests cover:
  • RSS fetching (mocked feedparser)
  • Guardian API fetching (mocked requests)
  • NewsData.io fetching (mocked requests)
  • Deduplication logic in get_trending_topics()
  • Graceful degradation when sources fail or keys are missing
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.trend_agent import (
    fetch_rss_headlines,
    fetch_guardian_news,
    fetch_newsdata,
    get_trending_topics,
)


# ─────────────────────────────────────────────────────────────────────────────
# RSS tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchRssHeadlines:
    """feedparser.parse is mocked to avoid real HTTP calls."""

    def _make_feed(self, titles: list[str]) -> MagicMock:
        feed = MagicMock()
        entries = []
        for i, t in enumerate(titles):
            entry = MagicMock()
            # Explicitly set .get() to return real strings — avoids MagicMock leaking into re.sub
            entry.get.side_effect = lambda key, default="", _t=t, _i=i: {
                "title": _t,
                "summary": f"Summary of {_t}",
                "description": "",
                "link": f"https://example.com/{_i}",
            }.get(key, default)
            entries.append(entry)
        feed.entries = entries
        return feed

    @patch("agents.trend_agent.feedparser.parse")
    def test_returns_headlines_from_all_feeds(self, mock_parse):
        """Should collect headlines from every configured feed."""
        from agents.trend_agent import RSS_FEEDS
        mock_parse.return_value = self._make_feed(["Title A", "Title B"])

        results = fetch_rss_headlines(max_per_feed=2)

        # Every feed contributes up to max_per_feed headlines
        assert len(results) == len(RSS_FEEDS) * 2
        assert all("title" in r for r in results)
        assert all("source" in r for r in results)

    @patch("agents.trend_agent.feedparser.parse")
    def test_skips_entries_with_empty_title(self, mock_parse):
        """Entries with empty titles should be silently dropped."""
        feed = MagicMock()
        empty_entry = MagicMock()
        empty_entry.get.side_effect = lambda key, default="": {"title": "", "summary": "x", "description": "", "link": ""}.get(key, default)
        good_entry = MagicMock()
        good_entry.get.side_effect = lambda key, default="": {"title": "Good Title", "summary": "x", "description": "", "link": "http://a.com"}.get(key, default)
        feed.entries = [empty_entry, good_entry]
        mock_parse.return_value = feed

        results = fetch_rss_headlines(max_per_feed=5)
        assert all(r["title"] != "" for r in results)

    @patch("agents.trend_agent.feedparser.parse", side_effect=Exception("Network error"))
    def test_gracefully_handles_feed_failure(self, _mock_parse):
        """A failing RSS feed should be logged and skipped — not crash the pipeline."""
        results = fetch_rss_headlines()
        assert isinstance(results, list)  # returns empty list, not exception


# ─────────────────────────────────────────────────────────────────────────────
# Guardian API tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchGuardianNews:
    def _make_guardian_response(self, titles: list[str]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": {
                "results": [
                    {
                        "webTitle": t,
                        "webUrl": f"https://theguardian.com/{i}",
                        "fields": {"trailText": f"Summary of {t}"},
                    }
                    for i, t in enumerate(titles)
                ]
            }
        }
        return mock_resp

    @patch("agents.trend_agent.settings")
    @patch("agents.trend_agent.requests.get")
    def test_returns_articles_when_key_set(self, mock_get, mock_settings):
        mock_settings.guardian_api_key = "fake-key"
        mock_get.return_value = self._make_guardian_response(["Climate Deal", "AI Breakthrough"])

        results = fetch_guardian_news()

        assert len(results) == 2
        assert results[0]["source"] == "The Guardian"
        assert results[0]["title"] == "Climate Deal"

    @patch("agents.trend_agent.settings")
    def test_returns_empty_when_no_key(self, mock_settings):
        mock_settings.guardian_api_key = None

        results = fetch_guardian_news()
        assert results == []

    @patch("agents.trend_agent.settings")
    @patch("agents.trend_agent.requests.get", side_effect=Exception("Network error"))
    def test_gracefully_handles_api_failure(self, _mock_get, mock_settings):
        mock_settings.guardian_api_key = "fake-key"

        results = fetch_guardian_news()
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# NewsData.io tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchNewsdata:
    def _make_newsdata_response(self, titles: list[str]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "results": [
                {
                    "title": t,
                    "description": f"Description of {t}",
                    "link": f"https://example.com/{i}",
                    "source_name": "Example News",
                }
                for i, t in enumerate(titles)
            ],
        }
        return mock_resp

    @patch("agents.trend_agent.settings")
    @patch("agents.trend_agent.requests.get")
    def test_returns_articles_when_key_set(self, mock_get, mock_settings):
        mock_settings.newsdata_api_key = "fake-key"
        mock_get.return_value = self._make_newsdata_response(["Trade War Update", "Space Mission"])

        results = fetch_newsdata()

        assert len(results) == 2
        assert "NewsData" in results[0]["source"]

    @patch("agents.trend_agent.settings")
    def test_returns_empty_when_no_key(self, mock_settings):
        mock_settings.newsdata_api_key = None

        results = fetch_newsdata()
        assert results == []

    @patch("agents.trend_agent.settings")
    @patch("agents.trend_agent.requests.get")
    def test_handles_non_success_status(self, mock_get, mock_settings):
        mock_settings.newsdata_api_key = "fake-key"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "error", "results": []}
        mock_get.return_value = mock_resp

        results = fetch_newsdata()
        assert results == []

    @patch("agents.trend_agent.settings")
    @patch("agents.trend_agent.requests.get", side_effect=Exception("Timeout"))
    def test_gracefully_handles_api_failure(self, _mock_get, mock_settings):
        mock_settings.newsdata_api_key = "fake-key"

        results = fetch_newsdata()
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTrendingTopics:
    @patch("agents.trend_agent.fetch_newsdata", return_value=[])
    @patch("agents.trend_agent.fetch_guardian_news", return_value=[])
    @patch("agents.trend_agent.fetch_rss_headlines")
    def test_deduplication_removes_similar_headlines(self, mock_rss, _mock_guardian, _mock_newsdata):
        """Headlines with >60% word overlap should be collapsed to one."""
        mock_rss.return_value = [
            {"title": "Global leaders sign historic climate deal in Geneva", "source": "BBC", "url": "", "summary": ""},
            {"title": "World leaders sign historic climate deal", "source": "Reuters", "url": "", "summary": ""},
            {"title": "Scientists discover new exoplanet with water", "source": "AP", "url": "", "summary": ""},
        ]

        results = get_trending_topics(max_topics=10)

        titles = [r["title"] for r in results]
        # The two climate headlines are duplicates — only one should survive
        climate_count = sum(1 for t in titles if "climate" in t.lower())
        assert climate_count == 1
        assert any("exoplanet" in t.lower() for t in titles)

    @patch("agents.trend_agent.fetch_newsdata", return_value=[])
    @patch("agents.trend_agent.fetch_guardian_news", return_value=[])
    @patch("agents.trend_agent.fetch_rss_headlines")
    def test_respects_max_topics_limit(self, mock_rss, _mock_guardian, _mock_newsdata):
        """Result list must not exceed max_topics."""
        mock_rss.return_value = [
            {"title": f"Unique headline number {i}", "source": "BBC", "url": "", "summary": ""}
            for i in range(20)
        ]

        results = get_trending_topics(max_topics=5)
        assert len(results) <= 5

    @patch("agents.trend_agent.fetch_newsdata", return_value=[])
    @patch("agents.trend_agent.fetch_guardian_news", return_value=[])
    @patch("agents.trend_agent.fetch_rss_headlines", return_value=[])
    def test_returns_empty_list_when_no_data(self, _mock_rss, _mock_guardian, _mock_newsdata):
        results = get_trending_topics()
        assert results == []

    @patch("agents.trend_agent.fetch_newsdata")
    @patch("agents.trend_agent.fetch_guardian_news")
    @patch("agents.trend_agent.fetch_rss_headlines")
    def test_merges_all_three_sources(self, mock_rss, mock_guardian, mock_newsdata):
        """Items from all three sources should appear in the merged result."""
        mock_rss.return_value = [
            {"title": "RSS story Alpha", "source": "BBC", "url": "", "summary": ""},
        ]
        mock_guardian.return_value = [
            {"title": "Guardian story Beta", "source": "The Guardian", "url": "", "summary": ""},
        ]
        mock_newsdata.return_value = [
            {"title": "NewsData story Gamma", "source": "NewsData: CNN", "url": "", "summary": ""},
        ]

        results = get_trending_topics(max_topics=10)
        titles = [r["title"] for r in results]

        assert any("Alpha" in t for t in titles)
        assert any("Beta" in t for t in titles)
        assert any("Gamma" in t for t in titles)
