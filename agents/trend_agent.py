"""
Trend Discovery Agent
──────────────────────
Aggregates trending news headlines from three free, no-auth (or free-key) sources:

  1. RSS feeds      — BBC, Reuters, NPR, Al Jazeera, AP News + Google News RSS
                      No API key needed. Real-time. Completely free.

  2. The Guardian   — Official OpenPlatform API. Free tier: 5 000 req/day, no
                      delay, high-quality journalism. Register at:
                      https://bopenplatform.theguardian.com/access/
                      Set GUARDIAN_API_KEY in .env (optional but recommended).

  3. NewsData.io    — Free tier: 200 credits/day, real-time. Register at:
                      https://newsdata.io/register
                      Set NEWSDATA_API_KEY in .env (optional).

Returns a deduplicated list of the top N trending topics as dicts with:
  title, summary, source, url
"""

import re
import requests
import feedparser

from utils.logger import logger
from config.settings import settings

# ── RSS feed catalogue ────────────────────────────────────────────────────────
# Standard editorial RSS feeds — always on, no key required
RSS_FEEDS: dict[str, str] = {
    "BBC News":           "https://feeds.bbci.co.uk/news/rss.xml",
    "Reuters Top News":   "https://feeds.reuters.com/reuters/topNews",
    "NPR News":           "https://feeds.npr.org/1001/rss.xml",
    "Al Jazeera":         "https://www.aljazeera.com/xml/rss/all.xml",
    "AP News":            "https://rsshub.app/apnews/topics/apf-topnews",
    # Google News RSS — curates trending topics from thousands of sources in real time
    "Google News World":  "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    "Google News Tech":   "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "Google News Sci":    "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
    "Google News Biz":    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
}


# ── Guardian API endpoint ────────────────────────────────────────────────────
_GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"

# ── NewsData.io endpoint ──────────────────────────────────────────────────────
_NEWSDATA_LATEST_URL = "https://newsdata.io/api/1/latest"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove simple HTML tags from a summary string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_rss_headlines(max_per_feed: int = 5) -> list[dict]:
    """
    Fetch the top N headlines from each configured RSS feed.
    Silently skips any feed that fails (network or parsing error).
    """
    headlines: list[dict] = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_feed:
                    break
                title = entry.get("title", "").strip()
                if not title:
                    continue
                headlines.append(
                    {
                        "title": title,
                        "summary": _strip_html(
                            entry.get("summary", entry.get("description", ""))
                        )[:300],
                        "source": source,
                        "url": entry.get("link", ""),
                    }
                )
                count += 1
            logger.debug("Fetched %d headlines from %s", count, source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RSS fetch failed for %s: %s", source, exc)

    return headlines


def fetch_guardian_news(max_results: int = 15) -> list[dict]:
    """
    Fetch latest top news from The Guardian OpenPlatform API.

    Free tier: 5 000 requests/day, no article delay, full access.
    API key: https://bopenplatform.theguardian.com/access/
    Set GUARDIAN_API_KEY in .env. Skipped gracefully if key is not set.

    Returns list of {title, summary, source, url} dicts.
    """
    api_key = settings.guardian_api_key
    if not api_key:
        logger.debug("GUARDIAN_API_KEY not set — skipping Guardian source.")
        return []

    try:
        params = {
            "api-key": api_key,
            "order-by": "newest",
            "page-size": max_results,
            "show-fields": "trailText",
            # Focus on news sections relevant to a news explainer channel
            "section": "world|technology|science|environment|business|politics",
        }
        response = requests.get(_GUARDIAN_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = data.get("response", {}).get("results", [])
        items: list[dict] = []
        for article in articles:
            title = article.get("webTitle", "").strip()
            if not title:
                continue
            items.append({
                "title": title,
                "summary": _strip_html(
                    article.get("fields", {}).get("trailText", "")
                )[:300],
                "source": "The Guardian",
                "url": article.get("webUrl", ""),
            })

        logger.debug("Fetched %d articles from The Guardian.", len(items))
        return items

    except Exception as exc:  # noqa: BLE001
        logger.warning("Guardian API fetch failed: %s", exc)
        return []


def fetch_newsdata(max_results: int = 10) -> list[dict]:
    """
    Fetch latest top news from NewsData.io.

    Free tier: 200 credits/day, real-time (no delay).
    API key: https://newsdata.io/register
    Set NEWSDATA_API_KEY in .env. Skipped gracefully if key is not set.

    Returns list of {title, summary, source, url} dicts.
    """
    api_key = settings.newsdata_api_key
    if not api_key:
        logger.debug("NEWSDATA_API_KEY not set — skipping NewsData.io source.")
        return []

    try:
        params = {
            "apikey": api_key,
            "language": "en",
            "country": "us,gb,au,ca",
            "size": max_results,
        }
        response = requests.get(_NEWSDATA_LATEST_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            logger.warning("NewsData.io returned status: %s", data.get("status"))
            return []

        articles = data.get("results", [])
        items: list[dict] = []
        for article in articles:
            title = (article.get("title") or "").strip()
            if not title:
                continue
            items.append({
                "title": title,
                "summary": _strip_html(article.get("description") or "")[:300],
                "source": f"NewsData: {article.get('source_name', 'Unknown')}",
                "url": article.get("link", ""),
            })

        logger.debug("Fetched %d articles from NewsData.io.", len(items))
        return items

    except Exception as exc:  # noqa: BLE001
        logger.warning("NewsData.io fetch failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_trending_topics(max_topics: int = 10) -> list[dict]:
    """
    Aggregate, deduplicate, and return the top trending news topics.

    Source priority (highest quality first):
      1. RSS feeds (BBC, Reuters, NPR, Al Jazeera, AP, Google News)
      2. The Guardian API
      3. NewsData.io

    Deduplication uses word-overlap: two headlines are considered the same
    story if >60% of their significant words match. The first occurrence
    (highest-priority source) wins.

    Returns a list of dicts: {title, summary, source, url}
    """
    rss_items      = fetch_rss_headlines(max_per_feed=4)
    guardian_items = fetch_guardian_news(max_results=15)
    newsdata_items = fetch_newsdata(max_results=10)

    all_items = rss_items + guardian_items + newsdata_items

    # Deduplicate by fuzzy title overlap
    seen_word_sets: list[frozenset] = []
    unique: list[dict] = []

    for item in all_items:
        if not item.get("title"):
            continue
        title_words = frozenset(item["title"].lower().split())
        is_duplicate = any(
            len(title_words & seen) / max(len(title_words), 1) > 0.6
            for seen in seen_word_sets
        )
        if not is_duplicate:
            seen_word_sets.append(title_words)
            unique.append(item)

    logger.info(
        "Trending topics: %d unique from %d total headlines", len(unique), len(all_items)
    )
    return unique[:max_topics]
