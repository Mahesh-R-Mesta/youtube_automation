"""
Unit tests for agents/visual_agent.py

Tests cover:
  • fetch_pexels_image() downloads best photo and saves it
  • fetch_pexels_image() falls back to "world news" on empty results
  • fetch_scene_images() skips re-download if file already exists
  • fetch_scene_images() uses previous image as placeholder on failure
  • HTTP errors raise correctly
"""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.visual_agent import fetch_pexels_image, fetch_scene_images, get_pexels_attribution


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pexels_response(photos: list[dict]) -> MagicMock:
    """Return a mock requests.Response with a Pexels photo search payload."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"photos": photos}
    return mock_resp


def _make_photo(photo_id: int = 1, width: int = 3000) -> dict:
    return {
        "id": photo_id,
        "width": width,
        "photographer": "Test Photographer",
        "src": {
            "large2x": f"https://example.com/photo_{photo_id}_2x.jpg",
            "large":   f"https://example.com/photo_{photo_id}.jpg",
            "original": f"https://example.com/photo_{photo_id}_orig.jpg",
        },
    }


def _make_img_response(content: bytes = b"\xff\xd8\xff\xe0\x00\x10fake_jpeg") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content.return_value = [content]
    return mock_resp


# ─────────────────────────────────────────────────────────────────────────────
# fetch_pexels_image
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchPexelsImage:
    @patch("agents.visual_agent.requests.get")
    def test_downloads_highest_quality_photo(self, mock_get, tmp_path):
        """Should select the photo with the largest width and save it."""
        mock_get.side_effect = [
            _make_pexels_response([
                _make_photo(1, width=1920),
                _make_photo(2, width=3840),   # ← best quality
                _make_photo(3, width=1280),
            ]),
            _make_img_response(),
        ]

        dest = tmp_path / "scene_01.jpg"
        result = fetch_pexels_image("climate summit", dest)

        assert result == dest
        assert dest.exists()

        # Verify the download URL corresponds to the widest photo (id=2)
        download_call = mock_get.call_args_list[1]
        assert "photo_2_2x" in download_call[0][0]

    @patch("agents.visual_agent.requests.get")
    def test_falls_back_to_world_news_when_no_results(self, mock_get, tmp_path):
        """If the keyword returns no photos, retry with 'world news'."""
        mock_get.side_effect = [
            _make_pexels_response([]),               # empty for original keyword
            _make_pexels_response([_make_photo()]),  # fallback keyword succeeds
            _make_img_response(),
        ]

        dest = tmp_path / "fallback.jpg"
        result = fetch_pexels_image("obscure keyword no results", dest)

        assert result == dest
        assert mock_get.call_count == 3  # search × 2 + download × 1

    @patch("agents.visual_agent.requests.get")
    def test_raises_runtime_error_when_both_searches_empty(self, mock_get, tmp_path):
        mock_get.side_effect = [
            _make_pexels_response([]),   # original keyword → empty
            _make_pexels_response([]),   # fallback → empty
        ]

        with pytest.raises(RuntimeError, match="no photos"):
            fetch_pexels_image("impossible keyword", tmp_path / "out.jpg")

    @patch("agents.visual_agent.requests.get")
    def test_http_error_propagates(self, mock_get, tmp_path):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            fetch_pexels_image("anything", tmp_path / "out.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# fetch_scene_images
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchSceneImages:
    @patch("agents.visual_agent.time.sleep")         # skip delays in tests
    @patch("agents.visual_agent.fetch_pexels_image")
    def test_downloads_one_image_per_keyword(self, mock_fetch, _mock_sleep, tmp_path):
        """Should call fetch_pexels_image once per keyword."""
        keywords = ["city skyline", "stock market", "scientist lab",
                    "government building", "protest crowd", "news studio"]

        def side_effect(keyword, output_path):
            output_path.write_bytes(b"fake")
            return output_path

        mock_fetch.side_effect = side_effect

        results = fetch_scene_images(keywords, tmp_path, run_id="20260324_090000")

        assert len(results) == 6
        assert mock_fetch.call_count == 6

    @patch("agents.visual_agent.time.sleep")
    @patch("agents.visual_agent.fetch_pexels_image")
    def test_skips_cached_images(self, mock_fetch, _mock_sleep, tmp_path):
        """If the destination file already exists, skip the download."""
        run_id = "20260324_090000"
        existing = tmp_path / f"{run_id}_scene_01.jpg"
        existing.write_bytes(b"cached")

        keywords = ["city skyline", "something new"]

        def side_effect(keyword, output_path):
            output_path.write_bytes(b"fresh")
            return output_path

        mock_fetch.side_effect = side_effect

        results = fetch_scene_images(keywords, tmp_path, run_id=run_id)

        # scene_01 was cached; only scene_02 should have triggered a download
        assert mock_fetch.call_count == 1
        assert len(results) == 2

    @patch("agents.visual_agent.time.sleep")
    @patch("agents.visual_agent.fetch_pexels_image")
    def test_uses_previous_image_as_placeholder_on_failure(
        self, mock_fetch, _mock_sleep, tmp_path
    ):
        """If one download fails, reuse the previous scene's image."""
        keywords = ["scene one", "scene two fails", "scene three"]
        first_img = tmp_path / "20260324_scene_01.jpg"
        first_img.write_bytes(b"img1")

        call_count = 0

        def side_effect(keyword, output_path):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Pexels API error")
            output_path.write_bytes(b"img")
            return output_path

        mock_fetch.side_effect = side_effect

        results = fetch_scene_images(keywords, tmp_path, run_id="20260324")

        assert len(results) == 3
        # result[1] should be a copy of result[0] (the placeholder)
        assert results[1] == results[0]


# ─────────────────────────────────────────────────────────────────────────────
# Attribution
# ─────────────────────────────────────────────────────────────────────────────

def test_pexels_attribution_is_non_empty():
    text = get_pexels_attribution()
    assert "Pexels" in text
    assert "https://www.pexels.com" in text
