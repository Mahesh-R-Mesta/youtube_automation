"""
Unit tests for agents/visual_agent.py

Tests cover:
  • _generate_schnell() calls the correct Pixazo endpoint and returns a URL
  • _generate_klein() calls the correct Pixazo endpoint and returns a URL
  • generate_image_from_prompt() uses Schnell by default; falls back to Klein
  • generate_image_from_prompt() uses Klein directly when use_quality_model=True
  • generate_scene_images() returns one Path per prompt
  • generate_scene_images() skips regeneration when files already exist
  • generate_scene_images() reuses the previous image on failure (placeholder)
  • Missing API key raises ValueError
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.visual_agent import (
    _generate_schnell,
    _generate_klein,
    generate_image_from_prompt,
    generate_scene_images,
    _SCHNELL_URL,
    _KLEIN_URL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

FAKE_IMG_BYTES = b"\xff\xd8\xff\xe0\x00\x10fake_jpeg_data"
FAKE_IMG_URL = "https://cdn.pixazo.ai/generated/test_image.jpg"


def _api_response(url: str = FAKE_IMG_URL) -> MagicMock:
    """Mock for a synchronous Pixazo generate endpoint response."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"output": url}
    return mock


def _download_response(content: bytes = FAKE_IMG_BYTES) -> MagicMock:
    """Mock for an image download (stream=True)."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.iter_content.return_value = [content]
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# _generate_schnell
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSchnell:
    @patch("agents.visual_agent.requests.post")
    def test_calls_correct_endpoint_and_returns_url(self, mock_post):
        mock_post.return_value = _api_response()
        url = _generate_schnell("a cinematic test scene")
        assert url == FAKE_IMG_URL
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == _SCHNELL_URL
        assert kwargs["json"]["prompt"] == "a cinematic test scene"
        assert kwargs["json"]["num_steps"] == 4

    @patch("agents.visual_agent.requests.post")
    def test_raises_on_missing_output_key(self, mock_post):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"error": "something went wrong"}
        mock_post.return_value = mock
        with pytest.raises(ValueError):
            _generate_schnell("test prompt")


# ─────────────────────────────────────────────────────────────────────────────
# _generate_klein
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateKlein:
    @patch("agents.visual_agent.requests.post")
    def test_calls_correct_endpoint_and_returns_url(self, mock_post):
        mock_post.return_value = _api_response()
        url = _generate_klein("a high-quality scene")
        assert url == FAKE_IMG_URL
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == _KLEIN_URL
        assert kwargs["json"]["steps"] == 25

    @patch("agents.visual_agent.requests.post")
    def test_raises_on_missing_output_key(self, mock_post):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {}
        mock_post.return_value = mock
        with pytest.raises(ValueError):
            _generate_klein("test prompt")


# ─────────────────────────────────────────────────────────────────────────────
# generate_image_from_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateImageFromPrompt:
    @patch("agents.visual_agent.requests.get")
    @patch("agents.visual_agent.requests.post")
    def test_uses_schnell_by_default(self, mock_post, mock_get, tmp_path):
        mock_post.return_value = _api_response()
        mock_get.return_value = _download_response()

        dest = tmp_path / "scene_01.jpg"
        result = generate_image_from_prompt("cinematic cityscape", dest)

        assert result == dest
        assert dest.exists()
        # POST should target the Schnell endpoint
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == _SCHNELL_URL

    @patch("agents.visual_agent.requests.get")
    @patch("agents.visual_agent.requests.post")
    def test_uses_klein_when_quality_flag_is_set(self, mock_post, mock_get, tmp_path):
        mock_post.return_value = _api_response()
        mock_get.return_value = _download_response()

        dest = tmp_path / "scene_hq.jpg"
        result = generate_image_from_prompt("epic landscape", dest, use_quality_model=True)

        assert result == dest
        assert mock_post.call_args[0][0] == _KLEIN_URL

    @patch("agents.visual_agent._generate_klein")
    @patch("agents.visual_agent._generate_schnell")
    @patch("agents.visual_agent._download_image")
    def test_falls_back_to_klein_when_schnell_fails(
        self, mock_dl, mock_schnell, mock_klein, tmp_path
    ):
        mock_schnell.side_effect = RuntimeError("Schnell unavailable")
        mock_klein.return_value = FAKE_IMG_URL
        mock_dl.return_value = tmp_path / "scene_01.jpg"

        dest = tmp_path / "scene_01.jpg"
        result = generate_image_from_prompt("sky panorama", dest)

        mock_schnell.assert_called_once()
        mock_klein.assert_called_once()
        assert result == dest

    @patch("agents.visual_agent.settings")
    def test_raises_value_error_on_missing_api_key(self, mock_settings, tmp_path):
        mock_settings.pixazo_api_key = ""
        with pytest.raises(ValueError, match="PIXAZO_API_KEY"):
            generate_image_from_prompt("test", tmp_path / "x.jpg")

    @patch("agents.visual_agent._generate_klein")
    @patch("agents.visual_agent._generate_schnell")
    def test_raises_runtime_error_when_both_models_fail(
        self, mock_schnell, mock_klein, tmp_path
    ):
        mock_schnell.side_effect = RuntimeError("Schnell failed")
        mock_klein.side_effect = RuntimeError("Klein failed")
        with pytest.raises(RuntimeError, match="Both Pixazo models failed"):
            generate_image_from_prompt("test", tmp_path / "x.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# generate_scene_images
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSceneImages:
    PROMPTS = [
        "Aerial view of a global summit, cinematic",
        "Data centre with blinking servers, dramatic lighting",
        "Busy stock market trading floor, wide angle",
    ]

    @patch("agents.visual_agent.time.sleep")
    @patch("agents.visual_agent.generate_image_from_prompt")
    def test_returns_one_path_per_prompt(self, mock_gen, mock_sleep, tmp_path):
        mock_gen.side_effect = [
            tmp_path / f"run_scene_0{i + 1}.jpg"
            for i in range(len(self.PROMPTS))
        ]
        paths = generate_scene_images(self.PROMPTS, tmp_path, "run")
        assert len(paths) == len(self.PROMPTS)
        assert mock_gen.call_count == len(self.PROMPTS)

    @patch("agents.visual_agent.time.sleep")
    @patch("agents.visual_agent.generate_image_from_prompt")
    def test_skips_existing_files(self, mock_gen, mock_sleep, tmp_path):
        # Pre-create the first scene file
        cached = tmp_path / "run_scene_01.jpg"
        cached.write_bytes(b"cached")

        mock_gen.return_value = tmp_path / "run_scene_02.jpg"

        paths = generate_scene_images(self.PROMPTS[:2], tmp_path, "run")

        # Only the second scene should trigger a generate call
        assert mock_gen.call_count == 1
        assert paths[0] == cached

    @patch("agents.visual_agent.time.sleep")
    @patch("agents.visual_agent.generate_image_from_prompt")
    def test_uses_previous_image_as_placeholder_on_failure(
        self, mock_gen, mock_sleep, tmp_path
    ):
        first_img = tmp_path / "run_scene_01.jpg"
        mock_gen.side_effect = [
            first_img,
            RuntimeError("API down"),
        ]
        paths = generate_scene_images(self.PROMPTS[:2], tmp_path, "run")

        # Second path should fall back to the first image
        assert paths[1] == first_img

    @patch("agents.visual_agent.time.sleep")
    @patch("agents.visual_agent.generate_image_from_prompt")
    def test_raises_when_first_scene_fails(self, mock_gen, mock_sleep, tmp_path):
        mock_gen.side_effect = RuntimeError("first scene failed")
        with pytest.raises(RuntimeError):
            generate_scene_images(self.PROMPTS[:1], tmp_path, "run")



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
