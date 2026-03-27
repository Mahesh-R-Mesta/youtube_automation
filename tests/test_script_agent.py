"""
Unit tests for agents/script_agent.py

Tests cover:
  • JSON response parsing (with and without markdown fences)
  • Scene parsing with |SCENE_N| markers
  • Keyword padding when LLM returns fewer than 6 items
  • generate_video_script() integration (all Gemini calls mocked)
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.script_agent import (
    _parse_json,
    parse_scenes,
    extract_scene_keywords,
    generate_video_script,
    VideoScript,
)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_json
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJson:
    def test_plain_json_dict(self):
        raw = '{"title": "Hello", "tags": ["a", "b"]}'
        result = _parse_json(raw)
        assert result["title"] == "Hello"
        assert result["tags"] == ["a", "b"]

    def test_strips_json_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json(raw)
        assert result["key"] == "value"

    def test_strips_plain_code_fence(self):
        raw = '```\n["item1", "item2"]\n```'
        result = _parse_json(raw)
        assert result == ["item1", "item2"]

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json("this is not json at all")


# ─────────────────────────────────────────────────────────────────────────────
# parse_scenes
# ─────────────────────────────────────────────────────────────────────────────

class TestParseScenes:
    def test_splits_six_scenes(self, sample_script):
        scenes = parse_scenes(sample_script)
        assert len(scenes) == 6

    def test_scenes_do_not_contain_marker(self, sample_script):
        scenes = parse_scenes(sample_script)
        for scene in scenes:
            assert "|SCENE_" not in scene

    def test_each_scene_is_non_empty(self, sample_script):
        scenes = parse_scenes(sample_script)
        for scene in scenes:
            assert len(scene.strip()) > 0

    def test_handles_missing_markers(self):
        bad_script = "Just a single block of text with no markers."
        scenes = parse_scenes(bad_script)
        # Should return [bad_script] — the whole text as one "scene"
        assert len(scenes) == 1

    def test_handles_extra_whitespace_between_scenes(self):
        script = "|SCENE_1|\n  First scene text.  \n\n|SCENE_2|\n  Second scene.  "
        scenes = parse_scenes(script)
        assert len(scenes) == 2
        assert scenes[0] == "First scene text."
        assert scenes[1] == "Second scene."


# ─────────────────────────────────────────────────────────────────────────────
# extract_scene_keywords — padding logic (Gemini mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractSceneKeywords:
    @patch("agents.script_agent._call_gemini")
    def test_pads_to_6_when_fewer_returned(self, mock_gemini):
        """If LLM returns only 4 keywords, the function must pad to 6."""
        mock_gemini.return_value = '["climate summit", "solar panels", "polar ice", "protest march"]'

        result = extract_scene_keywords("any script text")
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_truncates_to_6_when_more_returned(self, mock_gemini):
        """If LLM returns 8 keywords, only the first 6 should be used."""
        words = [f"keyword {i}" for i in range(8)]
        mock_gemini.return_value = json.dumps(words)

        result = extract_scene_keywords("any script text")
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_returns_exactly_6_when_correct(self, mock_gemini):
        words = [f"keyword {i}" for i in range(6)]
        mock_gemini.return_value = json.dumps(words)

        result = extract_scene_keywords("any script text")
        assert result == words


# ─────────────────────────────────────────────────────────────────────────────
# generate_video_script — full integration (all Gemini mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateVideoScript:
    def _mock_gemini_responses(self, mock_call):
        """Configure sequential Gemini mock responses for the 4 pipeline calls."""
        mock_call.side_effect = [
            # 1. select_topic
            json.dumps({
                "selected_headline": "Historic Climate Deal Signed by 195 Nations",
                "explanation": "Broad appeal, visual story potential.",
            }),
            # 2. write_script
            (
                "|SCENE_1|\nClimate deal hook.\n\n"
                "|SCENE_2|\nKey facts.\n\n"
                "|SCENE_3|\nScientist reaction.\n\n"
                "|SCENE_4|\nCriticism.\n\n"
                "|SCENE_5|\nImplementation challenges.\n\n"
                "|SCENE_6|\nCall to action. Like and subscribe!"
            ),
            # 3. generate_seo_metadata
            json.dumps({
                "title": "Historic Climate Deal Changes Everything | News 2026",
                "description": "Paragraph one.\nParagraph two.\nParagraph three.\n#WorldNews",
                "tags": [f"tag{i}" for i in range(15)],
                "thumbnail_text": "CLIMATE DEAL",
            }),
            # 4. extract_scene_keywords
            json.dumps([
                "climate summit meeting", "solar energy farm", "polar ice melting",
                "environmental protest", "government parliament", "earth satellite view",
            ]),
        ]

    @patch("agents.script_agent._call_gemini")
    def test_returns_video_script_dataclass(self, mock_call, sample_headlines):
        self._mock_gemini_responses(mock_call)

        result = generate_video_script(sample_headlines)

        assert isinstance(result, VideoScript)
        assert result.topic == "Historic Climate Deal Signed by 195 Nations"
        assert len(result.scenes) == 6
        assert len(result.scene_keywords) == 6
        assert len(result.tags) <= 15
        assert result.title != ""
        assert result.thumbnail_text != ""

    @patch("agents.script_agent._call_gemini")
    def test_title_is_truncated_to_100_chars(self, mock_call, sample_headlines):
        """SEO title must not exceed YouTube's 100-char limit."""
        long_title = "A" * 200

        mock_call.side_effect = [
            json.dumps({"selected_headline": "Topic", "explanation": "x"}),
            "|SCENE_1|\nText.\n|SCENE_2|\nText.\n|SCENE_3|\nText.\n|SCENE_4|\nText.\n|SCENE_5|\nText.\n|SCENE_6|\nText.",
            json.dumps({
                "title": long_title,
                "description": "desc",
                "tags": ["t"] * 15,
                "thumbnail_text": "TEXT",
            }),
            json.dumps([f"keyword {i}" for i in range(6)]),
        ]

        result = generate_video_script(sample_headlines)
        assert len(result.title) <= 100
