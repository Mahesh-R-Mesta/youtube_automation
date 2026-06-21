"""
Unit tests for agents/script_agent.py

Tests cover:
  • _parse_json() with/without markdown fences
  • parse_scenes() with |SCENE_N| markers
  • generate_spiritual_image_prompts() padding fallbacks
  • generate_spiritual_video_script() integration (all Gemini calls mocked)
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
    generate_spiritual_image_prompts,
    SpiritualScript,
)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_json
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJson:
    def test_plain_json_dict(self):
        raw = '{"title": "Gita 2:47", "caption": "Great teaching"}'
        result = _parse_json(raw)
        assert result["title"] == "Gita 2:47"

    def test_strips_json_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json(raw)
        assert result["key"] == "value"

    def test_strips_plain_code_fence(self):
        raw = '```\n["scene1", "scene2"]\n```'
        result = _parse_json(raw)
        assert result == ["scene1", "scene2"]

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json("not json at all ॐ")


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

    def test_each_scene_non_empty(self, sample_script):
        scenes = parse_scenes(sample_script)
        for scene in scenes:
            assert len(scene.strip()) > 0

    def test_handles_missing_markers(self):
        bad = "Just plain text, no markers."
        scenes = parse_scenes(bad)
        assert len(scenes) == 1

    def test_handles_extra_whitespace(self):
        script = "|SCENE_1|\n  First.  \n\n|SCENE_2|\n  Second.  "
        scenes = parse_scenes(script)
        assert len(scenes) == 2
        assert scenes[0] == "First."
        assert scenes[1] == "Second."


# ─────────────────────────────────────────────────────────────────────────────
# generate_spiritual_image_prompts — padding behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSpiritualImagePrompts:
    @patch("agents.script_agent._call_gemini")
    def test_returns_exactly_6_when_gemini_returns_4(self, mock_gemini, sample_script):
        mock_gemini.return_value = '["p1", "p2", "p3", "p4"]'
        result = generate_spiritual_image_prompts(
            script=sample_script,
            scripture="bhagavad_gita",
            image_style="tanjore",
        )
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_trims_to_6_when_gemini_returns_8(self, mock_gemini, sample_script):
        mock_gemini.return_value = json.dumps([f"prompt {i}" for i in range(8)])
        result = generate_spiritual_image_prompts(
            script=sample_script,
            scripture="puranas",
            image_style="cosmic",
        )
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_style_present_in_fallback(self, mock_gemini, sample_script):
        # Force fallback by returning only 1 prompt
        mock_gemini.return_value = '["only one prompt"]'
        result = generate_spiritual_image_prompts(
            script=sample_script,
            scripture="bhagavad_gita",
            image_style="cosmic",
        )
        # Padded prompts should contain the style name
        for prompt in result[1:]:
            assert "cosmic" in prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# generate_spiritual_video_script — integration (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSpiritualVideoScript:
    @patch("agents.script_agent._call_gemini")
    def test_returns_spiritual_script_dataclass(
        self, mock_gemini, sample_script, sample_topic_dict, sample_content_dict
    ):
        from agents.script_agent import generate_spiritual_video_script

        mock_meta = json.dumps({
            "title": "Nishkama Karma — Act Without Attachment",
            "caption": "Ancient wisdom for modern life.",
            "hashtags": [f"#tag{i}" for i in range(30)],
        })
        mock_prompts = json.dumps([f"spiritual prompt {i}" for i in range(6)])

        mock_gemini.side_effect = [
            sample_script,   # write_spiritual_script
            mock_meta,       # generate_spiritual_metadata
            mock_prompts,    # generate_spiritual_image_prompts
        ]

        result = generate_spiritual_video_script(
            topic_dict=sample_topic_dict,
            content_dict=sample_content_dict,
            image_style="tanjore",
        )

        assert isinstance(result, SpiritualScript)
        assert result.scripture == "bhagavad_gita"
        assert result.mode == "sloka"
        assert len(result.scenes) == 6
        assert len(result.scene_image_prompts) == 6
        assert result.title == "Nishkama Karma — Act Without Attachment"
        assert len(result.hashtags) == 30


import json
