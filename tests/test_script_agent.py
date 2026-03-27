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
    generate_scene_image_prompts,
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
# generate_scene_image_prompts — padding logic (Gemini mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSceneImagePrompts:
    @patch("agents.script_agent._call_gemini")
    def test_pads_to_6_when_fewer_returned(self, mock_gemini):
        """If LLM returns only 4 prompts, the function must pad to 6."""
        mock_gemini.return_value = '["cinematic climate summit", "solar panels wide", "polar ice landscape", "protest crowd"]'

        result = generate_scene_image_prompts("any script text")
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_truncates_to_6_when_more_returned(self, mock_gemini):
        """If LLM returns 8 prompts, only the first 6 should be used."""
        prompts = [f"cinematic scene {i}, photorealistic" for i in range(8)]
        mock_gemini.return_value = json.dumps(prompts)

        result = generate_scene_image_prompts("any script text")
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_returns_exactly_6_when_correct(self, mock_gemini):
        prompts = [f"cinematic scene {i}, photorealistic" for i in range(6)]
        mock_gemini.return_value = json.dumps(prompts)

        result = generate_scene_image_prompts("any script text")
        assert result == prompts


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
            # 4. generate_scene_image_prompts
            json.dumps([
                "aerial view of a climate summit, cinematic wide angle, 8K",
                "solar energy farm at dusk, photorealistic landscape",
                "polar ice melting, documentary drone shot, dramatic",
                "environmental protest crowd, silhouette against golden sky",
                "modern parliament building exterior, wide establishing shot",
                "earth from orbit with glowing city lights, space photography",
            ]),
        ]

    @patch("agents.script_agent._call_gemini")
    def test_returns_video_script_dataclass(self, mock_call, sample_headlines):
        self._mock_gemini_responses(mock_call)

        result = generate_video_script(sample_headlines)

        assert isinstance(result, VideoScript)
        assert result.topic == "Historic Climate Deal Signed by 195 Nations"
        assert len(result.scenes) == 6
        assert len(result.scene_image_prompts) == 6
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
            json.dumps([f"cinematic scene {i}, photorealistic 8K" for i in range(6)]),
        ]

        result = generate_video_script(sample_headlines)
        assert len(result.title) <= 100


# ─────────────────────────────────────────────────────────────────────────────
# write_script — language directive
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteScriptLanguage:
    @patch("agents.script_agent._call_gemini")
    def test_hindi_directive_injected_into_news_script_prompt(self, mock_gemini):
        mock_gemini.return_value = "|SCENE_1|\nX.\n|SCENE_2|\nX.\n|SCENE_3|\nX.\n|SCENE_4|\nX.\n|SCENE_5|\nX.\n|SCENE_6|\nX."
        from agents.script_agent import write_script
        write_script("AI revolution", language="hindi")
        prompt_used = mock_gemini.call_args[0][0]
        assert "Hindi" in prompt_used

    @patch("agents.script_agent._call_gemini")
    def test_english_directive_injected_by_default(self, mock_gemini):
        mock_gemini.return_value = "|SCENE_1|\nX.\n|SCENE_2|\nX.\n|SCENE_3|\nX.\n|SCENE_4|\nX.\n|SCENE_5|\nX.\n|SCENE_6|\nX."
        from agents.script_agent import write_script
        write_script("AI revolution")
        prompt_used = mock_gemini.call_args[0][0]
        assert "English" in prompt_used


# ─────────────────────────────────────────────────────────────────────────────
# write_history_script
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteHistoryScript:
    _BRIEF = {
        "topic": "The Dancing Plague of 1518",
        "era": "Early Modern Europe",
        "region": "Holy Roman Empire",
        "hook": "In the summer of 1518, hundreds of people could not stop dancing.",
        "key_figures": [{"name": "Frau Troffea", "role": "First dancer", "significance": "Started the epidemic"}],
        "timeline": [{"date": "July 1518", "event": "Frau Troffea begins dancing", "impact": "Triggered mass hysteria"}],
        "surprising_facts": ["Doctors prescribed more dancing as a cure.", "The city hired musicians to keep them going."],
        "turning_point": "When city authorities endorsed the dancing, it spread uncontrollably.",
        "legacy": "The event is studied as one of history's strangest cases of mass psychogenic illness.",
    }

    @patch("agents.script_agent._call_gemini")
    def test_returns_string_with_scene_markers(self, mock_gemini):
        mock_gemini.return_value = (
            "|SCENE_1|\nHook text.\n\n"
            "|SCENE_2|\nContext.\n\n"
            "|SCENE_3|\nRising tension.\n\n"
            "|SCENE_4|\nKey figure.\n\n"
            "|SCENE_5|\nThe turning point.\n\n"
            "|SCENE_6|\nLegacy. Subscribe for more."
        )
        from agents.script_agent import write_history_script
        script = write_history_script(self._BRIEF)
        assert isinstance(script, str)
        assert "|SCENE_1|" in script
        assert "|SCENE_6|" in script

    @patch("agents.script_agent._call_gemini")
    def test_injects_hook_into_prompt(self, mock_gemini):
        mock_gemini.return_value = "|SCENE_1|\nHook.\n|SCENE_2|\nText.\n|SCENE_3|\nText.\n|SCENE_4|\nText.\n|SCENE_5|\nText.\n|SCENE_6|\nText."
        from agents.script_agent import write_history_script
        write_history_script(self._BRIEF)
        prompt_used = mock_gemini.call_args[0][0]
        assert self._BRIEF["hook"] in prompt_used

    @patch("agents.script_agent._call_gemini")
    def test_hindi_directive_injected_into_history_script_prompt(self, mock_gemini):
        mock_gemini.return_value = "|SCENE_1|\nX.\n|SCENE_2|\nX.\n|SCENE_3|\nX.\n|SCENE_4|\nX.\n|SCENE_5|\nX.\n|SCENE_6|\nX."
        from agents.script_agent import write_history_script
        write_history_script(self._BRIEF, language="hindi")
        prompt_used = mock_gemini.call_args[0][0]
        assert "Hindi" in prompt_used

    @patch("agents.script_agent._call_gemini")
    def test_english_directive_injected_when_language_is_english(self, mock_gemini):
        mock_gemini.return_value = "|SCENE_1|\nX.\n|SCENE_2|\nX.\n|SCENE_3|\nX.\n|SCENE_4|\nX.\n|SCENE_5|\nX.\n|SCENE_6|\nX."
        from agents.script_agent import write_history_script
        write_history_script(self._BRIEF, language="english")
        prompt_used = mock_gemini.call_args[0][0]
        assert "English" in prompt_used


# ─────────────────────────────────────────────────────────────────────────────
# generate_history_image_prompts
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateHistoryImagePrompts:
    @patch("agents.script_agent._call_gemini")
    def test_returns_6_prompts(self, mock_gemini):
        prompts = [f"oil painting of scene {i}, {i} era" for i in range(6)]
        mock_gemini.return_value = json.dumps(prompts)
        from agents.script_agent import generate_history_image_prompts
        result = generate_history_image_prompts("script text", "Medieval", "Europe")
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_pads_to_6_when_fewer_returned(self, mock_gemini):
        mock_gemini.return_value = json.dumps(["only three prompts"] * 3)
        from agents.script_agent import generate_history_image_prompts
        result = generate_history_image_prompts("script", "Ancient", "Egypt")
        assert len(result) == 6

    @patch("agents.script_agent._call_gemini")
    def test_injects_era_and_region_into_prompt(self, mock_gemini):
        mock_gemini.return_value = json.dumps([f"prompt {i}" for i in range(6)])
        from agents.script_agent import generate_history_image_prompts
        generate_history_image_prompts("script", "Victorian England", "British Empire")
        prompt_used = mock_gemini.call_args[0][0]
        assert "Victorian England" in prompt_used
        assert "British Empire" in prompt_used


# ─────────────────────────────────────────────────────────────────────────────
# generate_history_video_script — full integration (all mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateHistoryVideoScript:
    def _mock_all(self, mock_call, mock_pick, mock_research):
        mock_pick.return_value = {
            "topic": "The Dancing Plague of 1518",
            "era": "Early Modern Europe",
            "region": "Holy Roman Empire",
            "hook": "In the summer of 1518, hundreds of people could not stop dancing.",
        }
        mock_research.return_value = {
            "topic": "The Dancing Plague of 1518",
            "era": "Early Modern Europe",
            "region": "Holy Roman Empire",
            "hook": "In the summer of 1518, hundreds of people could not stop dancing.",
            "key_figures": [],
            "timeline": [],
            "surprising_facts": [],
            "turning_point": "The city endorsed the dancing.",
            "legacy": "Studied as mass psychogenic illness.",
        }
        script_text = (
            "|SCENE_1|\nHook text here.\n\n"
            "|SCENE_2|\nContext.\n\n"
            "|SCENE_3|\nTension.\n\n"
            "|SCENE_4|\nFigures.\n\n"
            "|SCENE_5|\nClimax.\n\n"
            "|SCENE_6|\nLegacy. Subscribe for more hidden histories."
        )
        mock_call.side_effect = [
            script_text,  # write_history_script
            json.dumps({  # generate_seo_metadata
                "title": "The Dancing Plague of 1518 | Forgotten History",
                "description": "A strange epidemic swept through Strasbourg.",
                "tags": [f"tag{i}" for i in range(15)],
                "thumbnail_text": "DANCING PLAGUE",
            }),
            json.dumps([f"oil painting scene {i}, medieval style" for i in range(6)]),  # image prompts
        ]

    @patch("agents.history_agent.research_topic")
    @patch("agents.history_agent.pick_history_topic")
    @patch("agents.script_agent._call_gemini")
    def test_returns_history_script_dataclass(self, mock_call, mock_pick, mock_research):
        from agents.script_agent import generate_history_video_script, HistoryScript
        self._mock_all(mock_call, mock_pick, mock_research)
        result = generate_history_video_script()
        assert isinstance(result, HistoryScript)

    @patch("agents.history_agent.research_topic")
    @patch("agents.history_agent.pick_history_topic")
    @patch("agents.script_agent._call_gemini")
    def test_result_has_era_and_region(self, mock_call, mock_pick, mock_research):
        from agents.script_agent import generate_history_video_script, HistoryScript
        self._mock_all(mock_call, mock_pick, mock_research)
        result = generate_history_video_script()
        assert result.era == "Early Modern Europe"
        assert result.region == "Holy Roman Empire"

    @patch("agents.history_agent.research_topic")
    @patch("agents.history_agent.pick_history_topic")
    @patch("agents.script_agent._call_gemini")
    def test_result_has_6_scenes_and_6_prompts(self, mock_call, mock_pick, mock_research):
        from agents.script_agent import generate_history_video_script
        self._mock_all(mock_call, mock_pick, mock_research)
        result = generate_history_video_script()
        assert len(result.scenes) == 6
        assert len(result.scene_image_prompts) == 6

    @patch("agents.history_agent.research_topic")
    @patch("agents.history_agent.pick_history_topic")
    @patch("agents.script_agent._call_gemini")
    def test_passes_era_to_pick_history_topic(self, mock_call, mock_pick, mock_research):
        from agents.script_agent import generate_history_video_script
        self._mock_all(mock_call, mock_pick, mock_research)
        generate_history_video_script(era="Ancient Rome", theme="military")
        mock_pick.assert_called_once_with(era="Ancient Rome", theme="military")

