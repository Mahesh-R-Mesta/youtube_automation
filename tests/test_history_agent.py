"""
Unit tests for agents/history_agent.py

Tests cover:
  • pick_history_topic() returns correct keys from Gemini
  • pick_history_topic() falls back to curated list when Gemini fails
  • pick_history_topic() with fallback_index returns deterministic result
  • pick_history_topic() passes era/theme filters into the prompt
  • research_topic() returns correct keys from Gemini
  • research_topic() returns stub brief when Gemini fails
  • research_topic() preserves hook from topic_dict
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.history_agent import pick_history_topic, research_topic, _FALLBACK_TOPICS


# ─────────────────────────────────────────────────────────────────────────────
# pick_history_topic
# ─────────────────────────────────────────────────────────────────────────────

class TestPickHistoryTopic:
    def _valid_topic_response(self) -> str:
        return json.dumps({
            "topic": "The Plague of Justinian",
            "era": "Late Antiquity",
            "region": "Byzantine Empire",
            "hook": "It killed half the known world and no one remembers it.",
        })

    @patch("agents.history_agent._call_gemini")
    def test_returns_required_keys(self, mock_gemini):
        mock_gemini.return_value = self._valid_topic_response()
        result = pick_history_topic()
        assert {"topic", "era", "region", "hook"}.issubset(result.keys())
        assert result["topic"] == "The Plague of Justinian"

    @patch("agents.history_agent._call_gemini")
    def test_passes_era_filter_in_prompt(self, mock_gemini):
        mock_gemini.return_value = self._valid_topic_response()
        pick_history_topic(era="Ancient Rome")
        prompt_used = mock_gemini.call_args[0][0]
        assert "Ancient Rome" in prompt_used

    @patch("agents.history_agent._call_gemini")
    def test_passes_theme_filter_in_prompt(self, mock_gemini):
        mock_gemini.return_value = self._valid_topic_response()
        pick_history_topic(theme="forgotten women")
        prompt_used = mock_gemini.call_args[0][0]
        assert "forgotten women" in prompt_used

    @patch("agents.history_agent._call_gemini")
    def test_falls_back_to_curated_list_on_gemini_failure(self, mock_gemini):
        mock_gemini.side_effect = RuntimeError("API unavailable")
        result = pick_history_topic()
        # Must still return a valid topic dict from the fallback list
        assert {"topic", "era", "region", "hook"}.issubset(result.keys())
        assert result["topic"] in [t["topic"] for t in _FALLBACK_TOPICS]

    @patch("agents.history_agent._call_gemini")
    def test_falls_back_when_response_missing_keys(self, mock_gemini):
        mock_gemini.return_value = json.dumps({"topic": "Mystery Topic"})
        result = pick_history_topic()
        # Missing 'era', 'region', 'hook' → should trigger fallback
        assert result["topic"] in [t["topic"] for t in _FALLBACK_TOPICS]

    def test_fallback_index_returns_deterministic_result(self):
        result = pick_history_topic(fallback_index=0)
        assert result == _FALLBACK_TOPICS[0]

    def test_fallback_index_wraps_around(self):
        n = len(_FALLBACK_TOPICS)
        result_a = pick_history_topic(fallback_index=0)
        result_b = pick_history_topic(fallback_index=n)  # wraps to 0
        assert result_a == result_b


# ─────────────────────────────────────────────────────────────────────────────
# research_topic
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchTopic:
    _TOPIC_DICT = {
        "topic": "The Plague of Justinian",
        "era": "Late Antiquity",
        "region": "Byzantine Empire",
        "hook": "It killed half the known world and no one remembers it.",
    }

    def _valid_brief_response(self) -> str:
        return json.dumps({
            "topic": "The Plague of Justinian",
            "era": "Late Antiquity",
            "region": "Byzantine Empire",
            "key_figures": [
                {"name": "Justinian I", "role": "Emperor", "significance": "Ruled during the outbreak"},
                {"name": "Procopius", "role": "Historian", "significance": "Primary eyewitness account"},
            ],
            "timeline": [
                {"date": "541 AD", "event": "First appearance in Egypt", "impact": "Began spreading west"},
                {"date": "542 AD", "event": "Reached Constantinople", "impact": "Killed up to 10,000/day"},
            ],
            "surprising_facts": [
                "It may have killed up to 50 million people — a third of Europe",
                "Emperor Justinian himself survived after contracting the plague",
                "The outbreak recurred in waves for 200 years",
            ],
            "turning_point": "When the plague reached Constantinople in 542 AD, it paralyzed the entire administrative machine of the empire.",
            "legacy": "The Plague of Justinian ended Rome's last chance at reunification and accelerated the rise of the medieval world.",
        })

    @patch("agents.history_agent._call_gemini")
    def test_returns_required_keys(self, mock_gemini):
        mock_gemini.return_value = self._valid_brief_response()
        brief = research_topic(self._TOPIC_DICT)
        required = {"topic", "era", "region", "key_figures", "timeline",
                    "surprising_facts", "turning_point", "legacy"}
        assert required.issubset(brief.keys())

    @patch("agents.history_agent._call_gemini")
    def test_preserves_hook_from_topic_dict(self, mock_gemini):
        mock_gemini.return_value = self._valid_brief_response()
        brief = research_topic(self._TOPIC_DICT)
        assert brief["hook"] == self._TOPIC_DICT["hook"]

    @patch("agents.history_agent._call_gemini")
    def test_key_figures_is_a_list(self, mock_gemini):
        mock_gemini.return_value = self._valid_brief_response()
        brief = research_topic(self._TOPIC_DICT)
        assert isinstance(brief["key_figures"], list)
        assert len(brief["key_figures"]) > 0

    @patch("agents.history_agent._call_gemini")
    def test_timeline_is_a_list(self, mock_gemini):
        mock_gemini.return_value = self._valid_brief_response()
        brief = research_topic(self._TOPIC_DICT)
        assert isinstance(brief["timeline"], list)

    @patch("agents.history_agent._call_gemini")
    def test_returns_stub_brief_on_gemini_failure(self, mock_gemini):
        mock_gemini.side_effect = RuntimeError("API unavailable")
        brief = research_topic(self._TOPIC_DICT)
        # Must still return a usable dict with all required keys
        required = {"topic", "era", "region", "key_figures", "timeline",
                    "surprising_facts", "turning_point", "legacy"}
        assert required.issubset(brief.keys())
        assert brief["topic"] == self._TOPIC_DICT["topic"]

    @patch("agents.history_agent._call_gemini")
    def test_stub_includes_hook_on_failure(self, mock_gemini):
        mock_gemini.side_effect = RuntimeError("API unavailable")
        brief = research_topic(self._TOPIC_DICT)
        assert brief["hook"] == self._TOPIC_DICT["hook"]
