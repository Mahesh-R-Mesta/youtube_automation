"""
Unit tests for agents/scripture_agent.py

Tests cover:
  • pick_auto_topic() with Gemini mocked to succeed
  • pick_auto_topic() falls back to curated list when Gemini fails
  • _fallback_topic() respects scripture and mode filters
  • get_scripture_content() with Gemini mocked
  • get_scripture_content() returns safe placeholder on failure
"""

import sys
import os
import json
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.scripture_agent import (
    pick_auto_topic,
    get_scripture_content,
    _fallback_topic,
    _SCRIPTURE_KEYS,
    _MODE_KEYS,
)


# ─────────────────────────────────────────────────────────────────────────────
class TestPickAutoTopicGeminiSuccess:
    @patch("agents.scripture_agent._call_gemini")
    def test_returns_dict_with_required_keys(self, mock_gemini):
        mock_gemini.return_value = json.dumps({
            "scripture": "bhagavad_gita",
            "mode": "sloka",
            "topic": "Bhagavad Gita 2:47",
            "scripture_ref": "Chapter 2, Verse 47",
            "hook": "The secret to a fulfilling life...",
            "selection_reason": "Universally resonant.",
        })
        result = pick_auto_topic()
        for key in ("scripture", "mode", "topic", "scripture_ref", "hook"):
            assert key in result

    @patch("agents.scripture_agent._call_gemini")
    def test_scripture_filter_passed_in_prompt(self, mock_gemini):
        mock_gemini.return_value = json.dumps({
            "scripture": "puranas",
            "mode": "story",
            "topic": "Samudra Manthan",
            "scripture_ref": "Vishnu Purana, Book 1",
            "hook": "Gods and demons churned the cosmic ocean.",
            "selection_reason": "Epic visual narrative.",
        })
        result = pick_auto_topic(scripture="puranas")
        assert result["scripture"] == "puranas"
        # Verify the prompt included the scripture constraint
        call_args = mock_gemini.call_args[0][0]
        assert "puranas" in call_args.lower()


class TestPickAutoTopicFallback:
    @patch("agents.scripture_agent._call_gemini", side_effect=Exception("network error"))
    def test_falls_back_to_curated_when_gemini_fails(self, _mock_gemini):
        result = pick_auto_topic()
        required = {"scripture", "mode", "topic", "scripture_ref", "hook"}
        assert required.issubset(result.keys())
        assert result["scripture"] in _SCRIPTURE_KEYS
        assert result["mode"] in _MODE_KEYS

    @patch("agents.scripture_agent._call_gemini", side_effect=Exception("timeout"))
    def test_fallback_respects_scripture_filter(self, _mock):
        for scripture in _SCRIPTURE_KEYS:
            result = pick_auto_topic(scripture=scripture)
            assert result["scripture"] == scripture

    @patch("agents.scripture_agent._call_gemini", side_effect=Exception("timeout"))
    def test_fallback_respects_mode_filter(self, _mock):
        for mode in _MODE_KEYS:
            result = pick_auto_topic(mode=mode)
            assert result["mode"] == mode


class TestFallbackTopic:
    def test_returns_dict_with_required_keys(self):
        result = _fallback_topic()
        for key in ("scripture", "mode", "topic", "scripture_ref", "hook"):
            assert key in result

    def test_filters_by_scripture(self):
        for scripture in _SCRIPTURE_KEYS:
            result = _fallback_topic(scripture=scripture)
            assert result["scripture"] == scripture

    def test_filters_by_mode(self):
        for mode in _MODE_KEYS:
            result = _fallback_topic(mode=mode)
            assert result["mode"] == mode

    def test_invalid_scripture_returns_any_topic(self):
        # Unknown key should still return something from the full pool
        result = _fallback_topic(scripture="nonexistent_scripture")
        assert result["scripture"] in _SCRIPTURE_KEYS


class TestGetScriptureContent:
    @patch("agents.scripture_agent._call_gemini")
    def test_returns_parsed_content(self, mock_gemini):
        mock_gemini.return_value = json.dumps({
            "sanskrit_devanagari": "कर्मण्येवाधिकारस्ते",
            "transliteration": "karmany evadhikaras te",
            "word_by_word": "karma=action, evadhikaras=right",
            "english_meaning": "You have the right to perform your duty.",
            "context": "Spoken by Krishna to Arjuna.",
        })
        topic_dict = {"topic": "Gita 2:47", "scripture_ref": "Chapter 2, V47", "mode": "sloka"}
        result = get_scripture_content(topic_dict)
        assert result["transliteration"] == "karmany evadhikaras te"
        assert "कर्मण्" in result["sanskrit_devanagari"]

    @patch("agents.scripture_agent._call_gemini", side_effect=Exception("API error"))
    def test_returns_safe_placeholder_on_error(self, _mock):
        topic_dict = {"topic": "Gita 2:47", "scripture_ref": "Chapter 2", "mode": "sloka"}
        result = get_scripture_content(topic_dict)
        # Should not raise; returns minimal placeholder
        assert "sanskrit_devanagari" in result
        assert "ॉ" in result["sanskrit_devanagari"]   # Om placeholder
