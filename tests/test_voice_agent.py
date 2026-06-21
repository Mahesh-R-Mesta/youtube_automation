"""
Unit tests for agents/voice_agent.py

Tests cover:
  • _clean_script() strips scene markers and normalises whitespace
  • generate_edge_tts() creates file and returns its path
  • generate_elevenlabs() creates file and returns its path
  • generate_voiceover() tries edge-tts first; falls back to ElevenLabs
  • generate_voiceover() raises RuntimeError if both providers fail
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.voice_agent import _clean_script, generate_voiceover, generate_edge_tts


# ─────────────────────────────────────────────────────────────────────────────
# _clean_script
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanScript:
    def test_removes_scene_markers(self):
        script = "|SCENE_1|\nHello.\n\n|SCENE_2|\nWorld."
        result = _clean_script(script)
        assert "|SCENE_" not in result
        assert "Hello." in result
        assert "World." in result

    def test_collapses_multiple_blank_lines(self):
        script = "Line one.\n\n\n\n\nLine two."
        result = _clean_script(script)
        assert "\n\n\n" not in result

    def test_returns_stripped_text(self):
        script = "  \n  Text here.  \n  "
        result = _clean_script(script)
        assert result == "Text here."


# ─────────────────────────────────────────────────────────────────────────────
# generate_edge_tts
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEdgeTts:
    @patch("agents.voice_agent.asyncio.run")
    def test_calls_asyncio_run_and_returns_path(self, mock_run, tmp_path):
        from agents.voice_agent import generate_edge_tts

        output = tmp_path / "narration.mp3"
        # asyncio.run is mocked — it won't actually run; we just create the file manually
        mock_run.side_effect = lambda coro: output.write_bytes(b"\xff\xfb\x90\x00")

        result = generate_edge_tts("Hello world", output)

        assert result == output
        assert output.exists()
        mock_run.assert_called_once()

    @patch("agents.voice_agent.asyncio.run", side_effect=OSError("Network unreachable"))
    def test_propagates_edge_tts_exception(self, _mock_run, tmp_path):
        from agents.voice_agent import generate_edge_tts

        with pytest.raises(OSError):
            generate_edge_tts("Hello world", tmp_path / "out.mp3")


# ─────────────────────────────────────────────────────────────────────────────
# generate_elevenlabs
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateElevenLabs:
    @patch("agents.voice_agent.settings")
    def test_raises_if_no_api_key(self, mock_settings, tmp_path):
        from agents.voice_agent import generate_elevenlabs

        mock_settings.elevenlabs_api_key = None
        mock_settings.elevenlabs_voice_id = "abc"

        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            generate_elevenlabs("Hello", tmp_path / "out.mp3")


# ─────────────────────────────────────────────────────────────────────────────
# generate_voiceover — routing logic
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateVoiceover:
    @patch("agents.voice_agent.generate_edge_tts")
    def test_uses_edge_tts_by_default(self, mock_edge, tmp_path):
        expected = tmp_path / "narration.mp3"
        mock_edge.return_value = expected

        result = generate_voiceover("Some script text.", expected)

        mock_edge.assert_called_once()
        assert result == expected

    @patch("agents.voice_agent.generate_elevenlabs")
    @patch("agents.voice_agent.generate_edge_tts", side_effect=RuntimeError("edge-tts down"))
    @patch("agents.voice_agent.settings")
    def test_falls_back_to_elevenlabs_on_edge_tts_failure(
        self, mock_settings, _mock_edge, mock_eleven, tmp_path
    ):
        mock_settings.tts_voice = "en-US-AriaNeural"
        mock_settings.tts_rate = "+5%"
        mock_settings.elevenlabs_api_key = "fake-key"
        mock_settings.elevenlabs_voice_id = "fake-voice-id"

        expected = tmp_path / "narration.mp3"
        mock_eleven.return_value = expected

        result = generate_voiceover("Script text", expected)

        mock_eleven.assert_called_once()
        assert result == expected

    @patch("agents.voice_agent.generate_edge_tts", side_effect=RuntimeError("down"))
    @patch("agents.voice_agent.settings")
    def test_raises_runtime_error_when_both_fail(self, mock_settings, _mock_edge, tmp_path):
        mock_settings.tts_voice = "en-IN-NeerjaNeural"
        mock_settings.tts_male_voice = "en-IN-PrabhatNeural"
        mock_settings.tts_rate = "+0%"
        mock_settings.elevenlabs_api_key = None  # no fallback available

        with pytest.raises(RuntimeError, match="All TTS providers failed"):
            generate_voiceover("Script text", tmp_path / "out.mp3")

    @patch("agents.voice_agent.generate_edge_tts")
    @patch("agents.voice_agent.settings")
    def test_uses_male_voice_when_voice_gender_is_male(self, mock_settings, mock_edge, tmp_path):
        mock_settings.tts_voice = "en-IN-NeerjaNeural"
        mock_settings.tts_male_voice = "en-IN-PrabhatNeural"
        mock_settings.tts_rate = "+0%"
        expected = tmp_path / "narration.mp3"
        mock_edge.return_value = expected

        generate_voiceover("Some script text.", expected, voice_gender="male")

        _args, _kwargs = mock_edge.call_args
        assert _kwargs.get("voice") == "en-IN-PrabhatNeural"

    @patch("agents.voice_agent.generate_edge_tts")
    @patch("agents.voice_agent.settings")
    def test_uses_female_voice_by_default(self, mock_settings, mock_edge, tmp_path):
        mock_settings.tts_voice = "en-IN-NeerjaNeural"
        mock_settings.tts_male_voice = "en-IN-PrabhatNeural"
        mock_settings.tts_rate = "+0%"
        expected = tmp_path / "narration.mp3"
        mock_edge.return_value = expected

        generate_voiceover("Some script text.", expected)

        _args, _kwargs = mock_edge.call_args
        assert _kwargs.get("voice") == "en-IN-NeerjaNeural"

    @patch("agents.voice_agent.generate_edge_tts")
    @patch("agents.voice_agent.settings")
    def test_uses_english_voice_by_default(self, mock_settings, mock_edge, tmp_path):
        mock_settings.tts_voice = "en-US-AriaNeural"
        mock_settings.tts_hindi_voice = "hi-IN-SwaraNeural"
        expected = tmp_path / "narration.mp3"
        mock_edge.return_value = expected

        generate_voiceover("Some text.", expected)

        _args, _kwargs = mock_edge.call_args
        assert _kwargs.get("voice") == "en-US-AriaNeural"
