"""
Unit tests for pipeline/video_editor.py

Tests cover:
  • _ken_burns_clip() produces a VideoClip with the correct duration
  • _ken_burns_clip() all four directions produce valid frames without error
  • assemble_video(animated=False) calls ImageClip (static path; mocked)
  • assemble_video(animated=True) calls _ken_burns_clip (animated path; mocked)
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.video_editor import _ken_burns_clip, _KB_DIRECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_portrait_frame(w: int = 108, h: int = 192) -> np.ndarray:
    """Create a small solid-colour portrait test frame (H, W, 3)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 128  # red channel — distinguishable test pattern
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# _ken_burns_clip
# ─────────────────────────────────────────────────────────────────────────────

class TestKenBurnsClip:
    def test_clip_has_correct_duration(self):
        frame = _make_portrait_frame()
        clip = _ken_burns_clip(frame, duration=5.0, direction="zoom_in")
        assert abs(clip.duration - 5.0) < 0.01
        clip.close()

    @pytest.mark.parametrize("direction", _KB_DIRECTIONS)
    def test_all_directions_produce_valid_frame(self, direction):
        frame = _make_portrait_frame(w=108, h=192)
        clip = _ken_burns_clip(frame, duration=3.0, direction=direction)
        f0 = clip.get_frame(0.0)
        fmid = clip.get_frame(1.5)
        fend = clip.get_frame(2.99)
        assert f0.shape == (192, 108, 3)
        assert fmid.shape == (192, 108, 3)
        assert fend.shape == (192, 108, 3)
        clip.close()

    @pytest.mark.parametrize("direction", _KB_DIRECTIONS)
    def test_output_dtype_is_uint8(self, direction):
        frame = _make_portrait_frame(w=108, h=192)
        clip = _ken_burns_clip(frame, duration=2.0, direction=direction)
        f = clip.get_frame(0.5)
        assert f.dtype == np.uint8
        clip.close()

    def test_zoom_in_centre_frame_differs_between_start_and_end(self):
        frame = _make_portrait_frame(w=108, h=192)
        for y in range(192):
            for x in range(108):
                frame[y, x] = [x * 2 % 256, y % 256, 50]

        clip = _ken_burns_clip(frame, duration=4.0, direction="zoom_in")
        f_start = clip.get_frame(0.0)
        f_end = clip.get_frame(3.99)
        assert not np.array_equal(f_start, f_end)
        clip.close()

    def test_directions_cycle_via_modulus(self):
        assert len(_KB_DIRECTIONS) == 4
        assert "zoom_in" in _KB_DIRECTIONS
        assert "zoom_out" in _KB_DIRECTIONS
        assert "pan_up" in _KB_DIRECTIONS
        assert "pan_down" in _KB_DIRECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# assemble_video — Ken Burns integration
# ─────────────────────────────────────────────────────────────────────────────

class TestAssembleVideoAnimated:
    """
    We mock the heavy parts (audio loading, write_videofile) so these tests
    run without real media files, while still verifying the animated code path
    is exercised.
    """

    @patch("pipeline.video_editor._ken_burns_clip")
    @patch("pipeline.video_editor.ImageClip")
    @patch("pipeline.video_editor.get_audio_duration", return_value=12.0)
    @patch("pipeline.video_editor.AudioFileClip")
    @patch("pipeline.video_editor.concatenate_videoclips")
    def test_animated_true_calls_ken_burns(
        self,
        mock_concat,
        mock_audio_cls,
        mock_duration,
        mock_image_clip,
        mock_kb,
        tmp_path,
    ):
        """When animated=True, _ken_burns_clip should be called, not ImageClip."""
        from pipeline.video_editor import assemble_video

        # Build 6 tiny valid JPEG images
        from PIL import Image
        image_paths = []
        for i in range(6):
            p = tmp_path / f"scene_{i:02d}.jpg"
            Image.new("RGB", (64, 36), color=(i * 40, 100, 200)).save(str(p))
            image_paths.append(p)

        scenes = [f"Scene {i} narration text." for i in range(6)]
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"\xff\xfb\x90\x00" * 50)   # placeholder bytes
        output_path = tmp_path / "output.mp4"

        # Wire up mocks so write_videofile doesn't actually run
        mock_kb.return_value = MagicMock(duration=2.0)
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 12.0
        mock_audio_cls.return_value = mock_audio_instance
        mock_concat.return_value = MagicMock()
        mock_concat.return_value.with_audio.return_value = MagicMock()
        mock_concat.return_value.with_audio.return_value.write_videofile = MagicMock()

        assemble_video(
            image_paths=image_paths,
            scenes=scenes,
            audio_path=audio_path,
            output_path=output_path,
            animated=True,
        )

        # _ken_burns_clip must have been called once per scene
        assert mock_kb.call_count == 6
        # ImageClip must NOT have been called for scene clips
        mock_image_clip.assert_not_called()

    @patch("pipeline.video_editor._ken_burns_clip")
    @patch("pipeline.video_editor.ImageClip")
    @patch("pipeline.video_editor.get_audio_duration", return_value=12.0)
    @patch("pipeline.video_editor.AudioFileClip")
    @patch("pipeline.video_editor.concatenate_videoclips")
    def test_animated_false_calls_image_clip(
        self,
        mock_concat,
        mock_audio_cls,
        mock_duration,
        mock_image_clip,
        mock_kb,
        tmp_path,
    ):
        """When animated=False (default), ImageClip must be used, not _ken_burns_clip."""
        from pipeline.video_editor import assemble_video
        from PIL import Image

        image_paths = []
        for i in range(6):
            p = tmp_path / f"scene_{i:02d}.jpg"
            Image.new("RGB", (64, 36), color=(i * 40, 100, 200)).save(str(p))
            image_paths.append(p)

        scenes = [f"Scene {i} narration." for i in range(6)]
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"\xff\xfb\x90\x00" * 50)
        output_path = tmp_path / "output.mp4"

        mock_img_instance = MagicMock()
        mock_img_instance.with_duration.return_value = mock_img_instance
        mock_image_clip.return_value = mock_img_instance

        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 12.0
        mock_audio_cls.return_value = mock_audio_instance
        mock_concat.return_value = MagicMock()
        mock_concat.return_value.with_audio.return_value = MagicMock()
        mock_concat.return_value.with_audio.return_value.write_videofile = MagicMock()

        assemble_video(
            image_paths=image_paths,
            scenes=scenes,
            audio_path=audio_path,
            output_path=output_path,
            animated=False,
        )

        # ImageClip must have been called (once per scene frame)
        assert mock_image_clip.call_count == 6
        # _ken_burns_clip must NOT have been called
        mock_kb.assert_not_called()
