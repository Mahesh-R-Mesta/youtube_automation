"""
Video Editor
─────────────
Assembles the final YouTube video from:
  • Scene images (from Pexels)          — one per scene
  • Narration audio (from TTS)          — full voiceover
  • Background music (optional)         — royalty-free ambient track
  • Subtitle text                        — first sentence of each scene

Uses moviepy 2.x (v2 API). Key v1 → v2 changes applied throughout:
  clip.set_duration()  → clip.with_duration()
  clip.set_audio()     → clip.with_audio()
  clip.resize()        → clip.resized()
  clip.volumex()       → clip.with_volume_scaled()
  clip.subclip()       → clip.subclipped()

Subtitle rendering is done entirely with Pillow (pre-baked into each scene
frame as a numpy array) — no TextClip / ImageMagick dependency needed.

Output: 1920×1080, 24 fps, H.264/AAC .mp4
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    ImageClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

from config.settings import settings
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Font resolution
# ─────────────────────────────────────────────────────────────────────────────

def _get_font_path() -> str:
    """
    Return a usable TrueType font path.
    Preference order:
      1. Bundled OpenSans-Bold.ttf in assets/fonts/
      2. Windows Arial Bold (C:/Windows/Fonts/arialbd.ttf)
      3. Windows Arial Regular
    """
    candidates = [
        Path("assets/fonts/OpenSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),  # Linux fallback
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No TrueType font found. Place OpenSans-Bold.ttf in assets/fonts/ "
        "or ensure Arial is installed at C:/Windows/Fonts/."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pillow subtitle helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_px: int,
) -> list[str]:
    """Word-wrap text to fit within max_px pixels wide."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_px:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def _render_scene_frame(
    image_path: Path,
    subtitle_text: str,
    font_path: str,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Use Pillow to compose the final scene frame:
      • Image resized to (width, height)
      • Semi-transparent subtitle bar at the bottom (if subtitle_text is set)
      • Subtitle text drawn with drop shadow

    Returns a (height, width, 3) uint8 NumPy array suitable for moviepy ImageClip.
    """
    # Resize source image to target dimensions
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)

    if not subtitle_text.strip():
        return np.array(img)

    # ── Semi-transparent subtitle bar ────────────────────────────────────────
    bar_height = 110
    bar_y = height - bar_height

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    # Slightly gradient: fully opaque at bottom, 60% at top of bar
    for row_offset in range(bar_height):
        alpha = int(160 + (200 - 160) * (row_offset / bar_height))
        draw_ov.rectangle(
            [(0, bar_y + row_offset), (width, bar_y + row_offset + 1)],
            fill=(0, 0, 0, alpha),
        )

    img_rgba = img.convert("RGBA")
    img_with_bar = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    # ── Subtitle text ─────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(img_with_bar)

    try:
        font = ImageFont.truetype(font_path, size=34)
    except OSError:
        font = ImageFont.load_default()

    lines = _wrap_text(draw, subtitle_text, font, max_px=width - 80)

    y = bar_y + 18
    for line in lines[:2]:  # max 2 subtitle lines
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2

        # Drop shadow (2px offset, black)
        draw.text((x + 2, y + 2), line, fill=(0, 0, 0), font=font)
        # White text
        draw.text((x, y), line, fill=(255, 255, 255), font=font)

        line_height = bbox[3] - bbox[1]
        y += line_height + 6

    return np.array(img_with_bar)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_duration(audio_path: str | Path) -> float:
    """Return the duration of an audio file in seconds."""
    clip = AudioFileClip(str(audio_path))
    duration = clip.duration
    clip.close()
    return duration


# ─────────────────────────────────────────────────────────────────────────────
# Ken Burns animated clip
# ─────────────────────────────────────────────────────────────────────────────

_KB_DIRECTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right")


def _ken_burns_clip(
    frame: np.ndarray,
    duration: float,
    direction: str = "zoom_in",
) -> VideoClip:
    """
    Create a Ken Burns animated clip from a static numpy frame.

    Available directions (cycle through scenes for variety):
      zoom_in    — slow zoom from 1.0× to 1.12×, centred
      zoom_out   — slow zoom from 1.12× to 1.0×, centred
      pan_left   — slide crop window from right to left
      pan_right  — slide crop window from left to right

    Args:
        frame:     (H, W, 3) uint8 numpy array — the pre-rendered scene frame.
        duration:  Clip duration in seconds.
        direction: One of "zoom_in", "zoom_out", "pan_left", "pan_right".

    Returns:
        A moviepy VideoClip with the Ken Burns motion baked in.
    """
    h, w = frame.shape[:2]
    # Inner crop frame at maximum zoom (12% larger in each dimension)
    ZOOM = 1.12
    inner_w = int(w / ZOOM)
    inner_h = int(h / ZOOM)

    pil_src = Image.fromarray(frame)

    def make_frame(t: float) -> np.ndarray:
        progress = t / duration  # 0.0 → 1.0

        if direction == "zoom_in":
            # Scale factor grows from 1.0× to ZOOM×
            scale = 1.0 + (ZOOM - 1.0) * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2

        elif direction == "zoom_out":
            # Scale factor shrinks from ZOOM× to 1.0×
            scale = ZOOM - (ZOOM - 1.0) * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2

        elif direction == "pan_left":
            # Crop moves from right side to left side
            crop_w, crop_h = inner_w, inner_h
            max_offset = w - inner_w
            left = int(max_offset * (1.0 - progress))
            top = (h - inner_h) // 2

        else:  # pan_right
            # Crop moves from left side to right side
            crop_w, crop_h = inner_w, inner_h
            max_offset = w - inner_w
            left = int(max_offset * progress)
            top = (h - inner_h) // 2

        cropped = pil_src.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((w, h), Image.LANCZOS)
        return np.array(resized)

    clip = VideoClip(make_frame, duration=duration)
    return clip


# ─────────────────────────────────────────────────────────────────────────────
# Video assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_video(
    image_paths: list[Path],
    scenes: list[str],
    audio_path: Path,
    output_path: Path,
    music_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    animated: bool = False,
) -> Path:
    """
    Assemble the final 1080p video from scene images, narration, and music.

    Steps:
      1. Measure narration duration → divide equally across scenes
      2. Render each scene frame with Pillow (image + subtitle bar)
      3. Create moviepy clip per scene (static ImageClip or Ken Burns VideoClip)
      4. Concatenate all scenes
      5. Attach narration + optional background music (15% volume)
      6. Write H.264/AAC MP4

    Args:
        image_paths: One image per scene (ordered).
        scenes:      Scene narration text (used for subtitle extraction).
        audio_path:  Path to narration MP3.
        output_path: Destination MP4 path.
        music_path:  Optional background music path (auto-looped if shorter).
        width/height: Override output resolution (defaults from settings).
        animated:    If True, apply Ken Burns pan/zoom effect to each scene.

    Returns:
        Path to the rendered MP4 file.
    """
    w = width or settings.video_width
    h = height or settings.video_height
    font_path = _get_font_path()

    # ── Timing ────────────────────────────────────────────────────────────────
    total_duration = get_audio_duration(audio_path)
    num_scenes = len(image_paths)
    scene_duration = total_duration / num_scenes

    logger.info(
        "Assembling %d scenes × %.1fs = %.1fs total video",
        num_scenes, scene_duration, total_duration,
    )

    # ── Scene clips ───────────────────────────────────────────────────────────
    scene_clips = []
    for i, (img_path, scene_text) in enumerate(zip(image_paths, scenes)):
        # Use the first sentence as subtitle (max 120 chars)
        first_sentence = (scene_text.split(".")[0] + ".").strip()
        subtitle = first_sentence[:120]

        frame = _render_scene_frame(img_path, subtitle, font_path, w, h)

        if animated:
            # Cycle through Ken Burns directions for visual variety
            direction = _KB_DIRECTIONS[i % len(_KB_DIRECTIONS)]
            clip = _ken_burns_clip(frame, scene_duration, direction=direction)
        else:
            clip = ImageClip(frame).with_duration(scene_duration)

        scene_clips.append(clip)
        logger.debug("Scene %d/%d clip ready (animated=%s)", i + 1, num_scenes, animated)

    # ── Concatenate ───────────────────────────────────────────────────────────
    final_video = concatenate_videoclips(scene_clips)

    # ── Audio mix ─────────────────────────────────────────────────────────────
    narration = AudioFileClip(str(audio_path))

    if music_path and Path(music_path).exists():
        music = AudioFileClip(str(music_path)).with_volume_scaled(0.15)

        # Loop music to match video length
        if music.duration < total_duration:
            loops = math.ceil(total_duration / music.duration)
            music = concatenate_audioclips([music] * loops).subclipped(0, total_duration)
        else:
            music = music.subclipped(0, total_duration)

        mixed = CompositeAudioClip([narration, music])
        final_video = final_video.with_audio(mixed)
        logger.info("Audio: narration + background music (15%% vol)")
    else:
        final_video = final_video.with_audio(narration)
        logger.info("Audio: narration only (no background music found)")

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering video → %s", output_path.name)
    final_video.write_videofile(
        str(output_path),
        fps=settings.video_fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(output_path.parent / "temp_audio.m4a"),
        remove_temp=True,
        logger=None,  # suppress moviepy's internal verbose progress bar
    )

    # ── Cleanup clips ─────────────────────────────────────────────────────────
    for clip in scene_clips:
        clip.close()
    final_video.close()
    narration.close()

    logger.info("Video rendered successfully: %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_thumbnail(
    hero_image_path: Path,
    headline_text: str,
    thumbnail_text: str,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """
    Generate a YouTube thumbnail (1280×720 JPEG) using Pillow.

    Layout:
      • Hero image (full bleed, cropped to 16:9)
      • Dark gradient overlay on the bottom 45% for text contrast
      • "NEWS UPDATE" red label badge top-left
      • Bold all-caps headline text centered near the bottom

    Uses draw.textbbox() (Pillow v10+ API — replaces deprecated textsize()).

    Returns:
        Path to the saved JPEG file.
    """
    font_path = _get_font_path()

    # ── Base image ────────────────────────────────────────────────────────────
    img = Image.open(str(hero_image_path)).convert("RGBA")

    # Smart crop-resize to exact 16:9 thumbnail dimensions
    src_w, src_h = img.size
    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider — crop sides
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        # Source is taller — crop top/bottom
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 4  # Bias slightly upward for faces/horizons
        img = img.crop((0, top, src_w, top + new_h))

    img = img.resize((width, height), Image.LANCZOS)

    # ── Gradient overlay (bottom 45%) ─────────────────────────────────────────
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    grad_start = int(height * 0.55)

    for y in range(grad_start, height):
        alpha = int(210 * (y - grad_start) / (height - grad_start))
        draw_ov.rectangle([(0, y), (width, y + 1)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ── Fonts ──────────────────────────────────────────────────────────────────
    try:
        font_badge = ImageFont.truetype(font_path, size=30)
        font_main = ImageFont.truetype(font_path, size=74)
    except OSError:
        font_badge = ImageFont.load_default()
        font_main = ImageFont.load_default()

    # ── Badge: "NEWS UPDATE" ─────────────────────────────────────────────────
    badge_text = "TIME TALES"
    badge_bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    badge_w = badge_bbox[2] - badge_bbox[0] + 30
    badge_h = badge_bbox[3] - badge_bbox[1] + 16
    draw.rectangle([(30, 30), (30 + badge_w, 30 + badge_h)], fill=(210, 30, 30))
    draw.text((45, 38), badge_text, fill="white", font=font_badge)

    # ── Main headline text ────────────────────────────────────────────────────
    text = thumbnail_text[:40].upper()

    # Wrap if needed (max ~18 chars per line at 74px)
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = _wrap_text(temp_draw, text, font_main, max_px=width - 80)

    y_text = height - 50 - sum(
        draw.textbbox((0, 0), ln, font=font_main)[3] + 8 for ln in lines[:2]
    )

    for line in lines[:2]:
        bbox = draw.textbbox((0, 0), line, font=font_main)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = max(40, (width - text_w) // 2)

        # Drop shadow
        draw.text((x + 3, y_text + 3), line, fill=(0, 0, 0), font=font_main)
        # Yellow text (high contrast on dark overlay)
        draw.text((x, y_text), line, fill=(255, 220, 0), font=font_main)

        y_text += text_h + 8

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "JPEG", quality=95, optimize=True)

    logger.info("Thumbnail saved → %s", output_path.name)
    return output_path
