"""
Video Editor — Instagram Reels (Portrait 9:16)
───────────────────────────────────────────────
Assembles the final Instagram Reel from:
  • Scene images (from Pixazo Flux, portrait 720×1280)
  • Narration audio (Indian-accented English TTS)
  • Background music (optional, royalty-free)
  • Sanskrit text overlay (top zone — Devanagari + transliteration)
  • Narration subtitle (bottom zone)

Output: 1080×1920 portrait MP4 (9:16), 24 fps, H.264/AAC.

Frame layout:
  ┌────────────────────────────────┐
  │  TOP ZONE  (0 – 380px)        │  ← Sanskrit Devanagari text
  │  dark gradient overlay         │     + transliteration
  ├────────────────────────────────┤  ← thin golden divider
  │                                │
  │   SCENE IMAGE (fill frame)     │  ← AI-generated scene image
  │   scaled/cropped to 1080×1920  │     (Ken Burns animation)
  │                                │
  ├────────────────────────────────┤  ← thin golden divider
  │  BOTTOM ZONE (1480 – 1920px)  │  ← scripture ref (gold, small)
  │  dark gradient overlay         │     + narration subtitle (white)
  └────────────────────────────────┘

Thumbnail: 1080×1080 JPEG (Instagram square cover).

Uses moviepy 2.x (v2 API). Subtitle rendering via Pillow — no ImageMagick.
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


# ── Layout constants ──────────────────────────────────────────────────────────
_TOP_ZONE_H   = 380    # px: Sanskrit text zone height
_BOTTOM_ZONE_H = 440   # px: subtitle zone height
_GOLD = (218, 165, 32)   # goldenrod
_GOLD_LIGHT = (255, 215, 0)  # pure gold for accent lines
_DIVIDER_H = 3         # px: golden divider line thickness


# ─────────────────────────────────────────────────────────────────────────────
# Font resolution
# ─────────────────────────────────────────────────────────────────────────────

def _get_latin_font_path() -> str:
    """Resolve a TrueType Latin font for subtitles and titles."""
    candidates = [
        Path("assets/fonts/OpenSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No TrueType Latin font found. "
        "Place OpenSans-Bold.ttf in assets/fonts/ or ensure Arial is installed."
    )


def _get_devanagari_font_path() -> Optional[str]:
    """
    Resolve a TrueType font with Devanagari Unicode support.
    Returns None if no suitable font is found — caller should use transliteration.
    """
    candidates = [
        Path("assets/fonts/NotoSerifDevanagari-Bold.ttf"),
        Path("assets/fonts/NotoSansDevanagari-Bold.ttf"),
        Path("assets/fonts/NotoSansDevanagari-Regular.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),        # Segoe UI (Windows 8+)
        Path("C:/Windows/Fonts/seguisym.ttf"),       # Segoe UI Symbol
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    logger.warning(
        "No Devanagari font found. Sanskrit text will show as transliteration. "
        "Download 'Noto Serif Devanagari Bold' and place it in assets/fonts/ "
        "for proper script rendering."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pillow text helpers
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
        if (bbox[2] - bbox[0]) <= max_px:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_text_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
    y: int,
    color: tuple,
    shadow: bool = True,
    max_lines: int = 3,
    max_px: Optional[int] = None,
) -> int:
    """
    Draw centered text with optional drop shadow.
    Returns the y position after the last line drawn.
    """
    lines = _wrap_text(draw, text, font, max_px=max_px or (width - 60))
    for line in lines[:max_lines]:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        if shadow:
            draw.text((x + 2, y + 2), line, fill=(0, 0, 0, 200), font=font)
        draw.text((x, y), line, fill=color, font=font)
        y += text_h + 8
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Scene frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_spiritual_frame(
    image_path: Path,
    subtitle_text: str,
    scripture_ref: str,
    sanskrit_text: str,
    transliteration: str,
    latin_font_path: str,
    devanagari_font_path: Optional[str],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Compose the portrait Reel frame with spiritual UI overlays.

    Layout (top → bottom):
      • Full frame: scene image scaled to fill (w×h)
      • Top zone: dark gradient + Sanskrit text + transliteration
      • Golden divider: thin line at top/bottom zone boundaries
      • Bottom zone: dark gradient + scripture ref + narration subtitle

    Returns a (height, width, 3) uint8 NumPy array for moviepy.
    """
    # ── 1. Scale scene image to fill the full portrait frame ─────────────────
    img = Image.open(str(image_path)).convert("RGB")
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Centre-crop
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))
    full_frame = img.convert("RGBA")

    # ── 2. Top zone dark gradient overlay ────────────────────────────────────
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    # Gradient: fully opaque (alpha=220) at top, fading to 0 at zone bottom
    for row in range(_TOP_ZONE_H):
        alpha = int(220 * (1.0 - row / _TOP_ZONE_H))
        draw_ov.rectangle([(0, row), (width, row + 1)], fill=(10, 5, 20, alpha))

    # ── 3. Bottom zone dark gradient overlay ─────────────────────────────────
    bottom_start = height - _BOTTOM_ZONE_H
    for row in range(_BOTTOM_ZONE_H):
        alpha = int(220 * (row / _BOTTOM_ZONE_H))
        draw_ov.rectangle(
            [(0, bottom_start + row), (width, bottom_start + row + 1)],
            fill=(10, 5, 20, alpha),
        )

    # ── 4. Golden divider lines ───────────────────────────────────────────────
    draw_ov.rectangle(
        [(0, _TOP_ZONE_H), (width, _TOP_ZONE_H + _DIVIDER_H)],
        fill=(*_GOLD_LIGHT, 180),
    )
    draw_ov.rectangle(
        [(0, bottom_start - _DIVIDER_H), (width, bottom_start)],
        fill=(*_GOLD_LIGHT, 180),
    )

    frame_with_overlay = Image.alpha_composite(full_frame, overlay).convert("RGB")
    draw = ImageDraw.Draw(frame_with_overlay)

    # ── 5. Sanskrit text in top zone ─────────────────────────────────────────
    try:
        devanagari_font = (
            ImageFont.truetype(devanagari_font_path, size=44)
            if devanagari_font_path
            else None
        )
        latin_font_med = ImageFont.truetype(latin_font_path, size=28)
        latin_font_small = ImageFont.truetype(latin_font_path, size=22)
    except OSError:
        devanagari_font = None
        latin_font_med = ImageFont.load_default()
        latin_font_small = ImageFont.load_default()

    y_top = 22
    # OM symbol accent line
    draw.text((width // 2 - 14, y_top), "🕉", fill=_GOLD, font=latin_font_small)
    y_top += 34

    if devanagari_font and sanskrit_text:
        # Try to draw Devanagari — if it fails (encoding issues), fall back
        try:
            y_top = _draw_text_centered(
                draw, sanskrit_text, devanagari_font, width, y_top,
                color=_GOLD_LIGHT, max_lines=3,
            )
        except Exception:
            # Fall through to transliteration
            pass

    if transliteration:
        y_top += 6
        y_top = _draw_text_centered(
            draw, transliteration, latin_font_med, width, y_top,
            color=(220, 200, 160), shadow=True, max_lines=2,
        )

    # ── 6. Bottom zone text ───────────────────────────────────────────────────
    try:
        font_ref = ImageFont.truetype(latin_font_path, size=22)
        font_subtitle = ImageFont.truetype(latin_font_path, size=34)
    except OSError:
        font_ref = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()

    # Scripture reference (gold, small)
    if scripture_ref:
        ref_short = scripture_ref[:60]
        ref_bbox = draw.textbbox((0, 0), ref_short, font=font_ref)
        ref_w = ref_bbox[2] - ref_bbox[0]
        draw.text(
            ((width - ref_w) // 2, bottom_start + 12),
            ref_short,
            fill=_GOLD,
            font=font_ref,
        )

    # Narration subtitle (white, bold)
    if subtitle_text.strip():
        sub_y = bottom_start + 50
        _draw_text_centered(
            draw, subtitle_text, font_subtitle, width, sub_y,
            color=(255, 255, 255), shadow=True, max_lines=3,
        )

    return np.array(frame_with_overlay)


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

_KB_DIRECTIONS = ("zoom_in", "zoom_out", "pan_up", "pan_down")


def _ken_burns_clip(
    frame: np.ndarray,
    duration: float,
    direction: str = "zoom_in",
) -> VideoClip:
    """
    Create a slow Ken Burns motion clip from a static portrait frame.

    Directions:
      zoom_in   — slow zoom from 1.0× to 1.08× (centred)
      zoom_out  — slow zoom from 1.08× to 1.0× (centred)
      pan_up    — slide crop window from bottom to top
      pan_down  — slide crop window from top to bottom
    """
    h, w = frame.shape[:2]
    ZOOM = 1.08
    inner_h = int(h / ZOOM)
    inner_w = int(w / ZOOM)
    pil_src = Image.fromarray(frame)

    def make_frame(t: float) -> np.ndarray:
        progress = t / duration

        if direction == "zoom_in":
            scale = 1.0 + (ZOOM - 1.0) * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2

        elif direction == "zoom_out":
            scale = ZOOM - (ZOOM - 1.0) * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2

        elif direction == "pan_up":
            crop_w, crop_h = inner_w, inner_h
            max_offset = h - inner_h
            left = (w - inner_w) // 2
            top = int(max_offset * (1.0 - progress))

        else:  # pan_down
            crop_w, crop_h = inner_w, inner_h
            max_offset = h - inner_h
            left = (w - inner_w) // 2
            top = int(max_offset * progress)

        cropped = pil_src.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((w, h), Image.LANCZOS)
        return np.array(resized)

    return VideoClip(make_frame, duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# Video assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_video(
    image_paths: list[Path],
    scenes: list[str],
    audio_path: Path,
    output_path: Path,
    music_path: Optional[Path] = None,
    sanskrit_text: str = "ॐ",
    transliteration: str = "Om",
    scripture_ref: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    animated: bool = True,
) -> Path:
    """
    Assemble the final portrait Instagram Reel MP4.

    Steps:
      1. Measure narration duration → divide equally across scenes
      2. Render each portrait scene frame with Pillow (image + overlays)
      3. Apply Ken Burns animation per scene
      4. Concatenate all scene clips
      5. Attach narration + optional music (15% volume)
      6. Write H.264/AAC MP4

    Args:
        image_paths:    One portrait image per scene (ordered).
        scenes:         Scene narration text (used for subtitle extraction).
        audio_path:     Path to narration MP3.
        output_path:    Destination MP4 path.
        music_path:     Optional background music (auto-looped if shorter).
        sanskrit_text:  Devanagari Sanskrit text for top zone overlay.
        transliteration: Roman transliteration of the Sanskrit.
        scripture_ref:  Reference label shown above subtitle (e.g. "Gita 2:47").
        width/height:   Override portrait dimensions (default 1080×1920).
        animated:       Apply Ken Burns motion to scenes (default True).

    Returns:
        Path to the rendered MP4 file.
    """
    w = width or settings.video_width       # 1080
    h = height or settings.video_height     # 1920
    latin_font = _get_latin_font_path()
    deva_font = _get_devanagari_font_path()

    # ── Timing ────────────────────────────────────────────────────────────────
    total_duration = get_audio_duration(audio_path)
    num_scenes = len(image_paths)
    scene_duration = total_duration / num_scenes

    # Clamp to Instagram Reels max of 90 seconds
    if total_duration > settings.reel_max_duration:
        logger.warning(
            "Narration is %.1fs — exceeds Instagram Reels max of %ds. "
            "Consider shortening the script.",
            total_duration,
            settings.reel_max_duration,
        )

    logger.info(
        "Assembling %d scenes × %.1fs = %.1fs total reel",
        num_scenes, scene_duration, total_duration,
    )

    # ── Scene clips ───────────────────────────────────────────────────────────
    scene_clips = []
    for i, (img_path, scene_text) in enumerate(zip(image_paths, scenes)):
        # Extract first sentence as subtitle (max 100 chars)
        first_sentence = (scene_text.split(".")[0] + ".").strip()
        subtitle = first_sentence[:100]

        # Show Sanskrit only on first scene; subsequent scenes show scripture ref
        s_text = sanskrit_text if i == 0 else ""
        t_text = transliteration if i == 0 else ""

        frame = _render_spiritual_frame(
            image_path=img_path,
            subtitle_text=subtitle,
            scripture_ref=scripture_ref,
            sanskrit_text=s_text,
            transliteration=t_text,
            latin_font_path=latin_font,
            devanagari_font_path=deva_font,
            width=w,
            height=h,
        )

        if animated:
            direction = _KB_DIRECTIONS[i % len(_KB_DIRECTIONS)]
            clip = _ken_burns_clip(frame, scene_duration, direction=direction)
        else:
            clip = ImageClip(frame).with_duration(scene_duration)

        scene_clips.append(clip)
        logger.debug("Scene %d/%d rendered", i + 1, num_scenes)

    # ── Concatenate ───────────────────────────────────────────────────────────
    final_video = concatenate_videoclips(scene_clips)

    # ── Audio mix ─────────────────────────────────────────────────────────────
    narration = AudioFileClip(str(audio_path))

    if music_path and Path(music_path).exists():
        music = AudioFileClip(str(music_path)).with_volume_scaled(0.15)
        if music.duration < total_duration:
            loops = math.ceil(total_duration / music.duration)
            music = concatenate_audioclips([music] * loops).subclipped(0, total_duration)
        else:
            music = music.subclipped(0, total_duration)
        mixed = CompositeAudioClip([narration, music])
        final_video = final_video.with_audio(mixed)
        logger.info("Audio: narration + background music (15%% volume)")
    else:
        final_video = final_video.with_audio(narration)
        logger.info("Audio: narration only")

    # ── Write MP4 ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering portrait Reel → %s", output_path.name)

    final_video.write_videofile(
        str(output_path),
        fps=settings.video_fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(output_path.parent / "temp_audio.m4a"),
        remove_temp=True,
        logger=None,
    )

    for clip in scene_clips:
        clip.close()
    final_video.close()
    narration.close()

    logger.info("Reel rendered successfully: %s", output_path.name)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Instagram thumbnail (1080×1080 square)
# ─────────────────────────────────────────────────────────────────────────────

def generate_thumbnail(
    hero_image_path: Path,
    title_text: str,
    sanskrit_text: str,
    output_path: Path,
    size: int = 1080,
) -> Path:
    """
    Generate an Instagram square thumbnail (1080×1080 JPEG) using Pillow.

    Layout:
      • Hero image (square centre-crop)
      • Dark gradient overlay (top & bottom strips)
      • Sanskrit / title text centred
      • Golden accent lines
      • Small Om symbol

    Returns:
        Path to the saved JPEG file.
    """
    latin_font_path = _get_latin_font_path()
    deva_font_path = _get_devanagari_font_path()

    # ── Base image ────────────────────────────────────────────────────────────
    img = Image.open(str(hero_image_path)).convert("RGBA")
    src_w, src_h = img.size
    # Centre-crop to square
    sq = min(src_w, src_h)
    left = (src_w - sq) // 2
    top = (src_h - sq) // 3   # slight upward bias
    img = img.crop((left, top, left + sq, top + sq))
    img = img.resize((size, size), Image.LANCZOS)

    # ── Overlay gradient ──────────────────────────────────────────────────────
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    # Top strip (0 → 35%) — for Sanskrit text
    top_h = int(size * 0.35)
    for row in range(top_h):
        alpha = int(200 * (1.0 - row / top_h))
        draw_ov.rectangle([(0, row), (size, row + 1)], fill=(10, 5, 20, alpha))

    # Bottom strip (65% → 100%) — for title text
    bottom_start = int(size * 0.65)
    for row in range(size - bottom_start):
        alpha = int(210 * (row / (size - bottom_start)))
        draw_ov.rectangle(
            [(0, bottom_start + row), (size, bottom_start + row + 1)],
            fill=(10, 5, 20, alpha),
        )

    # Golden dividers
    draw_ov.rectangle([(0, top_h), (size, top_h + _DIVIDER_H)], fill=(*_GOLD_LIGHT, 160))
    draw_ov.rectangle([(0, bottom_start - _DIVIDER_H), (size, bottom_start)], fill=(*_GOLD_LIGHT, 160))

    final = Image.alpha_composite(img, overlay).convert("RGB")
    draw = ImageDraw.Draw(final)

    # ── Fonts ──────────────────────────────────────────────────────────────────
    try:
        font_deva = ImageFont.truetype(deva_font_path, size=54) if deva_font_path else None
        font_title = ImageFont.truetype(latin_font_path, size=52)
        font_small = ImageFont.truetype(latin_font_path, size=26)
    except OSError:
        font_deva = None
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # ── Top: Sanskrit text ────────────────────────────────────────────────────
    draw.text((size // 2 - 14, 18), "🕉", fill=_GOLD, font=font_small)
    y = 52
    if font_deva and sanskrit_text:
        try:
            y = _draw_text_centered(
                draw, sanskrit_text, font_deva, size, y,
                color=_GOLD_LIGHT, max_lines=2,
            )
        except Exception:
            pass

    # ── Bottom: title text ────────────────────────────────────────────────────
    title_short = title_text[:50]
    _draw_text_centered(
        draw, title_short, font_title, size, bottom_start + 10,
        color=(255, 255, 255), shadow=True, max_lines=2,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(str(output_path), "JPEG", quality=92, optimize=True)
    logger.info("Thumbnail saved → %s", output_path.name)
    return output_path

