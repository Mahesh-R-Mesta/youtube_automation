"""
Voiceover Agent
────────────────
Converts the narration script to Indian-accented English speech.

Primary  → edge-tts (Microsoft Edge TTS, free, no API key required)
Fallback → ElevenLabs (10K credits/month free tier)

Default voices:
  Female: en-IN-NeerjaNeural  — warm, natural Indian English
  Male:   en-IN-PrabhatNeural — clear, authoritative Indian English
"""

import asyncio
import re
from pathlib import Path

from utils.logger import logger
from config.settings import settings

# Clean-up pattern — strips script scene markers before sending to TTS
_SCENE_MARKER_RE = re.compile(r"\|SCENE_\d+\|")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_script(script: str) -> str:
    """Remove scene markers and normalise whitespace for TTS."""
    text = _SCENE_MARKER_RE.sub(" ", script)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# edge-tts  (primary — free, async API)
# ─────────────────────────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, voice: str, rate: str, output_path: Path) -> Path:
    """Async core for edge-tts generation."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(output_path))
    return output_path


def generate_edge_tts(
    text: str,
    output_path: str | Path,
    voice: str | None = None,
    rate: str | None = None,
) -> Path:
    """
    Generate speech with edge-tts and save as MP3.

    Args:
        text:        Narration text (scene markers already removed).
        output_path: Destination .mp3 file path.
        voice:       Edge TTS voice name (default from settings).
        rate:        Speech rate adjustment, e.g. "+0%" or "-5%".

    Returns:
        Path to the saved MP3 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voice = voice or settings.tts_voice
    rate = rate or settings.tts_rate

    asyncio.run(_edge_tts_async(text, voice, rate, output_path))
    logger.info("edge-tts audio saved → %s (voice=%s)", output_path.name, voice)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# ElevenLabs  (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def generate_elevenlabs(
    text: str,
    output_path: str | Path,
    voice_id: str | None = None,
) -> Path:
    """Generate speech with ElevenLabs and save as MP3."""
    from elevenlabs import ElevenLabs

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not settings.elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set in .env")

    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    voice_id = voice_id or settings.elevenlabs_voice_id

    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)

    logger.info("ElevenLabs audio saved → %s", output_path.name)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_voiceover(
    script: str,
    output_path: str | Path,
    voice_gender: str = "female",
) -> Path:
    """
    Generate voiceover for the spiritual narration script.
    Tries edge-tts first; falls back to ElevenLabs on failure.

    Args:
        script:       Full narration script (may contain |SCENE_N| markers).
        output_path:  Destination .mp3 file path.
        voice_gender: "female" (default) → en-IN-NeerjaNeural
                      "male"            → en-IN-PrabhatNeural

    Returns:
        Path to the saved MP3 file.

    Raises:
        RuntimeError: If all TTS providers fail.
    """
    clean_text = _clean_script(script)

    # Select Indian English voice based on gender preference
    if voice_gender == "male":
        voice = settings.tts_male_voice
    else:
        voice = settings.tts_voice  # female (Neerja) is the default

    # Primary: edge-tts (free, no API key)
    try:
        return generate_edge_tts(clean_text, output_path, voice=voice)
    except Exception as exc:
        logger.warning("edge-tts failed (%s). Trying ElevenLabs fallback...", exc)

    # Fallback: ElevenLabs
    if settings.elevenlabs_api_key:
        try:
            return generate_elevenlabs(clean_text, output_path)
        except Exception as exc:
            logger.error("ElevenLabs fallback also failed: %s", exc)

    raise RuntimeError(
        "All TTS providers failed. "
        "Check your network connection and API keys in .env."
    )

