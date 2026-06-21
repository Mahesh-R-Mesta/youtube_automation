from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Gemini ─────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key (required)")
    gemini_model: str = "gemini-2.5-flash"

    # ── ElevenLabs (optional fallback TTS) ────────────────────
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # ── Pixazo (AI image generation) ─────────────────────────
    # Free Flux 1 Schnell key at: https://api-console.pixazo.ai/api_keys
    pixazo_api_key: Optional[str] = None

    # ── Instagram ─────────────────────────────────────────────
    instagram_username: Optional[str] = None
    instagram_password: Optional[str] = None
    # Persisted instagrapi session — avoids repeated logins
    instagram_session_file: str = "instagram_session.json"

    # ── TTS (Indian English voices via edge-tts) ──────────────
    # Female: Neerja — warm, natural Indian English
    # Male:   Prabhat — clear, authoritative Indian English
    tts_voice: str = "en-IN-NeerjaNeural"
    tts_male_voice: str = "en-IN-PrabhatNeural"
    tts_rate: str = "+0%"

    # ── Video (Portrait 9:16 for Instagram Reels) ────────────
    video_fps: int = 24
    video_width: int = 1080
    video_height: int = 1920
    reel_max_duration: int = 90     # Instagram Reels max: 90 seconds

    # ── Spiritual content defaults ────────────────────────────
    # scripture: bhagavad_gita | upanishads | atharva_veda | mahabharata | puranas
    #            None → Gemini picks freely each run
    default_scripture: Optional[str] = None
    # mode: sloka | story | teaching
    default_mode: str = "sloka"
    # image_style: tanjore | vedic | cosmic | minimalist
    default_image_style: str = "tanjore"

    # ── Scheduler ─────────────────────────────────────────────
    schedule_hour: int = 9
    schedule_minute: int = 0

    # ── Output ────────────────────────────────────────────────
    output_dir: str = "output"


# Singleton instance — import and use directly:
#   from config.settings import settings
settings = Settings()
