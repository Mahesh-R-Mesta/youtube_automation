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

    # ── Trend Discovery (both optional — RSS feeds always run without keys) ──
    # The Guardian: free key at https://bopenplatform.theguardian.com/access/
    guardian_api_key: Optional[str] = None
    # NewsData.io: free key at https://newsdata.io/register
    newsdata_api_key: Optional[str] = None

    # ── TTS ───────────────────────────────────────────────────
    tts_voice: str = "en-US-AriaNeural"          # English default
    tts_hindi_voice: str = "hi-IN-SwaraNeural"   # Hindi female (edge-tts)
    tts_rate: str = "+5%"

    # ── Video ─────────────────────────────────────────────────
    video_fps: int = 24
    video_width: int = 1920
    video_height: int = 1080

    # ── YouTube ───────────────────────────────────────────────
    youtube_credentials_file: str = "client_secrets.json"
    youtube_privacy: str = "public"
    youtube_category_id: str = "25"          # 25 = News & Politics
    youtube_history_category_id: str = "27"  # 27 = Education

    # ── Scheduler ─────────────────────────────────────────────
    schedule_hour: int = 9
    schedule_minute: int = 0

    # ── History storytelling mode ─────────────────────────────
    # Both are optional overrides; leave empty to let Gemini choose freely.
    history_era: Optional[str] = None    # e.g. "Ancient Rome"
    history_theme: Optional[str] = None  # e.g. "forgotten women"

    # ── Output ────────────────────────────────────────────────
    output_dir: str = "output"


# Singleton instance — import and use directly:
#   from config.settings import settings
settings = Settings()
