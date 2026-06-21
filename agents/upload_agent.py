"""
Instagram Upload Agent
───────────────────────
Uploads the finished spiritual Reel to Instagram using instagrapi.

⚠️  instagrapi uses Instagram's private (unofficial) API.
    This is against Meta's Terms of Service and carries a risk of account
    suspension, especially on personal accounts.
    Recommendation: use a dedicated creator/test account.

Session caching:
  On first login, instagrapi saves a session JSON to disk.
  Subsequent runs reload the session — no repeated password logins.
  Set INSTAGRAM_SESSION_FILE in .env to customise the path.

Prerequisites:
  pip install instagrapi
  Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env
"""

import json
from pathlib import Path
from typing import Optional

from utils.logger import logger
from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """
    Return an authenticated instagrapi Client.
    Loads saved session if available; falls back to fresh login.
    Persists session after any login to disk.
    """
    from instagrapi import Client  # lazy import

    username = settings.instagram_username
    password = settings.instagram_password

    if not username or not password:
        raise ValueError(
            "INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD must be set in .env to upload Reels."
        )

    cl = Client()
    session_path = Path(settings.instagram_session_file)

    if session_path.exists():
        try:
            cl.load_settings(session_path)
            cl.login(username, password)   # validates + refreshes the cached session
            cl.dump_settings(session_path)
            logger.info("Instagram session loaded from %s", session_path.name)
            return cl
        except Exception as exc:
            logger.warning(
                "Cached Instagram session invalid (%s). Performing fresh login.", exc
            )

    # Fresh login
    cl.login(username, password)
    cl.dump_settings(session_path)
    logger.info("Instagram login successful. Session saved to %s", session_path.name)
    return cl


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

def upload_reel(
    video_path: Path,
    caption: str,
    cover_image_path: Optional[Path] = None,
) -> str:
    """
    Upload a video as an Instagram Reel.

    Args:
        video_path:        Path to the .mp4 file (max 90 seconds, 9:16 portrait).
        caption:           Instagram caption text (max 2200 characters).
        cover_image_path:  Optional cover image (.jpg); first frame used if None.

    Returns:
        Public Instagram post URL (e.g. "https://www.instagram.com/p/ABC123/").

    Raises:
        ValueError:   If Instagram credentials are not configured.
        RuntimeError: If the upload fails after instagrapi raises an error.
    """
    cl = _get_client()

    # Truncate caption to Instagram's 2200-character limit
    safe_caption = caption[:2200]

    logger.info("Uploading Reel: %s", video_path.name)

    try:
        media = cl.clip_upload(
            path=str(video_path),
            caption=safe_caption,
            thumbnail=str(cover_image_path) if cover_image_path and cover_image_path.exists() else None,
        )
        reel_url = f"https://www.instagram.com/p/{media.code}/"
        logger.info("Reel uploaded successfully → %s", reel_url)
        return reel_url
    except Exception as exc:
        raise RuntimeError(
            f"Instagram Reel upload failed: {exc}\n"
            "Check your credentials, account status, and network connection."
        ) from exc
