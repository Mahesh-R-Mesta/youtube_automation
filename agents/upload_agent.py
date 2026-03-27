"""
YouTube Upload Agent
─────────────────────
Handles OAuth2 authentication and video upload to the YouTube Data API v3.

Quota costs (default 10 000 units/day):
  • videos.insert   → 1 600 units  (~6 uploads/day on free quota)
  • thumbnails.set  →    50 units

First run opens a browser for OAuth2 consent; subsequent runs refresh
the cached token automatically (token.json).

Prerequisites:
  1. In Google Cloud Console, enable "YouTube Data API v3"
  2. Create OAuth 2.0 credentials → Desktop app → download as client_secrets.json
  3. Set YOUTUBE_CREDENTIALS_FILE=client_secrets.json in .env
  4. Run once interactively so the browser consent flow can complete
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from utils.logger import logger

# Only the upload scope is required — never request unnecessary permissions
_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_TOKEN_FILE = "token.json"


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────

def authenticate(credentials_file: str = "client_secrets.json") -> Credentials:
    """
    Obtain valid OAuth2 credentials for the YouTube Data API.

    Token caching:
      - Reads from token.json on disk if it exists.
      - Auto-refreshes expired tokens using the stored refresh_token.
      - Opens a browser window for first-time authorisation if no valid
        token is found.

    Args:
        credentials_file: Path to the client_secrets.json downloaded from
                          Google Cloud Console.

    Returns:
        Valid google.oauth2.credentials.Credentials object.
    """
    creds: Credentials | None = None

    token_path = Path(_TOKEN_FILE)
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            logger.info("YouTube OAuth2 token refreshed.")
        else:
            creds_path = Path(credentials_file)
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"YouTube credentials file not found: '{credentials_file}'.\n"
                    "Steps to fix:\n"
                    "  1. Go to https://console.cloud.google.com/\n"
                    "  2. Enable 'YouTube Data API v3'\n"
                    "  3. Create OAuth 2.0 credentials (Desktop app)\n"
                    "  4. Download as 'client_secrets.json'\n"
                    "  5. Set YOUTUBE_CREDENTIALS_FILE in .env"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
            creds = flow.run_local_server(port=0)
            logger.info("YouTube OAuth2 authorisation completed.")

        # Persist token for future runs
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

def upload_video(
    credentials: Credentials,
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    category_id: str = "25",
    privacy_status: str = "public",
) -> str:
    """
    Upload a video file to YouTube using a resumable, chunked upload.

    Args:
        credentials:    Authorised OAuth2 credentials.
        video_path:     Path to the .mp4 file.
        title:          Video title (truncated to 100 chars).
        description:    Video description (truncated to 5000 chars).
        tags:           List of tags (max 15 used).
        category_id:    YouTube category ID. 25 = News & Politics.
        privacy_status: "public", "unlisted", or "private".

    Returns:
        The YouTube video ID string (e.g. "dQw4w9WgXcQ").

    Raises:
        googleapiclient.errors.HttpError: On API errors.
    """
    youtube = build("youtube", "v3", credentials=credentials)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:15],
            "categoryId": category_id,
            "defaultLanguage": "en",
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10 MB per chunk
    )

    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    logger.info("Uploading '%s' to YouTube...", video_path.name)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            logger.info("Upload progress: %d%%", progress)

    video_id: str = response["id"]
    logger.info("Upload complete — video ID: %s", video_id)
    return video_id


def set_thumbnail(
    credentials: Credentials,
    video_id: str,
    thumbnail_path: Path,
) -> None:
    """
    Set a custom JPEG thumbnail on an uploaded video.

    Note: YouTube requires the channel to be verified (phone number) before
    custom thumbnails can be set. This call will silently log an error if the
    channel is unverified rather than crashing the pipeline.

    Costs 50 units of YouTube API quota.
    """
    youtube = build("youtube", "v3", credentials=credentials)

    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
        ).execute()
        logger.info("Thumbnail set for video %s", video_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not set thumbnail for video %s: %s — "
            "Ensure your YouTube channel is verified (phone number required for custom thumbnails).",
            video_id,
            exc,
        )
