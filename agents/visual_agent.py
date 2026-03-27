"""
Visual Sourcing Agent
──────────────────────
Downloads one landscape stock photo per scene from the Pexels API.

Free tier limits: 200 requests/hour, 20 000 requests/month.
Attribution is mandatory per Pexels Terms of Service — the attribution
string returned by get_pexels_attribution() must appear in the video
description (the orchestrator handles this automatically).

There is no official Python client for Pexels, so we use requests directly.
"""

import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.logger import logger
from config.settings import settings

# ── API endpoints ─────────────────────────────────────────────────────────────
_PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"

# Mandatory attribution text (Pexels ToS section 2.3)
PEXELS_ATTRIBUTION = (
    "Stock photos and videos provided by Pexels — https://www.pexels.com"
)

# Seconds to sleep between image downloads (keeps us well within rate limits)
_INTER_REQUEST_DELAY = 0.6


def _headers() -> dict[str, str]:
    return {"Authorization": settings.pexels_api_key}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _search_pexels(keyword: str, per_page: int = 15) -> list[dict]:
    """
    Search Pexels for landscape photos matching keyword.
    Returns the raw list of photo objects from the API response.
    Retries up to 3 times on network / 5xx errors.
    """
    params = {
        "query": keyword,
        "per_page": per_page,
        "orientation": "landscape",
        "size": "large",  # at least 1920px wide when available
    }
    response = requests.get(
        _PHOTO_SEARCH_URL,
        headers=_headers(),
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("photos", [])


def fetch_pexels_image(keyword: str, output_path: Path) -> Path:
    """
    Search Pexels for keyword, download the highest-quality result,
    and save it to output_path.

    Falls back to the generic keyword "world news" if no results found.

    Returns:
        Path to the downloaded JPEG file.
    """
    photos = _search_pexels(keyword)

    if not photos:
        logger.warning("No Pexels results for '%s'. Retrying with 'world news'.", keyword)
        photos = _search_pexels("world news")

    if not photos:
        raise RuntimeError(f"Pexels returned no photos even for fallback keyword. Check API key.")

    # Prefer the photo with the largest original width for best quality
    best = max(photos, key=lambda p: p.get("width", 0))

    # Prefer large2x (≥2560px) → large (1280px) → original
    src = best.get("src", {})
    img_url = src.get("large2x") or src.get("large") or src.get("original")

    if not img_url:
        raise RuntimeError(f"Could not determine download URL for Pexels photo id={best.get('id')}")

    # Stream download
    img_response = requests.get(img_url, timeout=30, stream=True)
    img_response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in img_response.iter_content(chunk_size=16_384):
            f.write(chunk)

    photographer = best.get("photographer", "Unknown")
    logger.debug(
        "Downloaded scene image for '%s' (by %s) → %s", keyword, photographer, output_path.name
    )
    return output_path


def fetch_scene_images(
    keywords: list[str],
    output_dir: Path,
    run_id: str,
) -> list[Path]:
    """
    Download one Pexels image per scene keyword.

    Args:
        keywords:   List of Pexels search keywords (one per scene).
        output_dir: Directory to save downloaded images.
        run_id:     Unique run identifier used in filenames.

    Returns:
        Ordered list of local image Paths corresponding to each scene.
        If a scene download fails, the previous scene's image is reused as a placeholder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for i, keyword in enumerate(keywords):
        dest = output_dir / f"{run_id}_scene_{i + 1:02d}.jpg"

        # Skip re-download if file already exists (e.g. pipeline retry)
        if dest.exists():
            logger.debug("Scene %d image already cached, skipping download.", i + 1)
            image_paths.append(dest)
            continue

        try:
            path = fetch_pexels_image(keyword, dest)
            image_paths.append(path)
            # Small delay between requests to stay well within rate limits
            time.sleep(_INTER_REQUEST_DELAY)
        except Exception as exc:
            logger.error("Failed to fetch image for scene %d ('%s'): %s", i + 1, keyword, exc)
            # Use the previous image as a fallback placeholder
            if image_paths:
                logger.warning("Using previous scene image as placeholder for scene %d.", i + 1)
                image_paths.append(image_paths[-1])
            else:
                raise

    logger.info("Fetched %d scene images.", len(image_paths))
    return image_paths


def get_pexels_attribution() -> str:
    """Return the mandatory attribution string for video descriptions."""
    return PEXELS_ATTRIBUTION
