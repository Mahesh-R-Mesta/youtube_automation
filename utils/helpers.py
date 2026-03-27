import re
from pathlib import Path

from utils.logger import logger


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and all parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    Remove or replace characters that are invalid in Windows/Linux filenames.
    Returns a safe string suitable for use as a filename stem (no extension).
    """
    # Replace filesystem-unsafe characters with underscores
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name)
    # Collapse runs of spaces and underscores into a single underscore
    name = re.sub(r"[\s_]+", "_", name)
    # Strip leading/trailing underscores and spaces
    name = name.strip("_ ")
    return name[:max_length] or "untitled"


def setup_output_dirs(output_dir: str = "output") -> dict[str, Path]:
    """
    Create all pipeline output subdirectories.
    Returns a dict mapping logical name → Path for easy reference.
    """
    base = Path(output_dir)
    dirs: dict[str, Path] = {
        "base": base,
        "audio": base / "audio",
        "images": base / "images",
        "videos": base / "videos",
        "thumbnails": base / "thumbnails",
    }
    for d in dirs.values():
        ensure_dir(d)
    logger.debug("Output directories ready under '%s'", output_dir)
    return dirs


def cleanup_old_outputs(output_dir: str = "output", keep_latest: int = 5) -> None:
    """
    Delete the oldest MP4 files from the videos output directory,
    keeping only the most recent `keep_latest` runs.
    """
    videos_dir = Path(output_dir) / "videos"
    if not videos_dir.exists():
        return

    video_files = sorted(
        videos_dir.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for f in video_files[keep_latest:]:
        try:
            f.unlink()
            logger.debug("Cleaned up old video: %s", f.name)
        except OSError as exc:
            logger.warning("Could not delete '%s': %s", f, exc)
