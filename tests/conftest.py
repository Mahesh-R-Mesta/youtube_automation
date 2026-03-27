"""
Shared pytest fixtures for the YouTube Automation test suite.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_headlines() -> list[dict]:
    return [
        {
            "title": "Historic Climate Deal Signed by 195 Nations at Global Summit",
            "summary": "World leaders gathered in Geneva to sign a landmark climate agreement.",
            "source": "BBC News",
            "url": "https://example.com/climate-deal",
        },
        {
            "title": "Scientists Discover Potential Cure for Rare Genetic Disease",
            "summary": "Researchers at MIT have identified a gene therapy approach.",
            "source": "Reuters",
            "url": "https://example.com/gene-therapy",
        },
        {
            "title": "Global Tech Stocks Surge Following AI Regulation Announcement",
            "summary": "Markets responded positively to new regulatory clarity.",
            "source": "AP News",
            "url": "https://example.com/tech-stocks",
        },
    ]


@pytest.fixture
def sample_script() -> str:
    return (
        "|SCENE_1|\n"
        "In a historic moment that could reshape the future of our planet, "
        "world leaders gathered in Geneva to sign an unprecedented climate deal.\n\n"
        "|SCENE_2|\n"
        "The agreement, backed by 195 nations, sets ambitious targets to cut carbon "
        "emissions by 60% before 2040.\n\n"
        "|SCENE_3|\n"
        "Scientists who have spent decades warning about climate change say this "
        "could be the turning point the world desperately needed.\n\n"
        "|SCENE_4|\n"
        "But not everyone is celebrating. Some critics argue the targets are still "
        "not aggressive enough to prevent the worst impacts.\n\n"
        "|SCENE_5|\n"
        "The real challenge now is implementation. History shows that international "
        "agreements don't always translate into real-world action.\n\n"
        "|SCENE_6|\n"
        "What do you think about this historic climate deal? Let us know in the "
        "comments below. Like and subscribe for daily news coverage."
    )


@pytest.fixture
def tmp_audio_file(tmp_path: Path) -> Path:
    """Create a minimal valid MP3 placeholder for tests."""
    audio = tmp_path / "narration.mp3"
    # Write a tiny valid-ish binary (tests mock AudioFileClip so content irrelevant)
    audio.write_bytes(b"\xff\xfb\x90\x00" * 100)
    return audio


@pytest.fixture
def tmp_image_file(tmp_path: Path) -> Path:
    """Create a small valid JPEG for tests that need an image file."""
    from PIL import Image
    img_path = tmp_path / "scene_01.jpg"
    img = Image.new("RGB", (200, 112), color=(100, 150, 200))
    img.save(str(img_path), "JPEG")
    return img_path
