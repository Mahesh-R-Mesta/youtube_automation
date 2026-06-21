"""
Shared pytest fixtures for the Spiritual Content test suite.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_topic_dict() -> dict:
    return {
        "scripture": "bhagavad_gita",
        "mode": "sloka",
        "topic": "Bhagavad Gita 2:47 — Nishkama Karma",
        "scripture_ref": "Chapter 2, Verse 47",
        "hook": "What if the secret to a fulfilling life is to stop caring about the outcome?",
        "selection_reason": "Universally resonant — one of the most cited verses in the Gita.",
    }


@pytest.fixture
def sample_content_dict() -> dict:
    return {
        "sanskrit_devanagari": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।",
        "transliteration": "karmaṇy evādhikāras te mā phaleṣu kadācana",
        "word_by_word": "karmaṇi=action, eva=indeed, adhikāraḥ=right/duty, te=your",
        "english_meaning": (
            "You have a right to perform your prescribed duties, "
            "but you are not entitled to the fruits of your actions."
        ),
        "context": (
            "Spoken by Lord Krishna to Arjuna on the battlefield of Kurukshetra. "
            "Krishna teaches that one should act without attachment to results, "
            "surrendering outcomes to the Divine."
        ),
    }


@pytest.fixture
def sample_script() -> str:
    return (
        "|SCENE_1|\n"
        "What if the secret to a fulfilling life is to stop caring about the outcome? "
        "Three thousand years ago, on the battlefield of Kurukshetra, Lord Krishna revealed this timeless truth.\n\n"
        "|SCENE_2|\n"
        "In Sanskrit: Karmanye vadhikaraste. Ma phaleshu kadachan. "
        "Your right is to the action alone — never to its fruit.\n\n"
        "|SCENE_3|\n"
        "This verse from the Bhagavad Gita, Chapter 2 Verse 47, is perhaps the most powerful lesson "
        "ever given on how to live without anxiety.\n\n"
        "|SCENE_4|\n"
        "When you focus only on doing your best — without obsessing over the reward — "
        "something extraordinary happens. You work from joy, not fear.\n\n"
        "|SCENE_5|\n"
        "Think of a student studying for exams. If they obsess over marks, anxiety paralyzes them. "
        "But if they focus only on learning — clarity and calm return.\n\n"
        "|SCENE_6|\n"
        "This is Nishkama Karma — the art of desireless action. "
        "Act fully. Give your best. Release the rest. Jai Shri Krishna."
    )


@pytest.fixture
def tmp_audio_file(tmp_path: Path) -> Path:
    """Create a minimal valid MP3 placeholder for tests."""
    audio = tmp_path / "narration.mp3"
    audio.write_bytes(b"\xff\xfb\x90\x00" * 100)
    return audio


@pytest.fixture
def tmp_image_file(tmp_path: Path) -> Path:
    """Create a small valid JPEG portrait image for tests."""
    from PIL import Image
    img_path = tmp_path / "scene_01.jpg"
    img = Image.new("RGB", (720, 1280), color=(80, 40, 120))  # purple — spiritual colour
    img.save(str(img_path), "JPEG")
    return img_path

