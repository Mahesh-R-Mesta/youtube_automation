"""
Scripture Agent
────────────────
Selects spiritual topics from Indian scriptures and retrieves their content.

Supported scriptures:
  bhagavad_gita — Gita slokas, chapters, philosophy
  upanishads    — Isha, Kena, Katha, Mundaka, Mandukya, Taittiriya, etc.
  atharva_veda  — mantras, hymns, healing & cosmic knowledge
  mahabharata   — stories, characters, moral teachings from the epic
  puranas       — deity stories from Vishnu, Shiva, Devi Bhagavata Puranas

Two public functions:
  pick_auto_topic(scripture, mode) → dict
      Gemini picks the best topic; falls back to curated list on failure.

  get_scripture_content(topic_dict) → dict
      Gemini fetches Sanskrit text, transliteration, meaning, and context.
"""

import json
import random
import re
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from config.prompts import SCRIPTURE_PICKER_PROMPT, SANSKRIT_EXTRACTOR_PROMPT
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Curated fallback topic list — used when Gemini is unavailable
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_TOPICS: list[dict] = [
    # Bhagavad Gita — slokas
    {
        "scripture": "bhagavad_gita", "mode": "sloka",
        "topic": "Bhagavad Gita 2:47 — Nishkama Karma",
        "scripture_ref": "Chapter 2, Verse 47",
        "hook": "What if the secret to a fulfilling life is to stop caring about the outcome?",
    },
    {
        "scripture": "bhagavad_gita", "mode": "sloka",
        "topic": "Bhagavad Gita 6:5 — Elevate Yourself",
        "scripture_ref": "Chapter 6, Verse 5",
        "hook": "The greatest enemy you will ever face lives inside your own mind.",
    },
    {
        "scripture": "bhagavad_gita", "mode": "sloka",
        "topic": "Bhagavad Gita 2:20 — The Eternal Soul",
        "scripture_ref": "Chapter 2, Verse 20",
        "hook": "Science says energy cannot be destroyed — the Gita said it thousands of years ago.",
    },
    {
        "scripture": "bhagavad_gita", "mode": "sloka",
        "topic": "Bhagavad Gita 9:22 — Divine Providence",
        "scripture_ref": "Chapter 9, Verse 22",
        "hook": "There is one ancient promise that has never been broken — not once in all of history.",
    },
    {
        "scripture": "bhagavad_gita", "mode": "teaching",
        "topic": "The Three Gunas — Tamas, Rajas, and Sattva",
        "scripture_ref": "Bhagavad Gita, Chapters 14–18",
        "hook": "Ancient India mapped the entire personality of a human being into just three forces.",
    },
    # Upanishads — slokas & teachings
    {
        "scripture": "upanishads", "mode": "sloka",
        "topic": "Aham Brahmasmi — I am Brahman",
        "scripture_ref": "Brihadaranyaka Upanishad 1.4.10",
        "hook": "Five thousand years ago, a sage spoke four words that would shake the foundations of philosophy.",
    },
    {
        "scripture": "upanishads", "mode": "sloka",
        "topic": "Tat Tvam Asi — Thou Art That",
        "scripture_ref": "Chandogya Upanishad 6.8.7",
        "hook": "What if you and the universe are not two different things — but one?",
    },
    {
        "scripture": "upanishads", "mode": "teaching",
        "topic": "The Katha Upanishad — Death's Teaching to Nachiketa",
        "scripture_ref": "Katha Upanishad 1.1–1.3",
        "hook": "A young boy walked into the realm of death — and came back with the secret of immortality.",
    },
    {
        "scripture": "upanishads", "mode": "teaching",
        "topic": "Prana — the Life Force in Vedic Philosophy",
        "scripture_ref": "Prashna Upanishad",
        "hook": "Every breath you take is a conversation with the universe.",
    },
    {
        "scripture": "upanishads", "mode": "sloka",
        "topic": "Mandukya Upanishad — The Four States of Consciousness",
        "scripture_ref": "Mandukya Upanishad, Verse 1–7",
        "hook": "Waking, dreaming, deep sleep — and a fourth state no one talks about.",
    },
    # Atharva Veda — mantras & teachings
    {
        "scripture": "atharva_veda", "mode": "sloka",
        "topic": "Prithvi Sukta — Hymn to Mother Earth",
        "scripture_ref": "Atharva Veda 12.1",
        "hook": "Over three thousand years ago, ancient India wrote the world's first environmental prayer.",
    },
    {
        "scripture": "atharva_veda", "mode": "teaching",
        "topic": "Healing Mantras of the Atharva Veda",
        "scripture_ref": "Atharva Veda, Book 4",
        "hook": "The oldest medical text in the world is not a book — it is a collection of sacred sounds.",
    },
    {
        "scripture": "atharva_veda", "mode": "sloka",
        "topic": "Skambha Sukta — The Cosmic Pillar",
        "scripture_ref": "Atharva Veda 10.7",
        "hook": "What holds the universe together? The Atharva Veda had an answer before any physicist.",
    },
    {
        "scripture": "atharva_veda", "mode": "teaching",
        "topic": "Brahmacharya — The Vedic Science of Energy Conservation",
        "scripture_ref": "Atharva Veda 11.5",
        "hook": "Ancient India discovered a technology for human excellence — and called it Brahmacharya.",
    },
    # Mahabharata — stories
    {
        "scripture": "mahabharata", "mode": "story",
        "topic": "Ekalavya — The Greatest Student the World Forgot",
        "scripture_ref": "Mahabharata, Adi Parva",
        "hook": "He became the greatest archer in the world — and then was asked to give it all up.",
    },
    {
        "scripture": "mahabharata", "mode": "story",
        "topic": "Karna — The Tragedy of the Greatest Warrior",
        "scripture_ref": "Mahabharata, Karna Parva",
        "hook": "He was born a king but raised as a charioteer's son — and the world never let him forget it.",
    },
    {
        "scripture": "mahabharata", "mode": "story",
        "topic": "Draupadi's Vastraharan — When Dharma Was Silent",
        "scripture_ref": "Mahabharata, Sabha Parva",
        "hook": "In the most sacred court in the world, every hero stayed silent — and a woman called on God.",
    },
    {
        "scripture": "mahabharata", "mode": "story",
        "topic": "Vidura Niti — The Wisdom of the Most Honest Man",
        "scripture_ref": "Mahabharata, Udyoga Parva",
        "hook": "He was the wisest man in the kingdom — and he was born a servant.",
    },
    {
        "scripture": "mahabharata", "mode": "story",
        "topic": "Ashwatthama's Curse — The Immortal's Burden",
        "scripture_ref": "Mahabharata, Sauptika Parva",
        "hook": "Some say he still walks the earth today — cursed to live forever with a wound that never heals.",
    },
    # Puranas — stories
    {
        "scripture": "puranas", "mode": "story",
        "topic": "The Samudra Manthan — Churning of the Cosmic Ocean",
        "scripture_ref": "Vishnu Purana, Book 1; Bhagavata Purana 8.6–8.12",
        "hook": "Gods and demons once stood side by side — not to fight, but to create.",
    },
    {
        "scripture": "puranas", "mode": "story",
        "topic": "Prahlada and Narasimha — When God Tore Through a Pillar",
        "scripture_ref": "Bhagavata Purana 7.8",
        "hook": "A five-year-old boy told his demon king father that God was everywhere — even in this pillar.",
    },
    {
        "scripture": "puranas", "mode": "story",
        "topic": "The Story of Savitri — Love That Defeated Death",
        "scripture_ref": "Markandeya Purana",
        "hook": "She followed the god of death for three days to bring her husband back — and she won.",
    },
    {
        "scripture": "puranas", "mode": "story",
        "topic": "Gajendra Moksha — The Elephant Who Called God",
        "scripture_ref": "Bhagavata Purana 8.2–8.4",
        "hook": "An elephant held by a crocodile for a thousand years did one thing that changed everything.",
    },
    {
        "scripture": "puranas", "mode": "story",
        "topic": "The Birth of Ganesha — Why the Elephant God Has No Father",
        "scripture_ref": "Shiva Purana, Rudra Samhita 3.13–3.15",
        "hook": "Goddess Parvati was alone one day — and created a son from nothing but her own divine will.",
    },
    {
        "scripture": "puranas", "mode": "teaching",
        "topic": "Dashavatara — The Ten Avatars and Evolution",
        "scripture_ref": "Bhagavata Purana 1.3",
        "hook": "Charles Darwin described evolution in 1859 — ancient India described it thousands of years before.",
    },
    {
        "scripture": "puranas", "mode": "story",
        "topic": "Dhruva — The Boy Who Moved God",
        "scripture_ref": "Bhagavata Purana 4.8–4.12",
        "hook": "A seven-year-old boy was rejected by his own father and went alone to find God in the jungle.",
    },
]

_SCRIPTURE_KEYS = {
    "bhagavad_gita", "upanishads", "atharva_veda", "mahabharata", "puranas",
}
_MODE_KEYS = {"sloka", "story", "teaching"}


# ─────────────────────────────────────────────────────────────────────────────
# Gemini helpers
# ─────────────────────────────────────────────────────────────────────────────

def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def _call_gemini(prompt: str, temperature: float = 0.7) -> str:
    client = _client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text.strip()


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def pick_auto_topic(
    scripture: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """
    Use Gemini to pick a compelling spiritual topic.
    Falls back to a random entry from the curated list if Gemini fails.

    Args:
        scripture: Optional scripture key to constrain selection
                   (bhagavad_gita | upanishads | atharva_veda | mahabharata | puranas).
        mode:      Optional mode key (sloka | story | teaching).

    Returns:
        A dict with keys: scripture, mode, topic, scripture_ref, hook.
    """
    scripture_filter = ""
    if scripture and scripture in _SCRIPTURE_KEYS:
        scripture_filter = (
            f"Constraint: Use ONLY the scripture '{scripture.replace('_', ' ').title()}'.\n"
        )
    if mode and mode in _MODE_KEYS:
        scripture_filter += f"Constraint: Use ONLY the mode '{mode}'.\n"

    prompt = SCRIPTURE_PICKER_PROMPT.format(scripture_filter=scripture_filter)

    try:
        response = _call_gemini(prompt, temperature=0.9)
        result = _parse_json(response)
        # Validate required keys
        required = {"scripture", "mode", "topic", "scripture_ref", "hook"}
        if not required.issubset(result.keys()):
            raise ValueError(f"Gemini response missing keys: {required - result.keys()}")
        logger.info("Gemini picked topic: [%s/%s] %s", result["scripture"], result["mode"], result["topic"])
        return result
    except Exception as exc:
        logger.warning("Gemini topic picker failed (%s). Using curated fallback.", exc)
        return _fallback_topic(scripture, mode)


def get_scripture_content(topic_dict: dict) -> dict:
    """
    Retrieve Sanskrit text, transliteration, meaning, and context for a topic.

    Args:
        topic_dict: Dict returned by pick_auto_topic() or provided manually.

    Returns:
        Dict with keys: sanskrit_devanagari, transliteration, word_by_word,
                        english_meaning, context.
        Returns safe placeholder values if Gemini fails.
    """
    prompt = SANSKRIT_EXTRACTOR_PROMPT.format(
        topic=topic_dict.get("topic", ""),
        scripture_ref=topic_dict.get("scripture_ref", ""),
        mode=topic_dict.get("mode", "sloka"),
    )

    try:
        response = _call_gemini(prompt, temperature=0.3)
        result = _parse_json(response)
        logger.info("Scripture content retrieved for: %s", topic_dict.get("topic", ""))
        return result
    except Exception as exc:
        logger.warning("get_scripture_content failed (%s). Using placeholder.", exc)
        return {
            "sanskrit_devanagari": "ॐ",
            "transliteration": "Om",
            "word_by_word": "Om — the primordial sound of the universe",
            "english_meaning": topic_dict.get("topic", "Spiritual wisdom from the scriptures"),
            "context": (
                "This profound teaching comes from India's ancient scriptural tradition, "
                "offering timeless wisdom for the modern seeker."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_topic(
    scripture: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """Return a random curated topic matching the optional filters."""
    pool = _FALLBACK_TOPICS

    if scripture and scripture in _SCRIPTURE_KEYS:
        filtered = [t for t in pool if t["scripture"] == scripture]
        if filtered:
            pool = filtered

    if mode and mode in _MODE_KEYS:
        filtered = [t for t in pool if t["mode"] == mode]
        if filtered:
            pool = filtered

    chosen = random.choice(pool)
    logger.info("Fallback topic: [%s/%s] %s", chosen["scripture"], chosen["mode"], chosen["topic"])
    return dict(chosen)
