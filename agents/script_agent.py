"""
Script Generation Agent
────────────────────────
Uses Google Gemini 2.5 Flash to:
  1. Write a full 6-scene narrated Reel script for a spiritual topic
  2. Generate Instagram metadata (caption + 30 hashtags)
  3. Generate Pixazo Flux image prompts per scene (style-aware)

Three content modes:
  sloka    — verse explanation (Bhagavad Gita, Upanishads, Atharva Veda)
  story    — scripture narrative (Mahabharata, Puranas, epic stories)
  teaching — philosophy concept explanation

All Gemini calls use the official google-genai SDK.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from config.prompts import (
    SLOKA_EXPLAINER_PROMPT,
    STORY_NARRATOR_PROMPT,
    TEACHING_EXPLAINER_PROMPT,
    SPIRITUAL_METADATA_PROMPT,
    SPIRITUAL_IMAGE_PROMPT,
    IMAGE_STYLE_DESCRIPTIONS,
)
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpiritualScript:
    scripture: str              # e.g., "bhagavad_gita"
    scripture_ref: str          # e.g., "Chapter 2, Verse 47"
    topic: str                  # e.g., "Bhagavad Gita 2:47 — Nishkama Karma"
    mode: str                   # sloka | story | teaching
    image_style: str            # tanjore | vedic | cosmic | minimalist
    # Sanskrit / on-screen text
    sanskrit_devanagari: str    # Devanagari Unicode text
    transliteration: str        # Roman transliteration (IAST)
    english_meaning: str        # English translation/summary
    # Video content
    full_script: str            # Complete narration with |SCENE_N| markers
    scenes: list[str]           # Per-scene narration text (6 items)
    scene_image_prompts: list[str]  # Flux AI image prompts (6 items)
    # Instagram metadata
    title: str                  # Short cover title (≤60 chars)
    caption: str                # Instagram caption (≤2200 chars)
    hashtags: list[str]         # 30 hashtags


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


def _parse_json(text: str) -> dict | list:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Script generation
# ─────────────────────────────────────────────────────────────────────────────

def write_spiritual_script(content: dict, mode: str) -> str:
    """
    Write a 6-scene narration script for the given content and mode.

    Args:
        content:  Dict containing topic, scripture_ref, hook, sanskrit_devanagari,
                  transliteration, english_meaning, context.
        mode:     "sloka", "story", or "teaching".

    Returns:
        Raw script text with |SCENE_1|...|SCENE_6| markers.
    """
    mode = mode.lower()

    if mode == "sloka":
        prompt = SLOKA_EXPLAINER_PROMPT.format(
            topic=content.get("topic", ""),
            scripture_ref=content.get("scripture_ref", ""),
            sanskrit_devanagari=content.get("sanskrit_devanagari", "ॐ"),
            english_meaning=content.get("english_meaning", ""),
            context=content.get("context", ""),
            hook=content.get("hook", ""),
        )
    elif mode == "story":
        prompt = STORY_NARRATOR_PROMPT.format(
            topic=content.get("topic", ""),
            scripture_ref=content.get("scripture_ref", ""),
            context=content.get("context", ""),
            hook=content.get("hook", ""),
        )
    elif mode == "teaching":
        prompt = TEACHING_EXPLAINER_PROMPT.format(
            topic=content.get("topic", ""),
            scripture_ref=content.get("scripture_ref", ""),
            sanskrit_devanagari=content.get("sanskrit_devanagari", "ॐ"),
            transliteration=content.get("transliteration", "Om"),
            context=content.get("context", ""),
            hook=content.get("hook", ""),
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'sloka', 'story', or 'teaching'.")

    script = _call_gemini(prompt, temperature=0.8)
    logger.info("Spiritual script generated: %d words (mode=%s)", len(script.split()), mode)
    return script


def generate_spiritual_metadata(
    topic: str,
    scripture_ref: str,
    mode: str,
    script: str,
) -> dict:
    """
    Generate Instagram caption and hashtags.

    Returns:
        Dict with keys: title, caption, hashtags.
    """
    excerpt = " ".join(script.split()[:120])
    prompt = SPIRITUAL_METADATA_PROMPT.format(
        topic=topic,
        scripture_ref=scripture_ref,
        mode=mode,
        script_excerpt=excerpt,
    )
    response = _call_gemini(prompt, temperature=0.5)
    metadata = _parse_json(response)
    logger.info("Instagram metadata generated: title='%s'", str(metadata.get("title", ""))[:60])
    return metadata


def generate_spiritual_image_prompts(
    script: str,
    scripture: str,
    image_style: str,
) -> list[str]:
    """
    Generate 6 style-aware Pixazo Flux image prompts for the script.

    Args:
        script:      Full narration script with |SCENE_N| markers.
        scripture:   Scripture key (used for thematic context in the prompt).
        image_style: One of: tanjore | vedic | cosmic | minimalist.

    Returns:
        List of exactly 6 prompt strings.
    """
    style_desc = IMAGE_STYLE_DESCRIPTIONS.get(
        image_style,
        IMAGE_STYLE_DESCRIPTIONS["tanjore"],  # default fallback
    )
    scripture_label = scripture.replace("_", " ").title()
    prompt = SPIRITUAL_IMAGE_PROMPT.format(
        image_style=image_style,
        scripture=scripture_label,
        script=script,
        style_description=style_desc,
    )
    response = _call_gemini(prompt, temperature=0.6)
    image_prompts: list[str] = _parse_json(response)

    # Ensure exactly 6 prompts — pad with themed fallbacks if model returned fewer
    _fallbacks = [
        f"Sacred temple interior with divine golden light filtering through ornate columns, "
        f"{image_style} art style, portrait 9:16 vertical, no text, no letters, no watermarks",

        f"Ancient Sanskrit manuscript scroll unfurled with glowing cosmic light surrounding it, "
        f"{image_style} art style, portrait 9:16 vertical, no text, no letters, no watermarks",

        f"Majestic lotus flower blooming in a serene celestial lake at sunrise, "
        f"{image_style} art style, portrait 9:16 vertical, no text, no letters, no watermarks",

        f"Divine Om symbol radiating golden light against a starfield sky, "
        f"{image_style} art style, portrait 9:16 vertical, no text, no letters, no watermarks",

        f"Sacred fire yajna ritual with glowing embers rising to the heavens, "
        f"{image_style} art style, portrait 9:16 vertical, no text, no letters, no watermarks",

        f"Peaceful ashram at dawn with Himalayan peaks in the background, "
        f"{image_style} art style, portrait 9:16 vertical, no text, no letters, no watermarks",
    ]
    while len(image_prompts) < 6:
        image_prompts.append(_fallbacks[len(image_prompts) % len(_fallbacks)])

    result = image_prompts[:6]
    logger.info("Spiritual image prompts generated (%d scenes, style=%s)", len(result), image_style)
    return result


def parse_scenes(script: str) -> list[str]:
    """
    Split the script into per-scene narration strings using |SCENE_N| markers.
    Returns a list of strings (6 items expected).
    """
    parts = re.split(r"\|SCENE_\d+\|", script)
    scenes = [s.strip() for s in parts if s.strip()]
    if len(scenes) != 6:
        logger.warning("Expected 6 scenes, got %d. Script markers may be malformed.", len(scenes))
    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# High-level pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_spiritual_video_script(
    topic_dict: dict,
    content_dict: dict,
    image_style: str,
) -> SpiritualScript:
    """
    Full pipeline: topic + content → SpiritualScript.

    Steps:
      1. Write 6-scene narration script (mode-specific)
      2. Generate Instagram metadata (caption + hashtags)
      3. Generate Flux image prompts (style-aware)
      4. Return SpiritualScript dataclass

    Args:
        topic_dict:   Output of scripture_agent.pick_auto_topic().
        content_dict: Output of scripture_agent.get_scripture_content().
        image_style:  "tanjore" | "vedic" | "cosmic" | "minimalist".

    Returns:
        SpiritualScript fully populated.
    """
    mode = topic_dict.get("mode", "sloka")
    scripture = topic_dict.get("scripture", "bhagavad_gita")
    topic = topic_dict.get("topic", "")
    scripture_ref = topic_dict.get("scripture_ref", "")

    # Merge hook into content_dict for prompt building
    merged = {**content_dict, **topic_dict}

    # 1. Script
    script = write_spiritual_script(merged, mode)

    # 2. Metadata
    meta = generate_spiritual_metadata(topic, scripture_ref, mode, script)

    # 3. Image prompts
    image_prompts = generate_spiritual_image_prompts(script, scripture, image_style)

    # 4. Parse scenes
    scenes = parse_scenes(script)

    return SpiritualScript(
        scripture=scripture,
        scripture_ref=scripture_ref,
        topic=topic,
        mode=mode,
        image_style=image_style,
        sanskrit_devanagari=content_dict.get("sanskrit_devanagari", "ॐ"),
        transliteration=content_dict.get("transliteration", "Om"),
        english_meaning=content_dict.get("english_meaning", ""),
        full_script=script,
        scenes=scenes,
        scene_image_prompts=image_prompts,
        title=meta.get("title", topic)[:60],
        caption=meta.get("caption", ""),
        hashtags=meta.get("hashtags", [])[:30],
    )


