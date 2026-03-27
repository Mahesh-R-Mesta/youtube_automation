"""
Script Generation Agent
────────────────────────
Uses Google Gemini 2.5 Flash to:
  1. Select the most YouTube-worthy topic from trending headlines
  2. Write a full 6-scene narrated video script
  3. Generate SEO-optimised title, description, tags, and thumbnail text
  4. Extract Pexels search keywords for each scene

All Gemini calls use the official google-genai SDK (NOT the deprecated
google-generativeai package which was EOL'd November 2025).
"""

import json
import re
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from config.prompts import (
    TOPIC_SELECTOR_PROMPT,
    SCRIPTWRITER_PROMPT,
    SEO_OPTIMIZER_PROMPT,
    KEYWORD_EXTRACTOR_PROMPT,
)
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoScript:
    topic: str
    title: str
    description: str
    tags: list[str]
    thumbnail_text: str
    full_script: str
    scenes: list[str]           # per-scene narration text (6 items)
    scene_keywords: list[str]   # Pexels search keyword per scene (6 items)


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
    """
    Call Gemini and return the response text.
    Retries up to 3 times with exponential back-off on transient errors.
    """
    client = _client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text.strip()


def _parse_json(text: str) -> dict | list:
    """
    Safely parse a JSON response from the LLM.
    Strips markdown code fences if the model wrapped the response.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def select_topic(headlines: list[dict]) -> dict:
    """
    Ask Gemini to pick the single best YouTube topic from trending headlines.
    Returns: {"selected_headline": str, "explanation": str}
    """
    headlines_text = "\n".join(
        f"{i + 1}. [{item['source']}] {item['title']}"
        for i, item in enumerate(headlines)
    )
    prompt = TOPIC_SELECTOR_PROMPT.format(headlines=headlines_text)
    response = _call_gemini(prompt, temperature=0.3)

    result = _parse_json(response)
    logger.info("Selected topic: %s", result.get("selected_headline", ""))
    return result


def write_script(topic: str) -> str:
    """
    Generate a full 6-scene narrated video script for the topic.
    Returns the raw script text with |SCENE_N| markers.
    """
    prompt = SCRIPTWRITER_PROMPT.format(topic=topic)
    script = _call_gemini(prompt, temperature=0.8)
    logger.info("Script generated: %d words", len(script.split()))
    return script


def generate_seo_metadata(topic: str, script: str) -> dict:
    """
    Generate YouTube SEO metadata: title, description, tags, thumbnail_text.
    """
    excerpt = " ".join(script.split()[:200])
    prompt = SEO_OPTIMIZER_PROMPT.format(topic=topic, script_excerpt=excerpt)
    response = _call_gemini(prompt, temperature=0.4)
    metadata = _parse_json(response)
    logger.info("SEO title: %s", str(metadata.get("title", ""))[:70])
    return metadata


def extract_scene_keywords(script: str) -> list[str]:
    """
    Extract one Pexels stock-photo search keyword per scene.
    Returns a list of exactly 6 strings.
    """
    prompt = KEYWORD_EXTRACTOR_PROMPT.format(script=script)
    response = _call_gemini(prompt, temperature=0.2)
    keywords: list[str] = _parse_json(response)

    # Ensure exactly 6 keywords — pad with fallbacks if needed
    fallbacks = ["world news", "global news", "breaking news", "news studio", "city street", "newspaper"]
    while len(keywords) < 6:
        keywords.append(fallbacks[len(keywords) % len(fallbacks)])

    result = keywords[:6]
    logger.info("Scene keywords: %s", result)
    return result


def parse_scenes(script: str) -> list[str]:
    """
    Split script text into individual scene strings using |SCENE_N| markers.
    Returns a list of per-scene narration text (may be fewer than 6 if model
    produced fewer markers — callers should handle this gracefully).
    """
    parts = re.split(r"\|SCENE_\d+\|", script)
    scenes = [s.strip() for s in parts if s.strip()]

    if len(scenes) != 6:
        logger.warning("Expected 6 scenes, got %d. Script markers may be malformed.", len(scenes))

    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_video_script(headlines: list[dict]) -> VideoScript:
    """
    Full pipeline: trending headlines → VideoScript object.

    Steps:
      1. Select best YouTube topic from headline list
      2. Write full 6-scene script
      3. Generate SEO metadata
      4. Extract scene-level Pexels keywords
    """
    # 1. Topic selection
    topic_result = select_topic(headlines)
    topic = topic_result["selected_headline"]

    # 2. Scriptwriting
    script = write_script(topic)

    # 3. SEO metadata
    seo = generate_seo_metadata(topic, script)

    # 4. Scene keywords
    keywords = extract_scene_keywords(script)
    scenes = parse_scenes(script)

    return VideoScript(
        topic=topic,
        title=seo.get("title", topic[:100]),
        description=seo.get("description", ""),
        tags=seo.get("tags", [])[:15],
        thumbnail_text=seo.get("thumbnail_text", "BREAKING NEWS"),
        full_script=script,
        scenes=scenes,
        scene_keywords=keywords,
    )
