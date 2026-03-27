"""
Script Generation Agent
────────────────────────
Uses Google Gemini 2.5 Flash to:
  1. Select the most YouTube-worthy topic from trending headlines
  2. Write a full 6-scene narrated video script
  3. Generate SEO-optimised title, description, tags, and thumbnail text
  4. Generate Flux AI image prompts for each scene (used by visual_agent)

Also provides the history storytelling pipeline:
  A. write_history_script(research_brief) → str
  B. generate_history_image_prompts(script, era, region) → list[str]
  C. generate_history_video_script(era, theme) → HistoryScript

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
    IMAGE_PROMPT_GENERATOR_PROMPT,
    HISTORY_SCRIPTWRITER_PROMPT,
    HISTORY_IMAGE_PROMPT_GENERATOR_PROMPT,
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
    scenes: list[str]               # per-scene narration text (6 items)
    scene_image_prompts: list[str]  # Flux AI image generation prompt per scene (6 items)


@dataclass
class HistoryScript(VideoScript):
    """VideoScript extended with history-specific metadata."""
    era: str = ""
    region: str = ""
    research_brief: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.research_brief is None:
            self.research_brief = {}


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
# Language helpers
# ─────────────────────────────────────────────────────────────────────────────

_HINDI_DIRECTIVE = (
    "Write the ENTIRE script in Hindi (Devanagari script). "
    "Use natural, conversational Hindi that sounds fluent when read aloud."
)
_ENGLISH_DIRECTIVE = "Write the script in English."


def _language_directive(language: str) -> str:
    """Return the language instruction string for a given language code."""
    return _HINDI_DIRECTIVE if language.lower() == "hindi" else _ENGLISH_DIRECTIVE


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


def write_script(topic: str, language: str = "english") -> str:
    """
    Generate a full 6-scene narrated video script for the topic.
    Returns the raw script text with |SCENE_N| markers.
    """
    prompt = SCRIPTWRITER_PROMPT.format(
        topic=topic,
        language_directive=_language_directive(language),
    )
    script = _call_gemini(prompt, temperature=0.8)
    logger.info("Script generated: %d words", len(script.split()))
    return script


def generate_seo_metadata(topic: str, script: str, language: str = "english") -> dict:
    """
    Generate YouTube SEO metadata: title, description, tags, thumbnail_text.
    """
    seo_language_directive = (
        "Write the title, description, and tags in Hindi (Devanagari)."
        if language.lower() == "hindi"
        else "Write the title, description, and tags in English."
    )
    excerpt = " ".join(script.split()[:200])
    prompt = SEO_OPTIMIZER_PROMPT.format(
        topic=topic,
        script_excerpt=excerpt,
        language_directive=seo_language_directive,
    )
    response = _call_gemini(prompt, temperature=0.4)
    metadata = _parse_json(response)
    logger.info("SEO title: %s", str(metadata.get("title", ""))[:70])
    return metadata


def generate_scene_image_prompts(script: str) -> list[str]:
    """
    Generate one Flux AI image generation prompt per scene.
    Each prompt is a rich, cinematic description optimised for Flux 1 Schnell.
    Returns a list of exactly 6 prompt strings.
    """
    prompt = IMAGE_PROMPT_GENERATOR_PROMPT.format(script=script)
    response = _call_gemini(prompt, temperature=0.5)
    image_prompts: list[str] = _parse_json(response)

    # Ensure exactly 6 prompts — pad with neutral fallbacks if needed
    fallbacks = [
        "cinematic aerial view of a city skyline at golden hour, photorealistic 8K wide angle",
        "dramatic close-up of a glowing digital globe with data streams, dark background, 8K",
        "sweeping documentary shot of a modern parliament building at dusk, cinematic wide",
        "abstract visualization of global connectivity, glowing network nodes, deep blue",
        "photorealistic crowd of silhouetted people at a public square, golden sunset backlight",
        "dramatic macro shot of a newspaper headline with shallow depth of field, cinematic",
    ]
    while len(image_prompts) < 6:
        image_prompts.append(fallbacks[len(image_prompts) % len(fallbacks)])

    result = image_prompts[:6]
    logger.info("Scene image prompts generated (%d scenes)", len(result))
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

def generate_video_script(headlines: list[dict], language: str = "english") -> VideoScript:
    """
    Full pipeline: trending headlines → VideoScript object.

    Steps:
      1. Select best YouTube topic from headline list
      2. Write full 6-scene script
      3. Generate SEO metadata
      4. Generate Flux AI image prompts per scene
    """
    # 1. Topic selection
    topic_result = select_topic(headlines)
    topic = topic_result["selected_headline"]

    # 2. Scriptwriting
    script = write_script(topic, language=language)

    # 3. SEO metadata
    seo = generate_seo_metadata(topic, script, language=language)

    # 4. Scene image prompts (replaces Pexels keyword extraction)
    image_prompts = generate_scene_image_prompts(script)
    scenes = parse_scenes(script)

    return VideoScript(
        topic=topic,
        title=seo.get("title", topic)[:100],
        description=seo.get("description", ""),
        tags=seo.get("tags", [])[:15],
        thumbnail_text=seo.get("thumbnail_text", "BREAKING NEWS"),
        full_script=script,
        scenes=scenes,
        scene_image_prompts=image_prompts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# History Storytelling Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def write_history_script(research_brief: dict, language: str = "english") -> str:
    """
    Write a Ken-Burns-style documentary narration script from a research brief.

    Args:
        research_brief: Output of history_agent.research_topic() — contains
                        topic, era, region, hook, key_figures, timeline, etc.
        language:       "english" (default) or "hindi".

    Returns:
        Raw script text with |SCENE_1|...|SCENE_6| markers.
    """
    import json as _json

    brief_text = _json.dumps(research_brief, indent=2, ensure_ascii=False)
    prompt = HISTORY_SCRIPTWRITER_PROMPT.format(
        topic=research_brief.get("topic", ""),
        era=research_brief.get("era", ""),
        region=research_brief.get("region", ""),
        research_brief=brief_text,
        hook=research_brief.get("hook", ""),
        language_directive=_language_directive(language),
    )
    script = _call_gemini(prompt, temperature=0.75)
    logger.info("History script generated: %d words", len(script.split()))
    return script


def generate_history_image_prompts(script: str, era: str, region: str) -> list[str]:
    """
    Generate 6 historically-styled Flux image prompts for the history script.
    Style is chosen automatically based on the era (oil painting, engraving, etc.).

    Returns a list of exactly 6 prompt strings.
    """
    prompt = HISTORY_IMAGE_PROMPT_GENERATOR_PROMPT.format(
        era=era,
        region=region,
        script=script,
    )
    response = _call_gemini(prompt, temperature=0.5)
    image_prompts: list[str] = _parse_json(response)

    fallbacks = [
        f"Ancient ruins at golden hour, {era} style, oil painting, dramatic chiaroscuro, landscape 16:9",
        f"Rolling hills and ancient architecture of {region}, watercolour illustration, warm tones, wide angle",
        f"Crowded marketplace in historical {region}, detailed engraving style, sepia tones, 16:9",
        f"Storm clouds over a historical fortress, {era}, cinematic oil painting, dramatic lighting",
        f"Ceremonial gathering under torch light, {era} period, illuminated manuscript style, gold detail",
        f"Solitary figure at a crossroads at dusk, {era}, photorealistic oil painting, landscape 16:9",
    ]
    while len(image_prompts) < 6:
        image_prompts.append(fallbacks[len(image_prompts) % len(fallbacks)])

    result = image_prompts[:6]
    logger.info("History scene image prompts generated (%d scenes)", len(result))
    return result


def generate_history_video_script(
    era: str | None = None,
    theme: str | None = None,
    language: str = "english",
) -> HistoryScript:
    """
    Full history pipeline: optional era/theme → HistoryScript.

    Steps:
      1. Pick a history topic (Gemini or curated fallback)
      2. Research the topic (Gemini structured brief)
      3. Write a Ken Burns-style documentary script
      4. Generate SEO metadata
      5. Generate historically-styled Flux image prompts
    """
    from agents.history_agent import pick_history_topic, research_topic

    # 1. Topic selection
    topic_dict = pick_history_topic(era=era, theme=theme)

    # 2. Research
    brief = research_topic(topic_dict)

    # 3. Script
    script = write_history_script(brief, language=language)

    # 4. SEO metadata — use the history topic as context
    seo = generate_seo_metadata(topic_dict["topic"], script, language=language)

    # 5. Image prompts (historically styled) — always English for Pixazo
    image_prompts = generate_history_image_prompts(
        script,
        era=brief.get("era", era or ""),
        region=brief.get("region", ""),
    )
    scenes = parse_scenes(script)

    return HistoryScript(
        topic=topic_dict["topic"],
        title=seo.get("title", topic_dict["topic"])[:100],
        description=seo.get("description", ""),
        tags=seo.get("tags", [])[:15],
        thumbnail_text=seo.get("thumbnail_text", "FORGOTTEN HISTORY"),
        full_script=script,
        scenes=scenes,
        scene_image_prompts=image_prompts,
        era=brief.get("era", era or ""),
        region=brief.get("region", ""),
        research_brief=brief,
    )
