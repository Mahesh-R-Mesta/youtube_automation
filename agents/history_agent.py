"""
History Research Agent
───────────────────────
Responsible for two steps in the history storytelling pipeline:

  1. pick_history_topic(era, theme) → dict
       Asks Gemini to choose one fascinating, under-covered historical story.
       Falls back to a curated list of 20 topics if Gemini fails.

  2. research_topic(topic_dict) → dict
       Asks Gemini for a structured research brief covering key figures,
       timeline, surprising facts, the turning point, and the legacy.

Both functions use the same Gemini client and retry logic as script_agent.
"""

import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types

from config.settings import settings
from config.prompts import HISTORY_TOPIC_PICKER_PROMPT, HISTORY_RESEARCHER_PROMPT
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Fallback topic list (used when Gemini is unavailable / returns bad JSON)
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_TOPICS = [
    {
        "topic": "The Library of Alexandria",
        "era": "Ancient World",
        "region": "North Africa / Egypt",
        "hook": "For seven centuries, it held the sum of all human knowledge — then, in a single generation, it vanished.",
    },
    {
        "topic": "The Silk Road's Forgotten Queens",
        "era": "Medieval Central Asia",
        "region": "Central Asia",
        "hook": "They ruled kingdoms at the crossroads of East and West — yet history almost erased every trace of them.",
    },
    {
        "topic": "The Aztec Florentine Codex",
        "era": "Early Modern (16th Century)",
        "region": "Mesoamerica",
        "hook": "A friar spent fifty years recording a civilization he was sent to destroy — and created the most complete portrait of the Aztec world.",
    },
    {
        "topic": "The Plague of Justinian",
        "era": "Late Antiquity (6th Century)",
        "region": "Byzantine Empire",
        "hook": "It killed half the known world and brought the mightiest empire of its age to its knees — yet most people have never heard of it.",
    },
    {
        "topic": "The Kingdom of Kush",
        "era": "Ancient Africa",
        "region": "Nubia / Sudan",
        "hook": "They conquered Egypt, ruled as pharaohs, and built more pyramids than any civilization on Earth — and history forgot them entirely.",
    },
    {
        "topic": "The Great Emu War",
        "era": "20th Century (1932)",
        "region": "Australia",
        "hook": "In 1932, the Australian Army declared war — on birds. What happened next defies belief.",
    },
    {
        "topic": "Zheng He's Treasure Fleets",
        "era": "Early Modern (15th Century)",
        "region": "East Asia / Indian Ocean",
        "hook": "A century before Columbus, a Chinese admiral commanded a fleet so vast it would not be matched for 500 years — then it was deliberately forgotten.",
    },
    {
        "topic": "The Dancing Plague of 1518",
        "era": "Early Modern Europe",
        "region": "Holy Roman Empire (Strasbourg)",
        "hook": "In the summer of 1518, hundreds of people in a single city could not stop dancing — and dozens danced themselves to death.",
    },
    {
        "topic": "The Haitian Revolution",
        "era": "18th-19th Century",
        "region": "Caribbean",
        "hook": "It was the only successful slave revolt in history — and the nations that claimed to love freedom spent the next two centuries punishing Haiti for it.",
    },
    {
        "topic": "The Real Robinson Crusoe",
        "era": "18th Century",
        "region": "Pacific Ocean",
        "hook": "A stubborn Scottish sailor was marooned alone on an island for four years — and his survival became the blueprint for one of literature's greatest novels.",
    },
    {
        "topic": "The Radium Girls",
        "era": "Early 20th Century",
        "region": "United States",
        "hook": "They were told the paint they swallowed was harmless — and their suffering changed labor law forever.",
    },
    {
        "topic": "The Mongol Postal System",
        "era": "Medieval (13th-14th Century)",
        "region": "Eurasia",
        "hook": "Before the internet, before the telegraph, one empire built a communication network that could carry a message 200 miles in a single day.",
    },
    {
        "topic": "The Night Witches of World War II",
        "era": "20th Century (WWII)",
        "region": "Eastern Europe / Soviet Union",
        "hook": "Flying ancient crop-dusters at night with no radar and no parachutes, they flew over 30,000 combat missions — and the Nazis were terrified of them.",
    },
    {
        "topic": "The Lost Colony of Roanoke",
        "era": "Late 16th Century",
        "region": "North America",
        "hook": "In 1590, an English ship arrived to resupply America's first colony — and found nothing but a single carved word.",
    },
    {
        "topic": "The Tulip Mania of 1637",
        "era": "Early Modern Europe",
        "region": "Dutch Republic",
        "hook": "In 17th-century Holland, a single flower bulb sold for the price of a house — and then, in a single morning, the market collapsed.",
    },
    {
        "topic": "Hypatia of Alexandria",
        "era": "Late Antiquity (4th-5th Century)",
        "region": "Egypt / Roman Empire",
        "hook": "She was the greatest mathematician of her age, a woman who taught men, and her murder marks the moment the ancient world began to die.",
    },
    {
        "topic": "The Great Molasses Flood",
        "era": "Early 20th Century (1919)",
        "region": "United States (Boston)",
        "hook": "On a warm January morning in Boston, a 15-foot wave moving at 35 miles per hour killed 21 people — and it was made entirely of molasses.",
    },
    {
        "topic": "The Children's Crusade",
        "era": "Medieval Europe (1212)",
        "region": "Europe / Mediterranean",
        "hook": "In 1212, thousands of children marched across Europe to reclaim the Holy Land — none of them ever made it.",
    },
    {
        "topic": "The Voynich Manuscript",
        "era": "Early Modern (15th Century)",
        "region": "Unknown / Europe",
        "hook": "For 600 years, the world's most brilliant codebreakers have failed to read a single word of this mysterious illustrated book.",
    },
    {
        "topic": "Nikola Tesla's Final Years",
        "era": "Early 20th Century",
        "region": "United States",
        "hook": "The man who lit the world spent his final years alone in a hotel room, feeding pigeons — while his greatest invention was deliberately destroyed.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini helpers (mirrors script_agent pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def _call_gemini(prompt: str, temperature: float = 0.6) -> str:
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
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def pick_history_topic(
    era: str | None = None,
    theme: str | None = None,
    fallback_index: int | None = None,
) -> dict:
    """
    Ask Gemini to pick one fascinating historical story.

    Args:
        era:            Optional era filter, e.g. "Ancient Rome", "Victorian England".
        theme:          Optional thematic focus, e.g. "forgotten women", "science".
        fallback_index: If set, skip Gemini and return this index from the curated list
                        (useful for deterministic testing).

    Returns:
        dict with keys: topic, era, region, hook
    """
    if fallback_index is not None:
        result = _FALLBACK_TOPICS[fallback_index % len(_FALLBACK_TOPICS)]
        logger.info("Using curated topic [%d]: %s", fallback_index, result["topic"])
        return result

    era_filter = f"Era filter: Focus on {era}.\n" if era else ""
    theme_filter = f"Theme filter: Focus on {theme}.\n" if theme else ""

    prompt = HISTORY_TOPIC_PICKER_PROMPT.format(
        era_filter=era_filter,
        theme_filter=theme_filter,
    )

    try:
        response = _call_gemini(prompt, temperature=0.8)
        result = _parse_json(response)

        # Validate required keys
        required = {"topic", "era", "region", "hook"}
        if not required.issubset(result.keys()):
            raise ValueError(f"Missing keys in response: {required - result.keys()}")

        logger.info("History topic selected: %s (%s)", result["topic"], result["era"])
        return result

    except Exception as exc:
        logger.warning(
            "Gemini topic picker failed (%s). Falling back to curated list.", exc
        )
        import random
        result = random.choice(_FALLBACK_TOPICS)
        logger.info("Fallback topic: %s", result["topic"])
        return result


def research_topic(topic_dict: dict) -> dict:
    """
    Generate a deep historical research brief for the given topic.

    Args:
        topic_dict: Output of pick_history_topic() — must contain topic, era, region, hook.

    Returns:
        Extended dict with keys: topic, era, region, key_figures, timeline,
        surprising_facts, turning_point, legacy.
        On failure, returns topic_dict augmented with stub values so the
        pipeline can continue.
    """
    topic = topic_dict["topic"]
    era = topic_dict["era"]
    region = topic_dict["region"]

    prompt = HISTORY_RESEARCHER_PROMPT.format(
        topic=topic,
        era=era,
        region=region,
    )

    try:
        response = _call_gemini(prompt, temperature=0.4)
        brief = _parse_json(response)

        # Ensure the hook from the picker is preserved
        brief.setdefault("hook", topic_dict.get("hook", ""))

        required = {"key_figures", "timeline", "surprising_facts", "turning_point", "legacy"}
        if not required.issubset(brief.keys()):
            raise ValueError(f"Research brief missing keys: {required - brief.keys()}")

        logger.info(
            "Research brief generated: %d key figures, %d timeline events",
            len(brief.get("key_figures", [])),
            len(brief.get("timeline", [])),
        )
        return brief

    except Exception as exc:
        logger.warning(
            "Gemini researcher failed for '%s' (%s). Using stub brief.", topic, exc
        )
        # Return a minimal stub so downstream pipeline steps can still run
        return {
            "topic": topic,
            "era": era,
            "region": region,
            "hook": topic_dict.get("hook", ""),
            "key_figures": [],
            "timeline": [],
            "surprising_facts": [],
            "turning_point": f"The pivotal moment in the story of {topic}.",
            "legacy": f"The story of {topic} continues to resonate today.",
        }
