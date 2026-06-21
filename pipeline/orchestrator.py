"""
Spiritual Content Pipeline Orchestrator
─────────────────────────────────────────
Wires all agents together using a CrewAI Flow.

Flow steps (sequential):
  init_run → pick_topic → generate_script → create_voiceover
           → fetch_visuals → compile_video → upload_to_instagram

State is tracked in SpiritualProductionState (Pydantic model).
All inter-step data lives on self.state — no argument passing.

Usage (via main.py):
    flow = SpiritualContentFlow()
    flow.kickoff(inputs={
        "scripture": "bhagavad_gita",   # or None for auto-pick
        "mode": "sloka",                # sloka | story | teaching
        "image_style": "tanjore",       # tanjore | vedic | cosmic | minimalist
        "voice_gender": "female",       # female | male
        "skip_upload": False,           # True = generate only
    })
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen

from agents.scripture_agent import pick_auto_topic, get_scripture_content
from agents.script_agent import generate_spiritual_video_script
from agents.voice_agent import generate_voiceover
from agents.visual_agent import generate_scene_images
from agents.upload_agent import upload_reel
from pipeline.video_editor import assemble_video, generate_thumbnail
from utils.helpers import setup_output_dirs, sanitize_filename, cleanup_old_outputs
from utils.logger import logger
from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state
# ─────────────────────────────────────────────────────────────────────────────

class SpiritualProductionState(BaseModel):
    # ── CLI inputs ────────────────────────────────────────────
    scripture: Optional[str] = None      # None → Gemini picks
    override_topic: Optional[str] = None # freeform topic override
    mode: str = "sloka"                  # sloka | story | teaching
    image_style: str = "tanjore"         # tanjore | vedic | cosmic | minimalist
    voice_gender: str = "female"         # female | male
    skip_upload: bool = False

    # ── Runtime ───────────────────────────────────────────────
    run_id: str = ""

    # ── Topic / content ───────────────────────────────────────
    topic: str = ""
    scripture_ref: str = ""
    sanskrit_devanagari: str = "ॐ"
    transliteration: str = "Om"
    english_meaning: str = ""

    # ── Script ────────────────────────────────────────────────
    full_script: str = ""
    scenes: list[str] = []
    scene_image_prompts: list[str] = []

    # ── Instagram metadata ────────────────────────────────────
    title: str = ""
    caption: str = ""
    hashtags: list[str] = []

    # ── File paths (strings for Pydantic compatibility) ───────
    audio_path: str = ""
    image_paths: list[str] = []
    video_path: str = ""
    thumbnail_path: str = ""

    # ── Result ────────────────────────────────────────────────
    instagram_url: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Flow
# ─────────────────────────────────────────────────────────────────────────────

class SpiritualContentFlow(Flow[SpiritualProductionState]):
    """End-to-end Instagram Reel production pipeline for spiritual content."""

    @start()
    def init_run(self) -> None:
        """Initialise run ID and create output directories."""
        self.state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_output_dirs(settings.output_dir)
        logger.info("=" * 60)
        logger.info("Spiritual content pipeline started  —  ID: %s", self.state.run_id)
        logger.info(
            "Scripture: %s | Mode: %s | Style: %s | Voice: %s",
            self.state.scripture or "auto",
            self.state.mode,
            self.state.image_style,
            self.state.voice_gender,
        )
        logger.info("=" * 60)

    @listen(init_run)
    def pick_topic(self) -> None:
        """Select the spiritual topic (Gemini auto-pick or manual override)."""
        if self.state.override_topic:
            # Manual topic: create a minimal topic dict for downstream steps
            logger.info("Manual topic override: %s", self.state.override_topic)
            topic_dict = {
                "scripture": self.state.scripture or settings.default_scripture or "bhagavad_gita",
                "mode": self.state.mode,
                "topic": self.state.override_topic,
                "scripture_ref": self.state.override_topic,
                "hook": self.state.override_topic,
            }
        else:
            topic_dict = pick_auto_topic(
                scripture=self.state.scripture,
                mode=self.state.mode,
            )

        # Store topic details on state
        self.state.topic = topic_dict["topic"]
        self.state.scripture_ref = topic_dict.get("scripture_ref", "")
        self.state.scripture = topic_dict.get("scripture", self.state.scripture or "bhagavad_gita")
        self.state.mode = topic_dict.get("mode", self.state.mode)

        # Retrieve Sanskrit text / content details
        content_dict = get_scripture_content(topic_dict)
        self.state.sanskrit_devanagari = content_dict.get("sanskrit_devanagari", "ॐ")
        self.state.transliteration = content_dict.get("transliteration", "Om")
        self.state.english_meaning = content_dict.get("english_meaning", "")

        # Store merged topic + content for script generation (attach hook)
        self._topic_dict = topic_dict
        self._content_dict = content_dict

        logger.info("Topic selected: [%s] %s", self.state.scripture, self.state.topic)

    @listen(pick_topic)
    def generate_script(self) -> None:
        """Generate the 6-scene script, image prompts, and Instagram metadata."""
        ss = generate_spiritual_video_script(
            topic_dict=self._topic_dict,
            content_dict=self._content_dict,
            image_style=self.state.image_style,
        )

        self.state.full_script = ss.full_script
        self.state.scenes = ss.scenes
        self.state.scene_image_prompts = ss.scene_image_prompts
        self.state.title = ss.title
        self.state.caption = ss.caption
        self.state.hashtags = ss.hashtags
        logger.info("Script ready: %d scenes, %d words", len(ss.scenes), len(ss.full_script.split()))

    @listen(generate_script)
    def create_voiceover(self) -> None:
        """Convert narration to Indian-accented English audio via edge-tts."""
        audio_path = (
            Path(settings.output_dir)
            / "audio"
            / f"{self.state.run_id}_narration.mp3"
        )
        self.state.audio_path = str(
            generate_voiceover(
                self.state.full_script,
                audio_path,
                voice_gender=self.state.voice_gender,
            )
        )

    @listen(create_voiceover)
    def fetch_visuals(self) -> None:
        """Generate one spiritual portrait AI image per scene via Pixazo Flux."""
        images_dir = Path(settings.output_dir) / "images"
        paths = generate_scene_images(
            image_prompts=self.state.scene_image_prompts,
            output_dir=images_dir,
            run_id=self.state.run_id,
        )
        self.state.image_paths = [str(p) for p in paths]

    @listen(fetch_visuals)
    def compile_video(self) -> None:
        """Assemble the portrait Reel MP4 and generate the Instagram thumbnail."""
        video_dir = Path(settings.output_dir) / "videos"
        safe_title = sanitize_filename(self.state.title, max_length=60)
        video_path = video_dir / f"{self.state.run_id}_{safe_title}.mp4"

        # Auto-detect background music
        music_path: Optional[Path] = None
        music_dir = Path("assets/music")
        if music_dir.exists():
            for ext in ("*.mp3", "*.wav", "*.ogg", "*.m4a"):
                candidates = list(music_dir.glob(ext))
                if candidates:
                    music_path = candidates[0]
                    break

        assembled = assemble_video(
            image_paths=[Path(p) for p in self.state.image_paths],
            scenes=self.state.scenes,
            audio_path=Path(self.state.audio_path),
            output_path=video_path,
            music_path=music_path,
            sanskrit_text=self.state.sanskrit_devanagari,
            transliteration=self.state.transliteration,
            scripture_ref=self.state.scripture_ref,
        )
        self.state.video_path = str(assembled)

        # Instagram square thumbnail (1080×1080)
        thumb_path = (
            Path(settings.output_dir)
            / "thumbnails"
            / f"{self.state.run_id}_thumb.jpg"
        )
        generate_thumbnail(
            hero_image_path=Path(self.state.image_paths[0]),
            title_text=self.state.title,
            sanskrit_text=self.state.sanskrit_devanagari,
            output_path=thumb_path,
        )
        self.state.thumbnail_path = str(thumb_path)

    @listen(compile_video)
    def upload_to_instagram(self) -> None:
        """Upload the Reel to Instagram (skipped if skip_upload=True)."""
        if self.state.skip_upload:
            logger.info("Upload skipped (--no-upload flag). Video saved at: %s", self.state.video_path)
            self.state.instagram_url = f"file://{self.state.video_path}"
            return

        if not settings.instagram_username or not settings.instagram_password:
            logger.warning(
                "Instagram credentials not set. Skipping upload. "
                "Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env."
            )
            self.state.instagram_url = f"file://{self.state.video_path}"
            return

        # Build full caption: caption text + newline + space-separated hashtags
        hashtag_block = " ".join(self.state.hashtags)
        full_caption = f"{self.state.caption}\n\n{hashtag_block}"

        try:
            url = upload_reel(
                video_path=Path(self.state.video_path),
                caption=full_caption,
                cover_image_path=Path(self.state.thumbnail_path) if self.state.thumbnail_path else None,
            )
            self.state.instagram_url = url
        except Exception as exc:
            logger.error("Instagram upload failed: %s", exc)
            logger.info(
                "Video is saved locally at: %s — upload manually if needed.",
                self.state.video_path,
            )
            self.state.instagram_url = f"file://{self.state.video_path}"
            return

        logger.info("=" * 60)
        logger.info("REEL PUBLISHED  ✓")
        logger.info("URL: %s", self.state.instagram_url)
        logger.info("Title: %s", self.state.title)
        logger.info("=" * 60)

        cleanup_old_outputs(settings.output_dir, keep_latest=5)


