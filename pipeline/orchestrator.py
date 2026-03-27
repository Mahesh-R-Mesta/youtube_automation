"""
Pipeline Orchestrator
──────────────────────
Wires all agents together using a CrewAI Flow (deterministic, event-driven).

Flow steps (sequential):
  init_run → discover_trends → generate_script → create_voiceover
          → fetch_visuals → compile_video → upload_to_youtube

State is tracked in VideoProductionState (a Pydantic model). Each step reads
from and writes to self.state — no data is passed as function arguments.

Usage:
    flow = YouTubeAutomationFlow()
    flow.kickoff()                                     # auto topic selection
    flow.kickoff(inputs={"override_topic": "My Topic"}) # manual topic
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen

from agents.trend_agent import get_trending_topics
from agents.script_agent import generate_video_script
from agents.voice_agent import generate_voiceover
from agents.visual_agent import generate_scene_images
from agents.upload_agent import authenticate, upload_video, set_thumbnail
from pipeline.video_editor import assemble_video, generate_thumbnail
from utils.helpers import setup_output_dirs, sanitize_filename, cleanup_old_outputs
from utils.logger import logger
from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline state
# ─────────────────────────────────────────────────────────────────────────────

class VideoProductionState(BaseModel):
    # Input (optional override from CLI)
    override_topic: Optional[str] = None

    # Runtime
    run_id: str = ""
    headlines: list[dict] = []

    # Script
    topic: str = ""
    full_script: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = []
    thumbnail_text: str = ""
    scenes: list[str] = []
    scene_image_prompts: list[str] = []

    # File paths (stored as strings for Pydantic compatibility)
    audio_path: str = ""
    image_paths: list[str] = []
    video_path: str = ""
    thumbnail_path: str = ""

    # YouTube result
    youtube_video_id: str = ""
    youtube_url: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Flow
# ─────────────────────────────────────────────────────────────────────────────

class YouTubeAutomationFlow(Flow[VideoProductionState]):
    """End-to-end YouTube video production pipeline as a CrewAI Flow."""

    @start()
    def init_run(self) -> None:
        """Initialise run ID and create output directory structure."""
        self.state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_output_dirs(settings.output_dir)
        logger.info("=" * 60)
        logger.info("Pipeline run started  —  ID: %s", self.state.run_id)
        logger.info("=" * 60)

    @listen(init_run)
    def discover_trends(self) -> None:
        """Fetch trending headlines from RSS + Reddit (or use CLI override)."""
        if self.state.override_topic:
            logger.info("Manual topic override: %s", self.state.override_topic)
            self.state.headlines = [
                {
                    "title": self.state.override_topic,
                    "source": "manual override",
                    "url": "",
                    "summary": "",
                }
            ]
        else:
            self.state.headlines = get_trending_topics(max_topics=10)

    @listen(discover_trends)
    def generate_script(self) -> None:
        """Use Gemini to select a topic, write a script, and generate SEO metadata."""
        vs = generate_video_script(self.state.headlines)

        self.state.topic = vs.topic
        self.state.full_script = vs.full_script
        self.state.title = vs.title
        self.state.description = vs.description
        self.state.tags = vs.tags
        self.state.thumbnail_text = vs.thumbnail_text
        self.state.scenes = vs.scenes
        self.state.scene_image_prompts = vs.scene_image_prompts

    @listen(generate_script)
    def create_voiceover(self) -> None:
        """Convert the script to narration audio (edge-tts → ElevenLabs fallback)."""
        audio_path = (
            Path(settings.output_dir)
            / "audio"
            / f"{self.state.run_id}_narration.mp3"
        )
        self.state.audio_path = str(
            generate_voiceover(self.state.full_script, audio_path)
        )

    @listen(create_voiceover)
    def fetch_visuals(self) -> None:
        """Generate one AI scene image per scene via Pixazo Flux."""
        images_dir = Path(settings.output_dir) / "images"
        paths = generate_scene_images(
            image_prompts=self.state.scene_image_prompts,
            output_dir=images_dir,
            run_id=self.state.run_id,
        )
        self.state.image_paths = [str(p) for p in paths]

    @listen(fetch_visuals)
    def compile_video(self) -> None:
        """Assemble the final MP4 and generate the thumbnail."""
        video_dir = Path(settings.output_dir) / "videos"
        safe_title = sanitize_filename(self.state.title, max_length=60)
        video_path = video_dir / f"{self.state.run_id}_{safe_title}.mp4"

        # Auto-detect background music (first audio file in assets/music/)
        music_dir = Path("assets/music")
        music_path: Optional[Path] = None
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
        )
        self.state.video_path = str(assembled)

        # Generate YouTube thumbnail from first scene image
        thumb_path = (
            Path(settings.output_dir)
            / "thumbnails"
            / f"{self.state.run_id}_thumb.jpg"
        )
        generate_thumbnail(
            hero_image_path=Path(self.state.image_paths[0]),
            headline_text=self.state.title,
            thumbnail_text=self.state.thumbnail_text,
            output_path=thumb_path,
        )
        self.state.thumbnail_path = str(thumb_path)

    @listen(compile_video)
    def upload_to_youtube(self) -> None:
        """Authenticate with YouTube and upload the finished video."""
        creds = authenticate(settings.youtube_credentials_file)

        video_id = upload_video(
            credentials=creds,
            video_path=Path(self.state.video_path),
            title=self.state.title,
            description=self.state.description,
            tags=self.state.tags,
            category_id=settings.youtube_category_id,
            privacy_status=settings.youtube_privacy,
        )
        self.state.youtube_video_id = video_id

        if self.state.thumbnail_path:
            set_thumbnail(creds, video_id, Path(self.state.thumbnail_path))

        self.state.youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        logger.info("=" * 60)
        logger.info("VIDEO PUBLISHED  ✓")
        logger.info("URL: %s", self.state.youtube_url)
        logger.info("Title: %s", self.state.title)
        logger.info("=" * 60)

        # Keep output directory tidy — delete oldest runs beyond the last 5
        cleanup_old_outputs(settings.output_dir, keep_latest=5)
