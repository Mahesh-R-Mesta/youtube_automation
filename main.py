"""
Spiritual Content Pipeline for Instagram Reels — Entry Point
──────────────────────────────────────────────────────────────
Generates a short spiritual Instagram Reel (≤90 seconds) from Indian scriptures
and uploads it to Instagram. Content is explained in Indian-accented English with
Sanskrit text displayed on screen.

Usage examples:
  # Auto-pick: Gemini chooses scripture, topic, mode, and image style
  python main.py --run-now --auto

  # Manual topic (Gemini still generates the full script)
  python main.py --run-now --topic "Bhagavad Gita 2:47 — Nishkama Karma"

  # Pick specific scripture
  python main.py --run-now --scripture bhagavad_gita
  python main.py --run-now --scripture upanishads --mode teaching
  python main.py --run-now --scripture puranas --mode story

  # Choose image style
  python main.py --run-now --auto --image-style cosmic
  python main.py --run-now --auto --image-style tanjore

  # Choose narrator voice gender
  python main.py --run-now --auto --voice male

  # Generate video only — skip Instagram upload
  python main.py --run-now --auto --no-upload

  # Daily scheduler (posts every day at 09:00 UTC)
  python main.py --schedule
  python main.py --schedule --scripture bhagavad_gita --image-style tanjore

Scriptures:
  bhagavad_gita  — Gita slokas and philosophy (default auto)
  upanishads     — Upanishad teachings and verses
  atharva_veda   — Vedic mantras and hymns
  mahabharata    — Epic stories and characters
  puranas        — Deity stories (Vishnu, Shiva, Devi Puranas)

Modes:
  sloka     — Explain a specific Sanskrit verse (default)
  story     — Narrate a scripture-based story
  teaching  — Explain a spiritual concept or philosophy

Image styles:
  tanjore    — Traditional gold-leaf South Indian painting (default)
  vedic      — Ancient manuscript / palm leaf aesthetic
  cosmic     — Celestial divine light and starfield
  minimalist — Modern spiritual minimalism

Environment:
  All configuration is in .env (copy .env.example → .env and fill in your keys).
  Required: GEMINI_API_KEY
  Optional: PIXAZO_API_KEY, INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD,
            ELEVENLABS_API_KEY
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

from utils.logger import logger


def _run_pipeline(
    scripture: str | None = None,
    override_topic: str | None = None,
    mode: str = "sloka",
    image_style: str = "tanjore",
    voice_gender: str = "female",
    skip_upload: bool = False,
) -> None:
    """Instantiate and execute the SpiritualContentFlow."""
    from pipeline.orchestrator import SpiritualContentFlow

    inputs: dict = {
        "scripture": scripture,
        "override_topic": override_topic,
        "mode": mode,
        "image_style": image_style,
        "voice_gender": voice_gender,
        "skip_upload": skip_upload,
    }
    flow = SpiritualContentFlow()
    flow.kickoff(inputs=inputs)


def _start_scheduler(
    scripture: str | None = None,
    mode: str = "sloka",
    image_style: str = "tanjore",
    voice_gender: str = "female",
    test_mode: bool = False,
) -> None:
    """Start APScheduler with a daily cron trigger at settings.schedule_hour:minute."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from config.settings import settings

    scheduler = BlockingScheduler(timezone="UTC")

    job_kwargs = {
        "scripture": scripture,
        "mode": mode,
        "image_style": image_style,
        "voice_gender": voice_gender,
        "skip_upload": False,
    }

    if test_mode:
        fire_at = datetime.now(timezone.utc) + timedelta(minutes=1)
        scheduler.add_job(
            _run_pipeline,
            trigger="date",
            run_date=fire_at,
            id="test_pipeline_run",
            kwargs=job_kwargs,
        )
        logger.info("TEST MODE: Pipeline fires once at %s UTC", fire_at.strftime("%H:%M:%S"))
    else:
        scheduler.add_job(
            _run_pipeline,
            trigger="cron",
            hour=settings.schedule_hour,
            minute=settings.schedule_minute,
            id="daily_spiritual_pipeline",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            kwargs=job_kwargs,
        )
        logger.info(
            "Scheduler started. Pipeline runs daily at %02d:%02d UTC.",
            settings.schedule_hour,
            settings.schedule_minute,
        )

    logger.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Instagram Spiritual Content Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    execution_group = parser.add_mutually_exclusive_group(required=True)
    execution_group.add_argument(
        "--run-now",
        action="store_true",
        help="Run the full pipeline once immediately and exit.",
    )
    execution_group.add_argument(
        "--schedule",
        action="store_true",
        help="Start the daily cron scheduler (blocking — runs until Ctrl+C).",
    )

    # Topic / content selection
    content_group = parser.add_mutually_exclusive_group()
    content_group.add_argument(
        "--auto",
        action="store_true",
        help="Let Gemini auto-pick the scripture, topic, and mode (default behaviour).",
    )
    content_group.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="TOPIC",
        help='Provide a specific topic string, e.g. "Bhagavad Gita 2:47".',
    )

    parser.add_argument(
        "--scripture",
        type=str,
        choices=["bhagavad_gita", "upanishads", "atharva_veda", "mahabharata", "puranas"],
        default=None,
        metavar="SCRIPTURE",
        help=(
            "Constrain auto-pick to a specific scripture. "
            "Choices: bhagavad_gita, upanishads, atharva_veda, mahabharata, puranas"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sloka", "story", "teaching"],
        default=None,
        metavar="MODE",
        help="Content mode: sloka (explain a verse), story (scripture narrative), teaching (philosophy concept).",
    )
    parser.add_argument(
        "--image-style",
        type=str,
        choices=["tanjore", "vedic", "cosmic", "minimalist"],
        default=None,
        metavar="STYLE",
        help="Visual style for generated images: tanjore, vedic, cosmic, or minimalist.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        choices=["female", "male"],
        default="female",
        metavar="GENDER",
        help="Narrator voice gender: 'female' (en-IN-NeerjaNeural, default) or 'male' (en-IN-PrabhatNeural).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Generate the video but skip the Instagram upload. Saves to output/videos/.",
    )
    parser.add_argument(
        "--test-schedule",
        action="store_true",
        help="With --schedule: fire job once in 1 minute instead of waiting for cron time.",
    )

    args = parser.parse_args()

    # Resolve defaults from settings if not provided on CLI
    from config.settings import settings

    resolved_mode = args.mode or settings.default_mode
    resolved_style = args.image_style or settings.default_image_style
    resolved_scripture = args.scripture or settings.default_scripture

    if args.run_now:
        parts = [
            f"scripture={resolved_scripture or 'auto'}",
            f"mode={resolved_mode}",
            f"style={resolved_style}",
            f"voice={args.voice}",
            "no-upload" if args.no_upload else "upload",
        ]
        logger.info("Starting pipeline (%s)...", ", ".join(parts))
        _run_pipeline(
            scripture=resolved_scripture,
            override_topic=args.topic if args.topic else None,
            mode=resolved_mode,
            image_style=resolved_style,
            voice_gender=args.voice,
            skip_upload=args.no_upload,
        )

    elif args.schedule:
        _start_scheduler(
            scripture=resolved_scripture,
            mode=resolved_mode,
            image_style=resolved_style,
            voice_gender=args.voice,
            test_mode=args.test_schedule,
        )


if __name__ == "__main__":
    main()

