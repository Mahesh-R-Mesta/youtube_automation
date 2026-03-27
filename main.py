"""
YouTube Automation Pipeline — Entry Point
──────────────────────────────────────────
Usage:
  # News pipeline — run once immediately
  python main.py --run-now

  # News pipeline — override topic
  python main.py --run-now --topic "Historic climate agreement signed by 195 nations"

  # History storytelling — let Gemini pick any topic
  python main.py --run-now --mode history

  # History storytelling — constrain era and/or theme
  python main.py --run-now --mode history --era "Ancient Rome"
  python main.py --run-now --mode history --theme "forgotten women"
  python main.py --run-now --mode history --era "Medieval Europe" --theme "science"

  # Hindi language — script + audio in Hindi
  python main.py --run-now --language hindi
  python main.py --run-now --mode history --era "Ancient India" --language hindi

  # Start the daily scheduler (news mode by default)
  python main.py --schedule
  python main.py --schedule --mode history

  # Test the scheduler by firing in 1 minute
  python main.py --schedule --test-schedule

Environment:
  All configuration is read from .env (see .env.example for required keys).
  Run `cp .env.example .env` then fill in your API keys before first use.
"""

import argparse
import sys
from datetime import datetime, timedelta

from utils.logger import logger


def _run_pipeline(
    mode: str = "news",
    override_topic: str | None = None,
    era: str | None = None,
    theme: str | None = None,
    language: str = "english",
) -> None:
    """Instantiate and execute the appropriate CrewAI pipeline Flow."""
    if mode == "history":
        from pipeline.orchestrator import HistoryStorytellerFlow
        inputs: dict = {"language": language}
        if era:
            inputs["era"] = era
        if theme:
            inputs["theme"] = theme
        flow = HistoryStorytellerFlow()
        flow.kickoff(inputs=inputs)
    else:
        from pipeline.orchestrator import YouTubeAutomationFlow
        inputs = {"language": language}
        if override_topic:
            inputs["override_topic"] = override_topic
        flow = YouTubeAutomationFlow()
        flow.kickoff(inputs=inputs)


def _start_scheduler(
    mode: str = "news",
    era: str | None = None,
    theme: str | None = None,
    language: str = "english",
    test_mode: bool = False,
) -> None:
    """
    Start APScheduler with a daily cron trigger.

    In test_mode, the job is scheduled to fire once 1 minute from now
    so you can verify the scheduler is working without waiting until 9 AM.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from config.settings import settings

    scheduler = BlockingScheduler(timezone="UTC")

    if test_mode:
        # Fire once in 1 minute for testing purposes
        fire_at = datetime.utcnow() + timedelta(minutes=1)
        scheduler.add_job(
            _run_pipeline,
            trigger="date",
            run_date=fire_at,
            id="test_pipeline_run",
            kwargs={"mode": mode, "era": era, "theme": theme, "language": language},
        )
        logger.info("TEST MODE: Pipeline will run once at %s UTC", fire_at.strftime("%H:%M:%S"))
    else:
        scheduler.add_job(
            _run_pipeline,
            trigger="cron",
            hour=settings.schedule_hour,
            minute=settings.schedule_minute,
            id="daily_youtube_pipeline",
            replace_existing=True,
            coalesce=True,          # collapse missed runs into one if scheduler was offline
            max_instances=1,        # never run two pipeline instances simultaneously
            kwargs={"mode": mode, "era": era, "theme": theme, "language": language},
        )
        logger.info(
            "Scheduler started. Pipeline runs daily at %02d:%02d UTC.",
            settings.schedule_hour,
            settings.schedule_minute,
        )

    logger.info("Press Ctrl+C to stop the scheduler.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="YouTube Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    execution_group = parser.add_mutually_exclusive_group(required=True)
    execution_group.add_argument(
        "--run-now",
        action="store_true",
        help="Execute the full pipeline once immediately and exit.",
    )
    execution_group.add_argument(
        "--schedule",
        action="store_true",
        help="Start the daily scheduler (blocking — runs until Ctrl+C).",
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="TOPIC",
        help="(news mode only) Override automatic trend discovery with a specific topic string.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["news", "history"],
        default="news",
        metavar="MODE",
        help="Pipeline mode: 'news' (default) or 'history'. History mode generates a storytelling video about a real historical event.",
    )
    parser.add_argument(
        "--era",
        type=str,
        default=None,
        metavar="ERA",
        help="(history mode) Optional era filter, e.g. 'Ancient Rome', 'Victorian England', '20th Century'.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        metavar="THEME",
        help="(history mode) Optional thematic focus, e.g. 'forgotten women', 'science discoveries', 'lost civilizations'.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["english", "hindi"],
        default="english",
        metavar="LANGUAGE",
        help="Script and audio language: 'english' (default) or 'hindi'. "
             "Hindi mode writes the script in Devanagari and uses the hi-IN-SwaraNeural voice.",
    )
    parser.add_argument(
        "--test-schedule",
        action="store_true",
        help="With --schedule: fire the job once in 1 minute instead of waiting for the cron time.",
    )

    args = parser.parse_args()

    if args.run_now:
        mode_label = f"{args.mode} mode"
        if args.language != "english":
            mode_label += f" [{args.language}]"
        if args.mode == "history" and (args.era or args.theme):
            filters = ", ".join(f for f in [args.era, args.theme] if f)
            mode_label += f" [{filters}]"
        logger.info("Starting pipeline (%s)...", mode_label)
        _run_pipeline(
            mode=args.mode,
            override_topic=args.topic,
            era=args.era,
            theme=args.theme,
            language=args.language,
        )
    elif args.schedule:
        _start_scheduler(
            mode=args.mode,
            era=args.era,
            theme=args.theme,
            language=args.language,
            test_mode=args.test_schedule,
        )


if __name__ == "__main__":
    main()
