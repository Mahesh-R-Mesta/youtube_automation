"""
YouTube Automation Pipeline — Entry Point
──────────────────────────────────────────
Usage:
  # Run the full pipeline immediately (one-shot)
  python main.py --run-now

  # Run with a specific topic (skips trend discovery)
  python main.py --run-now --topic "Historic climate agreement signed by 195 nations"

  # Start the daily scheduler (runs at SCHEDULE_HOUR:SCHEDULE_MINUTE from .env)
  python main.py --schedule

  # Test the scheduler by firing in 1 minute (useful for verifying cron config)
  python main.py --schedule --test-schedule

Environment:
  All configuration is read from .env (see .env.example for required keys).
  Run `cp .env.example .env` then fill in your API keys before first use.
"""

import argparse
import sys
from datetime import datetime, timedelta

from utils.logger import logger


def _run_pipeline(override_topic: str | None = None) -> None:
    """Instantiate and execute the CrewAI pipeline Flow."""
    from pipeline.orchestrator import YouTubeAutomationFlow

    inputs: dict = {}
    if override_topic:
        inputs["override_topic"] = override_topic

    flow = YouTubeAutomationFlow()
    flow.kickoff(inputs=inputs if inputs else None)


def _start_scheduler(test_mode: bool = False) -> None:
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
        help="Override automatic trend discovery with a specific topic string.",
    )
    parser.add_argument(
        "--test-schedule",
        action="store_true",
        help="With --schedule: fire the job once in 1 minute instead of waiting for the cron time.",
    )

    args = parser.parse_args()

    if args.run_now:
        logger.info("Starting pipeline (one-shot mode)...")
        _run_pipeline(override_topic=args.topic)
    elif args.schedule:
        _start_scheduler(test_mode=args.test_schedule)


if __name__ == "__main__":
    main()
