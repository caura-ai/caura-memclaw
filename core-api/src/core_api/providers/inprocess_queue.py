"""In-process async job queue backed by stdlib asyncio.

Implements the ``JobQueue`` protocol using fire-and-forget
``asyncio.Task`` objects.  Suitable for single-process deployments
where an external broker (Redis / RabbitMQ) is not needed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class InProcessQueue:
    """Async job queue that runs work in the current event loop."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._scheduled: list[tuple[Any, int, float]] = []  # (func, interval, last_run)
        self._ticker_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(self, func: Any, *args: Any, **kwargs: Any) -> str:
        """Fire-and-forget *func* as a background task."""
        job_id = str(uuid.uuid4())
        task = asyncio.create_task(func(*args, **kwargs))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.debug("Enqueued job %s", job_id)
        return job_id

    async def schedule(self, func: Any, *, cron: str) -> None:
        """Register *func* to run on a cron-like interval."""
        interval = self._parse_cron_interval(cron)
        self._scheduled.append((func, interval, time.monotonic()))
        if self._ticker_task is None or self._ticker_task.done():
            self._ticker_task = asyncio.create_task(self._ticker())
        logger.info("Scheduled job every %ds (cron: %s)", interval, cron)

    async def shutdown(self) -> None:
        """Cancel all tracked and scheduled tasks."""
        if self._ticker_task and not self._ticker_task.done():
            self._ticker_task.cancel()
        for task in list(self._tasks):
            task.cancel()
        all_tasks = list(self._tasks)
        if self._ticker_task:
            all_tasks.append(self._ticker_task)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_cron_interval(self, cron: str) -> int:
        """Convert a simple cron expression to an interval in seconds."""
        cron = cron.strip()

        # */N * * * *  ->  every N minutes
        m = re.fullmatch(r"\*/(\d+)\s+\*\s+\*\s+\*\s+\*", cron)
        if m:
            return int(m.group(1)) * 60

        # 0 */N * * *  ->  every N hours
        m = re.fullmatch(r"0\s+\*/(\d+)\s+\*\s+\*\s+\*", cron)
        if m:
            return int(m.group(1)) * 3600

        # 0 0 * * *  ->  daily
        if re.fullmatch(r"0\s+0\s+\*\s+\*\s+\*", cron):
            return 86400

        logger.warning("Unrecognised cron expression %r; defaulting to 3600s", cron)
        return 3600

    async def _ticker(self) -> None:
        """Wake every 60 s and fire any scheduled functions whose interval has elapsed."""
        while True:
            try:
                await asyncio.sleep(60)
                now = time.monotonic()
                for idx, (func, interval, last_run) in enumerate(self._scheduled):
                    if now - last_run >= interval:
                        await self.enqueue(func)
                        self._scheduled[idx] = (func, interval, now)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Ticker iteration failed, retrying in 5s")
                await asyncio.sleep(5)
