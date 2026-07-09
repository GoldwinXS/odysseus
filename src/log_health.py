# src/log_health.py
#
# Self-healing file logging for Windows.
#
# OBSERVED FAILURE (2026-07-09): the live server (PID 4652) stopped writing to
# data/logs/app.log at 16:36 while continuing to serve chat for 4+ hours — the
# stdlib RotatingFileHandler died silently (logging.Handler.handleError swallows
# every emit/rollover exception), leaving the app unobservable exactly while it
# was producing the day's worst stall spike. Two Windows-specific hazards:
#   - doRollover() renames the OPEN log file; any other handle on it (tail
#     viewer, AV scan, orphaned handle) makes the rename raise, and the handler
#     can be left with a closed/broken stream for the rest of the process.
#   - a transient OSError on write (disk pressure, handle invalidation) also
#     kills the stream, and every later emit fails silently.
#
# Fix = two independent layers:
#   1. SelfHealingRotatingFileHandler — rollover failures skip rotation instead
#      of breaking the stream (we keep appending to the current file and retry
#      on a later emit); a failed write closes and reopens the stream once and
#      retries the record.
#   2. start_log_canary() — an asyncio task that writes one heartbeat line per
#      interval, flushes, then checks the log file's mtime. If the file went
#      stale anyway (handler dead in a way layer 1 couldn't catch), it REBUILDS
#      the file handler from scratch and says so on the console. The heartbeat
#      also gives external monitors (autostart script, `Get-Item`) a liveness
#      signal: "process up but app.log mtime stale" now means logging is broken,
#      not maybe-idle.

import asyncio
import logging
import logging.handlers
import os
import time

logger = logging.getLogger(__name__)

CANARY_INTERVAL_S = 120
# mtime older than this after a flushed heartbeat => the file handler is dead.
CANARY_STALE_S = 30


class SelfHealingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that survives Windows rename/write failures.

    Rotation is best-effort: when the rename dance fails (file locked by a
    viewer/AV), we keep appending to the current file instead of losing the
    stream — an oversized log beats a silent one. A failed write reopens the
    stream once and retries the record before giving up on it."""

    def doRollover(self):
        try:
            super().doRollover()
        except Exception:
            # Rename failed (locked file). Make sure we still have an open
            # stream on the CURRENT file and carry on un-rotated.
            try:
                if self.stream is None or self.stream.closed:
                    self.stream = self._open()
            except Exception:
                pass

    def emit(self, record):
        try:
            if self.shouldRollover(record):
                self.doRollover()
            logging.FileHandler.emit(self, record)
        except Exception:
            # One reopen-and-retry; then fall back to stdlib error handling.
            try:
                try:
                    if self.stream:
                        self.stream.close()
                except Exception:
                    pass
                self.stream = self._open()
                logging.FileHandler.emit(self, record)
            except Exception:
                self.handleError(record)


def build_file_handler(log_file: str, formatter: logging.Formatter) -> logging.Handler:
    h = SelfHealingRotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    h.setFormatter(formatter)
    return h


def _rebuild_file_handler(log_file: str) -> None:
    """Tear down every file handler on the root logger and attach a fresh one."""
    root = logging.getLogger()
    formatter = None
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            formatter = h.formatter or formatter
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    if formatter is None:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root.addHandler(build_file_handler(log_file, formatter))


async def _canary_loop(log_file: str) -> None:
    while True:
        await asyncio.sleep(CANARY_INTERVAL_S)
        try:
            logger.info("log-canary")
            for h in logging.getLogger().handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            mtime = os.path.getmtime(log_file) if os.path.exists(log_file) else 0
            if time.time() - mtime > CANARY_STALE_S:
                # The heartbeat we just flushed never reached the file: the
                # handler is dead. Rebuild it and announce on both channels
                # (console always works; the new file handler proves itself
                # with this warning line).
                _rebuild_file_handler(log_file)
                logging.getLogger().warning(
                    "[log-health] file logging was dead (stale mtime) — handler rebuilt"
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Never let the canary die; console-print as the last resort.
            try:
                print(f"[log-health] canary iteration failed: {e}", flush=True)
            except Exception:
                pass


def start_log_canary(log_file: str) -> asyncio.Task:
    """Start the logging watchdog. Call from app lifespan startup."""
    return asyncio.create_task(_canary_loop(log_file), name="log-canary")
