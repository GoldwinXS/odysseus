import asyncio
import contextvars
import sys
import time
import collections
from typing import Optional, Callable, Awaitable, Tuple, Dict
from src.constants import MAX_OUTPUT_CHARS

# Cap on bash inside a sub-agent, set by the sub-agent runner for the duration of
# its loop. A sub-agent's whole run is bounded by the same wall-clock cap bash
# defaults to (both 600s), so one blocking foreground command — a dev server the
# model forgot to background — could burn the entire budget and leave the worker
# with nothing to report. When this is set, bash uses it instead of the global
# `agent_bash_timeout_seconds`, so a single command can't eat the run. contextvars
# are task-local, so a sub-agent's cap never leaks into the parent turn's bash.
# (Set by _run_subagent in model_interaction_tools.py — we can't edit agent_loop.py
# to thread it through, so a contextvar is the clean non-invasive route.)
_subagent_bash_timeout: contextvars.ContextVar = contextvars.ContextVar(
    "subagent_bash_timeout", default=None
)


def set_subagent_bash_timeout(seconds: Optional[int]):
    """Bind (or clear) the sub-agent bash cap for the current task context.

    Returns the contextvars token so the caller can reset() it on the way out —
    mirrors _active_workspace's set/reset pattern in tool_execution.py."""
    return _subagent_bash_timeout.set(seconds if (seconds and seconds > 0) else None)


def reset_subagent_bash_timeout(token) -> None:
    """Undo a set_subagent_bash_timeout so the binding never leaks past the run."""
    try:
        _subagent_bash_timeout.reset(token)
    except Exception:
        pass


# A hung foreground command (e.g. a dev server the model forgot to background)
# freezes the whole turn until this fires — at 1 hour that reads as a silent
# stall. Default to 10 minutes (covers real builds/installs) and make it
# tunable via the `agent_bash_timeout_seconds` setting. Progress SSE events
# still stream every PROGRESS_INTERVAL_S so long-but-live commands stay visible.
# Inside a sub-agent the runner-supplied cap (contextvar above) takes precedence
# so a single blocking command can't consume the sub-agent's whole time budget.
def _bash_timeout() -> int:
    _sub = _subagent_bash_timeout.get()
    if _sub and _sub > 0:
        return _sub
    try:
        from src.settings import get_setting
        v = int(get_setting("agent_bash_timeout_seconds", 600) or 600)
        return v if v > 0 else 600
    except Exception:
        return 600

DEFAULT_BASH_TIMEOUT = 60 * 60     # kept for import compatibility; live value via _bash_timeout()
DEFAULT_PYTHON_TIMEOUT = 60 * 60   # kept for import compatibility; live value via _python_timeout()


def _python_timeout() -> int:
    """Foreground wall-clock cap for the ```python``` tool. Mirrors _bash_timeout
    (sub-agent runner cap takes precedence, then the `agent_bash_timeout_seconds`
    setting, default 600s) — the python tool previously hardcoded 3600s, so a
    blocking call (or a server that slipped past the server-detect guard) could
    freeze a turn for a full hour or eat a sub-agent's whole budget."""
    return _bash_timeout()

PROGRESS_INTERVAL_S = 2.0
PROGRESS_TAIL_LINES = 12

async def _run_subprocess_streaming(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float,
    progress_cb: Optional[Callable[[Dict], Awaitable[None]]] = None,
) -> Tuple[str, str, Optional[int], bool]:
    started = time.time()
    stdout_full: list[str] = []
    stderr_full: list[str] = []
    tail = collections.deque(maxlen=PROGRESS_TAIL_LINES)

    async def _reader(stream, full_buf, label: str):
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            full_buf.append(decoded)
            if label == "err":
                tail.append(f"! {decoded}")
            else:
                tail.append(decoded)

    async def _progress_emitter():
        await asyncio.sleep(PROGRESS_INTERVAL_S)
        while True:
            if progress_cb:
                try:
                    await progress_cb({
                        "elapsed_s": round(time.time() - started, 1),
                        "tail": "\n".join(list(tail)),
                    })
                except Exception:
                    pass
            await asyncio.sleep(PROGRESS_INTERVAL_S)

    rd_out = asyncio.create_task(_reader(proc.stdout, stdout_full, "out"))
    rd_err = asyncio.create_task(_reader(proc.stderr, stderr_full, "err"))
    prog_task = asyncio.create_task(_progress_emitter()) if progress_cb else None

    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            pass
    except asyncio.CancelledError:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            pass
        for t in (rd_out, rd_err):
            t.cancel()
        if prog_task is not None:
            prog_task.cancel()
        raise
    finally:
        if prog_task is not None and not prog_task.done():
            prog_task.cancel()
            try:
                await prog_task
            except (asyncio.CancelledError, Exception):
                pass
        for t in (rd_out, rd_err):
            try:
                await asyncio.wait_for(t, timeout=1)
            except Exception:
                pass

    return (
        "\n".join(stdout_full),
        "\n".join(stderr_full),
        proc.returncode,
        timed_out,
    )

class BashTool:
    async def execute(self, content: str, ctx: dict) -> dict:
        from src.tool_execution import agent_cwd, _truncate
        progress_cb = ctx.get("progress_cb")
        _subproc_env = ctx.get("subproc_env")
        proc = await asyncio.create_subprocess_shell(
            content,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_subproc_env,
            cwd=agent_cwd(),
        )
        _bash_to = _bash_timeout()
        stdout, stderr, rc, timed_out = await _run_subprocess_streaming(
            proc,
            timeout=_bash_to,
            progress_cb=progress_cb,
        )
        if timed_out:
            return {"error": f"bash: timed out after {_bash_to}s — process killed. If this is a long-running server, background it (append ` &`) or write output to a file and tail it.", "exit_code": 124, "stdout": _truncate(stdout, MAX_OUTPUT_CHARS), "stderr": _truncate(stderr, MAX_OUTPUT_CHARS)}
        output = stdout.rstrip()
        err = stderr.rstrip()
        if err:
            output = (output + "\nSTDERR: " + err).strip() if output else "STDERR: " + err
        output = _truncate(output, MAX_OUTPUT_CHARS)
        return {"output": output or "(no output)", "exit_code": rc or 0}

class PythonTool:
    async def execute(self, content: str, ctx: dict) -> dict:
        from src.tool_execution import agent_cwd, _truncate
        # A server/event loop in the FOREGROUND python tool never returns and
        # hangs the whole turn (the bash tool auto-detaches servers; python does
        # not). Refuse and point the model at the path that works.
        try:
            from src import bg_jobs
            if bg_jobs.looks_long_running_python(content):
                return {
                    "error": (
                        "This looks like a long-lived server, which the python tool runs in "
                        "the FOREGROUND — it would never return and would hang your whole turn. "
                        "Start servers with the bash tool and `#!bg` as the first line, which "
                        "detaches it so your turn continues, e.g.:\n"
                        "```bash\n#!bg\npython -m http.server 8000 --bind 0.0.0.0\n```\n"
                        "Then confirm it's up with a quick curl and keep working."
                    ),
                    "exit_code": 1,
                }
        except Exception:
            pass
        _timeout = _python_timeout()
        progress_cb = ctx.get("progress_cb")
        _subproc_env = ctx.get("subproc_env")
        proc = await asyncio.create_subprocess_exec(
            (sys.executable or "python"), "-I", "-c", content,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_subproc_env,
            cwd=agent_cwd(),
        )
        stdout, stderr, rc, timed_out = await _run_subprocess_streaming(
            proc,
            timeout=_timeout,
            progress_cb=progress_cb,
        )
        if timed_out:
            return {"error": f"python: timed out after {_timeout}s — process killed", "exit_code": 124, "stdout": _truncate(stdout, MAX_OUTPUT_CHARS), "stderr": _truncate(stderr, MAX_OUTPUT_CHARS)}
        output = stdout.rstrip()
        err = stderr.rstrip()
        if err:
            output = (output + "\nSTDERR: " + err).strip() if output else "STDERR: " + err
        output = _truncate(output, MAX_OUTPUT_CHARS)
        return {"output": output or "(no output)", "exit_code": rc or 0}
