import json
import logging

logger = logging.getLogger(__name__)


class SendToSubagentTool:
    async def execute(self, content, ctx):
        """
        send_to_subagent: steer a RUNNING background sub-agent mid-flight. The
        parent agent uses this to inject guidance ("focus on X", "also check Y",
        "stop and summarize") into a sub-agent it dispatched with spawn_agent,
        WITHOUT cancelling it.

        Mechanics: each background sub-agent runs with an ephemeral steer-queue
        session (``_subagent_<id>``); this enqueues the message onto that queue
        via agent_runs.enqueue_steer, and the sub-agent's own agent loop drains
        it at the next round boundary (Claude-Code-style steering). Validates the
        target is a sub-agent that is (a) in THIS chat and (b) still running.

        Content (JSON): {"subagent_id": "sub_3", "message": "..."} — a bare
        string is treated as the message with no target (an error). Returns a
        result dict; no subprocess/filesystem.
        """
        from src import agent_runs, subagent_runs

        session_id = ctx.get("session_id")
        if not session_id:
            return "send_to_subagent: no session", {
                "error": "send_to_subagent can only be used inside a chat session.",
                "exit_code": 1,
            }

        sub_id, message = "", ""
        raw = (content or "").strip()
        try:
            parsed = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            parsed = {}
        if isinstance(parsed, dict):
            sub_id = str(parsed.get("subagent_id") or parsed.get("id") or "").strip()
            message = str(parsed.get("message") or parsed.get("text") or "").strip()
        else:
            message = raw

        if not sub_id:
            return "send_to_subagent: invalid", {
                "error": "send_to_subagent needs a `subagent_id` (e.g. 'sub_3') — the id "
                         "spawn_agent returned. Use manage_agents to list running ids.",
                "exit_code": 1,
            }
        if not message:
            return "send_to_subagent: invalid", {
                "error": "send_to_subagent needs a non-empty `message` to send to the sub-agent.",
                "exit_code": 1,
            }

        rec = subagent_runs.find_running(session_id, sub_id)
        if rec is None:
            return f"send_to_subagent: {sub_id} not running", {
                "error": (
                    f"No RUNNING sub-agent '{sub_id}' in this chat. It may have already "
                    "finished (check the chat for its result) or the id is wrong — call "
                    "manage_agents to list running sub-agents and their ids."
                ),
                "exit_code": 1,
            }

        queue_session = rec.get("queue_session")
        # Frame the steer so the sub-agent reads it as guidance from its dispatcher,
        # not as a fresh unrelated user message. Enqueued as kind="user" so it's
        # injected as an in-turn user message the worker acts on this round.
        framed = (
            "[Guidance from the agent that dispatched you — adjust your work "
            f"accordingly]\n{message}"
        )
        queued = bool(queue_session) and agent_runs.enqueue_steer(queue_session, framed, kind="user")
        if not queued:
            # Race: the sub-agent finished between find_running and enqueue.
            return f"send_to_subagent: {sub_id} just finished", {
                "error": (
                    f"Sub-agent '{sub_id}' is no longer accepting steering (it just "
                    "finished). Check the chat for its result."
                ),
                "exit_code": 1,
            }

        desc = f"send_to_subagent: {sub_id}"
        logger.info("Tool executed: %s (%d chars) → queue=%s", desc, len(message), queue_session)
        return desc, {
            "output": (
                f"Sent guidance to running sub-agent {sub_id}. It will pick this up at "
                "its next step and adjust. Do NOT wait for it — its result still posts "
                "into this chat when it finishes."
            ),
            "subagent_id": sub_id,
            "status": "running",
            "exit_code": 0,
        }

class AskUserTool:
    async def execute(self, content, ctx):
        """
        ask_user: the agent poses a multiple-choice question to the user to get a
        decision/clarification. This is a pure UI-control marker — no subprocess,
        no filesystem. It returns an `ask_user` payload that the agent loop turns
        into an `ask_user` SSE event and then ENDS the turn, so the chat waits for
        the user's selection (their choice arrives as the next message).
        """
        question, options, multi = "", [], False
        raw = (content or "").strip()
        try:
            parsed = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            parsed = {}

        if isinstance(parsed, dict):
            question = str(parsed.get("question", "")).strip()
            multi = bool(parsed.get("multi") or parsed.get("multiSelect"))
            # Normalize `options` to a real list. Models sometimes pass it as a
            # STRING (a JSON-encoded array, or newline-joined) instead of an
            # array — iterating that string would yield ONE OPTION PER CHARACTER
            # ("circles" → c,i,r,c,l,e,s). Coerce back to a list first.
            _raw_opts = parsed.get("options") or []
            if isinstance(_raw_opts, str):
                _s = _raw_opts.strip()
                try:
                    _raw_opts = json.loads(_s)
                except (ValueError, TypeError):
                    _raw_opts = [ln.strip() for ln in _s.splitlines() if ln.strip()]
                if isinstance(_raw_opts, str):   # json.loads gave back a bare string
                    _raw_opts = [_raw_opts]
            if not isinstance(_raw_opts, list):
                _raw_opts = [_raw_opts]
            for opt in _raw_opts:
                if isinstance(opt, dict):
                    label = str(opt.get("label", "")).strip()
                    descr = str(opt.get("description", "")).strip()
                elif isinstance(opt, str):
                    label, descr = opt.strip(), ""
                else:
                    continue
                if label:
                    options.append({"label": label, "description": descr})
        else:
            question = raw

        if not question or len(options) < 2:
            return "ask_user: invalid", {
                "error": (
                    "ask_user needs a non-empty `question` and at least 2 `options` "
                    "(each an object with a `label`, optional `description`)."
                ),
                "exit_code": 1,
            }

        options = options[:6]  # keep the choice list sane
        desc = f"ask_user: {question[:80]}"
        labels = ", ".join(o["label"] for o in options)
        result = {
            "ask_user": {"question": question, "options": options, "multi": multi},
            "output": f"Asked the user: {question}\nOptions: {labels}\nAwaiting their selection.",
            "exit_code": 0,
        }
        logger.info("Tool executed: %s (%d options, multi=%s)", desc, len(options), multi)
        return desc, result

class UpdatePlanTool:
    async def execute(self, content, ctx):
        """
        update_plan: the agent writes back to the active plan — tick an item done
        or revise steps (e.g. when the user asks to change something). Pure UI
        marker: returns a `plan_update` payload the agent loop turns into a
        `plan_update` SSE event; the frontend replaces the stored plan and refreshes
        the docked plan window. Does NOT end the turn.
        """
        raw = (content or "").strip()
        plan = ""
        try:
            parsed = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            parsed = {}

        if isinstance(parsed, dict) and parsed.get("plan"):
            plan = str(parsed.get("plan", "")).strip()
        else:
            plan = raw

        if not plan:
            return "update_plan: invalid", {
                "error": "update_plan needs a non-empty `plan` (the full updated checklist as markdown).",
                "exit_code": 1,
            }

        plan = plan[:8192]
        done = plan.count("- [x]") + plan.count("- [X]")
        total = done + plan.count("- [ ]")
        desc = f"update_plan: {done}/{total} done" if total else "update_plan"
        result = {
            "plan_update": {"plan": plan},
            "output": f"Plan updated ({done}/{total} steps complete)." if total else "Plan updated.",
            "exit_code": 0,
        }
        logger.info("Tool executed: %s", desc)
        return desc, result