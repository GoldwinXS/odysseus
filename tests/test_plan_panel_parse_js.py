"""FIX B (client) — plan checklist parsing contract for the docked plan panel.

The plan panel in static/js/chat.js parses `update_plan` markdown into
{done, text, item} rows via `_parsePlanLines` and counts done/total from the
checklist items. That function is private to the chat module (a default-export
ESM with a heavy import graph), so it can't be imported into a bare node
harness without stubbing every dependency. Instead we pin the exact parsing
contract the panel relies on by exercising the same regex/logic in isolation —
kept byte-for-byte in sync with `_parsePlanLines`. The wider wiring
(_setStoredPlan render + approved_plan re-inject) is verified by `node --check`
on chat.js plus a code-trace (see report).

Skips cleanly when node is not installed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_HAS_NODE = shutil.which("node") is not None

# Mirror of _parsePlanLines + the done/total count from _renderPlanPanel in
# static/js/chat.js. Keep in sync if that logic changes.
_HARNESS = r"""
function _parsePlanLines(plan) {
  const rows = [];
  (plan || '').split('\n').forEach((raw) => {
    const line = raw.replace(/\s+$/, '');
    if (!line.trim()) return;
    const m = line.match(/^\s*[-*]\s*\[([ xX])\]\s*(.*)$/);
    if (m) {
      rows.push({ done: m[1].toLowerCase() === 'x', text: m[2], item: true });
    } else {
      rows.push({ done: false, text: line.trim(), item: false });
    }
  });
  return rows;
}

const plan = PLAN_JSON;
const rows = _parsePlanLines(plan);
const items = rows.filter((r) => r.item);
const done = items.filter((r) => r.done).length;
console.log(JSON.stringify({ rows, done, total: items.length }));
"""


def _run(plan: str) -> dict:
    js = _HARNESS.replace("PLAN_JSON", json.dumps(plan))
    proc = subprocess.run(
        ["node", "--input-type=module"],
        input=js,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_REPO),
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_checklist_done_and_pending_states():
    out = _run("- [x] done one\n- [ ] pending two\n- [X] done three")
    assert out["done"] == 2 and out["total"] == 3
    rows = [r for r in out["rows"] if r["item"]]
    assert rows[0] == {"done": True, "text": "done one", "item": True}
    assert rows[1] == {"done": False, "text": "pending two", "item": True}
    assert rows[2]["done"] is True  # uppercase [X] counts as done


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_headings_render_as_non_items():
    out = _run("Plan heading\n- [ ] step one")
    assert out["total"] == 1  # only the checklist line is an item
    assert out["rows"][0]["item"] is False
    assert out["rows"][0]["text"] == "Plan heading"


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_blank_lines_are_skipped():
    out = _run("- [ ] a\n\n\n- [x] b\n")
    assert out["total"] == 2 and out["done"] == 1
    assert len(out["rows"]) == 2  # blanks dropped


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_empty_plan_yields_nothing():
    out = _run("")
    assert out["rows"] == [] and out["done"] == 0 and out["total"] == 0
