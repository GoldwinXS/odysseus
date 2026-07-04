"""manage_agents must be reachable from the model through BOTH tool-call
surfaces — the fenced/text parser and the native-function converter.

The bug that shipped: manage_agents was in TOOL_HANDLERS, ALWAYS_AVAILABLE,
FUNCTION_TOOL_SCHEMAS and the system prompt, but MISSING from TOOL_TAGS. So:

  * the fenced fence-regex (built from TOOL_TAGS) never matched
    ```manage_agents ...```, and
  * function_call_to_tool_block('manage_agents', ...) returned None at the
    `tool_type not in TOOL_TAGS` guard — the native call was silently dropped.

And even once reachable, the generic json.dumps fallback would hand the handler
'{"action": "stop sub_3"}', whose startswith("stop") never matches, so a cancel
silently degraded into a list. These tests exercise the parse + native-convert
layer the existing manage_agents tests bypassed — which is exactly how the bug
shipped.
"""
import sys
from unittest.mock import MagicMock

for mod in ['src.agent_tools', 'src.tool_parsing', 'src.tool_schemas', 'src.tool_execution']:
    sys.modules.pop(mod, None)
for mod in [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'core.models', 'core.database', 'core.auth'
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import src.agent_tools  # noqa: E402, F401
from src.agent_tools import TOOL_TAGS  # noqa: E402
from src.tool_parsing import parse_tool_blocks  # noqa: E402
from src.tool_schemas import function_call_to_tool_block  # noqa: E402


def test_manage_agents_is_in_tool_tags():
    # The root cause: without this entry the fence regex and the native
    # converter both drop the call.
    assert "manage_agents" in TOOL_TAGS


def test_fenced_manage_agents_list_parses():
    # ```manage_agents\n``` (empty body) is the no-arg "list" shape; it must
    # dispatch with empty content, not vanish.
    blocks = parse_tool_blocks('```manage_agents\n```')
    assert [(b.tool_type, b.content) for b in blocks] == [("manage_agents", "")]


def test_fenced_manage_agents_stop_parses():
    blocks = parse_tool_blocks('```manage_agents\nstop sub_3\n```')
    assert [(b.tool_type, b.content) for b in blocks] == [("manage_agents", "stop sub_3")]


def test_native_convert_stop_produces_handler_matchable_content():
    # The half that json.dumps would have broken: the handler's matcher does
    # content.strip().lower().startswith("stop"), so the converted content must
    # be the bare action string, NOT '{"action": "stop sub_3"}'.
    block = function_call_to_tool_block("manage_agents", '{"action": "stop sub_3"}')
    assert block is not None
    assert block.tool_type == "manage_agents"
    assert block.content == "stop sub_3"
    # Prove the handler's own matcher accepts it (this is what silently failed).
    assert block.content.strip().lower().startswith("stop")


def test_native_convert_list_and_empty_action():
    # Empty / "list" both map to the plain content the handler treats as list.
    assert function_call_to_tool_block("manage_agents", '{}').content == ""
    assert function_call_to_tool_block("manage_agents", '{"action": "list"}').content == "list"


def test_native_convert_no_args_is_list():
    # A native call with no arguments at all (schema marks action optional).
    block = function_call_to_tool_block("manage_agents", "")
    assert block is not None and block.content == ""
