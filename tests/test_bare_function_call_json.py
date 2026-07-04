"""Some Ollama /v1 models emit their tool call as plain message *content*
JSON instead of the structured tool_calls field.

qwen2.5-coder:14b via Ollama's OpenAI-compat endpoint answers
"what files can you see?" with the content:

    {"name": "get_workspace", "arguments": {}}

The agent loop then sees no native call and (with skip_fenced for
schema-carrying models) no fenced block, drops the turn, and the model
looks like it "can't use tools" and refuses. parse_tool_blocks must
recognize this shape when the name is a known tool, and strip_tool_blocks
must remove the raw JSON from display so it stays in lockstep.
"""
import src.agent_tools  # noqa: F401  prime before tool_parsing to avoid circular import
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks


def _tags(text, **kw):
    return [b.tool_type for b in parse_tool_blocks(text, **kw)]


def test_bare_function_call_dispatches_in_both_modes():
    # The exact shape qwen2.5-coder returns via Ollama /v1.
    text = '{"name": "get_workspace", "arguments": {}}'
    # skip_fenced=True mirrors a model sent native schemas (Ollama didn't
    # structure the call); skip_fenced=False mirrors fenced/local mode.
    assert _tags(text, skip_fenced=True) == ["get_workspace"]
    assert _tags(text, skip_fenced=False) == ["get_workspace"]


def test_multiline_and_args_are_parsed():
    text = '{\n  "name": "read_file",\n  "arguments": {"path": "README.md"}\n}'
    blocks = parse_tool_blocks(text, skip_fenced=True)
    assert len(blocks) == 1 and blocks[0].tool_type == "read_file"
    assert "README.md" in blocks[0].content


def test_prose_then_call_is_parsed_and_stripped():
    text = 'Sure, let me check.\n{"name": "ls", "arguments": {"path": "."}}'
    assert _tags(text, skip_fenced=True) == ["ls"]
    # The raw JSON must not survive into the displayed bubble.
    assert strip_tool_blocks(text, skip_fenced=True) == "Sure, let me check."


def test_function_envelope_is_unwrapped():
    text = '{"function": {"name": "get_workspace", "arguments": {}}}'
    assert _tags(text, skip_fenced=True) == ["get_workspace"]


def test_parameters_key_accepted():
    text = '{"name": "get_workspace", "parameters": {}}'
    assert _tags(text, skip_fenced=True) == ["get_workspace"]


def test_unknown_tool_name_is_not_executed():
    # Only names that resolve to a real tool tag may dispatch.
    text = '{"name": "frobnicate", "arguments": {}}'
    assert parse_tool_blocks(text, skip_fenced=True) == []
    assert strip_tool_blocks(text, skip_fenced=True) == text


def test_json_answer_without_args_key_is_not_executed():
    # A data structure that merely contains a "name" field is not a call.
    text = 'Here is the config: {"name": "ls"}'
    assert parse_tool_blocks(text, skip_fenced=True) == []
    assert strip_tool_blocks(text, skip_fenced=True) == text


def test_plain_prose_is_untouched():
    text = "I don't have direct access to your file system."
    assert parse_tool_blocks(text, skip_fenced=True) == []
    assert strip_tool_blocks(text, skip_fenced=True) == text


# --- Finding 2: fenced illustrative JSON must not execute under skip_fenced ---

def test_json_in_fence_is_illustrative_under_skip_fenced():
    # A native-schema model (skip_fenced=True) writing an EXAMPLE call inside a
    # ```json fence must not have it executed — that violates the skip_fenced
    # contract. The bare-JSON scanner used to reach inside the fence and run it.
    text = 'For example:\n```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```'
    assert _tags(text, skip_fenced=True) == []
    # ...and strip must leave the illustrative fence fully visible.
    assert strip_tool_blocks(text, skip_fenced=True) == text


def test_unfenced_json_still_executes_under_skip_fenced():
    # The fenced-skip must not suppress a genuine bare call sitting in prose.
    text = 'Sure.\n{"name": "read_file", "arguments": {"path": "x"}}'
    assert _tags(text, skip_fenced=True) == ["read_file"]


def test_json_in_fence_still_executes_when_not_skip_fenced():
    # Local fenced-block models (skip_fenced=False) DO execute a ```json fence
    # via Pattern 1; the bare-JSON scan is not what runs it there. Guard against
    # accidentally suppressing that path.
    text = '```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```'
    assert _tags(text, skip_fenced=False) == ["read_file"]


# --- Finding 3: parse/strip lockstep for Pattern 8 ---

def test_bash_fence_executes_and_trailing_json_survives():
    # A real bash fence dispatches (Pattern 1), so Pattern 8 never runs and the
    # trailing bare JSON never executed — it must NOT be stripped from display.
    text = '```bash\nls\n```\n{"name": "read_file", "arguments": {"path": "x"}}'
    tags = [b.tool_type for b in parse_tool_blocks(text, skip_fenced=False)]
    assert tags == ["bash"]
    stripped = strip_tool_blocks(text, skip_fenced=False)
    # bash fence gone, but the un-dispatched JSON stays visible.
    assert '{"name": "read_file"' in stripped


def test_lone_bare_call_still_stripped():
    # No earlier pattern → Pattern 8 dispatched it → it must be stripped.
    text = 'ok\n{"name": "ls", "arguments": {"path": "."}}'
    assert _tags(text, skip_fenced=True) == ["ls"]
    assert strip_tool_blocks(text, skip_fenced=True) == "ok"


# --- Finding 4: all bare calls execute; strings tracked only inside braces ---

def test_two_consecutive_bare_calls_both_execute():
    # Only the first used to convert; the second was silently lost.
    text = ('{"name": "ls", "arguments": {"path": "."}}\n'
            '{"name": "get_workspace", "arguments": {}}')
    assert _tags(text, skip_fenced=True) == ["ls", "get_workspace"]


def test_two_consecutive_bare_calls_both_stripped():
    text = ('{"name": "ls", "arguments": {"path": "."}}\n'
            '{"name": "get_workspace", "arguments": {}}')
    assert strip_tool_blocks(text, skip_fenced=True) == ""


def test_unpaired_quote_in_prose_does_not_suppress_call():
    # A stray apostrophe/quote in preceding prose used to flip the scanner into
    # "in string" at depth 0 and swallow the opening brace of the real call.
    text = 'I can\'t see it, let me check.\n{"name": "get_workspace", "arguments": {}}'
    assert _tags(text, skip_fenced=True) == ["get_workspace"]
