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
