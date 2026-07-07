"""ask_user must not split a STRING `options` into one option per character.

Models sometimes pass options as a JSON-encoded string or a newline-joined
string instead of an array; iterating that string char-by-char turned "circles"
into buttons c,i,r,c,l,e,s. The handler now normalizes options to a real list.
"""

import asyncio
import json

from src.agent_tools.interaction_tools import AskUserTool


def _run(payload):
    desc, result = asyncio.run(AskUserTool().execute(json.dumps(payload), {}))
    assert "ask_user" in result, result   # not the invalid-input error path
    return result["ask_user"]


def test_options_as_json_array_of_strings():
    r = _run({"question": "Pick a shape", "options": ["circles", "squares"]})
    labels = [o["label"] for o in r["options"]]
    assert labels == ["circles", "squares"]


def test_options_as_json_encoded_string_is_parsed_not_char_split():
    # The model sent options as a STRING containing a JSON array.
    r = _run({"question": "Pick a shape",
              "options": json.dumps([{"label": "circles"}, {"label": "squares"}])})
    labels = [o["label"] for o in r["options"]]
    assert labels == ["circles", "squares"]        # NOT ['c','i','r',...]


def test_options_as_newline_joined_string():
    r = _run({"question": "Pick a shape", "options": "circles\nsquares\ntriangles"})
    labels = [o["label"] for o in r["options"]]
    assert labels == ["circles", "squares", "triangles"]


def test_wellformed_object_options_unchanged():
    r = _run({"question": "Pick", "options": [
        {"label": "A", "description": "first"}, {"label": "B", "description": "second"}]})
    assert r["options"] == [
        {"label": "A", "description": "first"}, {"label": "B", "description": "second"}]
