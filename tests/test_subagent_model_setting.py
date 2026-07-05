"""Sub-agent model-role setting: persistence round-trip + endpoint cleanup.

The `subagent_model` role (default model for spawn_agent sub-agents) is
surfaced in the settings UI alongside the other model-role defaults, with an
endpoint + model selector and an ordered fallback chain, mirroring the Utility
role exactly. These tests pin the two things that silently break such a
setting:

1. The three persisted keys (`subagent_model`, `subagent_endpoint_id`,
   `subagent_model_fallbacks`) actually round-trip through the settings
   save/load path. `/api/auth/settings` only writes keys present in
   ``DEFAULT_SETTINGS`` (``for key in DEFAULT_SETTINGS``), so a key missing
   from the defaults is dropped on save even though the frontend sends it.
2. The model-routes endpoint-in-use tracking and clearing logic covers the
   sub-agent primary field and its fallback chain, so disabling/deleting an
   endpoint doesn't leave the sub-agent role pointing at a dead endpoint.
"""

from src import settings

from routes.model_routes import (
    _ENDPOINT_SETTING_FIELDS,
    _ENDPOINT_FALLBACK_FIELDS,
    _endpoint_settings_using_endpoint,
    _clear_endpoint_settings_for_endpoint,
)


SUBAGENT_KEYS = ("subagent_model", "subagent_endpoint_id", "subagent_model_fallbacks")


def _simulate_settings_save(body: dict) -> dict:
    """Mirror routes/auth_routes.py:set_settings — only DEFAULT_SETTINGS keys persist."""
    current = dict(settings.DEFAULT_SETTINGS)
    for key in settings.DEFAULT_SETTINGS:
        if key in body:
            current[key] = body[key]
    return current


def test_subagent_keys_are_default_settings():
    # Without these in DEFAULT_SETTINGS the /api/auth/settings save loop drops
    # them silently (mirrors task_model / task_endpoint_id / *_fallbacks).
    for key in SUBAGENT_KEYS:
        assert key in settings.DEFAULT_SETTINGS, f"{key} missing from DEFAULT_SETTINGS"
    assert settings.DEFAULT_SETTINGS["subagent_model"] == ""
    assert settings.DEFAULT_SETTINGS["subagent_endpoint_id"] == ""
    assert settings.DEFAULT_SETTINGS["subagent_model_fallbacks"] == []


def test_subagent_keys_are_per_user_like_other_roles():
    # Sub-agent role resolves per-user, exactly like default/utility/research.
    for key in SUBAGENT_KEYS:
        assert key in settings._PER_USER_KEYS, f"{key} not per-user resolvable"


def test_subagent_setting_roundtrips_through_save_and_load(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(settings_file))
    settings._invalidate_caches()

    body = {
        "subagent_endpoint_id": "cheap-ep",
        "subagent_model": "deepseek-chat",
        "subagent_model_fallbacks": [
            {"endpoint_id": "cheap-ep", "model": "deepseek-chat"},
            {"endpoint_id": "backup-ep", "model": "glm-4.5-flash"},
        ],
        # A key NOT in DEFAULT_SETTINGS must be ignored by the save loop.
        "not_a_real_setting": "should-be-dropped",
    }
    settings.save_settings(_simulate_settings_save(body))
    settings._invalidate_caches()

    loaded = settings.load_settings()
    assert loaded["subagent_endpoint_id"] == "cheap-ep"
    assert loaded["subagent_model"] == "deepseek-chat"
    assert loaded["subagent_model_fallbacks"] == [
        {"endpoint_id": "cheap-ep", "model": "deepseek-chat"},
        {"endpoint_id": "backup-ep", "model": "glm-4.5-flash"},
    ]
    assert "not_a_real_setting" not in loaded
    # Explicit override is distinguishable from a materialized default.
    assert settings.is_setting_overridden("subagent_model") is True


def test_subagent_blank_default_roundtrips_as_inherit(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(settings_file))
    settings._invalidate_caches()

    # Blank endpoint/model == "inherit the chat's model" (the helper caption).
    settings.save_settings(_simulate_settings_save({
        "subagent_endpoint_id": "",
        "subagent_model": "",
    }))
    settings._invalidate_caches()
    loaded = settings.load_settings()
    assert loaded["subagent_model"] == ""
    assert loaded["subagent_endpoint_id"] == ""


# ── model_routes endpoint-in-use tracking / clearing covers the sub-agent role ──

def test_subagent_role_registered_in_endpoint_field_maps():
    assert _ENDPOINT_SETTING_FIELDS.get("subagent_endpoint_id") == ("subagent_model", "Sub-agents")
    assert _ENDPOINT_FALLBACK_FIELDS.get("subagent_model_fallbacks") == "Sub-agent Model Fallbacks"


def test_subagent_endpoint_references_tracked_and_cleared():
    settings_dict = {
        "subagent_endpoint_id": "dead",
        "subagent_model": "primary",
        "subagent_model_fallbacks": [
            {"endpoint_id": "dead", "model": "fallback-a"},
            {"endpoint_id": "keep", "model": "fallback-b"},
        ],
    }
    assert _endpoint_settings_using_endpoint(settings_dict, "dead") == [
        "Sub-agents",
        "Sub-agent Model Fallbacks",
    ]
    assert _clear_endpoint_settings_for_endpoint(settings_dict, "dead") == [
        "Sub-agents",
        "Sub-agent Model Fallbacks",
    ]
    assert settings_dict["subagent_endpoint_id"] == ""
    assert settings_dict["subagent_model"] == ""
    assert settings_dict["subagent_model_fallbacks"] == [
        {"endpoint_id": "keep", "model": "fallback-b"},
    ]
