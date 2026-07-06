// static/js/reasoningEffort.js
// Reasoning-effort selector (Default/Off/Low/Medium/High) — Claude-app-style
// composer control. Persists PER SESSION via /api/session/{id}/prefs; a new
// chat starts from the global reasoning_effort_default setting.
//
// The provider/model support mapping mirrors apply_reasoning_effort in
// src/llm_core.py (client-side duplicate, intentionally kept in this one
// exported function — see effortSupport below — rather than a network
// round-trip just to grey out an unsupported selection).

import uiModule from './ui.js';

const API_BASE = window.location.origin;

export const EFFORT_VALUES = ['default', 'off', 'low', 'medium', 'high'];
const EFFORT_LABELS = { default: 'Default', off: 'Off', low: 'Low', medium: 'Medium', high: 'High' };

let _currentEffort = 'default';
let _globalDefault = 'default';
// Session id this module last loaded prefs for — guards against a slow GET
// /prefs response landing after the user has already switched sessions again.
let _loadedForSession = null;

function _hostOf(url) {
  try { return new URL(url).hostname.toLowerCase().replace(/\.$/, ''); }
  catch (_) { return ''; }
}
function _hostMatch(url, domain) {
  const h = _hostOf(url);
  return !!h && (h === domain || h.endsWith('.' + domain));
}
function _modelStartsWith(model, prefixes) {
  if (!model) return false;
  const m = String(model).toLowerCase().split('/').pop();
  return prefixes.some(p => m.startsWith(p));
}
function _modelContains(model, patterns) {
  if (!model) return false;
  const m = String(model).toLowerCase();
  return patterns.some(p => m.includes(p));
}

const _GEMINI_REASONING_MODEL_PATTERNS = ['gemini-2.5', 'gemini-3'];
const _ZAI_THINKING_MODEL_PATTERNS = ['glm-4.5', 'glm-4.6', 'glm-5'];
const _QWEN3_MODEL_PATTERNS = ['qwen3', 'qwen-3'];

/**
 * Mirrors llm_core.py's apply_reasoning_effort provider/model gating (plus
 * Anthropic, which IS supported there via the separate `effort=` kwarg path —
 * see that function's docstring for why it isn't a payload mutation).
 * Returns true if the CURRENT effort selector actually does something for
 * this model/endpoint, false if it would be a no-op (UI shows "(not
 * supported by this model)" in that case — see updateReasoningEffortUI).
 * "default" is always "supported" (it's always a no-op by definition, so
 * there's nothing to warn about).
 */
export function effortSupport(modelId, endpointUrl) {
  if (!modelId || !endpointUrl) return true;
  if (_hostMatch(endpointUrl, 'anthropic.com')) {
    return _modelContains(modelId, ['claude-3-7', 'claude-3.7', 'claude-opus-4', 'claude-sonnet-4', 'claude-haiku-4']);
  }
  if (_hostMatch(endpointUrl, 'openai.com')) {
    return _modelStartsWith(modelId, ['gpt-5']) || _modelStartsWith(modelId, ['o1', 'o3', 'o4']);
  }
  if (_hostMatch(endpointUrl, 'generativelanguage.googleapis.com')) {
    return _modelContains(modelId, _GEMINI_REASONING_MODEL_PATTERNS);
  }
  if (_hostMatch(endpointUrl, 'z.ai') || _hostMatch(endpointUrl, 'bigmodel.cn')) {
    return _modelContains(modelId, _ZAI_THINKING_MODEL_PATTERNS);
  }
  if (_hostMatch(endpointUrl, 'deepseek.com')) {
    return false; // no effort param exists on any DeepSeek chat model
  }
  // Ollama / self-hosted OpenAI-compat: only the qwen3 "/no_think" switch is
  // recognized, and only for "off" — but we don't know the SELECTED value
  // here (this just answers "does this knob do anything for this model at
  // all"), so qwen3 counts as supported (off works; low/med/high don't, but
  // that's a finer distinction than the tooltip needs).
  return _modelContains(modelId, _QWEN3_MODEL_PATTERNS);
}

export function getCurrentEffort() {
  return _currentEffort;
}

function _applyButtonLabel() {
  const btn = document.getElementById('reasoning-effort-btn');
  const label = document.getElementById('reasoning-effort-label');
  if (!label || !btn) return;
  label.textContent = EFFORT_LABELS[_currentEffort] || 'Default';
  btn.classList.toggle('effort-active', _currentEffort !== 'default');
  document.querySelectorAll('.reasoning-effort-item').forEach(item => {
    const on = item.dataset.effort === _currentEffort;
    item.classList.toggle('active', on);
    item.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  _refreshSupportIndicator();
}

function _refreshSupportIndicator() {
  const btn = document.getElementById('reasoning-effort-btn');
  if (!btn) return;
  const label = document.getElementById('reasoning-effort-label');
  const sm = window.sessionModule;
  const modelId = sm && sm.getCurrentModel ? sm.getCurrentModel() : null;
  const endpointUrl = sm && sm.getCurrentEndpointUrl ? sm.getCurrentEndpointUrl() : null;
  // Model-level support, independent of the current selection: on models with
  // no effort mapping at all (e.g. DeepSeek chat) every choice is a no-op, so
  // say it on the button itself ("N/A") instead of a subtle post-selection dim
  // that users read as "the knob is broken". effortSupport(null, ...) returns
  // true, so a not-yet-resolved model doesn't flash N/A.
  const modelSupported = effortSupport(modelId, endpointUrl);
  btn.classList.toggle('effort-unsupported', !modelSupported);
  if (!modelSupported && label) label.textContent = 'N/A';
  btn.title = modelSupported
    ? 'Reasoning effort'
    : 'Reasoning effort: not supported by this model — selection has no effect';
}

async function _fetchGlobalDefault() {
  try {
    const res = await fetch(`${API_BASE}/api/auth/settings`, { credentials: 'same-origin' });
    if (!res.ok) return 'default';
    const settings = await res.json();
    const val = settings && settings.reasoning_effort_default;
    return EFFORT_VALUES.includes(val) ? val : 'default';
  } catch (_) { return 'default'; }
}

async function _fetchSessionEffort(sessionId) {
  try {
    const res = await fetch(`${API_BASE}/api/session/${sessionId}/prefs`, { credentials: 'same-origin' });
    if (!res.ok) return null;
    const data = await res.json();
    const val = data && data.prefs && data.prefs.reasoning_effort;
    return EFFORT_VALUES.includes(val) ? val : null;
  } catch (_) { return null; }
}

async function _persistSessionEffort(sessionId, effort) {
  if (!sessionId) return; // pending (not-yet-materialized) chat — kept in memory only
  try {
    await fetch(`${API_BASE}/api/session/${sessionId}/prefs`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reasoning_effort: effort }),
    });
  } catch (_) { /* best-effort — the in-memory value still applies this session */ }
}

/**
 * Reload the composer's effort selection for a (possibly new) session.
 * Called on session switch / new chat. Falls back to the global default when
 * the session has no stored override (or has no id yet — a pending chat).
 */
export async function notifySessionChanged(sessionId) {
  const token = sessionId || null;
  _loadedForSession = token;
  let effort = null;
  if (sessionId) {
    effort = await _fetchSessionEffort(sessionId);
    // A slower-arriving response for a session the user already left — drop it.
    if (_loadedForSession !== token) return;
  }
  _currentEffort = effort || _globalDefault;
  _applyButtonLabel();
}

/** Call after the composer's model selection changes, to refresh the
 * "(not supported by this model)" tooltip state without touching the value. */
export function updateReasoningEffortUI() {
  _refreshSupportIndicator();
}

export async function initReasoningEffort() {
  const wrap = document.getElementById('reasoning-effort-wrap');
  const btn = document.getElementById('reasoning-effort-btn');
  const menu = document.getElementById('reasoning-effort-menu');
  if (!wrap || !btn || !menu) return;

  _globalDefault = await _fetchGlobalDefault();
  const sm = window.sessionModule;
  const sid = sm && sm.getCurrentSessionId ? sm.getCurrentSessionId() : null;
  await notifySessionChanged(sid);

  function _close() {
    menu.classList.add('hidden');
    btn.setAttribute('aria-expanded', 'false');
  }
  function _open() {
    menu.classList.remove('hidden');
    btn.setAttribute('aria-expanded', 'true');
    _refreshSupportIndicator();
  }

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (menu.classList.contains('hidden')) _open();
    else _close();
  });

  menu.querySelectorAll('.reasoning-effort-item').forEach((item) => {
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      const effort = item.dataset.effort;
      if (!EFFORT_VALUES.includes(effort)) return;
      _currentEffort = effort;
      _applyButtonLabel();
      _close();
      const curSid = sm && sm.getCurrentSessionId ? sm.getCurrentSessionId() : null;
      _persistSessionEffort(curSid, effort);
      if (uiModule && uiModule.showToast) {
        uiModule.showToast('Reasoning effort: ' + (EFFORT_LABELS[effort] || effort));
      }
    });
  });

  document.addEventListener('click', (e) => {
    if (!menu.classList.contains('hidden') && !menu.contains(e.target) && e.target !== btn) {
      _close();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !menu.classList.contains('hidden')) _close();
  });

  // A model pick doesn't change the effort VALUE, only whether it's supported
  // by the newly-picked model — refresh the tooltip/dimmed state only.
  document.addEventListener('odysseus:model-picked', _refreshSupportIndicator);
}

export default {
  EFFORT_VALUES,
  effortSupport,
  getCurrentEffort,
  notifySessionChanged,
  updateReasoningEffortUI,
  initReasoningEffort,
};
