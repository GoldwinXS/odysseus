// Standalone Node test for reasoningEffort.js's effortSupport() — the
// client-side duplicate of apply_reasoning_effort's provider/model gating
// (src/llm_core.py). Follows the same "strip the import, vm.runInContext the
// source" pattern as markdown_codefence_placeholder_regression.mjs, since
// this repo has no generic reusable JS test harness wired into pytest.
//
// Run with: node tests/reasoning_effort_client_regression.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modPath = path.join(__dirname, '..', 'static', 'js', 'reasoningEffort.js');
let src = fs.readFileSync(modPath, 'utf8');

src = src.replace(
  /import uiModule from '\.\/ui\.js';/,
  'const uiModule = { showToast() {} };'
);
src = src.replace(/export async function /g, 'async function ');
src = src.replace(/export function /g, 'function ');
src = src.replace(/export const /g, 'const ');
src = src.replace(/export default \{[\s\S]*?\};?\s*$/, '');
src += '\nthis.__effortSupport = effortSupport;\nthis.__getCurrentEffort = getCurrentEffort;\nthis.__EFFORT_VALUES = EFFORT_VALUES;';

const sandbox = {
  console,
  URL,
  window: { location: { origin: 'http://localhost' } },
  document: {
    getElementById() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
  },
};
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: modPath });

const effortSupport = sandbox.__effortSupport;

let failures = 0;
function check(name, actual, expected) {
  try {
    assert.equal(actual, expected, `${name}: expected ${expected}, got ${actual}`);
    console.log(`ok - ${name}`);
  } catch (e) {
    failures++;
    console.error(`FAIL - ${e.message}`);
  }
}

// Anthropic
check('anthropic supported model', effortSupport('claude-opus-4-6', 'https://api.anthropic.com/v1/messages'), true);
check('anthropic unsupported model', effortSupport('claude-3-5-sonnet', 'https://api.anthropic.com/v1/messages'), false);

// OpenAI
check('openai gpt-5', effortSupport('gpt-5', 'https://api.openai.com/v1/chat/completions'), true);
check('openai o3-mini', effortSupport('o3-mini', 'https://api.openai.com/v1/chat/completions'), true);
check('openai gpt-4o unsupported', effortSupport('gpt-4o', 'https://api.openai.com/v1/chat/completions'), false);

// Gemini
const GEMINI = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';
check('gemini 2.5 flash', effortSupport('gemini-2.5-flash', GEMINI), true);
check('gemini 1.5 pro unsupported', effortSupport('gemini-1.5-pro', GEMINI), false);

// Z.AI / GLM
check('glm-4.5', effortSupport('glm-4.5', 'https://api.z.ai/api/paas/v4/chat/completions'), true);
check('glm-4 unsupported', effortSupport('glm-4', 'https://api.z.ai/api/paas/v4/chat/completions'), false);
check('bigmodel.cn host', effortSupport('glm-5', 'https://open.bigmodel.cn/api/paas/v4/chat/completions'), true);

// DeepSeek — never supported
check('deepseek never supported', effortSupport('deepseek-reasoner', 'https://api.deepseek.com/v1/chat/completions'), false);

// Ollama / local qwen3
check('local qwen3', effortSupport('qwen3:14b', 'http://host.docker.internal:11434/v1/chat/completions'), true);
check('local llama unsupported', effortSupport('llama3.1:8b', 'http://localhost:11434/v1/chat/completions'), false);

// Missing model/endpoint → conservatively "supported" (no warning shown
// before the composer even knows what model is selected).
check('missing model is supported (no premature warning)', effortSupport('', 'https://api.openai.com/v1/chat/completions'), true);
check('missing endpoint is supported (no premature warning)', effortSupport('gpt-4o', ''), true);

if (failures > 0) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log('\nall ok');
