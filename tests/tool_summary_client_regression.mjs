// Standalone Node test for chatRenderer.js's buildToolSummary() — the
// Claude-app-style collapsed group summary ("Used N tools: ..."). This
// function has no DOM/module dependencies of its own, so rather than
// stubbing chatRenderer.js's full import chain (markdown.js, ui.js,
// providers.js, settings.js, spinner.js, escMenuStack.js, matchKey.js), we
// extract just its self-contained source slice (the phrase map + the two
// functions) via a bounded regex and eval that in isolation. Mirrors the
// spirit of markdown_codefence_placeholder_regression.mjs (vm-sandbox a
// trimmed slice of a real source file) without paying for the whole chain.
//
// Run with: node tests/tool_summary_client_regression.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rendererPath = path.join(__dirname, '..', 'static', 'js', 'chatRenderer.js');
const fullSrc = fs.readFileSync(rendererPath, 'utf8');

const startMarker = '// ---- Collapsed tool-group summary';
const endMarker = '\nexport function toolChipSummary';
const startIdx = fullSrc.indexOf(startMarker);
const endIdx = fullSrc.indexOf(endMarker);
assert.ok(startIdx >= 0, 'start marker not found — buildToolSummary block may have moved/been renamed');
assert.ok(endIdx > startIdx, 'end marker not found — buildToolSummary block may have moved/been renamed');

let slice = fullSrc.slice(startIdx, endIdx);
slice = slice.replace(/export function /g, 'function ');
slice += '\nthis.__buildToolSummary = buildToolSummary;';

const sandbox = { console };
vm.createContext(sandbox);
vm.runInContext(slice, sandbox, { filename: rendererPath });
const buildToolSummary = sandbox.__buildToolSummary;

let failures = 0;
function check(name, actual, expected) {
  try {
    assert.equal(actual, expected, `${name}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    console.log(`ok - ${name}`);
  } catch (e) {
    failures++;
    console.error(`FAIL - ${e.message}`);
  }
}

check('zero tools', buildToolSummary([]), 'Used 0 tools');

check(
  'single tool without caption uses generic phrase',
  buildToolSummary([{ tool: 'read_file' }]),
  'Used 1 tool: read a file'
);

check(
  'single tool WITH persisted caption uses the caption verbatim',
  buildToolSummary([{ tool: 'read_file', caption: 'Checking the renderer for sprite drawing' }]),
  'Checking the renderer for sprite drawing'
);

check(
  'multi-tool dedupes repeated phrases',
  buildToolSummary([{ tool: 'bash' }, { tool: 'bash' }, { tool: 'bash' }]),
  'Used 3 tools: ran a command'
);

check(
  'multi-tool keeps first 3 distinct phrases + and N more',
  buildToolSummary([
    { tool: 'read_file' }, { tool: 'grep' }, { tool: 'bash' },
    { tool: 'write_file' }, { tool: 'web_search' },
  ]),
  'Used 5 tools: read a file, searched code, ran a command, and 2 more'
);

check(
  'exactly 3 distinct phrases: no "and N more" suffix',
  buildToolSummary([{ tool: 'read_file' }, { tool: 'grep' }, { tool: 'bash' }]),
  'Used 3 tools: read a file, searched code, ran a command'
);

check(
  'get_workspace omitted from a multi-tool phrase list',
  buildToolSummary([{ tool: 'get_workspace' }, { tool: 'read_file' }]),
  'Used 2 tools: read a file'
);

check(
  'get_workspace ALONE still gets its own phrase (not omitted when alone)',
  buildToolSummary([{ tool: 'get_workspace' }]),
  'Used 1 tool: ' + 'get_workspace'.replace(/_/g, ' ')
);

check(
  'unmapped tool falls back to name with underscores as spaces',
  buildToolSummary([{ tool: 'some_future_tool' }, { tool: 'another_new_one' }]),
  'Used 2 tools: some future tool, another new one'
);

check(
  'ls and glob both map to "listed files" and dedupe',
  buildToolSummary([{ tool: 'ls' }, { tool: 'glob' }]),
  'Used 2 tools: listed files'
);

check(
  'spawn_agent maps to "ran a sub-agent"',
  buildToolSummary([{ tool: 'spawn_agent' }, { tool: 'bash' }]),
  'Used 2 tools: ran a sub-agent, ran a command'
);

check(
  'edit_file and write_file map to distinct phrases',
  buildToolSummary([{ tool: 'edit_file' }, { tool: 'write_file' }]),
  'Used 2 tools: edited a file, wrote a file'
);

check(
  'null/undefined events in the list are filtered out',
  buildToolSummary([{ tool: 'bash' }, null, undefined, { tool: 'grep' }]),
  'Used 2 tools: ran a command, searched code'
);

if (failures > 0) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log('\nall ok');
