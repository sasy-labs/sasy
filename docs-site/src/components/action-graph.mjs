// An illustrative shared session, not a captured trace of the two-run demo.
// Circles represent an LLM call and its output. Redundant history edges are omitted.
export const variants = [
  { id: 'allowed', title: 'Note has not arrived', outcome: 'ALLOW · Send to reviewer' },
  { id: 'blocked', title: 'Note arrives before the send', outcome: 'DENY · Redirected send' },
];

export function definitionsFor(variant = 'blocked') {
  const blocked = variant === 'blocked';
  // Successive rows leave 24 units between node borders.
  return [
    { id: 'task', kind: 'input', label: 'User task', detail: 'Review and send', x: 300, y: 44, inputs: [] },
    { id: 'spawn', kind: 'llm', label: 'Coordinator', detail: 'Spawn subagents', x: 300, y: 156, inputs: ['task'] },
    { id: 'report-plan', kind: 'llm', label: 'Summarizer', detail: 'Review report', x: 140, y: 300, inputs: ['spawn'] },
    { id: 'note-plan', kind: 'llm', label: 'Summarizer', detail: 'Review note', x: 460, y: 300, inputs: ['spawn'] },
    { id: 'read-report', kind: 'action', label: 'Read report', operation: 'read', x: 140, y: 412, inputs: ['report-plan'] },
    { id: 'report', kind: 'result', label: 'Sensitive report', detail: 'File contents', source: 'S', x: 140, y: 492, inputs: ['read-report'] },
    { id: 'report-summary', kind: 'llm', label: 'Summarizer', detail: 'Report summary', x: 140, y: 604, inputs: ['report'] },
    { id: 'read-note', kind: 'action', label: 'Read note', operation: 'read', x: 460, y: 412, inputs: ['note-plan'] },
    { id: 'note', kind: 'result', label: 'Untrusted note', detail: 'File contents', source: 'U', x: 460, y: 492, inputs: ['read-note'] },
    { id: 'note-summary', kind: 'llm', label: 'Summarizer', detail: 'Note summary', x: 460, y: 604, inputs: ['note'] },
    { id: 'draft', kind: 'llm', label: 'Coordinator', detail: blocked ? 'Follow the note' : 'Draft email', x: 300, y: 748,
      inputs: blocked ? ['spawn', 'report-summary', 'note-summary'] : ['spawn', 'report-summary'] },
    { id: 'send', kind: 'action', label: blocked ? 'Redirected send' : 'Send to reviewer', operation: 'send', x: 300, y: 860, inputs: ['draft'] },
  ];
}
export const definitions = definitionsFor();
const stagesFor = nodes => nodes.flatMap(node => node.kind === 'action'
  ? [{ id: node.id, phase: 'check' }, { id: node.id, phase: 'decision' }]
  : [{ id: node.id, phase: 'grow' }]);
export const stages = stagesFor(definitions);
export const lastStep = stages.length - 1;
export const actionStep = id => stages.findIndex(stage => stage.id === id && stage.phase === 'check');

export function backwardSlice(nodes, target) {
  const ancestors = new Set(), byId = new Map(nodes.map(node => [node.id, node]));
  const visit = id => {
    for (const input of byId.get(id)?.inputs ?? []) {
      if (input !== target && byId.has(input) && !ancestors.has(input)) {
        ancestors.add(input); visit(input);
      }
    }
  };
  visit(target);
  return ancestors;
}

export function labelsFor(nodes, id) {
  const related = backwardSlice(nodes, id); related.add(id);
  return ['U', 'S'].filter(label => nodes.some(node => related.has(node.id) && node.source === label));
}

export function decide(node) {
  return node.operation === 'send' && node.labels.includes('U') && node.labels.includes('S') ? 'deny' : 'allow';
}

const descriptions = {
  task: 'The user asks for a review and an email to the reviewer.',
  spawn: 'The coordinator spawns two summarizers, one for each file.',
  'report-plan': 'One summarizer chooses to read the report.',
  'note-plan': 'The other summarizer chooses to read the note.',
  report: 'Reading the sensitive report introduces S.',
  note: 'Reading the untrusted note introduces U. The note asks to redirect the email.',
  'report-summary': 'The report summary inherits S from the file contents.',
  'note-summary': 'The note summary inherits U from the file contents.',
};

export function frame(step = lastStep, variant = 'blocked') {
  const definitions = definitionsFor(variant), stages = stagesFor(definitions);
  const current = Math.max(0, Math.min(lastStep, Math.floor(step)));
  const stage = stages[current];
  const index = definitions.findIndex(node => node.id === stage.id);
  const nodes = definitions.slice(0, index + 1).map(node => ({ ...node, labels: labelsFor(definitions, node.id) }));
  const target = nodes.at(-1), slice = backwardSlice(nodes, target.id);
  const decision = stage.phase === 'decision' ? decide(target) : 'pending';
  let message = descriptions[target.id];
  if (target.id === 'draft') message = variant === 'blocked'
    ? 'Both summaries reach the coordinator. U and S influence the redirected email.'
    : 'Only the report summary reaches the coordinator. The note has not arrived.';
  if (stage.phase === 'check') message = target.operation === 'read'
    ? 'Check the read’s ancestors: neither sensitive nor untrusted content has entered this branch.'
    : variant === 'blocked'
      ? 'The send’s ancestors include both summaries, carrying U and S.'
      : 'The send’s ancestors include S; the U branch is outside its history.';
  if (stage.phase === 'decision') message = target.operation === 'read'
    ? 'ALLOW · The summarizer may read the file.'
    : decision === 'deny'
      ? 'DENY · Both U and S influenced this send. The email handler does not run.'
      : 'ALLOW · Only S influenced this send. The email may go to the intended reviewer.';
  const decisions = Object.fromEntries(stages.slice(0, current + 1).filter(s => s.phase === 'decision')
    .map(s => [s.id, decide(nodes.find(node => node.id === s.id))]));
  return { nodes, target, slice, current, last: lastStep, variant, phase: stage.phase, decision, decisions, message,
    edges: nodes.flatMap(node => node.inputs.map(from => ({ from, to: node.id }))) };
}

const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const dimensions = node => node.kind === 'llm' ? { rx: 60, ry: 60 } : { rx: 100, ry: 28 };
function shape(node, expand = 0, className = 'node-shape') {
  const { rx, ry } = dimensions(node);
  return node.kind === 'llm'
    ? `<circle class="${className}" cx="${node.x}" cy="${node.y}" r="${rx + expand}"/>`
    : `<rect class="${className}" x="${node.x - rx - expand}" y="${node.y - ry - expand}" width="${(rx + expand) * 2}" height="${(ry + expand) * 2}" rx="${node.kind === 'action' ? 7 : 24}"/>`;
}

export function graphMarkup(state, interactive = false) {
  const byId = new Map(state.nodes.map(node => [node.id, node]));
  const edges = state.edges.map(edge => {
    const from = byId.get(edge.from), to = byId.get(edge.to);
    const traced = state.slice.has(edge.from) && (state.slice.has(edge.to) || edge.to === state.target.id);
    let d, tip, direction = 'down';
    if (from.id === 'spawn' && ['report-plan', 'note-plan'].includes(to.id)) {
      const side = to.x < from.x ? -1 : 1;
      const startX = from.x + side * (dimensions(from).rx + 2);
      tip = { x: to.x, y: to.y - dimensions(to).ry - 4 };
      d = `M${startX} ${from.y} C${to.x} ${from.y} ${to.x} ${tip.y - 40} ${tip.x} ${tip.y}`;
    } else if (['report-summary', 'note-summary'].includes(from.id) && to.id === 'draft') {
      const fromLeft = from.x < to.x;
      tip = { x: to.x + (fromLeft ? -1 : 1) * (dimensions(to).rx + 4), y: to.y };
      direction = fromLeft ? 'right' : 'left';
      const startY = from.y + dimensions(from).ry + 2;
      d = `M${from.x} ${startY} C${from.x} ${to.y} ${from.x} ${to.y} ${tip.x} ${tip.y}`;
    } else {
      const start = { x: from.x, y: from.y + dimensions(from).ry + 2 };
      tip = { x: to.x, y: to.y - dimensions(to).ry - 4 };
      d = `M${start.x} ${start.y} V${tip.y}`;
    }
    // Arrowheads follow the final path direction and remain local to each edge.
    const end = direction === 'right'
      ? `M${tip.x - 7} ${tip.y - 5} L${tip.x} ${tip.y} L${tip.x - 7} ${tip.y + 5}`
      : direction === 'left'
        ? `M${tip.x + 7} ${tip.y - 5} L${tip.x} ${tip.y} L${tip.x + 7} ${tip.y + 5}`
        : `M${tip.x - 5} ${tip.y - 7} L${tip.x} ${tip.y} L${tip.x + 5} ${tip.y - 7}`;
    return `<g class="edge ${traced ? 'traced' : ''}" data-from="${edge.from}" data-to="${edge.to}"><path d="${d}"/><path d="${end}"/></g>`;
  }).join('');
  const nodes = state.nodes.map(node => {
    const target = node.id === state.target.id, inSlice = state.slice.has(node.id);
    const taint = node.labels.length === 2 ? 'mixed' : node.labels[0] === 'U' ? 'untrusted' : node.labels[0] === 'S' ? 'sensitive' : 'neutral';
    const outcome = state.decisions[node.id];
    const detail = node.kind === 'action' ? outcome ? `${outcome === 'allow' ? 'ALLOW' : 'DENY'} · tool action` : 'CHECK · tool action' : node.detail;
    const labelText = node.labels.length ? node.labels.join(' + ') : 'No U / S';
    const control = interactive && node.kind === 'action'
      ? `role="button" tabindex="0" data-inspect="${node.id}" aria-label="Inspect ${escape(node.label)} backward slice"` : '';
    const title = `${node.label}. ${node.kind === 'llm' ? 'LLM call and output. ' : ''}${detail}. Labels: ${labelText}.${inSlice ? ' In the backward slice.' : ''}`;
    const text = node.kind === 'llm'
      ? `<text class="node-label" x="${node.x}" y="${node.y - 12}">${node.label}</text><text class="node-detail" x="${node.x}" y="${node.y + 5}">LLM call</text><text class="llm-detail" x="${node.x}" y="${node.y + 22}">${node.detail}</text>`
      : `<text class="node-label" x="${node.x}" y="${node.y - 4}">${node.label}</text><text class="node-detail ${outcome ?? ''}" x="${node.x}" y="${node.y + 15}">${detail}</text>`;
    const badges = node.labels.map((label, i) => {
      const offset = node.labels.length - 1 - i;
      // Attach circular-node badges to the upper-right arc, not the bounding box.
      const angle = Math.PI / 4 + offset * Math.PI / 6;
      const cx = node.kind === 'llm'
        ? node.x + Math.round(dimensions(node).rx * Math.cos(angle) * 10) / 10
        : node.x + dimensions(node).rx - 8 - offset * 27;
      const cy = node.kind === 'llm'
        ? node.y - Math.round(dimensions(node).ry * Math.sin(angle) * 10) / 10
        : node.y - dimensions(node).ry - 3;
      return `<g class="badge ${label === 'U' ? 'untrusted' : 'sensitive'} ${node.source === label ? 'source' : ''}"><circle cx="${cx}" cy="${cy}" r="11"/><text x="${cx}" y="${cy + 4}">${label}</text></g>`;
    }).join('');
    return `<g class="graph-node ${taint} ${inSlice ? 'in-slice' : ''} ${target ? 'target' : 'outside'}" data-node="${node.id}" ${control}><title>${escape(title)}</title>${inSlice || target ? shape(node, 6, 'slice-halo') : ''}${shape(node)}${taint === 'mixed' ? shape(node, -4, 'second-border') : ''}${text}${badges}</g>`;
  }).join('');
  const annotations = state.variant === 'allowed' && byId.has('draft')
    ? '<text class="handoff" x="460" y="696" text-anchor="middle">Not yet delivered</text>' : '';
  return `${edges}${annotations}${nodes}`;
}

const lightPalette = { ink:'#243449', muted:'#59687b', paper:'#fff', line:'#c7d0dc', edge:'#a3adba', u:'#315bb5', ubg:'#edf2ff', s:'#975b06', sbg:'#fff4da', mix:'#f4eef8', trace:'#7854b1', tracebg:'#f4f0fa', allow:'#17613c', deny:'#b02b35' };
export const graphStyles = `
.flow-graph { ${Object.entries(lightPalette).map(([key, value]) => `--${key}:${value};`).join(' ')} font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
[data-theme='dark'] .flow-graph { --ink:#e7edf5; --muted:#adb9c9; --paper:#151b25; --line:#596576; --edge:#637084; --u:#9dbfff; --ubg:#1c3053; --s:#f7ca7e; --sbg:#3b3020; --mix:#302d46; --trace:#c5abf0; --tracebg:#29243b; --allow:#8de0b1; --deny:#ffa5a8; }
.flow-graph .edge { fill:none; stroke:var(--edge); stroke-width:1.5; }
.flow-graph .edge.traced { stroke:var(--trace); stroke-width:2.4; }
.flow-graph .node-shape { fill:var(--paper); stroke:var(--line); stroke-width:1.8; }
.flow-graph .untrusted .node-shape { fill:var(--ubg); stroke:var(--u); stroke-width:2.5; }
.flow-graph .sensitive .node-shape { fill:var(--sbg); stroke:var(--s); stroke-width:2.5; }
.flow-graph .mixed .node-shape { fill:var(--mix); stroke:var(--u); stroke-width:2.5; }
.flow-graph .second-border { fill:none; stroke:var(--s); stroke-width:2; }
.flow-graph .slice-halo { fill:none; stroke:var(--trace); stroke-width:1.5; stroke-dasharray:4 4; }
.flow-graph .neutral.in-slice .node-shape { fill:var(--tracebg); }
.flow-graph .target .slice-halo { stroke-width:2.5; stroke-dasharray:5 3; }
.flow-graph .outside:not(.in-slice) .node-shape { stroke-opacity:.5; }
.flow-graph .node-label { font-size:15px; font-weight:620; fill:var(--ink); text-anchor:middle; }
.flow-graph .node-detail { font-size:12px; fill:var(--muted); text-anchor:middle; }
.flow-graph .llm-detail { font-size:11px; fill:var(--muted); text-anchor:middle; }
.flow-graph .node-detail.allow { fill:var(--allow); font-weight:700; }
.flow-graph .node-detail.deny { fill:var(--deny); font-weight:700; }
.flow-graph .badge circle { fill:var(--paper); stroke-width:1.8; }
.flow-graph .badge.untrusted circle { stroke:var(--u); }
.flow-graph .badge.sensitive circle { stroke:var(--s); }
.flow-graph .badge.source.untrusted circle { fill:var(--ubg); }
.flow-graph .badge.source.sensitive circle { fill:var(--sbg); }
.flow-graph .badge text { font-size:12px; font-weight:700; text-anchor:middle; fill:var(--ink); }
.flow-graph .handoff { font-size:12px; fill:var(--muted); }
.flow-graph [data-inspect] { cursor:pointer; }
.flow-graph [data-inspect]:focus-visible { outline:none; }
.flow-graph [data-inspect]:focus-visible .node-shape { stroke:var(--trace); stroke-width:4; }
@media (prefers-reduced-motion:no-preference) { .flow-graph .target { animation:graph-arrive .3s ease-out; } @keyframes graph-arrive { from { opacity:.4; } to { opacity:1; } } }
`;

export function staticGraphSvg() {
  const panels = variants.map((variant, i) => `<g transform="translate(${10 + i * 620} 0)">
<text x="300" y="30" class="node-label">${variant.title}</text>
<text x="300" y="55" class="node-detail ${variant.id === 'allowed' ? 'allow' : 'deny'}">${variant.outcome}</text>
<g transform="translate(0 68)">${graphMarkup(frame(lastStep, variant.id))}</g></g>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" class="flow-graph" width="1240" height="1050" viewBox="0 0 1240 1050" role="img" aria-labelledby="title desc">
<title id="title">Arrival order changes the send’s ancestors</title>
<desc id="desc">Two illustrative variants, not a captured execution trace. A coordinator spawns two summarizers to review a sensitive report and an untrusted note. Left: the note has not reached the coordinator, so the send depends only on sensitive data and is allowed. Right: the note arrives before the send, which is redirected and denied because both labels reach it. Circles are LLM calls; rectangles are actions; pills are inputs and results. Dashed outlines mark action ancestors.</desc>
<style>${graphStyles.replace(/var\(--([a-z]+)\)/g, (_, key) => lightPalette[key])}</style><rect width="1240" height="1050" rx="12" fill="#fff"/>
${panels}
<g transform="translate(255 999)" class="badge untrusted"><circle r="13"/><text y="4">U</text></g><circle cx="289" cy="999" r="13" fill="#edf2ff" stroke="#315bb5" stroke-width="2.5"/><text x="312" y="1004" class="handoff">Untrusted</text>
<g transform="translate(460 999)" class="badge sensitive"><circle r="13"/><text y="4">S</text></g><circle cx="494" cy="999" r="13" fill="#fff4da" stroke="#975b06" stroke-width="2.5"/><text x="517" y="1004" class="handoff">Sensitive</text>
<circle cx="672" cy="999" r="14" fill="none" stroke="#7854b1" stroke-width="1.5" stroke-dasharray="4 4"/><text x="696" y="1004" class="handoff">Ancestors</text>
<text x="620" y="1035" text-anchor="middle" class="handoff">Circles: LLM calls · Rectangles: tool actions · Pills: inputs / results</text>
</svg>\n`;
}

export function mountActionGraph(root) {
  const marks = root.querySelector('[data-marks]'), status = root.querySelector('[data-status]');
  const replay = root.querySelector('[data-replay]'), previous = root.querySelector('[data-previous]'), next = root.querySelector('[data-next]');
  const variant = root.dataset.variant ?? 'blocked';
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let step = lastStep, timer = null;
  function stop() { if (timer !== null) window.clearInterval(timer); timer = null; replay.textContent = 'Replay'; }
  function draw() {
    const state = frame(step, variant);
    marks.innerHTML = graphMarkup(state, true);
    root.dataset.decision = state.decision; root.dataset.phase = state.phase;
    status.textContent = state.message;
    root.querySelector('[data-progress]').textContent = `Step ${step + 1} of ${lastStep + 1}`;
    root.querySelector('[data-focus]').textContent = `${state.phase === 'check' ? 'Check action' : state.phase === 'decision' ? 'Policy decision' : 'Record event'} · ${state.target.label}`;
    previous.disabled = step === 0; next.disabled = step === lastStep;
  }
  const select = id => { const index = actionStep(id); if (index < 0) return; stop(); step = index; draw(); };
  // Delegate events because each frame replaces the SVG nodes. Move focus to a
  // stable control after keyboard selection rather than leaving it on a removed node.
  marks.addEventListener('click', event => { const node = event.target.closest?.('[data-inspect]'); if (node) select(node.dataset.inspect); });
  marks.addEventListener('keydown', event => {
    const node = event.target.closest?.('[data-inspect]');
    if (node && ['Enter', ' '].includes(event.key)) { event.preventDefault(); select(node.dataset.inspect); next.focus(); }
  });
  replay.addEventListener('click', () => {
    if (timer !== null) { stop(); return; }
    step = 0; draw(); if (motion.matches) return;
    replay.textContent = 'Pause';
    timer = window.setInterval(() => { step++; if (step >= lastStep) stop(); draw(); }, 1800);
  });
  previous.addEventListener('click', () => { stop(); step = Math.max(0, step - 1); draw(); });
  next.addEventListener('click', () => { stop(); step = Math.min(lastStep, step + 1); draw(); });
  root.addEventListener('diagram-hidden', stop);
  motion.addEventListener('change', () => { if (motion.matches) stop(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  root.querySelector('[data-controls]').hidden = false;
  draw();
}

export function mountVariantTabs(root) {
  const tabs = [...root.querySelectorAll('[data-view]')];
  const panels = [...root.querySelectorAll('[data-action-graph]')];
  function select(index, focus = false) {
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
    });
    panels.forEach(panel => {
      panel.hidden = panel.dataset.variant !== tabs[index].dataset.view;
      if (panel.hidden) panel.dispatchEvent(new Event('diagram-hidden'));
    });
    if (focus) tabs[index].focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(index));
    tab.addEventListener('keydown', event => {
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length
        : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (next !== null) { event.preventDefault(); select(next, true); }
    });
  });
  select(0);
  root.querySelector('[data-view-tabs]').hidden = false;
}
