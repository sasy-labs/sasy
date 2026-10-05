import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { variants, definitionsFor, lastStep, actionStep, frame, labelsFor, graphMarkup, backwardSlice, staticGraphSvg, mountActionGraph, mountVariantTabs } from '../src/components/action-graph.mjs';

test('both variants propagate file labels through summarizer calls', () => {
  for (const {id} of variants) {
    const complete = frame(lastStep, id);
    assert.deepEqual(complete.nodes.filter(node => node.source).map(node => [node.id, node.source]), [['report','S'],['note','U']]);
    assert.deepEqual(complete.nodes.find(node => node.id === 'report-summary').labels, ['S']);
    assert.deepEqual(complete.nodes.find(node => node.id === 'note-summary').labels, ['U']);
    for (let step = 0; step <= lastStep; step++) {
      const state = frame(step,id);
      assert.ok(state.message);
      assert.ok(!state.slice.has(state.target.id));
      assert.ok(state.edges.every(edge => state.nodes.some(node => node.id === edge.from) && state.nodes.some(node => node.id === edge.to)));
      assert.ok(state.nodes.every(node => !backwardSlice(state.nodes,node.id).has(node.id)));
      if (state.phase !== 'decision') assert.equal(state.decision,'pending');
    }
  }
});

test('each action check highlights its ancestors before its decision', () => {
  for (const {id} of variants) for (const node of definitionsFor(id).filter(node => node.kind === 'action')) {
    const step = actionStep(node.id), check = frame(step,id), decision = frame(step+1,id);
    assert.equal(check.phase,'check'); assert.equal(check.decision,'pending');
    assert.equal(check.target.id,node.id); assert.equal(decision.target.id,node.id);
    assert.deepEqual(check.slice,backwardSlice(check.nodes,node.id));
    assert.ok(check.slice.has('task')); assert.ok(check.slice.has('spawn'));
    assert.equal(decision.phase,'decision');
    assert.equal(decision.decision,node.id === 'send' && id === 'blocked' ? 'deny' : 'allow');
    assert.match(graphMarkup(check),/neutral in-slice/);
  }
});

test('arrival before the send changes its ancestry and decision', () => {
  const allowed=frame(lastStep,'allowed'), blocked=frame(lastStep,'blocked');
  assert.deepEqual(allowed.target.labels,['S']); assert.equal(allowed.decision,'allow');
  assert.deepEqual(blocked.target.labels,['U','S']); assert.equal(blocked.decision,'deny');
  for (const state of [allowed,blocked]) {
    assert.ok(state.nodes.some(node=>node.id==='note-summary' && node.labels.includes('U')));
    assert.ok(state.slice.has('report-summary'));
    assert.equal(state.nodes.filter(node=>node.operation==='send').length,1);
    assert.ok(!state.nodes.some(node=>node.inputs.includes('send'))); // no claimed execution after authorization
  }
  for (const id of ['note-summary','note','read-note','note-plan']) {
    assert.ok(!allowed.slice.has(id)); assert.ok(blocked.slice.has(id));
  }
  const arrived = definitionsFor('allowed').map(node=>node.id==='draft' ? {...node,inputs:[...node.inputs,'note-summary']} : node);
  assert.deepEqual(labelsFor(arrived,'send'),['U','S']);
});

test('static comparison matches the renderer and includes accessible labels', () => {
  const poster=readFileSync(new URL('../public/diagrams/information-flow.svg',import.meta.url),'utf8');
  assert.equal(poster,staticGraphSvg());
  assert.match(poster,/not a captured execution trace/);
  assert.match(poster,/ALLOW · tool action/); assert.match(poster,/DENY · tool action/);
  assert.doesNotMatch(poster,/data-inspect=/);
  for (const {id} of variants) {
    const markup=graphMarkup(frame(lastStep,id),true);
    assert.match(markup,/<circle class="node-shape"/);
    assert.match(markup,/<rect class="node-shape"/);
    assert.match(markup,/data-inspect="send"/);
    assert.match(markup,/aria-label="Inspect/);
  }
});

test('playback, inspection, keyboard control and reduced motion remain usable', () => {
  const element = (dataset = {}) => ({dataset, attrs:{}, events:{}, textContent:'',
    addEventListener(name, callback) { this.events[name] = callback; },
    setAttribute(name, value) { this.attrs[name] = value; }, focus() { this.focused = true; }});
  const selectors = Object.fromEntries(['marks','status','focus','replay','previous','next','progress','controls'].map(name => [`[data-${name}]`,element()]));
  const root=element();
  root.querySelector=selector=>{assert.ok(selectors[selector],selector); return selectors[selector];};
  const motion=element();motion.matches=false;
  let tick;
  const oldWindow=globalThis.window, oldDocument=globalThis.document;
  globalThis.window={matchMedia:()=>motion,clearInterval:()=>{tick=undefined;},setInterval:callback=>{tick=callback;return 1;}};
  globalThis.document={...element(),hidden:false};
  try {
    mountActionGraph(root);
    assert.equal(root.dataset.decision,'deny');
    selectors['[data-marks]'].events.click({target:{closest:()=>({dataset:{inspect:'send'}})}});
    assert.equal(root.dataset.phase,'check');
    selectors['[data-next]'].events.click();
    assert.equal(root.dataset.decision,'deny');
    const replay=selectors['[data-replay]'], next=selectors['[data-next]'];
    replay.events.click();assert.equal(replay.textContent,'Pause');
    tick();root.events['diagram-hidden']();assert.equal(tick,undefined);assert.equal(replay.textContent,'Replay');
    while(!next.disabled) next.events.click();
    assert.equal(root.dataset.decision,'deny');
    selectors['[data-previous]'].events.click();assert.equal(root.dataset.phase,'check');
    let prevented=false;
    selectors['[data-marks]'].events.keydown({key:'Enter',preventDefault(){prevented=true;},target:{closest:()=>({dataset:{inspect:'read-note'}})}});
    assert.ok(prevented);assert.ok(next.focused);assert.equal(root.dataset.phase,'check');
    next.events.click();assert.equal(root.dataset.decision,'allow');
    replay.events.click();while(tick) tick();assert.equal(replay.textContent,'Replay');
    motion.matches=true;replay.events.click();assert.equal(tick,undefined);assert.equal(selectors['[data-previous]'].disabled,true);
    next.events.click();assert.equal(root.dataset.phase,'grow');
    motion.matches=false;replay.events.click();motion.matches=true;motion.events.change();assert.equal(tick,undefined);
    motion.matches=false;replay.events.click();globalThis.document.hidden=true;globalThis.document.events.visibilitychange();assert.equal(tick,undefined);
    assert.equal(selectors['[data-controls]'].hidden,false);
    root.dataset.variant='allowed'; mountActionGraph(root); assert.equal(root.dataset.decision,'allow');
  } finally {globalThis.window=oldWindow;globalThis.document=oldDocument;}
});

test('view tabs show one variant, support keyboard navigation, and stop hidden playback', () => {
  const element = dataset => ({dataset,attrs:{},events:{},
    setAttribute(name,value){this.attrs[name]=value;},
    addEventListener(name,callback){this.events[name]=callback;},
    focus(){this.focused=true;},
    dispatchEvent(event){assert.equal(event.type,'diagram-hidden');this.stopped=true;}});
  const tabs=variants.map(({id})=>element({view:id}));
  const panels=variants.map(({id})=>element({variant:id}));
  const tablist={hidden:true};
  mountVariantTabs({querySelector:()=>tablist,querySelectorAll:selector=>selector==='[data-view]'?tabs:panels});
  assert.equal(tablist.hidden,false);assert.deepEqual(panels.map(p=>p.hidden),[false,true]);
  tabs[1].events.click();
  assert.deepEqual(panels.map(p=>p.hidden),[true,false]);assert.equal(panels[0].stopped,true);
  assert.deepEqual(tabs.map(t=>t.attrs['aria-selected']),['false','true']);
  let prevented=false;
  tabs[1].events.keydown({key:'ArrowRight',preventDefault(){prevented=true;}});
  assert.ok(prevented);assert.equal(tabs[0].focused,true);
  assert.deepEqual(tabs.map(t=>t.tabIndex),[0,-1]);
  assert.deepEqual(panels.map(p=>p.hidden),[false,true]);
});
