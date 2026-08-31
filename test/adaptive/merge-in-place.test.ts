/**
 * Merge-in-place at the ceiling level (`maxMergeLevel`) — reachability.
 *
 * A ceiling merge consolidates N same-level entries into ONE broader entry at
 * that same level, pointing each source's `parentId` at it. Every ancestor walk
 * therefore has to keep climbing once it reaches the target level: stopping at
 * the first node whose level matches lands on a merged-away child, and since
 * sibling dedup keys on the returned id, each old source keeps emitting its own
 * recall pair. The merge spends an LLM call, the sources leave the unmerged set
 * so the trigger converges, and the rendered middle never shrinks.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import type { PickerInputs } from '../../src/adaptive/picker.js';
import { accountFrontier } from '../../src/adaptive/picker.js';
import { SummaryTree } from '../../src/adaptive/summary-tree.js';
import { renderLayout } from '../../src/adaptive/render-offsets.js';
import { topmostAtSameLevel } from '../../src/types/strategy.js';
import type { ChunkId } from '../../src/adaptive/folding-strategy.js';
import { MockChronicle } from './harness.js';

function inputsOf(ch: MockChronicle): PickerInputs {
  return {
    chunks: ch.chunks,
    summaries: ch.summaries,
    recallPairTokens: ch.recallPairTokens,
    headTokens: 0,
    tailTokens: 0,
    headChunkIds: new Set(),
    tailChunkIds: new Set(),
  };
}

/** 16 chunks → 8 L1s → 4 L2s → 2 L3s. Returns the two L3 ids. */
function pyramid(): { ch: MockChronicle; l3s: string[] } {
  const ch = new MockChronicle({ mergeThreshold: 2, recallPairTokens: 200 });
  for (let i = 0; i < 16; i++) {
    ch.addChunk({ id: `c-${i.toString().padStart(2, '0')}`, rawTokens: 1000 });
  }
  const l1s: string[] = [];
  for (let i = 0; i < 16; i += 2) {
    l1s.push(ch.produceL1([ch.chunks[i].id, ch.chunks[i + 1].id]).id);
  }
  const l2s: string[] = [];
  for (let i = 0; i < l1s.length; i += 2) {
    l2s.push(ch.produceUpper(2, [l1s[i], l1s[i + 1]]).id);
  }
  const l3s: string[] = [];
  for (let i = 0; i < l2s.length; i += 2) {
    l3s.push(ch.produceUpper(3, [l2s[i], l2s[i + 1]]).id);
  }
  return { ch, l3s };
}

function allAt(ch: MockChronicle, level: number): Map<ChunkId, number> {
  return new Map(ch.chunks.map((c): [ChunkId, number] => [c.id, level]));
}

test('merge-in-place: ancestorAt returns the broad L3, not the merged-away child', () => {
  const { ch, l3s } = pyramid();
  const broad = ch.consolidateInPlace(3, l3s).id;
  const tree = new SummaryTree(inputsOf(ch));

  // Both halves of the chronicle now resolve to the one consolidation…
  assert.equal(tree.ancestorAt('c-00', 3)?.id, broad);
  assert.equal(tree.ancestorAt('c-15', 3)?.id, broad);
  // …while the levels below are untouched.
  assert.equal(tree.ancestorAt('c-00', 1)?.id, 'L1-0');
  assert.equal(tree.ancestorAt('c-00', 2)?.id, 'L2-0');
  assert.equal(tree.maxLevel('c-00'), 3);
  assert.equal(tree.leavesUnder(broad).length, 16, 'consolidation covers every leaf');
});

test('merge-in-place: the consolidated group renders ONE recall pair', () => {
  const { ch, l3s } = pyramid();
  const broad = ch.consolidateInPlace(3, l3s).id;
  const inputs = inputsOf(ch);
  const tree = new SummaryTree(inputs);
  const frontier = allAt(ch, 3);

  const layout = renderLayout(inputs, tree, frontier);
  const recalls = layout.units.filter((u) => u.kind === 'recall');
  assert.deepEqual(recalls.map((u) => u.key), [broad], 'one pair for the whole consolidation');
  assert.equal(layout.totalTokens, 200);

  // The planner has to agree with the renderer, or the fold it budgets for
  // isn't the fold it gets.
  const accounted = accountFrontier(inputs, frontier);
  assert.equal(accounted.tokens, layout.totalTokens);
  assert.deepEqual(accounted.unrealizable, []);
});

test('merge-in-place: without a consolidation each L3 still renders its own pair', () => {
  const { ch, l3s } = pyramid();
  const inputs = inputsOf(ch);
  const tree = new SummaryTree(inputs);

  assert.equal(tree.ancestorAt('c-00', 3)?.id, l3s[0]);
  assert.equal(tree.ancestorAt('c-15', 3)?.id, l3s[1]);

  const layout = renderLayout(inputs, tree, allAt(ch, 3));
  assert.deepEqual(layout.units.filter((u) => u.kind === 'recall').map((u) => u.key), l3s);
  assert.equal(layout.totalTokens, 400);
});

test('merge-in-place: a same-level parent cycle terminates', () => {
  const { ch, l3s } = pyramid();
  const [a, b] = l3s;
  ch.summaries.get(a)!.parentId = b;
  ch.summaries.get(b)!.parentId = a;
  const tree = new SummaryTree(inputsOf(ch));

  // Corrupted store: no topmost exists. Any of the cycle's members is an
  // acceptable answer — hanging the compile is not.
  assert.ok(l3s.includes(tree.ancestorAt('c-00', 3)!.id));
  assert.ok(l3s.includes(tree.ancestorAt('c-15', 3)!.id));
});

test('topmostAtSameLevel: climbs same-level links, stops at a level change or a cycle', () => {
  type N = { id: string; level: number; parentId?: string };
  const nodes = new Map<string, N>();
  const add = (id: string, level: number, parentId?: string): N => {
    const n = { id, level, parentId };
    nodes.set(id, n);
    return n;
  };
  const top = (n: N): N => topmostAtSameLevel(n, (x) => x.parentId, (id) => nodes.get(id));

  // Chain of same-level consolidations: a → b → c, then c is merged upward.
  add('c', 3, 'up');
  add('b', 3, 'c');
  add('a', 3, 'b');
  add('up', 4);
  assert.equal(top(nodes.get('a')!).id, 'c', 'stops below the level change');
  assert.equal(top(nodes.get('up')!).id, 'up', 'a node with no parent is its own topmost');

  // A dangling parent pointer is not a reason to fail — return what we have.
  add('orphan', 3, 'missing');
  assert.equal(top(nodes.get('orphan')!).id, 'orphan');

  // Cycle: terminates rather than spinning.
  add('x', 3, 'y');
  add('y', 3, 'x');
  assert.ok(['x', 'y'].includes(top(nodes.get('x')!).id));
});
