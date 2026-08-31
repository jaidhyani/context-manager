/**
 * `maxMergeLevel` ceiling on the picker-demand merge path.
 *
 * `checkMergeThresholdRecursive` clamps its own enqueues at the ceiling, but
 * `enqueueMergeForRange` — the produce-op path the picker drives — took the
 * requested level at face value, so a demand for L_{ceiling+1} would mint a
 * level the hierarchy is not allowed to grow. Above the ceiling the request is
 * a merge-in-place instead: same-level sources, same-level target.
 */

import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, rmSync } from 'node:fs';

import { ContextManager, AutobiographicalStrategy } from '../src/index.js';
import type { ContentBlock } from '@animalabs/membrane';
import type { SummaryEntry } from '../src/types/index.js';

const BASE = './test-merge-ceiling';
let sequence = 0;
const paths: string[] = [];

after(() => {
  for (const path of paths) {
    if (existsSync(path)) rmSync(path, { recursive: true, force: true });
  }
});

class Probe extends AutobiographicalStrategy {
  seed(entry: SummaryEntry): void {
    this.pushSummary(entry);
  }
  demand(level: number, first: string, last: string): void {
    this.enqueueMergeForRange(level, first, last);
  }
  queueView(): Array<{ level: number; sourceIds: string[] }> {
    return this.mergeQueue.map((m) => ({ level: m.level, sourceIds: [...m.sourceIds] }));
  }
}

const t = (s: string): ContentBlock => ({ type: 'text', text: s });

/**
 * Real store, eight raw messages, and `count` unmerged summaries at `level`
 * spread across them. Chunks never close (targetChunkTokens is huge), so the
 * merge path is the only thing under test.
 */
async function fixture(level: number, count: number, maxMergeLevel?: number): Promise<{
  strategy: Probe;
  ids: string[];
  sourceIds: string[];
}> {
  const path = `${BASE}-${sequence++}`;
  paths.push(path);
  const strategy = new Probe({
    compressionModel: 'same-model',
    hierarchical: true,
    adaptiveResolution: true,
    targetChunkTokens: 100_000,
    recentWindowTokens: 0,
    headWindowTokens: 0,
    autoTickOnNewMessage: false,
    levels: { L3: { mergeCount: 2 } },
    ...(maxMergeLevel === undefined ? {} : { maxMergeLevel }),
  });
  const manager = await ContextManager.open({
    path,
    strategy,
    membrane: { complete: async () => ({ stopReason: 'end_turn', content: [] }) } as never,
  });
  const ids: string[] = [];
  for (let i = 0; i < 8; i++) {
    ids.push(manager.addMessage(i % 2 ? 'Claude' : 'User', [t(`raw-${i} ` + 'substance '.repeat(10))]));
  }
  const sourceIds: string[] = [];
  for (let n = 0; n < count; n++) {
    const id = `L${level}-seed-${n}`;
    sourceIds.push(id);
    strategy.seed({
      id,
      level,
      content: `authored ${id}`,
      tokens: 20,
      sourceLevel: level - 1,
      sourceIds: [`child-${id}`],
      sourceRange: { first: ids[n * 2]!, last: ids[n * 2 + 1]! },
      created: 1,
    });
  }
  return { strategy, ids, sourceIds };
}

test('a demand above the ceiling merges in place at the ceiling', async () => {
  const { strategy, ids, sourceIds } = await fixture(3, 4, 3);
  strategy.demand(4, ids[0]!, ids[7]!);

  const queue = strategy.queueView();
  assert.equal(queue.length, 1, 'one merge enqueued');
  assert.equal(queue[0].level, 3, 'clamped to the ceiling, not L4');
  assert.ok(
    queue[0].sourceIds.every((id) => sourceIds.includes(id)),
    'sources are the ceiling-level entries, not L3-minus-one',
  );
  assert.ok(queue[0].sourceIds.length >= 2, 'a real consolidation, not a rename');
});

test('a raised ceiling lets the same demand mint the deeper level', async () => {
  const { strategy, ids } = await fixture(3, 4, 4);
  strategy.demand(4, ids[0]!, ids[7]!);

  const queue = strategy.queueView();
  assert.equal(queue.length, 1);
  assert.equal(queue[0].level, 4, 'L4 is allowed when the ceiling permits it');
});

test('a demand at or below the ceiling is unaffected', async () => {
  const { strategy, ids, sourceIds } = await fixture(2, 4, 3);
  strategy.demand(3, ids[0]!, ids[7]!);

  const queue = strategy.queueView();
  assert.equal(queue.length, 1);
  assert.equal(queue[0].level, 3, 'L3 from L2 sources is a normal cross-level merge');
  assert.ok(queue[0].sourceIds.every((id) => sourceIds.includes(id)));
});

test('the default ceiling is 3', async () => {
  const { strategy, ids } = await fixture(3, 4);
  strategy.demand(5, ids[0]!, ids[7]!);

  const queue = strategy.queueView();
  assert.equal(queue.length, 1);
  assert.equal(queue[0].level, 3);
});
