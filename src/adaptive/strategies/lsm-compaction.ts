/**
 * LSM-compaction folding solver — layered cascade, boundary-local changes.
 *
 * Three zones in the compiled context:
 *
 *   [HEAD: system+tools] [MEMORY CACHE: L1 | L2 | bottom] [TAIL: raw live]
 *
 * The HEAD and TAIL are fixed (managed by the caller). The MIDDLE is purely
 * compressed — no raw content belongs there. Any raw chunk that ages out of
 * the tail into the middle is immediately promoted to L1.
 *
 * Within the memory cache, content is ordered oldest-first:
 *   [bottom (L3+)] [L2] [L1]
 * The most stable content (deep summaries) sits right after the head; the
 * most recently compressed sits right before the tail. Changes propagate
 * LEFT (deeper), so KV states for the recent "state of mind" near the
 * tail are perturbed least.
 *
 * Cascade uses CUT-FIRST-K: when a layer overflows, cut a fixed-size block
 * from its oldest end and promote the whole block one level deeper. This
 * gives PREDICTABLE fold frequency.
 *
 * On most turns: return the carried frontier unchanged (HOLD). This is
 * where the savings live — pure append, zero prefix invalidation.
 */

import type {
  FoldingSolver,
  FoldingSolution,
  FoldingBudget,
  ChunkId,
  ProduceRequest,
} from '../folding-strategy.js';
import type { PickerInputs, PickerChunk } from '../picker.js';
import { SummaryTree } from '../summary-tree.js';
import { renderLayout } from '../render-offsets.js';

/**
 * Cap on how much raw content step 1 (immediate promotion) may take into the
 * memory cache in a single compile. Tail-adjacent chunks are taken first; the
 * remainder is deferred to subsequent compiles.
 *
 * Why a cap exists: step 1 was the solver's only unbounded step. Every other
 * cascade step is cut-first-k with a size cap. On 2026-08-28 a 42-chunk tail
 * drain was promoted in one compile, overflowing L1, L2 and the bottom layer
 * simultaneously and re-creating 172,820 tokens against 12,417 read — worse
 * than the kv-stable fold (~121K median) this strategy exists to beat.
 *
 * Why THIS size: the token cap equals the default `promotionSize` (the L1->L2
 * cut). Intake can therefore add at most one cut's worth of content to L1 per
 * compile, which bounds the cascade to roughly one cut-first-k per layer —
 * the boundary-local behavior the design promises. Against a 60K tail
 * (`recentWindowTokens`), a worst-case full drain amortizes over ~8 compiles
 * instead of landing as one global rewrite.
 *
 * The chunk cap is a backstop for many-small-chunks, where the token cap would
 * not bind: it bounds the number of L1 summarization calls queued per compile.
 *
 * Raising these re-opens the defect; lowering them slows backlog drain. If the
 * `deferred` count in the `[lsm-cascade] immediate:` log stays persistently
 * above zero across many compiles, the cap is too tight for the message rate.
 */
const MAX_IMMEDIATE_PROMOTION_TOKENS = 8000;
const MAX_IMMEDIATE_PROMOTION_CHUNKS = 8;

export interface LsmCompactionOptions {
  /** [L1, L2, bottom] as fractions of the middle budget. */
  layerBudgetRatios?: [number, number, number];
  /** Deepest fold level to target. Default 4. */
  maxFoldLevel?: number;
  /** Tokens over threshold before cascade triggers. Default 2000. */
  cascadeHysteresis?: number;
  /** Cut size for L1->L2 promotions. Default 8000.
   *  L2->bottom derives from L2 threshold (40%). */
  promotionSize?: number;
}

export class LsmCompactionStrategy implements FoldingSolver {
  readonly name = 'lsm-compaction';

  private readonly ratios: [number, number, number];
  private readonly maxFoldLevel: number;
  private readonly hysteresis: number;
  private readonly promotionSize: number;
  private readonly middleBudget: number;

  constructor(
    options: LsmCompactionOptions,
    middleBudget: number,
  ) {
    this.ratios = options.layerBudgetRatios ?? [0.45, 0.35, 0.20];
    this.maxFoldLevel = options.maxFoldLevel ?? 4;
    this.hysteresis = options.cascadeHysteresis ?? 2000;
    this.promotionSize = options.promotionSize ?? 8000;
    this.middleBudget = middleBudget;
  }

  solve(inputs: PickerInputs, budget: FoldingBudget): FoldingSolution {
    const tree = new SummaryTree(inputs);
    const ordered = [...inputs.chunks].sort((a, b) => a.sequence - b.sequence);

    const headIds = inputs.headChunkIds;
    const tailIds = inputs.tailChunkIds;

    const middle = ordered.filter(
      (c) => !headIds.has(c.id) && !tailIds.has(c.id),
    );

    const carried = new Map<ChunkId, number>();
    let hasCarried = false;
    for (const c of inputs.chunks) {
      carried.set(c.id, c.currentResolution);
      if (c.currentResolution > 0) hasCarried = true;
    }

    const pinCaps = new Map<ChunkId, number>();
    const pinnedRaw = new Set<ChunkId>();
    for (const c of middle) {
      if (c.pinned) {
        pinnedRaw.add(c.id);
      } else if (c.pinLevel !== undefined) {
        if (c.pinLevel <= 0) pinnedRaw.add(c.id);
      } else if (c.pinMaxLevel !== undefined) {
        if (c.pinMaxLevel <= 0) pinnedRaw.add(c.id);
        else pinCaps.set(c.id, c.pinMaxLevel);
      }
    }

    const frontier = new Map<ChunkId, number>(carried);
    const produced: ProduceRequest[] = [];

    // Three-layer thresholds: [L1, L2, bottom].
    const thresholds = this.ratios.map((r) => Math.floor(this.middleBudget * r));

    if (!hasCarried) {
      this.bootstrap(middle, frontier, tree, pinnedRaw, pinCaps, produced);
    } else {
      this.cascade(middle, frontier, tree, pinnedRaw, pinCaps, thresholds, produced);
    }

    const totalTokens = this.computeRenderedTokens(inputs, tree, frontier);
    let exhausted = false;
    if (totalTokens > budget.totalBudget) {
      // Over budget. Before escalating, lift the step-1 intake cap and drain
      // any deferred raw chunks: they are the largest units in the middle
      // (raw shards, not recall pairs), so they are the cheapest thing to
      // reclaim, and emergencyCascade is itself unbounded.
      //
      // The cap is an optimization, never a correctness constraint. Holding
      // raw chunks back while over budget keeps the middle bloated and
      // re-enters the emergency cascade on EVERY compile until the backlog
      // clears — measured on a synthetic drain against a near-full middle as
      // 5 consecutive emergency cascades (~850 produce requests each) where
      // the uncapped solver took 1. That is worse than the defect the cap
      // fixes, so under pressure the cap yields and behavior falls back to
      // the original, bounded-by-the-drain-size path.
      this.immediateIntake(
        middle, frontier, tree, pinnedRaw, pinCaps, produced,
        Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY,
      );
      const afterDrain = this.computeRenderedTokens(inputs, tree, frontier);
      if (afterDrain > budget.totalBudget) {
        this.emergencyCascade(middle, frontier, tree, pinnedRaw, pinCaps, produced);
        const afterEmergency = this.computeRenderedTokens(inputs, tree, frontier);
        exhausted = afterEmergency > budget.totalBudget;
      }
    }

    return { frontier, produced, exhausted };
  }

  /**
   * Bootstrap: all middle chunks go into compressed layers (no level-0).
   * Oldest 25% -> bottom (L3), next 35% -> L2, newest 40% -> L1.
   */
  private bootstrap(
    middle: PickerChunk[],
    frontier: Map<ChunkId, number>,
    tree: SummaryTree,
    pinnedRaw: ReadonlySet<ChunkId>,
    pinCaps: ReadonlyMap<ChunkId, number>,
    produced: ProduceRequest[],
  ): void {
    const movable = middle.filter(
      (c) => !pinnedRaw.has(c.id) && c.pinLevel === undefined,
    );
    const n = movable.length;
    if (n === 0) return;

    // Oldest -> bottom, next -> L2, newest -> L1.
    const cuts = [
      Math.floor(n * 0.25),
      Math.floor(n * 0.60),
    ];
    const levelMap = [3, 2, 1]; // segment index -> fold level

    for (let i = 0; i < n; i++) {
      const c = movable[i];
      let segment: number;
      if (i < cuts[0]) segment = 0;       // oldest 25% -> bottom
      else if (i < cuts[1]) segment = 1;   // next 35% -> L2
      else segment = 2;                    // newest 40% -> L1

      let targetLevel = levelMap[segment];

      const cap = pinCaps.get(c.id);
      if (cap !== undefined && targetLevel > cap) {
        targetLevel = Math.max(1, cap);
      }

      const maxAvail = tree.maxLevel(c.id);
      if (targetLevel > maxAvail) {
        for (let l = maxAvail + 1; l <= targetLevel; l++) {
          this.emitProduce(c, l, tree, produced);
        }
        // Clamp to what exists, but never below 1 in the middle.
        targetLevel = Math.max(1, maxAvail);
      }

      targetLevel = Math.min(targetLevel, this.maxFoldLevel);
      // Middle must be at least L1.
      targetLevel = Math.max(1, targetLevel);

      frontier.set(c.id, targetLevel);
    }

    console.error(
      `[lsm-cascade] bootstrap: placed ${n} middle chunks` +
      ` (bottom=${cuts[0]}, L2=${cuts[1] - cuts[0]}, L1=${n - cuts[1]})`,
    );
  }

  /**
   * Three-zone cascade:
   * 1. Immediate promotion: any raw (level-0) middle chunk -> L1.
   * 2. L1 overflow -> cut-first-k oldest L1 chunks to L2.
   * 3. L2 overflow -> cut-first-k oldest L2 chunks to bottom.
   * 4. Bottom overflow -> deepen all bottom chunks.
   */
  private cascade(
    middle: PickerChunk[],
    frontier: Map<ChunkId, number>,
    tree: SummaryTree,
    pinnedRaw: ReadonlySet<ChunkId>,
    pinCaps: ReadonlyMap<ChunkId, number>,
    thresholds: number[],
    produced: ProduceRequest[],
  ): void {
    // Step 1: take raw middle chunks (aged out of the tail) into L1 — CAPPED.
    // See immediateIntake() for the cap and its reasoning.
    const { promoted: immediatePromotions } = this.immediateIntake(
      middle, frontier, tree, pinnedRaw, pinCaps, produced,
      MAX_IMMEDIATE_PROMOTION_TOKENS, MAX_IMMEDIATE_PROMOTION_CHUNKS,
    );

    // Step 2-3: Check L1 and L2 for overflow, cascade via cut-first-k.
    // Layers: [L1, L2, bottom] indexed as [0, 1, 2].
    const layers = this.partitionLayers(middle, frontier, pinnedRaw);
    const layerTokens = layers.map((chunks) =>
      this.computeLayerTokens(chunks, frontier, tree),
    );

    let cascaded = false;

    // k=0 is L1 (level 1), promotes to L2 (level 2).
    // k=1 is L2 (level 2), promotes to bottom (level 3).
    for (let k = 0; k < 2; k++) {
      const overflow = layerTokens[k] - (thresholds[k] + this.hysteresis);
      if (overflow <= 0) continue;

      cascaded = true;
      const fromLevel = k + 1;  // L1=1, L2=2
      const toLevel = k + 2;    // L2=2, bottom=3
      const layerChunks = layers[k];

      // Cut size: configurable for L1->L2, derived for L2->bottom.
      const cutSize = k === 0
        ? this.promotionSize
        : Math.floor(thresholds[k] * 0.40);

      let accumulated = 0;
      let promotedCount = 0;

      for (const c of layerChunks) {
        if (accumulated >= cutSize && promotedCount > 0) break;

        if (pinnedRaw.has(c.id) || c.pinLevel !== undefined) continue;

        const cap = pinCaps.get(c.id);
        if (cap !== undefined && toLevel > cap) continue;

        const currentTokens = this.chunkRenderedTokens(c, frontier.get(c.id) ?? 0, tree);
        accumulated += currentTokens;

        let effectiveTarget = Math.min(toLevel, this.maxFoldLevel);
        if (cap !== undefined) effectiveTarget = Math.min(effectiveTarget, cap);

        const maxAvail = tree.maxLevel(c.id);
        if (effectiveTarget > maxAvail) {
          this.emitProduce(c, effectiveTarget, tree, produced);
          effectiveTarget = Math.max(fromLevel, maxAvail);
        }

        if (effectiveTarget <= (frontier.get(c.id) ?? 0)) continue;

        frontier.set(c.id, effectiveTarget);
        promotedCount++;
      }

      if (promotedCount > 0) {
        console.error(
          `[lsm-cascade] L${fromLevel}→L${toLevel}: cut ${promotedCount} chunks` +
          ` (~${accumulated} tokens, cut target ${cutSize})`,
        );
      }

      // Recompute destination layer for cascading overflow check.
      layerTokens[k + 1] = this.computeLayerTokens(
        this.getLayerChunks(middle, frontier, pinnedRaw, k + 1),
        frontier,
        tree,
      );
    }

    // Step 4: Bottom overflow -> deepen.
    if (cascaded || immediatePromotions > 0) {
      const bottomTokens = this.computeLayerTokens(
        this.getLayerChunks(middle, frontier, pinnedRaw, 2),
        frontier,
        tree,
      );
      const bottomOverflow = bottomTokens - (thresholds[2] + this.hysteresis);
      if (bottomOverflow > 0) {
        this.compressBottom(middle, frontier, tree, pinnedRaw, pinCaps, produced);
      }
    }
  }

  /**
   * Step 1 intake: promote raw (level-0) middle chunks — chunks that just aged
   * out of the tail — up to L1, bounded by `limitTokens`/`limitChunks`.
   *
   * This step was the solver's ONLY unbounded step until 2026-08-28. Every
   * other cascade step is cut-first-k with a size cap; this one took every raw
   * chunk it found. A 42-chunk tail drain was therefore promoted in a single
   * compile, overflowing L1, L2 and the bottom layer at once and re-creating
   * 172,820 tokens against 12,417 read — worse than the kv-stable fold (~121K
   * median) this strategy exists to beat, and it lands exactly when the tail is
   * filling fastest. Bounding intake bounds L1 growth per compile, so at most
   * one cut-first-k fires per layer and the cascade stays boundary-local.
   *
   * ORDER — tail-adjacent (highest sequence) first, remainder deferred to
   * later compiles. `renderLayout` emits the middle in SOURCE SEQUENCE order,
   * and KV cost is governed by the earliest position at which the frontier
   * diverges (see render-offsets.ts header). Chunks that just aged out render
   * last in the middle, so taking those first puts the divergence point as late
   * as possible, and everything after it is compressed recall pairs rather than
   * raw shards. A deferred chunk renders raw at the position it already
   * occupied, so it contributes no divergence of its own.
   *
   * NO STARVATION: in steady state roughly one chunk ages out per compile (a
   * compile is an LLM call), so intake drains backlog several times faster than
   * it fills. Only a bulk drain builds a backlog, and it clears in a few
   * compiles. The deferred count is logged so a broken assumption is visible.
   *
   * The caller lifts the limits (Infinity) when the compile is over budget —
   * see solve(). The cap is an optimization, never a correctness constraint:
   * deferred raw chunks are the largest things in the middle, and holding them
   * back while over budget would re-enter the unbounded emergency cascade on
   * every compile until the backlog cleared, which measures WORSE than the
   * defect being fixed.
   */
  private immediateIntake(
    middle: PickerChunk[],
    frontier: Map<ChunkId, number>,
    tree: SummaryTree,
    pinnedRaw: ReadonlySet<ChunkId>,
    pinCaps: ReadonlyMap<ChunkId, number>,
    produced: ProduceRequest[],
    limitTokens: number,
    limitChunks: number,
  ): { promoted: number; deferred: number } {
    const rawMiddle: PickerChunk[] = [];
    for (const c of middle) {
      if (pinnedRaw.has(c.id) || c.pinLevel !== undefined) continue;
      if ((frontier.get(c.id) ?? 0) > 0) continue;
      const cap = pinCaps.get(c.id);
      if (cap !== undefined && cap < 1) continue;
      rawMiddle.push(c);
    }
    // Tail-adjacent first.
    const intakeOrder = [...rawMiddle].sort((a, b) => b.sequence - a.sequence);

    let promoted = 0;
    let intakeTokens = 0;
    let intakeCount = 0;
    let deferred = 0;

    for (const c of intakeOrder) {
      if (intakeCount >= limitChunks || intakeTokens >= limitTokens) {
        deferred++;
        continue;
      }
      // Charge the cap on the raw size taken in, on either path: a produce
      // request (an L1 summarization call) and a frontier move both cost real
      // work, and both perturb this compile.
      intakeTokens += c.rawTokens;
      intakeCount++;

      const maxAvail = tree.maxLevel(c.id);
      if (maxAvail < 1) {
        this.emitProduce(c, 1, tree, produced);
        // Leave at 0 until the L1 summary exists.
        continue;
      }

      frontier.set(c.id, 1);
      promoted++;
    }

    if (promoted > 0 || deferred > 0) {
      const capLabel = Number.isFinite(limitTokens)
        ? `caps ${limitChunks}/${limitTokens}`
        : 'caps LIFTED (over budget)';
      console.error(
        `[lsm-cascade] immediate: promoted ${promoted} raw middle chunks to L1` +
        ` (intake ${intakeCount} chunks / ~${intakeTokens} tokens, ${capLabel}` +
        `, deferred ${deferred})`,
      );
    }

    return { promoted, deferred };
  }

  private compressBottom(
    middle: PickerChunk[],
    frontier: Map<ChunkId, number>,
    tree: SummaryTree,
    pinnedRaw: ReadonlySet<ChunkId>,
    pinCaps: ReadonlyMap<ChunkId, number>,
    produced: ProduceRequest[],
  ): void {
    const bottomChunks = this.getLayerChunks(middle, frontier, pinnedRaw, 2);
    let compressed = 0;

    for (const c of bottomChunks) {
      if (pinnedRaw.has(c.id) || c.pinLevel !== undefined) continue;

      const current = frontier.get(c.id) ?? 0;
      let newLevel = current + 1;

      const cap = pinCaps.get(c.id);
      if (cap !== undefined && newLevel > cap) continue;
      newLevel = Math.min(newLevel, this.maxFoldLevel);

      if (newLevel > tree.maxLevel(c.id)) {
        this.emitProduce(c, newLevel, tree, produced);
        continue;
      }

      frontier.set(c.id, newLevel);
      compressed++;
    }

    if (compressed > 0) {
      console.error(
        `[lsm-cascade] bottom-rewrite: deepened ${compressed} chunks`,
      );
    }
  }

  private emergencyCascade(
    middle: PickerChunk[],
    frontier: Map<ChunkId, number>,
    tree: SummaryTree,
    pinnedRaw: ReadonlySet<ChunkId>,
    pinCaps: ReadonlyMap<ChunkId, number>,
    produced: ProduceRequest[],
  ): void {
    for (const c of middle) {
      if (pinnedRaw.has(c.id) || c.pinLevel !== undefined) continue;

      const current = frontier.get(c.id) ?? 0;
      let maxTarget = this.maxFoldLevel;

      const cap = pinCaps.get(c.id);
      if (cap !== undefined) maxTarget = Math.min(maxTarget, cap);

      const maxAvail = tree.maxLevel(c.id);
      const effectiveTarget = Math.min(maxTarget, maxAvail);

      if (effectiveTarget > current) {
        frontier.set(c.id, effectiveTarget);
      }

      if (maxTarget > maxAvail) {
        this.emitProduce(c, maxTarget, tree, produced);
      }
    }

    console.error(
      `[lsm-cascade] emergency: promoted all movable chunks to max available level`,
    );
  }

  /**
   * Partition middle chunks into three compressed layers.
   * Returns [L1, L2, bottom(L3+)], each sorted oldest-first.
   * Pinned-raw chunks and level-0 chunks are excluded (they shouldn't
   * exist in steady state but may during the immediate-promotion step).
   */
  private partitionLayers(
    middle: PickerChunk[],
    frontier: ReadonlyMap<ChunkId, number>,
    pinnedRaw: ReadonlySet<ChunkId>,
  ): [PickerChunk[], PickerChunk[], PickerChunk[]] {
    const l1: PickerChunk[] = [];
    const l2: PickerChunk[] = [];
    const bottom: PickerChunk[] = [];

    for (const c of middle) {
      if (pinnedRaw.has(c.id)) continue;
      const level = frontier.get(c.id) ?? 0;
      if (level <= 0) continue; // Raw — handled by immediate promotion.
      if (level === 1) l1.push(c);
      else if (level === 2) l2.push(c);
      else bottom.push(c);
    }

    return [l1, l2, bottom];
  }

  private getLayerChunks(
    middle: PickerChunk[],
    frontier: ReadonlyMap<ChunkId, number>,
    pinnedRaw: ReadonlySet<ChunkId>,
    layerIndex: number,
  ): PickerChunk[] {
    return this.partitionLayers(middle, frontier, pinnedRaw)[layerIndex];
  }

  private computeLayerTokens(
    chunks: PickerChunk[],
    frontier: ReadonlyMap<ChunkId, number>,
    tree: SummaryTree,
  ): number {
    let total = 0;
    const renderedSummaries = new Set<string>();

    for (const c of chunks) {
      const level = c.pinned ? 0 : (frontier.get(c.id) ?? 0);
      if (level === 0) {
        total += c.rawTokens;
        continue;
      }
      const ancestor = tree.ancestorAt(c.id, level);
      if (!ancestor) {
        total += c.rawTokens;
        continue;
      }
      if (!renderedSummaries.has(ancestor.id)) {
        renderedSummaries.add(ancestor.id);
        total += ancestor.recallTokens;
      }
    }

    return total;
  }

  private chunkRenderedTokens(
    c: PickerChunk,
    level: number,
    tree: SummaryTree,
  ): number {
    if (c.pinned || level === 0) return c.rawTokens;
    const ancestor = tree.ancestorAt(c.id, level);
    if (!ancestor) return c.rawTokens;
    return ancestor.recallTokens;
  }

  private computeRenderedTokens(
    inputs: PickerInputs,
    tree: SummaryTree,
    frontier: ReadonlyMap<ChunkId, number>,
  ): number {
    return renderLayout(inputs, tree, frontier).totalTokens;
  }

  private emitProduce(
    c: PickerChunk,
    level: number,
    tree: SummaryTree,
    produced: ProduceRequest[],
  ): void {
    const existing = produced.find(
      (p) =>
        p.level === level &&
        p.range.firstChunkId === c.id &&
        p.range.lastChunkId === c.id,
    );
    if (existing) return;

    const parentLevel = level - 1;
    const ancestor = parentLevel > 0 ? tree.ancestorAt(c.id, parentLevel) : null;
    const range = ancestor
      ? { firstChunkId: ancestor.sourceRange.first, lastChunkId: ancestor.sourceRange.last }
      : { firstChunkId: c.id, lastChunkId: c.id };

    produced.push({ level, range });
  }
}
