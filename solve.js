// Fast drop-in for makefun-diff 1.0.0 (make.fun's own line diff).
//
// run(input) returns exactly what makefun-diff 1.0.0's diffPatch returns for
// { old, new }: the same patch text, byte for byte. The edit script is the
// reference's: common prefix, then common suffix, then Myers's greedy search
// over the middle, keeping on each diagonal the path furthest into the old
// text and, on a tie, the deletion's path (jsdiff takes the insertion's),
// never moving past either text's end, pruning no edge diagonal, and
// following every run of equal lines as far as it goes; the first path to
// reach both ends is the script.
//
// Speed, versus the reference's linked-list paths (one object per (d, k)
// step, one per component):
// - Lines are interned to integer ids once, so every comparison in the
//   search is an int comparison. A text's last line, when it has no
//   newline, interns under the negative of its part's id, so it stays a
//   different line from the same part with a newline.
// - The search keeps only the furthest-reaching x per diagonal per d in
//   one flat Int32Array of snapshots and rebuilds the single winning path
//   by walking the snapshots back down, re-deriving each step's choice
//   with the same tie-break. The inner loop carries the next diagonal's
//   candidate in a register instead of re-reading it, and picks the
//   deletion or the insertion by comparing the x each reaches.
// - The patch is emitted a whole block of lines at a time (one join per
//   block, not one string per line).
// No allocation happens in the search loop.

const KEEP = 0;
const DELETE = 1;
const INSERT = 2;

/** Context lines a hunk shows before and after each change run. */
const CONTEXT = 3;

/** Furthest-reaching x per diagonal per d, snapshot d at index d*d, diagonal k at k+d. -1 = unreachable. */
let vs = new Int32Array(1025 * 1025);
let vsCapD = 1024;

function ensureVs(d) {
  if (d <= vsCapD) return vs;
  let nd = vsCapD;
  while (nd < d) nd *= 2;
  const nvs = new Int32Array((nd + 1) * (nd + 1));
  nvs.set(vs);
  vs = nvs;
  vsCapD = nd;
  return vs;
}

/**
 * A text's lines as split parts (each part without its "\n") and, in
 * parallel, each line's interned id. The map is shared by both texts so
 * equal lines share an id; a last line with no newline gets the negative
 * of its part's id, so it differs from the same part with a newline.
 * parts.length is the line count, plus one trailing "" when the text ends
 * in a newline.
 */
function tokenize(text, idsByLine) {
  const parts = text.split('\n');
  const last = parts.length - 1;
  const hasBare = parts[last] !== '';
  const n = last + (hasBare ? 1 : 0);
  const ids = new Int32Array(n);
  for (let i = 0; i < last; i++) {
    const part = parts[i];
    let id = idsByLine.get(part);
    if (id === undefined) {
      id = idsByLine.size;
      idsByLine.set(part, id);
    }
    ids[i] = id;
  }
  if (hasBare) {
    const part = parts[last];
    let id = idsByLine.get(part);
    if (id === undefined) {
      id = idsByLine.size;
      idsByLine.set(part, id);
    }
    ids[last] = -(id + 1);
  }
  return { parts, ids, hasBare };
}

/**
 * The edit script's runs for the middle slices a[0..N), b[0..M), as two
 * parallel arrays (kinds, counts) in forward order. Only called with the
 * common prefix and suffix already trimmed, so the first lines differ
 * (unless one side is empty).
 */
function myersRuns(a, b, N, M) {
  const kinds = [];
  const counts = [];
  if (N === 0) {
    if (M > 0) {
      kinds.push(INSERT);
      counts.push(M);
    }
    return { kinds, counts };
  }
  if (M === 0) {
    kinds.push(DELETE);
    counts.push(N);
    return { kinds, counts };
  }
  let V = vs;

  // d = 0: the initial snake along diagonal 0.
  let x = 0;
  let y = 0;
  while (x < N && y < M && a[x] === b[y]) {
    x++;
    y++;
  }
  V[0] = x;
  let dEnd = 0;
  let kEnd = 0;
  if (x !== N || y !== M) {
    let found = false;
    for (let d = 1; d <= N + M && !found; d++) {
      V = ensureVs(d);
      const base = d * d;
      const prevBase = (d - 1) * (d - 1);
      const lastJ = 2 * d;
      // The insertion candidate for j = 0 is V[d-1][-(d-1)], at prevBase.
      // Each step's deletion candidate is the previous step's insertion
      // candidate (both read V[prevBase + j - 2] / V[prevBase + j]), so
      // carry it in adPrev instead of reading it twice.
      let ad = V[prevBase];
      let adPrev = -1;
      for (let j = 0; j <= lastJ; j += 2) {
        const k = j - d;
        const rm = adPrev;
        // Each candidate's x after its edit, -1 when it cannot move:
        // deletion reaches rm + 1, insertion reaches ad. The deletion
        // wins on a tie (rm + 1 >= ad), which is the reference's
        // byDeletion >= byInsertion on oldPos values (rm, ad - 1).
        const xDel = rm !== -1 && rm < N ? rm + 1 : -1;
        const xIns = ad !== -1 && ad - k - 1 < M ? ad : -1;
        if (xDel < 0 && xIns < 0) {
          V[base + j] = -1;
        } else {
          let xx = xDel >= xIns ? xDel : xIns;
          let yy = xx - k;
          while (xx < N && yy < M && a[xx] === b[yy]) {
            xx++;
            yy++;
          }
          V[base + j] = xx;
          if (xx === N && yy === M) {
            dEnd = d;
            kEnd = k;
            found = true;
            break;
          }
        }
        adPrev = ad;
        ad = j <= lastJ - 4 ? V[prevBase + j + 2] : -1;
      }
    }
    if (!found) throw new Error('makefun-diff: the search ended without reaching both ends');
  }

  // Walk the winning path back down, re-deriving each step's choice.
  const revKinds = [];
  const revCounts = [];
  let d = dEnd;
  let j = kEnd + dEnd;
  x = V[d * d + j];
  while (d > 0) {
    const prevBase = (d - 1) * (d - 1);
    const rm = j >= 2 ? V[prevBase + j - 2] : -1;
    const ad = j <= 2 * d - 2 ? V[prevBase + j] : -1;
    const k = j - d;
    const xDel = rm !== -1 && rm < N ? rm + 1 : -1;
    const xIns = ad !== -1 && ad - k - 1 < M ? ad : -1;
    let xEdit;
    let isDel;
    let jPrev;
    if (xDel >= xIns) {
      xEdit = xDel;
      isDel = true;
      jPrev = j - 2;
    } else {
      xEdit = xIns;
      isDel = false;
      jPrev = j;
    }
    const keep = x - xEdit;
    if (keep > 0) {
      revKinds.push(KEEP);
      revCounts.push(keep);
    }
    revKinds.push(isDel ? DELETE : INSERT);
    revCounts.push(1);
    x = V[prevBase + jPrev];
    j = jPrev;
    d--;
  }
  if (x > 0) {
    revKinds.push(KEEP);
    revCounts.push(x);
  }
  // Reverse, merging runs of one kind that touch (same-kind edits with no
  // kept line between them are one run in the reference).
  for (let i = revKinds.length - 1; i >= 0; i--) {
    const kind = revKinds[i];
    const count = revCounts[i];
    const last = kinds.length - 1;
    if (last >= 0 && kinds[last] === kind) counts[last] += count;
    else {
      kinds.push(kind);
      counts.push(count);
    }
  }
  return { kinds, counts };
}

/**
 * One block of lines as the patch writes it: the op before every line, a
 * "%" line after a last line with no newline. parts holds the lines
 * without their "\n"; only a text's last line can lack one, so bareLast
 * is only consulted when the block reaches the end of its parts.
 */
function emitBlock(out, op, parts, from, to, bareLast) {
  if (from >= to) return;
  const sep = '\n' + op;
  if (bareLast && to === parts.length) {
    if (to - from > 1) out.push(op + parts.slice(from, to - 1).join(sep) + '\n');
    out.push(op + parts[to - 1] + '\n%\n');
  } else {
    out.push(op + parts.slice(from, to).join(sep) + '\n');
  }
}

/** makefun-diff 1.0.0's answer for one case's input. */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const idsByLine = new Map();
  const aTok = tokenize(input.old, idsByLine);
  const bTok = tokenize(input.new, idsByLine);
  const aParts = aTok.parts;
  const bParts = bTok.parts;
  const a = aTok.ids;
  const b = bTok.ids;
  const n = a.length;
  const m = b.length;

  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix && a[n - 1 - suffix] === b[m - 1 - suffix]) suffix++;
  const mid = myersRuns(
    a.subarray(prefix, n - suffix),
    b.subarray(prefix, m - suffix),
    n - prefix - suffix,
    m - prefix - suffix,
  );

  // The full runs: the prefix keep, the middle's runs, the suffix keep.
  const kinds = [];
  const counts = [];
  const pushRun = (kind, count) => {
    if (count <= 0) return;
    const last = kinds.length - 1;
    if (last >= 0 && kinds[last] === kind) counts[last] += count;
    else {
      kinds.push(kind);
      counts.push(count);
    }
  };
  pushRun(KEEP, prefix);
  for (let i = 0; i < mid.kinds.length; i++) pushRun(mid.kinds[i], mid.counts[i]);
  pushRun(KEEP, suffix);

  // The change runs: each a stretch of deletions and insertions between two kept lines, at old line o and new line p.
  const chO = [];
  const chP = [];
  const chD = [];
  const chI = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (let i = 0; i < kinds.length; i++) {
    const kind = kinds[i];
    const count = counts[i];
    if (kind === KEEP) {
      o += count;
      p += count;
      continue;
    }
    const last = chO.length - 1;
    if (last < 0 || chO[last] + chD[last] !== o || chP[last] + chI[last] !== p) {
      chO.push(o);
      chP.push(p);
      chD.push(0);
      chI.push(0);
    }
    if (kind === DELETE) {
      chD[chD.length - 1] += count;
      o += count;
      deleted += count;
    } else {
      chI[chI.length - 1] += count;
      p += count;
      inserted += count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (let i = 0; i < chO.length; i++) {
    const h = hunks[hunks.length - 1];
    const prev = h !== undefined ? h[h.length - 1] : -1;
    if (prev >= 0 && chO[i] - (chO[prev] + chD[prev]) <= 2 * CONTEXT) h.push(i);
    else hunks.push([i]);
  }

  const out = [`makefun-diff/1 old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (const h of hunks) {
    const first = h[0];
    const last = h[h.length - 1];
    const oldStart = Math.max(0, chO[first] - CONTEXT);
    const newStart = chP[first] - (chO[first] - oldStart);
    const after = Math.min(CONTEXT, n - (chO[last] + chD[last]));
    const oldEnd = chO[last] + chD[last] + after;
    const newEnd = chP[last] + chI[last] + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const ci of h) {
      emitBlock(out, '=', aParts, at, chO[ci], aTok.hasBare);
      emitBlock(out, '-', aParts, chO[ci], chO[ci] + chD[ci], aTok.hasBare);
      emitBlock(out, '+', bParts, chP[ci], chP[ci] + chI[ci], bTok.hasBare);
      at = chO[ci] + chD[ci];
    }
    emitBlock(out, '=', aParts, at, oldEnd, aTok.hasBare);
  }
  return out.join('');
}
