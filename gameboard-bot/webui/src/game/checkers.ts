// International Draughts bridge. Unlike Wall Chess (which has a parallel TS
// rules engine in ./engine), every draughts rule is driven by the Rust engine
// compiled to WASM — mandatory-maximal capture, flying kings, and
// promotion-on-stop are perft-validated there, so the browser never reimplements
// them. This module is a thin typed wrapper around the `ck_*` WASM exports plus
// the board geometry the renderer needs.
import init, {
  ck_analyze,
  ck_apply,
  ck_legal_moves,
  ck_status,
} from '../wasm/wallchess_wasm.js'

export type CkColor = 'white' | 'black'

// Board is the 50 dark squares only; pieces are stored as dark-square indices
// (0..49, PDN order). `kings` is the crowned subset of `white ∪ black`.
export type CkState = {
  white: number[]
  black: number[]
  kings: number[]
  stm: CkColor
  idle: number
}

// A multi-jump is ONE move: `captured` is every jumped square, `to` the final
// landing. A quiet move has an empty `captured`.
export type CkMove = { from: number; to: number; captured: number[] }

export type CkStatus = {
  terminal: boolean
  winner: CkColor | null
  draw: boolean
}

export type CkAnalysis = {
  move: CkMove | null
  // White win chance 0..100; black = 100 - white.
  white: number
  black: number
  depth: number
  stopped: boolean
  nodes: number
}

// Logistic squash for the win meter (matches the Wall Chess SCORE_K).
export const CK_K = 200

// Lazy, one-shot WASM init. Browser only — callers invoke from client effects.
// init() is the only runtime step; the module itself is statically imported.
let wasmReady: Promise<void> | null = null
function ensureWasm(): Promise<void> {
  if (!wasmReady) wasmReady = init().then(() => undefined)
  return wasmReady
}

// ---- board geometry (mirrors checkers.rs idx_to_rc / rc_to_idx) ------------
// Row 0 is the top; a square (r,c) on the 10×10 grid is playable iff r+c is odd.

export function isDark(r: number, c: number): boolean {
  return (r + c) % 2 === 1
}

/// Dark-square index (0..49) for a playable (r,c). Caller guarantees `isDark`.
export function rcToIndex(r: number, c: number): number {
  const within = r % 2 === 0 ? (c - 1) / 2 : c / 2
  return r * 5 + within
}

/// (row, col) on the 10×10 grid for a dark-square index.
export function indexToRC(i: number): { r: number; c: number } {
  const r = Math.floor(i / 5)
  const within = i % 5
  const c = r % 2 === 0 ? within * 2 + 1 : within * 2
  return { r, c }
}

// ---- engine wrappers -------------------------------------------------------

// The canonical opening: 20 men per side, White to move. Hardcoded (matching the
// Rust `State::initial` constants) so the first render needs no async hop; the
// test asserts it equals the WASM `ck_initial`.
export function initialState(): CkState {
  const seq = (lo: number, hi: number) =>
    Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
  return { white: seq(30, 49), black: seq(0, 19), kings: [], stm: 'white', idle: 0 }
}

export async function legalMoves(state: CkState): Promise<CkMove[]> {
  await ensureWasm()
  return ck_legal_moves(state) as CkMove[]
}

export async function applyMove(state: CkState, move: CkMove): Promise<CkState> {
  await ensureWasm()
  return ck_apply(state, move) as CkState
}

// WASM serializes `None` as `undefined`; normalize to `null` so the app's
// `CkStatus.winner` contract holds.
export async function status(state: CkState): Promise<CkStatus> {
  await ensureWasm()
  const s = ck_status(state) as { terminal: boolean; winner?: CkColor | null; draw: boolean }
  return { terminal: s.terminal, winner: s.winner ?? null, draw: s.draw }
}

// Run the Gen-1 bot + win-split. `nodeLimit` 0 means fixed-depth; > 0 budgets
// nodes for stable tail latency.
export async function analyze(
  state: CkState,
  depth: number,
  nodeLimit = 0,
  k = CK_K,
): Promise<CkAnalysis> {
  await ensureWasm()
  return ck_analyze(state, depth, BigInt(nodeLimit), k) as CkAnalysis
}
