// Wall Chess bot API. The move is chosen by the Rust engine compiled to WASM
// (src/wasm), loaded lazily in the browser — no server round-trip. The pure-TS
// engine (./bot) stays as a fallback if the WASM module fails to load.
import type { GameState, Move } from './engine'
import { chooseMove } from './bot'
import { parseGameState } from './validate'

// Default search depth (plies) for the WASM engine.
const BOT_DEPTH = 2
// Logistic squash constant for the 0..100 win meter (larger -> closer to 50/50).
const SCORE_K = 200

type WasmModule = typeof import('../wasm/wallchess_wasm.js')

// Lazy, one-shot WASM init. Browser only — callers invoke from client effects.
let wasmReady: Promise<WasmModule> | null = null
function loadWasm(): Promise<WasmModule> {
  if (!wasmReady) {
    wasmReady = (async () => {
      const mod = await import('../wasm/wallchess_wasm.js')
      await mod.default() // init(): fetch + instantiate the .wasm
      return mod
    })()
  }
  return wasmReady
}

// Opt-in trained-net bot (toy): enabled with `?net=1` in the URL. Heuristic
// alpha-beta stays the default; the net is lazy-loaded once and any failure
// falls back to the heuristic path below.
const NET_WEIGHTS_URL = '/wallnet.safetensors'
const NET_SIMS = 200
let netBotReady: Promise<NetBot> | null = null
function netBotEnabled(): boolean {
  return (
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).has('net')
  )
}
function getNetBot(): Promise<NetBot> {
  if (!netBotReady) netBotReady = loadNetBot(NET_WEIGHTS_URL)
  return netBotReady
}

// Bot move, same call shape as the old server function: `botMove({ data })`.
// Validates the state at the boundary, then runs the Rust search in WASM,
// falling back to the pure-TS engine if WASM is unavailable.
export async function botMove({ data }: { data: unknown }): Promise<Move> {
  const state = parseGameState(data)
  if (netBotEnabled()) {
    try {
      const bot = await getNetBot()
      const { move } = bot.analyze(state, NET_SIMS)
      if (move) return move
    } catch (err) {
      console.warn('net bot unavailable, using heuristic', err)
    }
  }
  try {
    const wasm = await loadWasm()
    return wasm.choose_move(state, BOT_DEPTH) as Move
  } catch (err) {
    console.warn('WASM bot unavailable, using TS fallback', err)
    return chooseMove(state)
  }
}

// Win-probability split for the UI meter. south + north === 100.
export type Analysis = { move: Move | null; south: number; north: number }

export async function analyzePosition(
  state: GameState,
  depth = BOT_DEPTH,
  k = SCORE_K,
): Promise<Analysis> {
  const wasm = await loadWasm()
  return wasm.analyze_state(state, depth, k) as Analysis
}

// One pruned successor in the state graph: a strong move + the resulting
// SOUTH win-chance (already scored). `score` is the raw side-to-move eval.
export type RankedMove = { move: Move; south: number; score: number }

// Top strong successors for the graph view — pruned in Rust so the graph stays
// light: at most `k` moves, useless walls dropped, blunders (worse than best by
// > `margin` centi-steps) dropped. Each comes pre-scored.
export async function topMoves(
  state: GameState,
  depth = BOT_DEPTH,
  k = 3,
  margin = 80,
  winK = SCORE_K,
): Promise<RankedMove[]> {
  const wasm = await loadWasm()
  return wasm.top_moves_js(state, depth, k, margin, winK) as RankedMove[]
}

// ---- trained-net bot (opt-in) ---------------------------------------------
// The net bot lives in a SEPARATE wasm bundle (src/wasm-net) because it bundles
// candle (~730KB extra). It is lazy-loaded only when actually used, so the
// default engine/graph bundle stays small. Rebuild it with:
//   wasm-pack build wasm --target web --release --features net
//   cp wasm/pkg/wallchess_wasm* webui/src/wasm-net/
type NetWasmModule = typeof import('../wasm-net/wallchess_wasm.js')

let netWasmReady: Promise<NetWasmModule> | null = null
function loadNetWasm(): Promise<NetWasmModule> {
  if (!netWasmReady) {
    netWasmReady = (async () => {
      const mod = await import('../wasm-net/wallchess_wasm.js')
      await mod.default()
      return mod
    })()
  }
  return netWasmReady
}

// A loaded net bot: pick a move via MCTS over the trained value/policy net.
// `sims` is the simulation budget per move (higher = stronger, slower).
export type NetBot = {
  analyze: (state: GameState, sims?: number) => Analysis
  free: () => void
}

// Fetch the safetensors weights and instantiate the net bot in wasm. The weights
// are produced by `trainer/train.py` and served as a static asset.
export async function loadNetBot(weightsUrl: string): Promise<NetBot> {
  const mod = await loadNetWasm()
  const buf = await fetch(weightsUrl).then((r) => {
    if (!r.ok) throw new Error(`fetch weights ${weightsUrl}: ${r.status}`)
    return r.arrayBuffer()
  })
  const bot = new mod.NetBot(new Uint8Array(buf))
  return {
    analyze: (state, sims = 200) => bot.best_move(state, sims) as Analysis,
    free: () => bot.free(),
  }
}

// A fully precomputed pruned state graph (BFS done entirely in Rust).
export type GraphNodeData = {
  key: string
  ply: number
  south: number
  state: GameState
}
export type GraphEdgeData = { from: string; to: string; move: Move }
export type GraphData = {
  nodes: GraphNodeData[]
  edges: GraphEdgeData[]
  capped: boolean
}

// Generate the whole pruned graph from `state` in one WASM call. `maxDepth`
// controls how far ahead Rust precomputes; `maxNodes` is the hard memory ceiling
// (the returned `capped` flag is true if it stopped early). Orders of magnitude
// faster than expanding node-by-node from JS.
export async function topGraph(
  state: GameState,
  depth = BOT_DEPTH,
  k = 5,
  margin = 400,
  maxDepth = 12,
  maxNodes = 1500,
  winK = SCORE_K,
): Promise<GraphData> {
  const wasm = await loadWasm()
  return wasm.generate_graph_js(
    state,
    depth,
    k,
    margin,
    winK,
    maxDepth,
    maxNodes,
  ) as GraphData
}
