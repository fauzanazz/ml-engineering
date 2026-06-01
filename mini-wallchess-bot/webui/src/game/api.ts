// Wall Chess bot API. The move is chosen by the Rust engine compiled to WASM
// (src/wasm), loaded lazily in the browser — no server round-trip. The pure-TS
// engine (./bot) stays as a fallback if the WASM module fails to load.
import {
  type Cell,
  type GameState,
  type Move,
  type Side,
  GOAL_ROW,
  distanceToGoal,
  isLegalMove,
  pawnMoves,
  stateKey,
} from './engine'
import { chooseMove } from './bot'
import { parseGameState } from './validate'

// Anti-oscillation. In an awkward race the bounded-depth search can drift: the
// true finishing path needs a temporary "down/away" step to route around a wall
// line that the horizon scores no better than holding still, and there is no
// repetition rule to force progress. So the opponent quietly races home
// (observed: bot at 98% win-prob oscillating 6,4<->6,5 for ~18 plies, then
// loses; and a wider wander over rows 7-9 for ~18 plies, also a loss). The Rust
// eval now carries a forward-progress tie-break that removes most of this, but
// the band-aid stays as a safety net for drifts that span beyond the search
// horizon. We remember the bot's recent cells AND the closest BFS distance it
// has reached; if the engine's chosen pawn move revisits a cell OR fails to beat
// that closest distance, we override it with the move that most reduces our own
// shortest-path distance (a 1-ply look gets this right), breaking the loop.
const HISTORY_LEN = 12 // long enough to catch wide multi-cell wanders, not just 2-cell shuffles
type Visit = { r: number; c: number; dist: number }
const botHistory: Record<Side, Visit[]> = { south: [], north: [] }

function sameCell(a: Cell, b: Cell): boolean {
  return a.r === b.r && a.c === b.c
}

function startCell(side: Side): Cell {
  return { r: side === 'south' ? 1 : 9, c: 5 }
}

function greedyProgressMove(state: GameState, side: Side): Move | null {
  const goal = GOAL_ROW[side]
  const moves = pawnMoves(state, side)
  if (moves.length === 0) return null
  let best = moves[0]
  let bestScore = Infinity
  for (const to of moves) {
    const dist = distanceToGoal(state.walls, to, goal)
    // progress first; tie-break AWAY from recently-visited cells to break loops
    const revisits = botHistory[side].some((v) => v.r === to.r && v.c === to.c) ? 1 : 0
    const score = dist * 10 + revisits
    if (score < bestScore) {
      bestScore = score
      best = to
    }
  }
  return { type: 'move', to: best }
}

function antiOscillate(state: GameState, chosen: Move): Move {
  const side = state.turn
  const hist = botHistory[side]
  const here = state.pawns[side]
  const opp = state.pawns[side === 'south' ? 'north' : 'south']
  const goal = GOAL_ROW[side]
  // Reset only on a genuinely fresh game (both pawns home, no walls). The old
  // "pawn on its start cell" test also fired mid-game when the bot wandered back
  // through its start cell, wiping the loop memory exactly when it was needed.
  if (
    state.walls.length === 0 &&
    sameCell(here, startCell(side)) &&
    sameCell(opp, startCell(side === 'south' ? 'north' : 'south'))
  ) {
    hist.length = 0
  }

  const distHere = distanceToGoal(state.walls, here, goal)
  const closest = hist.reduce((m, v) => Math.min(m, v.dist), Infinity)

  let out = chosen
  if (chosen.type === 'move') {
    const toDist = distanceToGoal(state.walls, chosen.to, goal)
    const revisiting = hist.some((v) => v.r === chosen.to.r && v.c === chosen.to.c)
    // Stagnation: enough history, yet the chosen move fails to get strictly
    // closer than we have already been — a drift, not real progress.
    const stagnating = hist.length >= 4 && toDist >= closest
    if (revisiting || stagnating) {
      const progress = greedyProgressMove(state, side)
      if (progress) out = progress
    }
  }
  hist.push({ r: here.r, c: here.c, dist: distHere })
  if (hist.length > HISTORY_LEN) hist.shift()
  return out
}

// Default search depth (plies) for the WASM engine. Depth 3: with the fixed
// eval (high w_wall) it beats depth 2 16-0 — as strong as depth 4 here but
// ~10x faster (one fewer ply over a ~130-wide tree), so move latency stays low
// in the browser. Depth 4 was correct but too slow in wasm.
const BOT_DEPTH = 3
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
const MOVE_BOOK_URL = '/counter-book.jsonl'
const NET_SIMS = 600
let netBotReady: Promise<NetBot> | null = null
let moveBookReady: Promise<Map<string, number>> | null = null
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
function getMoveBook(): Promise<Map<string, number>> {
  if (!moveBookReady) moveBookReady = loadMoveBook(MOVE_BOOK_URL)
  return moveBookReady
}

// Which engine picks the move. `undefined` keeps the legacy behavior: net when
// `?net=1` is set, heuristic otherwise. Arena mode passes an explicit engine per
// side so it can pit net vs heuristic regardless of the URL flag.
export type BotEngine = 'heuristic' | 'net'

// Bot move, same call shape as the old server function: `botMove({ data })`.
// Validates the state at the boundary, then runs the Rust search in WASM,
// falling back to the pure-TS engine if WASM is unavailable.
export async function botMove({
  data,
  engine,
  depth = BOT_DEPTH,
  sims = NET_SIMS,
}: {
  data: unknown
  engine?: BotEngine
  // Optional per-call overrides so bot variations (see ./bots) can run the same
  // engine at different strengths. Default to the legacy depth/sim budget.
  depth?: number
  sims?: number
}): Promise<Move> {
  const state = parseGameState(data)
  const useNet = engine === 'net' || (engine === undefined && netBotEnabled())
  if (useNet) {
    try {
      const bookMove = await chooseBookMove(state).catch((err) => {
        console.warn('move book unavailable, using net search', err)
        return null
      })
      if (bookMove) return antiOscillate(state, bookMove)
      const bot = await getNetBot()
      const { move } = bot.analyze(state, sims)
      if (move) return antiOscillate(state, move)
    } catch (err) {
      console.warn('net bot unavailable, using heuristic', err)
    }
  }
  try {
    const wasm = await loadWasm()
    return antiOscillate(state, wasm.choose_move(state, depth) as Move)
  } catch (err) {
    console.warn('WASM bot unavailable, using TS fallback', err)
    return antiOscillate(state, chooseMove(state))
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

type BookRecord = { type?: string; key?: string; a?: number }

async function loadMoveBook(bookUrl: string): Promise<Map<string, number>> {
  const text = await fetch(bookUrl).then((r) => {
    if (!r.ok) throw new Error(`fetch move book ${bookUrl}: ${r.status}`)
    return r.text()
  })
  const out = new Map<string, number>()
  for (const line of text.split('\n')) {
    if (!line.includes('"type":"book"')) continue
    const rec = JSON.parse(line) as BookRecord
    if (rec.type === 'book' && typeof rec.key === 'string' && typeof rec.a === 'number') {
      out.set(rec.key, rec.a)
    }
  }
  return out
}

async function chooseBookMove(state: GameState): Promise<Move | null> {
  const book = await getMoveBook()
  const action = book.get(stateKey(state))
  if (action === undefined) return null
  const move = mirrorMove(state.turn, indexToMove(action))
  return isLegalMove(state, move) ? move : null
}

function indexToMove(action: number): Move {
  if (action < 81) {
    return {
      type: 'move',
      to: { r: Math.floor(action / 9) + 1, c: (action % 9) + 1 },
    }
  }
  const wallAction = action - 81
  const vertical = wallAction >= 64
  const offset = vertical ? wallAction - 64 : wallAction
  return {
    type: 'wall',
    wall: {
      r: Math.floor(offset / 8) + 1,
      c: (offset % 8) + 1,
      o: vertical ? 'v' : 'h',
    },
  }
}

function mirrorMove(side: Side, move: Move): Move {
  if (side === 'south') return move
  if (move.type === 'move') {
    return { type: 'move', to: { r: 10 - move.to.r, c: 10 - move.to.c } }
  }
  return {
    type: 'wall',
    wall: {
      r: 9 - move.wall.r,
      c: 9 - move.wall.c,
      o: move.wall.o,
    },
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
