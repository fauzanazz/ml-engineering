// Graph Traveler — walk the game's state graph one node at a time.
//
//   • current node   : its board state + stats + win split
//   • connected nodes : every legal successor (the node's out-edges)
//   • galaxy view     : force-directed map of every node discovered so far
//
// The full state space is astronomically large (~10^42 legal positions), so the
// galaxy only ever shows the *explored* subgraph — nodes you have visited. A
// live counter shows how many have been discovered.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import MiniBoard from '../components/MiniBoard'
import ForceGraph, {
  type GraphEdge,
  type GraphNode,
} from '../components/ForceGraph'
import { analyzePosition, topGraph, topMoves } from '../game/api'
import {
  type GameState,
  type Move,
  applyMove,
  initialState,
  stateKey,
} from '../game/engine'

export const Route = createFileRoute('/graph')({ component: GraphTraveler })

// ~ legal-position count for 9x9 Quoridor (Glendenning, order of magnitude).
const STATE_SPACE = '≈ 10⁴²'
// Search depth used to rank/prune successors. depth-1 is the precompute/CPU
// balance sweet spot: ranking each node with a 1-ply search prunes blunders
// just as well for the graph view but is ~100x cheaper than depth-2 (measured:
// depth-12 graph ~100ms vs ~10s), so the whole graph generates near-instantly.
const GRAPH_DEPTH = 1
const TOP_K = 5 // keep at most this many strong moves per node
// Drop moves worse than the best by > this (centi-steps, ≈100 per board step).
// ~400 lets genuinely-different strong plans branch while still cutting
// blunders; tighter values (≤150) collapse the graph to one straight line.
const BLUNDER_MARGIN = 400
// "Generate to safe limit": the whole pruned graph is built in one Rust/WASM
// call up to this depth, capped at SAFE_NODE_CAP nodes (the memory ceiling).
// With rank_depth=1 the depth-12 graph (~740 states) builds in ~100ms, so this
// stays sub-second. The node cap is the hard stop so the graph never outgrows
// its budget regardless of depth.
const SAFE_DEPTH = 12
const SAFE_NODE_CAP = 2000

type NodeRec = {
  key: string
  state: GameState
  ply: number
  south?: number // win % for SOUTH (undefined until analyzed)
  children: Set<string>
  parents: Set<string>
  via: Map<string, Move> // childKey -> the move that reaches it (edge label)
}

function moveLabel(m: Move): string {
  if (m.type === 'move') return `move → (${m.to.r},${m.to.c})`
  return `wall ${m.wall.o.toUpperCase()} @ (${m.wall.r},${m.wall.c})`
}

// Colour a node by who it favours: SOUTH (lagoon) ↔ unknown (grey) ↔ NORTH.
function nodeColor(south?: number): string {
  if (south == null) return '#8a8178'
  return south >= 50 ? 'var(--lagoon)' : 'var(--pawn-north)'
}

function GraphTraveler() {
  const [nodes, setNodes] = useState<Map<string, NodeRec>>(() => {
    const root = initialState()
    const k = stateKey(root)
    return new Map([
      [
        k,
        {
          key: k,
          state: root,
          ply: 0,
          children: new Set(),
          parents: new Set(),
          via: new Map(),
        },
      ],
    ])
  })
  const [currentKey, setCurrentKey] = useState(() => stateKey(initialState()))
  const expanded = useRef<Set<string>>(new Set())
  const [generating, setGenerating] = useState(false)
  const [genInfo, setGenInfo] = useState<string | null>(null)

  const current = nodes.get(currentKey)!
  const neighbors = useMemo(
    () =>
      [...current.children]
        .map((k) => nodes.get(k))
        .filter((n): n is NodeRec => !!n),
    [current, nodes],
  )

  // Flatten the discovered subgraph into the force-graph's node/edge arrays.
  const graphNodes = useMemo<GraphNode[]>(
    () =>
      [...nodes.values()].map((n) => ({
        key: n.key,
        ply: n.ply,
        south: n.south,
        label: n.key === currentKey ? `ply ${n.ply}` : undefined,
      })),
    [nodes, currentKey],
  )
  const graphEdges = useMemo<GraphEdge[]>(() => {
    const seen = new Set<string>()
    const out: GraphEdge[] = []
    for (const n of nodes.values()) {
      for (const c of n.children) {
        const id = n.key < c ? `${n.key}__${c}` : `${c}__${n.key}`
        if (seen.has(id) || !nodes.has(c)) continue
        seen.add(id)
        out.push({ a: n.key, b: c })
      }
    }
    return out
  }, [nodes])

  // Expand the current node: discover its *pruned* strong successors (top-K,
  // useless walls + blunders removed by the Rust engine), wire edges, and fill
  // win splits. Children come back pre-scored, so no separate analysis pass.
  useEffect(() => {
    let cancelled = false
    const node = nodes.get(currentKey)
    if (!node) return
    ;(async () => {
      // 1) structural expansion (once per node) via the pruned ranker
      if (!expanded.current.has(currentKey)) {
        expanded.current.add(currentKey)
        const ranked = await topMoves(
          node.state,
          GRAPH_DEPTH,
          TOP_K,
          BLUNDER_MARGIN,
        )
        if (cancelled) return
        setNodes((prev) => {
          const next = new Map(prev)
          const cur = { ...next.get(currentKey)! }
          cur.children = new Set(cur.children)
          cur.via = new Map(cur.via)
          for (const { move, south } of ranked) {
            const child = applyMove(node.state, move)
            const ck = stateKey(child)
            const existing = next.get(ck)
            if (existing) {
              const e = { ...existing, parents: new Set(existing.parents) }
              e.parents.add(currentKey)
              if (e.south == null) e.south = south
              next.set(ck, e)
            } else {
              next.set(ck, {
                key: ck,
                state: child,
                ply: cur.ply + 1,
                south,
                children: new Set(),
                parents: new Set([currentKey]),
                via: new Map(),
              })
            }
            cur.children.add(ck)
            cur.via.set(ck, move)
          }
          next.set(currentKey, cur)
          return next
        })
      }

      // 2) score the focused node itself (children already carry their score)
      if (nodes.get(currentKey)?.south == null) {
        const a = await analyzePosition(node.state, GRAPH_DEPTH)
        if (cancelled) return
        setNodes((prev) => {
          const next = new Map(prev)
          const n = next.get(currentKey)
          if (n) next.set(currentKey, { ...n, south: a.south })
          return next
        })
      }
    })()
    return () => {
      cancelled = true
    }
    // re-run when we move to a new node or its children set changes
  }, [currentKey, nodes])

  const travel = useCallback((k: string) => setCurrentKey(k), [])

  // Generate the whole pruned graph up to a safe depth in ONE Rust/WASM call —
  // the BFS, dedup, scoring, and pruning all happen in the engine (shared
  // transposition table), so this is ~100x faster than expanding from JS. We
  // just rebuild the node map from the returned nodes + edges.
  const generateFull = useCallback(async () => {
    setGenerating(true)
    setGenInfo(null)
    try {
      const root = initialState()
      const t0 = performance.now()
      const g = await topGraph(
        root,
        GRAPH_DEPTH,
        TOP_K,
        BLUNDER_MARGIN,
        SAFE_DEPTH,
        SAFE_NODE_CAP,
      )
      const ms = Math.round(performance.now() - t0)

      const map = new Map<string, NodeRec>()
      for (const n of g.nodes) {
        map.set(n.key, {
          key: n.key,
          state: n.state,
          ply: n.ply,
          south: n.south,
          children: new Set(),
          parents: new Set(),
          via: new Map(),
        })
      }
      for (const e of g.edges) {
        const from = map.get(e.from)
        const to = map.get(e.to)
        if (!from || !to) continue
        from.children.add(e.to)
        from.via.set(e.to, e.move)
        to.parents.add(e.from)
      }
      const rootKey = stateKey(root)
      for (const k of map.keys()) expanded.current.add(k)
      setNodes(map)
      setCurrentKey(rootKey)
      setGenInfo(
        g.capped
          ? `Stopped at node cap (${map.size} states, ${ms} ms). Raise the cap for more.`
          : `Generated ${map.size} states to depth ${SAFE_DEPTH} in ${ms} ms.`,
      )
    } finally {
      setGenerating(false)
    }
  }, [])

  const reset = useCallback(() => {
    const root = initialState()
    const k = stateKey(root)
    expanded.current = new Set()
    setNodes(
      new Map([
        [
        k,
        {
          key: k,
          state: root,
          ply: 0,
          children: new Set(),
          parents: new Set(),
          via: new Map(),
        },
      ],
      ]),
    )
    setCurrentKey(k)
    setGenInfo(null)
  }, [])

  const south = current.south
  const discovered = nodes.size

  return (
    <main className="page-wrap px-4 py-8">
      <div className="mx-auto max-w-6xl">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="island-kicker mb-1">State graph</p>
            <h1 className="display-title text-3xl font-bold text-[var(--sea-ink)]">
              Graph Traveler
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-[var(--sea-ink-soft)]">
            <span>
              discovered{' '}
              <span className="font-bold text-[var(--sea-ink)]">{discovered}</span>{' '}
              / <span className="font-mono">{STATE_SPACE}</span> nodes
            </span>
            <button
              type="button"
              onClick={generateFull}
              disabled={generating}
              className="rounded-full bg-[var(--btn-primary-bg)] px-3 py-1.5 text-xs font-bold text-[var(--btn-primary-fg)] transition hover:opacity-90 disabled:opacity-50"
            >
              {generating ? 'Generating…' : `Generate graph (depth ${SAFE_DEPTH})`}
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-xs font-semibold text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
            >
              Reset to root
            </button>
          </div>
        </div>

        {genInfo && (
          <p className="mb-3 text-xs font-semibold text-[var(--sea-ink-soft)]">
            {genInfo}
          </p>
        )}

        <div className="grid gap-5 lg:grid-cols-[260px_1fr]">
          {/* ── current node ── */}
          <section className="island-shell rounded-2xl p-4">
            <p className="island-kicker mb-2">Current node</p>
            <MiniBoard state={current.state} size={220} />
            <dl className="mt-3 space-y-1 text-sm">
              <Row k="depth (ply)" v={`${current.ply}`} />
              <Row k="turn" v={current.state.turn.toUpperCase()} />
              <Row
                k="walls S / N"
                v={`${current.state.wallsLeft.south} / ${current.state.wallsLeft.north}`}
              />
              <Row
                k="out-degree"
                v={`${current.children.size || (current.state.winner ? 0 : '…')}`}
              />
              <Row
                k="winner"
                v={current.state.winner ? current.state.winner.toUpperCase() : '—'}
              />
            </dl>
            {/* win split */}
            <div className="mt-3">
              <div className="mb-1 flex justify-between text-xs font-semibold text-[var(--sea-ink)]">
                <span>S {south ?? '…'}</span>
                <span>N {south == null ? '…' : 100 - south}</span>
              </div>
              <div className="flex h-3 overflow-hidden rounded-full border border-[var(--line)] bg-[var(--chip-bg)]">
                <div
                  className="h-full bg-[var(--lagoon)] transition-[width] duration-300"
                  style={{ width: `${south ?? 50}%` }}
                />
                <div
                  className="h-full bg-[var(--pawn-north)] transition-[width] duration-300"
                  style={{ width: `${south == null ? 50 : 100 - south}%` }}
                />
              </div>
            </div>
          </section>

          {/* ── galaxy + connected ── */}
          <div className="flex flex-col gap-5">
            <section className="py-2">
              <div className="mb-2">
                <p className="island-kicker">Galaxy view — force graph</p>
                <p className="mt-1 text-xs text-[var(--sea-ink)]">
                  drag node · drag bg to orbit · wheel to zoom · click to fly
                </p>
              </div>
              <ForceGraph
                nodes={graphNodes}
                edges={graphEdges}
                currentKey={currentKey}
                onTravel={travel}
                height={420}
              />
            </section>

            <section className="py-2">
              <p className="island-kicker mb-2">
                Connected nodes ({neighbors.length})
              </p>
              {current.state.winner ? (
                <p className="text-sm text-[var(--sea-ink-soft)]">
                  Terminal node — {current.state.winner.toUpperCase()} has won. No
                  out-edges.
                </p>
              ) : (
                <div className="grid max-h-64 grid-cols-1 gap-1.5 overflow-y-auto sm:grid-cols-2">
                  {neighbors.map((n, i) => {
                    // recover the move that leads here for a readable label
                    const mv = current.via.get(n.key) ?? null
                    return (
                      <button
                        key={n.key}
                        type="button"
                        onClick={() => travel(n.key)}
                        className="flex items-center justify-between gap-2 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-2 text-left text-xs font-semibold text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
                      >
                        <span className="truncate">
                          {mv ? moveLabel(mv) : `node ${i}`}
                        </span>
                        <span
                          className="ml-2 inline-flex h-2.5 w-2.5 flex-shrink-0 rounded-full"
                          style={{ background: nodeColor(n.south) }}
                          title={n.south == null ? 'scoring…' : `S ${n.south}`}
                        />
                      </button>
                    )
                  })}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </main>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-[var(--sea-ink-soft)]">{k}</dt>
      <dd className="font-semibold text-[var(--sea-ink)]">{v}</dd>
    </div>
  )
}

