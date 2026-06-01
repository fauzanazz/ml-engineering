// Catalog of selectable bot variations. Arena (and any bot-vs-bot UI) picks a
// spec per side from this list, so the same engine can appear multiple times at
// different strengths and any matchup is possible — including a bot vs itself.
import type { BotEngine } from './api'

export type BotSpec = {
  id: string
  label: string
  engine: BotEngine
  // Heuristic search depth in plies (alpha-beta engine only).
  depth?: number
  // MCTS simulation budget per move (net engine only); higher = stronger/slower.
  sims?: number
}

export const BOTS: BotSpec[] = [
  { id: 'ab-fast', label: 'Alpha-Beta · fast (d2)', engine: 'heuristic', depth: 2 },
  { id: 'ab', label: 'Alpha-Beta (d3)', engine: 'heuristic', depth: 3 },
  { id: 'ab-deep', label: 'Alpha-Beta · deep (d4)', engine: 'heuristic', depth: 4 },
  { id: 'net-fast', label: 'MCTS Net · fast (60)', engine: 'net', sims: 60 },
  { id: 'net', label: 'MCTS Net (200)', engine: 'net', sims: 200 },
  { id: 'net-strong', label: 'MCTS Net · strong (600)', engine: 'net', sims: 600 },
]

const BOT_BY_ID = new Map(BOTS.map((b) => [b.id, b]))

// Resolve a bot id to its spec, falling back to the first catalog entry so a
// stale id (e.g. from an old replay/URL) never crashes the arena loop.
export function getBot(id: string): BotSpec {
  return BOT_BY_ID.get(id) ?? BOTS[0]
}
