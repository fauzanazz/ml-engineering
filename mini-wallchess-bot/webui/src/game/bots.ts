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
  // Optional node cap for budgeted heuristic search. Omitted means fixed depth.
  nodeLimit?: number
  // MCTS simulation budget per move (net engine only); higher = stronger/slower.
  sims?: number
}

export const D12_V2_BOT_ID = 'ab-d12-v2'
export const BROWSER_MAX_BOT_ID = D12_V2_BOT_ID

export const BOTS: BotSpec[] = [
  {
    id: D12_V2_BOT_ID,
    label: 'Alpha-Beta · D12-V2',
    engine: 'heuristic',
    depth: 12,
    nodeLimit: 600_000,
  },
  { id: 'ab-fast', label: 'Alpha-Beta · fast (d2)', engine: 'heuristic', depth: 2 },
  { id: 'ab', label: 'Alpha-Beta (d3)', engine: 'heuristic', depth: 3 },
  { id: 'ab-deep', label: 'Alpha-Beta · strong (d4)', engine: 'heuristic', depth: 4 },
  { id: 'ab-expert', label: 'Alpha-Beta · expert (d5)', engine: 'heuristic', depth: 5 },
  { id: 'ab-master', label: 'Alpha-Beta · master (d6)', engine: 'heuristic', depth: 6 },
  { id: 'ab-grandmaster', label: 'Alpha-Beta · grandmaster (d7)', engine: 'heuristic', depth: 7 },
  { id: 'ab-d8', label: 'Alpha-Beta · d8', engine: 'heuristic', depth: 8 },
  { id: 'ab-d9', label: 'Alpha-Beta · d9', engine: 'heuristic', depth: 9 },
  { id: 'ab-d10', label: 'Alpha-Beta · d10', engine: 'heuristic', depth: 10 },
  { id: 'net-fast', label: 'MCTS Net · fast (60)', engine: 'net', sims: 60 },
  { id: 'net', label: 'MCTS Net (200)', engine: 'net', sims: 200 },
  { id: 'net-strong', label: 'MCTS Net · strong (600)', engine: 'net', sims: 600 },
]

const BOT_BY_ID = new Map(BOTS.map((b) => [b.id, b]))
const BOT_ALIASES = new Map<string, string>([
  ['ab-browser-max', D12_V2_BOT_ID],
])

// Resolve a bot id to its spec, falling back to the first catalog entry so a
// stale id (e.g. from an old replay/URL) never crashes the arena loop.
export function getBot(id: string): BotSpec {
  return BOT_BY_ID.get(BOT_ALIASES.get(id) ?? id) ?? BOTS[0]
}
