import type { GameState, Move, Side } from './engine'

export type ReplayEngine = 'human' | 'heuristic' | 'net'

export type ReplayFrame = {
  ply: number
  turn: Side
  state: GameState  // state BEFORE the action
  action: Move
  win_prob_south: number
}

export type ReplayMeta = {
  id: string
  mode: 'bot' | 'friend' | 'arena'
  engines: { south: ReplayEngine; north: ReplayEngine }
  winner: Side | null
  ply_count: number
  started_at: string
}

export type ReplayRecord = ReplayMeta & { frames: ReplayFrame[] }

const STORAGE_KEY = 'wallchess_replays'
const MAX_REPLAYS = 30

export function makeId(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}

export function saveReplay(record: ReplayRecord): void {
  try {
    const all = listReplays()
    const updated = [record, ...all.filter((r) => r.id !== record.id)].slice(0, MAX_REPLAYS)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  } catch {
    // storage full or unavailable
  }
}

export function listReplays(): ReplayRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as ReplayRecord[]) : []
  } catch {
    return []
  }
}

export function deleteReplay(id: string): void {
  try {
    const updated = listReplays().filter((r) => r.id !== id)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  } catch {}
}

// Each line = one JSON object (JSONL / NDJSON).
// Line 0: metadata. Lines 1..N: one frame per ply.
// Each frame is a complete (state, action, player, outcome) tuple for training.
export function toJsonl(record: ReplayRecord): string {
  const lines: string[] = [
    JSON.stringify({
      type: 'meta',
      id: record.id,
      mode: record.mode,
      engines: record.engines,
      winner: record.winner,
      ply_count: record.ply_count,
      started_at: record.started_at,
    }),
  ]

  for (const f of record.frames) {
    lines.push(
      JSON.stringify({
        type: 'frame',
        ply: f.ply,
        turn: f.turn,
        state: f.state,
        action: f.action,
        win_prob_south: f.win_prob_south,
        // outcome from the perspective of the player who made this move
        outcome: record.winner
          ? record.winner === f.turn
            ? 'win'
            : 'loss'
          : 'draw',
      }),
    )
  }

  return lines.join('\n')
}

export function downloadReplay(record: ReplayRecord): void {
  const content = toJsonl(record)
  const blob = new Blob([content], { type: 'application/x-ndjson' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const date = record.started_at.slice(0, 10)
  a.download = `wallchess-${record.mode}-${date}-${record.id.slice(-6)}.jsonl`
  a.click()
  URL.revokeObjectURL(url)
}
