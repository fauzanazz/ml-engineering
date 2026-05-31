import {
  type Cell,
  type GameState,
  type Side,
  type Wall,
  SIZE,
  START_WALLS,
} from './engine'

// Pure parser for the bot endpoint's trust boundary — the request body is
// untrusted, so every field is parsed and range-checked rather than cast.
// Kept separate from the server-fn module so it can be unit-tested in plain
// node without evaluating `createServerFn`.

function parseCell(v: unknown, label: string): Cell {
  if (typeof v !== 'object' || v === null) throw new Error(`${label}: not an object`)
  const { r, c } = v as Record<string, unknown>
  if (!Number.isInteger(r) || (r as number) < 1 || (r as number) > SIZE) {
    throw new Error(`${label}.r out of range`)
  }
  if (!Number.isInteger(c) || (c as number) < 1 || (c as number) > SIZE) {
    throw new Error(`${label}.c out of range`)
  }
  return { r: r as number, c: c as number }
}

function parseWall(v: unknown, i: number): Wall {
  if (typeof v !== 'object' || v === null) throw new Error(`walls[${i}]: not an object`)
  const { r, c, o } = v as Record<string, unknown>
  if (!Number.isInteger(r) || (r as number) < 1 || (r as number) > SIZE - 1) {
    throw new Error(`walls[${i}].r out of range`)
  }
  if (!Number.isInteger(c) || (c as number) < 1 || (c as number) > SIZE - 1) {
    throw new Error(`walls[${i}].c out of range`)
  }
  if (o !== 'h' && o !== 'v') throw new Error(`walls[${i}].o invalid`)
  return { r: r as number, c: c as number, o }
}

function parseSide(v: unknown, label: string): Side {
  if (v !== 'south' && v !== 'north') throw new Error(`${label}: invalid side`)
  return v
}

export function parseGameState(raw: unknown): GameState {
  if (typeof raw !== 'object' || raw === null) throw new Error('state: not an object')
  const s = raw as Record<string, unknown>

  const pawns = s.pawns as Record<string, unknown> | undefined
  if (!pawns) throw new Error('state.pawns missing')

  const walls = Array.isArray(s.walls) ? s.walls : []
  if (walls.length > START_WALLS * 2) throw new Error('too many walls')

  const wl = s.wallsLeft as Record<string, unknown> | undefined
  if (!wl) throw new Error('state.wallsLeft missing')
  const clampWalls = (v: unknown, label: string): number => {
    if (!Number.isInteger(v) || (v as number) < 0 || (v as number) > START_WALLS) {
      throw new Error(`${label} out of range`)
    }
    return v as number
  }

  return {
    pawns: {
      south: parseCell(pawns.south, 'pawns.south'),
      north: parseCell(pawns.north, 'pawns.north'),
    },
    walls: walls.map(parseWall),
    wallsLeft: {
      south: clampWalls(wl.south, 'wallsLeft.south'),
      north: clampWalls(wl.north, 'wallsLeft.north'),
    },
    turn: parseSide(s.turn, 'turn'),
    winner: s.winner == null ? null : parseSide(s.winner, 'winner'),
  }
}
