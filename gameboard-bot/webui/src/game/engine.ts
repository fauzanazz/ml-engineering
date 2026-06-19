// Wall Chess rules engine. Pure, no IO — safe to run on client or server.
//
// Board is 9x9, cells addressed (r, c) with r, c in 1..9.
// SOUTH starts row 1 and must reach row 9; NORTH starts row 9, reaches row 1.
//
// Walls span two cells and sit on the grid lines *between* cells:
//   - 'h' (horizontal) wall at anchor (r, c) lies between rows r and r+1,
//     covering columns c and c+1 — it blocks vertical movement.
//   - 'v' (vertical) wall at anchor (r, c) lies between cols c and c+1,
//     covering rows r and r+1 — it blocks horizontal movement.
// Anchors range 1..8 (8 internal grid lines each axis).

export type Side = 'south' | 'north'
export type Cell = { r: number; c: number }
export type Orientation = 'h' | 'v'
export type Wall = { r: number; c: number; o: Orientation }

export type GameState = {
  pawns: Record<Side, Cell>
  walls: Wall[]
  wallsLeft: Record<Side, number>
  turn: Side
  winner: Side | null
}

export type Move =
  | { type: 'move'; to: Cell }
  | { type: 'wall'; wall: Wall }

export const SIZE = 9
export const START_WALLS = 10
export const GOAL_ROW: Record<Side, number> = { south: SIZE, north: 1 }

const DIRS = [
  { dr: 1, dc: 0 },
  { dr: -1, dc: 0 },
  { dr: 0, dc: 1 },
  { dr: 0, dc: -1 },
]

export function other(side: Side): Side {
  return side === 'south' ? 'north' : 'south'
}

export function initialState(): GameState {
  const mid = Math.ceil(SIZE / 2) // 5
  return {
    pawns: { south: { r: 1, c: mid }, north: { r: SIZE, c: mid } },
    walls: [],
    wallsLeft: { south: START_WALLS, north: START_WALLS },
    turn: 'south',
    winner: null,
  }
}

export function inBounds(cell: Cell): boolean {
  return cell.r >= 1 && cell.r <= SIZE && cell.c >= 1 && cell.c <= SIZE
}

function sameCell(a: Cell, b: Cell): boolean {
  return a.r === b.r && a.c === b.c
}

// Is movement between two orthogonally-adjacent cells blocked by a wall?
export function isBlocked(walls: Wall[], from: Cell, to: Cell): boolean {
  const dr = to.r - from.r
  const dc = to.c - from.c
  if (Math.abs(dr) + Math.abs(dc) !== 1) return true // not adjacent

  if (dc === 0) {
    // vertical move — blocked by a horizontal wall on the crossed line
    const lineR = Math.min(from.r, to.r) // wall anchor row sits on this line
    return walls.some(
      (w) => w.o === 'h' && w.r === lineR && (w.c === from.c || w.c === from.c - 1),
    )
  }
  // horizontal move — blocked by a vertical wall
  const lineC = Math.min(from.c, to.c)
  return walls.some(
    (w) => w.o === 'v' && w.c === lineC && (w.r === from.r || w.r === from.r - 1),
  )
}

// Legal pawn destinations for `side` (orthogonal step, straight jump over the
// opponent, or diagonal jump when a straight jump is blocked).
export function pawnMoves(state: GameState, side: Side): Cell[] {
  const pos = state.pawns[side]
  const opp = state.pawns[other(side)]
  const out: Cell[] = []
  const push = (cell: Cell) => {
    if (inBounds(cell) && !out.some((o) => sameCell(o, cell))) out.push(cell)
  }

  for (const { dr, dc } of DIRS) {
    const step: Cell = { r: pos.r + dr, c: pos.c + dc }
    if (!inBounds(step) || isBlocked(state.walls, pos, step)) continue

    if (!sameCell(step, opp)) {
      push(step)
      continue
    }
    // opponent occupies `step` — try to jump
    const beyond: Cell = { r: step.r + dr, c: step.c + dc }
    if (inBounds(beyond) && !isBlocked(state.walls, step, beyond)) {
      push(beyond)
    } else {
      // straight jump blocked → diagonal jumps around the opponent
      const perps =
        dr === 0
          ? [
              { dr: 1, dc: 0 },
              { dr: -1, dc: 0 },
            ]
          : [
              { dr: 0, dc: 1 },
              { dr: 0, dc: -1 },
            ]
      for (const p of perps) {
        const diag: Cell = { r: step.r + p.dr, c: step.c + p.dc }
        if (inBounds(diag) && !isBlocked(state.walls, step, diag)) push(diag)
      }
    }
  }
  return out
}

// Breadth-first reachability/shortest-path ignoring the opponent pawn (standard
// for wall-validity and bot heuristics). Returns step count, or Infinity.
export function distanceToGoal(walls: Wall[], from: Cell, goalRow: number): number {
  const seen = new Set<number>()
  const key = (cell: Cell) => cell.r * 16 + cell.c
  let frontier: Cell[] = [from]
  seen.add(key(from))
  let dist = 0
  while (frontier.length) {
    const next: Cell[] = []
    for (const cell of frontier) {
      if (cell.r === goalRow) return dist
      for (const { dr, dc } of DIRS) {
        const n: Cell = { r: cell.r + dr, c: cell.c + dc }
        if (!inBounds(n) || isBlocked(walls, cell, n)) continue
        if (seen.has(key(n))) continue
        seen.add(key(n))
        next.push(n)
      }
    }
    frontier = next
    dist++
  }
  return Infinity
}

export function hasPath(walls: Wall[], from: Cell, goalRow: number): boolean {
  return distanceToGoal(walls, from, goalRow) !== Infinity
}

function wallsConflict(a: Wall, b: Wall): boolean {
  // identical center (covers H/V crossing at the same intersection)
  if (a.r === b.r && a.c === b.c) return true
  // two horizontals on the same line overlapping a column
  if (a.o === 'h' && b.o === 'h' && a.r === b.r && Math.abs(a.c - b.c) < 2) {
    return true
  }
  // two verticals on the same line overlapping a row
  if (a.o === 'v' && b.o === 'v' && a.c === b.c && Math.abs(a.r - b.r) < 2) {
    return true
  }
  return false
}

export function canPlaceWall(
  state: GameState,
  wall: Wall,
  side: Side,
): boolean {
  if (state.wallsLeft[side] <= 0) return false
  if (wall.r < 1 || wall.r > SIZE - 1 || wall.c < 1 || wall.c > SIZE - 1) {
    return false
  }
  if (wall.o !== 'h' && wall.o !== 'v') return false
  if (state.walls.some((w) => wallsConflict(w, wall))) return false

  // must not trap either player
  const walls = [...state.walls, wall]
  return (
    hasPath(walls, state.pawns.south, GOAL_ROW.south) &&
    hasPath(walls, state.pawns.north, GOAL_ROW.north)
  )
}

export function isLegalMove(state: GameState, move: Move): boolean {
  if (state.winner) return false
  if (move.type === 'move') {
    return pawnMoves(state, state.turn).some((c) => sameCell(c, move.to))
  }
  return canPlaceWall(state, move.wall, state.turn)
}

// Apply a *legal* move and return the next state. Throws on illegal input so
// callers (and the API boundary) fail loudly rather than corrupting state.
export function applyMove(state: GameState, move: Move): GameState {
  if (!isLegalMove(state, move)) {
    throw new Error('illegal move')
  }
  const side = state.turn
  const next: GameState = {
    pawns: { ...state.pawns },
    walls: state.walls,
    wallsLeft: { ...state.wallsLeft },
    turn: other(side),
    winner: null,
  }
  if (move.type === 'move') {
    next.pawns[side] = move.to
    if (move.to.r === GOAL_ROW[side]) next.winner = side
  } else {
    next.walls = [...state.walls, move.wall]
    next.wallsLeft[side] = state.wallsLeft[side] - 1
  }
  return next
}

// All legal moves for the side to move: pawn steps first, then every legal
// wall placement. This is the out-degree of a node in the state graph.
export function legalMoves(state: GameState): Move[] {
  if (state.winner) return []
  const out: Move[] = pawnMoves(state, state.turn).map((to) => ({
    type: 'move',
    to,
  }))
  if (state.wallsLeft[state.turn] > 0) {
    for (let r = 1; r <= SIZE - 1; r++) {
      for (let c = 1; c <= SIZE - 1; c++) {
        for (const o of ['h', 'v'] as const) {
          const wall: Wall = { r, c, o }
          if (canPlaceWall(state, wall, state.turn)) {
            out.push({ type: 'wall', wall })
          }
        }
      }
    }
  }
  return out
}

// Canonical identity of a state — equal keys mean the same node in the graph
// (transposition). Walls are sorted so placement order does not matter.
export function stateKey(state: GameState): string {
  const w = [...state.walls]
    .sort((a, b) => a.r - b.r || a.c - b.c || a.o.localeCompare(b.o))
    .map((x) => `${x.o}${x.r}${x.c}`)
    .join(',')
  const p = state.pawns
  return [
    state.turn[0],
    `${p.south.r}${p.south.c}`,
    `${p.north.r}${p.north.c}`,
    `${state.wallsLeft.south}${state.wallsLeft.north}`,
    w,
  ].join('|')
}
