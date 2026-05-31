// Wall Chess bot. Deterministic (no randomness) so behaviour is testable.
//
// Strategy (intentionally simple, first pass):
//   1. If the opponent is at least as close to their goal as we are to ours,
//      look for a wall that delays them the most without hurting our own path
//      more than one step — place it if it gains ground.
//   2. Otherwise (or if no useful wall) step along our own shortest path.

import {
  type GameState,
  type Move,
  type Wall,
  GOAL_ROW,
  SIZE,
  canPlaceWall,
  distanceToGoal,
  other,
  pawnMoves,
} from './engine'

function bestAdvanceMove(state: GameState, side: Side): Move | null {
  const goal = GOAL_ROW[side]
  const moves = pawnMoves(state, side)
  if (moves.length === 0) return null

  let best = moves[0]
  let bestScore = Infinity
  for (const to of moves) {
    const dist = distanceToGoal(state.walls, to, goal)
    // tie-break: fewer rows from goal, then closer to centre column
    const rowGap = Math.abs(goal - to.r)
    const colGap = Math.abs(to.c - Math.ceil(SIZE / 2))
    const score = dist * 1000 + rowGap * 10 + colGap
    if (score < bestScore) {
      bestScore = score
      best = to
    }
  }
  return { type: 'move', to: best }
}

type Side = 'south' | 'north'

function bestWall(state: GameState, side: Side): { wall: Wall; gain: number } | null {
  if (state.wallsLeft[side] <= 0) return null
  const opp = other(side)
  const myGoal = GOAL_ROW[side]
  const oppGoal = GOAL_ROW[opp]
  const myDist = distanceToGoal(state.walls, state.pawns[side], myGoal)
  const oppDist = distanceToGoal(state.walls, state.pawns[opp], oppGoal)

  let pick: { wall: Wall; gain: number } | null = null
  for (let r = 1; r <= SIZE - 1; r++) {
    for (let c = 1; c <= SIZE - 1; c++) {
      for (const o of ['h', 'v'] as const) {
        const wall: Wall = { r, c, o }
        if (!canPlaceWall(state, wall, side)) continue
        const walls = [...state.walls, wall]
        const newOpp = distanceToGoal(walls, state.pawns[opp], oppGoal)
        const newMy = distanceToGoal(walls, state.pawns[side], myGoal)
        // delay the opponent without lengthening our own path by more than 1
        const gain = newOpp - oppDist - Math.max(0, newMy - myDist)
        if (gain <= 0) continue
        if (!pick || gain > pick.gain) pick = { wall, gain }
      }
    }
  }
  return pick
}

export function chooseMove(state: GameState): Move {
  if (state.winner) throw new Error('game already won')
  const side = state.turn
  const advance = bestAdvanceMove(state, side)

  const opp = other(side)
  const myDist = distanceToGoal(state.walls, state.pawns[side], GOAL_ROW[side])
  const oppDist = distanceToGoal(state.walls, state.pawns[opp], GOAL_ROW[opp])

  // only spend a wall defensively when not ahead
  if (oppDist <= myDist) {
    const wall = bestWall(state, side)
    if (wall && wall.gain >= 1) return { type: 'wall', wall: wall.wall }
  }

  if (advance) return advance
  // no pawn move available (fully boxed) — fall back to any legal wall
  const wall = bestWall(state, side)
  if (wall) return { type: 'wall', wall: wall.wall }
  throw new Error('no legal move')
}
