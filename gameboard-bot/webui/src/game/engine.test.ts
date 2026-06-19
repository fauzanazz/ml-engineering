import { describe, expect, it } from 'vitest'
import {
  type GameState,
  applyMove,
  canPlaceWall,
  distanceToGoal,
  initialState,
  isBlocked,
  isLegalMove,
  pawnMoves,
} from './engine'
import { chooseMove } from './bot'
import { parseGameState } from './validate'

describe('initialState', () => {
  it('places pawns centre of back rows with 10 walls each', () => {
    const s = initialState()
    expect(s.pawns.south).toEqual({ r: 1, c: 5 })
    expect(s.pawns.north).toEqual({ r: 9, c: 5 })
    expect(s.wallsLeft).toEqual({ south: 10, north: 10 })
    expect(s.turn).toBe('south')
  })
})

describe('pawnMoves', () => {
  it('south in centre has 4 orthogonal moves', () => {
    const s: GameState = {
      ...initialState(),
      pawns: { south: { r: 5, c: 5 }, north: { r: 9, c: 1 } },
    }
    expect(pawnMoves(s, 'south')).toHaveLength(4)
  })

  it('corner pawn has 2 moves', () => {
    const s = initialState() // south at r1c5 → up/left/right (down off-board)
    expect(pawnMoves(s, 'south')).toHaveLength(3)
  })

  it('jumps straight over an adjacent opponent', () => {
    const s: GameState = {
      ...initialState(),
      pawns: { south: { r: 5, c: 5 }, north: { r: 6, c: 5 } },
    }
    const moves = pawnMoves(s, 'south')
    expect(moves).toContainEqual({ r: 7, c: 5 }) // jumped over north
    expect(moves).not.toContainEqual({ r: 6, c: 5 }) // can't land on opponent
  })

  it('jumps diagonally when a wall is behind the opponent', () => {
    const s: GameState = {
      ...initialState(),
      pawns: { south: { r: 5, c: 5 }, north: { r: 6, c: 5 } },
      walls: [{ r: 6, c: 5, o: 'h' }], // wall behind north (between r6 and r7)
    }
    const moves = pawnMoves(s, 'south')
    expect(moves).not.toContainEqual({ r: 7, c: 5 }) // straight jump blocked
    expect(moves).toContainEqual({ r: 6, c: 4 })
    expect(moves).toContainEqual({ r: 6, c: 6 })
  })
})

describe('walls', () => {
  it('a horizontal wall blocks the vertical step it covers', () => {
    const walls = [{ r: 3, c: 5, o: 'h' as const }]
    expect(isBlocked(walls, { r: 3, c: 5 }, { r: 4, c: 5 })).toBe(true)
    expect(isBlocked(walls, { r: 3, c: 6 }, { r: 4, c: 6 })).toBe(true)
    expect(isBlocked(walls, { r: 3, c: 7 }, { r: 4, c: 7 })).toBe(false)
  })

  it('rejects overlapping same-orientation walls', () => {
    const s: GameState = { ...initialState(), walls: [{ r: 3, c: 5, o: 'h' }] }
    expect(canPlaceWall(s, { r: 3, c: 6, o: 'h' }, 'south')).toBe(false) // overlap
    expect(canPlaceWall(s, { r: 3, c: 5, o: 'v' }, 'south')).toBe(false) // crosses
    expect(canPlaceWall(s, { r: 3, c: 7, o: 'h' }, 'south')).toBe(true)
  })

  it('rejects a wall that fully traps a player', () => {
    // box south (at r1c5) in: walls on all reachable sides leaving no goal path
    const s: GameState = {
      ...initialState(),
      pawns: { south: { r: 1, c: 1 }, north: { r: 9, c: 9 } },
      walls: [{ r: 1, c: 1, o: 'v' }], // blocks rightward escape rows 1-2
    }
    // sealing the upward path out of the corner should be illegal
    expect(canPlaceWall(s, { r: 1, c: 1, o: 'h' }, 'south')).toBe(false)
  })
})

describe('distanceToGoal', () => {
  it('open board: south at r1 is 8 steps from row 9', () => {
    expect(distanceToGoal([], { r: 1, c: 5 }, 9)).toBe(8)
  })
})

describe('applyMove', () => {
  it('flips the turn and detects a win', () => {
    const s: GameState = {
      ...initialState(),
      pawns: { south: { r: 8, c: 5 }, north: { r: 2, c: 1 } },
    }
    const next = applyMove(s, { type: 'move', to: { r: 9, c: 5 } })
    expect(next.winner).toBe('south')
    expect(next.turn).toBe('north')
  })

  it('throws on an illegal move', () => {
    const s = initialState()
    expect(() => applyMove(s, { type: 'move', to: { r: 5, c: 5 } })).toThrow()
  })

  it('decrements walls on placement', () => {
    const s = initialState()
    const next = applyMove(s, { type: 'wall', wall: { r: 4, c: 4, o: 'h' } })
    expect(next.wallsLeft.south).toBe(9)
    expect(next.walls).toHaveLength(1)
  })
})

describe('bot', () => {
  it('returns a legal move from the initial position', () => {
    const s = initialState()
    const move = chooseMove(s)
    expect(isLegalMove(s, move)).toBe(true)
  })

  it('advances toward its goal when unobstructed', () => {
    const s = initialState()
    const move = chooseMove(s)
    expect(move).toEqual({ type: 'move', to: { r: 2, c: 5 } })
  })

  it('plays a full game to a winner without illegal moves', () => {
    let s = initialState()
    for (let i = 0; i < 400 && !s.winner; i++) {
      const move = chooseMove(s)
      expect(isLegalMove(s, move)).toBe(true)
      s = applyMove(s, move)
    }
    expect(s.winner).not.toBeNull()
  })
})

describe('parseGameState', () => {
  it('accepts a valid state', () => {
    expect(parseGameState(initialState())).toEqual(initialState())
  })

  it('rejects out-of-range pawns', () => {
    expect(() => parseGameState({ ...initialState(), pawns: { south: { r: 0, c: 5 }, north: { r: 9, c: 5 } } })).toThrow()
  })

  it('rejects a bad turn value', () => {
    expect(() => parseGameState({ ...initialState(), turn: 'east' })).toThrow()
  })
})
