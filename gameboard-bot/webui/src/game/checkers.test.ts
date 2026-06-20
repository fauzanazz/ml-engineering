import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { beforeAll, describe, expect, it } from 'vitest'
import init, {
  ck_analyze,
  ck_apply,
  ck_initial,
  ck_legal_moves,
  ck_status,
} from '../wasm/wallchess_wasm.js'
import type { CkAnalysis, CkMove, CkState, CkStatus } from './checkers'
import { indexToRC, initialState, isDark, rcToIndex, status as readStatus } from './checkers'

beforeAll(async () => {
  const wasm = await readFile(
    join(process.cwd(), 'src/wasm/wallchess_wasm_bg.wasm'),
  )
  await init({ module_or_path: wasm })
})

const sorted = (a: number[]) => [...a].sort((x, y) => x - y)

describe('board geometry', () => {
  it('round-trips every dark-square index and they are all dark', () => {
    for (let i = 0; i < 50; i++) {
      const { r, c } = indexToRC(i)
      expect(isDark(r, c)).toBe(true)
      expect(rcToIndex(r, c)).toBe(i)
    }
  })

  it('matches the WASM initial position', () => {
    const wasmInit = ck_initial() as CkState
    const tsInit = initialState()
    expect(sorted(wasmInit.white)).toEqual(sorted(tsInit.white))
    expect(sorted(wasmInit.black)).toEqual(sorted(tsInit.black))
    expect(wasmInit.kings).toEqual([])
    expect(wasmInit.stm).toBe('white')
    // White men occupy the bottom four rows (indices 30..49).
    expect(sorted(wasmInit.white)).toEqual(
      Array.from({ length: 20 }, (_, i) => 30 + i),
    )
  })
})

describe('opening position', () => {
  const start = initialState()

  it('has exactly 9 quiet opening moves (perft(1) oracle)', () => {
    const moves = ck_legal_moves(start) as CkMove[]
    expect(moves).toHaveLength(9)
    for (const m of moves) {
      expect(m.captured).toEqual([])
      expect(start.white).toContain(m.from)
    }
  })

  it('is non-terminal with no winner', async () => {
    // Raw WASM serializes the `None` winner as `undefined`.
    const raw = ck_status(start) as CkStatus
    expect(raw.terminal).toBe(false)
    expect(raw.draw).toBe(false)
    expect(raw.winner ?? null).toBeNull()
    // The bridge normalizes it to the `null` app contract.
    expect(await readStatus(start)).toEqual({
      terminal: false,
      winner: null,
      draw: false,
    })
  })

  it('analyzes to a legal move with a 0..100 split', () => {
    const a = ck_analyze(start, 8, 0n, 200) as CkAnalysis
    expect(a.move).not.toBeNull()
    expect(a.white + a.black).toBe(100)
    expect(a.depth).toBeGreaterThanOrEqual(1)
    const legal = ck_legal_moves(start) as CkMove[]
    expect(legal.some((m) => m.from === a.move?.from && m.to === a.move?.to)).toBe(
      true,
    )
  })

  it('applies an opening move and flips the side', () => {
    const moves = ck_legal_moves(start) as CkMove[]
    const next = ck_apply(start, moves[0]) as CkState
    expect(next.stm).toBe('black')
    expect(next.white).toHaveLength(20)
    expect(next.black).toHaveLength(20)
    expect(next.white).toContain(moves[0].to)
    expect(next.white).not.toContain(moves[0].from)
  })
})

describe('mandatory maximal capture (fidelity)', () => {
  // White man on 22 (row 4) with a Black man on 17 (up-left) and 11 empty
  // beyond: the only legal move is the jump 22→11, and quiet moves are illegal.
  const pos: CkState = {
    white: [22],
    black: [17],
    kings: [],
    stm: 'white',
    idle: 0,
  }

  it('returns only the forced capture', () => {
    const moves = ck_legal_moves(pos) as CkMove[]
    expect(moves).toEqual([{ from: 22, to: 11, captured: [17] }])
  })

  it('rejects a quiet move while a capture exists', () => {
    expect(() => ck_apply(pos, { from: 22, to: 18, captured: [] })).toThrow(
      /illegal move/,
    )
  })

  it('removes the captured man and ends the game (Black has no piece)', () => {
    const moves = ck_legal_moves(pos) as CkMove[]
    const next = ck_apply(pos, moves[0]) as CkState
    expect(next.black).toEqual([])
    expect(next.white).toEqual([11])
    const s = ck_status(next) as CkStatus
    expect(s.terminal).toBe(true)
    expect(s.winner).toBe('white')
  })
})
