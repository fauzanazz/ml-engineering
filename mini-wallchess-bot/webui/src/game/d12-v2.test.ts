import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { describe, expect, it, beforeAll } from 'vitest'
import init, {
  analyze_state_budgeted,
  analyze_state_budgeted_gen2,
} from '../wasm/wallchess_wasm.js'
import type { GameState, Move } from './engine'
import { BROWSER_MAX_BOT_ID, D12_GEN2_BOT_ID, D12_V2_BOT_ID, getBot } from './bots'

const D12_V2_NODES = 600_000n

beforeAll(async () => {
  const wasm = await readFile(join(process.cwd(), 'src/wasm/wallchess_wasm_bg.wasm'))
  await init({ module_or_path: wasm })
})

function bestMove(state: GameState): Move | null {
  return analyze_state_budgeted(state, 12, D12_V2_NODES, 200).move as Move | null
}

describe('bot catalog', () => {
  it('browser-max is the Gen-2 bot', () => {
    expect(BROWSER_MAX_BOT_ID).toBe(D12_GEN2_BOT_ID)
    expect(getBot(D12_GEN2_BOT_ID)).toMatchObject({
      engine: 'heuristic',
      depth: 12,
      nodeLimit: 600_000,
      gen2: true,
      label: 'Alpha-Beta · D12 Gen-2',
    })
  })

  it('keeps D12-V2 as the previous-generation bot', () => {
    expect(getBot(D12_V2_BOT_ID)).toMatchObject({
      engine: 'heuristic',
      depth: 12,
      nodeLimit: 600_000,
      label: 'Alpha-Beta · D12-V2',
    })
    // Old replays/URLs referencing the previous browser-max stay reproducible.
    expect(getBot('ab-browser-max').id).toBe(D12_V2_BOT_ID)
  })
})

describe('D12-V2 bot', () => {
  it('avoids the old replay side-step at ply 13', () => {
    const state: GameState = {
      pawns: { south: { r: 6, c: 5 }, north: { r: 3, c: 5 } },
      walls: [
        { r: 2, c: 4, o: 'h' },
        { r: 2, c: 6, o: 'h' },
        { r: 6, c: 4, o: 'h' },
      ],
      wallsLeft: { south: 8, north: 9 },
      turn: 'north',
      winner: null,
    }

    expect(bestMove(state)).toEqual({ type: 'move', to: { r: 4, c: 5 } })
  })

  it('avoids the old replay side-step at ply 15', () => {
    const state: GameState = {
      pawns: { south: { r: 6, c: 5 }, north: { r: 3, c: 4 } },
      walls: [
        { r: 2, c: 4, o: 'h' },
        { r: 2, c: 6, o: 'h' },
        { r: 3, c: 3, o: 'v' },
        { r: 6, c: 4, o: 'h' },
      ],
      wallsLeft: { south: 7, north: 9 },
      turn: 'north',
      winner: null,
    }

    expect(bestMove(state)).toEqual({ type: 'move', to: { r: 4, c: 4 } })
  })
})

describe('Gen-2 bot (deploy path)', () => {
  it('exposes the budgeted gen2 entry point', () => {
    expect(typeof analyze_state_budgeted_gen2).toBe('function')
  })

  it('returns a legal finishing move when a won race is in reach', () => {
    // North to move, one step from its goal (row 1) and the loser (south) has no
    // walls left — the exact-endgame resolver should see the win and finish.
    const state: GameState = {
      pawns: { south: { r: 4, c: 5 }, north: { r: 2, c: 5 } },
      walls: [],
      wallsLeft: { south: 0, north: 5 },
      turn: 'north',
      winner: null,
    }
    const res = analyze_state_budgeted_gen2(state, 12, D12_V2_NODES, 200)
    expect(res.move).toEqual({ type: 'move', to: { r: 1, c: 5 } })
    // North winning => south win-chance is near zero.
    expect(res.south).toBeLessThan(5)
  })
})
