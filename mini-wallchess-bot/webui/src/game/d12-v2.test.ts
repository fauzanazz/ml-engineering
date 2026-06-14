import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { describe, expect, it, beforeAll } from 'vitest'
import init, { analyze_state_budgeted } from '../wasm/wallchess_wasm.js'
import type { GameState, Move } from './engine'
import { BROWSER_MAX_BOT_ID, D12_V2_BOT_ID, getBot } from './bots'

const D12_V2_NODES = 600_000n

beforeAll(async () => {
  const wasm = await readFile(join(process.cwd(), 'src/wasm/wallchess_wasm_bg.wasm'))
  await init({ module_or_path: wasm })
})

function bestMove(state: GameState): Move | null {
  return analyze_state_budgeted(state, 12, D12_V2_NODES, 200).move as Move | null
}

describe('D12-V2 bot', () => {
  it('is the browser-max catalog bot', () => {
    expect(BROWSER_MAX_BOT_ID).toBe(D12_V2_BOT_ID)
    expect(getBot(D12_V2_BOT_ID)).toMatchObject({
      engine: 'heuristic',
      depth: 12,
      nodeLimit: 600_000,
      label: 'Alpha-Beta · D12-V2',
    })
    expect(getBot('ab-browser-max').id).toBe(D12_V2_BOT_ID)
  })

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
