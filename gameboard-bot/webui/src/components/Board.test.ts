import { createElement } from 'react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Board, { boardIntentAt, type Cell } from './Board'
import {
  type GameState,
  type Wall,
  canPlaceWall,
  initialState,
  pawnMoves,
} from '../game/engine'

afterEach(cleanup)

describe('boardIntentAt', () => {
  it('uses square centers for pawn moves', () => {
    expect(boardIntentAt(4.5, 7.5)).toEqual({
      type: 'move',
      cell: { r: 2, c: 5 },
    })
  })

  it('uses horizontal gaps for horizontal walls', () => {
    expect(boardIntentAt(4.6, 8.02)).toEqual({
      type: 'wall',
      wall: { r: 1, c: 5, o: 'h' },
    })
  })

  it('uses vertical gaps for vertical walls', () => {
    expect(boardIntentAt(5.02, 7.6)).toEqual({
      type: 'wall',
      wall: { r: 1, c: 5, o: 'v' },
    })
  })

  it('keeps near-center clicks as moves even inside a square with nearby gaps', () => {
    expect(boardIntentAt(4.55, 7.55)).toEqual({
      type: 'move',
      cell: { r: 2, c: 5 },
    })
  })

  it('requires a clear gap bias before choosing a wall', () => {
    expect(boardIntentAt(4.5, 7.76)).toEqual({
      type: 'move',
      cell: { r: 2, c: 5 },
    })
  })
})

// Board renders a single pointer surface (no explicit move/wall mode toggle):
// a click's intent — and legality — is inferred purely from where it lands,
// via boardIntentAt/eventIntent. These tests drive real clicks through that
// surface and assert on the resulting onMove/onPlaceWall calls.
describe('Board pointer clicks', () => {
  function mount(overrides: {
    state?: GameState
    legalTargets?: Cell[]
    interactive?: boolean
    canPlace?: (wall: Wall) => boolean
  } = {}) {
    const state = overrides.state ?? initialState()
    const onMove = vi.fn()
    const onPlaceWall = vi.fn()
    const { container } = render(
      createElement(Board, {
        state,
        legalTargets: overrides.legalTargets ?? pawnMoves(state, state.turn),
        interactive: overrides.interactive ?? true,
        canPlace:
          overrides.canPlace ??
          ((wall: Wall) => canPlaceWall(state, wall, state.turn)),
        onMove,
        onPlaceWall,
      }),
    )
    const boardEl = container.querySelector<HTMLDivElement>(
      'div.rounded-xl.border-4',
    )
    if (!boardEl) throw new Error('board pointer surface not found')
    // jsdom never lays elements out, so stub the rect eventIntent divides by:
    // 900x900 keeps 1 boardIntentAt unit == 100 client px, matching the
    // boardIntentAt fixtures above.
    vi.spyOn(boardEl, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      right: 900,
      bottom: 900,
      width: 900,
      height: 900,
      x: 0,
      y: 0,
      toJSON: () => {},
    } as DOMRect)
    return { boardEl, onMove, onPlaceWall }
  }

  function clickAt(el: HTMLElement, xUnits: number, yUnits: number) {
    // detail must be non-zero: Board treats detail === 0 as a synthetic
    // (keyboard-triggered) click and ignores it on the pointer surface.
    fireEvent.click(el, { detail: 1, clientX: xUnits * 100, clientY: yUnits * 100 })
  }

  it('calls onMove for a center click on a legal target', () => {
    const { boardEl, onMove, onPlaceWall } = mount()
    clickAt(boardEl, 4.5, 7.5) // south's only forward step from the start position
    expect(onMove).toHaveBeenCalledWith({ r: 2, c: 5 })
    expect(onPlaceWall).not.toHaveBeenCalled()
  })

  it('still moves when a legal-target click is only slightly closer to a gap', () => {
    const { boardEl, onMove, onPlaceWall } = mount()
    clickAt(boardEl, 4.5, 7.76)
    expect(onMove).toHaveBeenCalledWith({ r: 2, c: 5 })
    expect(onPlaceWall).not.toHaveBeenCalled()
  })

  it.each([
    { o: 'h' as const, x: 4.6, y: 8.02 },
    { o: 'v' as const, x: 5.02, y: 7.6 },
  ])(
    'calls onPlaceWall with the inferred $o orientation for a gap click',
    ({ o, x, y }) => {
      const { boardEl, onMove, onPlaceWall } = mount()
      clickAt(boardEl, x, y)
      expect(onPlaceWall).toHaveBeenCalledWith({ r: 1, c: 5, o })
      expect(onMove).not.toHaveBeenCalled()
    },
  )

  it('does not call onMove for a center click outside the legal targets', () => {
    const { boardEl, onMove, onPlaceWall } = mount()
    clickAt(boardEl, 0.5, 0.5) // cell r9,c1 — nowhere near south's legal steps
    expect(onMove).not.toHaveBeenCalled()
    expect(onPlaceWall).not.toHaveBeenCalled()
  })

  it('does not call onPlaceWall for a gap click on a wall that conflicts with an existing one', () => {
    const state: GameState = {
      ...initialState(),
      walls: [{ r: 1, c: 5, o: 'h' }],
    }
    const { boardEl, onMove, onPlaceWall } = mount({ state })
    clickAt(boardEl, 4.6, 8.02) // same anchor as the already-placed wall
    expect(onPlaceWall).not.toHaveBeenCalled()
    expect(onMove).not.toHaveBeenCalled()
  })

  it('does not call onMove or onPlaceWall when the board is not interactive', () => {
    const { boardEl, onMove, onPlaceWall } = mount({ interactive: false })
    clickAt(boardEl, 4.5, 7.5) // otherwise a legal move click
    expect(onMove).not.toHaveBeenCalled()
    expect(onPlaceWall).not.toHaveBeenCalled()
  })
})
