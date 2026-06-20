import { Crown } from 'lucide-react'
import {
  type CkMove,
  type CkState,
  isDark,
  rcToIndex,
} from '../game/checkers'

type Props = {
  state: CkState
  // Moves the human may play right now (empty when it is not the human's turn).
  legalMoves: CkMove[]
  // Dark-square index chosen as the move origin, or null.
  selected: number | null
  interactive: boolean
  // Flip the board so the human's pieces sit at the bottom (human plays Black).
  flipped: boolean
  lastMove: CkMove | null
  onSelect: (index: number | null) => void
  onMove: (move: CkMove) => void
}

const AXIS = Array.from({ length: 10 }, (_, i) => i) // 0..9

export default function CheckersBoard({
  state,
  legalMoves,
  selected,
  interactive,
  flipped,
  lastMove,
  onSelect,
  onMove,
}: Props) {
  const white = new Set(state.white)
  const black = new Set(state.black)
  const kings = new Set(state.kings)

  // Squares the human can pick up (distinct origins of the legal set), and the
  // destinations reachable from the currently selected origin.
  const origins = new Set(legalMoves.map((m) => m.from))
  const targets = new Map<number, CkMove>()
  if (selected != null) {
    for (const m of legalMoves) {
      if (m.from === selected) targets.set(m.to, m)
    }
  }
  const captureSquares = new Set<number>(
    selected != null ? legalMoves.filter((m) => m.from === selected).flatMap((m) => m.captured) : [],
  )

  function clickSquare(idx: number) {
    if (!interactive) return
    const target = targets.get(idx)
    if (target) {
      onMove(target)
      return
    }
    if (origins.has(idx)) {
      onSelect(idx === selected ? null : idx)
      return
    }
    onSelect(null)
  }

  return (
    <div
      className="grid aspect-square w-full overflow-hidden rounded-xl border-2 shadow-[var(--shadow-btn)]"
      style={{
        gridTemplateColumns: 'repeat(10, 1fr)',
        gridTemplateRows: 'repeat(10, 1fr)',
        borderColor: 'var(--board-frame)',
      }}
    >
      {AXIS.map((dr) =>
        AXIS.map((dc) => {
          const r = flipped ? 9 - dr : dr
          const c = flipped ? 9 - dc : dc
          const dark = isDark(r, c)
          const idx = dark ? rcToIndex(r, c) : -1

          const isWhite = dark && white.has(idx)
          const isBlack = dark && black.has(idx)
          const isKing = dark && kings.has(idx)
          const isOrigin = dark && interactive && origins.has(idx)
          const isTarget = dark && targets.has(idx)
          const isSelected = dark && idx === selected
          const isCapture = dark && captureSquares.has(idx)
          const inLastMove =
            dark &&
            lastMove != null &&
            (idx === lastMove.from || idx === lastMove.to)

          return (
            <button
              key={`${dr}-${dc}`}
              type="button"
              disabled={!dark || (!isOrigin && !isTarget && !isSelected)}
              onClick={() => dark && clickSquare(idx)}
              aria-label={dark ? `square ${idx + 1}` : undefined}
              className="relative flex items-center justify-center p-0"
              style={{
                background: dark ? 'var(--board-dark)' : 'var(--board-light)',
                cursor: isOrigin || isTarget || isSelected ? 'pointer' : 'default',
              }}
            >
              {/* last-move trail */}
              {inLastMove && (
                <span
                  className="pointer-events-none absolute inset-0"
                  style={{ background: 'var(--board-goal)' }}
                />
              )}
              {/* selected origin ring */}
              {isSelected && (
                <span
                  className="pointer-events-none absolute inset-0"
                  style={{ background: 'var(--selection)' }}
                />
              )}
              {/* capture marker on a jumped square */}
              {isCapture && !isWhite && !isBlack && (
                <span
                  className="pointer-events-none absolute h-1/5 w-1/5 rounded-full"
                  style={{ background: 'var(--danger)', opacity: 0.55 }}
                />
              )}

              {/* piece */}
              {(isWhite || isBlack) && (
                <span
                  className="relative flex items-center justify-center rounded-full"
                  style={{
                    width: '74%',
                    height: '74%',
                    background: isWhite ? 'var(--pawn-south)' : 'var(--pawn-north)',
                    border: `2px solid ${
                      isWhite ? 'var(--pawn-south-edge)' : 'var(--pawn-north-edge)'
                    }`,
                    boxShadow: isOrigin
                      ? '0 0 0 3px var(--lagoon)'
                      : '0 1px 3px rgba(26,21,16,0.35)',
                  }}
                >
                  {isKing && (
                    <Crown
                      size={14}
                      strokeWidth={2.5}
                      style={{
                        color: isWhite ? 'var(--lagoon-deep)' : 'var(--lagoon)',
                      }}
                    />
                  )}
                </span>
              )}

              {/* empty legal destination dot */}
              {isTarget && !isWhite && !isBlack && (
                <span
                  className="pointer-events-none absolute h-1/4 w-1/4 rounded-full"
                  style={{ background: 'var(--lagoon)', opacity: 0.8 }}
                />
              )}
            </button>
          )
        }),
      )}
    </div>
  )
}
