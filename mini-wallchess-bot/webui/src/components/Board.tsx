import { type CSSProperties, useState } from 'react'
import {
  type Cell,
  type GameState,
  type Move,
  type Orientation,
  type Side,
  type Wall,
  SIZE,
} from '../game/engine'

const STEP = 100 / SIZE // one cell as a % of the board
const WALL_TH = 2.6 // wall bar thickness, % of board
const HIT_CROSS = 6 // hit-strip cross-axis thickness, % of board
const HIT_LEN = STEP * 0.9 // hit-strip length along the wall (<1 cell → no overlap)

const ROWS = Array.from({ length: SIZE }, (_, i) => SIZE - i) // 9..1 top→bottom
const COLS = Array.from({ length: SIZE }, (_, i) => i + 1) // 1..9

function sameCell(a: Cell, b: Cell) {
  return a.r === b.r && a.c === b.c
}

// Geometry: board is a square; row 9 sits at the top, column 1 at the left.
// A horizontal wall (r,c) lies on the top edge of row r spanning cols c..c+1;
// a vertical wall (r,c) lies on the right edge of col c spanning rows r..r+1.
function wallStyle(w: Wall, thickness: number): CSSProperties {
  if (w.o === 'h') {
    return {
      top: `${(SIZE - w.r) * STEP}%`,
      left: `${(w.c - 1) * STEP}%`,
      width: `${2 * STEP}%`,
      height: `${thickness}%`,
      transform: 'translateY(-50%)',
    }
  }
  return {
    left: `${w.c * STEP}%`,
    top: `${(SIZE - 1 - w.r) * STEP}%`,
    width: `${thickness}%`,
    height: `${2 * STEP}%`,
    transform: 'translateX(-50%)',
  }
}

// Click target for placing a wall. Every wall is centred on its grid
// intersection (c*STEP, (SIZE-r)*STEP); the hit area is a short strip there —
// under one cell long, so neighbouring slots abut instead of overlapping (the
// 2-cell visual bar would intercept adjacent clicks).
function hitStyle(w: Wall): CSSProperties {
  const cx = w.c * STEP
  const cy = (SIZE - w.r) * STEP
  const along = `${HIT_LEN}%`
  const cross = `${HIT_CROSS}%`
  return {
    left: `${cx}%`,
    top: `${cy}%`,
    width: w.o === 'h' ? along : cross,
    height: w.o === 'h' ? cross : along,
    transform: 'translate(-50%, -50%)',
  }
}

type BoardProps = {
  state: GameState
  actionMode: 'move' | 'wall'
  orientation: Orientation
  legalTargets: Cell[]
  interactive: boolean
  canPlace: (wall: Wall) => boolean
  onMove: (to: Cell) => void
  onPlaceWall: (wall: Wall) => void
  goalTopLabel?: string
  goalBottomLabel?: string
  bestMove?: Move
}

export default function Board({
  state,
  actionMode,
  orientation,
  legalTargets,
  interactive,
  canPlace,
  onMove,
  onPlaceWall,
  goalTopLabel,
  goalBottomLabel,
  bestMove,
}: BoardProps) {
  const [hover, setHover] = useState<Wall | null>(null)
  const placing = interactive && actionMode === 'wall'

  // wall slots: anchors r,c in 1..8 for the chosen orientation
  const slots: Wall[] = []
  if (placing) {
    for (let r = 1; r <= SIZE - 1; r++) {
      for (let c = 1; c <= SIZE - 1; c++) slots.push({ r, c, o: orientation })
    }
  }
  const hoverValid = hover ? canPlace(hover) : false

  return (
    <div className="island-shell flex h-full flex-col rounded-[1.75rem] p-3">
      {goalTopLabel && (
        <p className="mb-1 flex-shrink-0 text-right text-[0.6rem] font-bold uppercase tracking-widest text-[var(--sea-ink-soft)] opacity-50">
          {goalTopLabel} ↑
        </p>
      )}
      <div
        className="relative min-h-0 flex-1 overflow-hidden rounded-xl border-4"
        style={{ borderColor: 'var(--board-frame)' }}
      >
        {/* squares */}
        <div
          className="grid h-full w-full"
          style={{ gridTemplateColumns: `repeat(${SIZE}, minmax(0, 1fr))` }}
        >
          {ROWS.map((r) =>
            COLS.map((c) => {
              const cell = { r, c }
              const isSouth = sameCell(state.pawns.south, cell)
              const isNorth = sameCell(state.pawns.north, cell)
              const isTarget =
                interactive &&
                actionMode === 'move' &&
                legalTargets.some((t) => sameCell(t, cell))
              const isBestMoveDest = bestMove?.type === 'move' && sameCell(bestMove.to, cell)
              const isLightSq = (r + c) % 2 === 0
              const isGoal = r === SIZE || r === 1

              return (
                <button
                  key={`${r}-${c}`}
                  type="button"
                  disabled={!isTarget}
                  onClick={() => isTarget && onMove(cell)}
                  aria-label={`${isGoal ? 'Goal row, ' : ''}cell row ${r} column ${c}`}
                  className={[
                    'relative flex items-center justify-center transition',
                    isLightSq
                      ? 'bg-[var(--board-light)]'
                      : 'bg-[var(--board-dark)]',
                    isTarget ? 'cursor-pointer' : 'cursor-default',
                  ].join(' ')}
                >
                  {isGoal && (
                    <span className="pointer-events-none absolute inset-0 bg-[var(--board-goal)]" />
                  )}
                  {isTarget && (
                    <span className="pointer-events-none h-[34%] w-[34%] rounded-full bg-[var(--lagoon-deep)] opacity-70" />
                  )}
                  {isBestMoveDest && (
                    <span className="pointer-events-none absolute h-[78%] w-[78%] rounded-full ring-[3px] ring-[#16a34a] opacity-90" />
                  )}
                  {(isSouth || isNorth) && (
                    <span
                      className={[
                        'relative h-[68%] w-[68%] rounded-full border shadow-md transition',
                        isSouth
                          ? 'bg-[var(--pawn-south)] border-[var(--pawn-south-edge)]'
                          : 'bg-[var(--pawn-north)] border-[var(--pawn-north-edge)]',
                        (isSouth ? state.turn === 'south' : state.turn === 'north') &&
                        !state.winner
                          ? isSouth
                            ? 'ring-2 ring-[var(--lagoon-deep)] ring-offset-1'
                            : 'ring-2 ring-white/80 ring-offset-1'
                          : '',
                      ].join(' ')}
                      aria-hidden="true"
                    />
                  )}
                </button>
              )
            }),
          )}
        </div>

        {/* wall-mode grid hint — faint lines at cell boundaries */}
        {placing && (
          <div className="pointer-events-none absolute inset-0">
            {Array.from({ length: SIZE - 1 }, (_, i) => (
              <div
                key={`gh-${i}`}
                className="absolute left-0 right-0"
                style={{
                  top: `${((i + 1) / SIZE) * 100}%`,
                  height: '1px',
                  background: 'rgba(26,21,16,0.10)',
                  transform: 'translateY(-50%)',
                }}
              />
            ))}
            {Array.from({ length: SIZE - 1 }, (_, i) => (
              <div
                key={`gv-${i}`}
                className="absolute top-0 bottom-0"
                style={{
                  left: `${((i + 1) / SIZE) * 100}%`,
                  width: '1px',
                  background: 'rgba(26,21,16,0.10)',
                  transform: 'translateX(-50%)',
                }}
              />
            ))}
          </div>
        )}

        {/* placed walls */}
        <div className="pointer-events-none absolute inset-0">
          {state.walls.map((w) => (
            <div
              key={`${w.o}-${w.r}-${w.c}`}
              className="absolute rounded-full bg-[var(--board-frame)] shadow"
              style={wallStyle(w, WALL_TH)}
            />
          ))}
          {bestMove?.type === 'wall' && (
            <div
              className="absolute rounded-full"
              style={{
                ...wallStyle(bestMove.wall, WALL_TH),
                background: '#16a34a',
                opacity: 0.85,
              }}
            />
          )}
          {hover && (
            <div
              className="absolute rounded-full"
              style={{
                ...wallStyle(hover, WALL_TH),
                background: hoverValid ? 'var(--lagoon-deep)' : 'var(--danger)',
                opacity: 0.7,
              }}
            />
          )}
        </div>

        {/* wall placement hit-strips */}
        {placing && (
          <div className="absolute inset-0">
            {slots.map((w) => (
              <button
                key={`slot-${w.o}-${w.r}-${w.c}`}
                type="button"
                aria-label={`Place ${w.o === 'h' ? 'horizontal' : 'vertical'} wall at ${w.r},${w.c}`}
                onPointerEnter={() => setHover(w)}
                onPointerLeave={() => setHover((h) => (h === w ? null : h))}
                onClick={() => {
                  if (canPlace(w)) {
                    onPlaceWall(w)
                    setHover(null)
                  }
                }}
                className="absolute cursor-pointer bg-transparent touch-none"
                style={hitStyle(w)}
              />
            ))}
          </div>
        )}
      </div>
      {goalBottomLabel && (
        <p className="mt-1 text-left text-[0.6rem] font-bold uppercase tracking-widest text-[var(--sea-ink-soft)] opacity-50">
          ↓ {goalBottomLabel}
        </p>
      )}
    </div>
  )
}

export type { Cell, Side }
