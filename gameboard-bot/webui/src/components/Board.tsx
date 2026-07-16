import {
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  useRef,
  useState,
} from 'react'
import {
  type Cell,
  type GameState,
  type Move,
  type Side,
  type Wall,
  SIZE,
} from '../game/engine'

const STEP = 100 / SIZE // one cell as a % of the board
const WALL_TH = 2.6 // wall bar thickness, % of board
const WALL_INTENT_MARGIN = 0.1 // board cells; avoids accidental walls near move targets

const ROWS = Array.from({ length: SIZE }, (_, i) => SIZE - i) // 9..1 top→bottom
const COLS = Array.from({ length: SIZE }, (_, i) => i + 1) // 1..9

function sameCell(a: Cell, b: Cell) {
  return a.r === b.r && a.c === b.c
}

type BoardIntent =
  | { type: 'move'; cell: Cell }
  | { type: 'wall'; wall: Wall }

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n))
}

export function boardIntentAt(x: number, y: number): BoardIntent {
  const col0 = clamp(Math.floor(x), 0, SIZE - 1)
  const row0 = clamp(Math.floor(y), 0, SIZE - 1)
  const centerX = col0 + 0.5
  const centerY = row0 + 0.5
  const centerDist = Math.hypot(x - centerX, y - centerY)

  const lineX = clamp(Math.round(x), 1, SIZE - 1)
  const lineY = clamp(Math.round(y), 1, SIZE - 1)
  const dx = Math.abs(x - lineX)
  const dy = Math.abs(y - lineY)
  const gapDist = Math.min(dx, dy)

  if (gapDist + WALL_INTENT_MARGIN < centerDist) {
    return {
      type: 'wall',
      wall: {
        r: SIZE - lineY,
        c: lineX,
        o: dx < dy ? 'v' : 'h',
      },
    }
  }

  return { type: 'move', cell: { r: SIZE - row0, c: col0 + 1 } }
}

function eventIntent(
  e: ReactMouseEvent<HTMLDivElement> | ReactPointerEvent<HTMLDivElement>,
  el: HTMLDivElement | null,
): BoardIntent | null {
  if (!el) return null
  const rect = el.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return null
  const x = ((e.clientX - rect.left) / rect.width) * SIZE
  const y = ((e.clientY - rect.top) / rect.height) * SIZE
  return boardIntentAt(x, y)
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


type BoardProps = {
  state: GameState
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
  legalTargets,
  interactive,
  canPlace,
  onMove,
  onPlaceWall,
  goalTopLabel,
  goalBottomLabel,
  bestMove,
}: BoardProps) {
  const boardRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<Wall | null>(null)
  const canTryWall = interactive && state.wallsLeft[state.turn] > 0
  const hoverValid = hover ? canPlace(hover) : false


  function handlePointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    if (!canTryWall) {
      setHover((h) => (h ? null : h))
      return
    }
    const intent = eventIntent(e, boardRef.current)
    const next = intent?.type === 'wall' ? intent.wall : null
    setHover((h) =>
      (h?.r === next?.r && h?.c === next?.c && h?.o === next?.o) || (!h && !next)
        ? h
        : next,
    )
  }

  function handleClick(e: ReactMouseEvent<HTMLDivElement>) {
    if (!interactive || e.detail === 0) return
    const intent = eventIntent(e, boardRef.current)
    if (!intent) return
    e.preventDefault()
    e.stopPropagation()
    if (intent.type === 'move') {
      if (legalTargets.some((t) => sameCell(t, intent.cell))) onMove(intent.cell)
      return
    }
    if (canPlace(intent.wall)) {
      onPlaceWall(intent.wall)
      setHover(null)
    }
  }

  return (
    <div className="border bg-card text-card-foreground shadow-sm flex h-full flex-col rounded-[1.75rem] p-3">
      {goalTopLabel && (
        <p className="mb-1 flex-shrink-0 text-right text-[0.6rem] font-bold uppercase tracking-widest text-muted-foreground opacity-50">
          {goalTopLabel} ↑
        </p>
      )}
      <div
        ref={boardRef}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHover(null)}
        onClickCapture={handleClick}
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
              const isTarget = interactive && legalTargets.some((t) => sameCell(t, cell))
              const isBestMoveDest = bestMove?.type === 'move' && sameCell(bestMove.to, cell)
              const isLightSq = (r + c) % 2 === 0
              const isGoal = r === SIZE || r === 1

              return (
                <button
                  key={`${r}-${c}`}
                  type="button"
                  aria-disabled={!isTarget}
                  tabIndex={isTarget ? 0 : -1}
                  onClick={(e) => {
                    if (e.detail === 0 && isTarget) onMove(cell)
                  }}
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
                    <span className="pointer-events-none h-[34%] w-[34%] rounded-full bg-primary opacity-70" />
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
                            ? 'ring-2 ring-primary ring-offset-1'
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

        {/* wall grid hint — faint lines at wall gaps */}
        {canTryWall && (
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
                background: hoverValid ? 'var(--primary)' : 'var(--destructive)',
                opacity: 0.7,
              }}
            />
          )}
        </div>

      </div>
      {goalBottomLabel && (
        <p className="mt-1 text-left text-[0.6rem] font-bold uppercase tracking-widest text-muted-foreground opacity-50">
          ↓ {goalBottomLabel}
        </p>
      )}
    </div>
  )
}

export type { Cell, Side }
