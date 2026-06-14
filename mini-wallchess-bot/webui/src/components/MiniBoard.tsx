// Compact SVG render of a game state: 9x9 grid, two pawns, placed walls.
// Pure/presentational — used for the current node and node previews.
import type { GameState } from '../game/engine'
import { SIZE } from '../game/engine'

type Props = {
  state: GameState
  /** Pixel size of the square board. */
  size?: number
}

export default function MiniBoard({ state, size = 220 }: Props) {
  const pad = 6
  const inner = size - pad * 2
  const cell = inner / SIZE
  // Board rows: row 9 (NORTH home) at top, row 1 (SOUTH home) at bottom.
  const cx = (c: number) => pad + (c - 0.5) * cell
  const cy = (r: number) => pad + (SIZE - r + 0.5) * cell
  // top-left corner of the gap line for a wall anchor (r,c)
  const lineX = (c: number) => pad + c * cell
  const lineY = (r: number) => pad + (SIZE - r) * cell

  const wallW = Math.max(2.5, cell * 0.16)

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label="board state"
    >
      <rect
        x={0}
        y={0}
        width={size}
        height={size}
        rx={10}
        fill="var(--board-dark)"
      />
      {/* cells */}
      {Array.from({ length: SIZE }, (_, ri) =>
        Array.from({ length: SIZE }, (_, ci) => {
          const r = ri + 1
          const c = ci + 1
          const light = (r + c) % 2 === 0
          return (
            <rect
              key={`${r}-${c}`}
              x={pad + (c - 1) * cell + 0.5}
              y={pad + (SIZE - r) * cell + 0.5}
              width={cell - 1}
              height={cell - 1}
              rx={2}
              fill={light ? 'var(--board-light)' : 'var(--board-dark)'}
            />
          )
        }),
      )}
      {/* goal rows tint */}
      <rect x={pad} y={pad} width={inner} height={cell} fill="var(--board-goal)" />
      <rect
        x={pad}
        y={pad + (SIZE - 1) * cell}
        width={inner}
        height={cell}
        fill="var(--board-goal)"
      />
      {/* walls */}
      {state.walls.map((w, i) => {
        if (w.o === 'h') {
          // horizontal: between rows w.r and w.r+1, spanning cols w.c, w.c+1
          return (
            <rect
              key={i}
              x={lineX(w.c - 1)}
              y={lineY(w.r) - wallW / 2}
              width={cell * 2}
              height={wallW}
              rx={wallW / 2}
              fill="var(--accent-text)"
            />
          )
        }
        // vertical: between cols w.c and w.c+1, spanning rows w.r, w.r+1
        return (
          <rect
            key={i}
            x={lineX(w.c) - wallW / 2}
            y={lineY(w.r + 1)}
            width={wallW}
            height={cell * 2}
            rx={wallW / 2}
            fill="var(--accent-text)"
          />
        )
      })}
      {/* pawns */}
      <circle
        cx={cx(state.pawns.south.c)}
        cy={cy(state.pawns.south.r)}
        r={cell * 0.32}
        fill="var(--pawn-south)"
        stroke="#000"
        strokeOpacity={0.35}
      />
      <circle
        cx={cx(state.pawns.north.c)}
        cy={cy(state.pawns.north.r)}
        r={cell * 0.32}
        fill="var(--pawn-north)"
        stroke="#000"
        strokeOpacity={0.25}
      />
    </svg>
  )
}
