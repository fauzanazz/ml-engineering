import { useCallback, useEffect, useState } from 'react'
import { Link, createFileRoute } from '@tanstack/react-router'
import { RotateCcw } from 'lucide-react'
import CheckersBoard from '../components/CheckersBoard'
import WinMeter from '../components/WinMeter'
import {
  type CkColor,
  type CkMove,
  type CkState,
  type CkStatus,
  analyze,
  applyMove,
  initialState,
  legalMoves,
  status as readStatus,
} from '../game/checkers'

export const Route = createFileRoute('/checkers')({ component: Checkers })

// Difficulty = bot search depth, with a node cap so tactical spikes stay snappy.
// d8 ≈ 2 ms/move, d12 ≈ 50 ms/move on a single core (see docs/checkers-gen1-bot).
const DIFFICULTY: Record<string, { label: string; depth: number; nodeLimit: number }> = {
  easy: { label: 'Easy', depth: 4, nodeLimit: 0 },
  medium: { label: 'Medium', depth: 8, nodeLimit: 200_000 },
  hard: { label: 'Hard', depth: 12, nodeLimit: 400_000 },
}
type DifficultyKey = keyof typeof DIFFICULTY

// Pause before the bot replies so its move is legible, not instant.
const BOT_THINK_MS = 350

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>()
  setTimeout(resolve, ms)
  return promise
}

const OK_STATUS: CkStatus = { terminal: false, winner: null, draw: false }

function Checkers() {
  const [humanColor, setHumanColor] = useState<CkColor>('white')
  const [difficulty, setDifficulty] = useState<DifficultyKey>('medium')
  const [state, setState] = useState<CkState>(initialState)
  const [selected, setSelected] = useState<number | null>(null)
  const [legal, setLegal] = useState<CkMove[]>([])
  const [status, setStatus] = useState<CkStatus>(OK_STATUS)
  const [whitePct, setWhitePct] = useState(50)
  const [thinking, setThinking] = useState(false)
  const [lastMove, setLastMove] = useState<CkMove | null>(null)
  // Bumped on every New Game / option change so in-flight effects abandon.
  const [gameId, setGameId] = useState(0)

  const { depth, nodeLimit } = DIFFICULTY[difficulty]
  const botColor: CkColor = humanColor === 'white' ? 'black' : 'white'

  const newGame = useCallback(() => {
    setState(initialState())
    setSelected(null)
    setLegal([])
    setStatus(OK_STATUS)
    setWhitePct(50)
    setThinking(false)
    setLastMove(null)
    setGameId((n) => n + 1)
  }, [])

  // The engine loop: after every position change, read terminal status + the
  // win split. On the human's turn, load the legal set for the board; on the
  // bot's turn, play the analysed move after a short pause. All WASM calls live
  // here (client only); `cancelled` discards results from a superseded position.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const s = await readStatus(state)
      if (cancelled) return
      setStatus(s)
      if (s.terminal) {
        setLegal([])
        setThinking(false)
        setWhitePct(s.winner === 'white' ? 100 : s.winner === 'black' ? 0 : 50)
        return
      }

      const botTurn = state.stm === botColor
      setThinking(botTurn)
      const a = await analyze(state, depth, nodeLimit)
      if (cancelled) return
      setWhitePct(a.white)

      if (botTurn) {
        if (!a.move) return
        await delay(BOT_THINK_MS)
        if (cancelled) return
        const next = await applyMove(state, a.move)
        if (cancelled) return
        setLastMove(a.move)
        setSelected(null)
        setThinking(false)
        setState(next)
      } else {
        const moves = await legalMoves(state)
        if (cancelled) return
        setLegal(moves)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [state, botColor, depth, nodeLimit, gameId])

  const handleMove = useCallback(
    async (move: CkMove) => {
      const next = await applyMove(state, move)
      setLastMove(move)
      setSelected(null)
      setLegal([])
      setState(next)
    },
    [state],
  )

  const interactive =
    !thinking && !status.terminal && state.stm === humanColor && legal.length > 0
  const mustCapture = legal.length > 0 && legal.every((m) => m.captured.length > 0)

  let banner: string
  if (status.terminal) {
    banner = status.draw
      ? 'Draw'
      : status.winner === humanColor
        ? 'You win!'
        : 'Bot wins'
  } else if (thinking) {
    banner = 'Bot is thinking…'
  } else if (state.stm === humanColor) {
    banner = mustCapture ? 'Your move — capture is mandatory' : 'Your move'
  } else {
    banner = 'Bot to move'
  }

  return (
    <main className="page-wrap flex min-h-[calc(100vh-var(--navbar-h))] flex-col items-center px-4 py-6">
      <div className="mb-4 text-center">
        <p className="island-kicker mb-1">International Draughts</p>
        <h1 className="display-title text-3xl font-bold text-[var(--sea-ink)] sm:text-4xl">
          {banner}
        </h1>
      </div>

      <div className="flex w-full max-w-3xl flex-col items-stretch gap-5 sm:flex-row sm:items-start">
        <div className="min-w-0 flex-1">
          <CheckersBoard
            state={state}
            legalMoves={interactive ? legal : []}
            selected={selected}
            interactive={interactive}
            flipped={humanColor === 'black'}
            lastMove={lastMove}
            onSelect={setSelected}
            onMove={handleMove}
          />
        </div>

        <aside className="flex shrink-0 flex-col gap-4 sm:w-56">
          <div className="island-shell flex h-28 rounded-xl p-3">
            <WinMeter
              south={whitePct}
              southLabel="White"
              northLabel="Black"
              loading={thinking}
            />
          </div>

          <div className="island-shell rounded-xl p-3">
            <p className="island-kicker mb-2 text-[0.6rem]">You play</p>
            <div className="flex gap-2">
              {(['white', 'black'] as const).map((color) => (
                <button
                  key={color}
                  type="button"
                  onClick={() => {
                    if (color === humanColor) return
                    setHumanColor(color)
                    newGame()
                  }}
                  className="flex-1 rounded-lg border px-2 py-1.5 text-sm font-semibold capitalize transition"
                  style={{
                    borderColor: 'var(--line)',
                    background:
                      color === humanColor ? 'var(--selection)' : 'var(--chip-bg)',
                    color: 'var(--sea-ink)',
                  }}
                >
                  {color}
                </button>
              ))}
            </div>
          </div>

          <div className="island-shell rounded-xl p-3">
            <p className="island-kicker mb-2 text-[0.6rem]">Difficulty</p>
            <div className="flex flex-col gap-1.5">
              {(Object.keys(DIFFICULTY) as DifficultyKey[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => {
                    if (key === difficulty) return
                    setDifficulty(key)
                    newGame()
                  }}
                  className="rounded-lg border px-2 py-1.5 text-sm font-semibold transition"
                  style={{
                    borderColor: 'var(--line)',
                    background:
                      key === difficulty ? 'var(--selection)' : 'var(--chip-bg)',
                    color: 'var(--sea-ink)',
                  }}
                >
                  {DIFFICULTY[key].label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="button"
            onClick={newGame}
            className="inline-flex items-center justify-center gap-2 rounded-full bg-[var(--btn-primary-bg)] px-5 py-2.5 text-sm font-bold text-[var(--btn-primary-fg)] shadow-[var(--shadow-btn)] transition hover:-translate-y-0.5 hover:opacity-90"
          >
            <RotateCcw size={16} />
            New Game
          </button>

          <Link
            to="/"
            className="text-center text-sm font-semibold text-[var(--sea-ink-soft)]"
          >
            ← Back
          </Link>
        </aside>
      </div>
    </main>
  )
}
