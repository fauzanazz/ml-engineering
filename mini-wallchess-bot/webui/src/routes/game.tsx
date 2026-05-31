import { useEffect, useRef, useState } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import {
  Bot,
  HelpCircle,
  Move as MoveIcon,
  RectangleHorizontal,
  RectangleVertical,
  RotateCcw,
  Square,
  Users,
} from 'lucide-react'
import Board from '../components/Board'
import HowToPlay from '../components/HowToPlay'
import WinMeter from '../components/WinMeter'
import { type BotEngine, analyzePosition, botMove } from '../game/api'
import {
  type Cell,
  type GameState,
  type Orientation,
  type Side,
  type Wall,
  GOAL_ROW,
  applyMove,
  canPlaceWall,
  distanceToGoal,
  initialState,
  pawnMoves,
} from '../game/engine'

type Mode = 'bot' | 'friend' | 'arena'

export const Route = createFileRoute('/game')({
  validateSearch: (search: Record<string, unknown>): { mode: Mode } => ({
    mode:
      search.mode === 'friend'
        ? 'friend'
        : search.mode === 'arena'
          ? 'arena'
          : 'bot',
  }),
  component: Game,
})

const BOT_SIDE: Side = 'north'

// Arena pits the trained net (south) against the heuristic (north) so you can
// watch them play. Delay between auto-moves keeps it watchable.
const ARENA_ENGINE: Record<Side, BotEngine> = { south: 'net', north: 'heuristic' }
const ARENA_MOVE_DELAY_MS = 550
// Stalling wall-wars never reach a goal; cap arena games and decide the cap by
// race progress (closer pawn wins) so it always resolves instead of looping.
const ARENA_PLY_CAP = 140
function raceWinner(s: GameState): Side {
  const ds = distanceToGoal(s.walls, s.pawns.south, GOAL_ROW.south)
  const dn = distanceToGoal(s.walls, s.pawns.north, GOAL_ROW.north)
  return ds <= dn ? 'south' : 'north'
}

function Game() {
  const { mode } = Route.useSearch()
  const [state, setState] = useState<GameState>(initialState)
  const [actionMode, setActionMode] = useState<'move' | 'wall'>('move')
  const [orientation, setOrientation] = useState<Orientation>('h')
  const [thinking, setThinking] = useState(false)
  const [botError, setBotError] = useState<string | null>(null)
  const [showHelp, setShowHelp] = useState(false)
  const [southWin, setSouthWin] = useState(50)
  const [analyzing, setAnalyzing] = useState(false)
  const [movesMade, setMovesMade] = useState(0)
  const [confirmLeave, setConfirmLeave] = useState(false)
  const plyRef = useRef(0)
  const navigate = useNavigate()

  useEffect(() => {
    document.body.classList.add('is-game')
    return () => document.body.classList.remove('is-game')
  }, [])

  // Which sides move automatically: the bot in vs-bot, both in arena, none in friend.
  const autoSides: Side[] =
    mode === 'bot' ? [BOT_SIDE] : mode === 'arena' ? ['south', 'north'] : []
  const isAutoTurn = !state.winner && autoSides.includes(state.turn)
  const isHumanTurn = mode !== 'arena' && (mode === 'friend' || state.turn !== BOT_SIDE)
  const interactive = !state.winner && !thinking && isHumanTurn
  const wallsLeft = state.wallsLeft[state.turn]

  // Auto turn: ask the engine for a move, then apply it. In arena both sides are
  // engines (net vs heuristic), paced by a short delay so it's watchable.
  useEffect(() => {
    if (!isAutoTurn) return
    // Arena: stop a never-ending wall-war and award the cap by race progress.
    if (mode === 'arena' && plyRef.current >= ARENA_PLY_CAP) {
      setState((s) => (s.winner ? s : { ...s, winner: raceWinner(s) }))
      return
    }
    const turn = state.turn
    let cancelled = false
    setThinking(true)
    setBotError(null)
    ;(async () => {
      try {
        if (mode === 'arena') {
          await new Promise((r) => setTimeout(r, ARENA_MOVE_DELAY_MS))
          if (cancelled) return
        }
        const engine = mode === 'arena' ? ARENA_ENGINE[turn] : undefined
        const move = await botMove({ data: state, engine })
        if (!cancelled) {
          setState((s) => (s.turn === turn && !s.winner ? applyMove(s, move) : s))
          plyRef.current += 1
        }
      } catch (err) {
        console.error('bot move failed', err)
        if (!cancelled) setBotError('Bot failed to respond. Hit Reset to try again.')
      } finally {
        if (!cancelled) setThinking(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [state, mode])

  // Win-probability meter: re-evaluate the position with the engine each change.
  useEffect(() => {
    if (state.winner) {
      setSouthWin(state.winner === 'south' ? 100 : 0)
      setAnalyzing(false)
      return
    }
    let cancelled = false
    setAnalyzing(true)
    ;(async () => {
      try {
        const a = await analyzePosition(state)
        if (!cancelled) setSouthWin(a.south)
      } catch (err) {
        console.warn('analysis failed', err)
      } finally {
        if (!cancelled) setAnalyzing(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [state])

  // Keyboard shortcuts: M=move, W=wall, H/V=orientation, R=reset, ?=help
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      switch (e.key) {
        case 'm': case 'M':
          if (interactive) setActionMode('move')
          break
        case 'w': case 'W':
          if (interactive && wallsLeft > 0) setActionMode('wall')
          break
        case 'h': case 'H':
          if (interactive && actionMode === 'wall') setOrientation('h')
          break
        case 'v': case 'V':
          if (interactive && actionMode === 'wall') setOrientation('v')
          break
        case 'r': case 'R':
          plyRef.current = 0
          setState(initialState())
          setActionMode('move')
          setThinking(false)
          setBotError(null)
          setShowHelp(false)
          break
        case '?':
          setShowHelp((s) => !s)
          break
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [interactive, actionMode, wallsLeft])

  // Warn before browser refresh/close mid-game.
  useEffect(() => {
    if (movesMade === 0 || state.winner) return
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [movesMade, state.winner])

  // Clear confirm panel when game ends.
  useEffect(() => {
    if (state.winner) setConfirmLeave(false)
  }, [state.winner])

  function handleChangeMode() {
    if (movesMade > 0 && !state.winner) {
      setConfirmLeave(true)
      return
    }
    navigate({ to: '/play' })
  }

  function handleMove(to: Cell) {
    setState((s) => applyMove(s, { type: 'move', to }))
    setActionMode('move')
    setMovesMade((n) => n + 1)
  }

  function handlePlaceWall(wall: Wall) {
    setState((s) => applyMove(s, { type: 'wall', wall }))
    setActionMode('move')
    setMovesMade((n) => n + 1)
  }

  function reset() {
    plyRef.current = 0
    setState(initialState())
    setActionMode('move')
    setThinking(false)
    setBotError(null)
    setMovesMade(0)
    setConfirmLeave(false)
  }

  const legalTargets = state.winner ? [] : pawnMoves(state, state.turn)
  const canPlace = (wall: Wall) =>
    !state.winner && canPlaceWall(state, wall, state.turn)

  const sideLabel = (s: typeof state.turn) =>
    mode === 'bot'
      ? s === 'south' ? 'You' : 'Bot'
      : mode === 'arena'
        ? s === 'south' ? 'Net' : 'Heuristic'
        : s === 'south' ? 'Player 1' : 'Player 2'
  const southIcon = mode === 'arena' ? Bot : Users
  const northIcon = mode === 'friend' ? Users : Bot

  const status = state.winner
    ? `${sideLabel(state.winner)} wins!`
    : thinking
      ? mode === 'arena' ? `${sideLabel(state.turn)} thinking…` : 'Bot thinking…'
      : `${sideLabel(state.turn)} to move`

  return (
    <main
      className="relative flex overflow-hidden px-3"
      style={{ height: 'calc(100dvh - var(--navbar-h))' }}
    >
      <div className="mx-auto flex w-full max-w-5xl gap-3 py-3">
        {/* sidebar: North → WinMeter → South */}
        <div className="hidden w-40 flex-shrink-0 flex-col gap-2 lg:flex">
          <PlayerCard
            side="north"
            label={sideLabel('north')}
            icon={northIcon}
            walls={state.wallsLeft.north}
            active={state.turn === 'north' && !state.winner}
          />
          <div className="flex-1 min-h-0">
            <WinMeter
              south={southWin}
              southLabel={sideLabel('south')}
              northLabel={sideLabel('north')}
              loading={analyzing}
            />
          </div>
          <PlayerCard
            side="south"
            label={sideLabel('south')}
            icon={southIcon}
            walls={state.wallsLeft.south}
            active={state.turn === 'south' && !state.winner}
          />
        </div>

        {/* board column */}
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          {/* top bar: status + controls */}
          <div className="flex flex-shrink-0 items-center justify-between gap-2">
            <span className="island-kicker" aria-live="polite" aria-atomic="true">{status}</span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleChangeMode}
                className="text-xs font-semibold text-[var(--sea-ink-soft)] hover:text-[var(--sea-ink)]"
              >
                ← Mode
              </button>
              <button
                type="button"
                onClick={() => setShowHelp(true)}
                aria-label="How to play"
                className="inline-flex items-center rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-1.5 text-[var(--sea-ink-soft)] transition hover:border-[var(--accent-text)] hover:text-[var(--sea-ink)]"
              >
                <HelpCircle size={14} />
              </button>
              <button
                type="button"
                onClick={reset}
                className="inline-flex items-center gap-1.5 rounded-full border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-xs font-semibold text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
              >
                <RotateCcw size={13} />
                Reset
              </button>
            </div>
          </div>

          {botError && (
            <p className="flex-shrink-0 text-xs font-semibold text-[var(--danger)]" role="alert">
              {botError}
            </p>
          )}

          {/* mobile player cards + WinMeter */}
          <div className="flex flex-shrink-0 gap-2 lg:hidden">
            <PlayerCard
              side="north"
              label={sideLabel('north')}
              icon={northIcon}
              walls={state.wallsLeft.north}
              active={state.turn === 'north' && !state.winner}
            />
            <PlayerCard
              side="south"
              label={sideLabel('south')}
              icon={southIcon}
              walls={state.wallsLeft.south}
              active={state.turn === 'south' && !state.winner}
            />
          </div>

          {/* board: fills remaining height, square constrained by whichever axis is smaller */}
          <div className="flex min-h-0 flex-1 items-center justify-center">
            <div
              className={[
                'aspect-square h-full max-w-full transition-opacity duration-300',
                thinking ? 'opacity-60' : '',
              ].join(' ')}
            >
              <Board
                state={state}
                actionMode={actionMode}
                orientation={orientation}
                legalTargets={legalTargets}
                interactive={interactive}
                canPlace={canPlace}
                onMove={handleMove}
                onPlaceWall={handlePlaceWall}
                goalTopLabel="P2"
                goalBottomLabel="P1"
              />
            </div>
          </div>

          {/* action controls */}
          <div className="flex flex-shrink-0 flex-wrap items-center justify-center gap-2">
            <div className="inline-flex overflow-hidden rounded-full border border-[var(--line)]">
              <ToggleBtn
                active={actionMode === 'move'}
                disabled={!interactive}
                onClick={() => setActionMode('move')}
              >
                <MoveIcon size={14} /> Move
              </ToggleBtn>
              <ToggleBtn
                active={actionMode === 'wall'}
                disabled={!interactive || wallsLeft <= 0}
                onClick={() => setActionMode('wall')}
              >
                <Square size={14} /> Wall
              </ToggleBtn>
            </div>

            {actionMode === 'wall' && interactive && (
              <div className="inline-flex overflow-hidden rounded-full border border-[var(--line)]">
                <ToggleBtn
                  active={orientation === 'h'}
                  onClick={() => setOrientation('h')}
                >
                  <RectangleHorizontal size={14} /> Horiz
                </ToggleBtn>
                <ToggleBtn
                  active={orientation === 'v'}
                  onClick={() => setOrientation('v')}
                >
                  <RectangleVertical size={14} /> Vert
                </ToggleBtn>
              </div>
            )}
          </div>

          <p className="flex-shrink-0 text-center text-xs text-[var(--sea-ink-soft)]">
            {actionMode === 'move'
              ? 'Tap a highlighted square to move.'
              : 'Tap a grid line to place a wall.'}
          </p>
          <span className="sr-only" aria-live="polite" aria-atomic="true">
            {actionMode === 'wall' ? 'Wall placement mode. Select a grid line to place a wall.' : ''}
          </span>
          <p className="hidden flex-shrink-0 text-center text-xs text-[var(--sea-ink-soft)] sm:block">
            <kbd className="rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1 font-mono text-[0.65rem] font-semibold">M</kbd>
            {' '}move{' · '}
            <kbd className="rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1 font-mono text-[0.65rem] font-semibold">W</kbd>
            {' '}wall{' · '}
            <kbd className="rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1 font-mono text-[0.65rem] font-semibold">H/V</kbd>
            {' '}orient{' · '}
            <kbd className="rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1 font-mono text-[0.65rem] font-semibold">R</kbd>
            {' '}reset{' · '}
            <kbd className="rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1 font-mono text-[0.65rem] font-semibold">?</kbd>
            {' '}help
          </p>
        </div>
      </div>

      {state.winner && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
          <div className="island-shell rise-in rounded-2xl p-6 text-center">
            <p className="display-title mb-3 text-2xl font-bold text-[var(--sea-ink)]">
              {mode === 'arena'
                ? `${sideLabel(state.winner)} wins! 🎉`
                : state.winner === 'south'
                  ? mode === 'bot' ? 'You win! 🎉' : 'Player 1 wins! 🎉'
                  : mode === 'bot' ? 'Bot wins.' : 'Player 2 wins! 🎉'}
            </p>
            <button
              type="button"
              onClick={reset}
              className="rounded-full bg-[var(--btn-primary-bg)] px-6 py-2.5 text-sm font-bold text-[var(--btn-primary-fg)] transition hover:opacity-90"
            >
              Play again
            </button>
          </div>
        </div>
      )}

      {confirmLeave && !state.winner && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
          <div className="island-shell rise-in rounded-2xl p-4" style={{ animationDuration: '300ms' }}>
            <p className="mb-3 text-sm text-[var(--sea-ink)]">
              Leave this game? Progress will be lost.
            </p>
            <div className="flex justify-center gap-2">
              <button
                type="button"
                onClick={() => navigate({ to: '/play' })}
                className="rounded-full bg-[var(--btn-primary-bg)] px-4 py-2 text-xs font-bold text-[var(--btn-primary-fg)] transition hover:opacity-90"
              >
                Leave game
              </button>
              <button
                type="button"
                onClick={() => setConfirmLeave(false)}
                className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] px-4 py-2 text-xs font-semibold text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
              >
                Keep playing
              </button>
            </div>
          </div>
        </div>
      )}

      {showHelp && <HowToPlay onClose={() => setShowHelp(false)} />}
    </main>
  )
}

function ToggleBtn({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean
  disabled?: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  const [flash, setFlash] = useState(false)
  const prevActive = useRef(active)

  useEffect(() => {
    if (active && !prevActive.current) {
      setFlash(true)
      const t = setTimeout(() => setFlash(false), 280)
      prevActive.current = true
      return () => clearTimeout(t)
    }
    if (!active) prevActive.current = false
  }, [active])

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      className={[
        'inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40',
        active
          ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-fg)]'
          : 'bg-[var(--chip-bg)] text-[var(--sea-ink)] hover:bg-[var(--link-bg-hover)]',
        flash ? 'ring-2 ring-[var(--lagoon-deep)] ring-offset-1' : '',
      ].join(' ')}
    >
      {children}
    </button>
  )
}

function PlayerCard({
  side,
  label,
  icon: Icon,
  walls,
  active,
}: {
  side: Side
  label: string
  icon: typeof Bot
  walls: number
  active: boolean
}) {
  const dot =
    side === 'south'
      ? 'bg-[var(--lagoon)]'
      : 'bg-[var(--pawn-north)]'

  return (
    <div
      className={[
        'py-1 transition',
        active ? 'opacity-100' : 'opacity-60',
      ].join(' ')}
    >
      <div className="flex items-center gap-3">
        <span className={`h-6 w-6 flex-shrink-0 rounded-full shadow ${dot}`} />
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-sm font-bold text-[var(--sea-ink)]">
            <Icon size={15} />
            {label}
          </div>
          <div className="text-xs text-[var(--sea-ink-soft)]">
            Walls left: <span className="font-semibold">{walls}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
