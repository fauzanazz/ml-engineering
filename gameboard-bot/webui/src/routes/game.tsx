import { useEffect, useRef, useState } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import {
  Bot,
  ChevronLeft,
  ChevronRight,
  Download,
  HelpCircle,
  History,
  Lightbulb,
  Loader2,
  Pause,
  Play,
  PlusSquare,
  RotateCcw,
  Users,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import Board from '../components/Board'
import HowToPlay from '../components/HowToPlay'
import ReplayList from '../components/ReplayList'
import WinMeter from '../components/WinMeter'
import { analyzePosition, botMove } from '../game/api'
import { type BotSpec, BOTS, BROWSER_MAX_BOT_ID, getBot } from '../game/bots'
import {
  type Cell,
  type GameState,
  type Move,
  type Side,
  type Wall,
  GOAL_ROW,
  applyMove,
  canPlaceWall,
  distanceToGoal,
  initialState,
  pawnMoves,
} from '../game/engine'
import {
  type ReplayEngine,
  type ReplayFrame,
  type ReplayRecord,
  downloadReplay,
  makeId,
  saveReplay,
} from '../game/replay'

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

// Arena pits two bots against each other. Each side picks any bot from the
// catalog (../game/bots) by id, so any matchup runs — including a bot vs an
// identical copy of itself.
type ArenaBots = { south: string; north: string }
const DEFAULT_ARENA_BOTS: ArenaBots = { south: BROWSER_MAX_BOT_ID, north: 'ab-d10' }
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
  const [thinking, setThinking] = useState(false)
  const [botError, setBotError] = useState<string | null>(null)
  const [showHelp, setShowHelp] = useState(false)
  const [southWin, setSouthWin] = useState(50)
  const [analyzing, setAnalyzing] = useState(false)
  const [movesMade, setMovesMade] = useState(0)
  const [confirmLeave, setConfirmLeave] = useState(false)
  const [arenaBots, setArenaBots] = useState<ArenaBots>(DEFAULT_ARENA_BOTS)
  const [botId, setBotId] = useState<string>(BROWSER_MAX_BOT_ID)
  const [confirmBotChange, setConfirmBotChange] = useState<string | null>(null)
  const [savedReplay, setSavedReplay] = useState<ReplayRecord | null>(null)
  type OverlayPanel = { type: 'list' }
  const [panel, setPanel] = useState<OverlayPanel | null>(null)
  type ReplayMode = { record: ReplayRecord; idx: number }
  const [replayMode, setReplayMode] = useState<ReplayMode | null>(null)
  const [replayBotMove, setReplayBotMove] = useState<Move | undefined>(undefined)
  const [botHelp, setBotHelp] = useState(false)
  const [helpMove, setHelpMove] = useState<Move | undefined>(undefined)
  const plyRef = useRef(0)
  const replayFramesRef = useRef<ReplayFrame[]>([])
  const gameIdRef = useRef(makeId())
  const gameStartedAtRef = useRef(new Date().toISOString())
  const winProbRef = useRef(50)
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
        let move: Move
        if (mode === 'arena') {
          const spec = getBot(arenaBots[turn])
          move = await botMove({
            data: state,
            engine: spec.engine,
            depth: spec.depth,
            nodeLimit: spec.nodeLimit,
            sims: spec.sims,
            gen2: spec.gen2,
          })
        } else {
          const botSpec = mode === 'bot' ? getBot(botId) : undefined
          move = await botMove({
            data: state,
            engine: botSpec?.engine,
            depth: botSpec?.depth,
            nodeLimit: botSpec?.nodeLimit,
            sims: botSpec?.sims,
            gen2: botSpec?.gen2,
          })
        }
        if (!cancelled) {
          replayFramesRef.current.push({
            ply: replayFramesRef.current.length,
            turn: state.turn,
            state,
            action: move,
            win_prob_south: winProbRef.current,
          })
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
  }, [state, mode, arenaBots, botId, isAutoTurn])

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

  // Keyboard shortcuts: R=reset, ?=help.
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      switch (e.key) {
        case 'r': case 'R':
          reset()
          setShowHelp(false)
          break
        case '?':
          setShowHelp((s) => !s)
          break
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [])

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

  // Keep winProbRef in sync so bot effect closure can read fresh value.
  useEffect(() => { winProbRef.current = southWin }, [southWin])

  // Auto-save completed game to localStorage.
  useEffect(() => {
    if (!state.winner || replayFramesRef.current.length === 0) return
    const engines: { south: ReplayEngine; north: ReplayEngine } =
      mode === 'bot'
        ? { south: 'human', north: getBot(botId).engine as ReplayEngine }
        : mode === 'arena'
          ? {
              south: getBot(arenaBots.south).engine as ReplayEngine,
              north: getBot(arenaBots.north).engine as ReplayEngine,
            }
          : { south: 'human', north: 'human' }
    const record: ReplayRecord = {
      id: gameIdRef.current,
      mode,
      engines,
      winner: state.winner,
      ply_count: replayFramesRef.current.length,
      started_at: gameStartedAtRef.current,
      frames: [...replayFramesRef.current],
    }
    saveReplay(record)
    setSavedReplay(record)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.winner])

  function pushFrame(s: GameState, action: Move) {
    replayFramesRef.current.push({
      ply: replayFramesRef.current.length,
      turn: s.turn,
      state: s,
      action,
      win_prob_south: winProbRef.current,
    })
  }

  function enterReplay(record: ReplayRecord) {
    setReplayMode({ record, idx: 0 })
    setPanel(null)
  }

  function exitReplay() {
    setReplayMode(null)
    reset()
  }

  function handleChangeMode() {
    if (movesMade > 0 && !state.winner) {
      setConfirmLeave(true)
      return
    }
    navigate({ to: '/play' })
  }

  function handleMove(to: Cell) {
    pushFrame(state, { type: 'move', to })
    setState((s) => applyMove(s, { type: 'move', to }))
    setMovesMade((n) => n + 1)
  }

  function handlePlaceWall(wall: Wall) {
    pushFrame(state, { type: 'wall', wall })
    setState((s) => applyMove(s, { type: 'wall', wall }))
    setMovesMade((n) => n + 1)
  }

  function reset() {
    plyRef.current = 0
    replayFramesRef.current = []
    gameIdRef.current = makeId()
    gameStartedAtRef.current = new Date().toISOString()
    setSavedReplay(null)
    setState(initialState())
    setThinking(false)
    setBotError(null)
    setMovesMade(0)
    setConfirmLeave(false)
  }

  function handleBotId(next: string) {
    if (next === botId) return
    if (movesMade > 0 && !state.winner) {
      setConfirmBotChange(next)
      return
    }
    setBotId(next)
    reset()
  }

  function confirmBotSwitch() {
    if (!confirmBotChange) return
    setBotId(confirmBotChange)
    setConfirmBotChange(null)
    reset()
  }

  function handleArenaBot(side: Side, id: string) {
    if (arenaBots[side] === id) return
    setArenaBots((b) => ({ ...b, [side]: id }))
    reset()
  }

  const legalTargets = state.winner ? [] : pawnMoves(state, state.turn)
  const canPlace = (wall: Wall) =>
    !state.winner && canPlaceWall(state, wall, state.turn)

  // Replay mode derived values
  const replayTotal = replayMode ? replayMode.record.frames.length : 0
  const replayBoardState = replayMode
    ? replayMode.idx < replayTotal
      ? replayMode.record.frames[replayMode.idx].state
      : applyMove(
          replayMode.record.frames[replayTotal - 1].state,
          replayMode.record.frames[replayTotal - 1].action,
        )
    : null

  // Bot recommendation for current replay position (fetched async)
  useEffect(() => {
    if (!replayMode || !replayBoardState || replayBoardState.winner) {
      setReplayBotMove(undefined)
      return
    }
    let cancelled = false
    setReplayBotMove(undefined)
    const spec = getBot(BROWSER_MAX_BOT_ID)
    botMove({
      data: replayBoardState,
      engine: spec.engine,
      depth: spec.depth,
      nodeLimit: spec.nodeLimit,
      sims: spec.sims,
      gen2: spec.gen2,
    })
      .then((move) => { if (!cancelled) setReplayBotMove(move) })
      .catch(() => {})
    return () => { cancelled = true }
  // replayBoardState changes every render — depend on idx + record id instead
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayMode?.record.id, replayMode?.idx])

  // Bot help: fetch best move recommendation for the human during vs-bot play
  useEffect(() => {
    if (!botHelp || mode !== 'bot' || !isHumanTurn || state.winner || replayMode) {
      setHelpMove(undefined)
      return
    }
    let cancelled = false
    setHelpMove(undefined)
    const spec = getBot(BROWSER_MAX_BOT_ID)
    botMove({
      data: state,
      engine: spec.engine,
      depth: spec.depth,
      nodeLimit: spec.nodeLimit,
      sims: spec.sims,
      gen2: spec.gen2,
    })
      .then((move) => { if (!cancelled) setHelpMove(move) })
      .catch(() => {})
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botHelp, state, isHumanTurn, mode, replayMode])

  // Win prob to display: in replay mode use stored frame value, else live analysis
  const displayWinProb = replayMode
    ? (replayMode.idx < replayTotal
        ? replayMode.record.frames[replayMode.idx].win_prob_south
        : (replayMode.record.winner === 'south' ? 100 : 0))
    : southWin

  const botLabel = getBot(botId).label
  const sideLabel = (s: typeof state.turn) =>
    mode === 'bot'
      ? s === 'south' ? 'You' : botLabel
      : mode === 'arena'
        ? getBot(arenaBots[s]).label
        : s === 'south' ? 'Player 1' : 'Player 2'
  const southIcon = mode === 'arena' ? Bot : Users
  const northIcon = mode === 'friend' ? Users : Bot

  const status = replayMode
    ? `Replay — ply ${replayMode.idx} / ${replayTotal}`
    : state.winner
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
            walls={replayMode ? replayBoardState!.wallsLeft.north : state.wallsLeft.north}
            active={replayMode
              ? replayBoardState!.turn === 'north' && !replayBoardState!.winner
              : state.turn === 'north' && !state.winner}
          />
          <div className="flex-1 min-h-0">
            <WinMeter
              south={displayWinProb}
              southLabel={sideLabel('south')}
              northLabel={sideLabel('north')}
              loading={replayMode ? false : analyzing}
            />
          </div>
          <PlayerCard
            side="south"
            label={sideLabel('south')}
            icon={southIcon}
            walls={replayMode ? replayBoardState!.wallsLeft.south : state.wallsLeft.south}
            active={replayMode
              ? replayBoardState!.turn === 'south' && !replayBoardState!.winner
              : state.turn === 'south' && !state.winner}
          />
        </div>

        {/* board column */}
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          {/* top bar: status + controls (controls hidden on desktop, live in right sidebar) */}
          <div className="flex flex-shrink-0 items-center justify-between gap-2">
            <span className="island-kicker flex items-center gap-1.5" aria-live="polite" aria-atomic="true">
              {thinking && !replayMode && (
                <Loader2 size={13} className="animate-spin text-primary" />
              )}
              {status}
            </span>
            <div className="flex items-center gap-2 lg:hidden">
              <button
                type="button"
                onClick={handleChangeMode}
                className="text-xs font-semibold text-muted-foreground hover:text-foreground"
              >
                ← Mode
              </button>
              <Button
                variant="outline"
                size="icon-sm"
                className="rounded-full"
                onClick={() => setPanel({ type: 'list' })}
                aria-label="Replay history"
              >
                <History size={14} />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                className="rounded-full"
                onClick={() => setShowHelp(true)}
                aria-label="How to play"
              >
                <HelpCircle size={14} />
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
                onClick={reset}
              >
                <RotateCcw size={13} />
                Reset
              </Button>
            </div>
          </div>

          {botError && (
            <p className="flex-shrink-0 text-xs font-semibold text-destructive" role="alert">
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

          {/* board: fills remaining space, square = min(width, height) via container query */}
          <div className="flex min-h-0 flex-1 items-center justify-center" style={{ containerType: 'size' }}>
            <div
              className={['transition-opacity duration-300', thinking ? 'opacity-60' : ''].join(' ')}
              style={{ width: 'min(100cqw, 100cqh)', height: 'min(100cqw, 100cqh)' }}
            >
              <Board
                state={replayMode ? replayBoardState! : state}
                legalTargets={replayMode ? [] : legalTargets}
                interactive={replayMode ? false : interactive}
                canPlace={replayMode ? () => false : canPlace}
                onMove={replayMode ? () => {} : handleMove}
                onPlaceWall={replayMode ? () => {} : handlePlaceWall}
                goalTopLabel="P2"
                goalBottomLabel="P1"
                bestMove={replayMode ? replayBotMove : (botHelp ? helpMove : undefined)}
              />
            </div>
          </div>

          {replayMode ? (
            <div className="flex flex-shrink-0 flex-col gap-2 lg:hidden">
              <input
                type="range"
                min={0}
                max={replayTotal}
                value={replayMode.idx}
                onChange={(e) => setReplayMode((r) => r ? { ...r, idx: Number(e.target.value) } : null)}
                className="w-full accent-primary"
              />
              <div className="flex items-center justify-center gap-2">
                <Button
                  variant="outline"
                  size="icon-sm"
                  className="rounded-full"
                  onClick={() => setReplayMode((r) => r ? { ...r, idx: Math.max(0, r.idx - 1) } : null)}
                  disabled={replayMode.idx === 0}
                >
                  <ChevronLeft size={14} />
                </Button>
                <Button
                  variant="outline"
                  size="icon-sm"
                  className="rounded-full"
                  onClick={() => setReplayMode((r) => r ? { ...r, idx: Math.min(replayTotal, r.idx + 1) } : null)}
                  disabled={replayMode.idx >= replayTotal}
                >
                  <ChevronRight size={14} />
                </Button>
                <Button
                  size="sm"
                  className="rounded-full gap-1.5"
                  onClick={exitReplay}
                >
                  <PlusSquare size={13} /> New Game
                </Button>
              </div>
            </div>
          ) : mode === 'arena' ? (
            <div className="flex flex-shrink-0 lg:hidden">
              <ArenaConfig bots={arenaBots} onChange={handleArenaBot} />
            </div>
          ) : (
            <div className="flex flex-col flex-shrink-0 gap-2 lg:hidden">
              {mode === 'bot' && (
                <div className="flex flex-shrink-0 flex-wrap items-center justify-center gap-2">
                  <div className="flex items-center gap-1.5">
                    <Bot size={14} className="text-muted-foreground" />
                    <select
                      value={botId}
                      onChange={(e) => handleBotId(e.target.value)}
                      className="rounded-full border border-border bg-secondary px-2 py-0.5 text-xs font-medium text-foreground focus:outline-none"
                    >
                      {BOTS.map((b) => (
                        <option key={b.id} value={b.id}>{b.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="inline-flex overflow-hidden rounded-full border border-border">
                    <ToggleBtn active={botHelp} onClick={() => setBotHelp((b) => !b)}>
                      <Lightbulb size={14} /> Bot Help
                    </ToggleBtn>
                  </div>
                </div>
              )}

              <p className="flex-shrink-0 text-center text-xs text-muted-foreground">
                Tap near a highlighted square center to move; tap near a grid gap to place a wall.
              </p>
              <span className="sr-only" aria-live="polite" aria-atomic="true">
                Board controls infer moves from square centers and walls from grid gaps.
              </span>
              <p className="flex-shrink-0 text-center text-xs text-muted-foreground">
                <kbd className="rounded border border-border bg-secondary px-1 font-mono text-[0.65rem] font-semibold">R</kbd>
                {' '}reset{' · '}
                <kbd className="rounded border border-border bg-secondary px-1 font-mono text-[0.65rem] font-semibold">?</kbd>
                {' '}help
              </p>
            </div>
          )}
        </div>

        {/* right sidebar: controls (desktop only) */}
        <div className="hidden w-56 flex-shrink-0 flex-col items-center gap-3 pt-1 lg:flex">
          {replayMode && (
            <InlineReplayPanel
              record={replayMode.record}
              idx={replayMode.idx}
              onIdx={(i) => setReplayMode((r) => r ? { ...r, idx: i } : null)}
              onExit={exitReplay}
            />
          )}
          {/* Mode on left, icon buttons on right */}
          <div className={['flex w-full items-center justify-between', replayMode ? 'hidden' : ''].join(' ')}>
            <button
              type="button"
              onClick={handleChangeMode}
              className="text-xs font-semibold text-muted-foreground hover:text-foreground"
            >
              ← Mode
            </button>
            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="icon-sm"
                className="rounded-full"
                onClick={() => setPanel({ type: 'list' })}
                aria-label="Replay history"
              >
                <History size={14} />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                className="rounded-full"
                onClick={() => setShowHelp(true)}
                aria-label="How to play"
              >
                <HelpCircle size={14} />
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
                onClick={reset}
              >
                <RotateCcw size={12} />
                Reset
              </Button>
            </div>
          </div>
          {!replayMode && (
            <>
              <div className="w-full border-t border-border" />
              {mode === 'arena' ? (
                <ArenaConfig bots={arenaBots} onChange={handleArenaBot} compact />
              ) : (
                <>
                  {mode === 'bot' && (
                    <div className="flex w-full items-center gap-1.5">
                      <Bot size={13} className="flex-shrink-0 text-muted-foreground" />
                      <select
                        value={botId}
                        onChange={(e) => handleBotId(e.target.value)}
                        className="w-full rounded-full border border-border bg-secondary px-2 py-0.5 text-xs font-medium text-foreground focus:outline-none"
                      >
                        {BOTS.map((b) => (
                          <option key={b.id} value={b.id}>{b.label}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  {mode === 'bot' && (
                    <div className="inline-flex overflow-hidden rounded-full border border-border">
                      <ToggleBtn compact active={botHelp} onClick={() => setBotHelp((b) => !b)}>
                        <Lightbulb size={13} /> Bot Help
                      </ToggleBtn>
                    </div>
                  )}

                  <p className="text-center text-xs text-muted-foreground">
                    Click near a highlighted square center to move; click near a grid gap to place a wall.
                  </p>
                  <span className="sr-only" aria-live="polite" aria-atomic="true">
                    Board controls infer moves from square centers and walls from grid gaps.
                  </span>
                  <p className="text-center text-xs text-muted-foreground leading-relaxed">
                    <kbd className="rounded border border-border bg-secondary px-1 font-mono text-[0.65rem] font-semibold">R</kbd>
                    {' '}reset{' · '}
                    <kbd className="rounded border border-border bg-secondary px-1 font-mono text-[0.65rem] font-semibold">?</kbd>
                    {' '}help
                  </p>
                </>
              )}
            </>
          )}
        </div>
      </div>

      {state.winner && !replayMode && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
          <div className="rise-in rounded-2xl border bg-card text-card-foreground shadow-sm p-6 text-center">
            <p className="display-title mb-4 text-2xl font-bold text-foreground">
              {mode === 'arena'
                ? `${sideLabel(state.winner)} wins! 🎉`
                : state.winner === 'south'
                  ? mode === 'bot' ? 'You win! 🎉' : 'Player 1 wins! 🎉'
                  : mode === 'bot' ? 'Bot wins.' : 'Player 2 wins! 🎉'}
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <Button
                className="rounded-full px-6 py-2.5 text-sm font-bold"
                onClick={reset}
              >
                Play again
              </Button>
              {savedReplay && (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-full gap-1.5"
                    onClick={() => enterReplay(savedReplay)}
                  >
                    ▶ Replay
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-full gap-1.5"
                    onClick={() => downloadReplay(savedReplay)}
                  >
                    <Download size={14} /> .jsonl
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {confirmBotChange && !state.winner && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
          <div className="rise-in rounded-2xl border bg-card text-card-foreground shadow-sm p-4" style={{ animationDuration: '300ms' }}>
            <p className="mb-3 text-sm text-foreground">
              Switch to {getBot(confirmBotChange).label}? Current match will reset.
            </p>
            <div className="flex justify-center gap-2">
              <Button
                size="sm"
                className="rounded-full"
                onClick={confirmBotSwitch}
              >
                Switch & reset
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
                onClick={() => setConfirmBotChange(null)}
              >
                Keep playing
              </Button>
            </div>
          </div>
        </div>
      )}

      {confirmLeave && !state.winner && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
          <div className="rise-in rounded-2xl border bg-card text-card-foreground shadow-sm p-4" style={{ animationDuration: '300ms' }}>
            <p className="mb-3 text-sm text-foreground">
              Leave this game? Progress will be lost.
            </p>
            <div className="flex justify-center gap-2">
              <Button
                size="sm"
                className="rounded-full"
                onClick={() => navigate({ to: '/play' })}
              >
                Leave game
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
                onClick={() => setConfirmLeave(false)}
              >
                Keep playing
              </Button>
            </div>
          </div>
        </div>
      )}

      {panel?.type === 'list' && (
        <ReplayList
          onView={(record) => enterReplay(record)}
          onClose={() => setPanel(null)}
        />
      )}

      {showHelp && <HowToPlay onClose={() => setShowHelp(false)} />}
    </main>
  )
}

function cellNote(cell: Cell): string {
  return String.fromCharCode(96 + cell.c) + cell.r
}

function moveNote(action: Move): string {
  if (action.type === 'move') return cellNote(action.to)
  const { wall } = action
  return `@${String.fromCharCode(96 + wall.c)}${wall.r}${wall.o}`
}

function InlineReplayPanel({
  record,
  idx,
  onIdx,
  onExit,
}: {
  record: ReplayRecord
  idx: number
  onIdx: (i: number) => void
  onExit: () => void
}) {
  const total = record.frames.length
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(800)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const activeRowRef = useRef<HTMLDivElement | null>(null)
  const onIdxRef = useRef(onIdx)
  useEffect(() => { onIdxRef.current = onIdx }, [onIdx])

  useEffect(() => {
    if (!playing) return
    timerRef.current = setTimeout(() => {
      if (idx >= total) {
        setPlaying(false)
        return
      }
      onIdxRef.current(idx + 1)
    }, speed)
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [playing, idx, speed, total])

  useEffect(() => {
    if (idx >= total) setPlaying(false)
  }, [idx, total])

  useEffect(() => {
    activeRowRef.current?.scrollIntoView({ block: 'nearest' })
  }, [idx])

  function togglePlay() {
    if (idx >= total) {
      onIdx(0)
      setPlaying(true)
    } else {
      setPlaying((p) => !p)
    }
  }

  // Group into pairs: each row = (rowNum, southPlyIdx, northPlyIdx)
  const rows: Array<{ n: number; s: number | null; no: number | null }> = []
  for (let i = 0; i < total; i += 2) {
    const sf = record.frames[i]
    const nf = i + 1 < total ? record.frames[i + 1] : null
    rows.push({
      n: Math.floor(i / 2) + 1,
      s: sf.turn === 'south' ? i : null,
      no: nf?.turn === 'north' ? i + 1 : null,
    })
  }

  const activeRow = Math.floor(idx / 2)

  return (
    <div className="flex h-full w-full flex-col gap-2">
      {/* header */}
      <div className="flex w-full flex-shrink-0 items-center justify-between">
        <Button
          size="sm"
          className="gap-1"
          onClick={onExit}
        >
          <PlusSquare size={13} /> New Game
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="rounded-full gap-1"
          onClick={() => downloadReplay(record)}
        >
          <Download size={11} /> .jsonl
        </Button>
      </div>

      <div className="w-full border-t border-border" />

      {/* move list */}
      <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-border" style={{ scrollbarWidth: 'thin' }}>
        {/* header row */}
        <div className="sticky top-0 grid grid-cols-[1.5rem_1fr_1fr] bg-secondary px-2 py-1 text-[0.6rem] font-bold uppercase tracking-wide text-muted-foreground">
          <span>#</span>
          <span className="text-center" style={{ color: 'var(--primary)' }}>S</span>
          <span className="text-center" style={{ color: 'var(--pawn-north)' }}>N</span>
        </div>
        {rows.map((row) => {
          const isActiveRow = row.n - 1 === activeRow
          return (
            <div
              key={row.n}
              ref={isActiveRow ? activeRowRef : null}
              className="grid grid-cols-[1.5rem_1fr_1fr] px-1 py-0.5 text-[0.7rem]"
            >
              <span className="flex items-center font-semibold text-muted-foreground opacity-60">{row.n}</span>
              {/* south cell */}
              <button
                type="button"
                onClick={() => row.s !== null && onIdx(row.s)}
                disabled={row.s === null}
                className={[
                  'rounded px-1 py-0.5 font-mono text-left transition disabled:opacity-30',
                  idx === row.s
                    ? 'bg-primary font-bold text-primary-foreground'
                    : 'hover:bg-secondary text-foreground',
                ].join(' ')}
              >
                {row.s !== null ? moveNote(record.frames[row.s].action) : '—'}
              </button>
              {/* north cell */}
              <button
                type="button"
                onClick={() => row.no !== null && onIdx(row.no)}
                disabled={row.no === null}
                className={[
                  'rounded px-1 py-0.5 font-mono text-left transition disabled:opacity-30',
                  idx === row.no
                    ? 'bg-primary font-bold text-primary-foreground'
                    : 'hover:bg-secondary text-foreground',
                ].join(' ')}
              >
                {row.no !== null ? moveNote(record.frames[row.no].action) : '—'}
              </button>
            </div>
          )
        })}
        {/* final state row */}
        <div
          ref={idx === total ? activeRowRef : null}
          className={[
            'px-2 py-1 text-center text-[0.65rem] font-semibold capitalize',
            idx === total ? 'bg-primary text-primary-foreground' : 'text-muted-foreground',
          ].join(' ')}
        >
          {record.winner ? `${record.winner} wins` : 'Draw'}
        </div>
      </div>

      {/* playback controls */}
      <div className="flex flex-shrink-0 flex-col gap-1.5">
        <input
          type="range"
          min={0}
          max={total}
          value={idx}
          onChange={(e) => {
            setPlaying(false)
            onIdx(Number(e.target.value))
          }}
          className="w-full accent-primary"
        />
        <div className="flex items-center justify-center gap-1.5">
          <button
            type="button"
            onClick={() => { setPlaying(false); onIdx(0) }}
            className="text-sm text-muted-foreground transition hover:text-foreground"
            title="Go to start"
          >
            ⏮
          </button>
          <Button
            variant="outline"
            size="icon-sm"
            className="rounded-full"
            onClick={() => { setPlaying(false); onIdx(Math.max(0, idx - 1)) }}
            disabled={idx === 0}
          >
            <ChevronLeft size={13} />
          </Button>
          <Button
            size="icon-sm"
            className="rounded-full"
            onClick={togglePlay}
          >
            {playing ? <Pause size={13} /> : <Play size={13} />}
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            className="rounded-full"
            onClick={() => { setPlaying(false); onIdx(Math.min(total, idx + 1)) }}
            disabled={idx >= total}
          >
            <ChevronRight size={13} />
          </Button>
          <button
            type="button"
            onClick={() => { setPlaying(false); onIdx(total) }}
            className="text-sm text-muted-foreground transition hover:text-foreground"
            title="Go to end"
          >
            ⏭
          </button>
        </div>
        <div className="flex items-center justify-center gap-1">
          {([300, 800, 1600] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSpeed(s)}
              className={[
                'rounded-full px-2 py-0.5 text-[0.6rem] font-semibold transition',
                speed === s
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              {s === 300 ? 'Fast' : s === 800 ? 'Normal' : 'Slow'}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// Per-side bot pickers for arena: choose any catalog bot for South and North
// (same bot on both sides is allowed). Uses dropdowns so the list scales as
// more variations are added.
function ArenaConfig({
  bots,
  onChange,
  compact,
}: {
  bots: ArenaBots
  onChange: (side: Side, id: string) => void
  compact?: boolean
}) {
  return (
    <div className="flex flex-wrap items-end justify-center gap-x-3 gap-y-2">
      <ArenaSidePicker label="South (P1)" side="south" value={bots.south} onChange={onChange} compact={compact} />
      <span className="pb-2 text-xs font-bold text-muted-foreground">vs</span>
      <ArenaSidePicker label="North (P2)" side="north" value={bots.north} onChange={onChange} compact={compact} />
    </div>
  )
}

function ArenaSidePicker({
  label,
  side,
  value,
  onChange,
  compact,
}: {
  label: string
  side: Side
  value: string
  onChange: (side: Side, id: string) => void
  compact?: boolean
}) {
  return (
    <label className="flex flex-col items-center gap-1">
      <span className="text-[0.65rem] font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-secondary pl-2.5 pr-1.5">
        <Bot size={compact ? 13 : 14} className="flex-shrink-0 text-muted-foreground" />
        <select
          value={value}
          onChange={(e) => onChange(side, e.target.value)}
          className={[
            'cursor-pointer bg-transparent font-semibold text-foreground outline-none',
            compact ? 'py-1.5 text-[0.7rem]' : 'py-2 text-xs',
          ].join(' ')}
        >
          {BOTS.map((b: BotSpec) => (
            <option key={b.id} value={b.id}>
              {b.label}
            </option>
          ))}
        </select>
      </div>
    </label>
  )
}

function ToggleBtn({
  active,
  disabled,
  compact,
  onClick,
  children,
}: {
  active: boolean
  disabled?: boolean
  compact?: boolean
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
        'inline-flex items-center gap-1 font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 whitespace-nowrap',
        compact ? 'px-3 py-1.5 text-[0.7rem]' : 'px-4 py-2 text-xs',
        active
          ? 'bg-primary text-primary-foreground'
          : 'bg-secondary text-foreground hover:bg-accent',
        flash ? 'ring-2 ring-primary ring-offset-1' : '',
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
      ? 'bg-[var(--pawn-south)] border border-[var(--pawn-south-edge)]'
      : 'bg-[var(--pawn-north)] border border-[var(--pawn-north-edge)]'

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
          <div className="flex items-center gap-1.5 text-sm font-bold text-foreground">
            <Icon size={15} />
            {label}
          </div>
          <div className="text-xs text-muted-foreground">
            Walls left: <span className="font-semibold">{walls}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
