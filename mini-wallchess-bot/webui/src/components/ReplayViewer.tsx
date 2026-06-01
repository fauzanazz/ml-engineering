import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Download, Pause, Play, X } from 'lucide-react'
import Board from './Board'
import WinMeter from './WinMeter'
import { type ReplayRecord, downloadReplay } from '../game/replay'
import { applyMove } from '../game/engine'

type Props = {
  record: ReplayRecord
  onClose: () => void
}

export default function ReplayViewer({ record, onClose }: Props) {
  const total = record.frames.length
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(800)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // state at position idx:
  //   0..total-1  → frames[idx].state  (before that ply's action)
  //   total       → result of last action = final board
  const boardState =
    idx < total
      ? record.frames[idx].state
      : applyMove(record.frames[total - 1].state, record.frames[total - 1].action)

  const winProb =
    idx < total
      ? record.frames[idx].win_prob_south
      : record.winner === 'south'
        ? 100
        : 0

  const frame = idx < total ? record.frames[idx] : null

  // autoplay using recursive setTimeout so speed changes take effect immediately
  useEffect(() => {
    if (!playing) return
    timerRef.current = setTimeout(() => {
      setIdx((i) => {
        if (i >= total) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, speed)
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [playing, idx, speed, total])

  useEffect(() => {
    if (idx >= total) setPlaying(false)
  }, [idx, total])

  function step(delta: number) {
    setPlaying(false)
    setIdx((i) => Math.max(0, Math.min(total, i + delta)))
  }

  function togglePlay() {
    if (idx >= total) {
      setIdx(0)
      setPlaying(true)
    } else {
      setPlaying((p) => !p)
    }
  }

  const southLabel = engineLabel('south', record)
  const northLabel = engineLabel('north', record)

  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
      <div
        className="island-shell rise-in relative flex w-full max-w-4xl flex-col gap-3 rounded-2xl p-4"
        style={{ maxHeight: 'calc(100dvh - 2rem)' }}
      >
        {/* header */}
        <div className="flex flex-shrink-0 items-center justify-between">
          <div>
            <span className="text-sm font-bold text-[var(--sea-ink)] capitalize">
              {record.mode} replay
            </span>
            <span className="ml-2 text-xs text-[var(--sea-ink-soft)]">
              {record.started_at.slice(0, 10)} · {record.ply_count} plies
              {record.winner ? ` · ${record.winner} wins` : ''}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => downloadReplay(record)}
              className="inline-flex items-center gap-1.5 rounded-full border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-xs font-semibold text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
            >
              <Download size={12} /> .jsonl
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-1.5 text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* main area */}
        <div className="flex min-h-0 flex-1 gap-3">
          {/* win meter */}
          <div className="hidden w-28 flex-shrink-0 flex-col gap-1 lg:flex">
            <span className="text-center text-[0.6rem] font-semibold text-[var(--sea-ink-soft)]">
              {northLabel}
            </span>
            <div className="min-h-0 flex-1">
              <WinMeter south={winProb} southLabel={southLabel} northLabel={northLabel} />
            </div>
            <span className="text-center text-[0.6rem] font-semibold text-[var(--sea-ink-soft)]">
              {southLabel}
            </span>
          </div>

          {/* board */}
          <div className="flex min-h-0 flex-1 items-center justify-center">
            <div className="aspect-square h-full max-w-full">
              <Board
                state={boardState}
                actionMode="move"
                orientation="h"
                legalTargets={[]}
                interactive={false}
                canPlace={() => false}
                onMove={() => {}}
                onPlaceWall={() => {}}
                goalTopLabel="P2"
                goalBottomLabel="P1"
              />
            </div>
          </div>

          {/* frame info */}
          <div className="hidden w-28 flex-shrink-0 flex-col gap-2 pt-1 lg:flex">
            <div className="island-shell rounded-xl p-3 text-xs">
              <div className="mb-1 font-bold text-[var(--sea-ink)]">
                Ply {idx} / {total}
              </div>
              {frame ? (
                <>
                  <div className="text-[var(--sea-ink-soft)]">
                    Turn: <span className="font-semibold">{frame.turn}</span>
                  </div>
                  <div className="text-[var(--sea-ink-soft)]">
                    Win: <span className="font-semibold">{Math.round(frame.win_prob_south)}%</span> S
                  </div>
                  <div className="mt-1 font-mono text-[0.6rem] text-[var(--sea-ink-soft)]">
                    {frame.action.type === 'move'
                      ? `→ r${frame.action.to.r}c${frame.action.to.c}`
                      : `⊞ r${frame.action.wall.r}c${frame.action.wall.c} ${frame.action.wall.o}`}
                  </div>
                </>
              ) : (
                record.winner && (
                  <div className="font-bold capitalize text-[var(--sea-ink)]">
                    {record.winner} wins!
                  </div>
                )
              )}
            </div>
          </div>
        </div>

        {/* controls */}
        <div className="flex flex-shrink-0 flex-col gap-2">
          <input
            type="range"
            min={0}
            max={total}
            value={idx}
            onChange={(e) => {
              setPlaying(false)
              setIdx(Number(e.target.value))
            }}
            className="w-full accent-[var(--lagoon-deep)]"
          />

          <div className="flex items-center justify-center gap-2">
            <button
              type="button"
              onClick={() => { setPlaying(false); setIdx(0) }}
              className="text-sm text-[var(--sea-ink-soft)] transition hover:text-[var(--sea-ink)]"
              title="Go to start"
            >
              ⏮
            </button>
            <button
              type="button"
              onClick={() => step(-1)}
              disabled={idx === 0}
              className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-2 text-[var(--sea-ink)] transition hover:border-[var(--accent-text)] disabled:opacity-40"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              type="button"
              onClick={togglePlay}
              className="rounded-full bg-[var(--btn-primary-bg)] p-2 text-[var(--btn-primary-fg)] transition hover:opacity-90"
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <button
              type="button"
              onClick={() => step(1)}
              disabled={idx >= total}
              className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-2 text-[var(--sea-ink)] transition hover:border-[var(--accent-text)] disabled:opacity-40"
            >
              <ChevronRight size={14} />
            </button>
            <button
              type="button"
              onClick={() => { setPlaying(false); setIdx(total) }}
              className="text-sm text-[var(--sea-ink-soft)] transition hover:text-[var(--sea-ink)]"
              title="Go to end"
            >
              ⏭
            </button>

            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="ml-2 rounded border border-[var(--line)] bg-[var(--chip-bg)] px-2 py-1 text-xs text-[var(--sea-ink)]"
            >
              <option value={300}>Fast</option>
              <option value={800}>Normal</option>
              <option value={1600}>Slow</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  )
}

function engineLabel(side: 'south' | 'north', record: ReplayRecord): string {
  const e = record.engines[side]
  if (e === 'human') return side === 'south' ? 'Player 1' : 'Player 2'
  if (e === 'net') return 'MCTS Net'
  return 'Alpha-Beta'
}
