import { useState } from 'react'
import { Download, Play, Trash2, X } from 'lucide-react'
import { type ReplayRecord, deleteReplay, downloadReplay, listReplays } from '../game/replay'

type Props = {
  onView: (record: ReplayRecord) => void
  onClose: () => void
}

export default function ReplayList({ onView, onClose }: Props) {
  const [replays, setReplays] = useState<ReplayRecord[]>(() => listReplays())

  function handleDelete(id: string) {
    deleteReplay(id)
    setReplays(listReplays())
  }

  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-[var(--overlay)] backdrop-blur-sm">
      <div
        className="island-shell rise-in flex w-full max-w-lg flex-col gap-3 rounded-2xl p-4"
        style={{ maxHeight: 'calc(100dvh - 4rem)' }}
      >
        <div className="flex flex-shrink-0 items-center justify-between">
          <h2 className="text-sm font-bold text-[var(--sea-ink)]">
            Saved Replays
            <span className="ml-2 text-xs font-normal text-[var(--sea-ink-soft)]">
              ({replays.length} / 30)
            </span>
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-1.5 text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
          >
            <X size={14} />
          </button>
        </div>

        {replays.length === 0 ? (
          <p className="py-8 text-center text-sm text-[var(--sea-ink-soft)]">
            No replays saved yet. Finish a game to record one.
          </p>
        ) : (
          <div
            className="flex flex-col gap-2 overflow-y-auto"
            style={{ maxHeight: 'calc(100dvh - 12rem)' }}
          >
            {replays.map((r) => (
              <div
                key={r.id}
                className="flex items-center gap-2 rounded-xl border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-xs font-bold capitalize text-[var(--sea-ink)]">
                      {r.mode}
                    </span>
                    {r.winner && (
                      <span className="rounded-full bg-[var(--icon-bg)] px-1.5 py-0.5 text-[0.6rem] font-semibold capitalize text-[var(--accent-text)]">
                        {r.winner} wins
                      </span>
                    )}
                    <span className="text-[0.65rem] text-[var(--sea-ink-soft)]">
                      {r.ply_count} plies
                    </span>
                  </div>
                  <div className="text-[0.65rem] text-[var(--sea-ink-soft)]">
                    {r.started_at.slice(0, 16).replace('T', ' ')} ·{' '}
                    {r.engines.south} vs {r.engines.north}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => onView(r)}
                  className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-1.5 text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
                  title="View replay"
                >
                  <Play size={12} />
                </button>
                <button
                  type="button"
                  onClick={() => downloadReplay(r)}
                  className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-1.5 text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
                  title="Download .jsonl"
                >
                  <Download size={12} />
                </button>
                <button
                  type="button"
                  onClick={() => handleDelete(r.id)}
                  className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] p-1.5 text-[var(--sea-ink)] transition hover:border-[var(--danger,#e53e3e)]"
                  title="Delete"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        <p className="flex-shrink-0 text-center text-[0.65rem] text-[var(--sea-ink-soft)]">
          JSONL format: 1 line per ply — state, action, win%, outcome. Ready for bot training.
        </p>
      </div>
    </div>
  )
}
