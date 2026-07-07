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
        className="rise-in flex w-full max-w-lg flex-col gap-3 rounded-2xl border bg-card text-card-foreground shadow-sm p-4"
        style={{ maxHeight: 'calc(100dvh - 4rem)' }}
      >
        <div className="flex flex-shrink-0 items-center justify-between">
          <h2 className="text-sm font-bold text-foreground">
            Saved Replays
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              ({replays.length} / 30)
            </span>
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-border bg-secondary p-1.5 text-foreground transition hover:border-primary"
          >
            <X size={14} />
          </button>
        </div>

        {replays.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
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
                className="flex items-center gap-2 rounded-xl border border-border bg-secondary px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-xs font-bold capitalize text-foreground">
                      {r.mode}
                    </span>
                    {r.winner && (
                      <span className="rounded-full bg-accent px-1.5 py-0.5 text-[0.6rem] font-semibold capitalize text-primary">
                        {r.winner} wins
                      </span>
                    )}
                    <span className="text-[0.65rem] text-muted-foreground">
                      {r.ply_count} plies
                    </span>
                  </div>
                  <div className="text-[0.65rem] text-muted-foreground">
                    {r.started_at.slice(0, 16).replace('T', ' ')} ·{' '}
                    {r.engines.south} vs {r.engines.north}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => onView(r)}
                  className="rounded-full border border-border bg-secondary p-1.5 text-foreground transition hover:border-primary"
                  title="View replay"
                >
                  <Play size={12} />
                </button>
                <button
                  type="button"
                  onClick={() => downloadReplay(r)}
                  className="rounded-full border border-border bg-secondary p-1.5 text-foreground transition hover:border-primary"
                  title="Download .jsonl"
                >
                  <Download size={12} />
                </button>
                <button
                  type="button"
                  onClick={() => handleDelete(r.id)}
                  className="rounded-full border border-border bg-secondary p-1.5 text-foreground transition hover:border-destructive"
                  title="Delete"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        <p className="flex-shrink-0 text-center text-[0.65rem] text-muted-foreground">
          JSONL format: 1 line per ply — state, action, win%, outcome. Ready for bot training.
        </p>
      </div>
    </div>
  )
}
