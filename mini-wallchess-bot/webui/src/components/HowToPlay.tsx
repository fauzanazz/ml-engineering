import { useEffect } from 'react'
import { X } from 'lucide-react'

const QUICK_RULES = [
  {
    n: '01',
    title: 'Move your pawn',
    body: 'On your turn, move one square orthogonally. Jump over your opponent if adjacent with no wall behind them.',
  },
  {
    n: '02',
    title: 'Place a wall',
    body: 'Instead of moving, place a 2-cell wall anywhere on the board. Walls block paths — use them wisely.',
  },
  {
    n: '03',
    title: 'Block, but never trap',
    body: 'Walls cannot completely block any player from reaching their goal row. Every placement is validated.',
  },
  {
    n: '04',
    title: 'First to cross wins',
    body: 'Player 1 starts at the bottom, Player 2 at the top. First to reach the opposite side wins.',
  },
]

const WALL_RULES = [
  ['ok', 'Walls occupy the gaps between cells, spanning exactly 2 cells.'],
  ['ok', 'Walls can be horizontal (block N–S) or vertical (block E–W).'],
  ['ok', 'Two walls cannot overlap or cross each other.'],
  ['no', 'A placement that leaves any player with no path to their goal row is illegal.'],
] as const

export default function HowToPlay({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center overflow-y-auto bg-[var(--overlay)] px-4 py-10 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="How to play Wall Chess"
      aria-describedby="howtoplay-desc"
      onClick={onClose}
    >
      <div
        className="island-shell rise-in relative w-full max-w-2xl rounded-[1.75rem] p-6 sm:p-9"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute right-5 top-5 rounded-full p-2 text-[var(--sea-ink-soft)] transition hover:bg-[var(--link-bg-hover)] hover:text-[var(--sea-ink)]"
        >
          <X size={20} />
        </button>

        <p className="island-kicker mb-2">How to Play</p>
        <h2 className="display-title mb-2 text-3xl font-bold text-[var(--sea-ink)]">
          Wall Chess in four rules
        </h2>
        <p id="howtoplay-desc" className="mb-6 max-w-xl text-sm text-[var(--sea-ink-soft)]">
          Race your pawn to the far side. Each turn: move one step, or drop a
          wall to slow your opponent down.
        </p>

        <div className="grid gap-3 sm:grid-cols-2">
          {QUICK_RULES.map((r) => (
            <article
              key={r.n}
              className="feature-card rounded-2xl p-4"
            >
              <span className="mb-1 block text-xs font-bold text-[var(--sea-ink-soft)]">
                {r.n}
              </span>
              <h3 className="mb-1 text-sm font-bold text-[var(--sea-ink)]">
                {r.title}
              </h3>
              <p className="m-0 text-sm leading-6 text-[var(--sea-ink-soft)]">
                {r.body}
              </p>
            </article>
          ))}
        </div>

        <div className="mt-5 rounded-2xl border border-[var(--line)] p-4">
          <p className="island-kicker mb-2">Wall Rules</p>
          <ul className="m-0 space-y-1.5 text-sm text-[var(--sea-ink-soft)]">
            {WALL_RULES.map(([kind, text]) => (
              <li key={text} className="flex gap-2">
                <span
                  className={
                    kind === 'ok'
                      ? 'font-bold text-[var(--palm)]'
                      : 'font-bold text-[var(--danger)]'
                  }
                  aria-hidden="true"
                >
                  {kind === 'ok' ? '✓' : '✗'}
                </span>
                <span>{text}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-5 rounded-2xl border border-[var(--line)] p-4">
          <p className="island-kicker mb-2">Keyboard shortcuts</p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3">
            {([
              ['M', 'Move pawn'],
              ['W', 'Place wall'],
              ['H / V', 'Horiz / Vert'],
              ['R', 'Reset game'],
              ['?', 'Toggle help'],
            ] as const).map(([key, label]) => (
              <div key={key} className="flex items-center gap-2">
                <kbd className="rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1.5 py-0.5 font-mono text-xs font-semibold text-[var(--sea-ink)] shadow-[var(--shadow-chip)]">
                  {key}
                </kbd>
                <span className="text-xs text-[var(--sea-ink-soft)]">{label}</span>
              </div>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="mt-6 w-full rounded-full border border-[var(--accent-border)] bg-[var(--icon-bg)] px-5 py-3 text-sm font-semibold text-[var(--accent-text)] transition hover:-translate-y-0.5 hover:bg-[var(--icon-bg-hover)]"
        >
          Got it
        </button>
      </div>
    </div>
  )
}
