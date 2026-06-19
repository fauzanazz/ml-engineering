type Props = {
  south: number
  southLabel?: string
  northLabel?: string
  loading?: boolean
}

export default function WinMeter({
  south,
  southLabel = 'Player 1',
  northLabel = 'Player 2',
  loading = false,
}: Props) {
  const s = Math.max(0, Math.min(100, Math.round(south)))
  const n = 100 - s
  return (
    <div
      className={[
        'flex h-full flex-col items-center py-1 transition-opacity',
        loading ? 'opacity-60' : 'opacity-100',
      ].join(' ')}
      aria-label={`Win chance: ${northLabel} ${n} percent, ${southLabel} ${s} percent`}
    >
      <span className="island-kicker mb-2 flex-shrink-0 text-[0.65rem]">Win chance</span>

      <div className="flex min-h-0 flex-1 w-full items-stretch gap-3">
        {/* vertical bar */}
        <div className="flex w-3.5 flex-col overflow-hidden rounded-full border border-[var(--line)] bg-[var(--chip-bg)]">
          <div
            className="w-full bg-[var(--pawn-north)] transition-[height] duration-500 ease-out"
            style={{ height: `${n}%` }}
          />
          <div
            className="w-full bg-[var(--lagoon)] transition-[height] duration-500 ease-out"
            style={{ height: `${s}%` }}
          />
        </div>

        {/* percentages */}
        <div className="flex flex-1 flex-col justify-between text-sm font-bold text-[var(--sea-ink)]">
          <span>{n}%</span>
          <span>{s}%</span>
        </div>
      </div>
    </div>
  )
}
