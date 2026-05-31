import { Link, createFileRoute } from '@tanstack/react-router'
import { Bot, Users } from 'lucide-react'

export const Route = createFileRoute('/play')({ component: Play })

const MODES = [
  {
    mode: 'bot',
    icon: Bot,
    title: 'vs Bot',
    body: 'Play against the computer. Good for learning the board.',
  },
  {
    mode: 'friend',
    icon: Users,
    title: 'vs Friend',
    body: 'Two players, one screen. Pass and play, turn by turn.',
  },
] as const

function Play() {
  return (
    <main className="page-wrap flex min-h-[70vh] flex-col items-center justify-center px-4 py-14">
      <div className="w-full max-w-2xl text-center">
        <p className="island-kicker mb-2">Choose a mode</p>
        <h1 className="display-title mb-8 text-4xl font-bold text-[var(--sea-ink)] sm:text-5xl">
          Who are you playing?
        </h1>

        <div className="grid gap-4 sm:grid-cols-2">
          {MODES.map(({ mode, icon: Icon, title, body }) => (
            <Link
              key={mode}
              to="/game"
              search={{ mode }}
              className="island-shell feature-card rise-in flex flex-col items-center gap-3 rounded-xl p-7 text-center no-underline"
            >
              <span className="flex h-14 w-14 items-center justify-center rounded-full bg-[var(--icon-bg)] text-[var(--accent-text)]">
                <Icon size={26} />
              </span>
              <h2 className="m-0 text-xl font-bold text-[var(--sea-ink)]">
                {title}
              </h2>
              <p className="m-0 text-sm text-[var(--sea-ink-soft)]">{body}</p>
            </Link>
          ))}
        </div>

        <Link
          to="/"
          className="mt-8 inline-block text-sm font-semibold text-[var(--sea-ink-soft)]"
        >
          ← Back
        </Link>
      </div>
    </main>
  )
}
