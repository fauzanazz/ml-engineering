import { useState } from 'react'
import { Link, createFileRoute } from '@tanstack/react-router'
import { BookOpen, Bot, Crown, Play, Route as RouteIcon } from 'lucide-react'
import HowToPlay from '../components/HowToPlay'
import { Button } from '@/components/ui/button'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const [showRules, setShowRules] = useState(false)

  return (
    <main className="page-wrap flex min-h-[70vh] flex-col items-center justify-center px-4 py-12">
      <section className="rise-in w-full max-w-4xl">
        <div className="mx-auto mb-9 max-w-2xl text-center">
          <p className="island-kicker mb-3">Gameboard</p>
          <h1 className="display-title mb-4 text-4xl font-bold leading-[1.05] tracking-tight text-foreground sm:text-5xl">
            Pick a board. Play the bot.
          </h1>
          <p className="mx-auto max-w-xl text-base text-muted-foreground sm:text-lg">
            Two strategy games share one Rust engine, one search core, and one quiet interface.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <article className="flex flex-col rounded-xl border bg-card p-6 text-card-foreground shadow-sm">
            <div className="mb-5 flex items-start justify-between gap-4">
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <RouteIcon size={24} />
              </span>
              <span className="rounded-full border bg-secondary px-3 py-1 text-xs font-bold text-secondary-foreground">
                AB-D10 deployed
              </span>
            </div>
            <h2 className="display-title mb-2 text-2xl font-bold text-foreground">
              Wall Chess
            </h2>
            <p className="mb-6 text-sm leading-6 text-muted-foreground">
              Race across a 9×9 board while placing walls that slow the opponent without trapping them.
            </p>
            <div className="mt-auto flex flex-col gap-2 sm:flex-row">
              <Button asChild size="lg" className="rounded-full">
                <Link to="/play">
                  <Play size={18} fill="currentColor" />
                  Choose Wall Chess
                </Link>
              </Button>
              <Button
                variant="outline"
                size="lg"
                type="button"
                onClick={() => setShowRules(true)}
                className="rounded-full"
              >
                <BookOpen size={18} />
                Read rules
              </Button>
            </div>
          </article>

          <article className="flex flex-col rounded-xl border bg-card p-6 text-card-foreground shadow-sm">
            <div className="mb-5 flex items-start justify-between gap-4">
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Crown size={24} />
              </span>
              <span className="rounded-full border bg-secondary px-3 py-1 text-xs font-bold text-secondary-foreground">
                Gen-1 search bot
              </span>
            </div>
            <h2 className="display-title mb-2 text-2xl font-bold text-foreground">
              International Draughts
            </h2>
            <p className="mb-6 text-sm leading-6 text-muted-foreground">
              Play 10×10 draughts with mandatory maximal capture, flying kings, and a WASM-backed bot.
            </p>
            <div className="mt-auto">
              <Button asChild size="lg" className="rounded-full">
                <Link to="/checkers">
                  <Bot size={18} />
                  Play Draughts
                </Link>
              </Button>
            </div>
          </article>
        </div>
      </section>

      {showRules && <HowToPlay onClose={() => setShowRules(false)} />}
    </main>
  )
}
