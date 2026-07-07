import { useState } from 'react'
import { Link, createFileRoute } from '@tanstack/react-router'
import { BookOpen, Play } from 'lucide-react'
import HowToPlay from '../components/HowToPlay'
import { Button } from '@/components/ui/button'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const [showRules, setShowRules] = useState(false)

  return (
    <main className="page-wrap flex min-h-[70vh] flex-col items-center justify-center px-4 py-14">
      <section className="rise-in relative w-full max-w-xl px-6 py-12 text-center sm:px-12 sm:py-16">
        <p className="island-kicker mb-3">Wall Chess</p>
        <h1 className="display-title mb-4 text-5xl font-bold leading-[1.02] tracking-tight text-foreground sm:text-6xl">
          Race to the other side.
        </h1>
        <p className="mx-auto mb-9 max-w-md text-base text-muted-foreground sm:text-lg">
          Move your pawn or drop a wall. First to cross the board wins. Play a
          bot or a friend on the same screen.
        </p>

        <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Button asChild size="lg" className="rounded-full w-full hover:-translate-y-0.5 sm:w-auto">
            <Link to="/play">
              <Play size={18} fill="currentColor" />
              Start
            </Link>
          </Button>
          <Button
            variant="outline"
            size="lg"
            type="button"
            onClick={() => setShowRules(true)}
            className="rounded-full w-full hover:-translate-y-0.5 sm:w-auto"
          >
            <BookOpen size={18} />
            How to Play
          </Button>
        </div>
      </section>

      {showRules && <HowToPlay onClose={() => setShowRules(false)} />}
    </main>
  )
}
