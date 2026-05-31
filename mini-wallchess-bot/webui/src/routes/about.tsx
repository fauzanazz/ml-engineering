import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <main className="page-wrap px-4 py-12">
      <section className="py-6 sm:py-8">
        <p className="island-kicker mb-2">About</p>
        <h1 className="display-title mb-3 text-4xl font-bold text-[var(--sea-ink)] sm:text-5xl">
          A board game with a machine learning brain.
        </h1>
        <div className="max-w-2xl space-y-4 text-base leading-8 text-[var(--sea-ink-soft)]">
          <p className="text-[var(--sea-ink)]">
            Wall Chess is a two-player abstract strategy game on a 9×9 grid.
            Move your pawn toward the opposite side, or spend a wall to slow
            your opponent down. First to cross wins.
          </p>
          <p>
            The bot is the real artifact. It runs a minimax search with
            alpha-beta pruning, compiled to WebAssembly from Rust. The win
            probability meter above the board shows its live evaluation of the
            position after every move.
          </p>
          <p>
            The Graph Traveler lets you walk the game's state space one node at
            a time, watching the engine score each position as you explore the
            tree.
          </p>
        </div>
      </section>
    </main>
  )
}
