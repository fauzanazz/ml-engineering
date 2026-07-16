import { Link, createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <main className="page-wrap px-4 py-12">
      <section className="py-6 sm:py-8">
        <p className="island-kicker mb-2">About</p>
        <h1 className="display-title mb-3 text-4xl font-bold text-foreground sm:text-5xl">
          Two board games. One Rust engine.
        </h1>
        <div className="max-w-2xl space-y-4 text-base leading-8 text-muted-foreground">
          <p className="text-foreground">
            Gameboard is a board-first playground for strategy bots. Wall Chess
            and International Draughts share the same Rust search core, then run
            in the browser through WebAssembly.
          </p>
          <p>
            Wall Chess is the mature game: a 9×9 pawn race where walls shape the
            path. The strongest current bot is alpha-beta search with a tuned
            evaluator and browser-safe node budget.
          </p>
          <p>
            International Draughts is the second game on the platform. The
            board uses the Rust/WASM engine for mandatory maximal capture,
            flying kings, promotion rules, terminal status, and bot analysis.
          </p>
          <p>
            The ML trainer remains part of the project, but it is not the whole
            product. Python/PyTorch handles policy-value experiments; Rust owns
            the playable engine and search behavior.
          </p>
          <div className="flex flex-col gap-3 pt-3 sm:flex-row">
            <Link
              to="/play"
              className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2.5 text-sm font-bold text-primary-foreground no-underline"
            >
              Play Wall Chess
            </Link>
            <Link
              to="/checkers"
              className="inline-flex items-center justify-center rounded-full border bg-card px-5 py-2.5 text-sm font-bold text-foreground no-underline"
            >
              Play Draughts
            </Link>
          </div>
        </div>
      </section>
    </main>
  )
}
