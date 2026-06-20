import { Link } from '@tanstack/react-router'

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--header-bg)] px-4 backdrop-blur-lg" style={{ height: 'var(--navbar-h)' }}>
      <nav className="page-wrap flex h-full items-center gap-x-3">
        <p className="m-0 flex-shrink-0">
          <Link
            to="/"
            className="inline-flex items-center gap-2 rounded-full border border-[var(--chip-line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm font-semibold text-[var(--sea-ink)] no-underline shadow-[var(--shadow-chip)]"
          >
            <span className="h-2 w-2 rounded-full bg-[var(--lagoon)]" />
            Wall Chess
          </Link>
        </p>

        <div className="flex items-center gap-x-4 text-sm font-semibold">
          <Link
            to="/"
            className="nav-link"
            activeProps={{ className: 'nav-link is-active' }}
          >
            Home
          </Link>
          <Link
            to="/checkers"
            className="nav-link"
            activeProps={{ className: 'nav-link is-active' }}
          >
            Checkers
          </Link>
          <Link
            to="/graph"
            className="nav-link"
            activeProps={{ className: 'nav-link is-active' }}
          >
            Graph
          </Link>
          <Link
            to="/about"
            className="nav-link"
            activeProps={{ className: 'nav-link is-active' }}
          >
            About
          </Link>
        </div>
      </nav>
    </header>
  )
}
