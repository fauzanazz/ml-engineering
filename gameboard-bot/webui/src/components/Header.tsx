import { Link } from '@tanstack/react-router'

const navLinkBase =
  'relative inline-flex items-center transition-colors after:absolute after:inset-x-0 after:-bottom-1.5 after:h-0.5 after:origin-left after:scale-x-0 after:bg-primary after:transition-transform after:duration-200 hover:text-foreground hover:after:scale-x-100'
const navLinkInactive = { className: 'text-muted-foreground' }
const navLinkActive = { className: 'text-foreground after:scale-x-100' }

export default function Header() {
  return (
    <header
      className="sticky top-0 z-50 border-b border-border bg-background/85 px-4 backdrop-blur-lg"
      style={{ height: 'var(--navbar-h)' }}
    >
      <nav className="page-wrap flex h-full items-center gap-x-3">
        <p className="m-0 flex-shrink-0">
          <Link
            to="/"
            className="inline-flex items-center gap-2 rounded-full border border-border bg-secondary px-3 py-1.5 text-sm font-semibold text-foreground no-underline shadow-sm"
          >
            <span className="h-2 w-2 rounded-full bg-primary" />
            <span className="font-heading">Wall Chess</span>
          </Link>
        </p>

        <div className="flex items-center gap-x-4 text-sm font-semibold">
          <Link to="/" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            Home
          </Link>
          <Link to="/checkers" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            Checkers
          </Link>
          <Link to="/graph" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            Graph
          </Link>
          <Link to="/about" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            About
          </Link>
        </div>
      </nav>
    </header>
  )
}
