import { Link } from '@tanstack/react-router'

const navLinkBase =
  'relative inline-flex min-h-10 items-center px-3 transition-colors after:absolute after:inset-x-3 after:bottom-0 after:h-0.5 after:origin-left after:scale-x-0 after:bg-primary after:transition-transform after:duration-200 hover:text-foreground hover:after:scale-x-100'
const navLinkInactive = { className: 'text-muted-foreground' }
const navLinkActive = { className: 'text-foreground after:scale-x-100' }

export default function Header() {
  return (
    <header
      className="sticky top-0 z-50 border-b border-border bg-background/85 px-4 backdrop-blur-lg"
      style={{ height: 'var(--navbar-h)' }}
    >
      <nav className="page-wrap flex h-full items-center justify-center">
        <div className="flex items-center gap-x-2 text-sm font-semibold">
          <Link to="/" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            Home
          </Link>
          <Link to="/play" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            Wall Chess
          </Link>
          <Link to="/checkers" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            Draughts
          </Link>
          <Link to="/about" className={navLinkBase} inactiveProps={navLinkInactive} activeProps={navLinkActive}>
            About
          </Link>
        </div>
      </nav>
    </header>
  )
}
