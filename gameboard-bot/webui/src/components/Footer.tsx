export default function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer className="site-footer mt-20 px-4 pb-10 pt-6 text-[var(--sea-ink-soft)]">
      <div className="page-wrap flex items-center justify-center text-center">
        <p className="m-0 text-xs">
          &copy; {year} Wall Chess
        </p>
      </div>
    </footer>
  )
}
