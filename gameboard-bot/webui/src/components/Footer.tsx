export default function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer className="mt-20 border-t bg-card px-4 pb-10 pt-6 text-muted-foreground">
      <div className="page-wrap flex items-center justify-center text-center">
        <p className="m-0 text-xs">
          &copy; {year} Gameboard
        </p>
      </div>
    </footer>
  )
}
