# Product

## Register

product

## Users

Two audiences sharing the same surface:

- **Casual players**: friends playing pass-and-play on one screen, or solo vs the bot. They want to pick up and play without reading docs — the board is the interface.
- **Developers and bot-curious**: people exploring the ML bot, reading source, testing the AI's strategy. They care about clarity and signal density, not ornamentation.

Both arrive via the landing page and end up at the board. The game loop is the product.

## Product Purpose

Wall Chess is a two-player abstract strategy game (pawn + wall placement on a 9×9 grid). The web app lets you play vs a bot or a friend on the same screen. The underlying ML bot is the engineering artifact; the UI is its showcase.

Success: a player lands, understands the rules, starts a game, and either finishes or comes back.

## Brand Personality

Minimal, focused, unhurried. White dominates (surfaces, backgrounds), black grounds (text, structure), brown touches sparingly (accent, warmth). No color competes with the board.

## References

- **Lichess.org**: focused board, no marketing clutter. The game is the entire page. Nothing competes with the board.

## Anti-references

- Teal/coastal color branding: the palette is now white/black/brown. Any drift back toward teal, green, or sea-tone accents breaks the identity.
- Dark terminal aesthetic: no hacker-green-on-black, no monospace-everything. Dark mode uses deep warm-black tones.
- Heavy gamification: no XP bars, streaks, or Duolingo-style reward overlays on top of the game.
- Warm-neutral maximalism: brown is 10% of the surface. It should feel rare, not everywhere.

## Design Principles

1. **Board first**: the game board is never subordinate to chrome. Header, panels, controls are quiet; the board is loud.
2. **Understand by playing**: rules are emergable from the board itself — highlighting, goal rows, pawn colors. Docs are a fallback, not onboarding.
3. **Restraint as identity**: white dominates, ink structures, brown accents once per cluster. The 60-30-10 ratio is the palette, not a guideline. Any drift toward color-heavy surfaces or warm-neutral maximalism breaks the identity.
4. **Signal over decoration**: every visual element earns its place by communicating game state or guiding a decision. Purely decorative layers are noise at the board.
5. **Same surface, two audiences**: developers reading code and casual players pressing buttons should both feel the interface was made for them. No mode-switching, no persona-targeted layers — clarity serves both.

## Accessibility & Inclusion

- Target: WCAG 2.1 AA at minimum.
- Dark mode: first-class, not an afterthought — deep ocean tones, not inverted light mode.
- Reduced motion: all animations must have a `prefers-reduced-motion` alternative (crossfade or instant).
- Board cells: keyboard-accessible, labelled with row/column for screen readers.
- Color alone must not be the only signal for game state (pawn color, turn indicator, hints).
