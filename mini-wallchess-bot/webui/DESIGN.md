---
name: Wall Chess
description: A focused, board-first strategy game where pawn and wall placement decide the match.
colors:
  ink: "#1a1510"
  ink-soft: "#5a524a"
  brown-accent: "#7a4f2c"
  brown-mid: "#c4956a"
  brown-deep: "#5c3820"
  white-base: "#ffffff"
  white-warm: "#fafaf8"
  white-sand: "#f5f0eb"
  danger: "#8b2424"
  board-light: "#f5f0eb"
  board-dark: "#d4c9be"
  board-frame: "#1a1510"
typography:
  display:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "clamp(2.5rem, 6vw, 3.75rem)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "clamp(1.75rem, 4vw, 2.5rem)"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.3
  body:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.69rem"
    fontWeight: 700
    letterSpacing: "0.16em"
rounded:
  sm: "7px"
  md: "16px"
  lg: "20px"
  xl: "28px"
  pill: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "28px"
  xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.white-base}"
    rounded: "{rounded.pill}"
    padding: "14px 32px"
  button-primary-hover:
    backgroundColor: "#2e2822"
  button-ghost:
    backgroundColor: "rgba(255,255,255,0.90)"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "6px 12px"
  button-ghost-hover:
    backgroundColor: "{colors.white-sand}"
  island-shell:
    backgroundColor: "rgba(255,255,255,0.96)"
    rounded: "{rounded.xl}"
    padding: "{spacing.lg}"
  feature-card:
    backgroundColor: "rgba(255,255,255,0.96)"
    rounded: "{rounded.lg}"
    padding: "28px"
  feature-card-hover:
    backgroundColor: "{colors.white-base}"
  player-card:
    backgroundColor: "rgba(255,255,255,0.96)"
    rounded: "{rounded.md}"
    padding: "{spacing.md}"
---

# Design System: Wall Chess

## 1. Overview

**Creative North Star: "The Quiet Board Room"**

Wall Chess is a white room with a game board in it. Surfaces are clean white. Text is near-black. Brown appears once or twice per screen — in a button, a link, a kicker — and nowhere else. The board is the only thing with texture or tonal range. Everything else defers.

The palette is a ratio, not a theme: 60% neutral, 30% ink, 10% brown. Neutral (white) is the primary — it fills all backgrounds and surfaces. Ink fills 30% as structure: text, borders, dividers, primary buttons. Brown fills 10% as accent: links, kickers, pawn gradient, focus rings. The system should feel like a well-typeset book that happens to have a game board in it.

The system explicitly rejects: teal or green color branding (replaced by this palette), warm-neutral maximalism (brown is 10%, not 40%), dark terminal chrome, and gamification overlays.

**Key Characteristics:**
- White-dominant surfaces: no colored backgrounds, no tinted neutrals, no cream
- Near-black structure: text, borders, primary buttons all pull from the same near-black ink
- Brown as accent only: one brown element per visual cluster — link, kicker, or button accent; never fill backgrounds
- Board-first: the board is the only element with real tonal range; everything else is quiet
- Dark mode: deep warm-black tones (not blue-black or gray), same ratio inverted

## 2. Colors: 60 / 30 / 10

The palette is a ratio. Neutral (white) is primary — 60% of any screen. Ink is secondary — 30%, all structure. Brown is accent — 10% maximum.

### Primary — Neutral (60%)
White fills all backgrounds and surfaces. This is the dominant presence; never compete with it.
- **White Base** (`#ffffff`): Page background, island-shell cards, primary surfaces.
- **White Warm** (`#fafaf8`): Near-white alternative for layering. Barely off-white.
- **White Sand** (`#f5f0eb`): Board light squares, `code` element background, subtle surface tint. The warmest neutral.

### Secondary — Ink (30%)
Ink fills all structure: text, borders, dividers, primary button fill, board frame. 30% means ink can appear in more structural roles than before — thicker borders, dividers, more prominent containment lines — but never as a background fill.
- **Ink** (`#1a1510`): All body text, headings, primary button fill, board frame, structural borders.
- **Ink Soft** (`#5a524a`): Secondary text, meta labels, muted captions. Same warm cast as Ink, lighter.

### Accent — Brown (10%)
Brown earns its place by being rare. One brown element per visual cluster.
- **Brown Accent** (`#7a4f2c`): Links, `accent-text`, kicker labels, active underlines. The defining brown. Never fills a background larger than an icon badge.
- **Brown Mid** (`#c4956a`): Pawn gradient top, decorative radial blob tints, hover glow effects.
- **Brown Deep** (`#5c3820`): Link hover, checkmark color in rules list.

### Danger
- **Danger** (`#8b2424`): Error states, illegal move indicators, ✗ symbols in rules list.

### Board Colors
- **Board Light** (`#f5f0eb`): Light checkerboard squares. Same as White Sand — the board reads as warm ivory at rest.
- **Board Dark** (`#d4c9be`): Dark checkerboard squares. Muted warm beige. Low contrast by design — the board is a background for the pawns.
- **Board Frame** (`#1a1510`): 4px border around the board grid. Same as Ink — the frame is structure, not decoration.

### Dark Mode
Inverted ratio: deep warm-black background (`#0f0d0b`), off-white text (`#f0ede8`), lighter brown accent (`#d4a876`). Surfaces become `rgba(22,18,14,0.95)`. Board squares: `#221e18` (light), `#181410` (dark). Frame: `#f0ede8`.

**The 60-30-10 Ratio.** Count visual weight per screen: 60% neutral white surfaces, 30% ink structure (text, borders, dividers, primary buttons), 10% brown accent. If brown exceeds 10%, reduce. If ink drops below 20%, structural elements are too light.

**The 10% Brown Rule.** Count brown elements per screen. If more than one distinct brown area appears per visual cluster (card, section, panel), reduce. Brown earns its place by being rare. A page with brown everywhere is not this system.

**The No-Teal Rule.** The palette was previously teal/coastal. Any reintroduction of teal, green, or seafoam tones — even subtle — breaks the identity. If a color reads as teal or green, replace it with white, ink-soft, or brown-mid.

## 3. Typography

**Display Font:** Fraunces (variable: `opsz` 9–144, `wgt` 500–700, with fallback Georgia, serif)
**Body Font:** Manrope (weights 400–800, with fallback ui-sans-serif, system-ui, sans-serif)

**Character:** Fraunces at display scale reads like a board-game almanac — authoritative, warm, unhurried. Manrope at body scale is clean and functional. Together they reinforce the "quiet book with a game board" tone without trying.

### Hierarchy
- **Display** (700, `clamp(2.5rem, 6vw, 3.75rem)`, line-height 1.02, tracking -0.02em): Hero headlines only. Fraunces. Max 6rem.
- **Headline** (700, `clamp(1.75rem, 4vw, 2.5rem)`, line-height 1.1, tracking -0.01em): Secondary headings, game mode selection. Fraunces.
- **Title** (700, 1.25rem, line-height 1.3): Card headings, player panel labels. Manrope.
- **Body** (400–500, 1rem, line-height 1.6): Descriptive copy, rules text. Manrope. Max line length 65–75ch.
- **Label** (700, 0.69rem, tracking 0.16em, uppercase): Kicker/eyebrow text, turn indicators. Manrope. One per section maximum.

**The Single Eyebrow Rule.** One kicker label per screen section. Not every section gets one.

## 4. Elevation

The system is flat-first. Surfaces are white with a thin border (`1px solid rgba(26,21,16,0.10)`). Shadows are shallow and warm — they signal containment, not depth. No glassmorphism, no backdrop blur, no inset glints.

Two effective layers: page (white), and contained surface (card/shell with thin border + minimal shadow). Nothing floats higher than the board shell.

### Shadow Vocabulary
- **Island Shadow** (`0 2px 12px rgba(26,21,16,0.06), 0 1px 3px rgba(26,21,16,0.04)`): Primary island-shell. Shallow warm-dark ambient. Not teal-tinted.
- **Feature Card Shadow** (`0 1px 4px rgba(26,21,16,0.05)`): Mode selection and secondary cards. Barely perceptible at rest.
- **Button Shadow** (`0 8px 24px rgba(26,21,16,0.18)`): Primary CTA button. Warm-dark shadow below the near-black pill.

**The Flat-By-Default Rule.** No backdrop-filter blur on any surface. No inset glints. The system is flat; warmth comes from color, not from glass effects.

**The Hover-Lift Rule.** Hover state on interactive cards uses `transform: translateY(-1px)` plus a slightly darker border. Shadow value does not change. Lift communicates interaction; shadow communicates containment.

## 5. Components

### Buttons
Near-black pill as primary. Ghost button for secondary actions.

- **Shape:** Fully pill (9999px radius)
- **Primary:** Ink (`#1a1510`) fill, white text, padding 14px 32px, font-weight 700, shadow `0 8px 24px rgba(26,21,16,0.18)`. Hover: opacity 0.88 + `translateY(-0.5px)`.
- **Ghost / Secondary:** `rgba(255,255,255,0.90)` fill, Ink text, border `1px solid rgba(26,21,16,0.12)`. Hover: fill `#f5f0eb`, border darkens to `rgba(26,21,16,0.22)`.
- **Brown Accent Button** (rare): `rgba(196,149,106,0.10)` fill, Brown Accent text, border `1px solid rgba(122,79,44,0.25)`. Used for dismissal/secondary modal actions only.
- **Focus:** `outline: 2px solid #7a4f2c` offset 2px on all buttons.
- **Transition:** 180ms ease on background-color, color, border-color, opacity, transform.

### Cards / Island Shells
Clean white, thin border. No glass.

- **Shape:** 28px radius (island-shell); 20px (feature cards); 16px (player panels).
- **Background:** `rgba(255,255,255,0.96)` — effectively white.
- **Border:** `1px solid rgba(26,21,16,0.10)`.
- **Shadow:** Island Shadow (see Elevation).
- **Hover (cards only):** `translateY(-1px)`, border darkens to `rgba(26,21,16,0.22)`.

### Game Board
The signature component. The only element with real tonal range.

- **Shell:** Island-shell with 28px radius, 12–16px padding.
- **Grid:** 9×9 CSS grid, `aspect-square`, 4px border in Board Frame color (`#1a1510`), overflow hidden, inner radius 12px.
- **Squares:** Alternate Board Light/Dark. Goal rows (row 1, row 9) have `rgba(196,149,106,0.14)` warm-brown overlay.
- **Pawns:** 68% of cell size, radial-gradient circle. South: brown warm (`#c4956a → #7a4f2c`). North: amber-gold (`#f0c27a → #c47d2f`). Active pawn: `ring-2 ring-white/70 ring-offset-1`.
- **Selection/Hints:** `ring-2 ring-inset` in Brown Accent (`#7a4f2c`).

### Navigation
White bar. Quiet.

- **Background:** `rgba(255,255,255,0.92)` with `backdrop-filter: blur(10px)`. Sticky with border-bottom `1px solid rgba(26,21,16,0.10)`.
- **Nav Links:** Manrope 600, Ink Soft at rest, Ink on hover/active. Underline `::after` pseudo-element with brown gradient (`#7a4f2c → #c4956a`), scale-x on hover (170ms ease).
- **Brand chip:** White pill with `1px solid rgba(26,21,16,0.12)` border, Ink text. Brown dot (gradient).

### Chips / Status Labels
- **Background:** `rgba(255,255,255,0.90)` with `1px solid rgba(26,21,16,0.12)` border.
- **Typography:** Label scale (0.69rem, 700, 0.16em tracking, uppercase). Brown kicker color.
- **Use:** Turn indicators, status chips. One per context.

## 6. Do's and Don'ts

### Do:
- **Do** keep surfaces white. White backgrounds, white cards, white header. The board is the only thing with tonal variation.
- **Do** use Ink (`#1a1510`) for primary CTA buttons. Black buttons on white: maximum contrast, no color needed.
- **Do** restrict brown to accent roles: links, kickers, pawn gradient, goal-row tint, focus rings. One brown element per visual cluster.
- **Do** use `border: 1px solid rgba(26,21,16,0.10)` on all cards and containers. A thin ink border is sufficient containment on white.
- **Do** keep the board as the most visually prominent element. Every chrome decision defers to the board.
- **Do** use Fraunces exclusively for display and headline scales. Manrope for everything below title level.
- **Do** treat dark mode as first-class: deep warm-black background, same ratio inverted. Test every new component in both modes.
- **Do** include `@media (prefers-reduced-motion: reduce)` for every animation.

### Don't:
- **Don't** reintroduce teal, green, or seafoam tones anywhere. The palette was previously coastal; that identity is replaced. Any color that reads as teal or green is a regression.
- **Don't** use brown as a fill color on backgrounds or cards. Brown is for text, borders, and accents — not surfaces.
- **Don't** use warm neutrals (cream, sand, parchment) as page or card backgrounds. White Base (`#ffffff`) is the background. White Sand (`#f5f0eb`) is for code, board squares, and subtle tints only.
- **Don't** add glassmorphism: no `backdrop-filter: blur` on cards, no inset glints, no translucent gradient surfaces. The system is flat.
- **Don't** add gamification chrome: XP bars, streaks, badges. The board is the reward.
- **Don't** use `border-left` greater than 1px as a colored accent stripe. Use a full border or nothing.
- **Don't** use `background-clip: text` with a gradient.
- **Don't** use border-radius greater than 28px on rectangular cards.
- **Don't** add marketing SaaS patterns: numbered section eyebrows, identical icon-card grids, hero metric templates.
- **Don't** put explanatory copy on the board during play. State communicates through visual signals, not text overlays.
