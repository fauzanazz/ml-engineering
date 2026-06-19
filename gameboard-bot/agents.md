# gameboard-bot

> Status: Multi-game platform — Wall Chess (deployed bot) + International Draughts (engine landed).
> Goal: One shared Rust engine (search + arena + encoders) that hosts many board games behind a single `Game` trait.

---

## Platform overview

`gameboard-bot` is a board-game bot platform. A single Rust core crate (`gameboard-core`) provides the engine — alpha-beta search with the full pruning/quiescence stack, a self-play arena, and feature encoders — and every game plugs in by implementing the `Game` trait (plus `Evaluator`/`Encoder`). The search monomorphizes per game, so adding a game does not touch the deployed bot's byte-for-byte behavior.

### Games

| Game | Board | trait `ID` | Status |
| --- | --- | --- | --- |
| Wall Chess (Quoridor variant) | 9×9, 81 cells | `wallchess` | Deployed bot (Gen-2: aggressive pruning + reweighted eval) |
| International Draughts | 10×10, 50 dark squares | `checkers` | Engine landed (perft d1–8 exact); CLI bin `checkers` |

### The `Game` seam

Each game is a flat module at the core crate root that implements `Game` (associated `State`/`Move`, the engine consts, and the move/terminal/hash methods), `Evaluator` (negamax-antisymmetric, terminal ±`WIN_SCORE`), and `Encoder` (feature vector + move-index mapping for NN work). `Player { P0, P1 }` is the generic two-valued turn marker shared by every game.

| Game | `Player` mapping | `ACTION_COUNT` | `FEATURE_LEN` | `MOVE_INDEX_SPACE` |
| --- | --- | --- | --- | --- |
| Wall Chess | South = P0, North = P1 | 209 | 300 | 384 |
| International Draughts | White = P0, Black = P1 | 2500 | 308 | 2500 |

### Docs

| Doc | Covers |
| --- | --- |
| `docs/multi-game-architecture-plan` | How the engine was genericized over `Game`; phases and golden-snapshot guards |
| `docs/game-trait-design` | The `Game`/`Evaluator`/`Encoder` trait contracts and the generic `Search<'a, E>` design |
| `docs/checkers-rules-and-encoding` | International Draughts rules, board indexing, and the encoder layout |

---

## Wall Chess

Quoridor-style pawn race on a 9×9 board with placeable walls.

### Quick Rules

**01 — Move your pawn**
On your turn, move one square orthogonally. Jump over your opponent if adjacent with no wall behind them.

**02 — Place a wall**
Instead of moving, place a 2-cell wall anywhere on the board. Walls block paths — use them wisely.

**03 — Block, but never trap**
Walls cannot completely block any player from reaching their goal row. Every placement is validated.

**04 — First to cross wins**
SOUTH starts on row 1, NORTH on row 9. The first player to reach the opposite side wins.

### Objective

Each player controls one pawn. The goal is to be the first to move your pawn to any cell on the opposite side of the board. SOUTH starts on row 1 and must reach row 9. NORTH starts on row 9 and must reach row 1.

### Setup

- 9×9 board with 81 cells
- Each player starts in the middle of their back row
- Each player receives 10 walls
- SOUTH moves first

### On Your Turn

On each turn you must do exactly one of two actions:

#### ♟ Move Your Pawn
Move to an orthogonally adjacent cell (up, down, left, right) that is not blocked by a wall. You may jump over your opponent if they are adjacent and there is no wall behind them. If there is a wall behind them, you may jump diagonally.

#### ▬ Place a Wall
Place one of your remaining walls on the board. Walls span 2 cells and can be horizontal or vertical. A wall cannot be placed if it completely blocks either player from reaching their goal row.

### Wall Rules

- ✓ Walls occupy the gaps between cells, spanning exactly 2 cells
- ✓ Walls can be horizontal (blocking north-south movement) or vertical (blocking east-west movement)
- ✓ Two walls cannot overlap or cross each other
- ✗ A wall placement that leaves any player with no possible path to their goal row is illegal

---

## International Draughts

10×10 checkers on the 50 dark squares (PDN-numbered 1–50). White (`P0`) moves first up the board; Black (`P1`) moves down.

### Quick Rules

**01 — Move your men diagonally forward**
A man steps one square diagonally forward to an empty dark square. White moves up the board, Black moves down.

**02 — Capture is mandatory and maximal**
If any capture exists you must capture, and you must take the line that removes the **maximum number** of pieces. Men capture by jumping an adjacent enemy into the empty square beyond — **in all four diagonal directions** (forward or backward). Captured pieces are removed only after the full capture sequence completes.

**03 — Kings fly**
A man that **stops** on the far back rank promotes to a king. A king slides any distance along a diagonal and captures any distance, landing on any empty square beyond a single jumped enemy.

**04 — Promote only on stopping**
A man promotes only if it *ends its move* on the back rank. Jumping *through* the back rank mid-capture does not promote (the "passing"/Turkish rule).

**05 — No move loses; idle kings draw**
A side with no legal move loses. 25 consecutive king-only, non-capturing moves per side ends the game in a draw.

### Setup

- 10×10 board; play on the 50 dark squares only
- 20 men per side, on the first four rows of each side
- White moves first (up the board, promotes on row 0); Black moves second (down the board, promotes on row 9)

### Objective

Capture or block all enemy pieces. You win when your opponent has no legal move on their turn — either every piece is captured or every piece is blocked.

### Capture Rules

- ✓ Captures are mandatory; when several capture lines exist you must play one that captures the most pieces
- ✓ Men capture forward and backward; kings capture along the full diagonal (flying capture)
- ✓ The same piece may capture multiple times in one turn, changing direction between jumps
- ✗ You may not jump your own pieces, and a piece already captured this turn cannot be jumped again

### Draw Rule

- The game is a draw after 25 king-only, non-capturing moves by each side (any man move or any capture resets the counter)
