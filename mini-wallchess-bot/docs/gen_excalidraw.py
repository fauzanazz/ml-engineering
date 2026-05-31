#!/usr/bin/env python3
"""Generate an Excalidraw scene of the wallchess bot AI architecture.

Output: docs/ai-architecture.excalidraw  (open at https://excalidraw.com)
Deterministic IDs/seeds so re-runs produce a stable file (no Math.random).
"""
import json
import itertools

_ctr = itertools.count(1)


def _id(prefix):
    return f"{prefix}-{next(_ctr)}"


elements = []


def box(x, y, w, h, label, *, fill="#e7f5ff", stroke="#1971c2", dashed=False,
        font=18, group=None):
    """A rounded rectangle with centered multi-line text bound to it."""
    rid = _id("rect")
    tid = _id("text")
    seed = (x * 31 + y * 17 + 7) % 2_000_000
    lines = label.split("\n")
    rect = {
        "id": rid, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke, "backgroundColor": fill,
        "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1, "opacity": 100, "groupIds": [group] if group else [],
        "frameId": None, "roundness": {"type": 3}, "seed": seed,
        "version": 1, "versionNonce": seed, "isDeleted": False,
        "boundElements": [{"type": "text", "id": tid}],
        "updated": 1, "link": None, "locked": False,
    }
    fontsize = font
    text_h = len(lines) * (fontsize + 6)
    text = {
        "id": tid, "type": "text", "x": x + 8, "y": y + (h - text_h) / 2,
        "width": w - 16, "height": text_h, "angle": 0,
        "strokeColor": stroke, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [group] if group else [],
        "frameId": None, "roundness": None, "seed": seed + 1,
        "version": 1, "versionNonce": seed + 1, "isDeleted": False,
        "boundElements": [], "updated": 1, "link": None, "locked": False,
        "fontSize": fontsize, "fontFamily": 1, "text": label,
        "textAlign": "center", "verticalAlign": "middle",
        "containerId": rid, "originalText": label, "lineHeight": 1.25,
    }
    elements.extend([rect, text])
    return rect


def label(x, y, text, *, color="#343a40", font=22, w=320):
    tid = _id("text")
    seed = (x * 13 + y * 29 + 3) % 2_000_000
    elements.append({
        "id": tid, "type": "text", "x": x, "y": y, "width": w,
        "height": font + 8, "angle": 0, "strokeColor": color,
        "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
        "seed": seed, "version": 1, "versionNonce": seed, "isDeleted": False,
        "boundElements": [], "updated": 1, "link": None, "locked": False,
        "fontSize": font, "fontFamily": 1, "text": text, "textAlign": "left",
        "verticalAlign": "top", "containerId": None, "originalText": text,
        "lineHeight": 1.25,
    })


def arrow(x1, y1, x2, y2, *, color="#495057", dashed=False, lbl=None):
    aid = _id("arrow")
    seed = (int(x1) * 7 + int(y2) * 11 + 5) % 2_000_000
    el = {
        "id": aid, "type": "arrow", "x": x1, "y": y1,
        "width": abs(x2 - x1), "height": abs(y2 - y1), "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 2}, "seed": seed, "version": 1,
        "versionNonce": seed, "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
        "points": [[0, 0], [x2 - x1, y2 - y1]], "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
    }
    elements.append(el)
    if lbl:
        label((x1 + x2) / 2 - 30, (y1 + y2) / 2 - 28, lbl, color=color, font=14, w=140)


def container(x, y, w, h, title, *, stroke="#adb5bd", fill="transparent"):
    seed = (x * 5 + y * 3 + 1) % 2_000_000
    elements.append({
        "id": _id("frame"), "type": "rectangle", "x": x, "y": y,
        "width": w, "height": h, "angle": 0, "strokeColor": stroke,
        "backgroundColor": fill, "fillStyle": "hachure", "strokeWidth": 1.5,
        "strokeStyle": "dotted", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": {"type": 3},
        "seed": seed, "version": 1, "versionNonce": seed, "isDeleted": False,
        "boundElements": [], "updated": 1, "link": None, "locked": False,
    })
    label(x + 14, y + 10, title, color=stroke, font=16)


# ─────────────────────────────────────────────────────────────────────────
# Title
label(40, 20, "Wall Chess Bot — AI Architecture", font=30, w=700)
label(40, 58, "Rust engine core  →  WASM bridge  →  React web UI   (ML model = future drop-in)",
      color="#868e96", font=16, w=900)

# ── LAYER 1: Rust core (core/) ──────────────────────────────────────────
container(20, 100, 430, 560, "core/  (Rust engine — pure, no IO)", stroke="#1971c2")

box(50, 150, 360, 70,
    "state.rs — bit-packed State\npawns u8x2 · h/v walls u64 · walls_left · turn",
    fill="#e7f5ff", stroke="#1971c2", font=14)

box(50, 245, 360, 86,
    "moves.rs\npawn_moves (step / jump / diagonal)\ndistance_to_goal (BFS) · legal_moves\ncan_place_wall (no-trap check)",
    fill="#e7f5ff", stroke="#1971c2", font=13)

box(50, 355, 360, 96,
    "eval.rs — Evaluator trait  ◀ ML SEAM\nHeuristic leaf score:\nw_path·(dist_opp-dist_me) + w_progress·(-dist_me)\n+ w_tempo + w_wall   →  win_prob 0..100",
    fill="#fff3bf", stroke="#f08c00", font=13)

box(50, 475, 360, 96,
    "search.rs\nnegamax + alpha-beta pruning\niterative deepening (depth 1..N)\nTransposition Table (HashMap<State>)",
    fill="#e7f5ff", stroke="#1971c2", font=14)

box(50, 595, 360, 50,
    "lib.rs  analyze(state, depth, k)\n-> (best_move, south 0..100, north 0..100)",
    fill="#d3f9d8", stroke="#2f9e44", font=14)

# internal arrows
arrow(230, 331, 230, 355)               # moves -> eval
arrow(120, 451, 120, 475)               # eval -> search (eval feeds leaves)
arrow(340, 475, 340, 451, lbl="leaf eval")  # search -> eval (calls)
arrow(230, 571, 230, 595)               # search -> analyze
arrow(150, 220, 150, 245)               # state -> moves
# TT self-loop hint
arrow(410, 500, 440, 500, color="#1971c2")
arrow(440, 540, 410, 540, color="#1971c2", lbl="memoize")

# ── LAYER 2: tuning harness ─────────────────────────────────────────────
container(20, 685, 430, 150, "core/  bins — tuning & eval (offline)", stroke="#9c36b5")
box(50, 725, 120, 90, "arena.rs\nbot vs bot\nweight / depth\nmatch -> W/L",
    fill="#f3d9fa", stroke="#9c36b5", font=13)
box(190, 725, 110, 90, "selfplay\nfull game\nlog 0..100\neach ply",
    fill="#f3d9fa", stroke="#9c36b5", font=13)
box(320, 725, 110, 90, "diag\nnodes/depth\nsearch\nhealth",
    fill="#f3d9fa", stroke="#9c36b5", font=13)
arrow(235, 595, 235, 595)  # noop guard
arrow(150, 645, 150, 725, color="#9c36b5", lbl="drive")

# ── LAYER 3: WASM bridge (wasm/) ────────────────────────────────────────
container(490, 230, 320, 230, "wasm/  (wasm-bindgen bridge)", stroke="#e8590c")
box(515, 275, 270, 80,
    "lib.rs  serde DTOs\nGameState/Move JSON  <->  Rust State\n(matches TS shapes exactly)",
    fill="#ffe8cc", stroke="#e8590c", font=13)
box(515, 370, 270, 70,
    "exports:\nchoose_move(state, depth)\nanalyze_state(state, depth, k)",
    fill="#ffe8cc", stroke="#e8590c", font=14)
arrow(410, 620, 515, 410, lbl="rlib dep")          # core -> wasm
arrow(650, 355, 650, 370)

# ── LAYER 4: web UI (webui/) ────────────────────────────────────────────
container(850, 150, 360, 470, "webui/  (React + TanStack, Vite)", stroke="#2f9e44")
box(880, 200, 300, 80,
    "src/wasm/  (generated)\nwallchess_wasm.js + _bg.wasm (84 KB)\nbuilt by:  bun run wasm",
    fill="#d3f9d8", stroke="#2f9e44", font=13)
box(880, 300, 300, 86,
    "game/api.ts\nlazy-load WASM (one-shot init)\nbotMove({data}) · analyzePosition()\nTS engine fallback if WASM fails",
    fill="#d3f9d8", stroke="#2f9e44", font=13)
box(880, 405, 300, 70,
    "routes/game.tsx\nbot useEffect -> botMove\nanalyze useEffect -> WinMeter",
    fill="#d3f9d8", stroke="#2f9e44", font=14)
box(880, 495, 300, 60,
    "components/WinMeter.tsx\n0..100 split bar  (south + north = 100)",
    fill="#d3f9d8", stroke="#2f9e44", font=14)
box(880, 570, 300, 36, "components/Board.tsx — interactive board",
    fill="#ebfbee", stroke="#2f9e44", font=13)

arrow(785, 400, 880, 360, lbl="import")   # wasm -> api
arrow(1030, 386, 1030, 405)               # api -> game
arrow(1030, 475, 1030, 495)               # game -> winmeter
arrow(1030, 475, 950, 570)                # game -> board

# ── LAYER 5: future ML ──────────────────────────────────────────────────
container(490, 500, 320, 200, "FUTURE — ML model (~1M params)", stroke="#c2255c")
box(515, 545, 270, 60,
    "trainer/ (Python, uv)\nself-play data -> value net",
    fill="#ffdeeb", stroke="#c2255c", font=13, dashed=True)
box(515, 615, 270, 60,
    "NetEvaluator: impl Evaluator\nload weights -> replace Heuristic",
    fill="#ffdeeb", stroke="#c2255c", font=13, dashed=True)
arrow(650, 605, 650, 615, dashed=True, color="#c2255c")
arrow(515, 575, 230, 451, dashed=True, color="#c2255c", lbl="swaps leaf eval")
arrow(300, 760, 515, 575, dashed=True, color="#c2255c", lbl="self-play data")

scene = {
    "type": "excalidraw", "version": 2,
    "source": "wallchess-bot/docs/gen_excalidraw.py",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

import os
out = os.path.join(os.path.dirname(__file__), "ai-architecture.excalidraw")
with open(out, "w") as f:
    json.dump(scene, f, indent=2)
print(f"wrote {out}  ({len(elements)} elements)")
