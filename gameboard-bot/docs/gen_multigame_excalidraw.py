#!/usr/bin/env python3
"""Generate an Excalidraw scene of the gameboard-bot multi-game architecture.

Data-driven: reads the layered diagram spec produced by the architecture-map
workflow (multigame-architecture.spec.json) and lays it out into bands.

  spec.json  ──(this script)──>  multigame-architecture.excalidraw
                                 (open at https://excalidraw.com)

Deterministic IDs/seeds + a fixed layout from each node's (layer, col), so
re-runs produce a stable file (no Math.random). Re-run after editing the spec:

    cd docs && python3 gen_multigame_excalidraw.py
"""
import json
import os
import textwrap
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "multigame-architecture.spec.json")))

# ── role → (fill, stroke); deferred boxes are drawn dashed ──────────────────
ROLE = {
    "generic":   ("#e7f5ff", "#1971c2"),  # blue   — game-agnostic engine
    "wallchess": ("#d3f9d8", "#2f9e44"),  # green  — Wall Chess specific
    "checkers":  ("#ffe8cc", "#e8590c"),  # orange — Checkers specific
    "shared":    ("#fff3bf", "#f08c00"),  # amber  — multi-game, not fully generic
    "external":  ("#f1f3f5", "#868e96"),  # gray   — third-party runtime
    "deferred":  ("#ffdeeb", "#c2255c"),  # pink   — planned, not yet wired
}

# ── edge category → (color, default dashed) ─────────────────────────────────
EDGE = {
    "seam":     ("#1971c2", False),  # impl Game/Evaluator/Encoder · Search via E::G
    "play":     ("#0c8599", False),  # PLAY flow: UI → wasm/bin → engine
    "train":    ("#9c36b5", False),  # TRAIN flow: data-gen → trainer → weights
    "deferred": ("#c2255c", True),   # not-yet-wired
    "default":  ("#adb5bd", False),
}

# ── layout constants ────────────────────────────────────────────────────────
BOX_W, BOX_H = 214, 74
GAP_X = 26
MARGIN_X = 40
TOP = 132
BAND_H, BAND_GAP = 132, 96
BOX_DY = 48                      # box offset below the band container's title
COL_PITCH = BOX_W + GAP_X

# ── primitive emitters (same scene schema as gen_excalidraw.py) ─────────────
_ctr = itertools.count(1)
elements = []


def _id(p):
    return f"{p}-{next(_ctr)}"


def box(x, y, w, h, text, *, fill, stroke, dashed=False, font=12):
    rid, tid = _id("rect"), _id("text")
    seed = (int(x) * 31 + int(y) * 17 + 7) % 2_000_000
    lines = text.split("\n")
    elements.append({
        "id": rid, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke, "backgroundColor": fill,
        "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 3}, "seed": seed, "version": 1,
        "versionNonce": seed, "isDeleted": False,
        "boundElements": [{"type": "text", "id": tid}], "updated": 1,
        "link": None, "locked": False,
    })
    text_h = len(lines) * (font + 6)
    elements.append({
        "id": tid, "type": "text", "x": x + 8, "y": y + (h - text_h) / 2,
        "width": w - 16, "height": text_h, "angle": 0, "strokeColor": stroke,
        "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
        "frameId": None, "roundness": None, "seed": seed + 1, "version": 1,
        "versionNonce": seed + 1, "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False, "fontSize": font,
        "fontFamily": 1, "text": text, "textAlign": "center",
        "verticalAlign": "middle", "containerId": rid, "originalText": text,
        "lineHeight": 1.25,
    })


def text_block(x, y, s, *, color="#343a40", font=20, w=360):
    lines = s.split("\n")
    tid = _id("text")
    seed = (int(x) * 13 + int(y) * 29 + 3) % 2_000_000
    elements.append({
        "id": tid, "type": "text", "x": x, "y": y, "width": w,
        "height": len(lines) * (font + 6), "angle": 0, "strokeColor": color,
        "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
        "frameId": None, "roundness": None, "seed": seed, "version": 1,
        "versionNonce": seed, "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False, "fontSize": font,
        "fontFamily": 1, "text": s, "textAlign": "left", "verticalAlign": "top",
        "containerId": None, "originalText": s, "lineHeight": 1.25,
    })


def container(x, y, w, h, title, *, stroke="#adb5bd"):
    seed = (int(x) * 5 + int(y) * 3 + 1) % 2_000_000
    elements.append({
        "id": _id("frame"), "type": "rectangle", "x": x, "y": y, "width": w,
        "height": h, "angle": 0, "strokeColor": stroke,
        "backgroundColor": "transparent", "fillStyle": "hachure",
        "strokeWidth": 1.5, "strokeStyle": "dotted", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 3}, "seed": seed, "version": 1,
        "versionNonce": seed, "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
    })
    text_block(x + 14, y + 9, title, color=stroke, font=15, w=w - 28)


def poly_arrow(pts, *, color="#adb5bd", dashed=False, width=2.0, opacity=100,
               lbl=None):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    seed = (int(x0) * 7 + int(ys[-1]) * 11 + 5) % 2_000_000
    elements.append({
        "id": _id("arrow"), "type": "arrow", "x": x0, "y": y0,
        "width": max(xs) - min(xs), "height": max(ys) - min(ys), "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": width,
        "strokeStyle": "dashed" if dashed else "solid", "roughness": 1,
        "opacity": opacity, "groupIds": [], "frameId": None,
        "roundness": {"type": 2}, "seed": seed, "version": 1,
        "versionNonce": seed, "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False, "points": rel,
        "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
    })
    if lbl:
        mx, my = pts[len(pts) // 2]
        text_block(mx - 40, my - 26, lbl, color=color, font=12, w=150)


# ── 1. resolve geometry from (layer order, col) ─────────────────────────────
layer_order = {ly["id"]: ly["order"] for ly in SPEC["layers"]}
layer_title = {ly["id"]: ly["title"] for ly in SPEC["layers"]}
max_col = {}
for n in SPEC["nodes"]:
    o = layer_order[n["layer"]]
    max_col[o] = max(max_col.get(o, 0), n["col"])

geo = {}
for n in SPEC["nodes"]:
    o = layer_order[n["layer"]]
    band_y = TOP + o * (BAND_H + BAND_GAP)
    x = MARGIN_X + n["col"] * COL_PITCH
    y = band_y + BOX_DY
    geo[n["id"]] = {
        "x": x, "y": y, "w": BOX_W, "h": BOX_H, "order": o,
        "cx": x + BOX_W / 2, "cy": y + BOX_H / 2, "top": y, "bottom": y + BOX_H,
        "left": x, "right": x + BOX_W, "band_y": band_y,
    }

scene_w = MARGIN_X + (max(max_col.values()) + 1) * COL_PITCH + MARGIN_X

# ── 2. title ────────────────────────────────────────────────────────────────
text_block(MARGIN_X, 24, SPEC["title"], font=30, w=1100)
text_block(MARGIN_X, 64, SPEC["subtitle"], color="#868e96", font=15, w=scene_w - 80)

# ── 3. layer bands (containers) ─────────────────────────────────────────────
band_stroke = {0: "#495057", 1: "#1971c2", 2: "#2f9e44", 3: "#e8590c", 4: "#9c36b5"}
for ly in SPEC["layers"]:
    o = ly["order"]
    band_y = TOP + o * (BAND_H + BAND_GAP)
    w = (max_col[o] + 1) * COL_PITCH - GAP_X + (MARGIN_X - 20) * 2
    container(20, band_y, w, BAND_H, f"{ly['id']} · {ly['title']}",
              stroke=band_stroke.get(o, "#adb5bd"))

# ── 4. edges (drawn under boxes so boxes stay readable) ─────────────────────
SEAM_TARGETS = {"core-game-trait", "core-evaluator-trait", "core-encoder-trait"}
SEAM_WORDS = ("impl Game", "impl Encoder", "impl Evaluator", "E::G", "type G",
              "param E")
PLAY_FROM = ("webui-", "wasm-", "bin-")
TRAIN_IDS = {"datagen-bins", "datagen-counterbook", "data-jsonl",
             "trainer-dataset", "trainer-train", "trainer-distill",
             "trainer-model", "trainer-encoding", "trainer-parity",
             "root-safetensors", "served-weights"}
role_of = {n["id"]: n["role"] for n in SPEC["nodes"]}

# curated marquee labels — keep the dense graph readable
KEY = {
    ("core-search", "core-game-trait"): "via E::G",
    ("core-search", "core-evaluator-trait"): "E: Evaluator",
    ("wallchess-glue", "core-game-trait"): "impl Game",
    ("checkers-glue", "core-game-trait"): "impl Game",
    ("core-arena", "core-search"): "BotConfig→Search",
    ("webui-api", "wasm-bundle-engine"): "PLAY",
    ("bin-checkers-cli", "core-search"): "draughts preset",
    ("datagen-bins", "data-jsonl"): "TRAIN: {z,f,pi}",
    ("root-safetensors", "served-weights"): "deploy weights",
    ("trainer-encoding", "wallchess-features"): "encoding parity",
}


def classify(e):
    if e["style"] == "dashed" or role_of.get(e["from"]) == "deferred" \
            or role_of.get(e["to"]) == "deferred":
        return "deferred"
    lbl = e.get("label", "")
    if e["to"] in SEAM_TARGETS or any(w in lbl for w in SEAM_WORDS):
        return "seam"
    if "TRAIN" in lbl or (e["from"] in TRAIN_IDS and e["to"] in TRAIN_IDS):
        return "train"
    if "PLAY" in lbl or (e["from"].startswith(PLAY_FROM)):
        return "play"
    if e["from"] in TRAIN_IDS or e["to"] in TRAIN_IDS:
        return "train"
    return "default"


# stagger intra-band routes through the gap below so they do not overlap
gap_seq = {}


def route(a, b):
    """Return an absolute polyline from node a to node b."""
    oa, ob = a["order"], b["order"]
    if oa == ob:                                   # intra-band → dip into gap
        k = gap_seq.get(oa, 0)
        gap_seq[oa] = k + 1
        gy = a["bottom"] + 26 + (k % 5) * 13
        return [(a["cx"], a["bottom"]), (a["cx"], gy), (b["cx"], gy),
                (b["cx"], b["bottom"])]
    if oa < ob:                                    # a above b
        return [(a["cx"], a["bottom"]), (b["cx"], b["top"])]
    return [(a["cx"], a["top"]), (b["cx"], b["bottom"])]  # a below b


for e in SPEC["edges"]:
    a, b = geo.get(e["from"]), geo.get(e["to"])
    if not a or not b:
        continue
    cat = classify(e)
    color, dflt_dash = EDGE[cat]
    dashed = dflt_dash or e["style"] == "dashed"
    span = abs(a["order"] - b["order"])
    # de-emphasize long edges (they unavoidably cross intervening bands)
    width = 1.4 if span >= 2 else 2.0
    opacity = 55 if span >= 2 else 100
    lbl = KEY.get((e["from"], e["to"]))
    poly_arrow(route(a, b), color=color, dashed=dashed, width=width,
               opacity=opacity, lbl=lbl)

# ── 5. nodes (boxes on top of edges) ────────────────────────────────────────
for n in SPEC["nodes"]:
    g = geo[n["id"]]
    fill, stroke = ROLE.get(n["role"], ROLE["shared"])
    box(g["x"], g["y"], g["w"], g["h"], n["label"], fill=fill, stroke=stroke,
        dashed=(n["role"] == "deferred"))

# ── 6. legend + notes below the last band ───────────────────────────────────
legend_y = TOP + (len(SPEC["layers"])) * (BAND_H + BAND_GAP) - BAND_GAP + 70
text_block(MARGIN_X, legend_y - 34, "Legend", font=20, w=300)
roles_in_order = ["generic", "wallchess", "checkers", "shared", "external",
                  "deferred"]
meaning = {ln.split(" = ")[0].strip(): ln.split(" = ", 1)[1]
           for ln in SPEC["legend"] if " = " in ln}
for i, r in enumerate(roles_in_order):
    fill, stroke = ROLE[r]
    sy = legend_y + i * 30
    box(MARGIN_X, sy, 22, 22, "", fill=fill, stroke=stroke,
        dashed=(r == "deferred"))
    text_block(MARGIN_X + 32, sy + 1, f"{r} — {meaning.get(r, '')}",
               color="#343a40", font=13, w=1000)

# edge-colour legend
ekey_y = legend_y + len(roles_in_order) * 30 + 16
edge_legend = [
    ("seam", "seam: impl Game/Evaluator/Encoder · Search reaches its game via E::G"),
    ("play", "PLAY flow: web UI / wasm / CLI → generic engine"),
    ("train", "TRAIN flow: data-gen CLIs → datasets → trainer → weights"),
    ("deferred", "deferred / not-yet-wired (dashed)"),
]
for i, (cat, desc) in enumerate(edge_legend):
    color, _ = EDGE[cat]
    yy = ekey_y + i * 26
    poly_arrow([(MARGIN_X, yy + 10), (MARGIN_X + 46, yy + 10)], color=color,
               dashed=(cat == "deferred"))
    text_block(MARGIN_X + 58, yy + 1, desc, color="#495057", font=13, w=1000)

# notes block
notes_y = ekey_y + len(edge_legend) * 26 + 28
text_block(MARGIN_X, notes_y - 30, "Notes", font=20, w=300)
wrapped = []
for nt in SPEC.get("notes", []):
    wrapped.append("\n".join(textwrap.wrap("•  " + nt, width=140,
                                           subsequent_indent="   ")))
text_block(MARGIN_X, notes_y, "\n".join(wrapped), color="#495057", font=12,
           w=scene_w - 80)

# ── 7. write scene ──────────────────────────────────────────────────────────
scene = {
    "type": "excalidraw", "version": 2,
    "source": "gameboard-bot/docs/gen_multigame_excalidraw.py",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
out = os.path.join(HERE, "multigame-architecture.excalidraw")
with open(out, "w") as f:
    json.dump(scene, f, indent=2)
print(f"wrote {out}  ({len(elements)} elements, "
      f"{len(SPEC['nodes'])} nodes, {len(SPEC['edges'])} edges)")
