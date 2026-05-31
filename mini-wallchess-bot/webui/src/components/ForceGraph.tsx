// Force-directed graph — 3D, hand-rolled, no graph library.
//
// Physics each frame: Coulomb repulsion + Hooke springs + gravity in 3D.
// Rendered via perspective projection onto a 2D canvas. Drag background to
// orbit, wheel to zoom, drag node to pin, click to travel.
import { useCallback, useEffect, useRef } from 'react'

export type GraphNode = {
  key: string
  /** depth layer (ply) — seeds initial x so the graph unrolls left→right. */
  ply: number
  /** SOUTH win % (undefined = not yet scored). Drives colour. */
  south?: number
  label?: string
}
export type GraphEdge = { a: string; b: string }

type Props = {
  nodes: GraphNode[]
  edges: GraphEdge[]
  currentKey: string
  onTravel: (key: string) => void
  height?: number
}

type Body = {
  key: string
  x: number; y: number; z: number
  vx: number; vy: number; vz: number
  ply: number
  south?: number
  label?: string
  pinned: boolean
}

// --- tunables ---
const REPULSION = 5200
const SPRING = 0.012
const SPRING_LEN = 70
const GRAVITY = 0.015
const DAMPING = 0.86
const MAX_V = 18
const FOV = 500  // perspective focal length

function colorFor(south: number | undefined, css: (v: string) => string): string {
  if (south == null) return '#8a8178'
  return south >= 50 ? css('--lagoon') : css('--pawn-north')
}

// Rotate by yaw (Y-axis) then pitch (X-axis), then perspective-project.
function projPt(
  x: number, y: number, z: number,
  yaw: number, pitch: number, scale: number,
  w: number, h: number,
) {
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw)
  const xr = x * cosY + z * sinY
  const yr = y
  const zr = -x * sinY + z * cosY
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch)
  const xrr = xr
  const yrr = yr * cosP - zr * sinP
  const zrr = yr * sinP + zr * cosP
  const d = (FOV * scale) / (FOV + zrr)
  return { sx: w / 2 + xrr * d, sy: h / 2 + yrr * d, depth: zrr, d }
}

// Back-project screen (sx,sy) to world (x,y) keeping world z=bz fixed.
// Uses the stored perspective factor d from when the drag began.
function unprojPt(
  sx: number, sy: number,
  bz: number, d: number,
  yaw: number, pitch: number,
  w: number, h: number,
) {
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw)
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch)
  const xrr = (sx - w / 2) / d
  const yrr = (sy - h / 2) / d
  // xrr = x*cosY + bz*sinY → x
  const dy = Math.abs(cosY) > 0.001 ? cosY : 0.001 * Math.sign(cosY || 1)
  const x = (xrr - bz * sinY) / dy
  // yrr = y*cosP + x*sinY*sinP - bz*cosY*sinP → y
  const dp = Math.abs(cosP) > 0.001 ? cosP : 0.001 * Math.sign(cosP || 1)
  const y = (yrr - x * sinY * sinP + bz * cosY * sinP) / dp
  return { x, y }
}

function hashStr(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return Math.abs(h)
}

export default function ForceGraph({
  nodes,
  edges,
  currentKey,
  onTravel,
  height = 420,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bodies = useRef<Map<string, Body>>(new Map())
  const view = useRef({ scale: 1, yaw: 0.4, pitch: 0.25 })
  const size = useRef({ w: 800, h: height })
  const hover = useRef<string | null>(null)
  const drag = useRef<{
    key: string | null
    orbiting: boolean
    px: number; py: number
    bz: number; d: number
  }>({ key: null, orbiting: false, px: 0, py: 0, bz: 0, d: 1 })
  const adj = useRef<Map<string, Set<string>>>(new Map())
  const moved = useRef(false)

  const css = useCallback((v: string) => {
    if (typeof window === 'undefined') return '#888'
    return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888'
  }, [])

  // Sync incoming nodes/edges into the simulation, preserving live positions.
  useEffect(() => {
    const m = bodies.current
    const present = new Set<string>()
    for (const n of nodes) {
      present.add(n.key)
      const b = m.get(n.key)
      if (b) {
        b.south = n.south; b.label = n.label; b.ply = n.ply
      } else {
        const hv = hashStr(n.key)
        m.set(n.key, {
          key: n.key,
          x: (n.ply - 1) * 60 + ((hv % 100) - 50) * 0.6,
          y: (((hv >> 7) % 100) - 50) * 1.2,
          z: (((hv >> 14) % 100) - 50) * 1.2,
          vx: 0, vy: 0, vz: 0,
          ply: n.ply, south: n.south, label: n.label, pinned: false,
        })
      }
    }
    for (const k of [...m.keys()]) if (!present.has(k)) m.delete(k)

    const a = new Map<string, Set<string>>()
    for (const e of edges) {
      if (!a.has(e.a)) a.set(e.a, new Set())
      if (!a.has(e.b)) a.set(e.b, new Set())
      a.get(e.a)!.add(e.b)
      a.get(e.b)!.add(e.a)
    }
    adj.current = a
  }, [nodes, edges])

  // Physics + render loop.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    let raf = 0

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      size.current = { w, h: height }
      canvas.width = w * dpr
      canvas.height = height * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const step = () => {
      const m = bodies.current
      const arr = [...m.values()]

      // 3D Coulomb repulsion — all pairs (O(n²))
      for (let i = 0; i < arr.length; i++) {
        const a = arr[i]
        for (let j = i + 1; j < arr.length; j++) {
          const b = arr[j]
          let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z
          let d2 = dx * dx + dy * dy + dz * dz
          if (d2 < 0.01) {
            dx = (hashStr(a.key + b.key) % 10) - 5
            dy = (hashStr(b.key + a.key) % 10) - 5
            dz = (hashStr(a.key + b.key + 'z') % 10) - 5
            d2 = dx * dx + dy * dy + dz * dz + 0.01
          }
          const f = REPULSION / d2, d = Math.sqrt(d2)
          const fx = (dx / d) * f, fy = (dy / d) * f, fz = (dz / d) * f
          a.vx += fx; a.vy += fy; a.vz += fz
          b.vx -= fx; b.vy -= fy; b.vz -= fz
        }
      }

      // 3D Hooke springs along edges
      for (const e of edges) {
        const a = m.get(e.a), b = m.get(e.b)
        if (!a || !b) continue
        const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.01
        const f = SPRING * (d - SPRING_LEN)
        const fx = (dx / d) * f, fy = (dy / d) * f, fz = (dz / d) * f
        a.vx += fx; a.vy += fy; a.vz += fz
        b.vx -= fx; b.vy -= fy; b.vz -= fz
      }

      // Gravity toward origin + integrate
      for (const b of arr) {
        if (b.pinned) { b.vx = 0; b.vy = 0; b.vz = 0; continue }
        const g = GRAVITY * 0.05
        b.vx += -b.x * g; b.vy += -b.y * g; b.vz += -b.z * g
        b.vx = Math.max(-MAX_V, Math.min(MAX_V, b.vx * DAMPING))
        b.vy = Math.max(-MAX_V, Math.min(MAX_V, b.vy * DAMPING))
        b.vz = Math.max(-MAX_V, Math.min(MAX_V, b.vz * DAMPING))
        b.x += b.vx; b.y += b.vy; b.z += b.vz
      }

      draw(ctx)
      raf = requestAnimationFrame(step)
    }

    const draw = (c: CanvasRenderingContext2D) => {
      const { w, h } = size.current
      const { yaw, pitch, scale } = view.current
      c.clearRect(0, 0, w, h)
      c.fillStyle = css('--board-dark') || '#181410'
      c.fillRect(0, 0, w, h)

      const hk = hover.current
      const nbrs = hk ? adj.current.get(hk) : undefined
      const m = bodies.current

      // project all bodies
      const ps = new Map<string, ReturnType<typeof projPt>>()
      for (const b of m.values()) ps.set(b.key, projPt(b.x, b.y, b.z, yaw, pitch, scale, w, h))

      // edges
      for (const e of edges) {
        const a = m.get(e.a), b = m.get(e.b)
        if (!a || !b) continue
        const pa = ps.get(e.a)!, pb = ps.get(e.b)!
        const hot = hk && (e.a === hk || e.b === hk)
        c.strokeStyle = hot ? css('--accent-text') : css('--line') || '#333'
        c.globalAlpha = hk ? (hot ? 0.9 : 0.15) : 0.45
        c.lineWidth = 1
        c.beginPath()
        c.moveTo(pa.sx, pa.sy)
        c.lineTo(pb.sx, pb.sy)
        c.stroke()
      }
      c.globalAlpha = 1

      // nodes — sort back-to-front for correct occlusion
      const sorted = [...m.values()].sort((a, b) => ps.get(a.key)!.depth - ps.get(b.key)!.depth)
      for (const b of sorted) {
        const p = ps.get(b.key)!
        const isCur = b.key === currentKey
        const isHover = b.key === hk
        const dim = !!(hk && !isHover && !(nbrs && nbrs.has(b.key)))
        c.globalAlpha = dim ? 0.25 : 1
        // perspective-scale the radius so far nodes shrink
        const r = Math.max(2, (isCur ? 9 : isHover ? 7 : 4.5) * p.d)
        c.beginPath()
        c.arc(p.sx, p.sy, r, 0, Math.PI * 2)
        c.fillStyle = colorFor(b.south, css)
        c.fill()
        if (isCur || isHover) {
          c.lineWidth = 2
          c.strokeStyle = css('--board-frame') || '#fff'
          c.stroke()
        }
        if (isCur || isHover || (nbrs && nbrs.has(b.key))) {
          c.globalAlpha = 1
          c.fillStyle = css('--sea-ink') || '#eee'
          c.font = `${Math.max(8, 12 * p.d)}px ui-sans-serif, system-ui`
          c.fillText(b.label ?? b.key.slice(0, 10), p.sx + r + 3, p.sy + 3)
        }
      }
      c.globalAlpha = 1
    }

    raf = requestAnimationFrame(step)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [edges, currentKey, css, height])

  // Hit-test by screen-space proximity.
  const pick = (sx: number, sy: number): Body | null => {
    const { yaw, pitch, scale } = view.current
    const { w, h } = size.current
    let best: Body | null = null, bestD = 18
    for (const b of bodies.current.values()) {
      const p = projPt(b.x, b.y, b.z, yaw, pitch, scale, w, h)
      const d = Math.hypot(p.sx - sx, p.sy - sy)
      if (d < bestD) { bestD = d; best = b }
    }
    return best
  }

  const onPointerDown = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top
    const hit = pick(sx, sy)
    moved.current = false
    if (hit) {
      hit.pinned = true
      const { yaw, pitch, scale } = view.current
      const { w, h } = size.current
      const p = projPt(hit.x, hit.y, hit.z, yaw, pitch, scale, w, h)
      drag.current = { key: hit.key, orbiting: false, px: sx, py: sy, bz: hit.z, d: p.d }
    } else {
      drag.current = { key: null, orbiting: true, px: sx, py: sy, bz: 0, d: 1 }
    }
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top
    const d = drag.current
    if (d.key) {
      const b = bodies.current.get(d.key)
      if (b) {
        const { yaw, pitch } = view.current
        const { w, h } = size.current
        const wp = unprojPt(sx, sy, d.bz, d.d, yaw, pitch, w, h)
        b.x = wp.x; b.y = wp.y; b.z = d.bz
        b.vx = 0; b.vy = 0; b.vz = 0
      }
      moved.current = true
    } else if (d.orbiting) {
      const dx = sx - d.px, dy = sy - d.py
      view.current.yaw += dx * 0.005
      view.current.pitch = Math.max(
        -Math.PI / 2 + 0.1,
        Math.min(Math.PI / 2 - 0.1, view.current.pitch + dy * 0.005),
      )
      d.px = sx; d.py = sy
      moved.current = true
    } else {
      hover.current = pick(sx, sy)?.key ?? null
    }
  }

  const onPointerUp = () => {
    const d = drag.current
    if (d.key) {
      const b = bodies.current.get(d.key)
      if (b) b.pinned = false
      if (!moved.current) onTravel(d.key)
    }
    drag.current = { key: null, orbiting: false, px: 0, py: 0, bz: 0, d: 1 }
  }

  const onWheel = (e: React.WheelEvent) => {
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
    view.current.scale = Math.max(0.25, Math.min(4, view.current.scale * factor))
  }

  const recenter = () => { view.current = { scale: 1, yaw: 0.4, pitch: 0.25 } }

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height, touchAction: 'none', cursor: 'grab' }}
        className="rounded-xl"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onWheel={onWheel}
      />
      <button
        type="button"
        onClick={recenter}
        className="absolute right-2 top-2 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-2 py-1 text-xs font-semibold text-[var(--sea-ink)] transition hover:border-[var(--accent-text)]"
      >
        Recenter
      </button>
    </div>
  )
}
