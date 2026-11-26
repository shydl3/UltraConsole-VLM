import type {
  CountingLine,
  CountingZone,
  PredBBox,
  PredInstanceKeypoints,
  PredMask,
} from '../store/useConsoleStore'

export type FitContain = {
  scale: number
  offX: number
  offY: number
  drawW: number
  drawH: number
}

export function fitContain(rectW: number, rectH: number, srcW: number, srcH: number): FitContain {
  const scale = Math.min(rectW / srcW, rectH / srcH)
  return {
    scale,
    drawW: srcW * scale,
    drawH: srcH * scale,
    offX: (rectW - srcW * scale) / 2,
    offY: (rectH - srcH * scale) / 2,
  }
}

// ---- 颜色：稳定哈希 trackId/cls 到色相 ----
function hashStr(s: string) {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h)
}
export function colorFor(key: string | number | undefined, alpha = 0.95): string {
  if (key === undefined || key === null) return `rgba(76, 255, 122, ${alpha})`
  const hue = hashStr(String(key)) % 360
  return `hsla(${hue}, 78%, 60%, ${alpha})`
}

// ---- 文本标签 ----
export function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color = 'rgba(255,255,255,0.95)') {
  ctx.font = '12px ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
  const padX = 6
  const padY = 4
  const w = ctx.measureText(text).width + padX * 2
  const h = 16 + padY * 2
  ctx.fillStyle = 'rgba(0,0,0,0.6)'
  ctx.fillRect(x, y - h, w, h)
  ctx.fillStyle = color
  ctx.fillText(text, x + padX, y - padY)
}

// ---- BBox ----
export function drawBoxes(
  ctx: CanvasRenderingContext2D,
  boxes: PredBBox[],
  fc: FitContain,
  showLabels: boolean,
) {
  ctx.lineWidth = 2
  for (const b of boxes) {
    const color = b.trackId !== undefined ? colorFor(b.trackId) : 'rgba(76,255,122,0.95)'
    ctx.strokeStyle = color
    const x = fc.offX + b.x1 * fc.scale
    const y = fc.offY + b.y1 * fc.scale
    const w = (b.x2 - b.x1) * fc.scale
    const h = (b.y2 - b.y1) * fc.scale
    ctx.strokeRect(x, y, w, h)
    if (showLabels && b.label) drawLabel(ctx, b.label, x, y, color)
  }
}

// ---- Masks（实例分割） ----
export function drawMasks(ctx: CanvasRenderingContext2D, masks: PredMask[], fc: FitContain) {
  for (const m of masks) {
    if (!m.points || m.points.length < 3) continue
    const fill = colorFor(m.trackId ?? m.cls, 0.18)
    const stroke = colorFor(m.trackId ?? m.cls, 0.85)
    ctx.beginPath()
    const [x0, y0] = m.points[0]
    ctx.moveTo(fc.offX + x0 * fc.scale, fc.offY + y0 * fc.scale)
    for (let i = 1; i < m.points.length; i++) {
      const [px, py] = m.points[i]
      ctx.lineTo(fc.offX + px * fc.scale, fc.offY + py * fc.scale)
    }
    ctx.closePath()
    ctx.fillStyle = fill
    ctx.fill()
    ctx.lineWidth = 1.5
    ctx.strokeStyle = stroke
    ctx.stroke()
  }
}

// ---- Keypoints（COCO-pose 17 关节骨架） ----
const COCO_SKELETON: [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
  [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
  [1, 3], [2, 4], [3, 5], [4, 6],
]
export function drawKeypoints(
  ctx: CanvasRenderingContext2D,
  insts: PredInstanceKeypoints[],
  fc: FitContain,
  minConf = 0.25,
) {
  for (const inst of insts) {
    const color = colorFor(inst.trackId ?? inst.cls, 0.95)
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = 2
    const pts = inst.points
    // skeleton
    for (const [a, b] of COCO_SKELETON) {
      if (a >= pts.length || b >= pts.length) continue
      const pa = pts[a]
      const pb = pts[b]
      if ((pa.conf ?? 1) < minConf || (pb.conf ?? 1) < minConf) continue
      ctx.beginPath()
      ctx.moveTo(fc.offX + pa.x * fc.scale, fc.offY + pa.y * fc.scale)
      ctx.lineTo(fc.offX + pb.x * fc.scale, fc.offY + pb.y * fc.scale)
      ctx.stroke()
    }
    // points
    for (const p of pts) {
      if ((p.conf ?? 1) < minConf) continue
      ctx.beginPath()
      ctx.arc(fc.offX + p.x * fc.scale, fc.offY + p.y * fc.scale, 3, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

// ---- 计数线 / 计数区 (归一化坐标 0..1) ----
export function drawCountingLines(
  ctx: CanvasRenderingContext2D,
  lines: CountingLine[],
  fc: FitContain,
  srcW: number,
  srcH: number,
) {
  ctx.lineWidth = 3
  ctx.strokeStyle = 'rgba(255, 215, 0, 0.95)'
  ctx.fillStyle = 'rgba(255, 215, 0, 0.95)'
  for (const l of lines) {
    const ax = fc.offX + l.a.x * srcW * fc.scale
    const ay = fc.offY + l.a.y * srcH * fc.scale
    const bx = fc.offX + l.b.x * srcW * fc.scale
    const by = fc.offY + l.b.y * srcH * fc.scale
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(bx, by)
    ctx.stroke()
    // 端点
    ctx.beginPath()
    ctx.arc(ax, ay, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(bx, by, 4, 0, Math.PI * 2)
    ctx.fill()
    if (l.name) drawLabel(ctx, l.name, ax, ay, 'rgba(255, 215, 0, 0.95)')
  }
}

export function drawCountingZones(
  ctx: CanvasRenderingContext2D,
  zones: CountingZone[],
  fc: FitContain,
  srcW: number,
  srcH: number,
) {
  ctx.lineWidth = 2
  for (const z of zones) {
    if (z.polygon.length < 3) continue
    ctx.strokeStyle = 'rgba(255, 99, 132, 0.95)'
    ctx.fillStyle = 'rgba(255, 99, 132, 0.12)'
    ctx.beginPath()
    const p0 = z.polygon[0]
    ctx.moveTo(fc.offX + p0.x * srcW * fc.scale, fc.offY + p0.y * srcH * fc.scale)
    for (let i = 1; i < z.polygon.length; i++) {
      const p = z.polygon[i]
      ctx.lineTo(fc.offX + p.x * srcW * fc.scale, fc.offY + p.y * srcH * fc.scale)
    }
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
    if (z.name) {
      const px = fc.offX + p0.x * srcW * fc.scale
      const py = fc.offY + p0.y * srcH * fc.scale
      drawLabel(ctx, z.name, px, py, 'rgba(255, 99, 132, 0.95)')
    }
  }
}

// ---- 轨迹（每个 trackId 最近 N 个中心点） ----
export type TrailMap = Map<number, { x: number; y: number; ts: number }[]>

export function pushTrail(trail: TrailMap, trackId: number, cx: number, cy: number, maxLen = 24) {
  const arr = trail.get(trackId) || []
  arr.push({ x: cx, y: cy, ts: performance.now() })
  if (arr.length > maxLen) arr.splice(0, arr.length - maxLen)
  trail.set(trackId, arr)
}

export function pruneTrails(trail: TrailMap, activeIds: Set<number>, ttlMs = 4000) {
  const now = performance.now()
  for (const [id, arr] of trail) {
    if (!activeIds.has(id) && arr.length > 0 && now - arr[arr.length - 1].ts > ttlMs) {
      trail.delete(id)
    }
  }
}

export function drawTrails(ctx: CanvasRenderingContext2D, trail: TrailMap, fc: FitContain) {
  ctx.lineWidth = 2
  for (const [id, pts] of trail) {
    if (pts.length < 2) continue
    ctx.strokeStyle = colorFor(id, 0.7)
    ctx.beginPath()
    const p0 = pts[0]
    ctx.moveTo(fc.offX + p0.x * fc.scale, fc.offY + p0.y * fc.scale)
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i]
      ctx.lineTo(fc.offX + p.x * fc.scale, fc.offY + p.y * fc.scale)
    }
    ctx.stroke()
  }
}

export type MagnifierGeometry = {
  sourceX: number
  sourceY: number
  sourceHalf: number
  displayHalf: number
}

/** Map the lens centre to source pixels for precise region drawing. */
export function magnifierGeometry(
  cursor: { x: number; y: number },
  rectW: number,
  rectH: number,
  srcW: number,
  srcH: number,
  zoom = 3,
  radius = 72,
): MagnifierGeometry | null {
  if (rectW <= 0 || rectH <= 0 || srcW <= 0 || srcH <= 0 || zoom <= 0 || radius <= 0) return null
  if (cursor.x < 0 || cursor.y < 0 || cursor.x > rectW || cursor.y > rectH) return null
  const fc = fitContain(rectW, rectH, srcW, srcH)
  const sourceX = (cursor.x - fc.offX) / fc.scale
  const sourceY = (cursor.y - fc.offY) / fc.scale
  if (sourceX < 0 || sourceY < 0 || sourceX > srcW || sourceY > srcH) return null
  return {
    sourceX,
    sourceY,
    sourceHalf: radius / (fc.scale * zoom),
    displayHalf: radius / zoom,
  }
}

/**
 * Draw a circular zoom lens over the drawing canvas. The underlying image is
 * sampled from the video/image and the canvas itself is magnified a second
 * time, so the in-progress polygon/line also appears enlarged.
 */
export function drawRegionMagnifier(
  ctx: CanvasRenderingContext2D,
  anchor: HTMLVideoElement | HTMLImageElement,
  canvas: HTMLCanvasElement,
  cursor: { x: number; y: number },
  rectW: number,
  rectH: number,
  srcW: number,
  srcH: number,
  options: { zoom?: number; radius?: number; color?: string } = {},
) {
  const zoom = options.zoom ?? 3
  const radius = options.radius ?? 72
  const color = options.color ?? 'rgba(45, 212, 255, 0.95)'
  const geometry = magnifierGeometry(cursor, rectW, rectH, srcW, srcH, zoom, radius)
  if (!geometry) return
  if (anchor instanceof HTMLVideoElement && anchor.readyState < 2) return
  if (anchor instanceof HTMLImageElement && !anchor.complete) return

  const { x, y } = cursor
  ctx.save()
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.clip()

  ctx.fillStyle = 'rgba(8, 12, 18, 0.94)'
  ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2)

  const { sourceX, sourceY, sourceHalf, displayHalf } = geometry
  ctx.drawImage(
    anchor,
    sourceX - sourceHalf, sourceY - sourceHalf, sourceHalf * 2, sourceHalf * 2,
    x - radius, y - radius, radius * 2, radius * 2,
  )

  // Magnify the already-drawn vectors/drafts from the overlay canvas. Source
  // coordinates are bitmap pixels; destination uses the current CSS transform.
  const dprX = canvas.width / rectW
  const dprY = canvas.height / rectH
  ctx.drawImage(
    canvas,
    (x - displayHalf) * dprX, (y - displayHalf) * dprY, displayHalf * 2 * dprX, displayHalf * 2 * dprY,
    x - radius, y - radius, radius * 2, radius * 2,
  )
  ctx.restore()

  ctx.save()
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)'
  ctx.lineWidth = 4
  ctx.stroke()
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.stroke()

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y)
  ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9)
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(x, y, 2.5, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  ctx.restore()
}
