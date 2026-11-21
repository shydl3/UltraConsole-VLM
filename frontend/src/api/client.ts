import type {
  ConsoleBootstrap, DeviceType, ModelInfo, Params,
  SystemStats,
} from './contracts'

export type * from './contracts'

const API_TOKEN = String(import.meta.env.VITE_ULTRA_API_TOKEN ?? '').trim()

function apiFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  const headers = new Headers(init.headers)
  if (API_TOKEN) headers.set('authorization', `Bearer ${API_TOKEN}`)
  return fetch(input, { ...init, headers })
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export async function apiHealth() {
  const res = await apiFetch('/api/health')
  return json<{ ok: boolean; ts: number }>(res)
}

export async function apiListModels() {
  const res = await apiFetch('/api/models')
  return json<ModelInfo[]>(res)
}

export async function apiGetState() {
  const res = await apiFetch('/api/state')
  return json<ConsoleBootstrap>(res)
}

export async function apiSelectModel(modelId: string) {
  const res = await apiFetch('/api/models/select', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ modelId }),
  })
  return json<ModelInfo>(res)
}

export async function apiSelectEngine(device: DeviceType) {
  const res = await apiFetch('/api/engine/select', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ device }),
  })
  return json<{ ok: boolean; device: DeviceType; warming: boolean }>(res)
}

export async function apiGetParams() {
  const res = await apiFetch('/api/params')
  return json<Params>(res)
}

export async function apiUpdateParams(p: Params) {
  const res = await apiFetch('/api/params', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(p),
  })
  return json<Params>(res)
}

export async function apiExportLogsCsv() {
  const res = await apiFetch('/api/logs/export.csv')
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.text()
}

export async function apiSystemStats() {
  const res = await apiFetch('/api/system/stats')
  return json<SystemStats>(res)
}

let sessionReady: Promise<void> | null = null

/**
 * Establish the HttpOnly auth cookie once so <video>/<img> elements can stream
 * media via Range requests. Without it, multi-gigabyte previews would have to be
 * downloaded entirely into browser memory as blobs.
 */
export function apiEnsureSession(): Promise<void> {
  if (!API_TOKEN) return Promise.resolve()
  if (!sessionReady) {
    sessionReady = apiFetch('/api/auth/session', { method: 'POST' })
      .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`) })
      .catch(() => { sessionReady = null })
  }
  return sessionReady
}

export function wsInferUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}/ws/infer`
}

export function wsAuthProtocols(): string[] | undefined {
  if (!API_TOKEN) return undefined
  const bytes = new TextEncoder().encode(API_TOKEN)
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  const encoded = btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '')
  return [`ultra-token.${encoded}`]
}

export async function apiVlm<T>(path: string, init: RequestInit = {}): Promise<T> {
  return json<T>(await apiFetch(`/api/vlm${path}`, init))
}

export async function apiVlmMedia(path: string): Promise<string> {
  // Cookie auth lets media elements stream directly; never blob whole videos.
  await apiEnsureSession()
  return `/api/vlm${path}`
}

/** One entry of the server-side video browser (see GET /api/vlm/browse). */
export type ServerBrowseListing = {
  root: string
  path: string
  parent: string | null
  dirs: { name: string; path: string }[]
  files: { name: string; path: string; size: number; mtime: number }[]
  truncated: boolean
}

/** A video that already lives on the server; playback and analysis use videoId, no upload. */
export type ServerVideo = { videoId: string; name: string; size: number }

export function apiVlmBrowse(path: string, signal?: AbortSignal) {
  return apiVlm<ServerBrowseListing>(`/browse?path=${encodeURIComponent(path)}`, { signal })
}

export async function apiVlmImportVideo(path: string): Promise<ServerVideo> {
  const meta = await apiVlm<{ videoId: string; fileName: string; sizeBytes: number }>('/videos/import', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  return { videoId: meta.videoId, name: meta.fileName, size: meta.sizeBytes }
}

export type VideoPreviewStatus = { state: 'uploading' | 'converting' | 'ready' | 'failed' | 'cancelled'; progress: number; error?: string }

export async function apiCreateVideoPreviewFromVideo(videoId: string) {
  return json<{ id: string }>(await apiFetch('/api/video/previews/from-video', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ videoId }),
  }))
}

export async function apiVideoPreviewStatus(id: string, signal: AbortSignal) {
  return json<VideoPreviewStatus>(await apiFetch(`/api/video/previews/${id}`, { signal }))
}

export async function apiDeleteVideoPreview(id: string) {
  await apiFetch(`/api/video/previews/${id}`, { method: 'DELETE' })
}

export async function apiVideoPreviewUrl(id: string, signal: AbortSignal) {
  await apiEnsureSession()
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError')
  return `/api/video/previews/${id}/source`
}

export async function apiLibrary<T>(path = '', init: RequestInit = {}): Promise<T> {
  return json<T>(await apiFetch(`/api/library${path}`, init))
}

export async function apiLibraryMedia(url: string, signal?: AbortSignal): Promise<string> {
  await apiEnsureSession()
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
  return url
}
