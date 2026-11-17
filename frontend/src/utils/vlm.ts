import type { EventsConfig } from '../api/contracts'
import type { VlmParameters } from './vlmParameters'

export type VlmEventDefinition = {
  id: string; description: string; enabled?: boolean; poiIds?: string[]; cueIds?: string[]; integrityIds?: string[]; triggerKinds?: string[]
  preSeconds?: number; postSeconds?: number; criteria?: string[]; uncertainWhen?: string[]
}
export type VlmProgress = {
  stage?: string; sampledFrames?: number; candidateCount?: number; trackingFraction?: number
  mediaTimeSec?: number; completedRequests?: number; totalRequests?: number; failedRequests?: number
  currentSegment?: number; segmentCount?: number; zoneCount?: number; integrityFraction?: number
}
export type VlmRun = {
  runId: string; videoId: string; fileName?: string; displayName?: string; state: string; error?: string
  progress?: VlmProgress
  summary?: { resultCount: number; reason?: string; dryRun?: boolean; failedRequests?: number }
  parameters?: VlmParameters
  startedAt?: number; finishedAt?: number; updatedAt?: number
}
/** A task row: a server run, or an upload that has not produced a run yet. */
export type VlmTask = {
  key: string; name: string; runId?: string; run?: VlmRun
  upload?: { uploaded: number; total: number }; error?: string
}
export type VlmCondition = {
  index: number; value: 'yes' | 'no' | 'unknown'; evidenceTimeSec?: number | null; frameIndex?: number | null
}
export type VlmIntegrityEvidence = {
  zoneId: string; label: string; onsetSec: number | null; offsetSec: number | null; confirmedSec: number | null
  peakRatio: number; sampleFps: number; confirmSec: number
  changed?: boolean; sampleCount?: number; baselineSec?: number | null
  before?: string; after?: string; diff?: string; context?: string
}
export type VlmResult = {
  resultId: string; trackId?: number | null; trackSegmentId?: string; poiId: string; triggerSec: number
  triggerKind: string; eventDefinitionId: string; description: string; decision: string
  inferenceStatus: string; modelDecisionConfidence: number | null; reason: string; error?: string
  modelDecision?: string; videoDescription?: string | null; deterministicDecision?: string
  eventCriteria?: string[]; conditions?: VlmCondition[]
  cueEvidence?: { cueIds: string[]; state: string; onsetSec: number | null }
  integrityEvidence?: VlmIntegrityEvidence
  previewPath?: string; downgradeReasons?: string[]; qualityWarnings?: string[]; postprocessVersion?: string
  segments?: { previewPath?: string; reason?: string; videoDescription?: string | null; decision?: string; conditions?: VlmCondition[]; rawOutputPath?: string; sampleManifest?: string; window: { startSec: number; endSec: number } }[]
}

export const activeVlm = (state: string) => ['queued', 'tracking', 'cues', 'integrity', 'understanding', 'cancelling'].includes(state)

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/**
 * Overall task percentage from the per-phase counters written by the worker.
 * Weights mirror the pipeline: segmentation 4%, pose tracking 40%, cue
 * detection 6%, integrity detection 4%, Qwen judgments the remaining 46%.
 */
export function vlmProgress(run: Pick<VlmRun, 'state'> & { progress?: VlmProgress }): { percent: number; stage: string | null } {
  const p = run.progress ?? {}
  const tracking = 4 + 40 * clamp(p.trackingFraction ?? 0, 0, 1)
  const total = p.totalRequests ?? 0
  const judgments = total > 0 ? 54 + 45 * clamp((p.completedRequests ?? 0) / total, 0, 1) : 54
  switch (run.state) {
    case 'queued': return { percent: 0, stage: 'queued' }
    case 'segment': return { percent: 4, stage: 'segment' }
    case 'tracking': return { percent: Math.round(tracking), stage: 'tracking' }
    case 'cues': return { percent: 46, stage: 'cues' }
    case 'integrity': return { percent: Math.round(50 + 4 * clamp(p.integrityFraction ?? 0, 0, 1)), stage: 'integrity' }
    case 'understanding': return { percent: Math.round(judgments), stage: 'understanding' }
    case 'completed':
    case 'partial': return { percent: 100, stage: 'understanding' }
    case 'cancelling': return { percent: Math.round(Math.max(tracking, judgments)), stage: 'cancelling' }
    default:
      // failed / cancelled / interrupted keep the last known position.
      return { percent: Math.round(Math.max(tracking, judgments)), stage: p.stage ?? null }
  }
}

/**
 * Local submissions keep their submission order and enrich with server progress;
 * runs the browser has not submitted (after a refresh) are appended as-is.
 */
export function mergeVlmTasks(local: VlmTask[], runs: VlmRun[]): VlmTask[] {
  const byRun = new Map(runs.map(run => [run.runId, run]))
  const seen = new Set<string>()
  const merged = local.map(task => {
    const run = task.runId ? byRun.get(task.runId) : undefined
    if (run) seen.add(run.runId)
    return { ...task, name: run?.displayName || run?.fileName || task.name, run: run ?? task.run }
  })
  for (const run of runs) {
    if (!seen.has(run.runId)) merged.push({ key: run.runId, name: run.displayName || run.fileName || run.runId, runId: run.runId, run })
  }
  return merged
}

export const CUE_DEFAULTS = { minOnRatio: 0.06, minR: 120, minRG: 50, minRB: 50, minOnSec: 0.2, minOffSec: 0.2, sampleFps: 10 }

export const INTEGRITY_DEFAULTS = {
  sampleFps: 0.5, baselineFrames: 5, pixelThreshold: 25, changeRatio: 0.04, confirmSec: 10,
  edgeWeight: 0.5, minBlobRatio: 0.005, alignTolerance: 0.02, ignoreDetections: true, vlmReview: false,
}

export type VlmPoi = {
  id: string; label: string; kind: string; points: number[][]; anchor: string
  entryDirection?: 'positive_to_negative' | 'negative_to_positive' | null
}

export function vlmPois(config: EventsConfig): VlmPoi[] {
  return [
    ...config.lines.map(l => ({ id: l.id, label: l.name || l.id, kind: 'line', points: [[l.a.x, l.a.y], [l.b.x, l.b.y]], anchor: 'bbox_center', entryDirection: l.entryDirection ?? null })),
    ...config.zones.map(z => ({ id: z.id, label: z.name || z.id, kind: 'zone', points: z.polygon.map(p => [p.x, p.y]), anchor: 'bbox_center' })),
  ]
}

export function vlmCues(config: EventsConfig) {
  return (config.cues || []).map(c => ({
    id: c.id, label: c.name || c.id, points: c.polygon.map(p => [p.x, p.y]),
    minOnRatio: c.minOnRatio ?? CUE_DEFAULTS.minOnRatio,
    minR: c.minR ?? CUE_DEFAULTS.minR,
    minRG: c.minRG ?? CUE_DEFAULTS.minRG,
    minRB: c.minRB ?? CUE_DEFAULTS.minRB,
    minOnSec: c.minOnSec ?? CUE_DEFAULTS.minOnSec,
    minOffSec: c.minOffSec ?? CUE_DEFAULTS.minOffSec,
    sampleFps: c.sampleFps ?? CUE_DEFAULTS.sampleFps,
  }))
}

export function vlmIntegrity(config: EventsConfig) {
  return (config.integrity || []).map(z => ({
    id: z.id, label: z.name || z.id, points: z.polygon.map(p => [p.x, p.y]),
    sampleFps: z.sampleFps ?? INTEGRITY_DEFAULTS.sampleFps,
    baselineFrames: z.baselineFrames ?? INTEGRITY_DEFAULTS.baselineFrames,
    pixelThreshold: z.pixelThreshold ?? INTEGRITY_DEFAULTS.pixelThreshold,
    changeRatio: z.changeRatio ?? INTEGRITY_DEFAULTS.changeRatio,
    confirmSec: z.confirmSec ?? INTEGRITY_DEFAULTS.confirmSec,
    edgeWeight: z.edgeWeight ?? INTEGRITY_DEFAULTS.edgeWeight,
    minBlobRatio: z.minBlobRatio ?? INTEGRITY_DEFAULTS.minBlobRatio,
    alignTolerance: z.alignTolerance ?? INTEGRITY_DEFAULTS.alignTolerance,
    ignoreDetections: z.ignoreDetections ?? INTEGRITY_DEFAULTS.ignoreDetections,
    vlmReview: z.vlmReview ?? INTEGRITY_DEFAULTS.vlmReview,
  }))
}

export function parseVlmEvents(text: string): VlmEventDefinition[] {
  const trimmed = text.trim()
  if (!trimmed) throw new Error('请输入至少一条事件描述 / Enter an event description')
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    const raw = JSON.parse(trimmed)
    const events: unknown = Array.isArray(raw) ? raw : raw.events
    if (!Array.isArray(events) || !events.length) throw new Error('JSON must contain a nonempty events array')
    for (const event of events) {
      if (!event || typeof event.id !== 'string' || typeof event.description !== 'string' || !event.description.trim()) {
        throw new Error('Each event requires id and description')
      }
    }
    return events as VlmEventDefinition[]
  }
  return trimmed.split('\n').map(s => s.trim()).filter(Boolean).map((description, index) => ({ id: `event_${index + 1}`, description }))
}
