import { memo, useEffect, useState } from 'react'
import { apiVlmMedia } from '../../api/client'
import { useT } from '../../i18n'
import { formatDuration } from '../../utils/library'
import { decisionKey, decisionTone, isFlagged } from '../../utils/review'
import type { VlmCondition, VlmResult } from '../../utils/vlm'
import { ModelExplanation } from '../vlm/ModelExplanation'
import { QualityWarnings } from '../vlm/QualityWarnings'
import styles from './ReviewDialog.module.css'

function EvidenceImage({ runId, path }: { runId: string; path: string }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    let alive = true
    let resource = ''
    void apiVlmMedia(`/runs/${runId}/artifacts/${path}`).then(value => {
      resource = value
      if (alive) setUrl(value)
      else if (value.startsWith('blob:')) URL.revokeObjectURL(value)
    }).catch(() => { /* The result and source video remain available. */ })
    return () => { alive = false; if (resource.startsWith('blob:')) URL.revokeObjectURL(resource) }
  }, [runId, path])
  return url ? <img className={styles.evidence} src={url} alt="Sampled target evidence" /> : null
}

function ConditionList({ conditions, criteria, onSeek }: {
  conditions?: VlmCondition[]; criteria?: string[]; onSeek: (sec: number) => void
}) {
  const t = useT()
  if (!conditions?.length) return null
  const label = (value: string) => t(value === 'yes' ? 'vlm.conditionYes' : value === 'no' ? 'vlm.conditionNo' : 'vlm.conditionUnknown')
  return <div className={styles.section}>
    <strong>{t('vlm.conditions')}</strong>
    <ul>{conditions.map(condition => <li key={condition.index}>
      {criteria?.[condition.index] ?? `#${condition.index}`} · {label(condition.value)}
      {condition.evidenceTimeSec != null && <button type="button" onClick={() => onSeek(condition.evidenceTimeSec as number)}>
        {t('vlm.evidenceAt')} {formatDuration(condition.evidenceTimeSec, t)}
      </button>}
    </li>)}</ul>
  </div>
}

export const ResultCard = memo(function ResultCard({ result, runId, ruleName, expanded, active, onToggle, onJump, onSeek }: {
  result: VlmResult
  runId: string
  ruleName?: string
  expanded: boolean
  active: boolean
  onToggle: (resultId: string) => void
  onJump: (resultId: string) => void
  onSeek: (sec: number) => void
}) {
  const t = useT()
  const integrity = result.triggerKind === 'integrity.change' ? result.integrityEvidence : undefined
  const cue = result.triggerKind === 'cue.on' ? result.cueEvidence : undefined
  return <article id={`result-${result.resultId}`} className={[styles.card, active ? styles.cardActive : ''].filter(Boolean).join(' ')}>
    <div className={styles.cardHead}>
      <button type="button" className={styles.cardTitle} aria-expanded={expanded} onClick={() => onToggle(result.resultId)}>
        {cue
          ? `${cue.cueLabel ?? cue.triggerCueId ?? result.cueId ?? 'C'} · ${formatDuration(result.triggerSec, t)} · ${cue.poiLabel ?? result.poiId}`
          : integrity ? `${integrity.label} · ${formatDuration(result.triggerSec, t)}` : `#${result.trackId} · ${formatDuration(result.triggerSec, t)} · ${result.poiId}`}
      </button>
      <span className={[styles.badge, styles[decisionTone(result)]].join(' ')}>{t(decisionKey(result))}</span>
    </div>
    <div className={styles.cardMeta}>
      {ruleName && <span className={styles.rule}>{ruleName}</span>}
      <span>{t(`eventRules.${result.triggerKind}`)}</span>
      {isFlagged(result) && <span>{t('review.flagged')}</span>}
      <button type="button" className={styles.jump} onClick={() => onJump(result.resultId)}>{t('review.jump')}</button>
    </div>
    <p className={styles.description}>{result.description}</p>
    {expanded && <>
      {result.deterministicDecision && <div className={styles.section}>
        {t('vlm.deterministic')}: {t(`vlm.${result.deterministicDecision}`)}
        {result.cueEvidence ? ` · ${t('vlm.cueState')}=${result.cueEvidence.state}${result.cueEvidence.onsetSec != null ? ` @${formatDuration(result.cueEvidence.onsetSec, t)}` : ''}` : ''}
      </div>}
      {cue && <div className={styles.section}>
        <strong>{t('vlm.cueTrigger')}</strong>
        <span>{cue.cueLabel ?? cue.triggerCueId} (#{cue.pairIndex ?? '?'}) · {t('vlm.cueInterval')} {cue.interval ? `${formatDuration(cue.interval[0], t)}–${formatDuration(cue.interval[1], t)}` : '—'}</span>
        {(cue.rawTriggerCount ?? 1) > 1 && <span> · {t('vlm.cueRawTriggers')} {cue.rawTriggerCount}</span>}
        <span> · {t('vlm.pairedPoi')}: {cue.poiLabel ?? cue.poiId} ({cue.poiKind ?? 'line'})</span>
        <span> · {cue.retreated
          ? `${t('vlm.retreated')}${cue.retreatSec != null ? ` @${formatDuration(cue.retreatSec, t)}` : ''}`
          : t('vlm.notRetreated')}</span>
        {cue.entrySec != null && <span> · {t('vlm.entryAt')} {formatDuration(cue.entrySec, t)}</span>}
      </div>}
      {integrity && <div className={styles.section}>
        <strong>{t('vlm.integrityEvidence')}</strong>
        {integrity.changed === false ? <span>
          {t('vlm.integrityNegative')}
          {integrity.sampleCount != null ? ` · ${t('vlm.integritySamples', { count: integrity.sampleCount })}` : ''}
          {integrity.baselineSec != null ? ` · ${t('vlm.integrityMonitorFrom')} ${formatDuration(integrity.baselineSec, t)}` : ''}
        </span> : <>
          <span>
            {t('vlm.integrityOnset')} {integrity.onsetSec != null ? formatDuration(integrity.onsetSec, t) : '—'} · {t('vlm.integrityPeak')} {(integrity.peakRatio * 100).toFixed(1)}%
            {integrity.offsetSec != null && integrity.onsetSec != null
              ? ` · ${t('vlm.integrityDuration')} ${formatDuration(integrity.offsetSec - integrity.onsetSec, t)}`
              : ` · ${t('vlm.integrityToEndTime')}`}
          </span>
          <div className={styles.integrityGrid}>
            {(['before', 'after', 'diff'] as const).map(key => {
              const path = integrity[key]
              if (!path) return null
              const caption = key === 'before' ? t('vlm.integrityBefore') : key === 'after' ? t('vlm.integrityAfter') : t('vlm.integrityDiff')
              return <figure key={key} className={styles.integrityFig}>
                <EvidenceImage runId={runId} path={path} />
                <figcaption>{caption}</figcaption>
              </figure>
            })}
          </div>
          {integrity.context && <EvidenceImage runId={runId} path={integrity.context} />}
        </>}
      </div>}
      <ConditionList conditions={result.conditions} criteria={result.eventCriteria} onSeek={onSeek} />
      <ModelExplanation videoDescription={result.videoDescription} reason={result.reason} />
      {result.modelDecision && result.modelDecision !== result.decision && <div className={styles.section}>
        {t('vlm.originalDecision')}: {t(`vlm.${result.modelDecision}`)}
      </div>}
      {result.modelDecisionConfidence != null && <div className={styles.section}>
        {t('vlm.confidence')}: {(result.modelDecisionConfidence * 100).toFixed(0)}%
      </div>}
      <div className={styles.section}><QualityWarnings warnings={result.qualityWarnings} /></div>
      {result.downgradeReasons?.length ? <div className={styles.section}>{t('vlm.review')}: {result.downgradeReasons.join(', ')}</div> : null}
      {result.error && <div className={styles.error}>{result.error}</div>}
      {(result.segments || []).map((segment, index) => <div key={index} className={styles.section}>
        <p className={styles.muted}>{formatDuration(segment.window.startSec, t)}–{formatDuration(segment.window.endSec, t)}</p>
        <ModelExplanation videoDescription={segment.videoDescription} reason={segment.reason} />
        <ConditionList conditions={segment.conditions} criteria={result.eventCriteria} onSeek={onSeek} />
        {segment.previewPath && !integrity && <EvidenceImage runId={runId} path={segment.previewPath} />}
      </div>)}
    </>}
  </article>
})
