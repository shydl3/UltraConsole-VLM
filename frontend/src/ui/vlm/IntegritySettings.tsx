import { useConsoleStore } from '../../store/useConsoleStore'
import { useT } from '../../i18n'
import { INTEGRITY_DEFAULTS } from '../../utils/vlm'
import styles from './CueSettings.module.css'

const FIELDS = [
  ['sampleFps', 0.1],
  ['baselineFrames', 1],
  ['pixelThreshold', 1],
  ['changeRatio', 0.01],
  ['confirmSec', 1],
  ['edgeWeight', 0.1],
  ['minBlobRatio', 0.001],
  ['alignTolerance', 0.005],
] as const

const TOGGLES = ['ignoreDetections', 'vlmReview'] as const

export function IntegritySettings() {
  const t = useT()
  const eventsConfig = useConsoleStore((s) => s.eventsConfig)
  const setEventsConfig = useConsoleStore((s) => s.setEventsConfig)
  const zones = eventsConfig.integrity || []
  if (!zones.length) return null

  function update(id: string, key: string, value: string) {
    const trimmed = value.trim()
    const num = trimmed === '' ? NaN : Number(trimmed)
    setEventsConfig({ integrity: zones.map(z => z.id === id ? { ...z, [key]: Number.isFinite(num) ? num : undefined } : z) })
  }

  function toggle(id: string, key: typeof TOGGLES[number], checked: boolean) {
    setEventsConfig({ integrity: zones.map(z => z.id === id ? { ...z, [key]: checked } : z) })
  }

  return <details className={styles.wrap} open>
    <summary>{t('vlmIntegrity.title')}</summary>
    <p className={styles.hint}>{t('vlmIntegrity.hint')}</p>
    {zones.map(zone => <fieldset key={zone.id} className={styles.cue}>
      <legend>{zone.name || zone.id}</legend>
      <div className={styles.grid}>
        {FIELDS.map(([key, step]) => <label key={key} className={styles.field}>
          <span>{t(`vlmIntegrity.${key}`)}</span>
          <input type="number" step={step} value={zone[key] ?? INTEGRITY_DEFAULTS[key]}
            onChange={e => update(zone.id, key, e.target.value)} />
        </label>)}
        {TOGGLES.map(key => <label key={key} className={styles.field}>
          <span>{t(`vlmIntegrity.${key}`)}</span>
          <input type="checkbox" checked={zone[key] ?? INTEGRITY_DEFAULTS[key]}
            onChange={e => toggle(zone.id, key, e.target.checked)} />
        </label>)}
      </div>
    </fieldset>)}
  </details>
}
