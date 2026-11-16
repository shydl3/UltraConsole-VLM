import { useT } from '../../i18n'
import { PRESET_RULE_ID, type RuleSnapshot } from '../../utils/eventRules'
import type { ReviewFacets } from '../../utils/review'
import styles from './ReviewDialog.module.css'

/** Per-rule verdict coverage for one run: every rule stays visible, positives and negatives alike. */
export function RuleCoverage({ rules, byEvent, onSelect }: {
  rules: RuleSnapshot['rules']
  byEvent: ReviewFacets['byEvent']
  onSelect: (eventId: string) => void
}) {
  const t = useT()
  if (!rules.length) return null
  return <section className={styles.coverage} aria-label={t('review.coverage')}>
    <strong>{t('review.coverage')}</strong>
    <ul>{rules.map(rule => {
      const counts = byEvent[rule.id]
      const total = counts ? counts.match + counts.no_match + counts.uncertain : 0
      const integrity = (rule.event.triggerKinds ?? []).includes('integrity.change')
      return <li key={rule.id} className={styles.coverageRow}>
        <button type="button" className={styles.coverageName} onClick={() => onSelect(rule.id)} title={rule.event.description}>
          {rule.name}{rule.id === PRESET_RULE_ID && <span className={styles.presetChip}>{t('eventRules.preset')}</span>}
        </button>
        <span className={styles.coverageCounts}>
          {!total && <span className={styles.coverageNone}>{t('review.notTriggered')}</span>}
          {!!counts?.match && <span className={[styles.badge, styles.match].join(' ')}>{integrity ? t('vlm.integrityPositiveShort') : t('vlm.match')} {counts.match}</span>}
          {!!counts?.no_match && <span className={[styles.badge, styles.no_match].join(' ')}>{integrity ? t('vlm.integrityNegativeShort') : t('vlm.no_match')} {counts.no_match}</span>}
          {!!counts?.uncertain && <span className={[styles.badge, styles.uncertain].join(' ')}>{t('vlm.uncertain')} {counts.uncertain}</span>}
        </span>
      </li>
    })}</ul>
  </section>
}
