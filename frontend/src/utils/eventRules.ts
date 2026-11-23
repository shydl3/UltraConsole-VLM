import type { RuleCategory, VlmEventDefinition } from './vlm'

export const ruleCategories = ['poi_event', 'forced_passage', 'integrity'] as const
export const triggerKinds = ['line.cross', 'zone.enter', 'zone.leave', 'initial_inside', 'cue.on', 'integrity.change'] as const
export type EventRule = Required<VlmEventDefinition> & { name: string; version: number; createdAt?: number; updatedAt?: number }
export type RuleLibrary = { revision: number; rules: EventRule[] }
export type RuleSelection = { id: string; version: number; poiIds: string[]; cueIds: string[]; integrityIds: string[] }
export type RuleSnapshot = { libraryRevision: number | null; rules: { id: string; name: string; version: number | null; event: VlmEventDefinition }[] }
export type RulePoi = { id: string; label: string; kind: string; pairIndex?: number; entryDirection?: string | null }
export const newRuleId = () => `event_${crypto.randomUUID().replaceAll('-', '')}`

export const categoryTriggers = (category: RuleCategory): string[] => category === 'forced_passage' ? ['cue.on']
  : category === 'integrity' ? ['integrity.change'] : ['line.cross', 'zone.enter', 'zone.leave']

// System-managed preset: auto-bound to every drawn integrity zone by the backend;
// the editor shows it locked and never treats the synthesized row as dirty.
export const PRESET_RULE_ID = 'preset_integrity'
export function presetRule(): EventRule {
  return { id: PRESET_RULE_ID, name: '完整性检测（预设）', description: '完整性检测区是否出现持续性画面改变（围墙、门体等静态设施被改动）', enabled: true, category: 'integrity', poiIds: [], cueIds: [], integrityIds: [], triggerKinds: ['integrity.change'], cueDedupeSec: 3, preSeconds: 4, postSeconds: 6, criteria: [], uncertainWhen: [], version: 0 }
}
export function newRule(description = '', category: RuleCategory = 'poi_event'): EventRule {
  return { id: newRuleId(), name: description.slice(0, 160), description, enabled: true, category, poiIds: [], cueIds: [], integrityIds: [], triggerKinds: categoryTriggers(category), cueDedupeSec: 3, preSeconds: category === 'forced_passage' ? 8 : 4, postSeconds: 6, criteria: [], uncertainWhen: [], version: 0 }
}
export function rulePayload(rule: EventRule) {
  return { id: rule.id, name: rule.name.trim(), description: rule.description.trim(), enabled: rule.enabled, category: rule.category, poiIds: [], cueIds: [], integrityIds: [], triggerKinds: rule.triggerKinds, cueDedupeSec: rule.cueDedupeSec, preSeconds: rule.preSeconds, postSeconds: rule.postSeconds, criteria: rule.criteria, uncertainWhen: rule.uncertainWhen }
}
export const ruleChanged = (rule: EventRule, saved?: EventRule) => (!saved && rule.id === PRESET_RULE_ID) ? false : !saved || JSON.stringify(rulePayload(rule)) !== JSON.stringify(rulePayload(saved))
export function ruleError(rule: EventRule): string {
  if (!rule.name.trim() || rule.name.length > 160) return '名称不能为空，最多 160 字 / Name: 1–160 characters'
  if (!rule.description.trim() || rule.description.length > 1500) return '事件描述不能为空，最多 1500 字 / Description: 1–1500 characters'
  if (!ruleCategories.includes(rule.category)) return '请选择有效的规则类别 / Select a valid rule category'
  if (!rule.triggerKinds.length || rule.triggerKinds.some(t => !triggerKinds.includes(t as typeof triggerKinds[number]))) return '请选择有效的触发类型 / Select valid triggers'
  if (rule.category === 'forced_passage' && (rule.triggerKinds.length !== 1 || rule.triggerKinds[0] !== 'cue.on')) return '强行通过规则由信号区触发 / Forced-passage rules are triggered by cue.on'
  if (rule.category === 'integrity' && !rule.triggerKinds.includes('integrity.change')) return '完整性规则由完整性变化触发 / Integrity rules use integrity.change'
  if (rule.category !== 'forced_passage' && rule.triggerKinds.includes('cue.on')) return '仅强行通过规则可使用 cue.on / cue.on is reserved for forced-passage rules'
  if (![rule.preSeconds, rule.postSeconds].every(n => Number.isFinite(n) && n >= 0 && n <= 60) || rule.preSeconds + rule.postSeconds <= 0) return '前后窗口各为 0–60 秒，合计必须大于 0 / Invalid time window'
  if ([rule.criteria, rule.uncertainWhen].some(list => list.length > 10 || list.some(s => !s.trim() || s.length > 500))) return '高级条件最多 10 条，每条 1–500 字 / Invalid conditions'
  return ''
}

const indexList = (values: (number | undefined)[]) => values.filter((value): value is number => typeof value === 'number')

/** Category-aware binding validation. Forced-passage needs paired line+cue channels. */
export function bindingError(rule: EventRule, ids: string[], pois: RulePoi[], integrity: RulePoi[] = [], cueIds: string[] = [], cues: RulePoi[] = []) {
  const category = rule.category ?? 'poi_event'
  if (!pois.length && integrity.length && category !== 'integrity') {
    return '当前仅绘制了完整性区，只能启用完整性规则 / Only integrity rules are available'
  }
  if (category === 'integrity') {
    return integrity.length ? '' : '完整性规则需要至少一个完整性区 / Integrity rule needs an integrity zone'
  }
  if (category === 'forced_passage') {
    if (ids.some(id => !pois.some(p => p.id === id))) return '计数 POI 绑定已失效，请重新选择 / Missing counting POI'
    if (cueIds.some(id => !cues.some(c => c.id === id))) return '信号区绑定已失效，请重新选择 / Missing signal ROI'
    const boundPois = ids.length ? pois.filter(p => ids.includes(p.id)) : pois
    const boundCues = cueIds.length ? cues.filter(c => cueIds.includes(c.id)) : cues
    const lines = boundPois.filter(p => p.kind === 'line')
    const zones = boundPois.filter(p => p.kind === 'zone')
    if (!lines.length) return '强行通过规则需要入口计数线 / Forced-passage rules need an entry counting line'
    if (!boundCues.length) return '强行通过规则需要信号区 / Forced-passage rules need a signal ROI'
    const unpairedLine = lines.find(line => line.pairIndex == null)
    const unpairedZone = ids.length ? zones.find(zone => zone.pairIndex == null) : undefined
    const unpairedCue = boundCues.find(cue => cue.pairIndex == null)
    if (unpairedLine || unpairedZone || unpairedCue) return '区域缺少配对编号，请重新绘制 / Region has no pair number'
    const noDirection = lines.find(line => !line.entryDirection)
    if (noDirection) return `请为入口计数线设置进入方向：${noDirection.label} / Set the entry direction`
    const lineIndexes = new Set(indexList(lines.map(line => line.pairIndex)))
    const cueIndexes = new Set(indexList(boundCues.map(cue => cue.pairIndex)))
    const missingCue = [...lineIndexes].filter(index => !cueIndexes.has(index))
    const missingLine = [...cueIndexes].filter(index => !lineIndexes.has(index))
    if (missingCue.length || missingLine.length) {
      const detail = [missingCue.length ? `缺信号区 #${missingCue.join(',#')}` : '', missingLine.length ? `缺入口线 #${missingLine.join(',#')}` : ''].filter(Boolean).join('；')
      return `计数线与信号区必须按编号一一对应：${detail} / Pair line and signal ROI by number`
    }
    const orphanZone = ids.length ? zones.find(zone => !lineIndexes.has(zone.pairIndex as number)) : undefined
    if (orphanZone) return `计数区 ${orphanZone.label} 没有对应的入口线/信号区 / Auxiliary zone has no channel`
    return ''
  }
  if (ids.some(id => !pois.some(p => p.id === id))) return 'POI 绑定已失效，请重新选择 / Missing POI binding'
  const targets = pois.filter(p => !ids.length || ids.includes(p.id))
  const poiOk = targets.some(p => rule.triggerKinds.some(t => p.kind === 'line' ? t === 'line.cross' : t === 'zone.enter' || t === 'zone.leave' || t === 'initial_inside'))
  const integrityOk = rule.triggerKinds.includes('integrity.change') && integrity.length > 0
  if (!poiOk && !integrityOk) return '没有匹配触发类型的 POI / No compatible POI'
  return ''
}
export function parseRuleImport(text: string): EventRule[] {
  const trimmed = text.trim()
  if (!trimmed) throw new Error('请输入规则 / Enter rules')
  if (!trimmed.startsWith('[') && !trimmed.startsWith('{')) return trimmed.split('\n').map(s => s.trim()).filter(Boolean).map(line => newRule(line))
  const parsed: unknown = JSON.parse(trimmed)
  const raw = Array.isArray(parsed) ? parsed : (parsed as { rules?: unknown; events?: unknown })?.rules ?? (parsed as { events?: unknown })?.events
  if (!Array.isArray(raw) || !raw.length) throw new Error('JSON 必须包含非空 rules 或 events 数组 / Missing rules array')
  const ids = new Set<string>()
  return raw.map((value: unknown, index) => {
    if (!value || typeof value !== 'object') throw new Error(`Invalid rule ${index + 1}`)
    const item = value as Record<string, unknown>
    if (typeof item.description !== 'string') throw new Error(`Rule ${index + 1}: description required`)
    const rule = newRule(item.description)
    for (const key of ['name', 'id'] as const) {
      if (item[key] !== undefined) { if (typeof item[key] !== 'string') throw new Error(`Invalid ${key}`); rule[key] = item[key] }
    }
    if (!/^[A-Za-z0-9_-]{1,80}$/.test(rule.id) || ids.has(rule.id)) throw new Error(`Invalid or duplicate ID: ${rule.id}`)
    ids.add(rule.id)
    for (const key of ['criteria', 'uncertainWhen', 'triggerKinds', 'poiIds', 'cueIds', 'integrityIds'] as const) {
      if (item[key] !== undefined) {
        if (!Array.isArray(item[key]) || item[key].some(s => typeof s !== 'string')) throw new Error(`Invalid ${key}`)
        rule[key] = item[key] as string[]
      }
    }
    if (item.category !== undefined) {
      if (typeof item.category !== 'string' || !ruleCategories.includes(item.category as RuleCategory)) throw new Error(`Invalid category: ${String(item.category)}`)
      rule.category = item.category as RuleCategory
      if (item.triggerKinds === undefined) rule.triggerKinds = categoryTriggers(rule.category)
      if (rule.category === 'forced_passage' && item.preSeconds === undefined) rule.preSeconds = 8
    }
    if (item.cueDedupeSec !== undefined) {
      if (typeof item.cueDedupeSec !== 'number' || item.cueDedupeSec < 0 || item.cueDedupeSec > 60) throw new Error('Invalid cueDedupeSec')
      rule.cueDedupeSec = item.cueDedupeSec
    }
    for (const key of ['preSeconds', 'postSeconds'] as const) {
      if (item[key] !== undefined) { if (typeof item[key] !== 'number') throw new Error(`Invalid ${key}`); rule[key] = item[key] }
    }
    if (item.enabled !== undefined) { if (typeof item.enabled !== 'boolean') throw new Error('Invalid enabled'); rule.enabled = item.enabled }
    const error = ruleError(rule)
    if (error) throw new Error(`${index + 1}: ${error}`)
    return rule
  })
}
