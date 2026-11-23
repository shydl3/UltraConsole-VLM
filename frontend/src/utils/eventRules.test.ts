import { describe, expect, it } from 'vitest'
import { bindingError, newRule, parseRuleImport, PRESET_RULE_ID, presetRule, ruleChanged, ruleError, rulePayload } from './eventRules'

describe('event rule imports and per-run bindings', () => {
  it('assigns stable independent IDs to text entries and preserves exported definitions', () => {
    const rows = parseRuleImport('事件一\n事件二')
    expect(rows).toHaveLength(2)
    expect(rows[0].id).not.toBe(rows[1].id)
    const exported = JSON.stringify({ rules: rows.map(rulePayload) })
    const imported = parseRuleImport(exported)
    expect(imported.map(rulePayload)).toEqual(rows.map(rulePayload))
    expect(ruleChanged({ ...rows[0], name: '修改名称' }, rows[0])).toBe(true)
    expect(rulePayload({ ...rows[0], version: 3 })).not.toHaveProperty('version')
  })
  it('reads existing advanced JSON without losing criteria or accepting invalid types', () => {
    const raw = { id: 'legacy', description: '检查', criteria: ['条件'], uncertainWhen: ['遮挡'], preSeconds: 2, postSeconds: 8, poiIds: ['old-zone'] }
    const [row] = parseRuleImport(JSON.stringify({ events: [raw] }))
    expect(row.criteria).toEqual(['条件'])
    expect(row.poiIds).toEqual(['old-zone'])
    expect(rulePayload(row).poiIds).toEqual([])
    expect(() => parseRuleImport(JSON.stringify([{ ...raw, enabled: 'false' }]))).toThrow()
    expect(() => parseRuleImport(JSON.stringify([raw, raw]))).toThrow()
  })
  it('does not silently broaden missing POIs or incompatible trigger types', () => {
    const row = newRule('检查')
    const pois = [{ id: 'gate', kind: 'line', label: '门' }]
    expect(bindingError(row, [], pois)).toBe('')
    expect(bindingError(row, ['removed'], pois)).not.toBe('')
    expect(bindingError({ ...row, triggerKinds: ['zone.enter'] }, [], pois)).not.toBe('')
  })
  it('supports integrity-only rules bound to integrity zones', () => {
    const row = newRule('围墙完整性', 'integrity')
    const zones = [{ id: 'wall', kind: 'integrity', label: 'W1' }]
    expect(bindingError(row, [], [], zones)).toBe('')
    expect(bindingError(row, [], [], [])).not.toBe('')
    // An integrity trigger never matches a line POI by itself.
    expect(bindingError(row, [], [{ id: 'gate', kind: 'line', label: '门' }], [])).not.toBe('')
    expect(rulePayload(row).integrityIds).toEqual([])
    const imported = parseRuleImport(JSON.stringify({ rules: [{ ...rulePayload(row), integrityIds: ['wall'] }] }))
    expect(imported[0].integrityIds).toEqual(['wall'])
    expect(imported[0].triggerKinds).toEqual(['integrity.change'])
  })
  it('keeps the synthesized preset row clean until the backend seeds it', () => {
    const preset = presetRule()
    expect(preset.id).toBe(PRESET_RULE_ID)
    expect(ruleChanged(preset, undefined)).toBe(false)
    expect(ruleChanged(preset, { ...preset, version: 1 })).toBe(false)
    expect(ruleChanged({ ...preset, description: '改' }, { ...preset, version: 1 })).toBe(true)
    expect(rulePayload(preset).triggerKinds).toEqual(['integrity.change'])
  })
})

describe('rule categories and signal-channel pairing', () => {
  const line = { id: 'l1', label: 'L1', kind: 'line', pairIndex: 1, entryDirection: 'negative_to_positive' }
  const zone = { id: 'z1', label: 'Z1', kind: 'zone', pairIndex: 1 }
  const cue = { id: 'c1', label: 'C1', kind: 'cue', pairIndex: 1 }
  const forced = newRule('强行通过未退回', 'forced_passage')

  it('defaults a forced-passage rule to cue.on with a longer lookback', () => {
    expect(forced.category).toBe('forced_passage')
    expect(forced.triggerKinds).toEqual(['cue.on'])
    expect(forced.preSeconds).toBe(8)
    const imported = parseRuleImport(JSON.stringify({ rules: [rulePayload(forced)] }))
    expect(imported[0].category).toBe('forced_passage')
    expect(imported[0].triggerKinds).toEqual(['cue.on'])
  })

  it('requires a line, a signal ROI and matching channel numbers', () => {
    expect(bindingError(forced, ['l1', 'z1'], [line, zone], [], ['c1'], [cue])).toBe('')
    expect(bindingError(forced, ['l1'], [line], [], [], [])).not.toBe('')
    expect(bindingError(forced, ['l1'], [line], [], ['c1'], [{ ...cue, pairIndex: 2 }])).not.toBe('')
    expect(bindingError(forced, ['l1'], [{ ...line, entryDirection: null }], [], ['c1'], [cue])).not.toBe('')
    expect(bindingError(forced, ['l1'], [{ ...line, pairIndex: undefined }], [], ['c1'], [cue])).not.toBe('')
    expect(bindingError(forced, ['z1'], [zone], [], ['c1'], [cue])).not.toBe('')
    expect(bindingError(forced, ['l1', 'z1'], [line, { ...zone, pairIndex: 2 }], [], ['c1'], [cue])).not.toBe('')
  })

  it('keeps cue.on out of non-forced-passage rules', () => {
    const row = { ...newRule('计数'), triggerKinds: ['cue.on' as const] }
    expect(ruleError(row)).not.toBe('')
  })
})
