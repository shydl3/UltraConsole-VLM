import { describe, expect, it } from 'vitest'
import { formatDuration } from './library'

const t = (key: string, params?: Record<string, string | number>) => {
  if (key === 'library.durationHms') return `${params?.h}小时${params?.m}分${params?.s}秒`
  if (key === 'library.durationMs') return `${params?.m}分${params?.s}秒`
  if (key === 'library.durationS') return `${params?.s}秒`
  return key
}

describe('formatDuration', () => {
  it('renders event times as hours/minutes/seconds instead of raw seconds', () => {
    expect(formatDuration(0, t)).toBe('0秒')
    expect(formatDuration(65, t)).toBe('1分5秒')
    expect(formatDuration(3661, t)).toBe('1小时1分1秒')
  })

  it('rounds to the nearest second and handles invalid values', () => {
    expect(formatDuration(59.6, t)).toBe('1分0秒')
    expect(formatDuration(null, t)).toBe('—')
    expect(formatDuration(Number.NaN, t)).toBe('—')
  })
})
