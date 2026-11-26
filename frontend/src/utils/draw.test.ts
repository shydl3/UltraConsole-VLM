import { describe, expect, it } from 'vitest'
import { magnifierGeometry } from './draw'

describe('magnifierGeometry', () => {
  it('maps the lens centre to source pixels and preserves the zoom factor', () => {
    const geometry = magnifierGeometry({ x: 480, y: 270 }, 960, 540, 1920, 1080, 3, 72)
    expect(geometry).not.toBeNull()
    expect(geometry!.sourceX).toBeCloseTo(960)
    expect(geometry!.sourceY).toBeCloseTo(540)
    expect(geometry!.sourceHalf).toBeCloseTo(48)
    expect(geometry!.displayHalf).toBeCloseTo(24)
  })

  it('accounts for letterbox offsets', () => {
    const geometry = magnifierGeometry({ x: 500, y: 500 }, 1000, 1000, 1920, 1080, 3, 72)
    expect(geometry).not.toBeNull()
    expect(geometry!.sourceX).toBeCloseTo(960)
    expect(geometry!.sourceY).toBeCloseTo(540)
  })

  it('returns null outside the drawable source area', () => {
    expect(magnifierGeometry({ x: 10, y: 10 }, 1000, 1000, 1920, 1080)).toBeNull()
    expect(magnifierGeometry({ x: -1, y: 500 }, 1000, 1000, 1920, 1080)).toBeNull()
    expect(magnifierGeometry({ x: 500, y: 500 }, 0, 1000, 1920, 1080)).toBeNull()
  })
})
