import { afterEach, expect, it, vi } from 'vitest'
import { apiCreateVideoPreviewFromVideo, apiDeleteVideoPreview, apiVideoPreviewStatus, apiVideoPreviewUrl } from '../api/client'
import { prepareVideoPreviewFromVideo } from './videoPreview'

vi.mock('../api/client', () => ({
  apiCreateVideoPreviewFromVideo: vi.fn(), apiDeleteVideoPreview: vi.fn(),
  apiVideoPreviewStatus: vi.fn(), apiVideoPreviewUrl: vi.fn(),
}))

afterEach(() => vi.resetAllMocks())

it('loads a converted preview and releases the server file', async () => {
  vi.mocked(apiCreateVideoPreviewFromVideo).mockResolvedValue({ id: 'id' })
  vi.mocked(apiVideoPreviewStatus).mockResolvedValue({ state: 'ready', progress: 100 })
  vi.mocked(apiVideoPreviewUrl).mockResolvedValue('/api/video/previews/id/source')
  vi.mocked(apiDeleteVideoPreview).mockResolvedValue(undefined)
  const progress = vi.fn()
  const preview = await prepareVideoPreviewFromVideo('vid', new AbortController().signal, progress)
  expect(apiCreateVideoPreviewFromVideo).toHaveBeenCalledWith('vid')
  expect(preview.url).toBe('/api/video/previews/id/source')
  expect(progress).toHaveBeenCalledWith(100)
  preview.release()
  expect(apiDeleteVideoPreview).toHaveBeenCalledWith('id')
})

it('cleans up a job if cancelled while starting', async () => {
  const controller = new AbortController()
  vi.mocked(apiCreateVideoPreviewFromVideo).mockImplementation(async () => {
    controller.abort()
    return { id: 'cancelled' }
  })
  vi.mocked(apiDeleteVideoPreview).mockResolvedValue(undefined)
  await expect(prepareVideoPreviewFromVideo('vid', controller.signal, vi.fn())).rejects.toThrow()
  expect(apiDeleteVideoPreview).toHaveBeenCalledWith('cancelled')
  expect(apiVideoPreviewStatus).not.toHaveBeenCalled()
})

it('surfaces conversion failure and removes the job', async () => {
  vi.mocked(apiCreateVideoPreviewFromVideo).mockResolvedValue({ id: 'bad' })
  vi.mocked(apiVideoPreviewStatus).mockResolvedValue({ state: 'failed', progress: 0, error: 'No video stream' })
  vi.mocked(apiDeleteVideoPreview).mockResolvedValue(undefined)
  await expect(prepareVideoPreviewFromVideo('vid', new AbortController().signal, vi.fn())).rejects.toThrow('No video stream')
  expect(apiDeleteVideoPreview).toHaveBeenCalledWith('bad')
})
