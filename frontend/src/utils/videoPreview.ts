import {
  apiCreateVideoPreviewFromVideo,
  apiDeleteVideoPreview,
  apiVideoPreviewStatus,
  apiVideoPreviewUrl,
} from '../api/client'

async function pollPreviewJob(id: string, signal: AbortSignal, progress: (value: number) => void) {
  let url = ''
  const release = () => {
    if (url.startsWith('blob:')) URL.revokeObjectURL(url)
    void apiDeleteVideoPreview(id).catch(() => {})
  }
  try {
    for (;;) {
      signal.throwIfAborted()
      const status = await apiVideoPreviewStatus(id, signal)
      if (status.state === 'failed' || status.state === 'cancelled') throw new Error(status.error || status.state)
      progress(status.progress)
      if (status.state === 'ready') break
      await new Promise<void>((resolve, reject) => {
        const onAbort = () => {
          clearTimeout(timer)
          reject(signal.reason)
        }
        const timer = setTimeout(() => {
          signal.removeEventListener('abort', onAbort)
          resolve()
        }, 1000)
        signal.addEventListener('abort', onAbort, { once: true })
        if (signal.aborted) onAbort()
      })
    }
    url = await apiVideoPreviewUrl(id, signal)
    signal.throwIfAborted()
    return { url, release }
  } catch (error) {
    release()
    throw error
  }
}

/** Browser fallback for server-imported videos: the bytes are already on the backend. */
export async function prepareVideoPreviewFromVideo(videoId: string, signal: AbortSignal, progress: (value: number) => void) {
  const { id } = await apiCreateVideoPreviewFromVideo(videoId)
  return pollPreviewJob(id, signal, progress)
}
