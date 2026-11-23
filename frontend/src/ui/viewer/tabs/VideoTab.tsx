import { useEffect, useMemo, useRef, useState } from 'react'
import { EventEditOverlay, type EventEditOverlayHandle } from '../widgets/EventEditOverlay'
import { newTrackerState } from '../../../utils/events'
import { createVideoReport, formatVideoTime, recordVideoFrame, seekVideo, VideoInferenceSession, videoReportCsv, type VideoReport } from '../../../utils/videoAnalysis'
import { useT } from '../../../i18n'
import { useConsoleStore } from '../../../store/useConsoleStore'
import { filterBBoxesByRoiNormalized } from '../../../utils/roi'
import { NeoButton } from '../../primitives/NeoButton'
import { TelemetryHUD } from '../widgets/TelemetryHUD'
import { VideoCanvasOverlay } from '../widgets/VideoCanvasOverlay'
import { RoiOverlay } from '../widgets/RoiOverlay'
import { ServerVideoDialog } from '../widgets/ServerVideoDialog'
import styles from './VideoTab.module.css'
import { VlmPanel } from '../../vlm/VlmPanel'
import { prepareVideoPreviewFromVideo } from '../../../utils/videoPreview'
import { apiVlmMedia, apiVlmThumbnail, type ServerVideo } from '../../../api/client'

export function VideoTab() {
  const t = useT()
  const params = useConsoleStore((s) => s.params)
  const osd = useConsoleStore((s) => s.osd)
  const setOSD = useConsoleStore((s) => s.setOSD)
  const setLastPred = useConsoleStore((s) => s.setLastPred)
  const lastPred = useConsoleStore((s) => s.lastPred)
  const roi = useConsoleStore((s) => s.roi)
  const setRoi = useConsoleStore((s) => s.setRoi)
  const pushLog = useConsoleStore((s) => s.pushLog)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const stillRef = useRef<HTMLImageElement | null>(null)
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const pickedVideoRef = useRef<ServerVideo | null>(null)
  const previewAbortRef = useRef<AbortController | null>(null)
  const previewReleaseRef = useRef<(() => void) | null>(null)
  const stillRequestRef = useRef('')
  const usingPreviewRef = useRef(false)
  const [stillUrl, setStillUrl] = useState('')
  const [stillReady, setStillReady] = useState(false)
  const [useStill, setUseStill] = useState(false)
  const [needsPreview, setNeedsPreview] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const [previewProgress, setPreviewProgress] = useState<number | null>(null)
  const reportRef = useRef<VideoReport | null>(null)
  const [report, setReport] = useState<VideoReport | null>(null)
  const [fileName, setFileName] = useState('')
  const [selectedVideo, setSelectedVideo] = useState<ServerVideo | null>(null)
  const [browserOpen, setBrowserOpen] = useState(false)
  const [semanticBusy, setSemanticBusy] = useState(false)
  const [running, setRunning] = useState(false)
  const [videoLoaded, setVideoLoaded] = useState(false)
  const [targetFps, setTargetFps] = useState(8)
  const [editTool, setEditTool] = useState<'none' | 'line' | 'zone' | 'cue' | 'integrity'>('none')
  const [roiMenuOpen, setRoiMenuOpen] = useState(false)
  const roiMenuRef = useRef<HTMLDivElement | null>(null)
  const editOverlayRef = useRef<EventEditOverlayHandle | null>(null)
  const [error, setError] = useState('')
  const modelId = useConsoleStore((s) => s.currentModelId)
  const eventsConfig = useConsoleStore((s) => s.eventsConfig)
  const lastLine = eventsConfig.lines[eventsConfig.lines.length - 1]

  const filteredPred = useMemo(() => {
    if (!lastPred) return undefined
    const classFiltered =
      params.classFilter.length > 0 ? lastPred.bboxes.filter((b) => params.classFilter.includes(b.cls)) : lastPred.bboxes

    const roiFiltered =
      roi.applyFilter && roi.polygon.length >= 3
        ? filterBBoxesByRoiNormalized({
            bboxes: classFiltered,
            roiPoly: roi.polygon,
            width: lastPred.width,
            height: lastPred.height,
          })
        : classFiltered

    return { ...lastPred, bboxes: roiFiltered }
  }, [lastPred, params.classFilter, roi.applyFilter, roi.polygon])
  const bboxes = useMemo(() => filteredPred?.bboxes ?? [], [filteredPred])
  // Drawing overlays anchor to the still until the explicit preview is playable.
  const overlayKey = `${selectedVideo?.videoId ?? 'none'}-${useStill ? `still-${stillReady}` : 'video'}`
  const anchorRef = useStill ? stillRef : videoRef
  // Drawing is valid on a loaded first frame even before a playable preview exists.
  const stageReady = videoLoaded || (useStill && stillReady)

  // ROI 子菜单：点击外部或按 Esc 关闭；分析中自动收起
  const roiMenuVisible = roiMenuOpen && !running
  useEffect(() => {
    if (!roiMenuVisible) return
    function onPointerDown(e: PointerEvent) {
      if (roiMenuRef.current && !roiMenuRef.current.contains(e.target as Node)) setRoiMenuOpen(false)
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setRoiMenuOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [roiMenuVisible])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      previewAbortRef.current?.abort()
      previewReleaseRef.current?.()
      useConsoleStore.getState().setLastPred(undefined)
      useConsoleStore.getState().resetCounters()
    }
  }, [])

  async function analyze() {
    const video = videoRef.current
    if (!video || !modelId || abortRef.current || previewAbortRef.current || !Number.isFinite(video.duration) || video.duration <= 0) return
    // HEVC 等编码可能只加载出时长而解不出画面（videoWidth 为 0）：先走兼容预览回退。
    if (!video.videoWidth || !video.videoHeight) {
      void onDecodeError()
      return
    }
    const controller = new AbortController()
    abortRef.current = controller
    video.pause()
    setRoi({ enabled: false })
    setEditTool('none')
    setRoiMenuOpen(false)
    setError('')
    setRunning(true)
    setLastPred(undefined)
    useConsoleStore.getState().resetCounters()
    const snapshot = useConsoleStore.getState()
    const analysis = createVideoReport(fileName, video.duration, targetFps, modelId, { ...snapshot.params, track: true }, snapshot.eventsConfig)
    reportRef.current = analysis
    setReport({ ...analysis })
    const tracker = newTrackerState()
    let session: VideoInferenceSession | undefined
    // Changing model, engine, parameters or rules invalidates a run rather than mixing results.
    const unsubscribe = useConsoleStore.subscribe((next) => {
      if (next.currentModelId !== snapshot.currentModelId || next.engine.device !== snapshot.engine.device ||
          next.params !== snapshot.params || next.eventsConfig !== snapshot.eventsConfig) controller.abort()
    })
    try {
      session = new VideoInferenceSession(controller.signal)
      const canvas = (captureCanvasRef.current ??= document.createElement('canvas'))
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('Canvas is unavailable')
      const total = Math.ceil(analysis.duration * analysis.fps)
      for (let index = 0; index < total; index++) {
        const mediaTime = index / analysis.fps
        await seekVideo(video, mediaTime, controller.signal)
        if (controller.signal.aborted) break
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        const jpeg = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
        // 0×0 画布会得到 "data:,"（空 payload），后端会拒收并断开整个会话。
        if (!jpeg) throw new Error(t('video.decodeError'))
        const pred = await session.infer(jpeg, analysis.params)
        if (controller.signal.aborted) break
        recordVideoFrame(analysis, tracker, pred, mediaTime)
        // Video events are processed synchronously here, independently of React render frequency.
        useConsoleStore.setState({ counters: analysis.counters, events: analysis.events.slice(-500) })
        setLastPred({ ...pred, source: 'video', mediaTime })
        setReport({ ...analysis })
      }
      analysis.status = controller.signal.aborted ? 'stopped' : 'completed'
    } catch (e) {
      analysis.status = controller.signal.aborted ? 'stopped' : 'error'
      if (!controller.signal.aborted) {
        setError(String(e))
        pushLog({ ts: Date.now() / 1000, level: 'ERROR', event: 'video.analysis.failed', msg: String(e), fields: {} })
      }
    } finally {
      unsubscribe()
      session?.close()
      if (abortRef.current === controller) {
        abortRef.current = null
        setRunning(false)
        setReport({ ...analysis })
      }
    }
  }

  async function onPick(video: ServerVideo) {
    if (abortRef.current || semanticBusy) return
    const element = videoRef.current
    if (!element) return
    previewAbortRef.current?.abort()
    previewAbortRef.current = null
    previewReleaseRef.current?.()
    previewReleaseRef.current = null
    pickedVideoRef.current = video
    stillRequestRef.current = ''
    usingPreviewRef.current = false
    setStillUrl('')
    setStillReady(false)
    setUseStill(false)
    setNeedsPreview(false)
    setPreparing(false)
    setPreviewProgress(null)
    setVideoLoaded(false)
    setError('')
    setLastPred(undefined)
    useConsoleStore.getState().resetCounters()
    setReport(null)
    reportRef.current = null
    setEditTool('none')
    setFileName(video.name)
    setSelectedVideo(video)
    try {
      // Server-side import + cookie-authenticated Range streaming: no upload,
      // no multi-gigabyte blob in the browser.
      const url = await apiVlmMedia(`/videos/${video.videoId}/source`)
      if (pickedVideoRef.current !== video || !videoRef.current) return
      videoRef.current.src = url
      videoRef.current.load()
      pushLog({ ts: Date.now() / 1000, level: 'INFO', event: 'video.loaded', msg: video.name, fields: {} })
    } catch (e) {
      if (pickedVideoRef.current === video) setError(String(e))
    }
  }

  /** Browser cannot decode the source: show its first frame and let the user draw. */
  async function onDecodeError() {
    if (previewAbortRef.current) return
    setVideoLoaded(false)
    abortRef.current?.abort()
    const video = pickedVideoRef.current
    if (!video) {
      setError(t('video.decodeError'))
      return
    }
    setUseStill(true)
    setNeedsPreview(true)
    setStillReady(false)
    setError('')
    if (stillRequestRef.current === video.videoId) return
    stillRequestRef.current = video.videoId
    try {
      const url = await apiVlmThumbnail(video.videoId)
      if (pickedVideoRef.current === video) setStillUrl(url)
    } catch (e) {
      if (pickedVideoRef.current === video) setError(`${t('video.decodeError')} ${String(e)}`)
    }
  }

  /** Explicit user action: build the playable preview (keeps the still visible meanwhile). */
  async function generatePreview() {
    const video = pickedVideoRef.current
    if (!video || previewAbortRef.current) return
    const controller = new AbortController()
    previewAbortRef.current = controller
    setPreparing(true)
    setPreviewProgress(null)
    setError('')
    try {
      const preview = await prepareVideoPreviewFromVideo(video.videoId, controller.signal, setPreviewProgress)
      if (controller.signal.aborted || pickedVideoRef.current !== video || !videoRef.current) {
        preview.release()
        return
      }
      previewReleaseRef.current = preview.release
      usingPreviewRef.current = true
      videoRef.current.src = preview.url
      videoRef.current.load()
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(`${t('video.prepareError')} ${String(error)}`)
        pushLog({ ts: Date.now() / 1000, level: 'ERROR', event: 'video.preview.failed', msg: String(error), fields: {} })
      }
    } finally {
      if (previewAbortRef.current === controller) {
        previewAbortRef.current = null
        setPreparing(false)
      }
    }
  }

  function exportReport() {
    const current = reportRef.current
    if (!current) return
    const url = URL.createObjectURL(new Blob([videoReportCsv(current)], { type: 'text/csv;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${current.fileName}.analysis.csv`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  function edit(mode: 'line' | 'zone' | 'cue' | 'integrity') {
    videoRef.current?.pause()
    setRoi({ enabled: false })
    if (editTool === mode) {
      // 再次点击同一工具 = 完成绘制：提交未闭合草稿后退出绘制模式。
      editOverlayRef.current?.finish()
      setEditTool('none')
    } else {
      setEditTool(mode)
    }
  }

  function flipEntryDirection() {
    const lines = eventsConfig.lines
    if (!lines.length) return
    const last = lines[lines.length - 1]
    const next = last.entryDirection === 'negative_to_positive' ? 'positive_to_negative'
      : last.entryDirection === 'positive_to_negative' ? null
        : 'negative_to_positive'
    useConsoleStore.getState().setEventsConfig({ lines: lines.map((l, i) => i === lines.length - 1 ? { ...l, entryDirection: next } : l) })
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.toolbar}>
        <NeoButton disabled={running || semanticBusy} onClick={() => setBrowserOpen(true)}>
          {fileName ? `${t('video.selectVideo')} · ${fileName}` : t('video.selectVideo')}
        </NeoButton>

        {!running ? (
          <NeoButton onClick={() => void analyze()} disabled={!videoLoaded || !modelId || semanticBusy || preparing}>
            {t('video.analyze')}
          </NeoButton>
        ) : (
          <NeoButton tone="danger" onClick={() => abortRef.current?.abort()}>{t('common.stop')}</NeoButton>
        )}
        <NeoButton onClick={exportReport} disabled={running || !report || report.frames === 0}>{t('video.exportReport')}</NeoButton>
        <NeoButton onClick={() => edit('line')} disabled={running || !stageReady} className={editTool === 'line' ? styles.toolLine : undefined}>
          {t(editTool === 'line' ? 'video.finishEdit' : 'video.drawLine')}
        </NeoButton>
        <NeoButton onClick={flipEntryDirection} disabled={running || !eventsConfig.lines.length}>
          {t('video.entryDirection')}{lastLine?.entryDirection ? (lastLine.entryDirection === 'negative_to_positive' ? ' →' : ' ←') : ` (${t('video.entryBoth')})`}
        </NeoButton>
        <NeoButton onClick={() => edit('zone')} disabled={running || !stageReady} className={editTool === 'zone' ? styles.toolZone : undefined}>
          {t(editTool === 'zone' ? 'video.finishEdit' : 'video.drawZone')}
        </NeoButton>
        <NeoButton onClick={() => edit('cue')} disabled={running || !stageReady} className={editTool === 'cue' ? styles.toolCue : undefined}>
          {t(editTool === 'cue' ? 'video.finishEdit' : 'video.drawCue')}
        </NeoButton>
        <NeoButton onClick={() => edit('integrity')} disabled={running || !stageReady} className={editTool === 'integrity' ? styles.toolIntegrity : undefined}>
          {t(editTool === 'integrity' ? 'video.finishEdit' : 'video.drawIntegrity')}
        </NeoButton>
        <NeoButton disabled={running || (!eventsConfig.lines.length && !eventsConfig.zones.length && !(eventsConfig.cues?.length) && !(eventsConfig.integrity?.length))} onClick={() => {
          useConsoleStore.getState().setEventsConfig({ lines: [], zones: [], cues: [], integrity: [] })
          setEditTool('none')
        }}>{t('video.clearRules')}</NeoButton>

        <div className={styles.divider} />

        <label className={styles.toggle}>
          FPS
          <input
            className={styles.num}
            type="number"
            min={1}
            max={30}
            value={targetFps}
            disabled={running}
            onChange={(e) => setTargetFps(Math.max(1, Math.min(30, Number(e.target.value) || 8)))}
          />
        </label>

        <div className={styles.divider} />

        <div className={styles.roiMenu} ref={roiMenuRef}>
          <NeoButton
            onClick={() => setRoiMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={roiMenuVisible}
            className={roi.enabled ? styles.roiActive : undefined}
          >
            {t('common.editRoi')} {roiMenuVisible ? '▴' : '▾'}
          </NeoButton>
          {roiMenuVisible && (
            <div className={styles.roiPop} role="menu" aria-label={t('common.editRoi')}>
              {!stageReady && <div className={styles.roiHint}>{t('common.roiNeedVideo')}</div>}
              <NeoButton role="menuitem" className={roi.enabled ? styles.roiActive : undefined} disabled={!stageReady || running} onClick={() => { setEditTool('none'); setRoi({ enabled: !roi.enabled }) }}>
                ROI{roi.enabled ? ' · on' : ''}
              </NeoButton>
              <NeoButton role="menuitem" onClick={() => setRoi({ mode: roi.mode === 'rect' ? 'poly' : 'rect' })} disabled={!roi.enabled}>
                {roi.mode === 'rect' ? t('common.rectangle') : t('common.polygon')}
              </NeoButton>
              <NeoButton role="menuitem" onClick={() => setRoi({ polygon: [], closed: false })} disabled={!roi.enabled || roi.polygon.length === 0}>
                {t('common.clearRoi')}
              </NeoButton>
              <NeoButton role="menuitem" onClick={() => setRoi({ closed: false })} disabled={!roi.enabled || !roi.closed}>
                {t('common.continueEdit')}
              </NeoButton>
            </div>
          )}
        </div>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={roi.applyFilter}
            onChange={(e) => setRoi({ applyFilter: e.target.checked })}
            disabled={!roi.enabled}
          />
          {t('common.showOnlyInsideRoi')}
        </label>

        <label className={styles.toggle}>
          <input type="checkbox" checked={osd.bbox} onChange={() => setOSD({ bbox: !osd.bbox })} />
          BBox
        </label>
        <label className={styles.toggle}>
          <input type="checkbox" checked={osd.labels} onChange={() => setOSD({ labels: !osd.labels })} />
          Labels
        </label>
        <div className={styles.spacer} />
        <div className={styles.meta}>bbox={bboxes.length}</div>
      </div>

      <div className={styles.meta}>{t('video.analysisHint')}</div>
      {editTool !== 'none' && <div className={styles.meta}>{t(editTool === 'line' ? 'video.lineHint' : editTool === 'zone' ? 'video.zoneHint' : editTool === 'cue' ? 'video.cueHint' : 'video.integrityHint')}</div>}
      {report && <div role="status" className={styles.meta}>
        {t(`video.${report.status}`)} · {formatVideoTime(report.analyzedSeconds)} / {formatVideoTime(report.duration)} · {report.frames} {t('video.frames')} · {report.events.length} {t('video.events')}
      </div>}
      {error && <div role="alert">{error}</div>}
      {needsPreview && <div className={styles.previewBar}>
        <span>{t('video.previewNeeded')}</span>
        <NeoButton onClick={() => void generatePreview()} disabled={preparing}>{t('video.generatePreview')}</NeoButton>
      </div>}
      {preparing && <div role="status" className={styles.meta}>
        {previewProgress === null ? t('video.preparingUpload') : `${t('video.preparingConvert')} ${previewProgress}%`}
        {' '}<NeoButton onClick={() => {
          previewAbortRef.current?.abort()
          previewAbortRef.current = null
          setPreparing(false)
          setError(t('video.prepareCancelled'))
        }}>{t('common.stop')}</NeoButton>
      </div>}
      <div className={styles.stage}>
        <video ref={videoRef} className={styles.video} controls={!running && !useStill} playsInline
          onLoadedData={() => {
            const v = videoRef.current
            const hasDuration = Number.isFinite(v?.duration) && (v?.duration ?? 0) > 0
            const decodable = (v?.videoWidth ?? 0) > 0 && (v?.videoHeight ?? 0) > 0
            setVideoLoaded(hasDuration && decodable)
            // 浏览器读得出时长却解不出画面（如 Linux 下无 HEVC 硬解）时不会触发 error 事件，
            // 这里主动切到首帧绘制模式，等用户点击“生成预览”再做转换。
            if (hasDuration && !decodable) { void onDecodeError(); return }
            if (decodable && usingPreviewRef.current) {
              usingPreviewRef.current = false
              setUseStill(false)
              setNeedsPreview(false)
              setStillReady(false)
            }
          }}
          onError={() => void onDecodeError()}
          onPlay={() => { if (abortRef.current) videoRef.current?.pause(); else setLastPred(undefined) }}
          onSeeking={() => { if (!abortRef.current) setLastPred(undefined) }}
        />
        {useStill && stillUrl && <img ref={stillRef} className={styles.still} src={stillUrl}
          alt={t('video.firstFrameAlt')}
          onLoad={() => {
            setStillReady(true)
            // The overlays observe resize; nudge them once the natural size is known.
            window.dispatchEvent(new Event('resize'))
          }}
          onError={() => setError(t('video.decodeError'))}
        />}
        {useStill && !stillUrl && <div className={styles.stillHint}>{t('video.loadingFirstFrame')}</div>}
        {!useStill && <VideoCanvasOverlay videoRef={videoRef} pred={filteredPred} showBBox={osd.bbox} showLabels={osd.labels} />}
        <RoiOverlay key={`roi-${overlayKey}`} anchorRef={anchorRef} />
        <EventEditOverlay ref={editOverlayRef} key={`edit-${editTool}-${overlayKey}`} anchorRef={anchorRef} enabled={editTool !== 'none'} mode={editTool === 'none' ? 'line' : editTool} />
        <TelemetryHUD telemetry={filteredPred?.telemetry} />
      </div>
      <VlmPanel video={selectedVideo} eventsConfig={eventsConfig} basicRunning={running || preparing} onBusyChange={setSemanticBusy}
        onSelectVideo={next => void onPick(next)} />
      {browserOpen && <ServerVideoDialog multiple={false} onClose={() => setBrowserOpen(false)} onPicked={videos => {
        setBrowserOpen(false)
        if (videos[0]) void onPick(videos[0])
      }} />}
    </div>
  )
}

