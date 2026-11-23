import { useCallback, useEffect, useRef, useState } from 'react'
import { apiVlmBrowse, apiVlmImportFolder, apiVlmImportVideo, type ServerBrowseListing, type ServerVideo } from '../../../api/client'
import { useT } from '../../../i18n'
import styles from './ServerVideoDialog.module.css'

const formatSize = (bytes: number) => bytes >= 1024 ** 3
  ? `${(bytes / 1024 ** 3).toFixed(2)} GiB`
  : bytes >= 1024 ** 2
    ? `${(bytes / 1024 ** 2).toFixed(1)} MiB`
    : `${Math.max(1, Math.round(bytes / 1024))} KiB`

type Picked = { name: string; size: number }

/**
 * Server-side video browser replacing the local file input: remote operators
 * pick recordings that already live on this machine; confirming imports them
 * by reference (hard link / reuse), so no bytes travel through the browser.
 *
 * In folder mode it imports every video directly inside the current folder in
 * one transaction. Batch recordings from one camera share the same geometry,
 * so the operator draws regions once and starts the whole batch.
 */
export function ServerVideoDialog({ multiple, folderImport = false, onClose, onPicked }: {
  multiple: boolean; folderImport?: boolean; onClose: () => void; onPicked: (videos: ServerVideo[]) => void
}) {
  const t = useT()
  const ref = useRef<HTMLDialogElement>(null)
  const [listing, setListing] = useState<ServerBrowseListing | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<Record<string, Picked>>({})
  const [importing, setImporting] = useState<{ current: number; total: number } | null>(null)
  const [folderBusy, setFolderBusy] = useState(false)
  const [folderErrors, setFolderErrors] = useState<{ name: string; error: string }[]>([])
  const [pendingFolder, setPendingFolder] = useState<ServerVideo[] | null>(null)
  const [folderNote, setFolderNote] = useState('')
  const busy = !!importing || folderBusy

  useEffect(() => { ref.current?.showModal() }, [])

  // Runs one listing request; the initial mount fetch only settles asynchronously
  // (loading starts true), while navigation also resets the flags from a handler.
  const fetchInto = useCallback((path: string) => {
    const controller = new AbortController()
    void apiVlmBrowse(path, controller.signal)
      .then(next => { if (!controller.signal.aborted) setListing(next) })
      .catch(e => { if (!controller.signal.aborted) setError(String(e)) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [])

  useEffect(() => fetchInto(''), [fetchInto])

  function load(path: string) {
    setLoading(true)
    setError('')
    setFolderErrors([])
    setPendingFolder(null)
    setFolderNote('')
    void fetchInto(path)
  }

  function toggle(path: string, picked: Picked) {
    setSelected(previous => {
      if (previous[path]) {
        const next = { ...previous }
        delete next[path]
        return next
      }
      // Single selection replaces instead of accumulating the previous choice.
      return multiple ? { ...previous, [path]: picked } : { [path]: picked }
    })
  }

  async function confirm() {
    const paths = Object.keys(selected)
    if (!paths.length || busy) return
    setError('')
    setImporting({ current: 0, total: paths.length })
    const picked: ServerVideo[] = []
    try {
      for (const [index, path] of paths.entries()) {
        setImporting({ current: index + 1, total: paths.length })
        picked.push(await apiVlmImportVideo(path))
      }
      onPicked(picked)
    } catch (e) {
      // Registered items stay valid server-side; retrying reuses them without a second copy.
      setError(String(e))
      if (picked.length) onPicked(picked)
    } finally {
      setImporting(null)
    }
  }

  async function importFolder() {
    if (!listing || busy) return
    setFolderBusy(true)
    setError('')
    setFolderErrors([])
    setPendingFolder(null)
    setFolderNote('')
    try {
      const result = await apiVlmImportFolder(listing.path)
      if (result.videos.length && !result.errors.length && !result.truncated) {
        onPicked(result.videos)
        return
      }
      setPendingFolder(result.videos.length ? result.videos : null)
      setFolderErrors(result.errors)
      setFolderNote(result.videos.length
        ? t('serverBrowser.folderPartial', { count: result.videos.length })
        : t('serverBrowser.folderEmpty'))
      if (result.truncated) setFolderNote(t('serverBrowser.folderTruncated', { count: result.videos.length }))
    } catch (e) {
      setError(String(e))
    } finally {
      setFolderBusy(false)
    }
  }

  const selectedCount = Object.keys(selected).length
  const videoCount = listing?.files.length ?? 0
  return <dialog ref={ref} className={styles.dialog} aria-label={t(folderImport ? 'serverBrowser.titleFolder' : multiple ? 'serverBrowser.titleMultiple' : 'serverBrowser.title')}
    onCancel={e => { e.preventDefault(); if (!busy) onClose() }}>
    <h2>{t(folderImport ? 'serverBrowser.titleFolder' : multiple ? 'serverBrowser.titleMultiple' : 'serverBrowser.title')}</h2>
    <div className={styles.hint}>{folderImport
      ? t('serverBrowser.folderHint')
      : t('serverBrowser.hint', { root: listing?.root ?? '' })}</div>
    <div className={styles.path}>
      <button type="button" disabled={!listing?.parent || busy}
        onClick={() => { if (listing?.parent) load(listing.parent) }}>{t('serverBrowser.parent')}</button>
      <div className={styles.current} title={listing?.path}>{listing?.path ?? '…'}</div>
      <button type="button" disabled={loading || busy}
        onClick={() => { if (listing) load(listing.path) }}>{t('serverBrowser.refresh')}</button>
    </div>
    <ul className={styles.list}>
      {loading && <li className={styles.empty}>{t('serverBrowser.loading')}</li>}
      {!loading && error && !listing && <li className={styles.empty}>{error}</li>}
      {!loading && listing && listing.dirs.map(dir => <li key={dir.path} className={`${styles.row} ${styles.dirRow}`}
        onClick={() => { if (!busy) load(dir.path) }}>
        <span>📁</span><span className={styles.name} title={dir.name}>{dir.name}</span>
      </li>)}
      {!loading && listing && listing.files.map(file => <li key={file.path}
        className={[styles.row, !folderImport && selected[file.path] ? styles.selected : ''].join(' ')}
        onClick={() => { if (!folderImport && !busy) toggle(file.path, { name: file.name, size: file.size }) }}>
        {folderImport
          ? <span aria-hidden="true">🎞️</span>
          : <input type="checkbox" readOnly checked={!!selected[file.path]} />}
        <span className={styles.name} title={file.name}>{file.name}</span>
        <span className={styles.meta}>{formatSize(file.size)}</span>
        <span className={styles.meta}>{new Date(file.mtime * 1000).toLocaleString()}</span>
      </li>)}
      {!loading && listing && !listing.dirs.length && !listing.files.length &&
        <li className={styles.empty}>{t('serverBrowser.empty')}</li>}
      {listing?.truncated && <li className={styles.empty}>{t('serverBrowser.truncated')}</li>}
    </ul>
    {error && listing && <div role="alert" className={styles.error}>{error}</div>}
    {folderImport && folderNote && <div className={styles.hint}>{folderNote}</div>}
    {folderImport && folderErrors.length > 0 && <div role="alert" className={styles.error}>
      {t('serverBrowser.folderErrors', { count: folderErrors.length })}
      <ul className={styles.errorList}>
        {folderErrors.slice(0, 8).map(item => <li key={item.name}>{item.name}: {item.error}</li>)}
        {folderErrors.length > 8 && <li>…</li>}
      </ul>
    </div>}
    <div className={styles.actions}>
      <span className={styles.count}>{folderImport
        ? t('serverBrowser.folderCount', { count: videoCount })
        : t('serverBrowser.selected', { count: selectedCount })}</span>
      <button type="button" disabled={busy} onClick={onClose}>{t('serverBrowser.cancel')}</button>
      {folderImport ? <>
        {pendingFolder && <button type="button" disabled={folderBusy} onClick={() => onPicked(pendingFolder)}>
          {t('serverBrowser.addImported', { count: pendingFolder.length })}
        </button>}
        <button type="button" className={styles.primary} disabled={!videoCount || busy} onClick={() => void importFolder()}>
          {folderBusy ? t('serverBrowser.folderImporting') : t('serverBrowser.importFolder', { count: videoCount })}
        </button>
      </> : <button type="button" disabled={!selectedCount || busy} onClick={() => void confirm()}>
        {importing ? t('serverBrowser.importing', { current: importing.current, total: importing.total })
          : t('serverBrowser.confirm', { count: selectedCount })}
      </button>}
    </div>
  </dialog>
}
