import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

const ACCEPTED = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

function isVideoFile(file) {
  if (!file) return false
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  return ACCEPTED.includes(ext) || file.type.startsWith('video/')
}

export default function UploadPage() {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [confidence, setConfidence] = useState(0.3)
  const [cleanup, setCleanup] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const pickFile = (f) => {
    if (f && isVideoFile(f)) {
      setFile(f)
      setError(null)
    } else if (f) {
      setError('Please select a video file (.mp4, .avi, .mov, .mkv, .webm).')
    }
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    pickFile(e.dataTransfer.files[0])
  }, [])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const onInputChange = (e) => pickFile(e.target.files[0])

  const onSubmit = async (e) => {
    e.preventDefault()
    if (!file) { setError('Please select a video file first.'); return }

    setSubmitting(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('video', file, file.name)
      form.append('confidence_threshold', String(confidence))
      form.append('cleanup', cleanup ? 'true' : 'false')

      const resp = await fetch('/jobs', { method: 'POST', body: form })
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`Server error ${resp.status}: ${text}`)
      }
      const { job_id } = await resp.json()
      navigate(`/jobs/${job_id}/progress`)
    } catch (err) {
      setError(err.message)
      setSubmitting(false)
    }
  }

  return (
    <div className="page">
      <h1>Cattle Vision — Behavior Inference</h1>

      <form onSubmit={onSubmit}>
        {/* Drop zone */}
        <div
          className={`dropzone${dragging ? ' active' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
          aria-label="Drop video file here or click to browse"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED.join(',')}
            style={{ display: 'none' }}
            onChange={onInputChange}
          />
          {file ? (
            <>
              <p>Selected video:</p>
              <p className="filename">{file.name}</p>
              <p>({(file.size / 1e6).toFixed(1)} MB)</p>
            </>
          ) : (
            <>
              <p>Drag & drop a video file here</p>
              <p>or click to browse</p>
              <p style={{ fontSize: '0.78rem', color: '#888', marginTop: '0.5rem' }}>
                {ACCEPTED.join('  ')}
              </p>
            </>
          )}
        </div>

        {/* Config */}
        <div className="config-form">
          <h2>Options</h2>
          <label>
            Confidence threshold
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={cleanup}
              onChange={(e) => setCleanup(e.target.checked)}
            />
            Delete intermediate files after job completes
          </label>
        </div>

        {error && <div className="error-box">{error}</div>}

        <button
          type="submit"
          className="btn-primary"
          disabled={submitting || !file}
        >
          {submitting ? 'Uploading…' : 'Run inference'}
        </button>
      </form>
    </div>
  )
}
