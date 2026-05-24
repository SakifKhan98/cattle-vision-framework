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
    <div className="max-w-xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold mb-6">Cattle Vision — Behavior Inference</h1>

      <form onSubmit={onSubmit}>
        {/* Drop zone */}
        <div
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors mb-5
            ${dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white hover:border-gray-400'}`}
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
            className="hidden"
            onChange={onInputChange}
          />
          {file ? (
            <>
              <p className="text-sm text-gray-500">Selected video:</p>
              <p className="font-semibold mt-1">{file.name}</p>
              <p className="text-sm text-gray-400 mt-1">({(file.size / 1e6).toFixed(1)} MB)</p>
            </>
          ) : (
            <>
              <p className="text-gray-600">Drag & drop a video file here</p>
              <p className="text-gray-400 text-sm mt-1">or click to browse</p>
              <p className="text-gray-300 text-xs mt-3">{ACCEPTED.join('  ')}</p>
            </>
          )}
        </div>

        {/* Config */}
        <div className="bg-white rounded-xl p-5 mb-5 space-y-4 border border-gray-100">
          <h2 className="text-base font-semibold text-gray-700">Options</h2>

          <label className="flex items-center gap-3 text-sm text-gray-700">
            <span className="w-44">Confidence threshold</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="w-20 border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </label>

          <label className="flex items-center gap-3 text-sm text-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={cleanup}
              onChange={(e) => setCleanup(e.target.checked)}
              className="w-4 h-4 rounded accent-blue-600"
            />
            Delete intermediate files after job completes
          </label>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 mb-5 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting || !file}
          className="bg-blue-600 text-white font-semibold px-6 py-2.5 rounded-lg
            hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors"
        >
          {submitting ? 'Uploading…' : 'Run inference'}
        </button>
      </form>
    </div>
  )
}
