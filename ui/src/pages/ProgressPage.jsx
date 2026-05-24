import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, Link } from 'react-router-dom'

const STAGE_NAMES = {
  1: 'Ingest',
  2: 'Detect',
  3: 'Track',
  4: 'Extract',
  5: 'Classify',
  6: 'Analyze',
  7: 'Render',
}

export default function ProgressPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const esRef = useRef(null)

  const [event, setEvent] = useState(null)  // last progress event
  const [failed, setFailed] = useState(null)

  useEffect(() => {
    const es = new EventSource(`/jobs/${jobId}/stream`)
    esRef.current = es

    es.onmessage = (e) => {
      let data
      try { data = JSON.parse(e.data) } catch { return }

      if (data.status === 'complete') {
        es.close()
        navigate(`/jobs/${jobId}/results`)
        return
      }
      if (data.status === 'failed') {
        es.close()
        setFailed(data.error || 'Unknown error')
        return
      }
      setEvent(data)
    }

    es.onerror = () => {
      es.close()
      setFailed('Lost connection to server. The job may still be running — refresh to check.')
    }

    return () => es.close()
  }, [jobId, navigate])

  const stageNum = event?.stage ?? 0
  const stageName = event?.stage_name ?? STAGE_NAMES[stageNum] ?? '…'
  const totalStages = event?.total_stages ?? 7
  const frame = event?.frame ?? 0
  const totalFrames = event?.total_frames ?? 0

  const stageProgress = totalStages > 0 ? (stageNum / totalStages) * 100 : 0
  const frameProgress = totalFrames > 0 ? (frame / totalFrames) * 100 : 0

  if (failed) {
    return (
      <div className="page">
        <h1>Job failed</h1>
        <div className="error-box">{failed}</div>
        <Link to="/"><button className="btn-secondary">Back to upload</button></Link>
      </div>
    )
  }

  return (
    <div className="page">
      <h1>Running inference…</h1>
      <p style={{ fontSize: '0.82rem', color: '#888', marginBottom: '1.25rem' }}>
        Job: {jobId}
      </p>

      <div className="progress-card">
        <div className="stage-label">
          {stageNum > 0
            ? `Stage ${stageNum} / ${totalStages}: ${stageName}`
            : 'Starting…'}
        </div>
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${stageProgress}%` }}
            role="progressbar"
            aria-valuenow={stageNum}
            aria-valuemax={totalStages}
          />
        </div>
        <div className="frame-count">
          {totalFrames > 0
            ? `Frame ${frame.toLocaleString()} / ${totalFrames.toLocaleString()}`
            : 'Waiting for first event…'}
        </div>
      </div>

      {totalFrames > 0 && (
        <div className="progress-card">
          <div className="stage-label">Frame progress</div>
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${frameProgress}%` }}
              role="progressbar"
              aria-valuenow={frame}
              aria-valuemax={totalFrames}
            />
          </div>
          <div className="frame-count">
            {Math.round(frameProgress)}%
          </div>
        </div>
      )}
    </div>
  )
}
