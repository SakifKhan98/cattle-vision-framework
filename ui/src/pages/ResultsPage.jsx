import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'

export default function ResultsPage() {
  const { jobId } = useParams()
  const [manifest, setManifest] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`/jobs/${jobId}/results`)
      .then((r) => {
        if (!r.ok) throw new Error(`Status ${r.status}`)
        return r.json()
      })
      .then(setManifest)
      .catch((e) => setError(e.message))
  }, [jobId])

  if (error) return (
    <div className="page">
      <h1>Results</h1>
      <div className="error-box">{error}</div>
      <Link to="/"><button className="btn-secondary">Back to upload</button></Link>
    </div>
  )

  if (!manifest) return (
    <div className="page"><p>Loading results…</p></div>
  )

  return (
    <div className="page">
      <h1>Inference complete</h1>
      <p style={{ fontSize: '0.82rem', color: '#888', marginBottom: '1.5rem' }}>
        Job: {jobId}
      </p>

      <div className="progress-card">
        <h2>Downloads</h2>
        <ul style={{ margin: 0, padding: '0 0 0 1.2rem', lineHeight: 2 }}>
          <li><a href={manifest.annotated_video} download>Annotated video (mp4)</a></li>
          <li><a href={manifest.activity_budget} download>Activity budget (CSV)</a></li>
          <li><a href={manifest.behavior_deviation} download>Behavior deviation (CSV)</a></li>
          {manifest.timelines?.length > 0 && (
            <li>
              Timelines:
              <ul>
                {manifest.timelines.map((url) => {
                  const name = url.split('/').pop()
                  return <li key={url}><a href={url} download>{name}</a></li>
                })}
              </ul>
            </li>
          )}
        </ul>
      </div>

      <Link to="/"><button className="btn-secondary">Run another video</button></Link>
    </div>
  )
}
