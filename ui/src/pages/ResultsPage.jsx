import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'

function DownloadLink({ href, label }) {
  return (
    <a
      href={href}
      download
      className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-800 hover:underline"
    >
      <span>↓</span>
      {label}
    </a>
  )
}

export default function ResultsPage() {
  const { jobId } = useParams()
  const [manifest, setManifest] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`/jobs/${jobId}/results`)
      .then((r) => { if (!r.ok) throw new Error(`Status ${r.status}`); return r.json() })
      .then(setManifest)
      .catch((e) => setError(e.message))
  }, [jobId])

  if (error) return (
    <div className="max-w-xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold mb-4">Results</h1>
      <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 mb-6 text-sm">
        {error}
      </div>
      <Link to="/"><button className="bg-gray-200 text-gray-700 font-semibold px-5 py-2.5 rounded-lg hover:bg-gray-300 transition-colors">Back to upload</button></Link>
    </div>
  )

  if (!manifest) return (
    <div className="max-w-xl mx-auto px-4 py-10 text-gray-400">Loading results…</div>
  )

  return (
    <div className="max-w-xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold mb-1">Inference complete</h1>
      <p className="text-xs text-gray-400 mb-8 font-mono">{jobId}</p>

      <div className="bg-white rounded-xl border border-gray-100 p-5 mb-6 space-y-3">
        <h2 className="text-base font-semibold text-gray-700 mb-1">Downloads</h2>
        <DownloadLink href={manifest.annotated_video} label="Annotated video (.mp4)" />
        <DownloadLink href={manifest.activity_budget} label="Activity budget (.csv)" />
        <DownloadLink href={manifest.behavior_deviation} label="Behavior deviation (.csv)" />
        {manifest.timelines?.length > 0 && (
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-2 mt-1">Timelines</p>
            <div className="space-y-2 pl-2">
              {manifest.timelines.map((url) => (
                <DownloadLink key={url} href={url} label={url.split('/').pop()} />
              ))}
            </div>
          </div>
        )}
      </div>

      <Link to="/">
        <button className="bg-gray-200 text-gray-700 font-semibold px-5 py-2.5 rounded-lg hover:bg-gray-300 transition-colors">
          Run another video
        </button>
      </Link>
    </div>
  )
}
