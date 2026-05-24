import { Routes, Route } from 'react-router-dom'
import UploadPage from './pages/UploadPage.jsx'
import ProgressPage from './pages/ProgressPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/jobs/:jobId/progress" element={<ProgressPage />} />
        <Route path="/jobs/:jobId/results" element={<ResultsPage />} />
      </Routes>
    </div>
  )
}
