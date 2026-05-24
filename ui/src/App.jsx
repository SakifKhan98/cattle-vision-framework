import { Routes, Route } from 'react-router-dom'
import UploadPage from './pages/UploadPage.jsx'
import ProgressPage from './pages/ProgressPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'
import './App.css'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/jobs/:jobId/progress" element={<ProgressPage />} />
      <Route path="/jobs/:jobId/results" element={<ResultsPage />} />
    </Routes>
  )
}
