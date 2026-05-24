import { useEffect, useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import UploadPage from './pages/UploadPage.jsx'
import ProgressPage from './pages/ProgressPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'

function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4"/>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  )
}

export default function App() {
  const [dark, setDark] = useState(
    () => localStorage.getItem('theme') === 'dark'
      || (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)
  )

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors duration-200">
      <header className="border-b border-gray-200 dark:border-gray-800 px-4 py-3 flex justify-end">
        <button
          onClick={() => setDark(d => !d)}
          aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400
            hover:text-gray-900 dark:hover:text-gray-100 transition-colors px-3 py-1.5
            rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
        >
          {dark ? <SunIcon /> : <MoonIcon />}
          {dark ? 'Light mode' : 'Dark mode'}
        </button>
      </header>

      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/jobs/:jobId/progress" element={<ProgressPage />} />
        <Route path="/jobs/:jobId/results" element={<ResultsPage />} />
      </Routes>
    </div>
  )
}
