import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import HomePage from './pages/HomePage'
import GeneratorPage from './pages/GeneratorPage'
import ResearchPage from './pages/ResearchPage'
import Navbar from './components/Navbar'
import ParticleCanvas from './components/ParticleCanvas'

export default function App() {
  return (
    <div className="noise min-h-screen bg-void">
      <ParticleCanvas />
      <Navbar />
      <Routes>
        <Route path="/"          element={<HomePage />} />
        <Route path="/generate"  element={<GeneratorPage />} />
        <Route path="/research"  element={<ResearchPage />} />
      </Routes>
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#1a1a2e',
            color: '#e8e8f0',
            border: '1px solid rgba(124,58,237,0.3)',
            fontFamily: 'DM Sans, sans-serif',
          },
        }}
      />
    </div>
  )
}
