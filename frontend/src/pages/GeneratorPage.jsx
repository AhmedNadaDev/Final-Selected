import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  Zap, Wand2, X, ChevronRight, Sparkles
} from 'lucide-react'
import toast from 'react-hot-toast'

import useGeneration, { STATUS } from '../hooks/useGeneration'
import { DEFAULT_SETTINGS } from '../utils/api'
import GenerationStatus from '../components/GenerationStatus'
import VideoPreview from '../components/VideoPreview'
import SettingsPanel from '../components/SettingsPanel'
import PromptHistory from '../components/PromptHistory'
import AudioPanel from '../components/AudioPanel'

const EXAMPLE_PROMPTS = [
  'A lone astronaut walking on Mars at sunrise, cinematic',
  'Neon-lit Tokyo alley in heavy rain, film noir style',
  'Ocean waves crashing in slow motion, golden hour',
  'Wolf running through a snowy pine forest at dusk',
  'Time-lapse of aurora borealis over icy mountains',
  'Futuristic city flyover at night, blade runner aesthetic',
]

export default function GeneratorPage() {
  const [prompt,   setPrompt]   = useState('')
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [history,  setHistory]  = useState([])
  const textareaRef = useRef(null)

  const { status, pct, message, result, error, generate, reset } = useGeneration()

  const isGenerating = status === STATUS.RUNNING || status === STATUS.QUEUED
  const isComplete   = status === STATUS.COMPLETED

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt')
      return
    }
    setHistory(h => [
      ...h,
      {
        id: Date.now(),
        prompt: prompt.trim(),
        timestamp: new Date().toLocaleTimeString(),
        status: 'generating...',
      }
    ])
    await generate(prompt.trim(), settings)
    setHistory(h =>
      h.map(item =>
        item.prompt === prompt.trim() && item.status === 'generating...'
          ? { ...item, status: status === STATUS.FAILED ? 'failed' : 'done' }
          : item
      )
    )
  }

  const handleReset = () => {
    reset()
    setPrompt('')
  }

  const handleSelectPrompt = (p) => {
    setPrompt(p)
    textareaRef.current?.focus()
  }

  return (
    <div className="relative min-h-screen pt-20 pb-16 px-4 md:px-6">
      {/* Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="orb w-[500px] h-[500px] bg-plasma/15 absolute -top-20 right-0 animate-float" />
        <div className="orb w-[400px] h-[400px] bg-arc/10 absolute bottom-20 left-0 animate-float"
             style={{ animationDelay: '3s' }} />
      </div>

      <div className="relative max-w-7xl mx-auto">
        {/* Page header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8"
        >
          <div className="flex items-center gap-2 text-xs font-mono text-neon mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-neon animate-pulse" />
            AI VIDEO GENERATOR
          </div>
          <h1 className="font-display font-extrabold text-4xl md:text-5xl text-bright">
            Text to{' '}
            <span className="gradient-text">Video</span>
          </h1>
          <p className="text-subtle mt-2">
            Describe a scene. Watch it come alive.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-[1fr_380px] gap-6">
          {/* ── Left Column: Prompt + Status + Video ── */}
          <div className="space-y-5">
            {/* Prompt area */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.6 }}
              className="glass rounded-2xl p-5 space-y-4"
            >
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono text-neon tracking-widest">PROMPT</label>
                {prompt && (
                  <button
                    onClick={() => setPrompt('')}
                    className="text-muted hover:text-subtle transition-colors"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>

              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleGenerate()
                  }}
                  disabled={isGenerating}
                  placeholder="A lone astronaut walking on Mars at sunrise, slow motion, cinematic, 4K quality..."
                  rows={5}
                  className="input-plasma w-full rounded-xl px-4 py-3 text-sm resize-none
                             disabled:opacity-50 disabled:cursor-not-allowed leading-relaxed"
                />
                <div className="absolute bottom-3 right-3 text-muted text-xs font-mono">
                  {prompt.length}/500
                </div>
              </div>

              {/* Negative prompt */}
              <div>
                <label className="text-xs font-mono text-subtle mb-1.5 block">
                  NEGATIVE PROMPT
                </label>
                <input
                  type="text"
                  value={settings.negative_prompt}
                  onChange={e => setSettings(s => ({ ...s, negative_prompt: e.target.value }))}
                  className="input-plasma w-full rounded-xl px-4 py-2.5 text-xs"
                />
              </div>

              {/* Example prompts */}
              <div>
                <div className="text-xs font-mono text-muted mb-2">TRY AN EXAMPLE</div>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_PROMPTS.slice(0, 4).map(p => (
                    <button
                      key={p}
                      onClick={() => setPrompt(p)}
                      className="text-xs glass rounded-lg px-3 py-1.5 text-subtle
                                 hover:text-neon hover:border-neon/30 transition-all"
                    >
                      {p.length > 40 ? p.slice(0, 40) + '…' : p}
                    </button>
                  ))}
                </div>
              </div>

              {/* Generate button */}
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className={`btn-plasma w-full py-4 rounded-xl text-white font-display font-semibold
                           text-base flex items-center justify-center gap-3 relative
                           disabled:opacity-50 disabled:cursor-not-allowed
                           ${!isGenerating && prompt.trim() ? 'animate-glow-pulse' : ''}`}
              >
                {isGenerating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Generating... {pct}%
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    Generate Video
                    <span className="text-xs opacity-60 font-mono ml-1">⌘↵</span>
                  </>
                )}
              </button>
            </motion.div>

            {/* Generation status */}
            <GenerationStatus
              status={status}
              pct={pct}
              message={message}
              error={error}
            />

            {/* Video preview */}
            {isComplete && result && (
              <VideoPreview result={result} onReset={handleReset} />
            )}
          </div>

          {/* ── Right Column: Settings + History ── */}
          <div className="space-y-5">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              <SettingsPanel
                settings={settings}
                onChange={setSettings}
                onReset={() => setSettings(DEFAULT_SETTINGS)}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25, duration: 0.6 }}
            >
              <AudioPanel
                settings={settings}
                onChange={setSettings}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
            >
              <PromptHistory
                history={history}
                onSelect={handleSelectPrompt}
                onClear={() => setHistory([])}
              />
            </motion.div>

            {/* Tips card */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="glass rounded-2xl p-5 space-y-3"
            >
              <div className="flex items-center gap-2 text-xs font-mono text-amber">
                <Sparkles size={13} />
                PROMPT TIPS
              </div>
              {[
                'Include camera style: "aerial", "slow motion", "POV"',
                'Mention lighting: "golden hour", "neon-lit", "moonlight"',
                'Add art style: "cinematic", "film noir", "anime"',
                'Describe motion explicitly: "running", "floating", "swirling"',
              ].map((tip, i) => (
                <div key={i} className="flex items-start gap-2">
                  <ChevronRight size={11} className="text-amber mt-0.5 flex-shrink-0" />
                  <span className="text-subtle text-xs leading-relaxed">{tip}</span>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}
