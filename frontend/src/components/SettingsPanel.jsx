import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Settings2, ChevronDown, RefreshCw, Info } from 'lucide-react'

function Label({ children, tip }) {
  const [show, setShow] = useState(false)
  return (
    <div className="flex items-center gap-1.5 mb-1.5 relative">
      <span className="text-xs font-mono text-subtle">{children}</span>
      {tip && (
        <button
          onMouseEnter={() => setShow(true)}
          onMouseLeave={() => setShow(false)}
          className="text-muted hover:text-subtle"
        >
          <Info size={11} />
        </button>
      )}
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute left-0 top-5 z-20 glass rounded-lg px-3 py-2 text-xs
                       text-subtle w-52 shadow-xl border-neon/20"
          >
            {tip}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function Slider({ label, tip, value, onChange, min, max, step = 1, fmt }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <Label tip={tip}>{label}</Label>
      <div className="flex items-center gap-3">
        <div className="relative flex-1 h-1.5 bg-border rounded-full">
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-plasma to-arc"
            style={{ width: `${pct}%` }}
          />
          <input
            type="range" min={min} max={max} step={step}
            value={value}
            onChange={e => onChange(step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value))}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>
        <span className="text-xs font-mono text-neon w-14 text-right">
          {fmt ? fmt(value) : value}
        </span>
      </div>
    </div>
  )
}

export default function SettingsPanel({ settings, onChange, onReset }) {
  const [open, setOpen] = useState(false)

  const set = (key) => (val) => onChange({ ...settings, [key]: val })

  const resolutions = ['576×320', '512×320', '320×256', '256×256']
  const resMap = {
    '576×320': [576, 320],
    '512×320': [512, 320],
    '320×256': [320, 256],
    '256×256': [256, 256],
  }
  const currentRes = `${settings.width}×${settings.height}`

  return (
    <div className="glass rounded-2xl overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4
                   hover:bg-card/50 transition-colors duration-200"
      >
        <div className="flex items-center gap-2.5 text-sm text-text font-display font-medium">
          <Settings2 size={16} className="text-neon" />
          Advanced Controls
        </div>
        <ChevronDown
          size={16}
          className={`text-subtle transition-transform duration-300 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-5 border-t border-border pt-4">

              <div className="rounded-lg border border-border/60 px-3 py-2 text-xs text-subtle leading-relaxed">
                Backend uses{' '}
                <span className="font-mono text-neon">cerspense/zeroscope_v2_576w</span>
                {' '}with DPMSolverMultistepScheduler at native{' '}
                <span className="font-mono">576×320</span> by default.
              </div>

              {/* Sampler */}
              <div className="space-y-3">
                <div className="text-xs font-mono text-neon tracking-widest">SAMPLER</div>
                <Slider
                  label="Inference Steps"
                  tip="Number of denoising steps. More steps = higher quality but slower."
                  value={settings.num_inference_steps}
                  onChange={set('num_inference_steps')}
                  min={5} max={100}
                />
                <Slider
                  label="Guidance Scale (CFG)"
                  tip="ε̃ = ε(∅) + w·(ε(c)−ε(∅)). Higher w = stronger prompt adherence."
                  value={settings.guidance_scale}
                  onChange={set('guidance_scale')}
                  min={1} max={20} step={0.5}
                  fmt={v => v.toFixed(1)}
                />
              </div>

              {/* Video params */}
              <div className="space-y-3">
                <div className="text-xs font-mono text-neon tracking-widest">VIDEO PARAMS</div>
                <Slider
                  label="Frames"
                  tip="Total number of frames to generate."
                  value={settings.num_frames}
                  onChange={set('num_frames')}
                  min={8} max={64}
                />
                <Slider
                  label="FPS"
                  tip="Frames per second of the output video."
                  value={settings.fps}
                  onChange={set('fps')}
                  min={4} max={30}
                />

                {/* Resolution selector */}
                <div>
                  <Label tip="Output video resolution. Higher = more VRAM needed.">Resolution</Label>
                  <div className="grid grid-cols-4 gap-2">
                    {resolutions.map(r => (
                      <button
                        key={r}
                        onClick={() => {
                          const [w, h] = resMap[r]
                          onChange({ ...settings, width: w, height: h })
                        }}
                        className={`py-1.5 rounded-lg text-xs font-mono transition-all
                                   ${currentRes === r
                                     ? 'bg-plasma text-white border border-plasma'
                                     : 'bg-card text-subtle border border-border hover:border-muted'}`}
                      >
                        {r}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Seed */}
              <div>
                <div className="text-xs font-mono text-neon tracking-widest mb-3">REPRODUCIBILITY</div>
                <Label tip="Fix the random seed for reproducible outputs. Leave blank for random.">Seed</Label>
                <input
                  type="number"
                  value={settings.seed ?? ''}
                  onChange={e => set('seed')(e.target.value === '' ? null : parseInt(e.target.value))}
                  placeholder="Random"
                  className="input-plasma w-full rounded-lg px-3 py-2 text-sm font-mono"
                />
              </div>

              {/* Reset */}
              <button
                onClick={onReset}
                className="btn-ghost w-full py-2.5 rounded-xl text-sm flex items-center
                           justify-center gap-2"
              >
                <RefreshCw size={13} />
                Reset to defaults
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
