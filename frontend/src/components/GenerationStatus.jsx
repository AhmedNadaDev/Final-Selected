import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, CheckCircle, XCircle, Clock } from 'lucide-react'
import { STATUS } from '../hooks/useGeneration'

function StatusIcon({ status }) {
  if (status === STATUS.RUNNING || status === STATUS.QUEUED)
    return <Loader2 size={18} className="animate-spin text-neon" />
  if (status === STATUS.COMPLETED)
    return <CheckCircle size={18} className="text-lime" />
  if (status === STATUS.FAILED)
    return <XCircle size={18} className="text-flare" />
  return <Clock size={18} className="text-subtle" />
}

const statusColors = {
  [STATUS.IDLE]:      'text-subtle',
  [STATUS.QUEUED]:    'text-arc',
  [STATUS.RUNNING]:   'text-neon',
  [STATUS.COMPLETED]: 'text-lime',
  [STATUS.FAILED]:    'text-flare',
}

const statusLabel = {
  [STATUS.IDLE]:      'Idle',
  [STATUS.QUEUED]:    'Queued',
  [STATUS.RUNNING]:   'Generating',
  [STATUS.COMPLETED]: 'Complete',
  [STATUS.FAILED]:    'Failed',
}

// Animated step indicators
const STEPS = [
  'Initializing pipeline',
  'Text encoding',
  'Denoising',
  'Decoding frames',
  'Video encoding',
]

export default function GenerationStatus({ status, pct, message, error }) {
  const isActive = status === STATUS.RUNNING || status === STATUS.QUEUED

  return (
    <AnimatePresence mode="wait">
      {status !== STATUS.IDLE && (
        <motion.div
          key="status-panel"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -16 }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          className="glass rounded-2xl p-5 space-y-4"
        >
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <StatusIcon status={status} />
              <span className={`font-display font-semibold text-sm ${statusColors[status]}`}>
                {statusLabel[status]}
              </span>
            </div>
            {isActive && (
              <span className="font-mono text-xs text-neon">{pct}%</span>
            )}
            {status === STATUS.COMPLETED && (
              <span className="font-mono text-xs text-lime">✓ Done</span>
            )}
          </div>

          {/* Progress bar */}
          {isActive && (
            <div className="space-y-2">
              <div className="h-1.5 bg-border rounded-full overflow-hidden">
                <motion.div
                  className="h-full progress-fill rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>
              <p className="text-xs text-subtle font-mono truncate">{message}</p>
            </div>
          )}

          {/* Completed bar */}
          {status === STATUS.COMPLETED && (
            <div className="h-1.5 bg-border rounded-full overflow-hidden">
              <div className="h-full w-full bg-gradient-to-r from-lime to-arc rounded-full" />
            </div>
          )}

          {/* Step dots (running) */}
          {isActive && (
            <div className="flex flex-wrap gap-2">
              {STEPS.map((step, i) => {
                const stepPct = (i + 1) / STEPS.length * 100
                const done = pct >= stepPct
                const active = pct >= (i / STEPS.length * 100) && !done
                return (
                  <div
                    key={step}
                    className={`flex items-center gap-1.5 text-xs font-mono px-2.5 py-1
                               rounded-full transition-all duration-300
                               ${done   ? 'bg-plasma/20 text-neon border border-plasma/30'
                               : active ? 'bg-card text-arc border border-arc/30 animate-pulse'
                               :          'bg-card/50 text-muted border border-border'}`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full
                                     ${done ? 'bg-neon' : active ? 'bg-arc' : 'bg-muted'}`} />
                    {step}
                  </div>
                )
              })}
            </div>
          )}

          {/* Error */}
          {status === STATUS.FAILED && error && (
            <div className="bg-flare/10 border border-flare/20 rounded-xl px-4 py-3">
              <p className="text-flare text-xs font-mono">{error}</p>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
