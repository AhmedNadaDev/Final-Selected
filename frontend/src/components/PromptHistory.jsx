import { motion, AnimatePresence } from 'framer-motion'
import { History, Play, Trash2 } from 'lucide-react'

export default function PromptHistory({ history, onSelect, onClear }) {
  return (
    <div className="glass rounded-2xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-border">
        <div className="flex items-center gap-2 text-sm font-display font-medium text-text">
          <History size={15} className="text-neon" />
          Recent Prompts
        </div>
        {history.length > 0 && (
          <button
            onClick={onClear}
            className="text-muted hover:text-flare transition-colors"
          >
            <Trash2 size={13} />
          </button>
        )}
      </div>

      <div className="max-h-72 overflow-y-auto">
        <AnimatePresence>
          {history.length === 0 ? (
            <div className="px-5 py-8 text-center text-muted text-xs font-mono">
              No history yet
            </div>
          ) : (
            history.slice().reverse().map((item, i) => (
              <motion.button
                key={item.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => onSelect(item.prompt)}
                className="w-full text-left px-5 py-3.5 flex items-start gap-3
                           hover:bg-card/60 transition-colors duration-150 group
                           border-b border-border/50 last:border-0"
              >
                <Play size={11} className="text-muted group-hover:text-neon mt-0.5
                                           transition-colors flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-text truncate">{item.prompt}</p>
                  <p className="text-xs text-muted mt-0.5 font-mono">
                    {item.timestamp} · {item.status}
                  </p>
                </div>
              </motion.button>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
