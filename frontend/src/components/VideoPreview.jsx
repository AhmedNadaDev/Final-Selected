import { useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, Play, RotateCcw, Zap, Activity, Film, Volume2 } from 'lucide-react'

export default function VideoPreview({ result, onReset }) {
  const videoRef = useRef(null)

  if (!result) return null

  const videoUrl = result.video_url || result.video_path

  const handleDownload = () => {
    const a = document.createElement('a')
    a.href = videoUrl
    a.download = `novacine-${result.job_id?.slice(0, 8) ?? 'video'}.mp4`
    a.click()
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 20 }}
        animate={{ opacity: 1, scale: 1,    y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="glass rounded-2xl overflow-hidden border-neon/20"
      >
        {/* Video element */}
        <div className="relative bg-void aspect-video group">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            autoPlay
            loop
            className="w-full h-full object-contain"
          />
          {/* Subtle vignette overlay */}
          <div className="absolute inset-0 pointer-events-none
                          bg-gradient-to-t from-void/40 via-transparent to-void/10" />
        </div>

        {/* Meta + actions */}
        <div className="p-5 space-y-4">
          {/* Stats row */}
          <div className="flex flex-wrap gap-3">
            {[
              { icon: Film,      label: 'Frames',     val: result.num_frames },
              { icon: Play,      label: 'FPS',        val: result.fps },
              { icon: Film,      label: 'Length',     val: `${result.duration_video_s ?? '—'}s` },
              { icon: Activity,  label: 'Motion Δ',   val: result.motion_score?.toFixed(4) ?? '—' },
              ...(result.clip_score != null && result.clip_score !== undefined
                ? [{ icon: Activity, label: 'CLIP pick', val: result.clip_score?.toFixed(4) }]
                : []),
              { icon: Zap,       label: 'Gen time',   val: `${result.duration_s}s` },
              ...(result.audio_url
                ? [{ icon: Volume2, label: 'Audio', val: 'on' }]
                : []),
            ].map(({ icon: Icon, label, val }) => (
              <div key={label}
                className="flex items-center gap-2 glass-bright rounded-xl px-3 py-2"
              >
                <Icon size={13} className={label === 'Audio' ? 'text-arc' : 'text-neon'} />
                <span className="text-xs text-subtle">{label}</span>
                <span className="text-xs font-mono text-bright">{val}</span>
              </div>
            ))}
          </div>

          {/* Audio-only fallback player (in case the video element can't decode AAC inline) */}
          {result.audio_url && (
            <audio
              src={result.audio_url}
              controls
              className="w-full mt-1"
              style={{ filter: 'invert(0.85)' }}
            />
          )}

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleDownload}
              className="btn-plasma flex-1 py-3 rounded-xl text-white text-sm
                         flex items-center justify-center gap-2"
            >
              <Download size={15} />
              Download MP4
            </button>
            <button
              onClick={onReset}
              className="btn-ghost px-5 py-3 rounded-xl text-sm flex items-center gap-2"
            >
              <RotateCcw size={14} />
              New
            </button>
          </div>

          {/* Job ID */}
          <p className="text-muted text-xs font-mono text-center">
            Job: {result.job_id}
          </p>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
