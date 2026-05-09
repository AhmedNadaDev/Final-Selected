import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, useInView } from 'framer-motion'
import {
  Zap, Layers, FlaskConical, BarChart3,
  ArrowRight, Play, ChevronDown,
  Cpu, Sparkles, Video
} from 'lucide-react'

/* ── Stagger helpers ── */
const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.16, 1, 0.3, 1] } },
}
const stagger = (delay = 0) => ({
  hidden: { opacity: 0, y: 30 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] } },
})

/* ── Feature card data ── */
const FEATURES = [
  {
    icon: Cpu,
    color: 'text-neon',
    bg: 'from-plasma/20 to-transparent',
    title: 'Latent Diffusion Core',
    desc: '3D U-Net with spatial + temporal attention for frame-coherent synthesis across the full denoising trajectory.',
  },
  {
    icon: Sparkles,
    color: 'text-arc',
    bg: 'from-arc/20 to-transparent',
    title: 'CLIP Reranking',
    desc: 'Enhancement B: generate N candidates, score each with CLIP-ViT-B/32, return the highest-fidelity clip.',
  },
  {
    icon: Video,
    color: 'text-flare',
    bg: 'from-flare/20 to-transparent',
    title: 'Temporal Smoothing',
    desc: 'Enhancement A: cosine noise schedule + inter-frame latent smoothing loss eliminates temporal flickering.',
  },
  {
    icon: BarChart3,
    color: 'text-amber',
    bg: 'from-amber/20 to-transparent',
    title: 'Research Metrics',
    desc: 'Auto-runs FVD, CLIP-SIM, SSIM, PSNR, LPIPS, and Flow Warping Error across ablation conditions.',
  },
  {
    icon: Layers,
    color: 'text-lime',
    bg: 'from-lime/20 to-transparent',
    title: 'DDPM / DDIM',
    desc: 'Switch between stochastic DDPM and deterministic DDIM — 10–50× faster with identical quality at 25 steps.',
  },
  {
    icon: FlaskConical,
    color: 'text-neon',
    bg: 'from-neon/20 to-transparent',
    title: 'Full Paper Support',
    desc: 'IEEE-format paper draft, architecture diagrams, ablation tables, and LaTeX export — all auto-generated.',
  },
]

/* ── Stats ── */
const STATS = [
  { value: '3D',    label: 'U-Net Architecture' },
  { value: '6+',   label: 'Evaluation Metrics' },
  { value: '2×',   label: 'Research Enhancements' },
  { value: '25',   label: 'Inference Steps (DDIM)' },
]

function FeatureCard({ icon: Icon, color, bg, title, desc, delay }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })
  return (
    <motion.div
      ref={ref}
      variants={stagger(delay)}
      initial="hidden"
      animate={inView ? 'show' : 'hidden'}
      className="glass rounded-2xl p-6 group hover:border-plasma/40 transition-all duration-300
                 hover:shadow-[0_0_30px_rgba(124,58,237,0.15)] relative overflow-hidden"
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${bg} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
      <div className={`w-10 h-10 rounded-xl bg-card flex items-center justify-center mb-4 ${color}
                       group-hover:scale-110 transition-transform duration-300`}>
        <Icon size={18} />
      </div>
      <h3 className="font-display font-semibold text-base text-bright mb-2">{title}</h3>
      <p className="text-subtle text-sm leading-relaxed">{desc}</p>
    </motion.div>
  )
}

function SamplePrompt({ text, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="glass rounded-xl px-4 py-3 flex items-center gap-3 group cursor-pointer
                 hover:border-neon/30 transition-all"
    >
      <div className="w-2 h-2 rounded-full bg-neon animate-pulse-slow flex-shrink-0" />
      <span className="text-subtle text-sm font-mono group-hover:text-text transition-colors">{text}</span>
    </motion.div>
  )
}

const SAMPLE_PROMPTS = [
  'A lone astronaut walking on a red Martian surface at dawn',
  'Neon-lit Tokyo street in heavy rain, cinematic style',
  'Aerial ocean waves crashing in slow motion, golden hour',
  'A wolf running through a snowy forest at midnight',
  'Time-lapse of flowers blooming in a sunlit meadow',
]

export default function HomePage() {
  const featRef = useRef(null)
  const featInView = useInView(featRef, { once: true, margin: '-80px' })

  return (
    <div className="relative min-h-screen">
      {/* Background orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="orb w-[600px] h-[600px] bg-plasma/20 absolute -top-40 -left-40 animate-float" />
        <div className="orb w-[500px] h-[500px] bg-arc/10 absolute top-1/3 right-0 animate-float" style={{ animationDelay: '2s' }} />
        <div className="orb w-[400px] h-[400px] bg-flare/8 absolute bottom-0 left-1/3 animate-float" style={{ animationDelay: '4s' }} />
        <div className="absolute inset-0"
          style={{
            backgroundImage: 'linear-gradient(rgba(124,58,237,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(124,58,237,0.04) 1px,transparent 1px)',
            backgroundSize: '48px 48px'
          }}
        />
      </div>

      {/* ── Hero ── */}
      <section className="relative pt-32 pb-24 px-6">
        <div className="max-w-5xl mx-auto text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 glass rounded-full px-4 py-2 mb-8 text-xs font-mono text-neon border-neon/20"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-neon animate-pulse" />
            Research-Grade · Production-Ready · Open Architecture
          </motion.div>

          {/* Headline */}
          <motion.h1
            variants={fadeUp}
            initial="hidden"
            animate="show"
            className="font-display font-extrabold text-5xl md:text-7xl lg:text-8xl leading-[0.95] mb-6 tracking-tight"
          >
            <span className="text-bright">Text into</span>
            <br />
            <span className="gradient-text text-glow">Cinema</span>
          </motion.h1>

          <motion.p
            variants={stagger(0.15)}
            initial="hidden"
            animate="show"
            className="text-subtle text-lg md:text-xl max-w-2xl mx-auto mb-12 leading-relaxed"
          >
            NovaCine transforms natural language into temporally coherent video clips via
            latent diffusion — with research-level enhancements, full evaluation metrics,
            and a publishable paper scaffold.
          </motion.p>

          {/* CTA buttons */}
          <motion.div
            variants={stagger(0.25)}
            initial="hidden"
            animate="show"
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link to="/generate"
              className="btn-plasma px-8 py-4 rounded-xl text-white text-base
                         flex items-center gap-2.5 animate-glow-pulse"
            >
              <Zap size={18} />
              Start Generating
              <ArrowRight size={16} />
            </Link>
            <Link to="/research"
              className="btn-ghost px-8 py-4 rounded-xl text-base flex items-center gap-2.5"
            >
              <FlaskConical size={18} />
              View Research
            </Link>
          </motion.div>

          {/* Scroll indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            className="mt-20 flex flex-col items-center gap-2 text-muted"
          >
            <span className="text-xs font-mono tracking-widest">SCROLL</span>
            <ChevronDown size={16} className="animate-bounce" />
          </motion.div>
        </div>
      </section>

      {/* ── Stats bar ── */}
      <section className="relative py-12 border-y border-border">
        <div className="max-w-5xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8">
          {STATS.map(({ value, label }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              className="text-center"
            >
              <div className="font-display font-bold text-4xl gradient-text mb-1">{value}</div>
              <div className="text-subtle text-sm">{label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Sample prompts ── */}
      <section className="relative py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="font-display font-bold text-3xl text-bright mb-3">
              What will you create?
            </h2>
            <p className="text-subtle">Try one of these prompts or write your own</p>
          </motion.div>
          <div className="grid gap-3">
            {SAMPLE_PROMPTS.map((p, i) => (
              <SamplePrompt key={p} text={p} delay={i * 0.08} />
            ))}
          </div>
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5 }}
            className="mt-8 text-center"
          >
            <Link to="/generate"
              className="btn-plasma px-6 py-3 rounded-xl text-white text-sm
                         inline-flex items-center gap-2"
            >
              <Play size={14} />
              Generate Now
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ── Features grid ── */}
      <section ref={featRef} className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate={featInView ? 'show' : 'hidden'}
            className="text-center mb-16"
          >
            <span className="text-xs font-mono text-neon tracking-widest uppercase mb-4 block">
              Architecture & Enhancements
            </span>
            <h2 className="font-display font-bold text-4xl md:text-5xl text-bright">
              Built for Research,<br />
              <span className="gradient-text">Deployed for Impact</span>
            </h2>
          </motion.div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f, i) => (
              <FeatureCard key={f.title} {...f} delay={i * 0.07} />
            ))}
          </div>
        </div>
      </section>

      {/* ── Architecture diagram teaser ── */}
      <section className="relative py-20 px-6">
        <div className="max-w-4xl mx-auto glass rounded-3xl p-10 text-center
                        border-plasma/20 hover:border-plasma/40 transition-colors duration-500">
          <div className="font-mono text-xs text-neon tracking-widest mb-4">PIPELINE OVERVIEW</div>
          <h3 className="font-display font-bold text-2xl text-bright mb-8">
            Latent Diffusion Video Synthesis
          </h3>
          {/* Mini pipeline diagram */}
          <div className="flex flex-wrap items-center justify-center gap-2 text-sm font-mono">
            {['Text Prompt','→','CLIP Encoder','→','3D U-Net','→','VAE Decoder','→','MP4 Video'].map((s, i) => (
              <span
                key={i}
                className={s === '→' ? 'text-muted' : 'glass px-3 py-1.5 rounded-lg text-neon text-xs'}
              >
                {s}
              </span>
            ))}
          </div>
          <p className="text-subtle text-sm mt-6 max-w-lg mx-auto">
            Prompts are encoded via CLIP text encoder, conditioned into a 3D U-Net
            via cross-attention, iteratively denoised in latent space, then decoded
            through a VAE into full-resolution video frames.
          </p>
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section className="relative py-28 px-6 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <h2 className="font-display font-extrabold text-5xl md:text-6xl text-bright mb-6">
            Ready to{' '}
            <span className="gradient-text text-glow">generate?</span>
          </h2>
          <p className="text-subtle text-lg mb-10">
            One prompt. Seconds. Cinematic AI video.
          </p>
          <Link to="/generate"
            className="btn-plasma px-10 py-5 rounded-2xl text-white text-lg
                       inline-flex items-center gap-3 animate-glow-pulse"
          >
            <Zap size={20} />
            Open Generator
            <ArrowRight size={18} />
          </Link>
        </motion.div>
      </section>
    </div>
  )
}
