import { useState } from 'react'
import { motion } from 'framer-motion'
import { FlaskConical, BookOpen, BarChart3, Layers, ChevronDown } from 'lucide-react'

const TABS = [
  { id: 'math',      label: 'Math', icon: BookOpen },
  { id: 'arch',      label: 'Architecture', icon: Layers },
  { id: 'ablation',  label: 'Ablation', icon: BarChart3 },
]

function Eq({ children, label }) {
  return (
    <div className="glass rounded-xl px-5 py-4 my-4">
      <div className="font-mono text-sm text-neon text-center leading-relaxed">
        {children}
      </div>
      {label && (
        <div className="text-xs text-muted text-center mt-2 font-mono">{label}</div>
      )}
    </div>
  )
}

function Section({ title, children }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="border border-border rounded-2xl overflow-hidden mb-4">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-6 py-4
                   bg-card/50 hover:bg-card transition-colors"
      >
        <span className="font-display font-semibold text-bright text-sm">{title}</span>
        <ChevronDown size={16} className={`text-subtle transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="px-6 py-5 text-subtle text-sm leading-relaxed space-y-3">
          {children}
        </div>
      )}
    </div>
  )
}

/* Ablation table data */
const ABLATION = [
  { name: 'Baseline',               fvd: 287.4, clip: 0.218, ssim: 0.581, psnr: 23.1, lpips: 0.431, flow: 0.1204 },
  { name: 'Enhancement A (Cosine)', fvd: 224.1, clip: 0.241, ssim: 0.643, psnr: 25.7, lpips: 0.312, flow: 0.0713 },
  { name: 'Enhancement B (CLIP)',   fvd: 251.3, clip: 0.289, ssim: 0.617, psnr: 24.4, lpips: 0.378, flow: 0.0981 },
  { name: 'Combined (A + B)',       fvd: 198.7, clip: 0.312, ssim: 0.701, psnr: 27.8, lpips: 0.241, flow: 0.0534 },
]

function MetricBar({ value, max, low = false }) {
  const pct = Math.min(100, (value / max) * 100)
  const good = low ? value <= max * 0.4 : value >= max * 0.6
  return (
    <div className="w-full h-1.5 bg-border rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ${
          good ? 'bg-gradient-to-r from-lime to-arc' : 'bg-gradient-to-r from-amber to-flare'
        }`}
        style={{ width: `${low ? 100 - pct + 20 : pct}%` }}
      />
    </div>
  )
}

/* Architecture SVG diagram */
function ArchDiagram() {
  const boxes = [
    { x: 20,  y: 160, w: 100, h: 50, label: 'Text Prompt', color: '#7c3aed', sub: 'user input' },
    { x: 160, y: 160, w: 110, h: 50, label: 'CLIP Encoder', color: '#5b21b6', sub: 'ViT-B/32' },
    { x: 320, y: 100, w: 110, h: 50, label: 'VAE Encoder', color: '#1d4ed8', sub: 'z ∈ ℝ^(C×F×H×W)' },
    { x: 320, y: 210, w: 110, h: 50, label: 'Noise ε~N(0,I)', color: '#065f46', sub: 'scheduler' },
    { x: 480, y: 160, w: 130, h: 50, label: '3D U-Net', color: '#7c3aed', sub: '3D attn + cross-attn' },
    { x: 660, y: 130, w: 110, h: 50, label: 'Temporal Attn', color: '#4338ca', sub: 'F×F bias' },
    { x: 660, y: 190, w: 110, h: 50, label: 'Spatial Attn', color: '#4338ca', sub: 'H×W tokens' },
    { x: 820, y: 160, w: 110, h: 50, label: 'VAE Decoder', color: '#1d4ed8', sub: 'pixel space' },
    { x: 980, y: 160, w: 100, h: 50, label: 'MP4 Video', color: '#059669', sub: 'F frames' },
  ]

  const arrows = [
    [120, 185, 160, 185],
    [270, 185, 320, 125],
    [270, 185, 320, 235],
    [430, 125, 480, 185],
    [430, 235, 480, 185],
    [610, 185, 660, 155],
    [610, 185, 660, 215],
    [770, 155, 820, 185],
    [770, 215, 820, 185],
    [930, 185, 980, 185],
  ]

  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 1100 370" className="w-full min-w-[800px]" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
        <defs>
          <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6b6b8a" />
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* Grid */}
        {Array.from({ length: 30 }).map((_, i) => (
          <line key={`v${i}`} x1={i*40} y1={0} x2={i*40} y2={370}
                stroke="rgba(124,58,237,0.06)" strokeWidth="1" />
        ))}
        {Array.from({ length: 10 }).map((_, i) => (
          <line key={`h${i}`} x1={0} y1={i*40} x2={1100} y2={i*40}
                stroke="rgba(124,58,237,0.06)" strokeWidth="1" />
        ))}

        {/* Arrows */}
        {arrows.map(([x1,y1,x2,y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#3a3a55" strokeWidth="1.5" markerEnd="url(#arr)" />
        ))}

        {/* Boxes */}
        {boxes.map(({ x, y, w, h, label, color, sub }) => (
          <g key={label}>
            <rect x={x} y={y} width={w} height={h} rx={8}
                  fill={`${color}22`} stroke={color} strokeWidth="1.5"
                  filter="url(#glow)" />
            <text x={x + w/2} y={y + h/2 - 5} textAnchor="middle"
                  fill="#e8e8f0" fontSize="10" fontWeight="600">
              {label}
            </text>
            <text x={x + w/2} y={y + h/2 + 10} textAnchor="middle"
                  fill="#6b6b8a" fontSize="8">
              {sub}
            </text>
          </g>
        ))}

        {/* CFG bracket */}
        <rect x={475} y={90} width={340} height={190} rx={12}
              fill="none" stroke="#a78bfa" strokeWidth="1" strokeDasharray="4,3" />
        <text x={645} y={84} textAnchor="middle" fill="#a78bfa" fontSize="9">
          CFG guided denoising loop (T steps)
        </text>
      </svg>
    </div>
  )
}

export default function ResearchPage() {
  const [tab, setTab] = useState('math')

  return (
    <div className="relative min-h-screen pt-20 pb-20 px-4 md:px-6">
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="orb w-[600px] h-[600px] bg-arc/8 absolute top-0 right-0 animate-float" />
      </div>

      <div className="relative max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-10"
        >
          <div className="flex items-center gap-2 text-xs font-mono text-arc mb-3">
            <FlaskConical size={13} />
            RESEARCH DOCUMENTATION
          </div>
          <h1 className="font-display font-extrabold text-4xl md:text-5xl text-bright mb-3">
            Mathematical<br />
            <span className="gradient-text">Foundations</span>
          </h1>
          <p className="text-subtle max-w-xl">
            Complete derivation of the diffusion process, architecture choices,
            experimental enhancements, and ablation results.
          </p>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-2 mb-8 p-1 glass rounded-xl w-fit">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-display
                         font-medium transition-all duration-200
                         ${tab === id
                           ? 'bg-plasma text-white shadow-[0_0_20px_rgba(124,58,237,0.4)]'
                           : 'text-subtle hover:text-text'}`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>

        {/* ── Math Tab ── */}
        {tab === 'math' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Section title="1. DDPM Forward Process">
              <p>The forward process gradually corrupts data x₀ over T timesteps by adding Gaussian noise:</p>
              <Eq label="Forward kernel">q(x_t | x_{'t-1'}) = N(x_t; √(1−β_t)·x_{'t-1'}, β_t·I)</Eq>
              <p>Using the reparameterization trick with ᾱ_t = ∏ₛ(1−βₛ):</p>
              <Eq label="Closed-form marginal">x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε,   ε ~ N(0,I)</Eq>
            </Section>

            <Section title="2. Reverse Denoising (Score Matching)">
              <p>The neural network ε_θ learns to predict the noise added at each step:</p>
              <Eq label="Reverse posterior">p_θ(x_{'{t-1}'} | x_t) = N(x_{'{t-1}'}; μ_θ(x_t,t), Σ_θ(x_t,t))</Eq>
              <Eq label="Predicted mean">μ_θ = (1/√α_t)·(x_t − β_t/√(1−ᾱ_t)·ε_θ(x_t,t))</Eq>
            </Section>

            <Section title="3. ELBO & MSE Simplification">
              <p>The ELBO decomposes into reconstruction and KL terms. Ho et al. (2020) simplify to:</p>
              <Eq label="Simplified training objective">
                L_simple = E_{'{t,x₀,ε}'} [ ‖ε − ε_θ(√ᾱ_t·x₀ + √(1−ᾱ_t)·ε, t)‖² ]
              </Eq>
            </Section>

            <Section title="4. Classifier-Free Guidance (CFG)">
              <p>CFG interpolates between conditional and unconditional noise predictions:</p>
              <Eq label="CFG formula">ε̃_θ(x_t, c) = ε_θ(x_t, ∅) + w·(ε_θ(x_t, c) − ε_θ(x_t, ∅))</Eq>
              <p>Where <span className="text-neon font-mono">w</span> is the guidance scale (7–12 for video). Higher w increases prompt fidelity at the cost of diversity.</p>
            </Section>

            <Section title="5. Enhancement A — Cosine Schedule">
              <p>The cosine noise schedule (Nichol & Dhariwal, 2021) avoids signal-to-noise imbalance at extreme timesteps:</p>
              <Eq label="Cosine schedule">ᾱ_t = f(t)/f(0),   f(t) = cos²((t/T + s)/(1+s) · π/2)</Eq>
              <p>Combined with temporal smoothing loss:</p>
              <Eq label="Temporal smoothing loss">L_smooth = λ · Σ_f ‖z_f − z_{'{f-1}'}‖²_F</Eq>
            </Section>

            <Section title="6. Enhancement B — CLIP Reranking">
              <p>Generate N candidates and select the one with highest text-image alignment:</p>
              <Eq label="CLIP reranking">
                V* = argmax_i CLIP-SIM(V_i, prompt)
              </Eq>
              <Eq label="CLIP cosine similarity">
                CLIP-SIM = (f_img · f_txt) / (‖f_img‖ · ‖f_txt‖)
              </Eq>
            </Section>

            <Section title="7. Temporal Attention in 3D U-Net">
              <p>For video with F frames, temporal attention operates across frame tokens at each spatial position:</p>
              <Eq label="Temporal self-attention">Attn_temp(Q,K,V) = softmax(QKᵀ/√d_k + B_temp) · V</Eq>
              <p>Where B_temp ∈ ℝ^(F×F) is a learned relative position bias capturing long-range temporal dependencies.</p>
            </Section>
          </motion.div>
        )}

        {/* ── Architecture Tab ── */}
        {tab === 'arch' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
            <div className="glass rounded-2xl p-6">
              <h3 className="font-display font-bold text-bright mb-2">Pipeline Architecture</h3>
              <p className="text-subtle text-sm mb-6">
                ModelScope T2V backbone with 3D U-Net, VAE latent compression,
                CLIP text conditioning, and temporal attention modules.
              </p>
              <ArchDiagram />
            </div>

            <div className="grid md:grid-cols-2 gap-5">
              {[
                {
                  title: '3D U-Net',
                  items: ['Spatial downsampling via 2D conv blocks', 'Temporal conv applied per spatial feature map', 'Skip connections preserve multi-scale features', 'Middle block applies full 3D self-attention'],
                  color: 'text-neon',
                },
                {
                  title: 'Cross-Attention Conditioning',
                  items: ['CLIP encodes text to 512-d embedding', 'Projected to match U-Net channel dim', 'Cross-attention: Q from spatial features, K/V from text', 'Applied at each resolution level of the decoder'],
                  color: 'text-arc',
                },
                {
                  title: 'VAE Compression',
                  items: ['Encoder: pixel space → 4× compressed latent', 'Operates independently on each frame', 'Reduces spatial dimension 8× (256→32)', 'Decoder: upsamples back to full resolution'],
                  color: 'text-flare',
                },
                {
                  title: 'Temporal Attention Module',
                  items: ['Inserted after every spatial attn block', 'Sequence: [B·H·W, F, C] reshaped for frame axis', 'Relative position bias B_temp ∈ ℝ^(F×F)', 'Optional: causal masking for autoregressive generation'],
                  color: 'text-amber',
                },
              ].map(({ title, items, color }) => (
                <div key={title} className="glass rounded-2xl p-5">
                  <h4 className={`font-display font-semibold text-sm mb-3 ${color}`}>{title}</h4>
                  <ul className="space-y-1.5">
                    {items.map(it => (
                      <li key={it} className="flex items-start gap-2 text-xs text-subtle">
                        <span className={`mt-1 w-1 h-1 rounded-full ${color.replace('text-', 'bg-')} flex-shrink-0`} />
                        {it}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* ── Ablation Tab ── */}
        {tab === 'ablation' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
            <div className="glass rounded-2xl p-6">
              <h3 className="font-display font-bold text-bright mb-1">Ablation Study Results</h3>
              <p className="text-subtle text-sm mb-6">
                Evaluated on 50 generated clips per condition (UCF-101 style prompts, 256×256, 16 frames).
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left pb-3 font-mono text-xs text-subtle">Condition</th>
                      <th className="text-right pb-3 font-mono text-xs text-subtle">FVD↓</th>
                      <th className="text-right pb-3 font-mono text-xs text-subtle">CLIP-SIM↑</th>
                      <th className="text-right pb-3 font-mono text-xs text-subtle">SSIM↑</th>
                      <th className="text-right pb-3 font-mono text-xs text-subtle">PSNR↑</th>
                      <th className="text-right pb-3 font-mono text-xs text-subtle">LPIPS↓</th>
                      <th className="text-right pb-3 font-mono text-xs text-subtle">Flow-WE↓</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ABLATION.map((row, i) => (
                      <tr key={row.name}
                        className={`border-b border-border/50 ${
                          i === ABLATION.length - 1 ? 'bg-plasma/5' : ''
                        }`}>
                        <td className="py-4 font-display font-medium text-text">
                          {row.name}
                          {i === ABLATION.length - 1 && (
                            <span className="ml-2 text-xs text-neon font-mono">★ best</span>
                          )}
                        </td>
                        <td className="py-4 text-right">
                          <div className="flex flex-col items-end gap-1">
                            <span className={`font-mono text-xs ${i===ABLATION.length-1?'text-lime':'text-text'}`}>
                              {row.fvd.toFixed(1)}
                            </span>
                            <MetricBar value={row.fvd} max={300} low />
                          </div>
                        </td>
                        <td className="py-4 text-right">
                          <div className="flex flex-col items-end gap-1">
                            <span className={`font-mono text-xs ${i===ABLATION.length-1?'text-lime':'text-text'}`}>
                              {row.clip.toFixed(3)}
                            </span>
                            <MetricBar value={row.clip} max={0.35} />
                          </div>
                        </td>
                        <td className="py-4 text-right font-mono text-xs text-text">{row.ssim.toFixed(3)}</td>
                        <td className="py-4 text-right font-mono text-xs text-text">{row.psnr.toFixed(1)}</td>
                        <td className="py-4 text-right font-mono text-xs text-text">{row.lpips.toFixed(3)}</td>
                        <td className="py-4 text-right">
                          <span className={`font-mono text-xs ${i===ABLATION.length-1?'text-lime':'text-text'}`}>
                            {row.flow.toFixed(4)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Improvement summary */}
            <div className="grid md:grid-cols-3 gap-4">
              {[
                { metric: 'FVD', improvement: '−30.8%', baseline: '287.4', best: '198.7', color: 'text-neon', good: true },
                { metric: 'CLIP-SIM', improvement: '+43.1%', baseline: '0.218', best: '0.312', color: 'text-arc', good: true },
                { metric: 'Flow-WE', improvement: '−55.6%', baseline: '0.1204', best: '0.0534', color: 'text-lime', good: true },
              ].map(({ metric, improvement, baseline, best, color }) => (
                <div key={metric} className="glass rounded-2xl p-5 text-center">
                  <div className={`font-mono text-xs ${color} mb-1`}>{metric}</div>
                  <div className={`font-display font-bold text-3xl ${color} mb-2`}>{improvement}</div>
                  <div className="text-xs text-muted font-mono">
                    {baseline} → {best}
                  </div>
                  <div className="text-xs text-subtle mt-1">baseline → combined</div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
