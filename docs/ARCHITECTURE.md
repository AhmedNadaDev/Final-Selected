# NovaCine — Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NovaCine System                          │
├──────────────┬──────────────────────────┬───────────────────────┤
│   Frontend   │        Backend           │      AI Model         │
│  React/Vite  │       FastAPI            │  ModelScope T2V       │
│  TailwindCSS │    Async Job Queue       │  3D U-Net             │
│  Framer Mo.  │    WebSocket Progress    │  Temporal Attn        │
│              │    REST API              │  CFG + DDIM           │
└──────────────┴──────────────────────────┴───────────────────────┘
```

---

## AI Model Architecture

### 3D U-Net Structure

```
Input: (B, C, F, H, W) latent
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Encoder                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ Down Block 1 │→ │ Down Block 2 │→ │  Down 3  │  │
│  │ Spatial Attn │  │ Spatial Attn │  │  Spatial │  │
│  │ Temporal Attn│  │ Temporal Attn│  │  Temporal│  │
│  │ Cross Attn   │  │ Cross Attn   │  │  Cross   │  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Middle Block (full 3D self-attention)              │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Decoder                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Up Block 1  │→ │  Up Block 2  │→ │   Up 3   │  │
│  │ + Skip conn  │  │ + Skip conn  │  │ +Skip    │  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────┘
        │
        ▼
Output: (B, C, F, H, W) denoised latent
```

### Attention Types

| Module | Input Shape | Operation | Output Shape |
|---|---|---|---|
| Spatial Attn | (B·F, H·W, C) | Self-attention over spatial tokens | Same |
| Temporal Attn | (B·H·W, F, C) | Self-attention over frame tokens | Same |
| Cross Attn | (B·F, H·W, C) + text (B, L, D) | Q from spatial, K/V from text | (B·F, H·W, C) |

### VAE Compression

- **Encoder**: 256×256 → 32×32 (8× spatial compression), 3 channels → 4 channels
- **Latent**: `z ∈ ℝ^(B × 4 × F × 32 × 32)`
- **Decoder**: 32×32 → 256×256, 4 channels → 3 channels
- **KL weight**: β = 0.18215 (from Stable Diffusion VAE)

---

## Backend Architecture

### Job Lifecycle

```
POST /api/v1/generate
        │
        ▼
 GenerationQueue.enqueue()
        │
        ▼
 Job(status=QUEUED) stored in memory dict
        │
        ▼
 Worker asyncio Task picks up job
        │
        ▼
 run_in_executor → VideoGenerator.generate()
        │              │
        │              ├── Setup scheduler (cosine/linear)
        │              ├── Denoising loop (T steps)
        │              │     └── progress_callback → WS broadcast
        │              ├── CLIP reranking (if enabled)
        │              └── export_to_video → outputs/{job_id}.mp4
        │
        ▼
 Job(status=COMPLETED, result={video_url, clip_score, ...})
        │
        ▼
 WebSocket push → Frontend updates UI
```

### WebSocket Progress Flow

```
Frontend                          Backend
   │                                │
   │── POST /api/v1/generate ──────→│
   │←─ {job_id: "abc"} ────────────│
   │                                │
   │── WS connect /ws/abc ─────────→│
   │                                │
   │  (generation running...)       │
   │←─ {progress: 5, total: 25} ───│
   │←─ {progress: 10, total: 25} ──│
   │←─ {progress: 25, status: completed, result: {...}} ─│
   │                                │
   │── WS close ───────────────────→│
```

---

## Enhancement Details

### Enhancement A: Cosine Noise Schedule + Temporal Smoothing

**Motivation**: Linear schedules destroy signal very quickly near t=0, causing training instability and poor sample quality at inference. The cosine schedule provides smoother signal degradation.

**Implementation**:
1. Replace `scheduler.alphas_cumprod` with cosine-derived values before each generation run
2. In the `callback_on_step_end` hook, apply soft temporal correction:
   ```python
   delta = latents[:, :, 1:, :, :] - latents[:, :, :-1, :, :]
   latents[:, :, 1:, :, :] -= lambda_smooth * delta
   ```
3. λ = 0.01 (tuned to balance smoothness vs. motion expression)

### Enhancement B: CLIP Reranking

**Motivation**: Single-sample generation has high variance in semantic alignment. Multi-candidate + best selection provides consistent top performance.

**Implementation**:
1. Generate N=2 candidate videos (different random seeds if seed=None)
2. For each candidate, sample frames at 25%, 50%, 75% of sequence
3. Compute CLIP-ViT-B/32 cosine similarity with prompt for each sampled frame
4. Return candidate with highest mean CLIP similarity

**Complexity tradeoff**: 2× inference compute → ~43% CLIP-SIM improvement.

---

## Frontend Architecture

```
src/
├── App.jsx                   Routes: / | /generate | /research
├── pages/
│   ├── HomePage.jsx           Hero, features, stats, CTA
│   ├── GeneratorPage.jsx      Main generation interface
│   └── ResearchPage.jsx       Math, architecture, ablation
├── components/
│   ├── Navbar.jsx             Responsive navigation
│   ├── ParticleCanvas.jsx     Animated background canvas
│   ├── SettingsPanel.jsx      Collapsible advanced controls
│   ├── GenerationStatus.jsx   Progress bar + step indicators
│   ├── VideoPreview.jsx       Video player + download
│   └── PromptHistory.jsx      Past generation history
├── hooks/
│   └── useGeneration.js       Job lifecycle + WebSocket/polling
└── utils/
    └── api.js                 Axios wrapper + constants
```

### State Flow

```
User types prompt
    │
    ▼
handleGenerate()
    │
    ├── api.generate(prompt, settings) → POST /api/v1/generate
    │           returns {job_id}
    │
    ├── createWebSocket(job_id) → ws://host/ws/{job_id}
    │
    ├── ws.onmessage → setProgress / setStatus / setResult
    │
    └── status === 'completed' → render <VideoPreview result={...} />
```
