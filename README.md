# NovaCine — Text-to-Video Generation System

NovaCine is a **full-stack text-to-video application**: a **React** web UI talks to a **FastAPI** backend that runs a **latent diffusion** pipeline (Hugging Face **Diffusers**, **ZeroScope V2 576w**) with optional **neural narration** merged with **ffmpeg**. Jobs run through an **async queue** with **WebSocket** progress updates; generated MP4 files are served from the backend under `/outputs`.

This repository also includes **research artifacts**: evaluation metrics, ablation scripts, an IEEE-style paper (`paper/`), and a standalone CLI that reuses the same generator code.

---

## Table of contents

1. [How the system works](#how-the-system-works)
2. [Models and ML stack](#models-and-ml-stack)
3. [What talks to what (integrations)](#what-talks-to-what-integrations)
4. [Repository layout](#repository-layout)
5. [What you need to run it](#what-you-need-to-run-it)
6. [Setup and run](#setup-and-run)
7. [HTTP API (summary)](#http-api-summary)
8. [Standalone CLI (no UI)](#standalone-cli-no-ui)
9. [Evaluation and research scripts](#evaluation-and-research-scripts)
10. [Troubleshooting](#troubleshooting)
11. [Academic context](#academic-context)

---

## How the system works

### End-to-end flow

1. **Browser → Frontend (Vite dev server or nginx in Docker)**  
   The UI submits generation settings and prompt to `/api/v1/generate`.

2. **Proxy**  
   - **Development:** Vite proxies `/api`, `/outputs`, and `/ws` to `http://localhost:8000` (see `frontend/vite.config.js`).  
   - **Docker:** nginx proxies those paths to the `backend` service (see `frontend/nginx.conf`).

3. **Backend**  
   - FastAPI (`backend/main.py`) registers routes under `/api/v1`, mounts static **`outputs/`** at `/outputs`, and exposes **`/ws/{job_id}`** for progress.  
   - Each request creates a **`GenerationConfig`** and **`GenerationQueue.enqueue()`** stores the job and processes it (default **one worker**).

4. **Generation**  
   - **`VideoGenerator`** (`backend/generator.py`) loads **`cerspense/zeroscope_v2_576w`** via **`DiffusionPipeline.from_pretrained`** (fp16 on CUDA), **`DPMSolverMultistepScheduler`**, **`enable_attention_slicing()`**, single-pass inference (default **576×320**, 25 steps).  
   - Video bytes are written under **`backend/outputs/{job_id}.mp4`**. If audio is enabled, **Edge TTS** writes MP3 and **`mux_audio_video`** produces **`{job_id}_with_audio.mp4`**.

5. **Progress**  
   The queue pushes JSON updates over the WebSocket so the UI can show step progress.

6. **No GPU / failed model load**  
   If the diffusion pipeline fails to load, the backend falls back to a **mock generator** (synthetic frames) so the API and UI remain testable without CUDA.

### Architecture at a glance

```text
┌─────────────┐     proxy      ┌──────────────────────────────────────────┐
│ React (UI)  │ ────────────► │ FastAPI (NovaCine API)                   │
│ Vite :5173  │               │  • POST /api/v1/generate                 │
│ or nginx:80 │ ◄── WS/json ─ │  • GET  /api/v1/jobs/{id}                │
└─────────────┘               │  • WS   /ws/{job_id}                     │
                              │  • GET  /outputs/* (StaticFiles)         │
                              │  • GenerationQueue → VideoGenerator      │
                              └─────────────────┬────────────────────────
                                                │
                    Hugging Face Hub (first run) │  CUDA (recommended)
                    caches weights               ▼
                              ┌──────────────────────────────────────────┐
                              │ ZeroScope V2 576w + Diffusers           │
                              │ Optional: Edge TTS, ffmpeg mux           │
                              └──────────────────────────────────────────┘
```

---

## Models and ML stack

| Component | Role | Identifier / notes |
|-----------|------|---------------------|
| **Text-to-video backbone** | Main diffusion model | **`cerspense/zeroscope_v2_576w`**, loaded with Hugging Face **Diffusers** `DiffusionPipeline`. FP16 on CUDA when available. |
| **Schedulers** | Denoising | **`DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)`** after load. |
| **Diagnostics** | Motion hint | Optional **`motion_score`** (mean absolute frame difference) in API results for debugging static output. |
| **Neural TTS (bonus)** | Narration audio | **Microsoft Edge Neural TTS** via Python package **`edge-tts`** (calls Microsoft’s online TTS service). Requires **network access** when synthesizing. |
| **Video / metrics** | Encoding & evaluation | **diffusers** `export_to_video`, **ffmpeg** (mux), **opencv**, **scikit-image**, **LPIPS**; optional CLIP only in `evaluation/run_metrics.py` for research metrics. |

Core Python libraries (see **`backend/requirements.txt`**): **PyTorch**, **diffusers**, **transformers**, **accelerate**, **torchvision**, **Pillow**, **loguru**, etc.

---

## What talks to what (integrations)

| Connection | Purpose |
|------------|---------|
| **Browser ↔ Backend** | REST (`/api/v1/*`), WebSocket (`/ws/{job_id}`), static video URLs (`/outputs/...`). CORS allows localhost frontends and `*` in dev (`backend/main.py`). |
| **Frontend dev server ↔ Backend** | Vite proxy to port **8000** for API, outputs, and WebSockets. |
| **Docker frontend ↔ Docker backend** | nginx **`proxy_pass`** to **`http://backend:8000`** for `/api/`, `/ws/`, `/outputs/`. |
| **Backend ↔ Hugging Face Hub** | On first load, **`from_pretrained`** downloads weights into **`~/.cache/huggingface`** (Docker volume **`model_cache`** maps there). |
| **Backend ↔ Microsoft TTS** | When **`enable_audio`** is true, **`edge-tts`** reaches Microsoft’s neural TTS endpoints over the internet. |
| **Backend ↔ filesystem** | Writes **`backend/outputs/*.mp4`** (and optional `.mp3`); Docker mounts **`./backend/outputs`** for persistence. |

There is **no separate database**: job state lives **in memory** in `GenerationQueue` (lost on server restart).

---

## Repository layout

```text
FinalProjectSelected/
├── frontend/              # React + Vite + Tailwind; proxies to backend in dev
├── backend/
│   ├── main.py            # FastAPI app, CORS, static /outputs, WebSocket
│   ├── api/routes.py      # /api/v1 REST endpoints
│   ├── config.py          # MODEL_ID, GenerationConfig (576×320 defaults)
│   ├── generator.py       # VideoGenerator (ZeroScope), TTS hook
│   ├── core/
│   │   └── queue_manager.py
│   └── utils/             # TTS, video mux (ffmpeg), helpers
├── ai-model/pipeline/     # Standalone CLI (imports backend core)
├── evaluation/            # Metrics (FVD, CLIP-SIM, SSIM, PSNR, LPIPS, FWE, …)
├── scripts/               # setup.ps1 / setup.sh, demos, weakness analysis, user study
├── paper/                 # LaTeX paper
└── docs/                  # API.md, ARCHITECTURE.md
```

---

## What you need to run it

### Minimum (API + UI smoke test)

- **Python 3** with **`backend/requirements.txt`** installed in a venv.
- **Node.js** (for `npm install` / `npm run dev` in `frontend/`).

Without a GPU, the diffusion model may fail to load and the backend will use the **mock** video generator.

### Recommended for real video generation

- **NVIDIA GPU** with a recent **CUDA** stack matching **PyTorch 2.2.x** (see project Dockerfile base: CUDA 12.1 runtime image).
- Enough **VRAM** for **ZeroScope V2 576w** (plan for several GB; reduce resolution/frames if you hit OOM).
- **ffmpeg** on PATH for audio muxing (included in the **backend Docker** image; install separately on bare metal if you use TTS).

### Optional

- **CLIP** (only if you run **`evaluation/run_metrics.py`** or other research scripts that compute CLIP-SIM):  
  `pip install git+https://github.com/openai/CLIP.git`
- **Internet** for first-time **Hugging Face** weight download and for **Edge TTS** when audio is enabled.

---

## Setup and run

### Windows (PowerShell)

```powershell
.\scripts\setup.ps1
cd backend; .\.venv\Scripts\Activate.ps1; uvicorn main:app --reload --port 8000
```

New terminal:

```powershell
cd frontend; npm run dev
```

Open **http://localhost:5173** — API and WebSocket traffic are proxied to **http://localhost:8000**.

### Linux / macOS

```bash
bash scripts/setup.sh
cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000
# new terminal
cd frontend && npm run dev
```

### Docker (GPU inference recommended)

```bash
docker-compose up --build
```

- Frontend: **http://localhost:5173** (mapped from container port 80).
- Backend: **http://localhost:8000**.
- Compose requests **NVIDIA** GPU for the `backend` service and persists Hugging Face cache in volume **`model_cache`**.

---

## HTTP API (summary)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/generate` | Queue a job (prompt, frames, fps, guidance, seed, optional TTS fields). |
| GET | `/api/v1/jobs/{job_id}` | Poll status, progress, result URLs, errors. |
| GET | `/api/v1/jobs` | List jobs. |
| DELETE | `/api/v1/jobs/{job_id}` | Remove a job from in-memory history. |
| GET | `/api/v1/tts/catalog` | Voices and emotions for the bonus TTS module. |
| GET | `/health` | Liveness JSON. |
| WS | `/ws/{job_id}` | Optional real-time progress (job dict as JSON). |

Static files: **`GET /outputs/...`** serves files from **`backend/outputs/`**.

Full detail: **`docs/API.md`**.

---

## Standalone CLI (no UI)

From the repo root (after backend dependencies are available):

```bash
python ai-model/pipeline/infer.py "Your prompt here" --frames 24 --steps 25 --width 576 --height 320
```

This imports **`VideoGenerator`** from **`backend/`** — same logic as the API.

---

## Evaluation and research scripts

Examples:

```bash
# Synthetic demo (no GPU)
python scripts/demo_ablation.py

# Metrics across output folders
python evaluation/run_metrics.py --baseline outputs/baseline --enh-a outputs/enh_a \
  --enh-b outputs/enh_b --combined outputs/combined --prompt "your reference prompt"

# Weakness analysis demo
python scripts/run_weakness_analysis.py --demo

# User study logger
python scripts/user_study.py
```

See existing **`paper/`** and **`docs/ARCHITECTURE.md`** for methodology and diagrams.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Log says **mock generator** / no real video | Pipeline failed to load (no GPU, OOM, or network issue downloading weights). Check logs from **`VideoGenerator._load_pipeline`**. |
| Low **`motion_score`** in API result | Possible near-static decode; try another seed or prompt; verify CUDA pipeline loaded (not mock). |
| **Audio** fails | **edge-tts** / network / **ffmpeg** missing on host. Docker image includes **ffmpeg**. |
| **WebSocket** works in dev but not in custom setup | Ensure same host/port as Vite proxy or nginx **`/ws/`** location with **Upgrade** headers. |
| Jobs disappear after restart | Expected: queue and job dict are **in-memory** only. |

---

## Academic context

NovaCine was developed for **AIE418 — Selected Topics in AI 2** (Alamein International University). It implements:

- **Phase 1–2** deliverables: theme, diffusion **text-to-video** stack (production uses **ZeroScope V2 576w**), mathematical write-up, weakness analysis, optional research enhancements documented in **`paper/`**, ablations, and IEEE-style paper.
- **Bonus**: neural **text-to-audio** via **Edge TTS**, muxed into the final MP4.

For deeper theory (DDPM/DDIM, CFG, schedules) and result tables, see **`paper/novacine_paper.tex`** and the previous sections of this README’s research lineage.

---

## License and third-party models

Respect the **license terms** of **Hugging Face** models (**`cerspense/zeroscope_v2_576w`** and any metrics/eval dependencies), **Diffusers**, optional **OpenAI CLIP** (evaluation scripts only), and **Microsoft** speech services when using **edge-tts**. This README does not substitute those licenses.
