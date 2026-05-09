# NovaCine API Documentation

Base URL: `http://localhost:8000`

---

## REST Endpoints

### POST `/api/v1/generate`
Submit a video generation job.

**Request body:**
```json
{
  "prompt": "A wolf running through a snowy forest",
  "negative_prompt": "blurry, distorted, low quality",
  "num_frames": 32,
  "fps": 8,
  "width": 256,
  "height": 256,
  "num_inference_steps": 25,
  "guidance_scale": 9.0,
  "seed": null,
  "use_ddim": true,
  "use_enhancement_a": true,
  "use_enhancement_b": true,
  "num_candidates": 2,

  "enable_audio": false,
  "audio_script": null,
  "audio_voice": "en-US-AriaNeural",
  "audio_rate": "+0%",
  "audio_pitch": "+0Hz",
  "audio_emotion": "neutral"
}
```

**Response:**
```json
{
  "job_id": "3f7a2b1c-...",
  "status":  "queued",
  "message": "Job queued successfully"
}
```

---

### GET `/api/v1/jobs/{job_id}`
Poll job status and result.

**Response:**
```json
{
  "job_id":      "3f7a2b1c-...",
  "status":      "running",
  "progress":    12,
  "total_steps": 25,
  "message":     "Denoising step 12/25",
  "result":      null,
  "error":       null,
  "created_at":  "2024-01-01T00:00:00",
  "completed_at": null
}
```

**Result object (when completed):**
```json
{
  "job_id":          "3f7a2b1c-...",
  "video_path":      "outputs/3f7a2b1c-..._with_audio.mp4",
  "video_url":       "/outputs/3f7a2b1c-..._with_audio.mp4",
  "audio_url":       "/outputs/3f7a2b1c-....mp3",
  "num_frames":      32,
  "fps":             8,
  "duration_video_s": 4.0,
  "clip_score":      0.2941,
  "duration_s":      47.3,
  "enhancements": {
    "cosine_schedule": true,
    "clip_reranking":  true,
    "tts_audio":       true
  }
}
```

---

### GET `/api/v1/jobs`
List all jobs.

### DELETE `/api/v1/jobs/{job_id}`
Remove a job from history.

### GET `/outputs/{file}.mp4`
Download generated video file (served as static file).

---

## Bonus — TTS endpoint

### GET `/api/v1/tts/catalog`
Returns the set of voices and emotion styles supported by the
Neural TTS engine.

```json
{
  "voices": [
    { "id": "en-US-AriaNeural",  "lang": "en-US", "gender": "Female", "label": "Aria (US, conversational)" },
    { "id": "ar-EG-SalmaNeural", "lang": "ar-EG", "gender": "Female", "label": "Salma (Egyptian Arabic)" },
    ...
  ],
  "emotions": ["neutral", "happy", "sad", "excited", "calm", "angry", "whispering", "narration"]
}
```

Returns `503` when the TTS engine is not installed.

---

## WebSocket `/ws/{job_id}`

Connect immediately after receiving a `job_id` for real-time progress.

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`)
ws.onmessage = (event) => {
  const job = JSON.parse(event.data)
  console.log(`${job.progress}/${job.total_steps} — ${job.message}`)
  if (job.status === 'completed') {
    console.log('Video URL:', job.result.video_url)
    if (job.result.audio_url) console.log('Audio URL:', job.result.audio_url)
  }
}
```

---

## Generation Parameters Reference

### Video
| Parameter             | Type   | Default | Range    | Description                              |
|-----------------------|--------|---------|----------|------------------------------------------|
| `prompt`              | string | required | 3–500   | Text description                         |
| `negative_prompt`     | string | preset  | —        | Things to avoid                          |
| `num_frames`          | int    | **32**  | 8–64     | Number of frames (32 ≈ 4 s at 8 fps)     |
| `fps`                 | int    | 8       | 4–30     | Output frames per second                 |
| `width` / `height`    | int    | 256     | 128–512  | Spatial resolution                       |
| `num_inference_steps` | int    | 25      | 5–100    | Denoising steps                          |
| `guidance_scale`      | float  | 9.0     | 1–20     | CFG weight                               |
| `seed`                | int    | random  | any      | Reproducibility                          |
| `use_ddim`            | bool   | true    | —        | DDIM (fast) vs DDPM (stochastic)         |
| `use_enhancement_a`   | bool   | true    | —        | Cosine schedule + temporal smoothing     |
| `use_enhancement_b`   | bool   | true    | —        | CLIP-rerank `num_candidates` candidates  |
| `num_candidates`      | int    | 2       | 1–4      | Candidate count for CLIP reranking       |

### Bonus — TTS narration
| Parameter        | Type   | Default              | Description                                              |
|------------------|--------|----------------------|----------------------------------------------------------|
| `enable_audio`   | bool   | false                | Render and mux a narration track                         |
| `audio_script`   | string | (uses `prompt`)      | Custom narration text (UTF-8, Arabic OK)                 |
| `audio_voice`    | string | `en-US-AriaNeural`   | Voice ID from `/api/v1/tts/catalog`                      |
| `audio_rate`     | string | `+0%`                | `-25%` … `+35%`                                          |
| `audio_pitch`    | string | `+0Hz`               | `-100Hz` … `+100Hz`                                      |
| `audio_emotion`  | string | `neutral`            | One of: neutral, happy, sad, excited, calm, angry, whispering, narration |

---

## Health Check

### GET `/health`
```json
{ "status": "ok", "version": "1.0.0" }
```
