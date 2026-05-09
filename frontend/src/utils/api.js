import axios from 'axios'

const BASE = '/api/v1'

export const api = {
  generate:   (payload) => axios.post(`${BASE}/generate`, payload),
  getJob:     (id)      => axios.get(`${BASE}/jobs/${id}`),
  listJobs:   ()        => axios.get(`${BASE}/jobs`),
  deleteJob:  (id)      => axios.delete(`${BASE}/jobs/${id}`),
  ttsCatalog: ()        => axios.get(`${BASE}/tts/catalog`),
}

export function createWebSocket(jobId) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host  = window.location.host
  return new WebSocket(`${proto}://${host}/ws/${jobId}`)
}

export const DEFAULT_SETTINGS = {
  negative_prompt:      'blurry, distorted, low quality, watermark, worst quality',
  // 32 frames @ 8 fps = 4 seconds (Phase-2 deliverable: clip ≥ 4s)
  num_frames:           32,
  fps:                  8,
  width:                256,
  height:               256,
  num_inference_steps:  25,
  guidance_scale:       9.0,
  seed:                 null,
  use_ddim:             true,
  use_enhancement_a:    true,
  use_enhancement_b:    true,
  num_candidates:       2,

  // Bonus: Text-to-Audio (Edge Neural TTS)
  enable_audio:         false,
  audio_script:         '',
  audio_voice:          'en-US-AriaNeural',
  audio_rate:           '+0%',
  audio_pitch:          '+0Hz',
  audio_emotion:        'neutral',
}

// Voices kept in sync with backend/utils/tts.py VOICES dict (fallback list).
export const FALLBACK_VOICES = [
  { id: 'en-US-AriaNeural',   label: 'Aria (US, conversational)',  lang: 'en-US', gender: 'Female' },
  { id: 'en-US-GuyNeural',    label: 'Guy (US, narrator)',          lang: 'en-US', gender: 'Male'   },
  { id: 'en-US-JennyNeural',  label: 'Jenny (US, friendly)',        lang: 'en-US', gender: 'Female' },
  { id: 'en-US-DavisNeural',  label: 'Davis (deep, dramatic)',      lang: 'en-US', gender: 'Male'   },
  { id: 'en-US-AmberNeural',  label: 'Amber (warm)',                lang: 'en-US', gender: 'Female' },
  { id: 'en-GB-RyanNeural',   label: 'Ryan (UK)',                   lang: 'en-GB', gender: 'Male'   },
  { id: 'en-GB-SoniaNeural',  label: 'Sonia (UK)',                  lang: 'en-GB', gender: 'Female' },
  { id: 'ar-EG-SalmaNeural',  label: 'Salma (Egyptian Arabic)',     lang: 'ar-EG', gender: 'Female' },
  { id: 'ar-EG-ShakirNeural', label: 'Shakir (Egyptian Arabic)',    lang: 'ar-EG', gender: 'Male'   },
]
export const FALLBACK_EMOTIONS = [
  'neutral', 'happy', 'sad', 'excited', 'calm', 'angry', 'whispering', 'narration',
]
