import { useState, useRef, useCallback, useEffect } from 'react'
import { api, createWebSocket } from '../utils/api'
import toast from 'react-hot-toast'

export const STATUS = {
  IDLE:      'idle',
  QUEUED:    'queued',
  RUNNING:   'running',
  COMPLETED: 'completed',
  FAILED:    'failed',
}

export default function useGeneration() {
  const [status,   setStatus]   = useState(STATUS.IDLE)
  const [progress, setProgress] = useState(0)
  const [total,    setTotal]    = useState(25)
  const [message,  setMessage]  = useState('')
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)
  const [jobId,    setJobId]    = useState(null)
  const wsRef = useRef(null)

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  useEffect(() => () => cleanup(), [cleanup])

  const pollFallback = useCallback(async (id) => {
    const iv = setInterval(async () => {
      try {
        const { data } = await api.getJob(id)
        setProgress(data.progress)
        setTotal(data.total_steps)
        setMessage(data.message)
        setStatus(data.status)
        if (data.status === STATUS.COMPLETED) {
          setResult(data.result)
          clearInterval(iv)
        } else if (data.status === STATUS.FAILED) {
          setError(data.error)
          clearInterval(iv)
        }
      } catch { clearInterval(iv) }
    }, 1500)
    return iv
  }, [])

  const generate = useCallback(async (prompt, settings) => {
    cleanup()
    setStatus(STATUS.QUEUED)
    setProgress(0)
    setResult(null)
    setError(null)
    setMessage('Submitting job...')

    try {
      const { data } = await api.generate({ prompt, ...settings })
      const id = data.job_id
      setJobId(id)

      // Try WebSocket first, fall back to polling
      try {
        const ws = createWebSocket(id)
        wsRef.current = ws

        ws.onmessage = (e) => {
          const job = JSON.parse(e.data)
          setProgress(job.progress)
          setTotal(job.total_steps)
          setMessage(job.message)
          setStatus(job.status)
          if (job.status === STATUS.COMPLETED) {
            setResult(job.result)
            toast.success('Video generated!')
            cleanup()
          } else if (job.status === STATUS.FAILED) {
            setError(job.error || 'Generation failed')
            toast.error('Generation failed')
            cleanup()
          }
        }

        ws.onerror = () => {
          cleanup()
          pollFallback(id)
        }
      } catch {
        pollFallback(id)
      }
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Request failed'
      setError(msg)
      setStatus(STATUS.FAILED)
      toast.error(msg)
    }
  }, [cleanup, pollFallback])

  const reset = useCallback(() => {
    cleanup()
    setStatus(STATUS.IDLE)
    setProgress(0)
    setTotal(25)
    setMessage('')
    setResult(null)
    setError(null)
    setJobId(null)
  }, [cleanup])

  const pct = total > 0 ? Math.round((progress / total) * 100) : 0

  return { status, progress, total, pct, message, result, error, jobId, generate, reset }
}
