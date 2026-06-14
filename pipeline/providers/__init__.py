"""
Cloud-provider clients (state-of-the-art quality), each key-gated and fallback-safe.

Every function returns None / False on a missing key, missing dependency, or any
API error — it NEVER raises — so callers degrade gracefully to the local engine.
This mirrors the pipeline's existing cascade philosophy.

- fal_provider    : Flux images, Kling image-to-video, Stable Audio music (fal.ai)
- hedra_provider  : Character-3 talking-photo lip-sync
- eleven_provider : ElevenLabs narration + word-level timestamps
"""
