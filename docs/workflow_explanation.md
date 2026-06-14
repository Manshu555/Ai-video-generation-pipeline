# AI Viral Reel Pipeline — Workflow & Full Change Log

> **Status:** Redesigned 2026-06-13/14. Default flow is now **faceless B-roll**
> (`VISUAL_MODE="stock"`): your story text → narration plan → real Pexels stock footage +
> kinetic captions → compiled reel. Published to GitHub (`Manshu555/Ai-video-generation-pipeline`,
> code-only — secrets/photos/outputs are gitignored). This document is authoritative; it supersedes
> the original auto-generated notes.
>
> **Verified end-to-end (2026-06-14):** faceless reel rendered for **story 1** (hardcoded script, all 5
> scenes real Pexels footage) and for **story 2** via **Ollama** (LLM planned the script + per-scene
> `stock_query`, all 5 scenes real Pexels footage). **Bug found & fixed:** llama3.2:3b sometimes returns
> scenes with an *empty* `voiceover_text` (→ silent B-roll, no captions). The script generator now
> **rejects + retries** any Ollama script with an empty/too-short voiceover, the normalizer **back-fills**
> any empty voiceover from the story, and the audio step **self-heals** timings-less voice files on
> re-run (see §2/§8). Pipeline-level fixes — not hand-patched outputs.

---

## 1. What the pipeline does

Turns a short business story into a **~30-second vertical reel (1080×1920, H.264/AAC)** through an
8-step flow, runnable two ways:

- **Streamlit wizard** — `app.py` (8-step UI: Story → Style → Script → Images → Clips → Audio → Assemble → Export)
- **Headless CLI** — `generate_video.py` (edit the constants at the top, run it; writes `output.mp4`)

The design goal of the redesign was **storytelling quality**, not just "text → pictures." Every stage
is a **cascade**: a state-of-the-art cloud engine first (if a key is present and has credit), then an
automatic fallback to a local/free engine, so the pipeline **never hard-fails** and costs $0 with an
empty `.env`.

The reference story is **"The Dishwasher Steak Heist"** (story id 1). It gets special treatment
(see §4) because it's the showcase reel.

**Default flow (faceless B-roll, `VISUAL_MODE="stock"`):**
```
your story text → LLM plans scenes + narration + a stock_query per scene (Ollama)
              → Pexels stock footage per scene (cover-cropped to 1080×1920, fit to narration)
              → edge-tts/ElevenLabs narration + word-synced kinetic captions
              → color arc + hero-text climax + music arc → output.mp4
```
No character, no talking-head — narration over real B-roll. This is the most reliable look on an
8 GB / RAM-limited machine and matches the classic "faceless" business-reel format. Set
`VISUAL_MODE="ai"` to switch back to the AI-character + lip-sync flow.

---

## 2. Everything that was done in the redesign (change log)

### Root-cause fixes
| # | Problem | Fix | File |
|---|---------|-----|------|
| 1 | Ollama (`llama3.2:3b`) failed almost every run → pipeline fell back to a **generic template that ignored the actual story** | **Hardcoded, hand-directed screenplay** `_DISHWASHER_SCRIPT` for story 1 (bypasses Ollama); Ollama still used for stories 2–9 | `pipeline/script_generator.py` |
| 2 | Streamlit crashed with `KeyError: 'voiceover_text'` on malformed LLM output | **Hardened `_normalize()`** — every scene field is guaranteed to exist; `app.py` reads use `.get()` | `pipeline/script_generator.py`, `app.py` |
| 3 | Kinetic captions never synced (empty `voice_XX.json`) | **edge-tts 7.x defaults to `boundary="SentenceBoundary"`** (≈2 events). Now passes `boundary="WordBoundary"` → real per-word timings | `pipeline/audio_generator.py` |
| 4 | Action scenes were flickery/gray AnimateDiff text-to-video | AnimateDiff **disabled** (`USE_ANIMATEDIFF=False`) — it's text-to-video, *ignores the curated still*, and collapses on 8 GB VRAM. Action scenes now use **Ken Burns on a good still** | `config.py`, `pipeline/video_generator.py` |
| 5 | SD 1.5 produced **gray granite textures** for the kitchen/restaurant scenes | The "cinematic" style prefix contained **"film grain"** and the scene prompts had **"cold grey/desaturated"** — weak SD 1.5 rendered literal grain. Removed all grade/grain words from prompts (mood now applied only in post) + trimmed prompts under the 77-token CLIP limit | `config.py`, `pipeline/image_generator.py`, `pipeline/script_generator.py` |
| 6 | Speaking scenes showed a **second person** in frame | The reference photo `IMG_5950-edited.jpg` had a 2nd person at the right edge. Cropped a clean single-subject portrait → `assets/character_refs/bob_clean.jpg`; SadTalker now **prefers the clean reference first** | `assets/character_refs/bob_clean.jpg`, `config.py`, `pipeline/lipsync_generator.py` |
| 7 | Local image quality ceiling | Reordered image cascade so **free Pollinations (Flux)** is tried before local SD 1.5 | `pipeline/image_generator.py` |
| 8 | AnimateDiff was slow (~50 s/step) / OOM | `release_image_pipeline()` frees the SD pipe + VRAM before the motion stage | `pipeline/image_generator.py`, `generate_video.py` |

### New capabilities added
- **Faceless B-roll flow (Pexels stock footage)** — `pipeline/providers/pexels_provider.py` searches real vertical stock clips per scene (by `stock_query`), cover-crops to 1080×1920, fits to the narration length, and NVENC-encodes. Default visual source (`VISUAL_MODE="stock"`); independent of the paid-cloud switch (free key). Each scene now carries a `stock_query`; the Ollama prompt + template + normalizer all populate it. On any miss it falls back to Ken Burns on a still.
- **ModelScope text-to-video engine** (`pipeline/modelscope_generator.py`) — `damo-vilab/text-to-video-ms-1.7b` as an **opt-in** action-scene motion engine (`USE_MODELSCOPE`, default off). Mirrors the AnimateDiff module (8 GB offload, NVENC). It's text-to-video (generates from the prompt, ignores the still) and 256×256/watermarked, so it sits *behind* Kling and above AnimateDiff in the action cascade and is off by default. See §7.
- **Cloud-provider layer** (`pipeline/providers/`) — key-gated, fallback-safe clients:
  - `fal_provider.py` — Flux images, Kling image-to-video, Stable Audio music
  - `hedra_provider.py` — Character-3 talking-photo lip-sync (REST)
  - `eleven_provider.py` — ElevenLabs narration + word-level timestamps
- **Cold→warm color arc** — `COLOR_GRADES` in `config.py`, applied once per scene in the assembler so the arc is identical across every motion engine.
- **Hero-text climax** — scene 4 renders full-frame kinetic typography ("HE DIDN'T HAVE A / **EGO PROBLEM.**") with the music dropping to near-silence (`assembler._apply_hero_text`).
- **Per-scene music arc** — each scene carries a `music_attenuation` (dB); the assembler builds a per-segment bed instead of a flat track.
- **`.env.example`** rewritten with the three relevant keys + free-tier signup URLs.

---

## 3. The 8-step workflow (current engines & cascades)

> Each row is a **fallback cascade**: left → right. Cloud steps are skipped automatically when
> `USE_CLOUD_PROVIDERS=False` or the key is missing/out of credit.

| Step | Stage | Cascade (best → fallback) | Key files |
|------|-------|---------------------------|-----------|
| 1 | **Script** | hardcoded screenplay (story 1) · Ollama `llama3.2:3b` · deterministic template | `script_generator.py` |
| 2 | **Style/Tone** | 4 styles × 3 tones (prompt control) | `config.py` |
| 3 | **Images** | fal Flux · Pollinations Flux (free) · local SDXL · local SD 1.5 · PIL placeholder | `image_generator.py`, `providers/fal_provider.py` |
| 4 | **Voice** | ElevenLabs (+ timestamps) · edge-tts (`WordBoundary`, rate `-5%`) · silent | `audio_generator.py`, `providers/eleven_provider.py` |
| 5 | **Visual — faceless (DEFAULT, `VISUAL_MODE="stock"`)** | Pexels stock footage (by `stock_query`) · Ken Burns on a still (fallback) | `video_generator.py`, `providers/pexels_provider.py` |
| 5 | **Motion — speaking** *(only `VISUAL_MODE="ai"`)* | Hedra Character-3 · SadTalker (local) · Ken Burns | `video_generator.py`, `providers/hedra_provider.py`, `lipsync_generator.py` |
| 5 | **Motion — action** *(only `VISUAL_MODE="ai"`)* | fal Kling image-to-video · ModelScope T2V *(opt-in)* · AnimateDiff *(disabled)* · Ken Burns on the still | `video_generator.py`, `providers/fal_provider.py`, `modelscope_generator.py` |
| 6 | **Music** | fal Stable Audio *(opt-in)* · bundled CC track · none | `assembler.py`, `audio_generator.py` |
| 7 | **Assemble** | MoviePy v2 + NVENC: color arc · kinetic captions · hero text · title cards · crossfades · music arc | `assembler.py` |
| 8 | **Export** | local file (`output.mp4`) · Google Drive upload (needs `credentials.json`) | `app.py`, `drive_uploader.py` |

**Scene routing** is automatic: each scene is tagged `shot_type = "speaking" | "action"`.
Speaking → lip-sync engines; action → motion/Ken Burns. Lip-sync is reserved for the scenes where
the character's face *is* the message (hook + climax + payoff), not every scene.

---

## 4. Story design — "The Dishwasher Steak Heist"

A hand-directed 5-scene screenplay with a deliberate **emotional + color arc**:

```
S1 HOOK         S2 THE CRIME    S3 THE COST      S4 THE LINE ★     S5 COMEBACK
intrigue      → cold shock    → lonely defeat  → revelation      → warmth / hope
warm amber      cold blue       desat. grey      contrast gold      full gold
SadTalker       Ken Burns       Ken Burns        SadTalker+hero     Ken Burns
(talking head)  (kitchen,pan)   (empty rest.)    ("EGO PROBLEM.")   (warm restaurant)
```

- **S4 is the climax**: shortest scene (~4.5 s), music drops to ≈ −50 dB, full-frame "EGO PROBLEM."
  hero typography over the darkened face. The words carry the moment.
- Captions are **word-by-word**, newest word in gold, synced to the TTS word timings.
- Total ≈ 38 s, **voice-driven** durations (no dead-air tails).

The exact voiceover lines, image prompts, camera moves, emotions, `hero_text`, and
`music_attenuation` per scene live in `_DISHWASHER_SCRIPT` in `pipeline/script_generator.py`.

---

## 5. Architecture — provider layer & fallback philosophy

```
generate_video.py / app.py  (orchestration)
        │
        ├── script_generator   → scenes[]  (hardcoded story1 | Ollama | template)
        ├── audio_generator     → voice_XX.mp3 + voice_XX.json (word timings)
        ├── image_generator     → scene_XX.png   → release_image_pipeline() (free VRAM)
        ├── video_generator     → scene_XX.mp4   (routes by shot_type)
        │       ├── speaking → providers.hedra → lipsync_generator(SadTalker) → ken_burns
        │       └── action   → providers.fal(Kling) → [ModelScope off] → [AnimateDiff off] → ken_burns
        └── assembler           → output.mp4   (color arc, captions, hero text, music, NVENC)

pipeline/providers/  (each: key-gated, returns None/False on any error — never raises)
        ├── fal_provider     (FAL_KEY)            Flux · Kling · Stable Audio
        ├── hedra_provider   (HEDRA_API_KEY)      Character-3 lip-sync
        └── eleven_provider  (ELEVENLABS_API_KEY) TTS + timestamps
```

**Principle:** a missing key or a failed API call silently degrades to the next engine. An empty
`.env` runs the entire pipeline locally and free.

### Configuration (`config.py`)
| Flag | Meaning |
|------|---------|
| `USE_CLOUD_PROVIDERS` | Master switch. **Currently `False`** (fal balance exhausted, no Eleven/Hedra keys) → fully local. |
| `PREFER_FAL_IMAGE / FAL_VIDEO / HEDRA / ELEVENLABS / FAL_MUSIC` | Per-stage cloud preference. |
| `USE_MODELSCOPE` | `False` — opt-in `damo-vilab/text-to-video-ms-1.7b` action engine (256×256, watermarked). See §7. |
| `USE_ANIMATEDIFF` | `False` — action scenes use Ken Burns instead (see §2 #4). |
| `COLOR_GRADES` | Per-emotion RGB multipliers (the cold→warm arc). |
| `SCENE_SPEECH_RATE` | edge-tts pace (`-5%`, more gravity). |
| `CHARACTER_REF_IMAGE` | `assets/character_refs/bob_clean.jpg` (single-subject crop). |

### `.env` keys (all optional)
`FAL_KEY` · `ELEVENLABS_API_KEY` · `HEDRA_API_KEY` (+ optional `ELEVENLABS_VOICE_ID`/`_MODEL`).
Signup URLs are in `.env.example`.

---

## 6. Diagnostics — why earlier outputs degraded (and the lip-sync analysis)

A detailed external diagnosis was reviewed. Here is how it maps to **what this pipeline actually does**
(important, because some of it assumes a different stack):

### Problem 1 — "Story 1 looks great, other storylines degrade"
**True, and the real mechanism is:** story 1 uses a **hardcoded, hand-directed screenplay**; stories
2–9 go through **Ollama**. When Ollama is *offline* they fall back to a generic template (weaker scene
prompts) — that was the original cause, **not** image overfitting. **With Ollama running** (see §8), the
arbitrary-story path now produces a proper story-tailored script: verified on story 2 (5 scenes, valid
JSON on the first attempt). So the fix is simply **keep Ollama up** (or wire a cloud LLM).
- ✅ *Their A (image quality inconsistency)* did bite us: SD 1.5 collapsed to gray textures on
  grainy/over-long prompts → fixed in §2 #5.
- ✅ *Their B (identity drift)* partially applies: speaking scenes use the **real reference photo**,
  action scenes use the **SD-generated character** — a real identity difference (see Limitations).
- ⚠️ *Their C (story-specific prompt engineering)* — yes; the fix is to give stories 2–9 the same
  hand-directed treatment or a reliable LLM (run Ollama, or wire a cloud LLM).

### Problem 2 — "Eyes/head moved but lips didn't"
**Our pipeline uses SadTalker, not LivePortrait.** SadTalker produces head motion **and** mouth-sync
in a single model from the driving audio, so the "wrong model order" / "LivePortrait isn't a lip-sync
model" reasons **don't apply here**. In our case, when a speaking scene looked off it was because:
- the **source still had a 2nd person** / the SD portrait **failed face detection** → SadTalker fell
  back or framed poorly (fixed in §2 #6 by cropping a clean reference and preferring it first), or
- the scene fell through to **Ken Burns** (no lip-sync at all) when SadTalker errored (e.g. S5 hit a
  memory error and became a Ken Burns shot).

The genuinely portable lessons from that analysis that we **did** adopt:
- **Use a single, clean, front-facing reference** with a clearly visible mouth → `bob_clean.jpg`.
- **Prefer one consistent character source** for all speaking scenes → SadTalker tries the reference first.
- (Audio note) edge-tts outputs MP3; SadTalker handles it via its own decoding, so the 16 kHz/mono
  caveat hasn't been an issue here — but it's worth keeping in mind if swapping lip-sync engines.

---

## 7. Known limitations & how to push quality higher

| Limitation | Why | How to fix |
|------------|-----|------------|
| Speaking face (real photo) ≠ action character (SD) | No local face-consistency engine | Add credit and use **Hedra** (lip-sync on a consistent portrait) + **fal Flux/Ideogram character-ref**; flip `USE_CLOUD_PROVIDERS=True` |
| Custom stories get a generic script | **Ollama offline** | `ollama serve` (model `llama3.2:3b` is on `D:`), or wire a cloud LLM into `script_generator` |
| No background music | `assets/music/` is empty | Drop an `.mp3` in `assets/music/`, or set `PREFER_FAL_MUSIC=True` with fal credit |
| Action scenes are Ken Burns, not true motion | AnimateDiff disabled (8 GB ceiling) | Top up **fal** → Kling image-to-video (already wired), enable **ModelScope** (see below), or enable AnimateDiff on a bigger GPU |
| Cloud visuals unavailable now | **fal balance exhausted**, Pollinations 402 | Top up fal at fal.ai/dashboard/billing |
| App labels say "AnimateDiff/SDXL" | Cosmetic, pre-redesign copy | Update the Step-5 info text + sidebar in `app.py` |

### Enabling ModelScope text-to-video (local action motion)
`damo-vilab/text-to-video-ms-1.7b` is wired as an opt-in action engine (`pipeline/modelscope_generator.py`).
It is **text-to-video** — it generates frames from the scene prompt and **ignores the curated still** —
and outputs **256×256, watermarked** clips, so it is generally *softer/lower quality than Ken Burns on a
good Flux/SD still*. It exists for users who want real generated motion without cloud credit.

To use it:
1. Set `USE_MODELSCOPE = True` in `config.py` (optionally also `USE_CLOUD_PROVIDERS=False` to skip Kling).
2. First run downloads ~3.5 GB to `D:\models\sd`.
3. Action scenes (S2/S3) will log `Scene N OK (ModelScope text-to-video)`; speaking scenes are unaffected.
4. Tunables in `modelscope_generator.py`: `MS_FRAMES` (default 16), `MS_STEPS` (25), `MS_SIZE` (256).
   If you OOM on the 8 GB card, drop `MS_FRAMES` to 8. Quick single-scene preview: `python test_modelscope_scene.py`.
On any error it falls back to Ken Burns, so the reel never breaks.

> **RAM requirement (verified the hard way):** loading the 1.7B model needs roughly **8–10 GB of
> *free system RAM*** during init. On this machine with only ~5.8 GB free, the OS **killed the load
> process** (silent exit, no Python error) — and HF's xet downloader also panicked for the same reason
> (`config.py` now sets `HF_HUB_DISABLE_XET=1`). The code is correct and the weights download fine; to
> actually run ModelScope, **close other apps to free ~10 GB RAM**, then retry `test_modelscope_scene.py`
> (weights are cached, so no re-download). Otherwise the action cascade simply uses Ken Burns.

---

## 8. Running & troubleshooting

**Headless:** edit story/style/tone/voice constants in `generate_video.py`, then
`D:\round2_venv\Scripts\python.exe generate_video.py` → writes `output.mp4` (root) and
`outputs/session_<ts>/final/output.mp4`.

**UI:** `D:\round2_venv\Scripts\streamlit.exe run app.py` → http://localhost:8501

**Ollama (needed only for non-hardcoded stories):** start the bundled server with
`& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve`. The model `llama3.2:3b` lives in the
**default** location (`~/.ollama/models`) — do **not** set `OLLAMA_MODELS=D:\models\ollama` (that dir is
empty; the server will report `{"models":[]}`). Confirm with `curl http://localhost:11434/api/tags`.

**Stock footage (Pexels):** put `PEXELS_API_KEY=` in `.env` (free at pexels.com/api). With it set,
every scene logs `Scene N OK (Pexels stock footage)`; without it, scenes fall back to Ken Burns.

**Resume / partial re-render:** set `RESUME_SESSION="session_<ts>"` and `REUSE_CACHED_CLIPS=True` in
`generate_video.py`; delete only the `clips/scene_XX.mp4` (and/or `images/scene_XX.png`) you want
regenerated — cached files are reused via per-file existence checks.

**Common signals in the log:**
- `[fal] ... Exhausted balance` / `403` → fal out of credit (expected now) → falls back to local.
- `[Image] ... Pollinations 402` → free Flux quota hit → falls back to SD 1.5.
- `[SadTalker] inference failed ... retrying with fallback source` → first source had no detectable
  face; it retried the clean reference.
- `[Audio] All TTS attempts failed, using silent audio for: ...` → usually means that scene's
  **`voiceover_text` was empty** (llama3.2:3b sometimes omits voiceovers). Now prevented upstream: the
  script generator **rejects + retries** Ollama scripts with empty/short voiceovers and falls back to the
  template if all attempts fail; the normalizer **back-fills** any empty voiceover. Genuine *transient*
  edge-tts failures are retried in **4 rounds** with backoff, and `generate_all_voiceovers` **self-heals**
  on re-run (a timings-less `voice_XX.mp3` is regenerated). An ElevenLabs key avoids edge-tts entirely.
- `[Script] Attempt N: ... empty/too-short voiceover — retrying` / `Ollama failed, using template
  script` → expected behavior when the local 3B model returns a weak/invalid script; the template
  guarantees non-silent scenes. For better custom-story scripts, run a stronger LLM.
- `OSError: [WinError 6] The handle is invalid` at the very end → harmless MoviePy reader GC teardown.

**Quick checks:**
- Word timings present: `outputs/<session>/audio/voice_01.json` is a non-empty `{word,start,end}` list.
- Final video: `ffprobe output.mp4` → 1080×1920, ~38 s, h264 + aac.

---

## 9. Future / agentic direction (unchanged from original intent)

The modular step functions make a `QualityAgent` retry loop straightforward: after each stage, score
the output (face visible? style consistent? duration right?) and **approve / retry-with-refined-prompt /
escalate**. With that loop plus a job queue, the pipeline becomes bulk overnight production. The highest-
leverage upgrades are **character LoRA / Flux character-ref** (perfect identity) and **ElevenLabs voice
clone** (branded narration).
