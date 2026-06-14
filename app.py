"""
AI Viral Reel Video Generation Pipeline
8-step Streamlit wizard: Story → Style → Script → Images → Clips → Audio → Assemble → Export
"""
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import torch as _torch
    _CUDA_OK = _torch.cuda.is_available()
except ImportError:
    _CUDA_OK = False

from config import (
    STYLE_PRESETS,
    TONE_OPTIONS,
    TTS_VOICES,
    DEFAULT_STYLE,
    DEFAULT_TONE,
    DEFAULT_TTS_VOICE,
    OUTPUTS_DIR,
)

GEMINI_API_KEY = ""   # not used (Ollama is primary)
FAL_KEY = ""          # fal.ai balance exhausted, using local SDXL
from pipeline.story_manager import load_stories, get_story_display, get_story_titles

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Reel Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOTAL_STEPS = 8
STEP_LABELS = [
    "Story", "Style", "Script", "Images", "Clips", "Audio", "Assemble", "Export"
]


# ── Session state helpers ─────────────────────────────────────────────────────
def init_session():
    defaults = {
        "step": 1,
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "story": None,
        "tone": DEFAULT_TONE,
        "style_key": DEFAULT_STYLE,
        "scenes": None,
        "image_paths": None,
        "clip_paths": None,
        "voice_paths": None,
        "music_path": None,
        "voice_id": DEFAULT_TTS_VOICE,
        "final_path": None,
        "drive_link": None,
        "regen_scene": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def session_dir() -> Path:
    path = OUTPUTS_DIR / st.session_state["session_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def go_to(step: int):
    st.session_state["step"] = step


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🎬 Reel Pipeline")
        st.caption("AI-generated viral video workflow")
        st.divider()

        current = st.session_state["step"]
        for i, label in enumerate(STEP_LABELS, 1):
            icon = "✅" if i < current else ("▶️" if i == current else "⬜")
            st.markdown(f"{icon} **Step {i}:** {label}")

        st.divider()
        st.markdown("**Pipeline Status**")
        st.markdown(f"{'✅' if _CUDA_OK else '⚠️'} GPU ({'RTX 4060 ready' if _CUDA_OK else 'Install CUDA PyTorch'})")
        st.markdown("✅ Ollama LLM (script gen)")
        st.markdown("✅ Local SDXL (image gen)")
        st.divider()

        if st.session_state.get("story"):
            st.markdown("**Current Session**")
            story = st.session_state["story"]
            st.caption(f"Story: {story['title']}")
            style = STYLE_PRESETS[st.session_state["style_key"]]
            st.caption(f"Style: {style['label']}")

        if st.button("🔄 Start New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Step 1: Story Selection ───────────────────────────────────────────────────
def step_story():
    st.header("Step 1: Choose Your Story")
    st.caption("Select a pre-written viral business story or write your own hook below.")

    stories = load_stories()
    titles = [f"#{s['id']}: {s['title']}" for s in stories]
    titles.append("Custom Story")

    col1, col2 = st.columns([1, 2])

    with col1:
        choice = st.radio("Select a story:", titles, key="story_choice")

    with col2:
        if choice != "Custom Story":
            idx = titles.index(choice)
            story = stories[idx]
            st.info(get_story_display(story))
        else:
            story = None
            st.markdown("**Write your custom story:**")

    if choice == "Custom Story":
        with st.form("custom_story_form"):
            title = st.text_input("Story Title", placeholder="The Day I Lost Everything")
            hook = st.text_area("Hook (1-2 sentences that grab attention)", height=80,
                                placeholder="He built a $10M company... then lost it all in 90 days.")
            raw_story = st.text_area("Full Story (background, conflict, resolution)", height=200)
            lesson = st.text_input("Core Lesson", placeholder="Speed beats perfection. Always.")
            audience = st.text_input("Target Audience", placeholder="startup founders, entrepreneurs")
            arc = st.text_input("Emotional Arc", placeholder="shock → struggle → breakthrough")

            if st.form_submit_button("Use This Story", type="primary"):
                if title and hook and raw_story and lesson:
                    st.session_state["story"] = {
                        "id": 0, "title": title, "hook": hook,
                        "raw_story": raw_story, "core_lesson": lesson,
                        "target_audience": audience or "general business audience",
                        "emotional_arc": arc or "failure → lesson → hope",
                        "tags": [],
                    }
                    go_to(2)
                    st.rerun()
                else:
                    st.error("Please fill in all required fields.")
    else:
        if st.button("Use This Story →", type="primary", key="use_story"):
            idx = titles.index(choice)
            st.session_state["story"] = stories[idx]
            go_to(2)
            st.rerun()


# ── Step 2: Style & Tone ──────────────────────────────────────────────────────
def step_style():
    st.header("Step 2: Choose Visual Style & Tone")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visual Style")
        style_options = {k: v["label"] for k, v in STYLE_PRESETS.items()}
        style_key = st.radio(
            "Pick the animation style for your video:",
            list(style_options.keys()),
            format_func=lambda k: style_options[k],
            key="style_selector",
            index=list(style_options.keys()).index(st.session_state["style_key"]),
        )
        style = STYLE_PRESETS[style_key]
        st.info(f"**{style['label']}**: {style['description']}")

    with col2:
        st.subheader("Narrative Tone")
        tone_key = st.radio(
            "How should the story be narrated?",
            list(TONE_OPTIONS.keys()),
            format_func=lambda k: k.title(),
            key="tone_selector",
            index=list(TONE_OPTIONS.keys()).index(st.session_state["tone"]),
        )
        st.info(TONE_OPTIONS[tone_key])

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="back2"):
            go_to(1); st.rerun()
    with col_next:
        if st.button("Generate Script →", type="primary", key="next2"):
            st.session_state["style_key"] = style_key
            st.session_state["tone"] = tone_key
            go_to(3)
            st.rerun()


# ── Step 3: Script Generation ─────────────────────────────────────────────────
def step_script():
    st.header("Step 3: Generate & Review Script")

    story = st.session_state["story"]
    style = STYLE_PRESETS[st.session_state["style_key"]]
    tone = st.session_state["tone"]

    if not st.session_state.get("scenes"):
        st.info(f"Generating 6-7 scene script for **{story['title']}** in {style['label']} style...")

        with st.spinner("Writing your reel script with local Ollama LLM..."):
            from pipeline.script_generator import generate_script
            try:
                scenes = generate_script(story, tone, style, session_dir())
                st.session_state["scenes"] = scenes
            except Exception as e:
                st.error(f"Script generation failed: {e}")
                return

    scenes = st.session_state["scenes"]
    st.success(f"Script ready! {len(scenes)} scenes, ~{sum(s['duration_seconds'] for s in scenes)}s total")

    for i, scene in enumerate(scenes):
        with st.expander(f"Scene {scene['scene_number']}: {scene.get('text_overlay', '')} ({scene['duration_seconds']}s)"):
            col1, col2 = st.columns(2)
            with col1:
                new_visual = st.text_area(
                    "Visual Description",
                    value=scene["visual_description"],
                    key=f"visual_{i}",
                    height=100,
                )
                new_overlay = st.text_input("Text Overlay", value=scene.get("text_overlay", ""), key=f"overlay_{i}")
            with col2:
                new_voice = st.text_area(
                    "Voiceover Text",
                    value=scene.get("voiceover_text", ""),
                    key=f"voice_{i}",
                    height=100,
                )
                new_motion = st.selectbox(
                    "Camera Motion",
                    ["slow push-in", "pan left", "pan right", "tilt up", "tilt down", "static hold", "zoom out"],
                    index=["slow push-in","pan left","pan right","tilt up","tilt down","static hold","zoom out"].index(
                        scene.get("camera_motion", "static hold")
                    ),
                    key=f"motion_{i}",
                )
            scenes[i].update({
                "visual_description": new_visual,
                "voiceover_text": new_voice,
                "text_overlay": new_overlay,
                "camera_motion": new_motion,
            })

    # Save edits
    with open(session_dir() / "script.json", "w") as f:
        json.dump(scenes, f, indent=2)
    st.session_state["scenes"] = scenes

    st.divider()
    col_back, col_regen, col_next = st.columns(3)
    with col_back:
        if st.button("← Back", key="back3"):
            go_to(2); st.rerun()
    with col_regen:
        if st.button("🔄 Regenerate Script", key="regen3"):
            st.session_state["scenes"] = None
            st.session_state["image_paths"] = None
            st.rerun()
    with col_next:
        if st.button("Generate Images →", type="primary", key="next3"):
            go_to(4); st.rerun()


# ── Step 4: Image Generation ──────────────────────────────────────────────────
def step_images():
    st.header("Step 4: Generate Scene Illustrations")
    st.caption("Each scene gets a custom illustrated still with your character.")

    scenes = st.session_state["scenes"]
    style = STYLE_PRESETS[st.session_state["style_key"]]
    sdir = session_dir()

    if not st.session_state.get("image_paths"):
        _gpu_msg = "GPU (RTX 4060)" if _CUDA_OK else "CPU (slow — CUDA PyTorch not yet installed)"
        st.info(f"Generating images with local SDXL on {_gpu_msg}. First scene loads the model (~30s).")
        progress = st.progress(0, text="Starting image generation...")
        status_area = st.empty()
        paths = []

        from pipeline.image_generator import generate_scene_image
        for i, scene in enumerate(scenes):
            status_area.info(f"Generating image for scene {scene['scene_number']}/{len(scenes)}...")
            try:
                path = generate_scene_image(scene, style, sdir)
                paths.append(path)
            except Exception as e:
                st.warning(f"Scene {scene['scene_number']} image failed: {e}")
                paths.append(None)
            progress.progress((i + 1) / len(scenes), text=f"Scene {i+1}/{len(scenes)} done")

        st.session_state["image_paths"] = [str(p) if p else None for p in paths]
        progress.empty()
        status_area.empty()

    image_paths = [Path(p) if p else None for p in st.session_state["image_paths"]]

    st.success("All scene images generated!")
    cols = st.columns(min(len(scenes), 4))
    for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
        col = cols[i % 4]
        with col:
            if img_path and img_path.exists():
                st.image(str(img_path), caption=f"Scene {scene['scene_number']}: {scene.get('text_overlay','')}", use_container_width=True)
            else:
                st.markdown(f"Scene {scene['scene_number']}: *generation failed*")

            if st.button(f"🔄 Redo Scene {scene['scene_number']}", key=f"redo_img_{i}"):
                if img_path and img_path.exists():
                    img_path.unlink()
                # Mark for regeneration
                paths_list = list(st.session_state["image_paths"])
                paths_list[i] = None
                st.session_state["image_paths"] = paths_list
                st.session_state["clip_paths"] = None
                st.rerun()

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="back4"):
            go_to(3); st.rerun()
    with col_next:
        if st.button("Animate Scenes →", type="primary", key="next4"):
            go_to(5); st.rerun()


# ── Step 5: Video Generation ──────────────────────────────────────────────────
def step_clips():
    st.header("Step 5: Animate Each Scene")
    st.caption("Converting illustrated stills into 8-10 second video clips.")

    scenes = st.session_state["scenes"]
    image_paths = [Path(p) if p else None for p in (st.session_state.get("image_paths") or [])]
    style = STYLE_PRESETS[st.session_state["style_key"]]
    sdir = session_dir()

    st.info("Speaking scenes → SadTalker lip-sync · action scenes → AnimateDiff motion · "
            "Ken Burns fallback. GPU (NVENC) encoding.")

    if not st.session_state.get("clip_paths"):
        # Speaking scenes need their voiceover first (SadTalker is audio-driven).
        from pipeline.audio_generator import (
            generate_all_voiceovers, scene_duration_from_voice, get_audio_duration,
        )
        voice_paths = st.session_state.get("voice_paths")
        if not voice_paths:
            with st.spinner("Generating voiceovers (needed for lip-sync)..."):
                voice_paths = [str(p) for p in generate_all_voiceovers(
                    scenes, st.session_state["voice_id"], sdir)]
                st.session_state["voice_paths"] = voice_paths
        vpaths = [Path(p) if p else None for p in voice_paths]

        target_durations = []
        for sc, vp in zip(scenes, vpaths):
            if sc.get("shot_type") == "speaking":
                target_durations.append(get_audio_duration(vp) or scene_duration_from_voice(vp))
            else:
                target_durations.append(scene_duration_from_voice(vp))

        progress = st.progress(0, text="Starting animation...")
        status_area = st.empty()
        paths = []

        from pipeline.video_generator import animate_scene
        for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
            kind = "talking head (SadTalker)" if scene.get("shot_type") == "speaking" else "motion (AnimateDiff)"
            status_area.info(f"Animating scene {scene['scene_number']}/{len(scenes)} — {kind}... "
                             f"(first run loads the model; can take a few minutes)")
            try:
                path = animate_scene(img_path, scene, sdir,
                                     target_duration=target_durations[i],
                                     style=style, voice_path=vpaths[i])
                paths.append(str(path))
            except Exception as e:
                st.warning(f"Scene {scene['scene_number']} animation failed: {e}")
                paths.append(None)
            progress.progress((i + 1) / len(scenes), text=f"Scene {i+1}/{len(scenes)} animated")

        st.session_state["clip_paths"] = paths
        progress.empty()
        status_area.empty()

    clip_paths = [Path(p) if p else None for p in st.session_state["clip_paths"]]

    st.success("All scenes animated!")
    cols = st.columns(min(len(scenes), 3))
    for i, (scene, clip_path) in enumerate(zip(scenes, clip_paths)):
        col = cols[i % 3]
        with col:
            if clip_path and clip_path.exists():
                st.video(str(clip_path))
                st.caption(f"Scene {scene['scene_number']}")
            else:
                st.markdown(f"Scene {scene['scene_number']}: *failed*")

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="back5"):
            go_to(4); st.rerun()
    with col_next:
        if st.button("Configure Audio →", type="primary", key="next5"):
            go_to(6); st.rerun()


# ── Step 6: Audio Settings ────────────────────────────────────────────────────
def step_audio():
    st.header("Step 6: Generate Voiceover & Music")

    scenes = st.session_state["scenes"]
    sdir = session_dir()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Voice Selection")
        voice_label = st.selectbox(
            "Choose narrator voice:",
            list(TTS_VOICES.keys()),
            index=list(TTS_VOICES.values()).index(st.session_state["voice_id"])
            if st.session_state["voice_id"] in TTS_VOICES.values() else 0,
        )
        selected_voice = TTS_VOICES[voice_label]

    with col2:
        st.subheader("Background Music")
        from pipeline.audio_generator import get_available_music_tracks
        music_options = get_available_music_tracks()
        music_choice = st.selectbox("Background music track:", ["No music"] + [k for k in music_options if k != "No music"])
        music_path = music_options.get(music_choice)

    if st.button("Generate Voiceovers", type="primary", key="gen_audio"):
        st.session_state["voice_id"] = selected_voice
        st.session_state["music_path"] = str(music_path) if music_path else None

        progress = st.progress(0)
        from pipeline.audio_generator import generate_voiceover
        paths = []
        for i, scene in enumerate(scenes):
            out_path = sdir / "audio" / f"voice_{scene['scene_number']:02d}.mp3"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                generate_voiceover(scene.get("voiceover_text", ""), selected_voice, out_path)
                paths.append(str(out_path))
            except Exception as e:
                st.warning(f"Voice gen scene {scene['scene_number']}: {e}")
                paths.append(None)
            progress.progress((i + 1) / len(scenes))

        st.session_state["voice_paths"] = paths
        st.success("Voiceovers ready!")

        # Preview first clip
        if paths and paths[0]:
            st.audio(paths[0])

    if st.session_state.get("voice_paths"):
        st.info("Voiceovers already generated. Re-generate above to change voice.")
        for i, (scene, vp) in enumerate(zip(scenes, st.session_state["voice_paths"])):
            if vp and Path(vp).exists():
                with st.expander(f"Scene {scene['scene_number']} preview"):
                    st.audio(vp)
                    st.caption(scene.get("voiceover_text", ""))

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="back6"):
            go_to(5); st.rerun()
    with col_next:
        ready = bool(st.session_state.get("voice_paths"))
        if st.button("Assemble Reel →", type="primary", key="next6", disabled=not ready):
            st.session_state["music_path"] = str(music_path) if music_path else None
            go_to(7); st.rerun()


# ── Step 7: Assembly ──────────────────────────────────────────────────────────
def step_assemble():
    st.header("Step 7: Assemble Final Reel")
    st.caption("Combining all clips, audio, captions and transitions into one 60-second video.")

    if not st.session_state.get("final_path"):
        scenes = st.session_state["scenes"]
        clip_paths = [Path(p) for p in st.session_state["clip_paths"] if p]
        voice_paths = [Path(p) if p else None for p in st.session_state.get("voice_paths", [])]
        music_path = Path(st.session_state["music_path"]) if st.session_state.get("music_path") else None
        sdir = session_dir()

        story = st.session_state["story"]
        safe_title = "".join(c for c in story["title"] if c.isalnum() or c in " _-")[:40]
        output_filename = f"{safe_title.replace(' ', '_')}_reel.mp4"

        with st.spinner("Assembling your reel... this takes 1-3 minutes"):
            from pipeline.assembler import assemble_reel
            try:
                final_path = assemble_reel(
                    clip_paths=clip_paths,
                    voice_paths=voice_paths,
                    scenes=scenes,
                    session_dir=sdir,
                    music_path=music_path,
                    output_filename=output_filename,
                )
                st.session_state["final_path"] = str(final_path)
            except Exception as e:
                st.error(f"Assembly failed: {e}")
                st.exception(e)
                return

    final_path = Path(st.session_state["final_path"])
    st.success(f"Reel assembled successfully!")
    st.video(str(final_path))

    duration_info = f"File: {final_path.stat().st_size / 1024 / 1024:.1f} MB"
    st.caption(duration_info)

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="back7"):
            go_to(6); st.rerun()
    with col_next:
        if st.button("Export & Upload →", type="primary", key="next7"):
            go_to(8); st.rerun()


# ── Step 8: Export ────────────────────────────────────────────────────────────
def step_export():
    st.header("Step 8: Export & Upload")
    st.success("Your reel is ready!")

    final_path = Path(st.session_state["final_path"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Download")
        with open(final_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Final Reel (MP4)",
                data=f,
                file_name=final_path.name,
                mime="video/mp4",
                use_container_width=True,
            )

        st.video(str(final_path))

    with col2:
        st.subheader("Upload to Google Drive")
        if st.session_state.get("drive_link"):
            st.success(f"Already uploaded!")
            st.markdown(f"[Open in Google Drive]({st.session_state['drive_link']})")
        else:
            folder_name = st.text_input("Drive folder name", value="Viral Reel Videos")
            if st.button("Upload to Google Drive", use_container_width=True):
                with st.spinner("Uploading..."):
                    from pipeline.drive_uploader import upload_to_drive
                    link = upload_to_drive(final_path, folder_name)
                    if link:
                        st.session_state["drive_link"] = link
                        st.success("Uploaded!")
                        st.markdown(f"[Open in Google Drive]({link})")
                    else:
                        st.error("Upload failed. Check credentials.json is present.")

    st.divider()
    st.subheader("Make Another Video")
    story = st.session_state["story"]
    style = STYLE_PRESETS[st.session_state["style_key"]]
    st.info(f"Current: **{story['title']}** | Style: **{style['label']}**")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎬 New Story, Same Style", use_container_width=True):
            # Keep style, reset story/content
            for k in ["story","scenes","image_paths","clip_paths","voice_paths","final_path","drive_link"]:
                st.session_state[k] = None
            st.session_state["session_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            go_to(1); st.rerun()
    with col_b:
        if st.button("🔄 Full Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Main router ───────────────────────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()

    current_step = st.session_state["step"]

    # Progress bar
    st.progress(current_step / TOTAL_STEPS, text=f"Step {current_step} of {TOTAL_STEPS}: {STEP_LABELS[current_step-1]}")
    st.divider()

    step_funcs = {
        1: step_story,
        2: step_style,
        3: step_script,
        4: step_images,
        5: step_clips,
        6: step_audio,
        7: step_assemble,
        8: step_export,
    }
    step_funcs[current_step]()


if __name__ == "__main__":
    main()
