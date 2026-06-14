"""
Generates a structured 6-7 scene script using local Ollama (llama3.2:3b).
Falls back to a template-based script if Ollama is unavailable.
"""
import json
import re
import time
from pathlib import Path

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, CHARACTER_DESCRIPTION, CLIP_DURATION_TARGET

SYSTEM_PROMPT = """You are an expert viral short-form video scriptwriter for social media reels.
Turn the given business story into a punchy ~30-second vertical reel script with exactly 5 scenes.

STRICT RULES:
- Each voiceover_text MUST be 15-22 words (NEVER fewer than 15). Complete, vivid sentences — short lines leave dead air.
- Every scene has a "shot_type": either "speaking" or "action".
    * "speaking" = the character talks straight to camera. Use for scene 1 (the HOOK) and the final 1-2 scenes (the LESSON / CALL TO ACTION).
       For speaking scenes, visual_description MUST be a frontal CLOSE-UP PORTRAIT of {character_description} looking directly at the camera, talking, head and shoulders framed, clean simple background.
    * "action" = the character is DOING something (an action/gesture/movement). Use for the middle conflict/escalation scenes.
       For action scenes, visual_description shows {character_description} mid-action in the scene environment.
- The character ({character_description}) MUST appear in every scene.
- camera_motion: pick one of: slow push-in | pan left | pan right | tilt up | static hold | zoom out
- text_overlay: 3-5 words, ALL CAPS, punchy
- stock_query: a 3-6 word literal visual phrase to search STOCK FOOTAGE for this scene's B-roll.
    Use concrete, filmable nouns/scenes (e.g. "busy restaurant kitchen night", "empty office at dusk",
    "hands typing laptop closeup"). NO character names, NO abstract concepts, NO text.
- Emotional arc across the 5 scenes: hook -> conflict -> escalation -> lesson -> payoff/CTA
- Output ONLY a valid JSON array. No markdown. No explanation. Nothing else.

JSON format:
[{{"scene_number":1,"duration_seconds":6,"shot_type":"speaking","visual_description":"...","voiceover_text":"...","stock_query":"busy city street morning","camera_motion":"slow push-in","text_overlay":"THE HOOK","emotion":"tension"}},...]"""

USER_PROMPT = """Story: {title}
Hook: {hook}
Background: {raw_story}
Core lesson: {core_lesson}
Tone: {tone} | Style: {style_label}

Output the 5-scene JSON array now."""


def _call_ollama(system: str, user: str) -> str:
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_PORTRAIT = (
    f"frontal close-up portrait of {CHARACTER_DESCRIPTION} looking directly at the camera, "
    f"talking, head and shoulders framed, clean simple background, soft key light"
)


# ── Hand-directed screenplay for Story 1: "The Dishwasher Steak Heist" ─────────
# Ollama (llama3.2:3b) fails to return valid JSON almost every run and falls back
# to a generic template that ignores the actual story. This hand-directed script
# replaces that path for story 1 — story-accurate, cinematically blocked, with a
# cold→warm emotional/color arc and a silent hero-text climax (scene 4).
# Each visual_description is the scene-specific part; image_generator wraps it with
# the style prefix + character description. Scenes flagged no_character omit the
# character from the prompt (empty-kitchen shot).
_DISHWASHER_SCRIPT = [
    {
        "scene_number": 1, "shot_type": "speaking", "emotion": "tension",
        "stock_query": "empty upscale restaurant interior evening",
        "duration_seconds": 5.5, "camera_motion": "slow push-in",
        "text_overlay": "THE HEIST", "hero_text": False, "music_attenuation": -28.0,
        "voiceover_text": "He built a successful construction company. Then he opened a restaurant. It was gone in eight months.",
        "visual_description": ("looking directly at the camera with quiet regret, warm amber "
                               "restaurant candlelight from below, head-and-shoulders close-up, "
                               "shallow depth of field, talking to camera"),
        "motion_prompt": "",
    },
    {
        "scene_number": 2, "shot_type": "action", "emotion": "shock", "no_character": True,
        "stock_query": "commercial restaurant kitchen night chef",
        "duration_seconds": 6.0, "camera_motion": "pan left",
        "text_overlay": "THE THEFT", "hero_text": False, "music_attenuation": -28.0,
        "voiceover_text": "Every night, his dishwasher threw premium steaks into the dumpster, to sneak back and steal them. Bob never knew.",
        # NOTE: no color/grade words here (they collapse weak SD 1.5 into textures);
        # the cold-blue mood is applied in post via COLOR_GRADES["shock"].
        "visual_description": ("interior of a commercial restaurant kitchen at night, stainless steel "
                               "counters, raw premium steaks on a tray, an open back door to a dark "
                               "alley with a large metal dumpster, overhead fluorescent lighting, "
                               "cinematic wide establishing shot, empty, no people"),
        "motion_prompt": "slow cinematic pan left across the kitchen toward the dark exit door, subtle steam drift, ominous still dread",
    },
    {
        "scene_number": 3, "shot_type": "action", "emotion": "dramatic",
        "stock_query": "empty restaurant chairs on tables closed",
        "duration_seconds": 5.5, "camera_motion": "zoom out",
        "text_overlay": "THE COST", "hero_text": False, "music_attenuation": -28.0,
        "voiceover_text": "He was buried in the numbers. Too proud to truly know his own people. So he never saw it coming.",
        # NOTE: mood (desaturated grey) applied in post via COLOR_GRADES["dramatic"].
        "visual_description": ("standing alone in the middle of an empty closed restaurant at night, "
                               "dining chairs turned upside-down on the tables around him, one warm "
                               "overhead light above, long shadows, cinematic medium-wide shot"),
        "motion_prompt": "very slow camera pull back revealing the empty restaurant, the lone figure small and still, melancholic",
    },
    {
        "scene_number": 4, "shot_type": "speaking", "emotion": "revelation",
        "stock_query": "lone man silhouette window rain night moody",
        "duration_seconds": 4.5, "camera_motion": "static hold",
        "text_overlay": "", "hero_text": True, "music_attenuation": -50.0,
        "hero_line1": "HE DIDN'T HAVE A", "hero_line2": "EGO PROBLEM.",
        "voiceover_text": "He didn't have a restaurant problem. He had an ego problem.",
        "visual_description": ("extreme close-up of the face, Rembrandt lighting with one side warmly "
                               "lit and deep shadow on the other, intense direct eye contact, wisdom "
                               "and hard-won self-awareness, plain dark background"),
        "motion_prompt": "",
    },
    {
        "scene_number": 5, "shot_type": "speaking", "emotion": "triumph",
        "stock_query": "busy fine dining restaurant happy diners",
        "duration_seconds": 5.8, "camera_motion": "slow push-in",
        "text_overlay": "THE COMEBACK", "hero_text": False, "music_attenuation": -22.0,
        "voiceover_text": "He read every book on hospitality. He listened. His next restaurant earned rave reviews around the world.",
        "visual_description": ("warm golden upscale restaurant interior behind him, candlelight bokeh, "
                               "calm confident smile, looking at camera with warmth and wisdom, "
                               "medium portrait"),
        "motion_prompt": "",
    },
]


def _normalize(scenes: list[dict], story: dict) -> list[dict]:
    """Harden scene dicts so a missing field can never crash the pipeline or UI."""
    scenes = list(scenes)[:5]
    n = len(scenes)
    for i, scene in enumerate(scenes):
        scene["scene_number"] = i + 1
        # voiceover_text drives narration AND captions — it must never be empty,
        # or the scene plays as silent B-roll with no captions. Back-fill from the
        # story if the LLM left it blank (last-resort; the Ollama path also retries).
        scene["voiceover_text"] = (scene.get("voiceover_text")
                                   or scene.get("voiceover") or "").strip()
        if not scene["voiceover_text"]:
            scene["voiceover_text"] = (
                story.get("core_lesson") or story.get("hook")
                or scene.get("visual_description") or "").strip()
        scene.setdefault("duration_seconds", 6)
        scene.setdefault("camera_motion", "static hold")
        scene.setdefault("text_overlay", "")
        scene.setdefault("emotion", "neutral")
        scene.setdefault("hero_text", False)
        scene.setdefault("music_attenuation", -28.0)
        scene.setdefault("motion_prompt", "")
        # Stock-footage search phrase for the faceless B-roll flow; derive from the
        # visual description if the LLM omitted it.
        if not scene.get("stock_query"):
            scene["stock_query"] = (scene.get("visual_description")
                                    or scene.get("text_overlay") or "").strip()[:80]
        # Back-fill shot_type if missing: scene 1 + last two = speaking
        if scene.get("shot_type") not in ("speaking", "action"):
            scene["shot_type"] = "speaking" if (i == 0 or i >= n - 2) else "action"
        if not scene.get("visual_description"):
            scene["visual_description"] = (
                _PORTRAIT if scene["shot_type"] == "speaking"
                else f"{CHARACTER_DESCRIPTION} mid-action in the scene environment"
            )
        # Speaking scenes need a clean frontal portrait for the lip-sync engine
        if scene["shot_type"] == "speaking":
            vd = scene["visual_description"].lower()
            if not any(k in vd for k in ("portrait", "close-up", "close up", "looking", "face")):
                scene["visual_description"] = _PORTRAIT
    return scenes


def _template_script(story: dict) -> list[dict]:
    """Deterministic 5-scene fallback script when Ollama is unavailable."""
    templates = [
        {"shot_type": "speaking", "visual_description": _PORTRAIT, "stock_query": "thoughtful entrepreneur looking out office window",
         "voiceover_text": story["hook"], "camera_motion": "slow push-in", "text_overlay": "THE HOOK", "emotion": "curiosity"},
        {"shot_type": "action", "visual_description": f"{CHARACTER_DESCRIPTION} at a cluttered desk, head in hands, papers scattered, dim office, visibly stressed", "stock_query": "stressed businessman cluttered desk dim office",
         "voiceover_text": "Everyone thought he had it all figured out. He was the very last one to realize that he didn't.", "camera_motion": "pan left", "text_overlay": "THE PROBLEM", "emotion": "tension"},
        {"shot_type": "action", "visual_description": f"{CHARACTER_DESCRIPTION} pacing an empty room, gesturing in thought, moody dramatic shadows around him", "stock_query": "person walking alone city street night",
         "voiceover_text": "Then came the single moment that changed everything for him. One hard conversation he almost didn't have.", "camera_motion": "static hold", "text_overlay": "THE TURN", "emotion": "dramatic"},
        {"shot_type": "speaking", "visual_description": _PORTRAIT, "stock_query": "sunrise over city skyline hopeful",
         "voiceover_text": story["core_lesson"], "camera_motion": "tilt up", "text_overlay": "THE LESSON", "emotion": "revelation"},
        {"shot_type": "speaking", "visual_description": _PORTRAIT, "stock_query": "diverse team celebrating success office",
         "voiceover_text": "If you are building something right now, save this video. You will absolutely need it one day.", "camera_motion": "slow push-in", "text_overlay": "SAVE THIS", "emotion": "connection"},
    ]
    return [{"scene_number": i + 1, "duration_seconds": 6, **t} for i, t in enumerate(templates)]


def generate_script(story: dict, tone: str, style: dict, session_dir: Path) -> list[dict]:
    # Story 1 ("The Dishwasher Steak Heist") uses a hand-directed screenplay —
    # Ollama is unreliable and its template fallback ignores the actual story.
    if story.get("id") == 1:
        print("[Script] Using hand-directed screenplay for 'The Dishwasher Steak Heist'")
        scenes = [dict(s) for s in _DISHWASHER_SCRIPT]
    else:
        system = SYSTEM_PROMPT.format(character_description=CHARACTER_DESCRIPTION)
        user = USER_PROMPT.format(
            title=story["title"],
            hook=story["hook"],
            raw_story=story["raw_story"][:800],  # Trim for context window
            core_lesson=story["core_lesson"],
            tone=tone,
            style_label=style["label"],
        )

        scenes = None
        for attempt in range(3):
            try:
                print(f"[Script] Calling Ollama {OLLAMA_MODEL} (attempt {attempt+1})...")
                raw = _call_ollama(system, user)

                # Strip markdown fences if present
                raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
                raw = re.sub(r"\n?```$", "", raw)

                # Extract JSON array
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    raw = match.group(0)

                parsed = json.loads(raw)
                # Accept only a well-formed script where EVERY scene has a real
                # voiceover (>=5 words). llama3.2:3b sometimes emits scenes with an
                # empty voiceover_text → silent B-roll with no captions. Reject &
                # retry rather than ship that.
                if isinstance(parsed, list) and len(parsed) >= 4:
                    weak = [s for s in parsed
                            if len((s.get("voiceover_text") or "").split()) < 5]
                    if weak:
                        print(f"[Script] Attempt {attempt+1}: {len(weak)} scene(s) had "
                              f"empty/too-short voiceover — retrying")
                        scenes = None
                    else:
                        scenes = parsed
                        print(f"[Script] Got {len(scenes)} scenes from Ollama")
                        break
            except Exception as e:
                print(f"[Script] Attempt {attempt+1} failed: {e}")
                time.sleep(3)

        if not scenes:
            print("[Script] Ollama failed, using template script")
            scenes = _template_script(story)

    scenes = _normalize(scenes, story)

    out_path = session_dir / "script.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)

    return scenes


def load_script(session_dir: Path) -> list[dict]:
    with open(session_dir / "script.json", encoding="utf-8") as f:
        return json.load(f)
