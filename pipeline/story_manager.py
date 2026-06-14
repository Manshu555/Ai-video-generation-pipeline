import json
from pathlib import Path
from typing import Optional
from config import STORIES_FILE


def load_stories() -> list[dict]:
    with open(STORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_story_by_id(story_id: int) -> Optional[dict]:
    for story in load_stories():
        if story["id"] == story_id:
            return story
    return None


def get_story_titles() -> list[tuple[int, str]]:
    """Return list of (id, display_title) for UI dropdowns."""
    return [(s["id"], f"#{s['id']}: {s['title']}") for s in load_stories()]


def get_story_display(story: dict) -> str:
    """Human-readable story card for UI preview."""
    return (
        f"**Hook:** {story['hook']}\n\n"
        f"**Core Lesson:** {story['core_lesson']}\n\n"
        f"**Target Audience:** {story['target_audience']}\n\n"
        f"**Emotional Arc:** {story['emotional_arc']}"
    )
