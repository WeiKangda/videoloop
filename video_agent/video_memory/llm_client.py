# Modified from Qwen-MM-Plugins for VideoLoop; see README.md in this folder.
"""Model calls for the graph-memory build pipeline, through VideoLoop's model route, and JSON extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..llm_api import openai_completion


@dataclass
class BuildSettings:
    """The model and the frames of a build. Upstream's values, except where the model forces a change."""

    model: str
    api_base: str = ""
    api_key: str = ""
    effort: str = ""
    temperature: float = 0.7  # Dropped on the way out by llm.omit_temperature
    max_output_tokens: int = 65536
    timeout: float = 1200
    max_retries: int = 8
    fps: float = 0.5
    max_dim: int = 720  # Frame width in pixels
    max_frames: int = 100  # Upstream sends up to 250; Claude takes at most 100 images in a request
    max_mb: float | None = None  # Drop frames from a request above this many MB of base64


def extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating ```json fences and surrounding prose."""
    text = text.strip()
    think_end = text.find("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    if not text.startswith(("{", "[")):
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                text = text[i:]
                break
    bracket = "]" if text.startswith("[") else "}"
    idx = text.rfind(bracket)
    if idx != -1:
        text = text[: idx + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[error] JSON parse failed: {text[:500]}")
        raise


def call_model(
    prompt: str,
    settings: BuildSettings,
    frame_items: list[tuple[str, str]] | None = None,
    usage_tag: str = "memory_build",
) -> str:
    """Call the model with base64 frames or plain text, and return its text."""
    content: list[dict] = []
    if frame_items:
        ts_lines = [f"Frame {i + 1}: {label}" for i, (_, label) in enumerate(frame_items)]
        ts_block = "Frame timestamps, relative to the start of this segment (corresponding to frames in order):\n" + "\n".join(ts_lines)
        for url, _ in frame_items:
            content.append({"type": "image_url", "image_url": {"url": url}})
        prompt = f"{ts_block}\n\n{prompt}"
    content.append({"type": "text", "text": prompt})
    result = openai_completion(
        model=settings.model,
        messages=[{"role": "user", "content": content if frame_items else prompt}],
        api_base=settings.api_base,
        api_key=settings.api_key,
        max_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        reasoning_effort=settings.effort or None,
        usage_tag=usage_tag,
    )
    if result.get("finish_reason") == "length":
        print(f"[warning] the reply stopped at max_tokens={settings.max_output_tokens}")
    return result.get("content", "")
