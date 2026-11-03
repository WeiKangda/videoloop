#!/usr/bin/env python3
"""Build the video memory of long videos ahead of a run. See docs/video_index_pilot.md.

  python scripts/build_video_memory.py --items data/items_test.jsonl --videos data/videos \
      --model MODEL --api-base URL

Each video gets <out>/<video stem>/ with:
  graph_memory.json, embeddings.npz  what the agent's memory tools read
  build_stats.json                   status, settings, time and tokens; score_videoindex.py reads it
  usage.jsonl                        one line per model call, including calls that failed
  01_macros.json, subgraphs/, ...    checkpoints, so that a stopped build carries on where it stopped

A finished video is skipped, so this can be stopped and started again with the same command.
The build gets the frames and the transcript of one video. It never reads the question, the options
or the answer of an item, only where its video is and how long it runs.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from run_videoindex import DEFAULT_TRANSCRIPTS, MEMORY_REQUIRED_FROM_S, ROOT, git_output, jsonl_rows, load_items, summarize_usage, write_json

DEFAULT_OUT = "dataset/videoindex_v1_1/video_memory"
MAX_FAILURES_IN_ROW = 2


def select_items(ap, a, items):
    if a.only:
        wanted = {x for x in a.only.split(",") if x}
        unknown = wanted - {i["item_id"] for i in items}
        if unknown:
            ap.error(f"--only: no such item: {sorted(unknown)[0]}")
        items = [i for i in items if i["item_id"] in wanted]
    else:
        items = [i for i in items if i["duration_s"] >= a.min_duration]
    if a.shard:
        index, _, count = a.shard.partition("/")
        items = items[int(index)::int(count)]
    audio_only = [i["item_id"] for i in items if i.get("width") is None]
    if audio_only:
        print(f"skipping {len(audio_only)} with no video stream: {', '.join(audio_only)}")
    return [i for i in items if i.get("width") is not None]


def video_name(item):
    return os.path.basename(item["video"])


def stem(item):
    return os.path.splitext(video_name(item))[0]


def transcript_of(a, item):
    """The transcript to give the build, or None for a video with no sound."""
    if a.no_asr or not item.get("audio"):
        return None
    return os.path.join(a.transcripts, f"{stem(item)}.json")


def read_stats(a, item):
    path = Path(a.out) / stem(item) / "build_stats.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_inputs(a, items, settings_key):
    problems = []
    for item in items:
        if not os.path.exists(os.path.join(a.videos, video_name(item))):
            problems.append(f"{item['item_id']}: video not found: {video_name(item)}")
        transcript = transcript_of(a, item)
        if transcript and not os.path.exists(transcript):
            problems.append(f"{item['item_id']}: no transcript in {a.transcripts}; run scripts/transcribe_videoindex.py, or pass --no-asr")
        before = read_stats(a, item).get("settings")
        if before and before != settings_key:
            changed = next(k for k in settings_key if before.get(k) != settings_key[k])
            problems.append(f"{item['item_id']}: {a.out} was built with a different {changed}; use a new --out")
    return problems


def build_one(a, item, settings, settings_key, cfg_commit):
    """Build one video, or carry on with it. Returns its build_stats.json."""
    from video_agent import llm_api
    from video_agent.video_memory.build_graph import _has_subgraph_content, build_graph_memory

    folder = Path(a.out) / stem(item)
    folder.mkdir(parents=True, exist_ok=True)
    previous = read_stats(a, item)
    lock = threading.Lock()
    usage_log = open(folder / "usage.jsonl", "a", encoding="utf-8", buffering=1)

    def sink(record):
        with lock:
            usage_log.write(json.dumps(record) + "\n")

    llm_api.set_usage_sink(sink)
    timings = dict(previous.get("phase_s") or {})
    started = time.time()
    memory = None
    error = "stopped"
    try:
        memory = build_graph_memory(
            os.path.join(a.videos, video_name(item)),
            settings,
            video_key=stem(item),
            output_dir=str(folder),
            concurrency=a.workers,
            min_scene_sec=a.min_scene_sec,
            max_scene_sec=a.max_scene_sec,
            transcript_path=transcript_of(a, item),
            timings=timings,
        )
        error = ""
    except Exception as e:
        error = str(e)[:500]
    finally:
        llm_api.set_usage_sink(None)
        usage_log.close()
        stats = {
            "status": "done" if memory else "failed",
            "error": error,
            "video": video_name(item),
            "duration_s": item["duration_s"],
            "settings": settings_key,
            "wall_s": round(previous.get("wall_s", 0) + time.time() - started, 1),
            "phase_s": timings,
            "usage": summarize_usage(jsonl_rows(folder / "usage.jsonl")),
            "commit": cfg_commit,
        }
        if memory:
            stats.update({
                "n_macro_events": len(memory.macro_events),
                "n_super_events": len(memory.super_events),
                "n_empty_subgraphs": sum(1 for m in memory.macro_events if not _has_subgraph_content(m)),
                "n_nodes": len(memory.get_all_nodes()),
            })
        write_json(folder / "build_stats.json", stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", required=True, help="folder with <sha>.mp4")
    ap.add_argument("--out", default=str(ROOT / DEFAULT_OUT))
    ap.add_argument("--transcripts", default=str(ROOT / DEFAULT_TRANSCRIPTS))
    ap.add_argument("--no-asr", action="store_true", help="build from frames alone")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="", help="OpenAI-compatible endpoint; not needed with a backend")
    ap.add_argument("--effort", default="", help="default: video_memory.build_effort from the config")
    ap.add_argument("--config", default=str(ROOT / "configs" / "config_videoindex.yaml"))
    ap.add_argument("--only", default="", help="comma-separated item ids, whatever their length")
    ap.add_argument("--min-duration", type=float, default=MEMORY_REQUIRED_FROM_S, help="seconds; ignored with --only")
    ap.add_argument("--shard", default="", help="I/N: do every Nth video starting at I (from 0), to run N of these side by side")
    ap.add_argument("--workers", type=int, default=8, help="model calls at a time, per video")
    ap.add_argument("--fps", type=float, default=0, help="frames per second sent; default: from the config")
    ap.add_argument("--max-dim", type=int, default=0, help="frame width in pixels; default: from the config")
    ap.add_argument("--max-frames", type=int, default=0, help="frames per segment; default: from the config")
    ap.add_argument("--min-scene-sec", type=float, default=30.0)
    ap.add_argument("--max-scene-sec", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="list what would be built; calls no model")
    a = ap.parse_args()
    a.out = str(Path(a.out).resolve())
    a.videos = str(Path(a.videos).resolve())
    a.transcripts = str(Path(a.transcripts).resolve())

    os.environ["CONFIG_FILE"] = str(Path(a.config).resolve())
    sys.path.insert(0, str(ROOT))
    from video_agent.config import get_config

    cfg = get_config()
    memory_cfg = cfg.video_memory
    embed_backend = os.environ.get("VIDEOLOOP_EMBED_BACKEND") or memory_cfg.embed_backend
    settings_key = {
        "model": a.model,
        "effort": a.effort or memory_cfg.build_effort,
        "fps": a.fps or memory_cfg.build_fps,
        "max_dim": a.max_dim or memory_cfg.build_max_dim,
        "max_frames": a.max_frames or memory_cfg.build_max_frames,
        "min_scene_sec": a.min_scene_sec,
        "max_scene_sec": a.max_scene_sec,
        "asr": not a.no_asr,
        "retrieval": "hybrid" if embed_backend else "bm25",
        "embed_backend": embed_backend,
    }

    items = select_items(ap, a, load_items(a.items))
    problems = check_inputs(a, items, settings_key)
    if problems:
        print("\n".join(problems[:20]), file=sys.stderr)
        ap.error(f"{len(problems)} input problems")
    finished = [i for i in items if read_stats(a, i).get("status") == "done"]
    todo = [i for i in items if i not in finished]
    hours = sum(i["duration_s"] for i in todo) / 3600
    print(f"{len(finished)} already built, {len(todo)} to build, {hours:.1f} h of video")
    if a.dry_run:
        for item in items:
            print(f"{'kept' if item in finished else 'run '} {item['item_id']} {item['duration_s']:.0f}s"
                  f"{' transcript' if transcript_of(a, item) else ''}")
        return

    missing = [m for m in ("numpy", "cv2") if not importlib.util.find_spec(m)]
    if missing:
        ap.error(f'needs {", ".join(missing)}: pip install -e ".[video-memory]"')
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        ap.error(f"{', '.join(missing)} not on PATH")
    api_base = a.api_base or os.environ.get("MAIN_AGENT_API_BASE", "")
    # With neither, openai_completion would fall back to the Gemini API and send it the frames.
    if not (api_base or os.environ.get("VIDEOLOOP_LLM_BACKEND") or cfg.llm.backend):
        ap.error("no model route: pass --api-base, or set VIDEOLOOP_LLM_BACKEND")

    from video_agent.video_memory.llm_client import BuildSettings

    settings = BuildSettings(
        model=a.model,
        api_base=api_base,
        api_key=os.environ.get("MAIN_AGENT_API_KEY", ""),  # Stays in the environment, never in argv
        effort=settings_key["effort"],
        max_output_tokens=memory_cfg.build_max_output_tokens,
        timeout=memory_cfg.build_timeout,
        fps=settings_key["fps"],
        max_dim=settings_key["max_dim"],
        max_frames=settings_key["max_frames"],
        max_mb=cfg.main_agent.max_batch_mb,
    )
    commit = git_output("rev-parse", "HEAD")
    failed = in_row = 0
    for n, item in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {item['item_id']}", flush=True)
        stats = build_one(a, item, settings, settings_key, commit)
        if stats["status"] == "done":
            in_row = 0
            print(f"[{n}/{len(todo)}] {item['item_id']} done: {stats['n_macro_events']} segments, "
                  f"{stats['n_empty_subgraphs']} empty, {stats['wall_s']:.0f}s", flush=True)
            continue
        failed += 1
        in_row += 1
        print(f"[{n}/{len(todo)}] {item['item_id']} FAILED: {stats['error']}", file=sys.stderr, flush=True)
        if in_row >= MAX_FAILURES_IN_ROW:
            print(f"stopped after {in_row} failures in a row: fix the cause, then run this command again", file=sys.stderr)
            break
    if failed:
        sys.exit(1)
    print(f"done; run with: python scripts/run_videoindex.py ... --video-memory {a.out}")


if __name__ == "__main__":
    main()
