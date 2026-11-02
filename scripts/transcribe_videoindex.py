#!/usr/bin/env python3
"""Transcribe the Video-Index v1.1 videos ahead of the run, with faster-whisper.

  python scripts/transcribe_videoindex.py --items data/items_test.jsonl --videos data/videos \
      --model-dir MODELS/large-v3 --threads 12

VideoLoop reads <out>/<video stem>.json at the start of a question, as in the paper's
flow, so a run needs no Whisper in its sandbox. A video with no audio track gets an empty
transcript without loading the model. A finished video is skipped, so this can be stopped
and started again. elapsed_s in each file is what score_videoindex.py reports as ASR time.
"""

import argparse
import json
import os
import time
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "dataset" / "videoindex_v1_1" / "transcripts_whisperx"


def write_json(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", required=True, help="folder with <sha>.mp4")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model-dir", required=True, help="a faster-whisper model folder, or a model name")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads per video (0 = faster-whisper's default)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--only", default="", help="comma-separated item ids")
    ap.add_argument("--shard", default="", help="I/N: do every Nth video starting at I (from 0), to run N of these side by side")
    a = ap.parse_args()

    with open(a.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if a.only:
        wanted = set(a.only.split(","))
        items = [i for i in items if i["item_id"] in wanted]
        missing = wanted - {i["item_id"] for i in items}
        if missing:
            ap.error(f"--only: no such item: {sorted(missing)[0]}")
    if a.shard:
        index, _, count = a.shard.partition("/")
        items = items[int(index)::int(count)]
    os.makedirs(a.out, exist_ok=True)

    model = None
    done = skipped = 0
    for n, item in enumerate(items, 1):
        name = os.path.basename(item["video"])
        stem = os.path.splitext(name)[0]
        path = os.path.join(a.out, f"{stem}.json")
        if os.path.exists(path):
            skipped += 1
            continue
        if not item.get("audio"):
            write_json(path, {"segments": [], "language": None, "elapsed_s": 0.0, "model": None})
            done += 1
            continue
        video = os.path.join(a.videos, name)
        if not os.path.exists(video):
            ap.error(f"video not found: {video}")
        if model is None:
            from faster_whisper import WhisperModel
            model = WhisperModel(a.model_dir, device=a.device, compute_type=a.compute_type, cpu_threads=a.threads)
        started = time.time()
        segments, info = model.transcribe(video, beam_size=a.beam_size)
        rows = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        elapsed = time.time() - started
        write_json(path, {"segments": rows, "language": info.language, "elapsed_s": round(elapsed, 1),
                          "model": os.path.basename(a.model_dir.rstrip("/"))})
        done += 1
        print(f"[{n}/{len(items)}] {item['item_id']}: {len(rows)} segments, {elapsed:.0f}s "
              f"for {item['duration_s']:.0f}s of video", flush=True)
    print(f"{done} written, {skipped} already there, in {a.out}")


if __name__ == "__main__":
    main()
