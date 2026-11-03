#!/usr/bin/env python3
"""Run VideoLoop on Video-Index v1.1, one process per question. See docs/video_index_pilot.md.

  python scripts/run_videoindex.py --items data/items_test.jsonl --videos data/videos \
      --out results/videoindex/run1 --model MODEL --api-base URL --env-prefix ENV \
      --cpu-slots '0-5,48-53;6-11,54-59'

One attempt per question. A question is killed at --timeout seconds, however far it got, and the
process is gone with all of its memory, so nothing carries over between questions. A question with
a result.json is not run again, so a stopped run can be started again with the same command.

For each question <out>/<item_id>/ holds:
  result.json        status, the raw answer, how it was obtained, token counts per kind of call
  trajectory.jsonl   each step, written as it happens, with base64 images taken out
  usage.jsonl        one line per model call, including calls that failed
  child.log          what the agent logged

The answer is only checked afterwards, by score_videoindex.py. The agent's process gets the question
and the options, never the answer field, and its sandbox can see the one video and nothing of the dataset.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATA_URL = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")
DEFAULT_TRANSCRIPTS = "dataset/videoindex_v1_1/transcripts_whisperx"
CONFIG_KEYS = ("model", "orchestrator_model", "effort", "orchestrator_effort", "timeout", "max_iterations", "config_sha256", "video_memory")
MEMORY_REQUIRED_FROM_S = 1800  # With --video-memory, a video this long must have one. Shorter ones use it if there is one.


def item_dir_name(item_id):
    return re.sub(r"[^A-Za-z0-9._-]", "_", item_id)


def load_items(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lettered_options(item):
    """Always 'A. text', since VideoLoop takes options starting 'A ' as already lettered."""
    return [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(item["options"])]


def write_json(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def jsonl_rows(path):
    """The lines of a JSONL file; a last line cut off by a kill is skipped."""
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def summarize_usage(rows):
    out = {}
    for r in rows:
        tag = out.setdefault(r.get("tag", "?"), {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                 "cache_read_tokens": 0, "cache_write_tokens": 0})
        tag["failed_calls" if r.get("error") else "calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            tag[k] += r.get(k, 0) or 0
    return out


def summarize_trajectory(rows):
    tools = {}
    images = 0
    for r in rows:
        if r.get("step_type") == "tool_use":
            name = (r.get("content") or {}).get("name", "?")
            tools[name] = tools.get(name, 0) + 1
        elif r.get("step_type") == "tool_result" and isinstance(r.get("content"), dict):
            images += sum(1 for x in r["content"].get("attachments", []) if x.get("type") == "image")
        elif r.get("step_type") == "frame_batch":
            images += (r.get("content") or {}).get("n_images", 0)
    return {"n_tool_calls": sum(tools.values()), "tool_calls": tools, "n_images": images}


def container_name(item_id):
    return "videoloop_" + hashlib.sha256(item_id.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- one question


def run_child(a, item, item_dir):
    """One question, in its own process. Writes child_result.json and the two logs."""
    sys.path.insert(0, str(ROOT))
    from video_agent import llm_api, transcript_cache
    from video_agent.agent import VideoUnderstandingAgent
    from video_agent.config import get_config

    cfg = get_config()
    cfg.memory.summarizer_model = a.orchestrator_model or a.model
    if a.effort:
        cfg.main_agent.reasoning_effort = a.effort
    if a.orchestrator_effort:
        cfg.memory.summarizer_reasoning_effort = a.orchestrator_effort
    cfg.sandbox.env_prefix = a.env_prefix
    cfg.sandbox.extra_ro_binds = a.extra_ro_bind
    cfg.sandbox.work_root = str(item_dir / "sandbox")
    cfg.sandbox.cpuset_cpus = a.slot or None
    cfg.sandbox.container_name = container_name(item["item_id"])
    cfg.video_memory.memory_dir = a.video_memory
    transcript_cache._EXTRA_CACHE_DIRS.insert(0, Path(a.transcripts))

    usage_log = open(item_dir / "usage.jsonl", "a", encoding="utf-8", buffering=1)
    llm_api.set_usage_sink(lambda record: usage_log.write(json.dumps(record) + "\n"))
    trajectory_log = open(item_dir / "trajectory.jsonl", "a", encoding="utf-8", buffering=1)

    def on_step(step_type, content, _trajectory):
        line = json.dumps({"step_type": step_type, "content": content, "t": time.time()}, ensure_ascii=False, default=str)
        trajectory_log.write(DATA_URL.sub("[image]", line) + "\n")

    agent = VideoUnderstandingAgent(
        model=a.model, api_base=a.api_base, max_iterations=a.max_iterations, verbose=False, step_callback=on_step,
    )
    agent._current_question_id = item["item_id"]
    video = os.path.join(a.videos, os.path.basename(item["video"]))
    result = agent.run(video_path=video, question=item["question"], options=lettered_options(item))

    memory = getattr(agent, "_current_memory", None)
    orchestrator = getattr(memory, "orchestrator", None)
    write_json(item_dir / "child_result.json", {
        "answer": result.answer,
        "answer_source": result.answer_source,
        "reasoning": result.reasoning,
        "error": result.error,
        "video_memory": agent._video_memory is not None,
        "orchestrator_calls_made": getattr(orchestrator, "calls_made", None),
        "orchestrator_calls_failed": getattr(orchestrator, "calls_failed", None),
    })


# --------------------------------------------------------------------------- the whole run


class Running:
    """Process groups running now, so that Ctrl-C leaves nothing behind."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pids = set()

    def add(self, pid):
        with self.lock:
            self.pids.add(pid)

    def remove(self, pid):
        with self.lock:
            self.pids.discard(pid)

    def kill_all(self):
        with self.lock:
            for pid in self.pids:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def run_question(a, item, slot, running):
    """Run one child process to its end or its timeout, then write result.json."""
    item_dir = Path(a.out) / item_dir_name(item["item_id"])
    item_dir.mkdir(parents=True, exist_ok=True)
    # Only what a run needs from an earlier attempt that never got a result.json: nothing.
    for name in ("usage.jsonl", "trajectory.jsonl", "child_result.json", "child.log"):
        (item_dir / name).unlink(missing_ok=True)
    shutil.rmtree(item_dir / "sandbox", ignore_errors=True)

    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", item["item_id"], "--slot", slot or ""] + a.passthrough
    if slot:
        cmd = ["taskset", "-c", slot] + cmd
    env = dict(os.environ, CONFIG_FILE=str(Path(a.config).resolve()))

    started = time.time()
    timed_out = False
    with open(item_dir / "child.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT, start_new_session=True)
        running.add(p.pid)
        try:
            p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        finally:
            running.remove(p.pid)
    wall_s = time.time() - started

    shutil.rmtree(item_dir / "sandbox", ignore_errors=True)
    if a.runtime == "docker":
        subprocess.run(["docker", "rm", "-f", container_name(item["item_id"])], capture_output=True)

    child_path = item_dir / "child_result.json"
    child = json.loads(child_path.read_text(encoding="utf-8")) if child_path.exists() else {}
    if timed_out:
        status, error = "timeout", ""
    elif child.get("error"):
        status, error = "error", child["error"]
    elif child.get("answer") is not None:
        status, error = "succeeded", ""
    else:
        status, error = "crashed", f"exit code {p.returncode}, see child.log"
    result = {
        "item_id": item["item_id"],
        "status": status,
        "error": error,
        "answer": None if timed_out else child.get("answer"),
        "answer_source": None if timed_out else child.get("answer_source"),
        "reasoning": None if timed_out else child.get("reasoning"),
        "wall_s": round(wall_s, 1),
        "slot": slot or "",
        "video_memory": child.get("video_memory", False),
        "orchestrator_calls_made": child.get("orchestrator_calls_made"),
        "orchestrator_calls_failed": child.get("orchestrator_calls_failed"),
        "usage": summarize_usage(jsonl_rows(item_dir / "usage.jsonl")),
        **summarize_trajectory(jsonl_rows(item_dir / "trajectory.jsonl")),
    }
    write_json(item_dir / "result.json", result)
    return result


def should_retry(result):
    """An outage can be run again. A request too large for the model is the question's result."""
    if result["status"] not in ("error", "crashed"):
        return False
    error = (result.get("error") or "").lower()
    return "too long" not in error and "too large" not in error


def select_items(ap, a, items):
    if a.only:
        wanted = [x for x in a.only.split(",") if x]
        known = {i["item_id"] for i in items}
        unknown = [x for x in wanted if x not in known]
        if unknown:
            ap.error(f"--only: no such item: {unknown[0]}")
        items = [i for i in items if i["item_id"] in set(wanted)]
    if a.min_duration:
        items = [i for i in items if i["duration_s"] >= a.min_duration]
    return items[:a.limit] if a.limit else items


def memory_ready(memory_dir, video_name):
    """Whether scripts/build_video_memory.py finished this video."""
    folder = os.path.join(memory_dir, os.path.splitext(video_name)[0])
    stats = os.path.join(folder, "build_stats.json")
    if not os.path.exists(stats):
        return False
    with open(stats, encoding="utf-8") as f:
        return json.load(f).get("status") == "done"


def check_inputs(a, items):
    """Problems that would waste a run, as strings: missing videos, transcripts and memories."""
    problems = []
    for item in items:
        name = os.path.basename(item["video"])
        if not os.path.exists(os.path.join(a.videos, name)):
            problems.append(f"{item['item_id']}: video not found: {name}")
        transcript = os.path.join(a.transcripts, os.path.splitext(name)[0] + ".json")
        if not os.path.exists(transcript) and not a.allow_missing_transcripts:
            problems.append(f"{item['item_id']}: no transcript in {a.transcripts}")
        has_frames = item.get("width") is not None
        if a.video_memory and has_frames and item["duration_s"] >= MEMORY_REQUIRED_FROM_S and not memory_ready(a.video_memory, name):
            problems.append(f"{item['item_id']}: no finished memory in {a.video_memory}")
    return problems


def git_output(*args):
    """What a git command prints here, or '' with no git or no repository."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def check_sandbox(a, item):
    """Start and stop the sandbox once, so that a broken setup shows before any question runs."""
    from video_agent.bwrap_runtime import BwrapRuntime
    video = os.path.join(a.videos, os.path.basename(item["video"]))
    with BwrapRuntime(video_path=video, env_prefix=a.env_prefix, extra_ro_binds=a.extra_ro_bind,
                      network_disabled=a.network_disabled):
        pass


def run_config(a):
    with open(a.config, "rb") as f:
        config_sha256 = hashlib.sha256(f.read()).hexdigest()
    head = git_output("rev-parse", "HEAD")
    dirty = bool(git_output("status", "--porcelain"))
    return {
        "model": a.model,
        "orchestrator_model": a.orchestrator_model or a.model,
        "effort": a.effort,
        "orchestrator_effort": a.orchestrator_effort,
        "timeout": a.timeout,
        "max_iterations": a.max_iterations,
        "video_memory": a.video_memory or None,
        "config": a.config,
        "config_sha256": config_sha256,
        "commit": head,
        "commit_dirty": dirty,
    }


def dry_run(a, items):
    sys.path.insert(0, str(ROOT))
    for item in items:
        done = (Path(a.out) / item_dir_name(item["item_id"]) / "result.json").exists()
        audio_only = " audio-only" if item.get("width") is None else ""
        memory = " memory" if a.video_memory and memory_ready(a.video_memory, os.path.basename(item["video"])) else ""
        print(f"{'kept' if done else 'run '} {item['item_id']} {len(item['options'])} options "
              f"{item['duration_s']:.0f}s{audio_only}{memory}")
    if a.runtime == "bwrap" and items:
        from video_agent.bwrap_runtime import BwrapRuntime
        video = os.path.join(a.videos, os.path.basename(items[0]["video"]))
        sandbox = BwrapRuntime(video_path=video, env_prefix=a.env_prefix, extra_ro_binds=a.extra_ro_bind,
                               network_disabled=a.network_disabled)
        sandbox._bwrap = "bwrap"
        sandbox.root = "<out>/<item_id>/sandbox"
        print("\nsandbox of the first question:\n  " + " ".join(sandbox._argv()) + " bash --norc -c COMMAND")
        print(f"  PATH={sandbox._path_entries()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", required=True, help="folder with <sha>.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--orchestrator-model", default="", help="default: --model")
    ap.add_argument("--api-base", default="", help="OpenAI-compatible endpoint; not needed with a backend")
    ap.add_argument("--effort", default="", help="main agent; default: from the config")
    ap.add_argument("--orchestrator-effort", default="", help="default: from the config")
    ap.add_argument("--config", default=str(ROOT / "configs" / "config_videoindex.yaml"))
    ap.add_argument("--transcripts", default=str(ROOT / DEFAULT_TRANSCRIPTS))
    ap.add_argument("--video-memory", default="", help="folder from scripts/build_video_memory.py; a video with a memory there gets the memory tools")
    ap.add_argument("--env-prefix", default="", help="Python environment for the sandbox's shell (bwrap)")
    ap.add_argument("--extra-ro-bind", action="append", default=[], help="SRC[:DST] mounted read-only (bwrap)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per question")
    ap.add_argument("--max-iterations", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpu-slots", default="", help="'LIST;LIST;...': one taskset CPU list per question running at a time (replaces --workers)")
    ap.add_argument("--only", default="", help="comma-separated item ids")
    ap.add_argument("--min-duration", type=float, default=0, help="only videos at least this many seconds long")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-missing-transcripts", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="run questions again whose status is error or crashed, except a request too large for the model; for outages such as expired credentials")
    ap.add_argument("--max-consecutive-errors", type=int, default=5, help="start no new question after this many errors in a row (0 = never stop)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run and print the sandbox; calls no model")
    ap.add_argument("--child", default="", help=argparse.SUPPRESS)
    ap.add_argument("--slot", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.config = str(Path(a.config).resolve())
    a.out = str(Path(a.out).resolve())
    a.videos = str(Path(a.videos).resolve())
    a.transcripts = str(Path(a.transcripts).resolve())
    a.env_prefix = str(Path(a.env_prefix).resolve()) if a.env_prefix else ""
    a.video_memory = str(Path(a.video_memory).resolve()) if a.video_memory else ""

    items = load_items(a.items)
    if a.child:
        item = next(i for i in items if i["item_id"] == a.child)
        run_child(a, item, Path(a.out) / item_dir_name(a.child))
        return

    os.environ["CONFIG_FILE"] = a.config
    sys.path.insert(0, str(ROOT))
    from video_agent.config import get_config
    cfg = get_config()
    a.runtime = cfg.sandbox.runtime
    a.network_disabled = cfg.sandbox.network_disabled

    items = select_items(ap, a, items)
    problems = check_inputs(a, items)
    if problems:
        print("\n".join(problems[:20]), file=sys.stderr)
        ap.error(f"{len(problems)} input problems; for transcripts, run scripts/transcribe_videoindex.py; "
                 "for memories, run scripts/build_video_memory.py")
    if a.video_memory and not importlib.util.find_spec("numpy"):
        ap.error('--video-memory needs numpy: pip install -e ".[video-memory]"')
    if a.runtime == "bwrap" and not a.dry_run and not shutil.which("bwrap"):
        ap.error("bwrap is not installed")
    slots = [s.strip() for s in a.cpu_slots.split(";") if s.strip()]
    if slots and not shutil.which("taskset"):
        ap.error("--cpu-slots: taskset is not installed")
    workers = len(slots) or a.workers
    if a.dry_run:
        dry_run(a, items)
        return
    # With neither, openai_completion would fall back to the Gemini API and send it the frames.
    if not (a.api_base or os.environ.get("MAIN_AGENT_API_BASE") or os.environ.get("VIDEOLOOP_LLM_BACKEND") or cfg.llm.backend):
        ap.error("no model route: pass --api-base, or set VIDEOLOOP_LLM_BACKEND")

    if a.runtime == "bwrap" and items:
        try:
            check_sandbox(a, items[0])
        except RuntimeError as e:
            ap.error(str(e))

    os.makedirs(a.out, exist_ok=True)
    config_path = Path(a.out) / "run_config.json"
    now = run_config(a)
    if config_path.exists():
        before = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [k for k in CONFIG_KEYS if before.get(k) != now.get(k)]
        if changed:
            ap.error(f"{a.out} was started with a different {changed[0]}; use a new --out")
    else:
        write_json(config_path, now)
    if now["commit_dirty"]:
        print("warning: this checkout has uncommitted changes", file=sys.stderr)

    # What the child needs, without the parent's own flags. Keys stay in the environment, never in argv.
    a.passthrough = ["--items", str(Path(a.items).resolve()), "--videos", a.videos, "--out", a.out,
                     "--model", a.model, "--transcripts", a.transcripts,
                     "--max-iterations", str(a.max_iterations), "--config", a.config]
    for flag, value in (("--orchestrator-model", a.orchestrator_model), ("--api-base", a.api_base),
                        ("--effort", a.effort), ("--orchestrator-effort", a.orchestrator_effort),
                        ("--env-prefix", a.env_prefix), ("--video-memory", a.video_memory)):
        if value:
            a.passthrough += [flag, value]
    for spec in a.extra_ro_bind:
        a.passthrough += ["--extra-ro-bind", spec]

    todo = []
    for item in items:
        path = Path(a.out) / item_dir_name(item["item_id"]) / "result.json"
        if path.exists():
            if a.retry_errors and should_retry(json.loads(path.read_text(encoding="utf-8"))):
                path.unlink()
            else:
                continue
        todo.append(item)
    print(f"{len(items) - len(todo)} already done, {len(todo)} to run, {workers} at a time", flush=True)

    free_slots = queue.Queue()
    for s in slots or [""] * workers:
        free_slots.put(s)
    running = Running()
    stop = threading.Event()
    lock = threading.Lock()
    finished = [0]
    consecutive_errors = [0]

    def task(item):
        if stop.is_set():
            return None
        slot = free_slots.get()
        try:
            if stop.is_set():
                return None
            result = run_question(a, item, slot, running)
        finally:
            free_slots.put(slot)
        with lock:
            finished[0] += 1
            consecutive_errors[0] = consecutive_errors[0] + 1 if result["status"] in ("error", "crashed") else 0
            print(f"[{finished[0]}/{len(todo)}] {item['item_id']} {result['status']} answer={result['answer']} "
                  f"{result['wall_s']:.0f}s {result['n_tool_calls']} tools {result['error'][:120]}", flush=True)
            if a.max_consecutive_errors and consecutive_errors[0] >= a.max_consecutive_errors:
                stop.set()
        return result

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(task, item) for item in todo]
    try:
        for future in futures:
            future.result()
    except BaseException as e:
        stop.set()
        running.kill_all()
        pool.shutdown(wait=True, cancel_futures=True)
        print("stopped; questions with no result.json will run next time", file=sys.stderr)
        if isinstance(e, KeyboardInterrupt):
            sys.exit(130)
        raise
    pool.shutdown(wait=True)
    if stop.is_set():
        print(f"stopped after {a.max_consecutive_errors} errors in a row: fix the cause, then run this "
              "command again with --retry-errors", file=sys.stderr)
        sys.exit(2)
    print(f"done; score with: python scripts/score_videoindex.py --run {a.out} --items {a.items}")


if __name__ == "__main__":
    main()
