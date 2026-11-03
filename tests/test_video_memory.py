"""Tests for the video memory: the build, the index, the agent's tools and the score.

Like test_videoindex_pilot.py, these run the real scripts, real bubblewrap and real ffmpeg against the
strict fake model. The videos are made up here, and no real model or embedding has been through this.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("CONFIG_FILE", "config_videoindex.yaml")

pytest.importorskip("numpy")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

import score_videoindex as score  # noqa: E402
from fake_model_server import MEMORY_SCRIPT, FakeModelServer  # noqa: E402
from test_videoindex_pilot import MODEL, Experiment  # noqa: E402
from video_agent import tool_handlers  # noqa: E402
from video_agent.tools import format_tools_for_gemini, format_tools_for_openai, get_all_tools  # noqa: E402
from video_agent.video_memory.build_graph import _fit_to_size, step3_hierarchical_aggregation  # noqa: E402
from video_agent.video_memory.embeddings import EmbeddingIndex, load_retrieval_index, save_retrieval_index  # noqa: E402
from video_agent.video_memory.llm_client import BuildSettings, extract_json  # noqa: E402
from video_agent.video_memory.loader import load_toolkit  # noqa: E402
from video_agent.video_memory.schema import MacroEvent  # noqa: E402
from video_agent.video_memory.tool_schemas import VIDEO_MEMORY_TOOL_NAMES  # noqa: E402

CAN_BUILD = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
CAN_SANDBOX = bool(CAN_BUILD and shutil.which("bwrap"))
needs_ffmpeg = pytest.mark.skipif(not CAN_BUILD, reason="needs ffmpeg and ffprobe")
needs_sandbox = pytest.mark.skipif(not CAN_SANDBOX, reason="needs bwrap, ffmpeg and ffprobe")
TRANSCRIPT = [{"start": 1.0, "end": 3.0, "text": " The chef talks about pancakes."}]


def make_scenes_video(path, colors=("red", "blue", "green", "yellow", "white"), scene_s=12, sound=True):
    """Hard cuts between flat colors, with a tone. Five scenes of 12 s make 60 s."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for color in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=5:d={scene_s}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={scene_s * len(colors)}"]
    chain = "".join(f"[{i}:v]" for i in range(len(colors))) + f"concat=n={len(colors)}:v=1:a=0[v]"
    cmd += ["-filter_complex", chain, "-map", "[v]"]
    if sound:
        cmd += ["-map", f"{len(colors)}:a", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)


class MemoryExperiment(Experiment):
    def __init__(self, tmp_path, server):
        super().__init__(tmp_path, server)
        self.memory = tmp_path / "memory"

    def add_long(self, item_id, mode="memory", answer="C", gold=None, duration_s=2000.0):
        """A video with scenes and sound, whose item says it is long, so that the rules for long videos apply."""
        self.add(item_id, mode=mode, answer=answer, gold=gold, transcript=False)
        make_scenes_video(self.videos / f"{item_id}.mp4")
        item = self.items[-1]
        item["duration_s"] = duration_s
        item["audio"] = {"codec": "aac", "sample_rate": 44100, "channels": 1}
        (self.transcripts / f"{item_id}.json").write_text(json.dumps({"segments": TRANSCRIPT, "elapsed_s": 5.0}))

    def build(self, *extra, check=True, env=None):
        self.write_items()
        cmd = [sys.executable, str(ROOT / "scripts" / "build_video_memory.py"), "--items", str(self.items_path),
               "--videos", str(self.videos), "--out", str(self.memory), "--transcripts", str(self.transcripts),
               "--model", MODEL, "--api-base", self.server.base, "--workers", "2",
               "--min-scene-sec", "8", "--max-scene-sec", "30"]
        p = subprocess.run(cmd + list(extra), capture_output=True, text=True, timeout=300, env={**os.environ, **(env or {})})
        if check:
            assert p.returncode == 0, p.stdout + p.stderr
        return p

    def stats(self, item_id):
        return json.loads((self.memory / item_id / "build_stats.json").read_text())

    def graph(self, item_id):
        return json.loads((self.memory / item_id / "graph_memory.json").read_text())

    def requests(self, kind):
        return [r for r in self.server.requests if r["kind"] == kind]

    def run_with_memory(self, *extra, **kw):
        return self.run("--video-memory", str(self.memory), "--max-iterations", "30", *extra, **kw)

    def score_sessions(self, *extra):
        cmd = [sys.executable, str(ROOT / "scripts" / "score_videoindex.py"), "--run", str(self.out), "--items", str(self.items_path)]
        p = subprocess.run(cmd + list(extra), capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stdout + p.stderr
        with open(self.out / "sessions.csv", newline="") as f:
            return p.stdout, {r["item_id"]: r for r in csv.DictReader(f)}


@pytest.fixture
def experiment(tmp_path):
    server = FakeModelServer().start()
    yield MemoryExperiment(tmp_path, server)
    server.stop()


# --------------------------------------------------------------------------- the build


@needs_ffmpeg
def test_a_build_breaks_no_rule_of_the_model_and_records_what_it_spent(experiment):
    experiment.add_long("v1")
    experiment.build()
    assert experiment.server.violations == []

    stats = experiment.stats("v1")
    assert stats["status"] == "done" and stats["error"] == ""
    assert stats["n_macro_events"] >= 2 and stats["n_empty_subgraphs"] == 0
    assert stats["usage"]["memory_subgraph"]["calls"] == stats["n_macro_events"]
    assert stats["usage"]["memory_aggregate"]["calls"] == 1
    assert stats["usage"]["memory_subgraph"]["input_tokens"] > 0
    assert stats["wall_s"] > 0 and set(stats["phase_s"]) == {"segment", "subgraph", "aggregate", "index"}
    assert stats["settings"]["retrieval"] == "bm25"

    subgraphs = experiment.requests("build_subgraph")
    assert {r["effort"] for r in subgraphs} == {"medium"}
    assert {r["max_tokens"] for r in subgraphs} == {65536}
    assert all(r["tools"] == [] and r["n_images"] >= 4 for r in subgraphs)
    assert all(r["n_images"] == 0 for r in experiment.requests("build_aggregate"))


@needs_ffmpeg
def test_times_are_relative_in_the_request_and_absolute_in_the_memory(experiment):
    experiment.add_long("v1")
    experiment.build()
    for request in experiment.requests("build_subgraph"):
        assert "Frame 1: 00:00:00\n" in request["task"], "every segment's frames count from its own start"

    macros = experiment.graph("v1")["macro_events"]
    assert macros[0]["time_range"][0] == 0
    assert macros[-1]["time_range"][1] == pytest.approx(60, abs=1)
    assert all(a["time_range"][1] == b["time_range"][0] for a, b in zip(macros, macros[1:])), "no gaps"
    for macro in macros:
        start = macro["time_range"][0]
        assert macro["subgraph"]["micro_events"][0]["time_range"] == [start + 1, start + 2]
        assert macro["subgraph"]["on_screen_texts"][0]["time_range"] == [start, start + 3]
        assert macro["subgraph"]["entities"][0]["visual_grounding"]["primary_time_sec"] == start + 1


@needs_ffmpeg
def test_the_transcript_reaches_the_segment_it_belongs_to(experiment):
    experiment.add_long("v1")
    experiment.build()
    macros = experiment.graph("v1")["macro_events"]
    assert macros[0]["asr_text"] == "The chef talks about pancakes."
    assert all(not m["asr_text"] for m in macros[1:])
    with_text = [r for r in experiment.requests("build_subgraph") if "Audio transcript (ASR)" in r["task"]]
    assert len(with_text) == 1 and "talks about pancakes" in with_text[0]["task"]


@needs_ffmpeg
def test_the_build_never_sees_the_question_options_or_answer(experiment):
    experiment.add_long("v1", gold="GOLDANSWERTOKEN")
    experiment.items[-1]["question"] = "QUESTIONTOKEN"
    experiment.items[-1]["options"] = ["OPTIONTOKEN"] * 4
    seen = []
    handle = experiment.server.handle
    experiment.server.handle = lambda body: (seen.append(json.dumps(body)), handle(body))[1]
    experiment.build()
    assert seen
    memory_text = (experiment.memory / "v1" / "graph_memory.json").read_text()
    for token in ("GOLDANSWERTOKEN", "QUESTIONTOKEN", "OPTIONTOKEN"):
        assert not any(token in s for s in seen)
        assert token not in memory_text


@needs_ffmpeg
def test_the_memory_keeps_the_file_name_and_not_where_the_video_lives(experiment):
    experiment.add_long("v1")
    experiment.build()
    assert experiment.graph("v1")["video_path"] == "v1.mp4"
    assert str(experiment.tmp) not in (experiment.memory / "v1" / "graph_memory.json").read_text()


@needs_ffmpeg
def test_a_segment_never_gets_more_frames_than_the_model_takes(experiment):
    experiment.add("long", transcript=False)
    # With a 250 s minimum, 400 s of one color stays one segment: 201 frames at 0.5 fps before the cap
    make_scenes_video(experiment.videos / "long.mp4", colors=("red",), scene_s=400, sound=False)
    experiment.build("--only", "long", "--no-asr", "--min-scene-sec", "250", "--max-scene-sec", "300")
    assert experiment.server.violations == []
    # 99 when the last timestamp falls on the very end of the video, where ffmpeg finds no frame
    assert max(r["n_images"] for r in experiment.requests("build_subgraph")) in (99, 100)


def test_frames_are_dropped_evenly_to_fit_the_request():
    frames = [("x" * 1000, f"t{i}") for i in range(100)]
    assert _fit_to_size(frames, None) is frames
    assert _fit_to_size(frames, 1) is frames
    kept = _fit_to_size(frames, 0.05)  # About 52 kB
    assert 40 <= len(kept) <= 52
    assert kept[0][1] == "t0" and kept[-1][1] in ("t97", "t98", "t99")
    assert len(_fit_to_size([("x" * 10_000_000, "t")], 0.001)) == 1


@needs_ffmpeg
def test_a_failed_request_is_tried_again_within_the_build(experiment):
    experiment.add_long("v1")
    experiment.server.fail_subgraphs = 1
    experiment.build()
    stats = experiment.stats("v1")
    assert stats["status"] == "done" and stats["n_empty_subgraphs"] == 0
    assert stats["usage"]["memory_subgraph"]["failed_calls"] == 1


@needs_ffmpeg
def test_a_stopped_build_carries_on_without_paying_again(experiment):
    experiment.add_long("v1")
    experiment.server.fail_aggregations = 1
    p = experiment.build(check=False)
    assert p.returncode == 1
    first = experiment.stats("v1")
    assert first["status"] == "failed" and "refused by the test" in first["error"]
    assert not (experiment.memory / "v1" / "graph_memory.json").exists()
    n_subgraphs = len(experiment.requests("build_subgraph"))
    assert n_subgraphs >= 2

    experiment.build()
    assert len(experiment.requests("build_subgraph")) == n_subgraphs, "subgraphs come from disk the second time"
    second = experiment.stats("v1")
    assert second["status"] == "done"
    assert (second["usage"]["memory_aggregate"]["calls"], second["usage"]["memory_aggregate"]["failed_calls"]) == (1, 1)
    assert second["usage"]["memory_subgraph"]["calls"] == n_subgraphs, "the first attempt's spending stays counted"
    assert second["wall_s"] >= first["wall_s"]


@needs_ffmpeg
def test_a_finished_video_is_skipped(experiment):
    experiment.add_long("v1")
    experiment.build()
    n = len(experiment.server.requests)
    p = experiment.build()
    assert "1 already built, 0 to build" in p.stdout and len(experiment.server.requests) == n


@needs_ffmpeg
@pytest.mark.parametrize("fail", ["refused", "empty"])
def test_a_build_with_most_segments_empty_stops_before_aggregating(experiment, fail):
    experiment.add_long("v1")
    if fail == "refused":
        experiment.server.fail_subgraphs = 10 ** 6
    else:
        experiment.server.empty_subgraphs = True
    p = experiment.build(check=False)
    assert p.returncode == 1
    stats = experiment.stats("v1")
    assert stats["status"] == "failed" and "have no subgraph" in stats["error"]
    assert not experiment.requests("build_aggregate")
    assert not (experiment.memory / "v1" / "graph_memory.json").exists()


@needs_ffmpeg
def test_two_failures_in_a_row_stop_the_build_and_one_does_not(experiment):
    for name in ("a", "b", "c"):
        experiment.add_long(name)
    experiment.server.fail_subgraphs = 10 ** 6
    p = experiment.build(check=False)
    assert p.returncode == 1 and "stopped after 2 failures in a row" in p.stderr
    assert not (experiment.memory / "c").exists()


def test_a_change_of_settings_is_refused(experiment):
    experiment.add("v1", transcript=False)
    folder = experiment.memory / "v1"
    folder.mkdir(parents=True)
    settings = {"model": MODEL, "effort": "medium", "fps": 0.5, "max_dim": 720, "max_frames": 100, "min_scene_sec": 8.0,
                "max_scene_sec": 30.0, "asr": False, "retrieval": "bm25", "embed_backend": ""}
    (folder / "build_stats.json").write_text(json.dumps({"status": "failed", "settings": settings}))
    assert experiment.build("--only", "v1", "--no-asr", "--dry-run").returncode == 0
    p = experiment.build("--only", "v1", "--no-asr", "--fps", "1", "--dry-run", check=False)
    assert p.returncode != 0 and "built with a different fps" in p.stderr


def test_which_videos_get_built(experiment):
    for item_id, seconds, width in (("short", 600.0, 320), ("long", 2000.0, 320), ("sound_only", 2500.0, None), ("longer", 9000.0, 640)):
        experiment.items.append({"item_id": item_id, "question": "q", "options": ["a", "b"], "answer": "A", "video": f"videos/{item_id}.mp4",
                                 "duration_s": seconds, "width": width, "audio": None})
        (experiment.videos / f"{item_id}.mp4").touch()
    p = experiment.build("--dry-run", "--no-asr")
    assert "run  long 2000s" in p.stdout and "run  longer 9000s" in p.stdout
    assert "short" not in p.stdout and "run  sound_only" not in p.stdout and "skipping 1 with no video stream: sound_only" in p.stdout
    assert "2 to build, 3.1 h of video" in p.stdout
    assert "run  short 600s" in experiment.build("--dry-run", "--no-asr", "--only", "short").stdout
    assert "2 to build" in experiment.build("--dry-run", "--no-asr", "--shard", "0/2").stdout
    assert "0 to build" in experiment.build("--dry-run", "--no-asr", "--shard", "1/2").stdout
    assert not experiment.server.requests


def test_a_missing_transcript_stops_the_build_before_any_model_call(experiment):
    experiment.items.append({"item_id": "v", "question": "q", "options": ["a", "b"], "answer": "A", "video": "videos/v.mp4",
                             "duration_s": 2000.0, "width": 320, "audio": {"codec": "aac"}})
    (experiment.videos / "v.mp4").touch()
    p = experiment.build(check=False)
    assert p.returncode != 0 and "no transcript" in p.stderr and not experiment.server.requests


def build_settings(server):
    return BuildSettings(model=MODEL, api_base=server.base, effort="medium", timeout=30, max_retries=1)


def test_a_long_video_is_aggregated_in_windows():
    server = FakeModelServer().start()
    macros = [MacroEvent(macro_id=f"macro_{i:04d}", label=f"scene_{i:04d}", time_range=[i * 10.0, i * 10.0 + 10]) for i in range(7)]
    try:
        root, supers, macro_rels, super_rels = step3_hierarchical_aggregation(macros, build_settings(server), window_size=3)
    finally:
        server.stop()
    assert server.violations == []
    assert root.title == "A chef at work"
    assert sum(r["kind"] == "build_aggregate" for r in server.requests) >= 3
    assigned = [m for s in supers for m in s.sub_macro_ids]
    assert sorted(set(assigned)) == [m.macro_id for m in macros]
    assert all(m.super_id for m in macros)
    assert len({s.super_id for s in supers}) == len(supers)


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here it is: {"a": 1} done', '<think>x</think>{"a": 1}'])
def test_json_is_read_from_what_models_write(text):
    assert extract_json(text) == {"a": 1}


# --------------------------------------------------------------------------- the index


def write_embed_module(tmp_path, name="fake_embed", body=None):
    (tmp_path / f"{name}.py").write_text(body or (
        "import hashlib\n"
        "def embed(texts):\n"
        "    out = []\n"
        "    for text in texts:\n"
        "        v = [0.0] * 16\n"
        "        for word in text.lower().split():\n"
        "            v[hashlib.md5(word.strip('.:,').encode()).digest()[0] % 16] += 1.0\n"
        "        out.append(v)\n"
        "    return out\n"
    ))
    return f"{name}:embed"


NODES = [
    {"node_id": "ev_1", "node_type": "event", "text": "flips the pancake: a chef flips a pancake", "macro_id": "m1"},
    {"node_id": "ev_2", "node_type": "event", "text": "pours coffee: a waiter pours coffee", "macro_id": "m2"},
    {"node_id": "ocr_1", "node_type": "on_screen_text", "text": "SCORE 12-7: a scoreboard", "macro_id": "m1"},
]


def test_with_no_backend_search_is_bm25_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("VIDEOLOOP_EMBED_BACKEND", raising=False)
    index = load_retrieval_index(save_retrieval_index(str(tmp_path), NODES, None))
    assert index.embeddings is None
    results = index.search("chef and pancake", top_k=5)
    assert [r["node_id"] for r in results] == ["ev_1"] and results[0]["cosine"] == 0.0
    assert index.search("waiter", node_types=["on_screen_text"]) == []
    assert [r["node_id"] for r in index.search("scoreboard", node_types=["on_screen_text"])] == ["ocr_1"]
    assert index.search("nothing here matches") == []


def test_with_a_backend_search_is_hybrid_and_falls_back_to_bm25(tmp_path, monkeypatch):
    sys.path.insert(0, str(tmp_path))
    monkeypatch.setenv("VIDEOLOOP_EMBED_BACKEND", write_embed_module(tmp_path))
    vectors = np.asarray(__import__("fake_embed").embed([n["text"] for n in NODES]), dtype=np.float32)
    index = load_retrieval_index(save_retrieval_index(str(tmp_path), NODES, vectors))
    assert index.embeddings.shape == (3, 16)
    results = index.search("pours coffee", top_k=3)
    assert results[0]["node_id"] == "ev_2" and results[0]["cosine"] > 0.5
    assert len(results) == 3, "dense search ranks every node, BM25 only the matching ones"

    write_embed_module(tmp_path, "broken_embed", "def embed(texts):\n    raise RuntimeError('down')\n")
    monkeypatch.setenv("VIDEOLOOP_EMBED_BACKEND", "broken_embed:embed")
    fallback = index.search("pours coffee", top_k=3)
    assert [r["node_id"] for r in fallback] == ["ev_2"] and fallback[0]["cosine"] == 0.0
    assert index._dense_disabled


def test_a_query_of_another_dimension_falls_back_to_bm25(tmp_path, monkeypatch):
    sys.path.insert(0, str(tmp_path))
    write_embed_module(tmp_path, "short_embed", "def embed(texts):\n    return [[1.0, 0.0] for _ in texts]\n")
    monkeypatch.setenv("VIDEOLOOP_EMBED_BACKEND", "short_embed:embed")
    index = EmbeddingIndex()
    index.nodes = NODES
    index._set_embeddings(np.ones((3, 16), dtype=np.float32))
    index._build_sparse_index()
    assert [r["node_id"] for r in index.search("coffee")] == ["ev_2"] and index._dense_disabled


@needs_ffmpeg
def test_a_build_with_a_backend_stores_vectors_a_build_without_stores_none(experiment, tmp_path):
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir()
    spec = write_embed_module(embed_dir, "built_embed")
    experiment.add_long("with_vectors")
    experiment.build("--only", "with_vectors", env={"VIDEOLOOP_EMBED_BACKEND": spec, "PYTHONPATH": str(embed_dir)})
    index = load_retrieval_index(str(experiment.memory / "with_vectors" / "embeddings.npz"))
    assert index.embeddings.shape == (len(index.nodes), 16)
    assert experiment.stats("with_vectors")["settings"]["retrieval"] == "hybrid"

    experiment.add_long("no_vectors")
    experiment.build("--only", "no_vectors")
    assert load_retrieval_index(str(experiment.memory / "no_vectors" / "embeddings.npz")).embeddings is None


@needs_ffmpeg
def test_a_build_stopped_at_the_index_only_redoes_the_index(experiment, tmp_path):
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir()
    write_embed_module(embed_dir, "flaky_embed", "def embed(texts):\n    raise RuntimeError('embedding service down')\n")
    env = {"VIDEOLOOP_EMBED_BACKEND": "flaky_embed:embed", "PYTHONPATH": str(embed_dir)}
    experiment.add_long("v1")
    assert experiment.build(check=False, env=env).returncode == 1
    assert "embedding service down" in experiment.stats("v1")["error"]
    index_path = experiment.memory / "v1" / "embeddings.npz"
    assert (experiment.memory / "v1" / "graph_memory.json").exists() and not index_path.exists()

    n_requests = len(experiment.server.requests)
    write_embed_module(embed_dir, "flaky_embed")
    experiment.build(env=env)
    assert len(experiment.server.requests) == n_requests, "nothing but the index is built again"
    assert experiment.stats("v1")["status"] == "done"
    assert load_retrieval_index(str(index_path)).embeddings is not None


@needs_ffmpeg
def test_the_tools_answer_from_the_built_memory(experiment, monkeypatch):
    monkeypatch.delenv("VIDEOLOOP_EMBED_BACKEND", raising=False)
    experiment.add_long("v1")
    experiment.build()
    toolkit = load_toolkit(str(experiment.memory / "v1"))

    class Agent:
        _video_memory = toolkit
        _include_tools = []

    call = lambda name, **args: tool_handlers.dispatch(Agent, name, args)  # noqa: E731
    assert json.loads(call("get_summary"))["title"] == "A chef at work"
    supers = json.loads(call("get_super_events"))
    assert supers[0]["super_id"] == "super_00" and supers[0]["num_macros"] >= 1
    macros = json.loads(call("get_macro_events", super_id="super_00"))
    assert macros and macros[0]["key_entities"] == ["Chef", "ent_002"]  # The pancake has no name
    assert len(json.loads(call("get_macro_events"))) == experiment.stats("v1")["n_macro_events"]
    subgraph = json.loads(call("get_subgraph", macro_id="macro_0000"))
    assert subgraph["events"][0]["time_range"] == "00:00:01-00:00:02"
    assert subgraph["ocr_texts"][0]["text"] == "SCORE 12-7"
    assert {r["label"] for r in subgraph["key_relations"]} == {"USES_TOOL", "CAUSES"}
    assert "not found" in json.loads(call("get_subgraph", macro_id="macro_9999"))["error"]

    found = json.loads(call("search_nodes", query="chef flips the pancake"))["results"]
    assert found[0]["node_type"] == "Event" and found[0]["parent_macro"]["macro_id"].startswith("macro_")
    assert found[0]["time_range"] and found[0]["neighbors"]
    entities = json.loads(call("search_nodes", query="chef", node_types=["entity"]))["results"]
    assert entities and all(r["node_type"] == "Entity" for r in entities)
    assert json.loads(call("search_ocr_text", query="scoreboard"))[0]["text"].startswith("SCORE 12-7")
    spoken = json.loads(call("search_asr_text", query="pancakes"))
    assert spoken[0]["macro_id"] == "macro_0000" and "pancakes" in spoken[0]["text"]
    in_time = json.loads(call("search_by_time", start_sec=0, end_sec=1))
    assert [m["macro_id"] for m in in_time] == ["macro_0000"]
    assert [m["macro_id"] for m in json.loads(call("search_by_time", start_time="00:00:00", end_time="00:00:01"))] == ["macro_0000"]
    assert call("search_by_time", start_time="not a time").startswith("Error: Invalid start_time")
    assert call("search_by_time", start_sec=5, end_sec=1).startswith("Error:")
    listed = json.loads(call("enumerate_events", query="flips the pancake"))
    assert listed["total_matches"] == experiment.stats("v1")["n_macro_events"]
    starts = [m["time_range"] for m in listed["matches"]]
    assert starts == sorted(starts) and "analyze_frames" in listed["note"] and "read_video" not in listed["note"]


def test_a_memory_that_is_not_complete_is_refused(tmp_path):
    (tmp_path / "graph_memory.json").write_text("{}")
    with pytest.raises(RuntimeError, match="No complete memory"):
        load_toolkit(str(tmp_path))


def test_the_tool_lists_agree():
    assert set(tool_handlers._VIDEO_MEMORY_CALLS) == set(VIDEO_MEMORY_TOOL_NAMES)
    assert score.MEMORY_TOOLS == set(VIDEO_MEMORY_TOOL_NAMES)
    assert {name for name, _ in MEMORY_SCRIPT} == set(VIDEO_MEMORY_TOOL_NAMES), "the end-to-end script calls each one"
    assert all(name in tool_handlers._HANDLERS for name in VIDEO_MEMORY_TOOL_NAMES)


def test_memory_tools_come_only_by_name():
    names = lambda tools: {t["function"]["name"] for t in tools}  # noqa: E731
    assert not names(get_all_tools()) & set(VIDEO_MEMORY_TOOL_NAMES), "no whitelist means upstream's tools"
    assert not names(format_tools_for_openai(include_tools=["execute_bash"])) & set(VIDEO_MEMORY_TOOL_NAMES)
    assert names(format_tools_for_openai(include_tools=["execute_bash", "get_subgraph"])) == {"execute_bash", "get_subgraph"}


def test_tools_with_no_arguments_are_fit_for_gemini():
    declarations = format_tools_for_gemini(include_tools=["get_summary", "get_subgraph"])[0]["functionDeclarations"]
    by_name = {d["name"]: d for d in declarations}
    assert "parameters" not in by_name["get_summary"]
    assert by_name["get_subgraph"]["parameters"]["type"] == "OBJECT"
    assert by_name["get_subgraph"]["parameters"]["required"] == ["macro_id"]


def test_a_memory_tool_on_a_video_with_no_memory_says_so():
    class Agent:
        _video_memory = None
        _include_tools = []

    assert tool_handlers.dispatch(Agent, "get_summary", {}) == "Error: this video has no memory."


# --------------------------------------------------------------------------- a question


@needs_sandbox
def test_a_question_with_memory_end_to_end(experiment):
    experiment.add_long("q1", gold="GOLDANSWERTOKEN")
    experiment.build()
    seen = []
    handle = experiment.server.handle
    experiment.server.handle = lambda body: (seen.append(json.dumps(body)), handle(body))[1]
    experiment.run_with_memory()
    assert experiment.server.violations == []
    assert not any("GOLDANSWERTOKEN" in s for s in seen)

    r = experiment.result("q1")
    assert (r["status"], r["answer"], r["answer_source"], r["video_memory"]) == ("succeeded", "C", "submit", True)
    assert all(r["tool_calls"][name] == 1 for name in VIDEO_MEMORY_TOOL_NAMES)
    assert r["tool_calls"]["analyze_frames"] == 3

    main = experiment.requests("main")
    assert all(set(VIDEO_MEMORY_TOOL_NAMES) <= set(x["tools"]) for x in main)
    assert all("<VIDEO_MEMORY>" in x["system"] for x in main)
    assert any("leads to verify against frames" in x["task"] for x in experiment.requests("orchestrator"))

    steps = experiment.trajectory("q1")
    results = [s["content"] for s in steps if s["step_type"] == "tool_result"]
    assert any("flips the pancake" in str(x) for x in results)
    assert not any("Error" in str(x) for x in results[:len(MEMORY_SCRIPT)])
    assert any(s["step_type"] == "video_memory" for s in steps)


@needs_sandbox
def test_the_score_shows_the_build_apart_and_added(experiment):
    experiment.add_long("q1")
    experiment.add("plain", mode="normal", answer="B")
    experiment.build("--only", "q1")
    experiment.run_with_memory()

    text, rows = experiment.score_sessions("--video-memory", str(experiment.memory))
    stats = experiment.stats("q1")
    build_cost = sum(score.cost_of(u, [4, 5, 0.2, 20]) for u in stats["usage"].values())
    row = rows["q1"]
    assert row["video_memory"] == "1" and row["n_memory_calls"] == str(len(MEMORY_SCRIPT))
    assert float(row["build_cost_usd"]) == pytest.approx(build_cost, abs=1e-3) and build_cost > 0
    assert float(row["cost_with_build_usd"]) == pytest.approx(float(row["cost_usd"]) + build_cost, abs=1e-3)
    assert float(row["build_wall_s"]) == stats["wall_s"]
    assert float(row["wall_with_build_s"]) == pytest.approx(float(row["wall_s"]) + stats["wall_s"], abs=0.1)
    plain = rows["plain"]
    assert plain["video_memory"] == "0" and plain["build_cost_usd"] == "" and plain["n_memory_calls"] == "0"
    assert plain["cost_with_build_usd"] == plain["cost_usd"]
    assert "Video memory: 1 questions used one, with 9 memory tool calls." in text
    assert "1 videos" in text and "not in the cost or wall time above" in text

    base = experiment.tmp / "base.csv"
    with open(base, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "correct", "no_answer", "cost_usd", "wall_s", "n_tool_calls", "n_read_image"])
        writer.writeheader()
        writer.writerow({"item_id": "q1", "correct": 0, "no_answer": 0, "cost_usd": 1, "wall_s": 100, "n_tool_calls": 5, "n_read_image": 5})
    text, _ = experiment.score_sessions("--video-memory", str(experiment.memory), "--baseline", str(base))
    assert "cost with memory build ($)" in text and "wall time with memory build (s)" in text

    text, _ = experiment.score_sessions()
    assert "1 questions used a memory whose build isn't counted" in text
    assert "cost with memory build" not in text


@needs_sandbox
def test_without_the_flag_a_built_memory_changes_nothing(experiment):
    experiment.add_long("q1", mode="normal")
    experiment.build()
    experiment.run()
    assert experiment.server.violations == []
    assert experiment.result("q1")["video_memory"] is False
    main = experiment.requests("main")
    assert main and not any(set(VIDEO_MEMORY_TOOL_NAMES) & set(x["tools"]) for x in main)
    assert not any("VIDEO_MEMORY" in x["system"] for x in main)
    assert not any("leads to verify" in x["task"] for x in experiment.requests("orchestrator"))


@needs_sandbox
def test_with_the_flag_a_video_with_no_memory_runs_plain(experiment):
    experiment.add("short", mode="normal")
    experiment.memory.mkdir()
    experiment.run_with_memory()
    assert experiment.server.violations == []
    assert experiment.result("short")["video_memory"] is False
    assert not any(set(VIDEO_MEMORY_TOOL_NAMES) & set(x["tools"]) for x in experiment.requests("main"))


@needs_sandbox
def test_a_broken_memory_fails_the_question_in_the_open(experiment):
    experiment.add("q1", mode="normal")
    folder = experiment.memory / "q1"
    folder.mkdir(parents=True)
    (folder / "graph_memory.json").write_text("{ not json")
    (folder / "embeddings.npz").write_text("nor this")
    experiment.run_with_memory()
    r = experiment.result("q1")
    assert r["status"] == "error" and r["answer"] is None and r["error"]
    assert not experiment.requests("main"), "no question runs quietly without its memory"


@needs_sandbox
def test_the_sandbox_cannot_see_the_memory(experiment):
    experiment.add_long("q1", mode="escape")
    experiment.build()
    target = experiment.memory / "q1" / "graph_memory.json"
    assert target.exists()
    experiment.server.probes = [f"cat {target}; echo probe-done"]
    experiment.run_with_memory()
    output = " ".join(str(s["content"]) for s in experiment.trajectory("q1") if s["step_type"] == "tool_result")
    assert "probe-done" in output and "No such file" in output and "flips the pancake" not in output


def test_a_long_video_with_no_finished_memory_stops_the_run(experiment):
    experiment.items.append({"item_id": "q1", "question": "q", "options": ["a", "b"], "answer": "A", "video": "videos/q1.mp4",
                             "duration_s": 2000.0, "width": 320, "audio": None})
    (experiment.videos / "q1.mp4").touch()
    (experiment.transcripts / "q1.json").write_text("{}")
    experiment.memory.mkdir()
    p = experiment.run_with_memory(check=False)
    assert p.returncode != 0 and "no finished memory" in p.stderr and not experiment.server.requests

    folder = experiment.memory / "q1"
    folder.mkdir()
    (folder / "build_stats.json").write_text(json.dumps({"status": "failed"}))
    p = experiment.run_with_memory(check=False)
    assert p.returncode != 0 and "no finished memory" in p.stderr


def test_dry_run_shows_which_questions_have_a_memory_and_min_duration_selects(experiment):
    for item_id, seconds in (("short", 600.0), ("long", 2000.0)):
        experiment.items.append({"item_id": item_id, "question": "q", "options": ["a", "b"], "answer": "A", "video": f"videos/{item_id}.mp4",
                                 "duration_s": seconds, "width": 320, "audio": None})
        (experiment.videos / f"{item_id}.mp4").touch()
        (experiment.transcripts / f"{item_id}.json").write_text("{}")
    folder = experiment.memory / "long"
    folder.mkdir(parents=True)
    (folder / "build_stats.json").write_text(json.dumps({"status": "done"}))
    text = experiment.run_with_memory("--dry-run").stdout
    assert "run  long 2 options 2000s memory" in text and "run  short 2 options 600s\n" in text
    text = experiment.run_with_memory("--dry-run", "--min-duration", "1800").stdout
    assert "long" in text and "short" not in text.split("sandbox")[0]


@needs_sandbox
def test_a_run_started_without_memory_cannot_go_on_with_it(experiment):
    experiment.add("q1", mode="normal")
    experiment.run("--max-iterations", "30")
    experiment.memory.mkdir()
    p = experiment.run_with_memory(check=False)
    assert p.returncode != 0 and "different video_memory" in p.stderr
    old = json.loads((experiment.out / "run_config.json").read_text())
    assert old["video_memory"] is None
