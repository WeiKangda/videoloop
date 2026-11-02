"""Tests for the Video-Index pilot: the request shape, the sandbox, the runner and the score.

The tests at the end run the real runner, real bubblewrap and real ffmpeg against the strict fake
model in fake_model_server.py, and are skipped where bwrap or ffmpeg is missing. No real model and
no Video-Index data: the questions and videos are made up here.
"""

import base64
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("CONFIG_FILE", "config_videoindex.yaml")

import run_videoindex  # noqa: E402
import score_videoindex as score  # noqa: E402
from fake_model_server import FakeModelServer, violation_in  # noqa: E402
from video_agent import llm_api  # noqa: E402
from video_agent.agent import VideoUnderstandingAgent  # noqa: E402
from video_agent.bwrap_runtime import BwrapRuntime  # noqa: E402
from video_agent.config_loader import get_iteration_warning, get_task_prompt, option_letters_text  # noqa: E402
from video_agent.message_utils import messages_api_view  # noqa: E402
from video_agent.tool_handlers import _frames_to_base64, _split_frame_batches  # noqa: E402

CAN_SANDBOX = bool(shutil.which("bwrap") and shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_sandbox = pytest.mark.skipif(not CAN_SANDBOX, reason="needs bwrap, ffmpeg and ffprobe")
TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


# --------------------------------------------------------------------------- the request shape


def test_view_ends_on_a_user_turn():
    view = messages_api_view([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert [m["role"] for m in view] == ["user", "assistant", "user"]


def test_view_merges_the_ack_into_the_next_assistant_turn():
    original = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Understood."},
        {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    view = messages_api_view(original)
    assert [m["role"] for m in view] == ["user", "assistant", "tool"]
    assert view[1]["content"] == "Understood."
    assert view[1]["tool_calls"] == [TOOL_CALL]
    assert "tool_calls" not in original[1], "the input must stay as it was"


def test_view_drops_empty_assistant_turns():
    view = messages_api_view([
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}, {"role": "user", "content": "go on"},
    ])
    assert [m["role"] for m in view] == ["user", "user"]


def test_view_shrinks_images_from_any_code_path():
    buf = io.BytesIO()
    Image.new("RGB", (3000, 1500)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    block = {"type": "image_url", "image_url": {"url": url}}
    original = [{"role": "user", "content": [{"type": "text", "text": "q"}, block]}]
    view = messages_api_view(original, max_image_long_side=2000)
    shrunk = view[0]["content"][1]["image_url"]["url"]
    assert shrunk.startswith("data:image/jpeg;base64,")
    assert Image.open(io.BytesIO(base64.b64decode(shrunk.split(",", 1)[1]))).size == (2000, 1000)
    assert view[0]["content"][0] == {"type": "text", "text": "q"}
    assert original[0]["content"][1]["image_url"]["url"] == url, "the input must stay as it was"
    assert messages_api_view(original)[0]["content"][1]["image_url"]["url"] == url


def test_usage_openai_style_includes_cached_tokens():
    u = llm_api._normalize_openai_usage({"prompt_tokens": 1000, "completion_tokens": 50,
                                         "prompt_tokens_details": {"cached_tokens": 200}, "cache_creation_input_tokens": 100})
    assert (u["promptTokenCount"], u["cacheReadTokenCount"], u["cacheWriteTokenCount"]) == (1000, 200, 100)


def test_usage_anthropic_style_adds_cached_tokens():
    u = llm_api._normalize_openai_usage({"input_tokens": 10, "output_tokens": 50, "cache_read_input_tokens": 200,
                                         "cache_creation_input_tokens": 100})
    assert (u["promptTokenCount"], u["candidatesTokenCount"]) == (310, 50)


def test_usage_gateway_reporting_only_uncached_prompt_tokens():
    u = llm_api._normalize_openai_usage({"prompt_tokens": 10, "completion_tokens": 5, "cache_read_input_tokens": 200})
    assert u["promptTokenCount"] == 210


def test_backend_gets_the_request_and_usage_is_reported(monkeypatch):
    seen = {}

    def backend(**kw):
        seen.update(kw)
        return {"content": "hi", "thinking_content": "", "tool_calls": [], "finish_reason": "stop",
                "usage": {"promptTokenCount": 7, "candidatesTokenCount": 3}}

    monkeypatch.setitem(sys.modules, "fake_backend_module", type(sys)("fake_backend_module"))
    sys.modules["fake_backend_module"].complete = backend
    monkeypatch.setenv("VIDEOLOOP_LLM_BACKEND", "fake_backend_module:complete")
    records = []
    llm_api.set_usage_sink(records.append)
    try:
        result = llm_api.openai_completion(model="m", messages=[{"role": "user", "content": "q"}], api_base="", api_key="",
                                           temperature=1.0, reasoning_effort="max", usage_tag="main_agent", tool_choice="none")
    finally:
        llm_api.set_usage_sink(None)
    assert result["content"] == "hi"
    assert seen["temperature"] is None, "config_videoindex.yaml omits temperature"
    assert (seen["reasoning_effort"], seen["tool_choice"]) == ("max", "none")
    assert (records[0]["tag"], records[0]["input_tokens"], records[0]["output_tokens"]) == ("main_agent", 7, 3)


def test_backend_is_retried_when_it_may_pass_and_not_when_it_cant(monkeypatch):
    monkeypatch.setattr(llm_api.time, "sleep", lambda s: None)
    calls = []

    def throttled_once(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("ThrottlingException: too many requests")
        return {"content": "ok", "usage": {}}

    assert llm_api._call_backend_with_retry(throttled_once, 8, model="m")["content"] == "ok"
    assert len(calls) == 2

    def rejected(**kw):
        calls.append(kw)
        raise RuntimeError("invalid_request_error: bad request")

    calls.clear()
    with pytest.raises(RuntimeError):
        llm_api._call_backend_with_retry(rejected, 8, model="m")
    assert len(calls) == 1


def test_default_config_sends_what_upstream_sent(monkeypatch):
    from video_agent import config as config_module
    monkeypatch.setattr(config_module, "_cached_config", config_module.AgentConfig())
    sent = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, payload=json)
        return Response()

    monkeypatch.setattr(llm_api.httpx, "post", post)
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    llm_api.openai_completion(model="m", messages=messages, api_base="http://x", api_key="k", temperature=0.0)
    assert sent["payload"]["temperature"] == 0.0
    assert sent["payload"]["messages"] == messages, "no rewriting unless the config asks for it"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert "tool_choice" not in sent["payload"] and "reasoning_effort" not in sent["payload"]


def test_no_key_sends_no_authorization_header(monkeypatch):
    sent = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    def post(url, headers, json, timeout):
        sent.update(headers=headers)
        return Response()

    monkeypatch.setattr(llm_api.httpx, "post", post)
    llm_api.openai_completion(model="m", messages=[{"role": "user", "content": "q"}], api_base="http://x", api_key="")
    assert "Authorization" not in sent["headers"]


def test_claude_models_get_openai_format_tools():
    agent = VideoUnderstandingAgent(model="us.anthropic.claude-opus-5-5", api_base="http://x", verbose=False)
    tools = agent._get_tools_for_provider()
    assert {t["function"]["name"] for t in tools} == {"execute_bash", "submit_answer", "analyze_frames", "transcribe_audio"}
    assert agent._use_gemini_native is False


@pytest.mark.parametrize("body, part", [
    ({"temperature": 1.0, "messages": [{"role": "user", "content": "q"}]}, "temperature"),
    ({"tool_choice": "required", "messages": [{"role": "user", "content": "q"}]}, "tool_choice"),
    ({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}, "prefill"),
    ({"messages": [{"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
                   {"role": "tool", "tool_call_id": "c1", "content": "r"}]}, "no tools declared"),
    ({"tools": [{}], "messages": [{"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
                                  {"role": "user", "content": "q"}]}, "no result"),
    ({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "q"}]}, "empty assistant"),
])
def test_the_fake_model_is_strict(body, part):
    assert part in violation_in(body)


# --------------------------------------------------------------------------- option letters


def test_option_letters_in_words():
    assert option_letters_text(4) == "A, B, C, or D"
    assert option_letters_text(2) == "A or B"
    assert option_letters_text(8) == "A, B, C, D, E, F, G, or H"
    assert option_letters_text(0) == "A, B, C, or D"


def test_four_options_keep_the_papers_prompt():
    text = get_task_prompt("/v.mp4", "q", "A. a", thinking=True)
    assert "the final answer (A, B, C, or D) via submit_answer" in text
    assert "Pick A, B, C, or D and submit" in get_iteration_warning(1, 50)


def test_more_options_change_the_prompts():
    assert "(A, B, C, D, or E)" in get_task_prompt("/v.mp4", "q", "A. a", option_letters="A, B, C, D, or E")
    assert "Pick A or B and submit" in get_iteration_warning(1, 50, "A or B")


def test_extracting_a_letter_covers_all_options():
    extract = VideoUnderstandingAgent._extract_answer_from_text
    assert extract("The answer: H", letters="A-H") == "H"
    assert extract("The answer: H") is None


def test_options_are_lettered_even_when_the_text_starts_with_a():
    options = run_videoindex.lettered_options({"options": ["A person walks", "A dog runs"]})
    assert options == ["A. A person walks", "B. A dog runs"]


# --------------------------------------------------------------------------- frames


def write_image(path, size, fmt="PNG"):
    Image.new("RGB", size, (10, 120, 200)).save(path, format=fmt)


def test_large_frames_are_shrunk_and_small_ones_left_alone(tmp_path):
    big, small = tmp_path / "big.png", tmp_path / "small.jpg"
    write_image(big, (3000, 1500))
    write_image(small, (640, 360), "JPEG")
    urls = _frames_to_base64([str(big), str(small)], max_frames=2, max_long_side=2000)
    size_of = lambda url: Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size  # noqa: E731
    assert size_of(urls[0]) == (2000, 1000) and urls[0].startswith("data:image/jpeg")
    assert size_of(urls[1]) == (640, 360)


def test_frames_stay_as_they_are_without_a_cap(tmp_path):
    big = tmp_path / "big.png"
    write_image(big, (3000, 1500))
    url = _frames_to_base64([str(big)], max_frames=1)[0]
    assert url.startswith("data:image/png") and Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (3000, 1500)


def test_batches_split_by_count_and_by_size():
    urls = ["x" * 100] * 5
    paths = [f"/f{i}.jpg" for i in range(5)]
    assert [len(u) for u, _ in _split_frame_batches(urls, paths, 2)] == [2, 2, 1]
    assert [len(u) for u, _ in _split_frame_batches(urls, paths, 64, max_mb=250 / 1024 / 1024)] == [2, 2, 1]
    assert [p for _, ps in _split_frame_batches(urls, paths, 2) for p in ps] == paths
    assert len(_split_frame_batches(urls, paths, 64)) == 1


# --------------------------------------------------------------------------- the score


def result(status="succeeded", answer="B", source="submit", error="", usage=None, **kw):
    return {"item_id": "i", "status": status, "answer": answer, "answer_source": source, "error": error,
            "wall_s": 10.0, "usage": usage or {}, "n_tool_calls": 2, "n_images": 3, **kw}


ITEM = {"item_id": "i", "options": ["a", "b", "c"], "answer": "B", "duration_s": 90.0, "audio": None, "video": "videos/x.mp4"}
PRICES = [4, 5, 0.2, 20]


@pytest.mark.parametrize("raw, letter", [
    ("C", "C"), ("c", "C"), (" C ", "C"), ("(C)", "C"), ("C.", "C"), ("C. text", "C"), ("C) text", "C"), ("C: text", "C"),
    ("D", None), ("Z", None), ("", None), (None, None), ("1", None), ("Because", None), ("the answer is C", None),
])
def test_parse_letter(raw, letter):
    assert score.parse_letter(raw, 3) == letter


@pytest.mark.parametrize("kw, kind, correct", [
    ({}, "", 1),
    ({"answer": "A"}, "", 0),
    ({"source": "forced_submit"}, "", 1),
    ({"source": "text"}, "", 1),
    ({"source": "default"}, "forced_default", 0),
    ({"source": "extracted"}, "forced_extracted", 0),
    ({"answer": "Q"}, "invalid_letter", 0),
    ({"status": "timeout", "answer": None, "source": None}, "timeout", 0),
    ({"status": "crashed", "answer": None, "source": None}, "crashed", 0),
    ({"status": "error", "answer": None, "source": None, "error": "Input is too long"}, "request_too_large", 0),
    ({"status": "error", "answer": None, "source": None, "error": "503"}, "api_error", 0),
])
def test_what_counts_as_an_answer(kw, kind, correct):
    row = score.score_session(ITEM, result(**kw), PRICES)
    assert (row["no_answer_kind"], row["correct"], row["no_answer"]) == (kind, correct, int(bool(kind)))


def test_cost_bills_each_kind_of_token():
    usage = {"main_agent": {"input_tokens": 1_000_000, "cache_read_tokens": 200_000, "cache_write_tokens": 100_000, "output_tokens": 50_000},
             "orchestrator": {"input_tokens": 500_000, "output_tokens": 10_000}}
    row = score.score_session(ITEM, result(usage=usage), PRICES)
    main = 0.7 * 4 + 0.1 * 5 + 0.2 * 0.2 + 0.05 * 20
    assert row["cost_main_usd"] == pytest.approx(main, abs=1e-3)
    assert row["cost_orchestrator_usd"] == pytest.approx(0.5 * 4 + 0.01 * 20, abs=1e-3)
    assert row["cost_usd"] == pytest.approx(main + 2.2, abs=1e-3)
    assert (row["all_input_tokens"], row["input_tokens"]) == (1_500_000, 1_200_000)


def test_mcnemar_known_values():
    assert score.mcnemar_p(0, 0) == 1.0
    assert score.mcnemar_p(0, 6) == pytest.approx(2 / 64)
    assert score.mcnemar_p(3, 3) == 1.0
    assert score.mcnemar_p(1, 9) == pytest.approx(2 * (1 + 10) / 1024)


def test_duration_bins_match_the_baselines():
    assert [score.duration_bin(s) for s in (59, 60, 599, 600, 3599, 3600)] == [
        "<1 min", "1-10 min", "1-10 min", "10-60 min", "10-60 min", ">=60 min"]


def test_comparison_pairs_by_question_and_compares_cost_only_when_both_finished():
    def row(item_id, correct, no_answer=0, cost=1.0):
        return {"item_id": item_id, "correct": correct, "no_answer": no_answer, "cost_usd": cost, "wall_s": 10,
                "n_tool_calls": 1, "n_read_image": 1, "duration_bin": "<1 min"}
    run = [row("a", 1), row("b", 1), row("c", 0), row("d", 0, 1, 99.0), row("only_in_run", 1)]
    base = [row("a", 1), row("b", 0), row("c", 0), row("d", 0), row("only_in_base", 1)]
    text = "\n".join(score.compare(run, base))
    assert "Paired on 4 questions" in text
    assert "only this run 1; only the baseline 0" in text
    assert "Finished in both: 3 questions" in text
    assert "+0%" in text, "the 99 dollars of the unfinished question must be left out"


# --------------------------------------------------------------------------- the sandbox, without running it


def make_runtime(tmp_path, **kw):
    runtime = BwrapRuntime(work_root=str(tmp_path / "work"), **kw)
    runtime._bwrap = "bwrap"
    runtime.root = os.path.realpath(str(tmp_path / "work"))
    for name in ("testbed", "outputs", "tmp", "models"):
        os.makedirs(os.path.join(runtime.root, name), exist_ok=True)
    return runtime


def test_sandbox_mounts_one_video_and_no_network(tmp_path):
    video = tmp_path / "abc.mp4"
    video.write_bytes(b"x")
    argv = make_runtime(tmp_path, video_path=str(video), env_prefix="/env", extra_ro_binds=["/models/w:/w"])._argv()
    text = " ".join(argv)
    assert f"--ro-bind {video} /videos/abc.mp4" in text
    assert "--ro-bind /env /env" in text and "--ro-bind /models/w /w" in text
    assert "--unshare-net" in argv and "--unshare-pid" in argv and "--die-with-parent" in argv
    assert str(tmp_path / "items") not in text and "--ro-bind / /" not in text
    assert "HOME" not in text, "the environment is set on the process, with nothing inherited"


def test_sandbox_environment_has_no_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("MAIN_AGENT_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    env = make_runtime(tmp_path, env_prefix="/env")._env()
    assert "secret" not in env.values()
    assert env["PATH"].startswith("/env/bin:")


def test_copy_paths_stay_inside_the_sandbox(tmp_path):
    runtime = make_runtime(tmp_path)
    secret = tmp_path / "items.jsonl"
    secret.write_text("answers")
    link = Path(runtime.root) / "outputs" / "frame_t1.jpg"
    link.symlink_to(secret)
    assert runtime._host_path("/outputs/frame_t1.jpg") is None, "a symlink must not reach a host file"
    assert runtime._host_path("/outputs/../../etc/passwd") is None
    assert runtime._host_path("/videos/x.mp4") is None
    assert runtime._host_path("/outputs/a/b.jpg") == os.path.join(runtime.root, "outputs", "a", "b.jpg")
    assert runtime.copy_from_container("/outputs/frame_t1.jpg", str(tmp_path / "out" / "x")) is False


@needs_sandbox
def test_sandbox_refuses_to_start_without_ffmpeg(tmp_path, monkeypatch):
    runtime = BwrapRuntime(work_root=str(tmp_path / "work"))
    monkeypatch.setattr(runtime, "execute", lambda command, timeout=None: (1, "") if "ffmpeg" in command else (0, ""))
    with pytest.raises(RuntimeError, match="ffmpeg or ffprobe"):
        runtime.start()
    assert runtime.root is None


def test_copy_round_trip_keeps_the_original_name(tmp_path):
    runtime = make_runtime(tmp_path)
    src = tmp_path / "t.txt"
    src.write_text("hello")
    assert runtime.copy_to_container(str(src), "/outputs/sub/transcript.txt")
    out = tmp_path / "host"
    out.mkdir()
    assert runtime.copy_from_container("/outputs/sub/transcript.txt", str(out / "transcript.txt"))
    assert (out / "transcript.txt").read_text() == "hello"


# --------------------------------------------------------------------------- the whole run


MODEL = "us.anthropic.claude-opus-5-5"


def make_video(path, seconds=4):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def make_video_with_sound(path, seconds=4):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=5",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True)


def make_audio_only(path):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:a", "aac", "-f", "mp4", str(path)], check=True)


class Experiment:
    """Made-up items, videos and transcripts, with helpers to run the scripts on them."""

    def __init__(self, tmp_path, server):
        self.tmp = tmp_path
        self.server = server
        self.videos = tmp_path / "data" / "videos"
        self.transcripts = tmp_path / "transcripts"
        self.videos.mkdir(parents=True)
        self.transcripts.mkdir()
        self.items_path = tmp_path / "data" / "items.jsonl"
        self.out = tmp_path / "out"
        self.items = []

    def add(self, item_id, mode="normal", answer="C", gold=None, n_options=4, audio_only=False, transcript=True, sound=False):
        name = f"{item_id}.mp4"
        if audio_only:
            make_audio_only(self.videos / name)
        elif sound:
            make_video_with_sound(self.videos / name)
        else:
            make_video(self.videos / name)
        self.items.append({
            "item_id": item_id, "benchmark": "Fake", "capability_group": "g", "question": f"What is shown? MODE={mode} ANSWER={answer}",
            "options": [f"option {i}" for i in range(n_options)], "answer": gold or answer, "answer_idx": 0, "chance": 0.25,
            "video": f"videos/{name}", "video_id": item_id, "duration_s": 4.0, "width": None if audio_only else 320,
            "height": None if audio_only else 240, "fps": None if audio_only else "5/1",
            "audio": {"codec": "aac", "sample_rate": 44100, "channels": 1} if audio_only else None,
        })
        if transcript:
            segments = [{"start": 0.0, "end": 2.0, "text": "a spoken line"}] if audio_only else []
            (self.transcripts / f"{item_id}.json").write_text(json.dumps({"segments": segments, "elapsed_s": 5.0}))

    def write_items(self):
        self.items_path.write_text("\n".join(json.dumps(i) for i in self.items) + "\n")

    def run(self, *extra, config=None, check=True, env=None, out=None):
        self.write_items()
        cmd = [sys.executable, str(ROOT / "scripts" / "run_videoindex.py"), "--items", str(self.items_path),
               "--videos", str(self.videos), "--out", str(out or self.out), "--model", MODEL, "--api-base", self.server.base,
               "--transcripts", str(self.transcripts), "--max-iterations", "12"]
        if config:
            cmd += ["--config", str(config)]
        p = subprocess.run(cmd + list(extra), capture_output=True, text=True, timeout=300, env={**os.environ, **(env or {})})
        if check:
            assert p.returncode == 0, p.stdout + p.stderr
        return p

    def result(self, item_id, out=None):
        return json.loads(((out or self.out) / item_id / "result.json").read_text())

    def trajectory(self, item_id):
        lines = (self.out / item_id / "trajectory.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def score(self, *extra):
        cmd = [sys.executable, str(ROOT / "scripts" / "score_videoindex.py"), "--run", str(self.out), "--items", str(self.items_path)]
        p = subprocess.run(cmd + list(extra), capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stdout + p.stderr
        with open(self.out / "sessions.csv", newline="") as f:
            return p.stdout, {r["item_id"]: r for r in csv.DictReader(f)}


@pytest.fixture
def experiment(tmp_path):
    server = FakeModelServer().start()
    yield Experiment(tmp_path, server)
    server.stop()


@needs_sandbox
def test_a_question_end_to_end_breaks_no_rule_of_the_model(experiment):
    experiment.add("q1", answer="C")
    experiment.add("q2", answer="B", gold="D")
    experiment.run()
    assert experiment.server.violations == []

    r = experiment.result("q1")
    assert (r["status"], r["answer"], r["answer_source"]) == ("succeeded", "C", "submit")
    assert r["n_images"] == 3 and r["tool_calls"]["analyze_frames"] == 3
    assert r["usage"]["main_agent"]["calls"] >= 7 and r["usage"]["orchestrator"]["calls"] >= 1
    assert r["usage"]["main_agent"]["cache_read_tokens"] > 0

    kinds = {x["kind"] for x in experiment.server.requests}
    assert kinds == {"main", "analysis", "orchestrator"}
    assert {x["effort"] for x in experiment.server.requests if x["kind"] in ("main", "analysis")} == {"max"}
    assert any(x["effort"] == "high" for x in experiment.server.requests if x["kind"] == "orchestrator")
    assert all(x["tool_choice"] == "none" for x in experiment.server.requests if x["kind"] == "analysis")
    assert max(x["n_images"] for x in experiment.server.requests if x["kind"] == "analysis") == 1

    text, rows = experiment.score()
    assert rows["q1"]["correct"] == "1" and rows["q2"]["correct"] == "0"
    assert "2 questions, 50.0% correct" in text
    assert float(rows["q1"]["cost_usd"]) > 0

    for line in (experiment.out / "q1" / "trajectory.jsonl").read_text().splitlines():
        assert ";base64," not in line
    assert not (experiment.out / "q1" / "sandbox").exists(), "frames must not stay on disk"
    assert "answer" not in " ".join(experiment.server.requests[0].keys())


@needs_sandbox
def test_the_answer_field_never_reaches_the_model(experiment):
    experiment.add("q1", answer="C", gold="GOLDANSWERTOKEN")
    seen = []
    handle = experiment.server.handle
    experiment.server.handle = lambda body: (seen.append(json.dumps(body)), handle(body))[1]
    experiment.run()
    assert seen and not any("GOLDANSWERTOKEN" in s for s in seen)


@needs_sandbox
def test_upstream_request_shape_is_what_the_strict_model_rejects(experiment, tmp_path):
    """Shows the fixes matter: with them off, the first look at a frame fails."""
    config = (ROOT / "configs" / "config_videoindex.yaml").read_text()
    off = tmp_path / "off.yaml"
    off.write_text(config.replace("messages_api_safe: true", "messages_api_safe: false"))
    experiment.add("q1")
    experiment.run(config=off)
    r = experiment.result("q1")
    assert r["status"] == "error" and "no tools declared" in r["error"]
    assert r["usage"]["main_agent"]["failed_calls"] == 1, "a 400 must not be retried"


@needs_sandbox
def test_more_options_and_an_audio_only_file(experiment):
    experiment.add("eight", answer="H", n_options=8)
    experiment.add("sound", mode="audio", answer="B", audio_only=True)
    experiment.run()
    assert experiment.server.violations == []
    eight = experiment.result("eight")
    assert (eight["answer"], eight["answer_source"]) == ("H", "submit")
    sound = experiment.result("sound")
    assert (sound["answer"], sound["answer_source"]) == ("B", "submit"), "an audio-only file must not bounce forever"
    _, rows = experiment.score()
    assert rows["eight"]["correct"] == "1" and rows["sound"]["correct"] == "1"


@needs_sandbox
def test_transcript_is_asked_for_only_when_there_is_audio(experiment):
    experiment.add("music", answer="B", sound=True)  # An audio track with no speech: its transcript is empty
    experiment.add("silent", answer="B")
    experiment.run()
    assert experiment.server.violations == []
    music, silent = experiment.result("music"), experiment.result("silent")
    assert music["tool_calls"].get("transcribe_audio") == 1 and music["answer_source"] == "submit"
    assert "transcribe_audio" not in silent["tool_calls"] and silent["answer_source"] == "submit"


@needs_sandbox
def test_budget_runs_out_on_bounced_submits_then_the_agent_is_asked(experiment):
    experiment.add("q1", mode="bounced", answer="B")
    experiment.run()
    assert experiment.server.violations == []
    r = experiment.result("q1")
    assert (r["answer"], r["answer_source"]) == ("B", "forced_submit")
    _, rows = experiment.score()
    assert rows["q1"]["correct"] == "1"


@needs_sandbox
def test_a_refusal_at_the_end_counts_as_no_answer(experiment):
    experiment.add("q1", mode="force_refused", answer="A", gold="A")
    experiment.run()
    r = experiment.result("q1")
    assert r["answer_source"] in ("default", "extracted"), r
    _, rows = experiment.score()
    assert rows["q1"]["no_answer"] == "1" and rows["q1"]["correct"] == "0", "the default A must not score even when A is right"


@needs_sandbox
def test_an_answer_stated_in_words_counts_after_the_evidence(experiment):
    experiment.add("q1", mode="late_text", answer="H", n_options=8)
    experiment.run()
    assert experiment.server.violations == []
    r = experiment.result("q1")
    assert (r["answer"], r["answer_source"]) == ("H", "text"), r
    _, rows = experiment.score()
    assert rows["q1"]["correct"] == "1"


@needs_sandbox
def test_a_letter_only_found_in_free_text_is_not_an_answer(experiment):
    experiment.add("q1", mode="text_only", answer="H", n_options=8)
    experiment.run()
    assert experiment.server.violations == []
    r = experiment.result("q1")
    assert (r["answer"], r["answer_source"]) == ("H", "extracted"), r
    _, rows = experiment.score()
    assert rows["q1"]["no_answer_kind"] == "forced_extracted" and rows["q1"]["correct"] == "0"


@needs_sandbox
def test_orchestrator_tool_loop(experiment):
    experiment.add("reads", mode="orch_read", answer="A")
    experiment.add("loops", mode="orch_loop", answer="A")
    experiment.run()
    assert experiment.server.violations == []
    assert experiment.result("reads")["status"] == "succeeded"
    r = experiment.result("loops")
    assert r["status"] == "succeeded"
    log = (experiment.out / "loops" / "child.log").read_text()
    assert "exhausted budget" in log and "has no attribute" not in log


@needs_sandbox
def test_timeout_kills_the_question_and_keeps_its_usage(experiment):
    experiment.add("q1", mode="slow")
    experiment.run("--timeout", "8")
    r = experiment.result("q1")
    assert r["status"] == "timeout" and r["answer"] is None and r["wall_s"] < 40
    assert not (experiment.out / "q1" / "sandbox").exists()
    text, rows = experiment.score()
    assert rows["q1"]["no_answer_kind"] == "timeout" and rows["q1"]["correct"] == "0"
    leftovers = subprocess.run(["pgrep", "-f", "run_videoindex.py --child"], capture_output=True, text=True).stdout.split()
    assert leftovers == []


@needs_sandbox
def test_killing_a_question_ends_what_it_runs_in_the_sandbox(experiment):
    experiment.add("q1", mode="hang_bash")
    experiment.run("--timeout", "6")
    assert experiment.result("q1")["status"] == "timeout"
    # Exited processes can stay as zombies where nothing reaps them, as in a container with no init.
    rows = subprocess.run(["ps", "-eo", "stat=,args="], capture_output=True, text=True).stdout.splitlines()
    alive = [r for r in rows if not r.strip().startswith("Z")]
    assert not [r for r in alive if "sleep 987" in r or " bwrap " in f" {r} "]


@needs_sandbox
def test_a_broken_sandbox_stops_before_any_model_call(experiment):
    experiment.add("q1")
    p = experiment.run("--extra-ro-bind", "/nonexistent/folder", check=False)
    assert p.returncode != 0 and "can't start a sandbox" in p.stderr
    assert experiment.server.requests == [] and not list(experiment.out.glob("*/result.json"))


@needs_sandbox
def test_errors_are_not_retried_and_resume_keeps_finished_questions(experiment):
    experiment.add("ok", answer="C")
    experiment.add("bad", mode="http400")
    experiment.run()
    assert experiment.result("bad")["status"] == "error"
    assert experiment.result("ok")["status"] == "succeeded"
    calls_before = len(experiment.server.requests)
    p = experiment.run()
    assert "2 already done, 0 to run" in p.stdout
    assert len(experiment.server.requests) == calls_before

    p = experiment.run("--retry-errors")
    assert "2 already done, 0 to run" in p.stdout, "too large for the model is a result, and stays"
    _, rows = experiment.score()
    assert rows["bad"]["no_answer_kind"] == "request_too_large"


@needs_sandbox
def test_retry_errors_runs_an_outage_again(experiment):
    experiment.add("q1", mode="http403")
    experiment.run()
    r = experiment.result("q1")
    assert r["status"] == "error" and "expired" in r["error"]
    assert r["usage"]["main_agent"]["failed_calls"] == 1, "an expired token must not be retried eight times"
    p = experiment.run("--retry-errors")
    assert "0 already done, 1 to run" in p.stdout


@needs_sandbox
def test_too_many_errors_in_a_row_stop_the_run(experiment):
    for i in range(4):
        experiment.add(f"bad{i}", mode="http400")
    p = experiment.run("--max-consecutive-errors", "2", check=False)
    assert p.returncode == 2 and "errors in a row" in p.stderr
    assert len(list(experiment.out.glob("*/result.json"))) == 2, "the remaining questions must stay unrun"


@needs_sandbox
def test_a_different_model_cant_continue_a_run(experiment):
    experiment.add("q1")
    experiment.run()
    experiment.write_items()
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_videoindex.py"), "--items", str(experiment.items_path),
                        "--videos", str(experiment.videos), "--out", str(experiment.out), "--model", "other-model",
                        "--api-base", experiment.server.base, "--transcripts", str(experiment.transcripts)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "different model" in p.stderr


@needs_sandbox
def test_missing_inputs_stop_before_any_model_call(experiment):
    experiment.add("q1", transcript=False)
    p = experiment.run(check=False)
    assert p.returncode != 0 and "no transcript" in p.stderr
    assert experiment.server.requests == []


@needs_sandbox
def test_no_route_stops_before_any_frame_can_leave(experiment):
    experiment.add("q1")
    experiment.write_items()
    env = {k: v for k, v in os.environ.items() if k not in ("MAIN_AGENT_API_BASE", "VIDEOLOOP_LLM_BACKEND")}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_videoindex.py"), "--items", str(experiment.items_path),
                        "--videos", str(experiment.videos), "--out", str(experiment.out), "--model", MODEL,
                        "--transcripts", str(experiment.transcripts)], capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode != 0 and "no model route" in p.stderr


@needs_sandbox
def test_dry_run_calls_no_model_and_shows_the_sandbox(experiment):
    experiment.add("q1")
    p = experiment.run("--dry-run")
    assert experiment.server.requests == []
    assert "run  q1 4 options" in p.stdout and "--unshare-net" in p.stdout
    assert not experiment.out.exists()


@needs_sandbox
def test_cpu_slots(experiment):
    experiment.add("q1")
    experiment.add("q2")
    experiment.run("--cpu-slots", "0;0")
    assert {experiment.result("q1")["slot"], experiment.result("q2")["slot"]} == {"0"}


@needs_sandbox
def test_what_the_sandbox_can_see(tmp_path):
    secret_env = "SECRET_VALUE_IN_ENV"
    probes = [
        "ls / | tr '\\n' ' '",
        "env | tr '\\n' ' '",
        f"cat {tmp_path}/data/items.jsonl",
        "ls /videos | tr '\\n' ' '",
        "touch /videos/new 2>&1; touch /usr/new 2>&1; echo wrote",
        "cat /sys/class/net/*/operstate 2>&1 | tr '\\n' ' '; ls /sys/class/net | tr '\\n' ' '",
        "command -v ffmpeg; command -v python3; pwd",
    ]
    server = FakeModelServer(probes=probes).start()
    try:
        e = Experiment(tmp_path, server)
        e.add("q1", mode="escape")
        e.add("other")
        e.run("--only", "q1", env={"MAIN_AGENT_API_KEY": secret_env, "AWS_SECRET_ACCESS_KEY": secret_env})
        outputs = [str(s["content"]) for s in e.trajectory("q1") if s["step_type"] == "tool_result"]
        joined = "\n".join(outputs)
        assert secret_env not in joined, "keys must not reach the sandbox"
        assert "MODE=escape" not in joined, "the dataset must not be readable"
        assert "No such file" in outputs[2]
        assert outputs[3].strip() == "q1.mp4", "only this question's video"
        assert "Read-only" in outputs[4] and "wrote" in outputs[4]
        assert "eth0" not in outputs[5], "no network device but loopback"
        assert "/testbed" in outputs[6] and "ffmpeg" in outputs[6]
        assert server.violations == []
    finally:
        server.stop()


@needs_sandbox
def test_long_commands_and_files_in_the_sandbox(tmp_path):
    runtime = BwrapRuntime(work_root=str(tmp_path / "work"), network_disabled=True)
    with runtime:
        code, output = runtime.execute("cat > /tmp/long << 'EOF'\n" + "a" * 300_000 + "\nEOF\nwc -c < /tmp/long")
        assert (code, output) == (0, "300001"), "execve refuses one argument over 128 KB, so this must not go through argv"
        code, output = runtime.execute("echo out; echo err >&2; exit 3")
        assert code == 3 and output == "out\n\n[STDERR]\nerr", "the same output as DockerRuntime"
        assert runtime.execute("sleep 5", timeout=1)[0] == 0, "a short per-call timeout isn't enforced, as in DockerRuntime"
        root = runtime.root
        assert os.path.isdir(root)
    assert not os.path.exists(root)


@needs_sandbox
def test_large_frames_go_through_with_more_than_twenty_in_a_request(tmp_path):
    """Over 20 images in a request, each must be at most 2000 px; the fake model checks."""
    server = FakeModelServer(probes=[
        "ffmpeg -y -v error -f lavfi -i testsrc=duration=5:size=3000x1500:rate=5 /outputs/frame_%02d.jpg",
    ]).start()
    handle = server.main_turn
    frames = [f"/outputs/frame_{i:02d}.jpg" for i in range(1, 26)]

    def main_turn(messages, task):
        video = "/videos/big.mp4"
        step = server.steps[video]
        if step == 1:
            server.steps[video] += 1
            return 200, server.reply("", [server.call("analyze_frames", {"image_paths": frames, "question": "What is on screen?"})])
        if step == 2:
            server.steps[video] += 1
            return 200, server.reply("", [server.call("submit_answer", {"answer": "A", "reasoning": "seen"})])
        return handle(messages, task)

    server.main_turn = main_turn
    try:
        e = Experiment(tmp_path, server)
        e.add("big", mode="escape", answer="A")
        config = tmp_path / "gate_off.yaml"
        text = (ROOT / "configs" / "config_videoindex.yaml").read_text()
        config.write_text(text.replace("min_vlm_calls_before_submit: 3", "min_vlm_calls_before_submit: 1")
                              .replace("min_iterations_before_submit: 6", "min_iterations_before_submit: 0"))
        e.run(config=config)
        assert server.violations == []
        r = e.result("big")
        assert r["n_images"] == 25 and r["answer"] == "A", r
        assert max(x["n_images"] for x in server.requests if x["kind"] == "analysis") == 25
    finally:
        server.stop()


def run_check(server, *extra, config=None):
    env = {**os.environ, "CONFIG_FILE": str(config or ROOT / "configs" / "config_videoindex.yaml")}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_model_endpoint.py"), "--model", MODEL,
                           "--api-base", server.base, *extra], capture_output=True, text=True, timeout=120, env=env)


def test_endpoint_check_passes_on_a_good_route():
    server = FakeModelServer().start()
    try:
        p = run_check(server)
    finally:
        server.stop()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "8/8 passed" in p.stdout and "FAIL" not in p.stdout
    assert server.violations == []


def test_endpoint_check_catches_a_route_that_drops_the_effort():
    server = FakeModelServer().start()
    server.drop_effort = True
    try:
        p = run_check(server)
    finally:
        server.stop()
    assert p.returncode == 1 and "FAIL effort" in p.stdout


def test_endpoint_check_finds_every_shape_the_strict_model_rejects(tmp_path):
    config = tmp_path / "unsafe.yaml"
    text = (ROOT / "configs" / "config_videoindex.yaml").read_text()
    config.write_text(text.replace("messages_api_safe: true", "messages_api_safe: false").replace("omit_temperature: true", "omit_temperature: false"))
    server = FakeModelServer().start()
    try:
        p = run_check(server, "--skip-effort", config=config)
    finally:
        server.stop()
    assert p.returncode == 1
    assert "FAIL text" in p.stdout and "temperature" in p.stdout


def test_endpoint_check_needs_a_route():
    env = {k: v for k, v in os.environ.items() if k != "VIDEOLOOP_LLM_BACKEND"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_model_endpoint.py"), "--model", MODEL],
                       capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode != 0 and "no model route" in p.stderr


def test_transcribe_script_skips_silent_videos_and_finished_work(tmp_path):
    stub = tmp_path / "stub" / "faster_whisper"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        "from types import SimpleNamespace as N\n"
        "import os\n"
        "class WhisperModel:\n"
        "    def __init__(self, path, device, compute_type, cpu_threads):\n"
        "        open(os.environ['STUB_LOG'], 'a').write('load\\n')\n"
        "    def transcribe(self, path, beam_size):\n"
        "        open(os.environ['STUB_LOG'], 'a').write(os.path.basename(path) + '\\n')\n"
        "        return iter([N(start=0.0, end=1.5, text=' hello')]), N(language='en')\n")
    items = tmp_path / "items.jsonl"
    rows = [{"item_id": "s", "video": "videos/silent.mp4", "audio": None, "duration_s": 3.0},
            {"item_id": "t", "video": "videos/talk.mp4", "audio": {"codec": "aac"}, "duration_s": 3.0}]
    items.write_text("\n".join(json.dumps(r) for r in rows))
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "talk.mp4").write_bytes(b"x")
    out = tmp_path / "out"
    log = tmp_path / "log"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "stub"), "STUB_LOG": str(log)}
    cmd = [sys.executable, str(ROOT / "scripts" / "transcribe_videoindex.py"), "--items", str(items), "--videos", str(videos),
           "--out", str(out), "--model-dir", "models/large-v3"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode == 0, p.stderr
    assert json.loads((out / "silent.json").read_text())["segments"] == []
    talk = json.loads((out / "talk.json").read_text())
    assert talk["segments"][0]["text"] == " hello" and talk["model"] == "large-v3" and "elapsed_s" in talk
    assert log.read_text().splitlines() == ["load", "talk.mp4"]

    log.write_text("")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    assert "0 written, 2 already there" in p.stdout and log.read_text() == ""

    shard = subprocess.run(cmd + ["--out", str(tmp_path / "shard"), "--shard", "0/2"], capture_output=True, text=True, timeout=60, env=env)
    assert (tmp_path / "shard" / "silent.json").exists() and not (tmp_path / "shard" / "talk.json").exists(), shard.stderr


@needs_sandbox
def test_comparison_with_a_baseline_csv(experiment, tmp_path):
    experiment.add("q1", answer="C")
    experiment.add("q2", answer="B")
    experiment.add("q3", answer="B", gold="D")
    experiment.run()
    base = tmp_path / "base.csv"
    with open(base, "w", newline="") as f:
        w = csv.DictWriter(f, ["item_id", "correct", "no_answer", "cost_usd", "wall_s", "n_tool_calls", "n_read_image"])
        w.writeheader()
        w.writerow({"item_id": "q1", "correct": 1, "no_answer": 0, "cost_usd": 1, "wall_s": 100, "n_tool_calls": 30, "n_read_image": 10})
        w.writerow({"item_id": "q2", "correct": 0, "no_answer": 0, "cost_usd": 1, "wall_s": 100, "n_tool_calls": 30, "n_read_image": 10})
        w.writerow({"item_id": "q3", "correct": 0, "no_answer": 1, "cost_usd": "", "wall_s": 3600, "n_tool_calls": 90, "n_read_image": 90})
    text, _ = experiment.score("--baseline", str(base), "--transcripts", str(experiment.transcripts))
    assert "Paired on 3 questions" in text
    assert "only this run 1; only the baseline 0" in text
    assert "Finished in both: 2 questions" in text
    assert "ASR, done ahead of the run" in text
    assert (experiment.out / "summary.md").exists()
