"""A stand-in for a model endpoint, as strict about requests as Claude Opus 5.5 behind an OpenAI-compatible route.

It rejects with a 400 what the real API rejects: sampling parameters, forced tool_choice, a conversation
ending on an assistant turn, tool history with no tools declared, empty assistant turns, orphaned tool calls
or results, and more than 20 images when one of them is over 2000 px. Every rejection is kept in
``violations``, so a test can require that a run caused none.

The main agent's replies follow a script chosen by a token in the question text, MODE=... and ANSWER=...
Each question is told apart by its video path.
"""

import base64
import io
import json
import re
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


def _text_of(content):
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def _image_urls(messages):
    urls = []
    for m in messages:
        if isinstance(m.get("content"), list):
            urls += [p["image_url"]["url"] for p in m["content"]
                     if isinstance(p, dict) and p.get("type") == "image_url"]
    return urls


def violation_in(body):
    """The reason the real API would answer 400, or None."""
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            return f"sampling parameter {key} is not supported"
    choice = body.get("tool_choice")
    if choice not in (None, "auto", "none"):
        return f"tool_choice {choice} is not supported"
    messages = body.get("messages") or []
    if not messages:
        return "messages is empty"
    if messages[-1].get("role") == "assistant":
        return "prefill: the conversation ends on an assistant turn"
    has_tool_history = any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)
    if has_tool_history and not body.get("tools"):
        return "tool history with no tools declared"
    pending = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            if m.get("tool_call_id") not in pending:
                return f"tool result with no tool call: {m.get('tool_call_id')}"
            pending.discard(m["tool_call_id"])
            continue
        if pending:
            return f"tool calls with no result: {sorted(pending)}"
        if role == "assistant":
            if not _text_of(m.get("content")).strip() and not m.get("tool_calls"):
                return f"empty assistant turn at {i}"
            pending = {c["id"] for c in m.get("tool_calls") or []}
        elif role == "system" and i != 0:
            return "system message after the start"
    if pending:
        return f"tool calls with no result: {sorted(pending)}"
    urls = _image_urls(messages)
    if len(urls) > 100:
        return f"{len(urls)} images, the limit is 100"
    if len(urls) > 20:
        for url in urls:
            size = Image.open(io.BytesIO(base64.b64decode(url.split(";base64,", 1)[1]))).size
            if max(size) > 2000:
                return f"more than 20 images and one is {size}"
    return None


class FakeModelServer:
    def __init__(self, probes=None):
        self.requests = []
        self.violations = []
        self.lock = threading.Lock()
        self.steps = defaultdict(int)
        self.probes = probes or []  # Commands the escape script runs in the sandbox
        self.drop_effort = False  # True acts like a route that loses reasoning_effort
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status, reply = outer.handle(body)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except BrokenPipeError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, body):
        messages = body.get("messages") or []
        names = [t["function"]["name"] for t in body.get("tools") or []]
        is_main = "execute_bash" in names
        task = " ".join(_text_of(m.get("content")) for m in messages if m.get("role") == "user")
        kind = "main" if is_main and body.get("tool_choice") != "none" else "analysis" if is_main else "orchestrator"
        with self.lock:
            self.requests.append({
                "kind": kind, "effort": body.get("reasoning_effort"), "model": body.get("model"),
                "n_images": len(_image_urls(messages)), "tool_choice": body.get("tool_choice"),
                "max_tokens": body.get("max_tokens"),
            })
        problem = violation_in(body)
        if problem:
            with self.lock:
                self.violations.append(problem)
            return 400, {"error": {"type": "invalid_request_error", "message": problem}}
        if kind == "analysis":
            return 200, self.reply("I see a test pattern with colored bars.")
        if not is_main and ("submit_answer" in names or "Reply with" in task or "What color" in task):
            return 200, self.check_turn(body, names, task)
        if kind == "orchestrator":
            if "read_file" in names and not any(m.get("role") == "tool" for m in messages) and "MODE=orch_read" in task:
                return 200, self.reply("", [self.call("read_file", {"path": "/testbed/memory/working_memory.md"})])
            if "read_file" in names and "MODE=orch_loop" in task:
                return 200, self.reply("", [self.call("read_file", {"path": "/testbed/memory/working_memory.md", "max_chars": len(messages)})])
            return 200, self.reply("## Current Understanding\nA test pattern.\n\n## Key Evidence\n- colored bars\n")
        return self.main_turn(messages, task)

    def check_turn(self, body, names, task):
        """Replies for the requests of check_model_endpoint.py."""
        if "Call submit_answer with answer B" in task:
            return self.reply("", [self.call("submit_answer", {"answer": "B"})])
        if "n^2 + 1" in task:
            reply = self.reply("17")
            if body.get("reasoning_effort") != "low" and not self.drop_effort:
                reply["usage"]["completion_tokens"] = 900
            return reply
        return self.reply("It is red." if "What color" in task else "ready")

    def main_turn(self, messages, task):
        mode = (re.search(r"MODE=(\w+)", task) or [None, "normal"])[1]
        answer = (re.search(r"ANSWER=(\w+)", task) or [None, "A"])[1]
        video = (re.search(r"/videos/\S+?\.mp4", task) or [""])[0]
        with self.lock:
            step = self.steps[video]
            self.steps[video] += 1
        last = _text_of(messages[-1].get("content"))
        forced = "submit_answer" in last and "CRITICAL" in last
        if "No transcript obtained yet" in last:
            return 200, self.reply("", [self.call("transcribe_audio", {"video_path": video})])

        if mode == "http400":
            return 400, {"error": {"type": "invalid_request_error", "message": "Input is too long for requested model."}}
        if mode == "http403":
            return 403, {"error": {"type": "authentication_error", "message": "authentication failed: the token expired"}}
        if mode == "slow":
            time.sleep(60)
        submit = self.reply("", [self.call("submit_answer", {"answer": answer, "reasoning": "from the script"})])
        if mode == "bounced":  # Always submits; the gate bounces it until the budget runs out
            return 200, submit
        if mode == "force_refused":
            if forced or "FINAL WARNING" in task or "RIGHT NOW" in task:
                return 200, self.reply("I can't tell.")
            return 200, self.bash(f"echo step {step}")
        if mode == "hang_bash":  # Left running in the sandbox when the question is killed
            return 200, self.bash("sleep 987")
        if mode == "text_only":  # Never calls a tool, and states the answer in words
            return 200, self.reply(f"The answer is {answer}.")
        if mode == "never":
            return (200, submit) if forced else (200, self.bash(f"echo step {step}"))
        if mode == "audio":  # No frames to look at, so only the iteration minimum applies
            return (200, submit) if step >= 6 else (200, self.bash(f"echo step {step}"))
        if mode == "escape":
            return (200, submit) if step >= len(self.probes) else (200, self.bash(self.probes[step]))
        # The normal script: three rounds of extracting a frame and looking at it, then submit
        n_round, phase = divmod(step, 2)
        if n_round >= 3:
            return (200, self.reply(f"The answer is {answer}.")) if mode == "late_text" else (200, submit)
        frame = f"/outputs/frame_t{n_round + 1}.jpg"
        if phase == 0:
            return 200, self.bash(f"ffmpeg -y -v error -ss {n_round} -i {video} -frames:v 1 {frame}")
        return 200, self.reply("", [self.call("analyze_frames", {"image_paths": [frame], "question": "What is on screen?"})])

    def bash(self, command):
        return self.reply("", [self.call("execute_bash", {"command": command})])

    def call(self, name, args):
        return {"id": f"call_{name}_{time.time_ns()}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}

    def reply(self, text, tool_calls=None):
        message = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
                      "prompt_tokens_details": {"cached_tokens": 200}, "cache_creation_input_tokens": 100},
        }
