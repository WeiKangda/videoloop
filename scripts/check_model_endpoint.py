#!/usr/bin/env python3
"""Check that the model route handles every request shape VideoLoop sends. See docs/video_index_pilot.md.

  MAIN_AGENT_API_KEY=... python scripts/check_model_endpoint.py --model MODEL --api-base URL
  VIDEOLOOP_LLM_BACKEND=my_module:complete python scripts/check_model_endpoint.py --model MODEL

Uses configs/config_videoindex.yaml unless CONFIG_FILE says otherwise. Goes through the same function as a run (llm_api.openai_completion), with an HTTP endpoint or a
Python backend, and sends a few small requests, a few cents in all. Each check is one thing a run
depends on; a failed one here would otherwise fail on every question. The last check compares the
output tokens at two efforts, because a route that drops the effort still answers: it just thinks
at the model's default, and the run is then not at the effort you meant. Exit code 1 if any check fails.
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CONFIG_FILE", "config_videoindex.yaml")

from video_agent.config import get_config  # noqa: E402
from video_agent.llm_api import openai_completion  # noqa: E402

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final answer.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "reasoning": {"type": "string"}},
            "required": ["answer"],
        },
    },
}
TOOL_HISTORY = [
    {"role": "system", "content": "You answer questions about a test image."},
    {"role": "user", "content": "What color is the test image? First look it up."},
    {"role": "assistant", "content": "Understood. Continuing analysis based on my memory."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_check_1", "type": "function", "function": {
        "name": "submit_answer", "arguments": json.dumps({"answer": "unknown"})}}]},
    {"role": "tool", "tool_call_id": "call_check_1", "content": "Not submitted. Look at the image first."},
]
EFFORT_PROBE = ("Without using any tool: how many integers n with 1 <= n <= 300 make n^2 + 1 a prime number? "
                "Reply with the number only.")


def red_png_data_url():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--effort", default="", help="default: main_agent.reasoning_effort from the config")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--skip-effort", action="store_true", help="skip the effort check, the one that costs the most")
    a = ap.parse_args()

    cfg = get_config()
    effort = a.effort or cfg.main_agent.reasoning_effort
    key = os.environ.get("MAIN_AGENT_API_KEY", "")
    backend = os.environ.get("VIDEOLOOP_LLM_BACKEND") or cfg.llm.backend
    if not backend and not a.api_base:
        ap.error("no model route: pass --api-base, or set VIDEOLOOP_LLM_BACKEND")
    print(f"model {a.model}, route: " + (f"backend {backend}" if backend else f"HTTP {a.api_base}"))
    print(f"omit_temperature={cfg.llm.omit_temperature} messages_api_safe={cfg.llm.messages_api_safe} effort={effort}\n")

    def call(messages, **kw):
        kw.setdefault("reasoning_effort", effort)
        kw.setdefault("max_tokens", 16000)
        return openai_completion(model=a.model, messages=messages, api_base=a.api_base, api_key=key, temperature=1.0,
                                 timeout=a.timeout, max_retries=2, **kw)

    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append(True)
            print(f"PASS {name}: {detail}")
        except Exception as e:  # noqa: BLE001 - every failure here is the finding
            results.append(False)
            print(f"FAIL {name}: {str(e)[:400]}")

    usage_seen = {}

    def text_reply():
        r = call([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Reply with the word ready."}])
        assert r["content"].strip(), "empty reply"
        usage_seen.update(r.get("usage") or {})
        return repr(r["content"].strip()[:40])

    def image_reply():
        r = call([{"role": "user", "content": [
            {"type": "text", "text": "What color is this square? Reply with one word."},
            {"type": "image_url", "image_url": {"url": red_png_data_url()}},
        ]}])
        assert "red" in r["content"].lower(), f"expected red, got {r['content'][:60]!r}"
        return "the model saw the image"

    def tool_call():
        r = call([{"role": "user", "content": "Call submit_answer with answer B."}], tools=[SUBMIT_TOOL])
        assert r["tool_calls"], f"no tool call, finish_reason={r['finish_reason']}, text={r['content'][:80]!r}"
        args = json.loads(r["tool_calls"][0]["function"]["arguments"])
        assert args.get("answer") == "B", f"arguments were {args}"
        return "submit_answer(answer=B)"

    def after_tool_result():
        r = call(TOOL_HISTORY, tools=[SUBMIT_TOOL])
        assert r["content"].strip() or r["tool_calls"], "empty reply"
        return "accepted tool history, a double assistant turn and no thinking blocks"

    def text_only_with_tool_history():
        messages = TOOL_HISTORY + [{"role": "user", "content": [
            {"type": "text", "text": "[Frames from your analysis request. Describe what you observe in detail.]"},
            {"type": "image_url", "image_url": {"url": red_png_data_url()}},
        ]}]
        r = call(messages, tools=[SUBMIT_TOOL], tool_choice="none")
        assert not r["tool_calls"], "called a tool despite tool_choice none"
        assert r["content"].strip(), f"no text, finish_reason={r['finish_reason']}"
        return "tool_choice none works; this is the call after every analyze_frames"

    def trailing_assistant():
        r = call(TOOL_HISTORY + [{"role": "assistant", "content": "The result said to look at the image."}], tools=[SUBMIT_TOOL])
        assert r["content"].strip() or r["tool_calls"], "empty reply"
        return "a turn ending on assistant text goes through"

    def usage_fields():
        assert usage_seen.get("promptTokenCount") and usage_seen.get("candidatesTokenCount"), (
            f"no token counts in usage: {usage_seen}; the score would report a cost of zero")
        return ", ".join(f"{k}={v}" for k, v in usage_seen.items())

    def effort_is_honored():
        low = call([{"role": "user", "content": EFFORT_PROBE}], reasoning_effort="low", max_tokens=64000)
        high = call([{"role": "user", "content": EFFORT_PROBE}], reasoning_effort=effort, max_tokens=64000)
        low_out, high_out = low["usage"].get("candidatesTokenCount", 0), high["usage"].get("candidatesTokenCount", 0)
        assert high_out > low_out, (f"output tokens: low {low_out}, {effort} {high_out}; "
                                    "the route may be dropping the effort, or this probe was too easy to tell")
        return f"output tokens: low {low_out}, {effort} {high_out}"

    check("text", text_reply)
    check("image", image_reply)
    check("tool call", tool_call)
    check("after a tool result", after_tool_result)
    check("text only, tools declared", text_only_with_tool_history)
    check("ends on assistant", trailing_assistant)
    check("usage", usage_fields)
    if not a.skip_effort:
        check("effort", effort_is_honored)

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
