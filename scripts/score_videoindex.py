#!/usr/bin/env python3
"""Score a run_videoindex.py run, and compare it with another run question by question.

  python scripts/score_videoindex.py --run results/videoindex/run1 --items data/items_test.jsonl \
      [--baseline BASELINE/sessions.csv] [--transcripts DIR] [--prices 4,5,0.2,20]

Writes <run>/sessions.csv and <run>/summary.md, and prints the summary. Standard library only.

An answer counts only if the agent submitted it: its own submit_answer call, a submit_answer it made
when told the budget had run out (forced_submit), or a plain statement such as "the answer is C" after
several turns with no tool call (text). VideoLoop's fallbacks, a regular expression over the conversation
(extracted) and the default "A" (default), count as no answer, and so do a timeout, an error and a letter
that isn't one of the options. No answer counts as wrong. All of these are listed by kind in the summary.

--baseline takes any CSV with item_id, correct, no_answer and, where present, cost_usd, wall_s,
n_tool_calls and n_read_image, such as the sessions.csv from another run. The comparison is paired by
question, on the questions in both: a percentile bootstrap over questions (2,000 resamples, fixed seed) for
a 95% interval on each difference, and an exact McNemar test on accuracy. Cost, tokens and time are compared
only on questions where both finished, because a run killed at its limit has no full cost to compare.

--prices is input,cache write,cache read,output in dollars per million tokens. The default is the list price
of Claude Opus 5.5. Every input token that wasn't reported as cached is billed at the input price, so a model
call whose gateway reports no cache fields is billed in full.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import statistics
from collections import Counter

BINS = ["<1 min", "1-10 min", "10-60 min", ">=60 min"]
BOOTSTRAP = 2000
ANSWERED_SOURCES = {"submit", "forced_submit", "text"}


def duration_bin(s):
    return "<1 min" if s < 60 else "1-10 min" if s < 600 else "10-60 min" if s < 3600 else ">=60 min"


def parse_letter(raw, n_options):
    """The option letter in a submitted answer, or None. 'C', 'c', '(C)', 'C.' and 'C. text' are letters."""
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) == 1:
        letter = text.upper()
    else:
        m = re.match(r"\(?([A-Z])\)?(?:[\s.:,)\-]|$)", text)
        letter = m.group(1) if m else None
    if letter and "A" <= letter <= "Z" and ord(letter) - 65 < n_options:
        return letter
    return None


def no_answer_kind(result, letter):
    """Why a run has no answer, or '' when it has one."""
    status = result.get("status")
    if status == "timeout":
        return "timeout"
    if status == "crashed":
        return "crashed"
    if status == "error":
        error = (result.get("error") or "").lower()
        return "request_too_large" if "too long" in error or "too large" in error else "api_error"
    source = result.get("answer_source")
    if source not in ANSWERED_SOURCES:
        return f"forced_{source}" if source else "no_answer"
    return "" if letter else "invalid_letter"


def cost_of(usage, prices):
    """Dollars for one kind of call. input_tokens includes the cached tokens."""
    p_in, p_write, p_read, p_out = prices
    read, write = usage.get("cache_read_tokens", 0), usage.get("cache_write_tokens", 0)
    uncached = max(usage.get("input_tokens", 0) - read - write, 0)
    return (uncached * p_in + write * p_write + read * p_read + usage.get("output_tokens", 0) * p_out) / 1e6


def score_session(item, result, prices, asr_s=None):
    letter = parse_letter(result.get("answer"), len(item["options"]))
    kind = no_answer_kind(result, letter)
    correct = not kind and letter == item["answer"]
    usage = result.get("usage") or {}
    totals = {k: sum(u.get(k, 0) for u in usage.values())
              for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
    main = usage.get("main_agent", {})
    other = [u for tag, u in usage.items() if tag != "main_agent"]
    return {
        "item_id": item["item_id"],
        "benchmark": item.get("benchmark", ""),
        "capability": item.get("capability_group", ""),
        "duration_s": item["duration_s"],
        "duration_bin": duration_bin(item["duration_s"]),
        "has_audio": int(bool(item.get("audio"))),
        "status": result.get("status"),
        "correct": int(correct),
        "no_answer": int(bool(kind)),
        "no_answer_kind": kind,
        "answer": letter or "",
        "answer_source": result.get("answer_source") or "",
        "cost_usd": round(sum(cost_of(u, prices) for u in usage.values()), 4),
        "cost_main_usd": round(cost_of(main, prices), 4),
        "cost_orchestrator_usd": round(sum(cost_of(u, prices) for u in other), 4),
        "wall_s": result.get("wall_s"),
        "asr_s": asr_s if asr_s is not None else "",
        "n_tool_calls": result.get("n_tool_calls", 0),
        "n_read_image": result.get("n_images", 0),
        "input_tokens": max(totals["input_tokens"] - totals["cache_read_tokens"] - totals["cache_write_tokens"], 0),
        "cache_read_tokens": totals["cache_read_tokens"],
        "cache_write_tokens": totals["cache_write_tokens"],
        "output_tokens": totals["output_tokens"],
        "all_input_tokens": totals["input_tokens"],
        "main_calls": main.get("calls", 0),
        "orchestrator_calls": sum(u.get("calls", 0) for u in other),
        "failed_calls": sum(u.get("failed_calls", 0) for u in usage.values()),
    }


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_interval(pairs, stat, n=BOOTSTRAP, seed=0):
    """95% percentile interval of stat over resamples of the (a, b) pairs."""
    rng = random.Random(seed)
    values = sorted(stat([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(n))
    return values[int(0.025 * n)], values[int(0.975 * n) - 1]


def number(row, field):
    v = row.get(field, "")
    return None if v in ("", "None", None) else float(v)


def med(rows, field):
    values = [number(r, field) for r in rows]
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:,.{digits}f}"


def accuracy(rows):
    return 100 * sum(int(r["correct"]) for r in rows) / len(rows) if rows else 0.0


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def summarize_run(rows):
    total_cost = sum(float(r["cost_usd"]) for r in rows)
    out = [f"{len(rows)} questions, {accuracy(rows):.1f}% correct ({sum(int(r['correct']) for r in rows)}), "
           f"no answer counted as wrong.", ""]
    kinds = Counter(r["no_answer_kind"] for r in rows if r["no_answer_kind"])
    out.append("No answer: " + (", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none") + ".")
    out.append("How answers were obtained: "
               + ", ".join(f"{k or 'none'} {v}" for k, v in Counter(r["answer_source"] for r in rows).most_common()) + ".")
    failed = sum(int(r["failed_calls"]) for r in rows)
    out += [f"Model calls that failed: {failed}.", ""]
    out += table(["per question", "median", "mean", "total"], [
        [label, fmt(med(rows, f)), fmt(statistics.mean(float(r[f]) for r in rows)), fmt(sum(float(r[f]) for r in rows))]
        for label, f in (("cost, all ($)", "cost_usd"), ("cost, main agent ($)", "cost_main_usd"),
                         ("cost, orchestrator ($)", "cost_orchestrator_usd"), ("wall time (s)", "wall_s"),
                         ("tool calls", "n_tool_calls"), ("images sent", "n_read_image"),
                         ("main agent calls", "main_calls"), ("orchestrator calls", "orchestrator_calls"),
                         ("all input tokens", "all_input_tokens"), ("cache read tokens", "cache_read_tokens"),
                         ("output tokens", "output_tokens"))
    ])
    if total_cost == 0:
        out += ["", "**All costs are zero: the endpoint reported no usage.**"]
    asr = [number(r, "asr_s") for r in rows]
    asr = [v for v in asr if v is not None]
    if asr:
        out += ["", f"ASR, done ahead of the run and not in wall time: {sum(asr) / 3600:.1f} h in all, "
                    f"median {statistics.median(asr):.0f}s per question."]
    out += ["", "By video length:", ""]
    out += table(["length", "n", "correct %", "no answer", "median cost ($)", "median wall (s)"], [
        [b, len(g), f"{accuracy(g):.1f}", sum(int(r["no_answer"]) for r in g), fmt(med(g, "cost_usd")), fmt(med(g, "wall_s"), 0)]
        for b in BINS for g in [[r for r in rows if r["duration_bin"] == b]] if g
    ])
    return out


def paired_rows(rows, base):
    by_id = {r["item_id"]: r for r in base}
    return [(by_id[r["item_id"]], r) for r in rows if r["item_id"] in by_id]


def paired_accuracy(pairs):
    both_right = sum(1 for b, r in pairs if int(b["correct"]) and int(r["correct"]))
    only_run = sum(1 for b, r in pairs if not int(b["correct"]) and int(r["correct"]))
    only_base = sum(1 for b, r in pairs if int(b["correct"]) and not int(r["correct"]))
    diff = 100 * (only_run - only_base) / len(pairs)
    low, high = bootstrap_interval(pairs, lambda s: 100 * sum(int(r["correct"]) - int(b["correct"]) for b, r in s) / len(s))
    return both_right, only_run, only_base, diff, low, high


def compare(rows, base):
    pairs = paired_rows(rows, base)
    if not pairs:
        return ["No question is in both runs."]
    base_rows = [b for b, _ in pairs]
    run_rows = [r for _, r in pairs]
    both_right, only_run, only_base, diff, low, high = paired_accuracy(pairs)
    out = [f"Paired on {len(pairs)} questions.", ""]
    out.append(f"Correct: this run {accuracy(run_rows):.1f}%, baseline {accuracy(base_rows):.1f}%; difference "
               f"{diff:+.1f} points, 95% interval {low:+.1f} to {high:+.1f}.")
    out.append(f"Both right {both_right}; only this run {only_run}; only the baseline {only_base}; "
               f"exact McNemar p = {mcnemar_p(only_run, only_base):.3f}.")
    finished = [(b, r) for b, r in pairs if not int(b["no_answer"]) and not int(r["no_answer"])]
    out += ["", f"Finished in both: {len(finished)} questions.", ""]
    metric_rows = []
    for label, field in (("cost ($)", "cost_usd"), ("wall time (s)", "wall_s"), ("tool calls", "n_tool_calls"), ("images", "n_read_image")):
        valid = [(number(b, field), number(r, field)) for b, r in finished]
        valid = [(x, y) for x, y in valid if x is not None and y is not None]
        if not valid:
            continue
        if sum(x for x, _ in valid) <= 0:
            continue

        def change(sample):
            denominator = sum(x for x, _ in sample)
            return 100 * (sum(y for _, y in sample) / denominator - 1) if denominator else 0.0

        ci = bootstrap_interval(valid, change)
        metric_rows.append([label, fmt(statistics.median(y for _, y in valid)), fmt(statistics.median(x for x, _ in valid)),
                            f"{change(valid):+.0f}%", f"{ci[0]:+.0f}% to {ci[1]:+.0f}%"])
    out += table(["", "median, this run", "median, baseline", "change in total", "95% interval"], metric_rows)
    out += ["", "By video length:", ""]
    length_rows = []
    for b in BINS:
        group = [(x, y) for x, y in pairs if y["duration_bin"] == b]
        if not group:
            continue
        _, only_r, only_b, d, lo, hi = paired_accuracy(group)
        length_rows.append([b, len(group), f"{accuracy([y for _, y in group]):.1f}", f"{accuracy([x for x, _ in group]):.1f}",
                            f"{d:+.1f}", f"{lo:+.1f} to {hi:+.1f}", f"{only_r} / {only_b}"])
    out += table(["length", "n", "this run %", "baseline %", "difference", "95% interval", "only this run / only baseline"], length_rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--baseline", help="a CSV with item_id, correct and no_answer, paired by question")
    ap.add_argument("--transcripts", help="folder of transcripts, to report ASR time")
    ap.add_argument("--prices", default="4,5,0.2,20", help="input,cache write,cache read,output, $ per million tokens")
    a = ap.parse_args()
    prices = [float(x) for x in a.prices.split(",")]
    if len(prices) != 4:
        ap.error("--prices takes four numbers")

    with open(a.items, encoding="utf-8") as f:
        items = {i["item_id"]: i for i in (json.loads(line) for line in f if line.strip())}
    rows = []
    for path in sorted(glob.glob(os.path.join(a.run, "*", "result.json"))):
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        item = items[result["item_id"]]
        asr_s = None
        if a.transcripts:
            transcript = os.path.join(a.transcripts, os.path.splitext(os.path.basename(item["video"]))[0] + ".json")
            if os.path.exists(transcript):
                with open(transcript, encoding="utf-8") as f:
                    asr_s = json.load(f).get("elapsed_s")
        rows.append(score_session(item, result, prices, asr_s))
    if not rows:
        ap.error(f"no result.json under {a.run}")

    with open(os.path.join(a.run, "sessions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# {os.path.basename(os.path.normpath(a.run))}", ""] + summarize_run(rows)
    if a.baseline:
        with open(a.baseline, newline="", encoding="utf-8") as f:
            base = list(csv.DictReader(f))
        lines += ["", f"## Against {os.path.basename(a.baseline)}", ""] + compare(rows, base)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.run, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
