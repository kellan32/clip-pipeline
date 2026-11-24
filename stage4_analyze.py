"""Stage 4: ask Claude whether each chat spike is a good clip.

For each spike, sends the transcript + chat from N seconds before to N seconds after,
together with the instructions in prompt.txt. Claude replies with JSON:
clippable, score, start_seconds, end_seconds, reason, hook_text, caption.

Needs the ANTHROPIC_API_KEY environment variable.

Usage:  python3 stage4_analyze.py <folder ID from stage 1>
"""
import json
import os
import sys

import anthropic

from common import ROOT, banner, hms, load_config, load_json, save_json, work_dir

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "clippable": {"type": "boolean"},
        "score": {"type": "integer"},
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "reason": {"type": "string"},
        "hook_text": {"type": "string"},
        "caption": {"type": "string"},
    },
    "required": ["clippable", "score", "start_seconds", "end_seconds",
                 "reason", "hook_text", "caption"],
    "additionalProperties": False,
}


def build_moment_text(spike, transcript, chat, before, after, duration):
    lo = max(0, spike["time"] - before)
    hi = min(duration, spike["time"] + after)
    lines = [f"Stream moment from {lo:.0f}s to {hi:.0f}s "
             f"(chat spiked at {spike['time']:.0f}s).", "", "TRANSCRIPT:"]
    said = [s for s in transcript if s["end"] >= lo and s["start"] <= hi]
    lines += [f"[{s['start']:.1f}s] {s['text']}" for s in said] or ["(no speech)"]
    lines += ["", "CHAT:"]
    msgs = [m for m in chat if lo <= m["t"] <= hi]
    lines += [f"[{m['t']:.0f}s] {m['user']}: {m['text']}" for m in msgs] or ["(no chat)"]
    return "\n".join(lines), lo, hi


def ask_claude(client, cfg, instructions, moment_text):
    response = client.messages.create(
        model=cfg["model"],
        max_tokens=16000,
        system=instructions,
        messages=[{"role": "user", "content": moment_text}],
        output_config={"format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to judge this moment")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 stage4_analyze.py <folder ID, e.g. kick_01a1ed52>")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY isn't set. Add it to your environment variables first.")
    cfg = load_config()
    folder = work_dir(cfg, sys.argv[1])
    instructions = (ROOT / cfg["prompt_file"]).read_text()
    transcript = load_json(folder / "transcript.json")
    chat = load_json(folder / "chat.json")
    spikes = load_json(folder / "spikes.json")
    duration = load_json(folder / "info.json")["duration"]

    out_path = folder / "analysis.json"
    results = load_json(out_path) if out_path.exists() else {}  # resume if re-run
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    tokens_in = tokens_out = 0

    for spike in spikes:
        key = str(spike["time"])
        if key in results and "error" not in results[key]:
            continue
        moment, lo, hi = build_moment_text(spike, transcript, chat,
                                           cfg["seconds_before_spike"],
                                           cfg["seconds_after_spike"], duration)
        print(f"  Spike #{spike['rank']} at {hms(spike['time'])}... ", end="", flush=True)
        try:
            verdict, usage = ask_claude(client, cfg, instructions, moment)
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            # Keep the cut points sensible: inside the stream, end after start.
            verdict["start_seconds"] = max(0.0, min(float(verdict["start_seconds"]), duration))
            verdict["end_seconds"] = max(verdict["start_seconds"] + 1,
                                         min(float(verdict["end_seconds"]), duration))
            verdict["spike_time"] = spike["time"]
            results[key] = verdict
            print(f"score {verdict['score']}/10")
        except anthropic.AuthenticationError:
            raise SystemExit("\nThe API key was rejected. Check ANTHROPIC_API_KEY is correct.")
        except (anthropic.APIError, RuntimeError, json.JSONDecodeError, StopIteration) as e:
            results[key] = {"spike_time": spike["time"], "error": str(e)}
            print(f"failed ({e})")
        save_json(out_path, results)

    good = [r for r in results.values() if "error" not in r]
    keep = sorted([r for r in good if r["clippable"] and r["score"] >= cfg["min_score"]],
                  key=lambda r: -r["score"])
    banner("STAGE 4 DONE — Claude's verdicts")
    print(f"  Moments judged: {len(good)}   Failed: {len(results) - len(good)}")
    print(f"  Clips scoring {cfg['min_score']}+: {len(keep)}")
    for r in keep:
        print(f"    {r['score']}/10  {hms(r['start_seconds'])}-{hms(r['end_seconds'])}  "
              f"\"{r['hook_text']}\"")
    if tokens_in:
        print(f"  Tokens used this run: {tokens_in} in / {tokens_out} out")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
