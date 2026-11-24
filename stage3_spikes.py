"""Stage 3: find the moments where chat goes wild.

Counts chat messages in fixed windows (default 10 seconds). Messages that are hype
emotes/words (W, KEKW, LUL, ...) count extra. The busiest windows become "spikes".

Usage:  python3 stage3_spikes.py <folder ID from stage 1>
"""
import re
import sys

from common import banner, hms, load_config, load_json, save_json, work_dir

EMOTE_RE = re.compile(r"\[emote:\d+:([^\]]+)\]")


def is_hype(text, hype_words):
    """True if the message contains a hype word or hype emote."""
    for name in EMOTE_RE.findall(text):
        if any(name.lower().endswith(w) for w in hype_words):
            return True
    plain = EMOTE_RE.sub(" ", text).lower()
    tokens = re.findall(r"[a-z0-9?]+|[^\sa-z0-9]", plain)
    return any(t in hype_words for t in tokens) or any(
        w in plain for w in hype_words if not w.isascii())  # emoji like 😂 💀


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 stage3_spikes.py <folder ID, e.g. kick_01a1ed52>")
    cfg = load_config()
    folder = work_dir(cfg, sys.argv[1])
    chat = load_json(folder / "chat.json")
    duration = load_json(folder / "info.json")["duration"]

    win = cfg["window_seconds"]
    hype_words = {str(w).lower() for w in cfg["emotes"]}
    n_windows = int(duration // win) + 1
    counts = [0] * n_windows
    hype = [0] * n_windows
    for m in chat:
        i = int(m["t"] // win)
        if 0 <= i < n_windows:
            counts[i] += 1
            if is_hype(m["text"], hype_words):
                hype[i] += 1

    # Score: every message = 1, hype messages = emote_weight instead.
    w = cfg["emote_weight"]
    scores = [counts[i] - hype[i] + hype[i] * w for i in range(n_windows)]

    # Pick the top windows, skipping any too close to one already picked.
    picked = []
    for i in sorted(range(n_windows), key=lambda i: scores[i], reverse=True):
        if scores[i] <= 0 or len(picked) >= cfg["top_spikes"]:
            break
        t = i * win
        if all(abs(t - p["time"]) >= cfg["min_gap_seconds"] for p in picked):
            picked.append({"time": t, "score": round(scores[i], 1),
                           "messages": counts[i], "hype_messages": hype[i]})

    avg = sum(scores) / n_windows
    for rank, p in enumerate(picked, 1):
        p["rank"] = rank
        p["times_normal"] = round(p["score"] / avg, 1) if avg else 0
    save_json(folder / "spikes.json", picked)

    banner("STAGE 3 DONE — chat spikes")
    print(f"  Chat messages: {len(chat)}   ({sum(hype)} counted as hype)")
    print(f"  Average activity: {avg:.1f} per {win}s window")
    print(f"  Top {len(picked)} spikes (spaced at least {cfg['min_gap_seconds']}s apart):\n")
    print("   #   time     msgs  hype  vs normal")
    for p in picked:
        bar = "█" * int(p["score"])
        print(f"  {p['rank']:>2}  {hms(p['time'])}   {p['messages']:>3}   {p['hype_messages']:>3}   "
              f"{p['times_normal']:>4}x  {bar}")
    print(f"\n  Saved to: {folder / 'spikes.json'}")


if __name__ == "__main__":
    main()
