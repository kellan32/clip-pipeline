"""Stage 3: find the moments where chat goes wild.

Counts chat messages in fixed windows (default 10 seconds). Messages that are hype
emotes/words (W, KEKW, LUL, ...) count extra, and "clip it"-type messages count a lot extra.
On Twitch, clips that viewers made of the stream (from stage 1) add a big bonus scaled by
their view count. The highest-scoring windows become "spikes".

Usage:  python3 stage3_spikes.py <folder ID from stage 1>
"""
import math
import re
import sys

from common import banner, hms, load_config, load_json, save_json, work_dir

EMOTE_RE = re.compile(r"\[emote:\d+:([^\]]+)\]")
URL_RE = re.compile(r"\S*https?://\S+|\S*\.(?:tv|com)/\S+")


def clip_call_regex(phrases):
    """Regex matching any of the "clip it" phrases as whole words, e.g. "clip" but not "clipboard"."""
    alts = sorted((re.escape(str(p).lower()).replace("\\ ", r"\s+") for p in phrases), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])")


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
    calls = [0] * n_windows
    call_re = clip_call_regex(cfg.get("clip_phrases") or [])
    for m in chat:
        i = int(m["t"] // win)
        if 0 <= i < n_windows:
            counts[i] += 1
            if call_re.pattern != r"(?<![a-z0-9])(?:)(?![a-z0-9])" and call_re.search(
                    URL_RE.sub(" ", m["text"]).lower()):
                calls[i] += 1
            elif is_hype(m["text"], hype_words):
                hype[i] += 1

    # Score: every message = 1, hype messages = emote_weight, "clip it" messages = clip_call_weight.
    w = cfg["emote_weight"]
    cw = cfg.get("clip_call_weight", 5)
    scores = [counts[i] - hype[i] - calls[i] + hype[i] * w + calls[i] * cw for i in range(n_windows)]

    # Viewer clips (Twitch only): each adds a bonus to every window it covers. The bonus grows
    # fast with views, so one clip with 2,000 views beats a pile of clips nobody watched
    # (1 view ≈ +2, 50 views ≈ +58, 500 views ≈ +146, 2,000 views ≈ +220 with weight 20).
    viewer_clips_path = folder / "viewer_clips.json"
    viewer_clips = load_json(viewer_clips_path) if viewer_clips_path.exists() else []
    vw = cfg.get("viewer_clip_weight", 20)
    for c in viewer_clips:
        first = max(0, int(c["start"] // win))
        last = min(n_windows - 1, int((c["start"] + c["duration"]) // win))
        bonus = vw * math.log10(1 + c["views"]) ** 2
        for i in range(first, last + 1):
            scores[i] += bonus

    # Pick the top windows, skipping any too close to one already picked.
    picked = []
    for i in sorted(range(n_windows), key=lambda i: scores[i], reverse=True):
        if scores[i] <= 0 or len(picked) >= cfg["top_spikes"]:
            break
        t = i * win
        if all(abs(t - p["time"]) >= cfg["min_gap_seconds"] for p in picked):
            near = [c for c in viewer_clips
                    if c["start"] - win <= t <= c["start"] + c["duration"] + win]
            picked.append({"time": t, "score": round(scores[i], 1),
                           "messages": counts[i], "hype_messages": hype[i], "clip_calls": calls[i],
                           "viewer_clips": [{k: c[k] for k in ("start", "duration", "views", "title")}
                                            for c in near]})

    avg = sum(scores) / n_windows
    for rank, p in enumerate(picked, 1):
        p["rank"] = rank
        p["times_normal"] = round(p["score"] / avg, 1) if avg else 0
    save_json(folder / "spikes.json", picked)

    banner("STAGE 3 DONE — chat spikes")
    print(f"  Chat messages: {len(chat)}   ({sum(hype)} counted as hype, {sum(calls)} saying 'clip it')")
    if viewer_clips:
        print(f"  Viewer clips:  {len(viewer_clips)}   (most viewed: {viewer_clips[0]['views']} views "
              f"at {hms(viewer_clips[0]['start'])})")
    print(f"  Average activity: {avg:.1f} per {win}s window")
    print(f"  Top {len(picked)} spikes (spaced at least {cfg['min_gap_seconds']}s apart):\n")
    print("   #   time     msgs  hype  clip  viewer clips  vs normal")
    for p in picked:
        vc = sum(c["views"] for c in p["viewer_clips"])
        vc_txt = f"{len(p['viewer_clips'])} ({vc} views)" if p["viewer_clips"] else "-"
        print(f"  {p['rank']:>2}  {hms(p['time'])}   {p['messages']:>3}   {p['hype_messages']:>3}   "
              f"{p['clip_calls']:>3}  {vc_txt:<13} {p['times_normal']:>5}x")
    print(f"\n  Saved to: {folder / 'spikes.json'}")


if __name__ == "__main__":
    main()
