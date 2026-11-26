"""YouTube training session -> raw 9:16 clips for TikTok (no captions, music or effects).

Usage:
  python3 yt_clips.py <YouTube link or video file> [options]

Options:
  --count N            how many clips (default from config.yaml, 5)
  --length S           target clip length in seconds (default 60; clips stay within 40-75 s)
  --min-length S / --max-length S   change the allowed range
  --start T --end T    only look at this part of the video (T like 1:20 or 1:02:03 or 80)
  --manual "1:20-2:15,4:00-5:05"    cut exactly these ranges instead of auto-picking
  --crop smart|center  follow the players (default) or plain centre crop
  --channel NAME --source-url URL --title TEXT   credits for clips.csv (useful for local files)
  --debug              also save tracking contact sheets in work/<id>/debug/

Works alongside the Kick/Twitch pipeline (run.py) and shares its config.yaml and helpers.
Finished clips land in output/<video-title>/ with a thumbnail each and clips.csv.
Re-running reuses the download and analysis, so trying different options is quick.
"""
import argparse
import re
import shutil
import sys
import time

from common import ROOT, banner, hms, load_config, save_json
import yt_crop
import yt_render
import yt_select
import yt_source


def parse_time(s):
    s = s.strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    if not 2 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise SystemExit(f"Couldn't read the time '{s}'. Use 80, 1:20 or 1:02:03.")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def parse_ranges(text):
    out = []
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        m = re.fullmatch(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*", chunk)
        if not m:
            raise SystemExit(f"Couldn't read the range '{chunk.strip()}'. Use e.g. 1:20-2:15.")
        a, b = parse_time(m.group(1)), parse_time(m.group(2))
        if b <= a:
            raise SystemExit(f"Range '{chunk.strip()}' ends before it starts.")
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description="YouTube training video -> vertical clips",
                                 usage="python3 yt_clips.py <YouTube link or video file> [options]")
    ap.add_argument("source")
    ap.add_argument("--count", type=int)
    ap.add_argument("--length", type=float)
    ap.add_argument("--min-length", type=float)
    ap.add_argument("--max-length", type=float)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--manual")
    ap.add_argument("--crop", choices=["smart", "center"])
    ap.add_argument("--channel")
    ap.add_argument("--source-url")
    ap.add_argument("--title")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    full_cfg = load_config()
    cfg = full_cfg["youtube"]
    count = a.count or cfg["clip_count"]
    target = a.length or cfg["clip_length"]
    min_len = a.min_length or min(cfg["min_length"], target)
    max_len = a.max_length or max(cfg["max_length"], target)
    crop_mode = a.crop or cfg["crop"]
    if count < 1 or not 0 < min_len <= target <= max_len:
        raise SystemExit("Check --count / --length / --min-length / --max-length.")

    banner("STEP 1 — video")
    folder, info = yt_source.get_source(a.source, cfg, full_cfg, a.title, a.channel, a.source_url)
    dur = info["duration"]
    print(f"  {info['title']}  ({info['channel'] or 'channel unknown'})")
    print(f"  {info['width']}x{info['height']} @ {info['fps']:g} fps, {hms(dur)}, "
          f"audio: {'yes' if info['has_audio'] else 'no'}")
    if info["height"] < 1080 and info["width"] > info["height"]:
        print(f"  Note: source is only {info['height']}p, so the vertical crop is upscaled "
              f"{1920 / info['height']:.1f}x and will look softer than a 1080p source.")
    area = yt_source.detect_letterbox(info)
    if area != (0, 0, info["width"], info["height"]):
        print(f"  Black bars found: using the {area[2]}x{area[3]} picture inside them.")

    lo = parse_time(a.start) if a.start else 0.0
    hi = min(parse_time(a.end), dur) if a.end else dur
    if hi <= lo:
        raise SystemExit("--end must be after --start.")

    data = None
    skipped = []
    if a.manual:
        banner("STEP 2 — your ranges")
        clips = []
        for s, e in parse_ranges(a.manual):
            if s >= dur:
                print(f"  Skipping {hms(s)}-{hms(e)}: the video is only {hms(dur)} long.")
                continue
            e = min(e, dur)
            if not min_len <= e - s <= max_len:
                print(f"  Note: {hms(s)}-{hms(e)} is {e - s:.0f}s (outside {min_len:.0f}-{max_len:.0f}s); "
                      "cutting it anyway.")
            clips.append({"start": s, "end": e, "score": "", "start_on": "manual", "end_on": "manual"})
    else:
        banner("STEP 2 — finding the drills")
        proxy = yt_source.make_proxy(folder, info)
        data = yt_select.analyse(folder, proxy, info, cfg)
        clips, skipped = yt_select.pick_clips(data, cfg, count, target, min_len, max_len, lo, hi)
        for s, e, why in skipped:
            if e > lo and s < hi:
                print(f"  skipping {hms(s)}-{hms(e)} ({why})")
        if not clips:
            raise SystemExit(
                f"\nNo stretch of {min_len:.0f}-{max_len:.0f}s of continuous drill footage found.\n"
                "Try a lower --min-length, or pick ranges yourself with --manual.")
        if len(clips) < count:
            print(f"  Only {len(clips)} good {min_len:.0f}-{max_len:.0f}s stretches found "
                  f"(asked for {count}).")
    if not clips:
        raise SystemExit("Nothing to cut.")

    banner("STEP 3 — following the action")
    model = yt_select.load_yolo(cfg) if crop_mode == "smart" else None
    tracks = []
    for i, c in enumerate(clips, 1):
        length = c["end"] - c["start"]
        if crop_mode == "center":
            tracks.append({"t": [0.0], "x": [0.5], "mode": "center", "thumb_t": length / 2})
            continue
        if data is not None:
            cuts = [x - c["start"] for x in data["cuts"] if c["start"] < x < c["end"]]
        else:
            cuts = [x - c["start"] for x in
                    yt_select.detect_cuts(info["source_path"], c["start"], c["end"], info["fps"])]
        print(f"  clip {i:02d}: {hms(c['start'])}-{hms(c['end'])} ...", end="", flush=True)
        tr = yt_crop.track_clip(info, c, cfg, model, cuts, area,
                                folder / "debug" if a.debug else None, f"clip_{i:02d}")
        tracks.append(tr)
        print(f" {tr['mode']} crop ({tr.get('found', 0):.0%} of samples had players)")

    banner("STEP 4 — rendering")
    out_dir = ROOT / full_cfg["output_folder"] / yt_source.slugify(info["title"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("clip_*.mp4")) + list(out_dir.glob("clip_*.jpg")):
        old.unlink()          # clear clips from an earlier run of this video
    tmp = folder / "render_tmp"
    tmp.mkdir(exist_ok=True)
    rows = []
    for i, (c, tr) in enumerate(zip(clips, tracks), 1):
        name = f"clip_{i:02d}"
        mp4, jpg = out_dir / f"{name}.mp4", out_dir / f"{name}.jpg"
        print(f"  {name}.mp4 ({c['end'] - c['start']:.0f}s)...", flush=True)
        yt_render.render_clip(info, c, tr, area, cfg, mp4, tmp)
        yt_render.thumbnail(mp4, tr["thumb_t"], jpg)
        rows.append({
            "clip": i, "file": mp4.name, "thumbnail": jpg.name,
            "start": yt_render.stamp(c["start"]), "end": yt_render.stamp(c["end"]),
            "start_seconds": round(c["start"], 2), "end_seconds": round(c["end"], 2),
            "duration_seconds": round(c["end"] - c["start"], 2),
            "score": c["score"], "crop": tr["mode"],
            "source_url": info.get("webpage_url", ""), "channel": info.get("channel", ""),
            "channel_url": info.get("channel_url", ""), "video_title": info["title"],
        })
    shutil.rmtree(tmp, ignore_errors=True)
    yt_render.write_csv(out_dir / "clips.csv", rows)
    save_json(folder / "last_run_tracks.json", {"clips": clips, "tracks": tracks})

    banner("DONE — clips ready")
    print(f"  {'clip':<12}{'from':>11}{'to':>11}{'len':>6}  crop")
    for r in rows:
        size = (out_dir / r["file"]).stat().st_size / 1e6
        print(f"  {r['file']:<12}{hms(r['start_seconds']):>11}{hms(r['end_seconds']):>11}"
              f"{r['duration_seconds']:>5.0f}s  {r['crop']:<13}{size:5.1f} MB")
    print(f"\n  Source:  {info.get('webpage_url') or info['source_path']}")
    print(f"  Credit:  {info.get('channel') or '(unknown — pass --channel)'}")
    print(f"  Folder:  {out_dir}")
    print(f"  Log:     {out_dir / 'clips.csv'}")
    if a.debug:
        print(f"  Tracking previews: {folder / 'debug'}")
    print(f"  Took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
