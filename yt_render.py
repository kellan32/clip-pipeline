"""YouTube pipeline, step 4: render the vertical clips, thumbnails and clips.csv.

The picture is scaled up to 1920 px tall first (Lanczos), then a 1080-wide window is cut out at
the position the camera path says — done in that order so the window can move in steps finer than
one source pixel, which keeps slow pans smooth. No captions, music, zooms or effects are added.
"""
import csv
import json
import re
import subprocess

import numpy as np

from common import hms


def _even(v):
    return int(round(v / 2)) * 2


def loudnorm_filter(src, start, length, target):
    """Two-pass loudness normalisation: measure first, then apply a clean linear gain."""
    base = f"loudnorm=I={target}:TP=-1.5:LRA=11"
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
         "-vn", "-af", base + ":print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr)
    if not m:
        return base
    s = json.loads(m.group(0))
    if s["input_i"] in ("-inf", "inf") or float(s["input_i"]) < -70:   # silence
        return None
    return (f"{base}:measured_I={s['input_i']}:measured_TP={s['input_tp']}"
            f":measured_LRA={s['input_lra']}:measured_thresh={s['input_thresh']}"
            f":offset={s['target_offset']}:linear=true")


def render_clip(info, clip, track, area, cfg, out_path, tmp_dir):
    bx, by, bw, bh = area
    W, H = cfg["width"], cfg["height"]
    src_fps = info["fps"]
    fps = min(src_fps, cfg["max_fps"])
    length = clip["end"] - clip["start"]

    vf = []
    if (bw, bh) != (info["width"], info["height"]):
        vf.append(f"crop={bw}:{bh}:{bx}:{by}")       # remove baked-in black bars
    if src_fps > cfg["max_fps"] + 0.5:
        vf.append(f"fps={cfg['max_fps']}")
    flags = "lanczos+accurate_rnd+full_chroma_int"
    sw = _even(bw * H / bh)                            # picture width once scaled to 1920 tall
    if sw <= W:
        # Source is 9:16 or narrower: fill the width and centre crop vertically.
        vf += [f"scale={W}:-2:flags={flags}", f"crop={W}:{H}"]
    else:
        ts = np.array(track["t"])
        xs = np.array(track["x"])
        n_frames = int(np.ceil(length * fps)) + 2
        ft = np.arange(n_frames) / fps
        centre = np.interp(ft, ts, xs) if len(ts) > 1 else np.full(n_frames, xs[0])
        left = np.clip(np.round(centre * sw - W / 2), 0, sw - W).astype(int)
        cmd_file = tmp_dir / (out_path.stem + "_crop.txt")
        lines, last = [], None
        for t, x in zip(ft, left):
            if x != last:
                lines.append(f"{t:.4f} crop@follow x {x};")
                last = x
        cmd_file.write_text("\n".join(lines) + "\n")
        vf += [f"sendcmd=f={cmd_file.name}",
               f"scale={sw}:{H}:flags={flags}",
               f"crop@follow=w={W}:h={H}:x={left[0]}:y=0:exact=1"]
    vf += ["setsar=1", "format=yuv420p"]

    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{clip['start']:.3f}",
           "-i", info["source_path"]]
    af = None
    if info["has_audio"]:
        af = loudnorm_filter(info["source_path"], clip["start"], length, cfg["loudness"])
    if af is None:   # no (or silent) audio: add a silent track so every clip plays the same way
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        amap = ["-map", "0:v:0", "-map", "0:a:0", "-af", af]
    color = []
    if info.get("color_primaries") == "bt709" or (info.get("color_space") or "").startswith("bt709"):
        color = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    gop = int(round(fps * 2))
    cmd += ["-t", f"{length:.3f}", "-vf", ",".join(vf), *amap,
            "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(gop), "-r", f"{fps:.3f}",
            *color,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", "-shortest", str(out_path)]
    subprocess.run(cmd, check=True, cwd=tmp_dir)


def thumbnail(clip_path, t, out_path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(clip_path),
                    "-frames:v", "1", "-q:v", "2", str(out_path)], check=True)


def write_csv(path, rows):
    cols = ["clip", "file", "thumbnail", "start", "end", "start_seconds", "end_seconds",
            "duration_seconds", "score", "crop", "source_url", "channel", "channel_url",
            "video_title"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def stamp(seconds):
    """1:02:03.45 style (hms() from common, plus hundredths)."""
    cs = int(round(seconds * 100))
    return f"{hms(cs // 100)}.{cs % 100:02d}"
