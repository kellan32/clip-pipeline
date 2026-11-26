"""YouTube pipeline, step 1: get the video (YouTube link or local file) and a fast analysis copy.

Everything lands in work/yt_<video id>/ (or work/local_<name>/) and is reused on the next run.
"""
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from common import ROOT, load_json, save_json

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")


def slugify(text, max_len=60):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-") or "video"


def youtube_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def probe(path):
    """Width, height, fps, duration, colour info and whether there's an audio track."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    if not 1 <= fps <= 240:
        fps = 30.0
    # Phone footage can be stored sideways with a rotation flag.
    rotation = 0
    for sd in v.get("side_data_list", []):
        rotation = int(sd.get("rotation", 0) or 0)
    w, h = int(v["width"]), int(v["height"])
    if abs(rotation) in (90, 270):
        w, h = h, w
    return {
        "width": w, "height": h, "fps": round(fps, 3),
        "duration": float(data["format"].get("duration") or v.get("duration") or 0),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
        "color_space": v.get("color_space"), "color_primaries": v.get("color_primaries"),
        "color_transfer": v.get("color_transfer"), "codec": v.get("codec_name"),
    }


def download_youtube(url, folder, cfg):
    import yt_dlp
    max_h = cfg["max_height"]
    opts = {
        # Best video up to 1080p (any codec) + best audio, merged. Falls back to a single file.
        "format": f"bv*[height<={max_h}]+ba/b[height<={max_h}]/bv*+ba/b",
        "format_sort": [f"res:{max_h}", "fps", "vbr"],
        "outtmpl": str(folder / "source.%(ext)s"),
        "merge_output_format": "mkv",   # mkv holds any codec mix without re-encoding
        "quiet": True, "no_warnings": True, "noprogress": True,
        "noplaylist": True,
        # Use the system certificate store (needed behind HTTPS proxies with their own CA).
        "compat_opts": {"no-certifi"},
    }
    print("Downloading video (best quality up to "
          f"{max_h}p — this can take a few minutes)...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.sanitize_info(ydl.extract_info(url, download=True))
    return {
        "id": info.get("id"), "title": info.get("title") or info.get("id"),
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "webpage_url": info.get("webpage_url") or url,
        "upload_date": info.get("upload_date") or "",
    }


def find_source(folder):
    files = [p for p in folder.glob("source.*") if p.suffix.lower() in VIDEO_EXTS]
    return files[0] if files else None


def get_source(arg, cfg, wcfg, title=None, channel=None, source_url=None):
    """Return (work folder, info dict). `arg` is a YouTube link or a local video path."""
    local = Path(arg).expanduser()
    if local.exists() and local.is_file():
        local = local.resolve()
        tag = hashlib.sha1(str(local).encode()).hexdigest()[:6]
        vid = f"local_{slugify(local.stem, 30)}_{tag}"
        folder = ROOT / wcfg["work_folder"] / vid
        folder.mkdir(parents=True, exist_ok=True)
        info_path = folder / "info.json"
        info = load_json(info_path) if info_path.exists() else {
            "id": vid, "title": local.stem, "channel": "", "channel_url": "",
            "webpage_url": "", "upload_date": ""}
        info["source_path"] = str(local)
        print(f"Using local file: {local}")
    else:
        if not re.match(r"https?://", arg):
            raise SystemExit(f"'{arg}' isn't a file that exists or a web link.")
        yid = youtube_id(arg)
        vid = f"yt_{yid}" if yid else "web_" + hashlib.sha1(arg.encode()).hexdigest()[:10]
        folder = ROOT / wcfg["work_folder"] / vid
        folder.mkdir(parents=True, exist_ok=True)
        info_path = folder / "info.json"
        src = find_source(folder)
        if src and info_path.exists():
            print("Video already downloaded, skipping.")
            info = load_json(info_path)
        else:
            try:
                info = download_youtube(arg, folder, cfg)
            except Exception as e:  # yt-dlp raises its own error types
                raise SystemExit(
                    f"\nCouldn't download that video:\n  {e}\n\n"
                    "If downloads are blocked here, download the video yourself and run:\n"
                    "  python3 yt_clips.py /path/to/video.mp4 --source-url <link> --channel \"<name>\"")
            src = find_source(folder)
            if not src:
                raise SystemExit("yt-dlp finished but no video file was saved.")
        info["source_path"] = str(src)

    # Command-line credits override whatever we know (handy for local files).
    if title:
        info["title"] = title
    if channel:
        info["channel"] = channel
    if source_url:
        info["webpage_url"] = source_url
    info.update(probe(info["source_path"]))
    save_json(folder / "info.json", info)
    return folder, info


def make_proxy(folder, info):
    """Small, quick-to-read copy used for scene detection and scoring (same timeline)."""
    proxy = folder / "proxy.mp4"
    if proxy.exists() and proxy.stat().st_mtime >= Path(info["source_path"]).stat().st_mtime:
        return proxy
    fps = min(info["fps"], 30)
    print("Making a small analysis copy of the video...")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", info["source_path"], "-an",
         "-vf", f"fps={fps},scale=960:540:force_original_aspect_ratio=decrease,"
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-g", str(int(fps)),
         str(proxy) + ".part.mp4"], check=True)
    Path(str(proxy) + ".part.mp4").rename(proxy)
    return proxy


def detect_letterbox(info, samples=12):
    """Find black bars baked into the picture. Returns (x, y, w, h) of the real picture area."""
    W, H, dur = info["width"], info["height"], info["duration"]
    found = []
    for i in range(samples):
        t = dur * (i + 0.5) / samples
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-ss", f"{t:.2f}", "-i", info["source_path"], "-t", "1",
             "-vf", "cropdetect=limit=0.09:round=2:reset=0", "-an", "-f", "null", "-"],
            capture_output=True, text=True)
        crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
        if crops:
            found.append(tuple(int(v) for v in crops[-1]))
    full = (0, 0, W, H)
    if not found:
        return full
    (cw, ch, cx, cy), n = Counter(found).most_common(1)[0]
    # Only trust it if most samples agree and the bars are real (>2% of the frame).
    if n < 0.75 * len(found) or (cw > 0.98 * W and ch > 0.98 * H) or cw < W / 2 or ch < H / 2:
        return full
    return (cx, cy, cw, ch)
