"""Small helpers shared by every stage."""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def vod_id_from_url(url):
    """Turn a Kick or Twitch VOD link into a short folder name."""
    m = re.search(r"kick\.com/[^/]+/videos/([0-9a-f-]+)", url)
    if m:
        return "kick_" + m.group(1)[:8]
    m = re.search(r"twitch\.tv/videos/(\d+)", url)
    if m:
        return "twitch_" + m.group(1)
    raise SystemExit(f"Couldn't recognise this link as a Kick or Twitch VOD: {url}")


def work_dir(cfg, vod_id):
    d = ROOT / cfg["work_folder"] / vod_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def banner(text):
    print("\n" + "=" * 60 + f"\n  {text}\n" + "=" * 60)
