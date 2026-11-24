"""Stage 1: download a VOD's video, audio and chat.

Usage:  python3 stage1_download.py <VOD link>
"""
import datetime as dt
import subprocess
import sys
import time

import requests
import yt_dlp

from common import banner, hms, load_config, load_json, save_json, vod_id_from_url, work_dir


def download_video(url, folder, cfg):
    info_path = folder / "info.json"
    video_path = folder / "video.mp4"
    opts = {
        "format": f"best[height<={cfg['video_quality']}]/best",
        "outtmpl": str(folder / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Use the system certificate store (needed behind HTTPS proxies with their own CA).
        "compat_opts": {"no-certifi"},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        if video_path.exists() and info_path.exists():
            print("Video already downloaded, skipping.")
            return load_json(info_path)
        print("Downloading video (this can take several minutes)...")
        info = ydl.extract_info(url, download=True)
        info = ydl.sanitize_info(info)
    save_json(info_path, {k: info.get(k) for k in (
        "id", "title", "channel", "channel_id", "timestamp", "duration", "webpage_url", "extractor")})
    return load_json(info_path)


def extract_audio(folder):
    audio_path = folder / "audio.wav"
    if audio_path.exists():
        print("Audio already extracted, skipping.")
        return
    print("Extracting audio...")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(folder / "video.mp4"),
                    "-vn", "-ac", "1", "-ar", "16000", str(audio_path)], check=True)


def download_kick_chat(info, folder):
    """Kick's chat replay returns ~5 seconds of messages per request, so step through the VOD."""
    chat_path = folder / "chat.json"
    if chat_path.exists():
        print("Chat already downloaded, skipping.")
        return load_json(chat_path)

    start = dt.datetime.fromtimestamp(info["timestamp"], dt.UTC)
    end = start + dt.timedelta(seconds=info["duration"])
    url = f"https://kick.com/api/v2/channels/{info['channel_id']}/messages"
    partial_path = folder / "chat.partial.json"
    if partial_path.exists():  # resume an interrupted download
        p = load_json(partial_path)
        messages, t = p["messages"], dt.datetime.fromisoformat(p["next"])
        print(f"Resuming chat download from {hms((t - start).total_seconds())}...")
    else:
        messages, t = [], start
        print("Downloading chat (about 1 request per second to stay under Kick's limits)...")
    seen = {(m["t"], m["user"], m["text"]) for m in messages}
    session = requests.Session()
    requests_since_save = 0
    while t < end:
        stamp = t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        for attempt in range(8):
            try:
                r = session.get(url, params={"start_time": stamp}, timeout=20)
                if r.status_code == 429:  # Kick says slow down: wait and retry
                    wait = int(r.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
                    print(f"\n  Kick asked us to slow down, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                batch = r.json()["data"]["messages"]
                break
            except requests.RequestException as e:
                print(f"\n  Network hiccup ({e}), retrying...")
                time.sleep(5 * (attempt + 1))
        else:
            save_json(partial_path, {"messages": messages, "next": t.isoformat()})
            raise SystemExit(f"Chat download kept failing at {stamp}. "
                             "Progress is saved: run the same command again to resume.")
        for m in batch:
            created = dt.datetime.fromisoformat(m["created_at"].replace("Z", "+00:00"))
            msg = {
                "t": round((created - start).total_seconds(), 1),
                "user": m["sender"]["username"],
                "text": m["content"],
            }
            key = (msg["t"], msg["user"], msg["text"])
            if key in seen:
                continue
            seen.add(key)
            messages.append(msg)
        # A very busy 5-second window may be cut short; if so, continue from the last message.
        last = (dt.datetime.fromisoformat(batch[-1]["created_at"].replace("Z", "+00:00"))
                if batch else None)
        if len(batch) >= 25 and last and last > t:
            t = last
        else:
            t += dt.timedelta(seconds=5)
        done = (t - start).total_seconds() / info["duration"]
        print(f"\r  {min(done, 1):.0%}  ({len(messages)} messages)", end="", flush=True)
        requests_since_save += 1
        if requests_since_save >= 25:
            save_json(partial_path, {"messages": messages, "next": t.isoformat()})
            requests_since_save = 0
        time.sleep(1.0)  # stay under Kick's rate limit
    print()
    messages.sort(key=lambda m: m["t"])
    save_json(chat_path, messages)
    partial_path.unlink(missing_ok=True)
    return messages


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 stage1_download.py <VOD link>")
    url = sys.argv[1]
    cfg = load_config()
    vod_id = vod_id_from_url(url)
    folder = work_dir(cfg, vod_id)

    info = download_video(url, folder, cfg)
    extract_audio(folder)
    if info["extractor"].startswith("kick"):
        chat = download_kick_chat(info, folder)
    else:
        raise SystemExit("Twitch chat download isn't built yet — Kick only for now.")

    size_mb = (folder / "video.mp4").stat().st_size / 1e6
    busiest = max(range(0, int(info["duration"]), 60),
                  key=lambda s: sum(1 for m in chat if s <= m["t"] < s + 60), default=0)
    banner("STAGE 1 DONE — download")
    print(f"  Stream:     {info['title']}  ({info['channel']})")
    print(f"  Length:     {hms(info['duration'])}")
    print(f"  Video:      {size_mb:.0f} MB")
    print(f"  Chat:       {len(chat)} messages from {len({m['user'] for m in chat})} chatters")
    print(f"  Busiest minute of chat starts at {hms(busiest)}")
    print(f"  Saved in:   {folder}")
    print(f"  Folder ID for next stages: {vod_id}")


if __name__ == "__main__":
    main()
