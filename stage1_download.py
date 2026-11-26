"""Stage 1: download a VOD's video, audio and chat.

Usage:  python3 stage1_download.py <VOD link>
"""
import datetime as dt
import re
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
        # Download HLS streams with yt-dlp itself (works reliably behind the session proxy).
        "external_downloader": {"m3u8": "native"},
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

TWITCH_GQL = "https://gql.twitch.tv/gql"
# Twitch has no official API for VOD chat replay; this is the public client ID yt-dlp uses.
TWITCH_CLIENT_ID = "ue6666qo983tsx6so1t0vnawi233wa"
TWITCH_COMMENTS_QUERY = """
query($id: ID!, $offset: Int, $cursor: Cursor) {
  video(id: $id) {
    comments(contentOffsetSeconds: $offset, after: $cursor) {
      edges { cursor node { contentOffsetSeconds commenter { login } message { fragments { text } } } }
      pageInfo { hasNextPage }
    }
  }
}"""


def download_twitch_chat(info, folder):
    """Page through a Twitch VOD's chat replay (about 1 request per second)."""
    chat_path = folder / "chat.json"
    if chat_path.exists():
        print("Chat already downloaded, skipping.")
        return load_json(chat_path)

    partial_path = folder / "chat.partial.json"
    if partial_path.exists():  # resume an interrupted download
        p = load_json(partial_path)
        messages, cursor = p["messages"], p["next"]
        print(f"Resuming chat download ({len(messages)} messages so far)...")
    else:
        messages, cursor = [], None
        print("Downloading chat (about 1 request per second)...")
    vod = str(info["id"]).lstrip("v")
    session = requests.Session()
    session.headers.update({"Client-ID": TWITCH_CLIENT_ID, "Content-Type": "text/plain;charset=UTF-8"})
    requests_since_save = 0
    while True:
        variables = {"id": vod, "cursor": cursor} if cursor else {"id": vod, "offset": 0}
        for attempt in range(8):
            try:
                r = session.post(TWITCH_GQL, json={"query": TWITCH_COMMENTS_QUERY, "variables": variables},
                                 timeout=20)
                if r.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"\n  Twitch asked us to slow down, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                body = r.json()
                if body.get("errors"):
                    raise requests.RequestException(body["errors"][0].get("message"))
                comments = body["data"]["video"]["comments"]
                break
            except (requests.RequestException, KeyError, TypeError, ValueError) as e:
                print(f"\n  Network hiccup ({e}), retrying...")
                time.sleep(5 * (attempt + 1))
        else:
            save_json(partial_path, {"messages": messages, "next": cursor})
            raise SystemExit("Chat download kept failing. "
                             "Progress is saved: run the same command again to resume.")
        if comments is None:
            break  # VOD has no chat replay
        edges = comments["edges"] or []
        for e in edges:
            n = e["node"]
            messages.append({
                "t": float(n["contentOffsetSeconds"]),
                "user": (n.get("commenter") or {}).get("login", "?"),
                "text": "".join(f.get("text", "") for f in n["message"]["fragments"]),
            })
        if edges:
            cursor = edges[-1]["cursor"]
            done = edges[-1]["node"]["contentOffsetSeconds"] / info["duration"]
            print(f"\r  {min(done, 1):.0%}  ({len(messages)} messages)", end="", flush=True)
        if not comments["pageInfo"]["hasNextPage"] or not edges or not cursor:
            break
        requests_since_save += 1
        if requests_since_save >= 25:
            save_json(partial_path, {"messages": messages, "next": cursor})
            requests_since_save = 0
        time.sleep(1.0)
    print()
    messages.sort(key=lambda m: m["t"])
    save_json(chat_path, messages)
    partial_path.unlink(missing_ok=True)
    return messages


def twitch_gql(session, query, variables):
    """One Twitch GraphQL request with a few retries. Returns the "data" part."""
    for attempt in range(5):
        try:
            r = session.post(TWITCH_GQL, json={"query": query, "variables": variables}, timeout=20)
            r.raise_for_status()
            body = r.json()
            if body.get("errors"):
                raise requests.RequestException(body["errors"][0].get("message"))
            return body["data"]
        except (requests.RequestException, ValueError) as e:
            print(f"\n  Network hiccup ({e}), retrying...")
            time.sleep(3 * (attempt + 1))
    return None


CLIP_FIELDS = "slug title viewCount durationSeconds videoOffsetSeconds video { id }"
CHANNEL_CLIPS_QUERY = """
query($login: String!, $period: ClipsPeriod, $cursor: Cursor) {
  user(login: $login) {
    clips(first: 100, after: $cursor, criteria: {period: $period, sort: VIEWS_DESC}) {
      edges { cursor node { %s } }
      pageInfo { hasNextPage }
    }
  }
}""" % CLIP_FIELDS
CLIP_BY_SLUG_QUERY = "query($slug: ID!) { clip(slug: $slug) { %s } }" % CLIP_FIELDS
CLIP_LINK_RE = re.compile(r"(?:clips\.twitch\.tv/|twitch\.tv/[^/\s]+/clip/)([A-Za-z0-9_-]+)")


def download_twitch_viewer_clips(info, folder, chat):
    """Find the clips viewers made of this VOD, with their view counts.

    Viewer clips are the strongest sign a moment landed: someone cared enough to clip it,
    and the view count shows how far it spread. Two sources: the channel's most-viewed
    clips, and clip links people pasted in chat.
    """
    clips_path = folder / "viewer_clips.json"
    if clips_path.exists():
        print("Viewer clips already fetched, skipping.")
        return load_json(clips_path)
    print("Looking up clips viewers made of this stream...")
    vod = str(info["id"]).lstrip("v")
    session = requests.Session()
    session.headers.update({"Client-ID": TWITCH_CLIENT_ID, "Content-Type": "text/plain;charset=UTF-8"})
    found = {}

    def keep(node):
        if node and (node.get("video") or {}).get("id") == vod and node.get("videoOffsetSeconds") is not None:
            found[node["slug"]] = {
                "start": node["videoOffsetSeconds"], "duration": node["durationSeconds"],
                "views": node["viewCount"], "title": node["title"], "slug": node["slug"]}

    data = twitch_gql(session, "query($id: ID!) { video(id: $id) { owner { login } } }", {"id": vod})
    login = (((data or {}).get("video") or {}).get("owner") or {}).get("login")
    if login:
        age_days = (time.time() - info["timestamp"]) / 86400
        period = "LAST_WEEK" if age_days < 6 else "LAST_MONTH" if age_days < 29 else "ALL_TIME"
        cursor = None
        for _ in range(15):  # up to 1,500 of the channel's top clips
            data = twitch_gql(session, CHANNEL_CLIPS_QUERY, {"login": login, "period": period, "cursor": cursor})
            clips = ((data or {}).get("user") or {}).get("clips")
            if not clips or not clips["edges"]:
                break
            for e in clips["edges"]:
                keep(e["node"])
            cursor = clips["edges"][-1]["cursor"]
            if not clips["pageInfo"]["hasNextPage"] or not cursor:
                break
            time.sleep(0.5)

    linked = {m for msg in chat for m in CLIP_LINK_RE.findall(msg["text"])} - set(found)
    for slug in list(linked)[:300]:
        data = twitch_gql(session, CLIP_BY_SLUG_QUERY, {"slug": slug})
        keep((data or {}).get("clip"))
        time.sleep(0.3)

    clips = sorted(found.values(), key=lambda c: -c["views"])
    save_json(clips_path, clips)
    print(f"  Found {len(clips)} viewer clips of this stream"
          + (f" (top one: {clips[0]['views']} views)" if clips else ""))
    return clips


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
        chat = download_twitch_chat(info, folder)
        viewer_clips = download_twitch_viewer_clips(info, folder, chat)

    size_mb = (folder / "video.mp4").stat().st_size / 1e6
    busiest = max(range(0, int(info["duration"]), 60),
                  key=lambda s: sum(1 for m in chat if s <= m["t"] < s + 60), default=0)
    banner("STAGE 1 DONE — download")
    print(f"  Stream:     {info['title']}  ({info['channel']})")
    print(f"  Length:     {hms(info['duration'])}")
    print(f"  Video:      {size_mb:.0f} MB")
    print(f"  Chat:       {len(chat)} messages from {len({m['user'] for m in chat})} chatters")
    if not info["extractor"].startswith("kick"):
        print(f"  Viewer clips: {len(viewer_clips)}")
    print(f"  Busiest minute of chat starts at {hms(busiest)}")
    print(f"  Saved in:   {folder}")
    print(f"  Folder ID for next stages: {vod_id}")


if __name__ == "__main__":
    main()
