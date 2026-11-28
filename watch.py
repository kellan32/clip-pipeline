"""Watch streamers and say when a stream has ended and its VOD is ready to clip.

Streamers to watch are listed in config.yaml under watch_streamers, e.g.
  watch_streamers: [twitch:stableronaldo, kick:xqc]

Usage:
  python3 watch.py                 check everyone; prints one line per streamer
  python3 watch.py --done <link>   mark a VOD as handled so it isn't picked up again

Each line starts with one of:
  READY <vod link>   the stream ended, the VOD is finished, and it hasn't been clipped yet
  LIVE               still streaming (the VOD isn't finished yet)
  NOTHING            nothing new
  ERROR              couldn't check (network, unknown streamer...)
A stream that ended more than watch_max_age_hours ago is never READY, so an old VOD isn't
picked up by mistake (for example if the list of handled VODs was lost).
"""
import sys
from datetime import datetime, timezone

import requests

from common import ROOT, load_config, load_json, save_json

STATE = ROOT / "work" / "watch_state.json"
TWITCH_GQL = "https://gql.twitch.tv/gql"
TWITCH_CLIENT_ID = "ue6666qo983tsx6so1t0vnawi233wa"   # same public ID stage 1 uses
HEADERS = {"User-Agent": "Mozilla/5.0 (clip-pipeline)"}


def hours_ago(dt):
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


def check_twitch(login):
    q = ('query($l:String!){user(login:$l){stream{id createdAt}'
         'videos(first:1,type:ARCHIVE,sort:TIME){edges{node{id createdAt lengthSeconds status}}}}}')
    r = requests.post(TWITCH_GQL, json={"query": q, "variables": {"l": login}}, timeout=20,
                      headers={**HEADERS, "Client-ID": TWITCH_CLIENT_ID})
    r.raise_for_status()
    user = r.json()["data"]["user"]
    if user is None:
        raise ValueError(f"no Twitch user called {login}")
    edges = user["videos"]["edges"]
    if user["stream"]:
        started = datetime.fromisoformat(user["stream"]["createdAt"].replace("Z", "+00:00"))
        return "LIVE", f"live for {hours_ago(started):.1f}h", None
    if not edges:
        return "NOTHING", "no past broadcasts saved", None
    v = edges[0]["node"]
    if v["status"] != "RECORDED":
        return "LIVE", "stream just ended, VOD still finishing", None
    start = datetime.fromisoformat(v["createdAt"].replace("Z", "+00:00"))
    ended = hours_ago(start) - v["lengthSeconds"] / 3600
    return "VOD", f"{v['lengthSeconds'] / 3600:.1f}h stream, ended {ended:.1f}h ago", \
        (f"https://www.twitch.tv/videos/{v['id']}", ended)


def check_kick(slug):
    ch = requests.get(f"https://kick.com/api/v2/channels/{slug}", headers=HEADERS, timeout=20)
    ch.raise_for_status()
    if ch.json().get("livestream"):
        return "LIVE", "live now", None
    vids = requests.get(f"https://kick.com/api/v2/channels/{slug}/videos", headers=HEADERS,
                        timeout=20)
    vids.raise_for_status()
    vids = vids.json()
    if not vids:
        return "NOTHING", "no past broadcasts saved", None
    v = vids[0]
    if v.get("is_live"):
        return "LIVE", "stream just ended, VOD still finishing", None
    start = datetime.fromisoformat(v["start_time"]).replace(tzinfo=timezone.utc)
    length = v["duration"] / 1000 / 3600
    ended = hours_ago(start) - length
    return "VOD", f"{length:.1f}h stream, ended {ended:.1f}h ago", \
        (f"https://kick.com/{slug}/videos/{v['video']['uuid']}", ended)


def main():
    state = load_json(STATE) if STATE.exists() else {"done": []}
    if "--done" in sys.argv:
        link = sys.argv[sys.argv.index("--done") + 1]
        if link not in state["done"]:
            state["done"].append(link)
        STATE.parent.mkdir(exist_ok=True)
        save_json(STATE, state)
        print(f"Marked as handled: {link}")
        return

    cfg = load_config()
    max_age = cfg.get("watch_max_age_hours", 12)
    for entry in cfg.get("watch_streamers") or []:
        site, _, name = entry.partition(":")
        try:
            status, note, vod = (check_kick if site == "kick" else check_twitch)(name)
        except Exception as e:   # one broken check shouldn't stop the others
            print(f"ERROR    {entry}: {e}")
            continue
        if status != "VOD":
            print(f"{status:8} {entry}: {note}")
        elif vod[0] in state["done"]:
            print(f"NOTHING  {entry}: latest VOD already clipped ({note})")
        elif vod[1] > max_age:
            print(f"NOTHING  {entry}: latest VOD is too old to auto-clip ({note})")
        else:
            print(f"READY    {vod[0]}  {entry}: {note}")


if __name__ == "__main__":
    main()
