# Clip pipeline: instructions for Claude

The user pastes a Kick (or Twitch) VOD link and wants short vertical clips plus hooks.
Speak plainly to them: they aren't technical.

## Setup (if the environment's setup script hasn't already done it)
```bash
pip install -q -r requirements.txt
```
ffmpeg must be installed. No ANTHROPIC_API_KEY is needed: Claude (you) does the scoring in-session.

## Workflow
1. **Check the VOD length first** and give a time estimate: Kick about 20–25 min per hour of stream, Twitch about 13–15 min per hour
   (Kick chat ≈ 12 min/hour, Twitch chat ≈ 8–9 min/hour; transcription ≈ 10–16 min/hour on 4 CPUs).
2. **Stage 1 + 2 in parallel.** Run `python3 -u stage1_download.py <link>` in the background, logging to a file.
   As soon as its log shows "Downloading chat", start `python3 -u stage2_transcribe.py <folder id>` in the
   background too (it only needs audio.wav). Folder id = `kick_` + first 8 chars of the VOD id.
3. **Stage 3:** `python3 stage3_spikes.py <folder id>` → `work/<id>/spikes.json` (top 15 chat spikes).
4. **Stage 4: score the moments yourself (no API).** For each spike, read the transcript from 60 s before to
   30 s after, plus a sample of chat, and judge it by the rules in `prompt.txt`. Write
   `work/<id>/analysis.json` in the same format stage4_analyze.py produces (keyed by spike time as a string:
   clippable, score, start_seconds, end_seconds, reason, hook_text, caption, spike_time). Check the cut points
   against transcript word timings so clips start just before the setup and end just after the payoff.
   Score down bits built on ethnic stereotypes or sexual jokes (they get flagged on TikTok).
   If the user wants more clips, look at spikes beyond the top 15.
5. **Stage 5:** the user's preference is **raw 9:16 clips with no captions or hook burned in**, plus 2–3 hook
   ideas per clip written in chat. Cut with:
   `ffmpeg -ss START -i work/<id>/video.mp4 -t LEN -vf "crop=ih*9/16:ih,scale=1080:1920,setsar=1" -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 160k -movflags +faststart output/<id>/no_text/clip_NN_name.mp4`
   (`python3 stage5_render.py <id>` makes the captioned versions if they ask for those.)
   Grab a frame from each clip and look at it to check the people are in frame; if they're cut off, offer
   `crop_mode: blur` or a shifted crop.
6. Send the clips to the user as files, with a short table: time in stream, what happens, hook ideas.

## Progress updates
Track overall progress (video 5%, chat 40%, transcription 35%, finding moments 12%, cutting clips 8%).
Use a background monitor so you wake at each quarter mark. At 25/50/75/100% post a message whose first line is
exactly `DING 25%` (etc.), and send a push notification if you have a tool for it.

## Known issues
- yt-dlp needs `compat_opts: {"no-certifi"}` behind the session's HTTPS proxy (already set in stage1).
- Twitch chat uses the unofficial GraphQL endpoint with the public client ID yt-dlp uses (user approved). Folder id = `twitch_` + VOD number.
- Twitch serves each VOD from one of many `*.cloudfront.net` video servers; a new one may be blocked by the network allowlist and need adding (exact host, user approves). Live/unfinished VODs fail to download: wait until the stream ends.
- Kick rate-limits chat: keep the 1 request/second pacing.
