CLIP PIPELINE — finds hype moments in Kick/Twitch VODs and makes vertical TikTok clips
=====================================================================================

WHAT'S IN HERE
  config.yaml        all the settings (score cutoff, window sizes, model, caption style...)
  prompt.txt         the instructions Claude uses to judge each moment
  run.py             runs every stage in order
  stage1_download.py   download video, audio and chat
  stage2_transcribe.py transcribe speech with timestamps (Whisper)
  stage3_spikes.py     find the busiest chat moments (emote spam counts extra)
  stage4_analyze.py    ask Claude to score each moment (needs ANTHROPIC_API_KEY)
  stage5_render.py     cut 9:16 clips with captions + a review .txt for each
  common.py            shared helpers

HOW TO USE IT IN A NEW CLAUDE CODE SESSION
  1. Start a session with this repo (or upload clip-pipeline.zip) and say:
       "Unzip this, install what it needs, and run it on <your VOD link>"
  2. Finished clips land in:  output/<vod id>/
     Each clip_XX_scoreN.mp4 has a matching .txt with the score, hook and caption.

NOTES
  - Twitch: video works; Twitch chat download isn't built yet (Kick chat works).
  - Your API key is read from the ANTHROPIC_API_KEY environment variable — never stored in these files.
  - Kick limits chat downloads, so chat takes ~10 minutes for a 45-minute VOD.
