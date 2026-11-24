"""Run the whole pipeline on one VOD, start to finish.

Usage:  python3 run.py <Kick or Twitch VOD link>

Each stage skips work that's already done, so it's safe to run again after a failure.
"""
import subprocess
import sys

from common import vod_id_from_url


def stage(script, *args):
    result = subprocess.run([sys.executable, script, *args])
    if result.returncode != 0:
        raise SystemExit(f"\nStopped: {script} failed (see the message above). "
                         "Fix it and run the same command again — finished steps are skipped.")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 run.py <VOD link>")
    url = sys.argv[1]
    vod_id = vod_id_from_url(url)
    stage("stage1_download.py", url)
    stage("stage2_transcribe.py", vod_id)
    stage("stage3_spikes.py", vod_id)
    stage("stage4_analyze.py", vod_id)
    stage("stage5_render.py", vod_id)


if __name__ == "__main__":
    main()
