"""Stage 6: add word-by-word captions and a title bar to the cut clips.

Takes the raw clips in output/<id>/no_text/, transcribes each one, burns in captions
(2-3 words at a time, middle of the frame) and a title bar at the top, and saves the result
in output/<id>/ready/. The raw clips are left untouched.

Titles come from output/<id>/titles.json, e.g. {"clip_01_falls_in_pool": "LACY falls in the pool"}.
Title style: name WHO it's about and WHAT happened, plainly: "[STREAMER] [does thing] [where/with who]".

Usage:
  python3 stage6_caption.py <id> --transcripts        print each clip's words (to help write titles)
  python3 stage6_caption.py <id>                      caption every clip that has a title
  python3 stage6_caption.py <id> clip_01_falls_in_pool --title "LACY falls in the pool"
"""
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from common import ROOT, banner, load_config, load_json, save_json
from stage5_render import ass_escape, ass_time


def transcribe(clip, cache, cfg, model_box):
    """Word-level transcript of one clip (cached so re-runs are quick)."""
    if cache.exists() and cache.stat().st_mtime >= clip.stat().st_mtime:
        return load_json(cache)
    if not model_box:
        from faster_whisper import WhisperModel
        print(f"  Loading Whisper '{cfg['whisper_model']}' model...")
        model_box.append(WhisperModel(cfg["whisper_model"], device="cpu", compute_type="int8"))
    # Decode with ffmpeg ourselves (faster-whisper's own loader clashes with the installed PyAV).
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-f", "s16le", "-ac", "1",
                          "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    segments, _ = model_box[0].transcribe(audio, language=cfg.get("whisper_language"),
                                          word_timestamps=True)
    words = []
    for s in segments:
        for w in s.words or []:
            # Whisper splits things like "$2,000" or "y'all" into pieces with no space before them.
            if words and w.word.strip() and not w.word.startswith(" "):
                words[-1]["word"] += w.word.strip()
                words[-1]["end"] = round(w.end, 2)
            elif w.word.strip():
                words.append({"start": round(w.start, 2), "end": round(w.end, 2),
                              "word": w.word.strip()})
    save_json(cache, words)
    return words


def group_words(words, max_words):
    """Split words into 2-3 word chunks, breaking early at pauses and sentence ends."""
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > 0.5
                    or cur[-1]["word"][-1] in ".?!,"):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def clip_length(clip):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def write_ass(path, words, title, length, cfg):
    w, h = 1080, 1920
    font = cfg.get("caption_font", "Inter")
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Title bar: black text on a white box, top centre, whole clip.
        f"Style: Title,{font},{cfg.get('title_size', 62)},&H00000000,&H00000000,&H00FFFFFF,"
        f"&H00FFFFFF,1,0,0,0,100,100,0,0,3,22,0,8,70,70,{int(h * 0.09)},1",
        # Captions: bold white, thick black outline, dead centre (middle third).
        f"Style: Cap,{font},{cfg.get('caption_size', 96)},&H00FFFFFF,&H00FFFFFF,&H00000000,"
        f"&H80000000,1,0,0,0,100,100,2,0,1,8,3,5,70,70,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 1,{ass_time(0)},{ass_time(length)},Title,,0,0,0,,{ass_escape(title)}",
    ]
    groups = group_words(words, cfg.get("caption_words", 3))
    for i, g in enumerate(groups):
        start = g[0]["start"]
        end = g[-1]["end"] + 0.25
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0]["start"])
        end = min(end, length)
        if end <= start:
            continue
        text = " ".join(x["word"] for x in g).upper()
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{ass_escape(text)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(groups)


def main():
    args = sys.argv[1:]
    title_arg = None
    if "--title" in args:
        i = args.index("--title")
        title_arg = args[i + 1]
        del args[i:i + 2]
    show = "--transcripts" in args
    args = [a for a in args if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    cfg = load_config()
    vod_id, only = args[0], [a.removesuffix(".mp4") for a in args[1:]]
    if title_arg and len(only) != 1:
        raise SystemExit("--title needs exactly one clip name, e.g. clip_01_falls_in_pool")

    base = ROOT / cfg["output_folder"] / vod_id
    raw_dir, ready = base / "no_text", base / "ready"
    clips = sorted(raw_dir.glob("*.mp4"))
    if only:
        clips = [c for c in clips if c.stem in only]
    if not clips:
        raise SystemExit(f"No clips found in {raw_dir}")
    (ready / ".words").mkdir(parents=True, exist_ok=True)

    titles_file = base / "titles.json"
    titles = load_json(titles_file) if titles_file.exists() else {}
    if title_arg:
        titles[clips[0].stem] = title_arg
        save_json(titles_file, titles)

    model_box, made, missing = [], [], []
    for clip in clips:
        words = transcribe(clip, ready / ".words" / f"{clip.stem}.json", cfg, model_box)
        if show:
            print(f"\n== {clip.stem}  (title: {titles.get(clip.stem, 'none yet')})")
            print("   " + " ".join(w["word"] for w in words))
            continue
        title = titles.get(clip.stem)
        if not title:
            missing.append(clip.stem)
            continue
        length = clip_length(clip)
        subs = ready / f"{clip.stem}.ass"
        n = write_ass(subs, words, title, length, cfg)
        out = ready / f"{clip.stem}.mp4"
        print(f"  Captioning {clip.stem} ({length:.0f}s, {n} captions)...")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip),
                        "-vf", f"ass={subs.name}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-c:a", "copy", "-movflags", "+faststart", str(out)],
                       check=True, cwd=ready)
        subs.unlink()
        made.append(out)

    if show:
        return
    banner("STAGE 6 DONE — captioned clips")
    for out in made:
        print(f"  {out.name}  ({out.stat().st_size / 1e6:.1f} MB)  \"{titles[out.stem]}\"")
    if missing:
        print("  No title yet (skipped): " + ", ".join(missing))
        print(f"  Add titles to {titles_file} or use --title, then run again.")
    print(f"  Folder: {ready}")


if __name__ == "__main__":
    main()
