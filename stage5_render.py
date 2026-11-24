"""Stage 5 + 6: cut the best clips, make them vertical (9:16) with burned-in captions,
and save each one with a text file (score, hook, caption) for review.

Usage:  python3 stage5_render.py <folder ID from stage 1>
        python3 stage5_render.py <folder ID> --test    (one sample clip, no Claude needed)
"""
import subprocess
import sys

from common import ROOT, banner, hms, load_config, load_json, work_dir


def ass_escape(text):
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def make_subtitles(path, words, clip_start, clip_end, hook, cfg):
    """Write an .ass subtitle file: hook text at the top, word-by-word captions lower down."""
    w, h = cfg["clip_width"], cfg["clip_height"]
    font = cfg["caption_font"]
    length = clip_end - clip_start
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Hook: black text on a white box, top centre.
        f"Style: Hook,{font},78,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,1,0,0,0,"
        f"100,100,0,0,3,18,0,8,80,80,{int(h * 0.12)},1",
        # Captions: big white text, thick black outline, lower middle.
        f"Style: Cap,{font},92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,"
        f"100,100,0,0,1,7,3,2,60,60,{int(h * 0.28)},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hook_end = length if cfg["hook_seconds"] <= 0 else min(cfg["hook_seconds"], length)
    if hook:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(hook_end)},Hook,,0,0,0,,"
                     f"{ass_escape(hook.upper())}")

    clip_words = [x for x in words if x["end"] > clip_start and x["start"] < clip_end and x["word"]]
    groups, current = [], []
    for x in clip_words:
        gap = x["start"] - current[-1]["end"] if current else 0
        if current and (len(current) >= cfg["words_per_caption"] or gap > 0.6):
            groups.append(current)
            current = []
        current.append(x)
    if current:
        groups.append(current)
    for i, g in enumerate(groups):
        start = g[0]["start"] - clip_start
        natural_end = g[-1]["end"] - clip_start + 0.25
        next_start = groups[i + 1][0]["start"] - clip_start if i + 1 < len(groups) else natural_end
        end = min(natural_end, next_start, length)
        text = " ".join(x["word"] for x in g).upper()
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{ass_escape(text)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(groups)


def render(video, subs, out, start, end, cfg):
    w, h = cfg["clip_width"], cfg["clip_height"]
    if cfg["crop_mode"] == "blur":
        vf = (f"split[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,"
              f"crop={w}:{h},gblur=sigma=30[bg];[b]scale={w}:-2[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,ass={subs.name}")
    else:
        vf = f"crop=ih*9/16:ih,scale={w}:{h},setsar=1,ass={subs.name}"
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{end - start:.2f}",
           "-filter_complex" if cfg["crop_mode"] == "blur" else "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, cwd=subs.parent)


def write_review_file(path, clip, info, rank):
    path.write_text(
        f"CLIP {rank:02d}\n"
        f"Score:    {clip['score']}/10\n"
        f"Hook:     {clip['hook_text']}\n"
        f"Caption:  {clip['caption']}\n"
        f"\n"
        f"Why:      {clip['reason']}\n"
        f"From:     {hms(clip['start_seconds'])} to {hms(clip['end_seconds'])} "
        f"({clip['end_seconds'] - clip['start_seconds']:.0f}s)\n"
        f"Stream:   {info['title']} ({info['channel']})\n"
        f"Link:     {info['webpage_url']}\n",
        encoding="utf-8")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    test = "--test" in sys.argv
    if len(args) != 1:
        raise SystemExit("Usage: python3 stage5_render.py <folder ID> [--test]")
    cfg = load_config()
    vod_id = args[0]
    folder = work_dir(cfg, vod_id)
    info = load_json(folder / "info.json")
    words = [x for s in load_json(folder / "transcript.json") for x in s["words"]]

    if test:
        spike = load_json(folder / "spikes.json")[0]
        clips = [{"score": 0, "clippable": True,
                  "start_seconds": max(0, spike["time"] - 25), "end_seconds": spike["time"] + 10,
                  "hook_text": "Test clip (not picked by Claude)",
                  "caption": "Sample clip to check the look #test",
                  "reason": "Test render around the busiest chat moment, made without Claude."}]
        out_dir = ROOT / cfg["output_folder"] / vod_id / "test"
    else:
        analysis = load_json(folder / "analysis.json")
        clips = sorted([c for c in analysis.values()
                        if "error" not in c and c["clippable"] and c["score"] >= cfg["min_score"]],
                       key=lambda c: (-c["score"], c["start_seconds"]))
        out_dir = ROOT / cfg["output_folder"] / vod_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clips:
        banner("STAGE 5 DONE — no clips")
        print(f"  No moments scored {cfg['min_score']}+ — nothing to cut.")
        print("  Try lowering min_score in config.yaml, or a stream with a busier chat.")
        return

    made = []
    for rank, clip in enumerate(clips, 1):
        name = f"clip_{rank:02d}_score{clip['score']}"
        subs = out_dir / f"{name}.ass"
        n_caps = make_subtitles(subs, words, clip["start_seconds"], clip["end_seconds"],
                                clip["hook_text"], cfg)
        print(f"  Rendering {name} ({clip['end_seconds'] - clip['start_seconds']:.0f}s, "
              f"{n_caps} captions)...")
        render(folder / "video.mp4", subs, out_dir / f"{name}.mp4",
               clip["start_seconds"], clip["end_seconds"], cfg)
        subs.unlink()
        write_review_file(out_dir / f"{name}.txt", clip, info, rank)
        made.append(name)

    banner("STAGE 5 + 6 DONE — clips ready for review")
    for name in made:
        size = (out_dir / f"{name}.mp4").stat().st_size / 1e6
        print(f"  {name}.mp4  ({size:.1f} MB)  + {name}.txt")
    print(f"  Folder: {out_dir}")


if __name__ == "__main__":
    main()
