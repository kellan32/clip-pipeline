"""Stage 2: transcribe the stream's audio with timestamps (Whisper).

Usage:  python3 stage2_transcribe.py <folder ID from stage 1>
"""
import os
import subprocess
import sys
import time

import numpy as np

# Download the model with plain HTTPS (the "xet" method needs extra servers allowed).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from faster_whisper import WhisperModel

from common import banner, hms, load_config, load_json, save_json, work_dir


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 stage2_transcribe.py <folder ID, e.g. kick_01a1ed52>")
    cfg = load_config()
    folder = work_dir(cfg, sys.argv[1])
    audio = folder / "audio.wav"
    out = folder / "transcript.json"
    if not audio.exists():
        raise SystemExit(f"No audio found in {folder}. Run stage 1 first.")
    if out.exists():
        print("Transcript already exists, skipping. (Delete transcript.json to redo it.)")
        return

    print(f"Loading Whisper '{cfg['whisper_model']}' model (downloads once the first time)...")
    model = WhisperModel(cfg["whisper_model"], device="cpu", compute_type="int8")

    # Work through the audio in 30-minute pieces so long streams don't run out of memory.
    # Each finished piece is saved, so a crash resumes where it stopped.
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(audio)], capture_output=True, text=True,
                               check=True).stdout.strip())
    chunk = 1800
    partial = folder / "transcript.partial.json"
    result = load_json(partial) if partial.exists() else []
    done_until = result[-1]["chunk_end"] if result else 0.0
    result = [r for r in result if "chunk_end" not in r]

    started = time.time()
    print(f"Transcribing {hms(dur)} of audio in {int(-(-dur // chunk))} pieces...")
    pos = done_until
    while pos < dur:
        # Read this piece with ffmpeg ourselves (16 kHz mono), which avoids a version clash
        # in faster-whisper's own audio loader.
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(pos), "-t", str(chunk),
                              "-i", str(audio), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                             capture_output=True, check=True).stdout
        samples = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        del raw
        segments, info = model.transcribe(
            samples,
            language=cfg.get("whisper_language"),
            word_timestamps=True,       # needed for word-by-word captions in stage 5
            vad_filter=True,            # skip long silences
        )
        for seg in segments:
            result.append({
                "start": round(pos + seg.start, 2),
                "end": round(pos + seg.end, 2),
                "text": seg.text.strip(),
                "words": [{"start": round(pos + w.start, 2), "end": round(pos + w.end, 2),
                           "word": w.word.strip()} for w in (seg.words or [])],
            })
            print(f"\r  {min(pos + seg.end, dur) / dur:.0%}", end="", flush=True)
        del samples
        pos = min(pos + chunk, dur)
        save_json(partial, result + [{"chunk_end": pos}])
        print(f"\r  {pos / dur:.0%}", end="", flush=True)
    print()
    save_json(out, result)
    partial.unlink(missing_ok=True)

    words = sum(len(s["words"]) for s in result)
    banner("STAGE 2 DONE — transcription")
    print(f"  Took:      {hms(time.time() - started)}")
    print(f"  Segments:  {len(result)}   Words: {words}")
    print("  First few lines:")
    for s in result[:4]:
        print(f"    [{hms(s['start'])}] {s['text']}")
    print(f"  Saved to:  {out}")


if __name__ == "__main__":
    main()
