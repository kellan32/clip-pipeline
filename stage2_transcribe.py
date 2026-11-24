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

from common import banner, hms, load_config, save_json, work_dir


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

    # Read the audio with ffmpeg ourselves (16 kHz mono), which avoids a version clash
    # in faster-whisper's own audio loader.
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(audio), "-f", "s16le",
                          "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    samples = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

    started = time.time()
    segments, info = model.transcribe(
        samples,
        language=cfg.get("whisper_language"),
        word_timestamps=True,       # needed for word-by-word captions in stage 5
        vad_filter=True,            # skip long silences
    )
    print(f"Transcribing {hms(info.duration)} of audio...")
    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                      for w in (seg.words or [])],
        })
        print(f"\r  {seg.end / info.duration:.0%}", end="", flush=True)
    print()
    save_json(out, result)

    words = sum(len(s["words"]) for s in result)
    banner("STAGE 2 DONE — transcription")
    print(f"  Took:      {hms(time.time() - started)}")
    print(f"  Segments:  {len(result)}   Words: {words}")
    print(f"  Language:  {info.language}")
    print("  First few lines:")
    for s in result[:4]:
        print(f"    [{hms(s['start'])}] {s['text']}")
    print(f"  Saved to:  {out}")


if __name__ == "__main__":
    main()
