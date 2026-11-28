"""Tell voices apart in a clip, so captions can be coloured per speaker.

Only says yes when it's easy: one clear main voice (stays white) and other voices that sound
clearly different from it (coloured). If anything looks unsure, it returns None and the captions stay all white.

Uses a small free voice-matching model (3D-Speaker ERes2Net; it matches voices, not words, so any language works) run with sherpa-onnx; it is
downloaded once into models/.
"""
import urllib.request

import numpy as np

from common import ROOT

MODEL_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/" + MODEL_NAME)
MODEL_PATH = ROOT / "models" / MODEL_NAME
SR = 16000

_extractor = []


def _embedder():
    if not _extractor:
        import sherpa_onnx
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(exist_ok=True)
            print("  Downloading voice-matching model (once, 40 MB)...")
            tmp = MODEL_PATH.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.rename(MODEL_PATH)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL_PATH), num_threads=2)
        _extractor.append(sherpa_onnx.SpeakerEmbeddingExtractor(cfg))
    return _extractor[0]


def _embed(audio, start, end):
    ext = _embedder()
    s = ext.create_stream()
    s.accept_waveform(SR, audio[int(start * SR):int(end * SR)])
    s.input_finished()
    v = np.array(ext.compute(s))
    return v / (np.linalg.norm(v) + 1e-9)


def _sentences(words, max_gap):
    """Split the words into short lines: a pause or a sentence end starts a new line."""
    out, cur = [], []
    for i, w in enumerate(words):
        if cur and (w["start"] - words[cur[-1]]["end"] > max_gap
                    or words[cur[-1]]["word"][-1] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def _cluster(embs, threshold):
    """Merge lines whose voices match (average similarity above threshold)."""
    groups = [[i] for i in range(len(embs))]
    sim = embs @ embs.T
    while len(groups) > 1:
        best, pair = -1, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = sim[np.ix_(groups[a], groups[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if best < threshold:
            break
        a, b = pair
        groups[a] += groups.pop(b)
    return groups


def label_speakers(audio, words, cfg):
    """Return (labels, message). labels[i] is 0 for the main voice (white) or 1 for anyone
    else (coloured), or None when it isn't confident (then every caption stays white)."""
    same = cfg.get("speaker_same_threshold", 0.40)
    is_main = cfg.get("speaker_main_threshold", 0.45)
    is_other = cfg.get("speaker_other_threshold", 0.20)

    if not words:
        return None, "no speech"
    lines = _sentences(words, cfg.get("speaker_pause_seconds", 0.3))

    def span(t):
        a, b = words[t[0]]["start"], words[t[-1]]["end"]
        pad = max(0.0, (1.0 - (b - a)) / 2)      # very short lines get a little audio either side
        return max(0.0, a - pad), b + pad, b - a

    spans = [span(t) for t in lines]
    embs = np.array([_embed(audio, a, b) for a, b, _ in spans])
    durs = np.array([d for _, _, d in spans])
    total = durs.sum()
    if total < 4:
        return None, "not enough speech to compare voices"

    # The main voice is the biggest group of clearly matching longer lines.
    longer = [i for i in range(len(lines)) if durs[i] >= 1.0]
    if len(longer) < 3:
        return None, "not enough speech to compare voices"
    groups = _cluster(embs[longer], same)
    main = max(groups, key=lambda g: durs[[longer[i] for i in g]].sum())
    main = [longer[i] for i in main]
    if durs[main].sum() < 0.4 * total:
        return None, "no clear main voice"

    labels, unsure, other_time, other_lines = [None] * len(words), 0.0, 0.0, 0
    for k, t in enumerate(lines):
        ref = [i for i in main if i != k] or main   # don't compare a line with itself
        cent = embs[ref].mean(axis=0)
        score = float(embs[k] @ (cent / np.linalg.norm(cent)))
        if score >= is_main:
            lab = 0
        elif score <= is_other:
            lab = 1
            other_time += durs[k]
            other_lines += 1
        else:
            lab = 0
            unsure += durs[k]
        for i in t:
            labels[i] = lab

    if other_time < 1.5:
        return None, "only one voice"
    if unsure > cfg.get("speaker_max_unsure", 0.05) * total:
        return None, f"{unsure / total:.0%} of the speech was hard to place"
    switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    if switches > total / 2.0:
        return None, "too much quick back-and-forth"
    return labels, (f"main voice {1 - other_time / total:.0%}, other voice {other_time / total:.0%},"
                    f" {other_lines} lines in colour")
