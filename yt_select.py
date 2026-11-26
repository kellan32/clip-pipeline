"""YouTube pipeline, step 2: find the drills worth clipping.

How it decides:
  * Scene detection (PySceneDetect) finds the camera cuts.
  * Four times a second we measure motion and how much of the picture is grass;
    every couple of seconds YOLO counts players standing on the pitch.
  * Title cards, logos, end screens and frozen frames (no motion / no players / no pitch)
    and talking heads (one or two big people, little motion) are marked as "skip".
  * Clips may only start/end on a camera cut, or failing that on a pause in the action
    (a dip in motion), and never contain a "skip" section. Busy, player-filled windows
    closest to the target length win.
"""
import subprocess

import numpy as np

from common import load_json, save_json

STEP = 0.25                 # feature grid: 4 samples per second
GRID_W, GRID_H = 384, 216     # enough to see far-away players move


# ---------------------------------------------------------------- scenes
def detect_cuts(video, start=None, end=None, fps=30.0):
    """Times (seconds) of hard camera cuts."""
    from scenedetect import AdaptiveDetector, detect
    scenes = detect(str(video), AdaptiveDetector(min_scene_len=max(1, int(fps * 0.6))),
                    start_time=start, end_time=end)
    return [round(s[0].get_seconds(), 3) for s in scenes[1:]]


# ---------------------------------------------------------------- features
def read_frames(video, fps, w, h, start=None, duration=None):
    """Yield raw RGB frames from ffmpeg at a fixed sample rate."""
    cmd = ["ffmpeg", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-vf", f"fps={fps},scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def green_mask(rgb):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)


def player_motion(prev, cur, mask=None):
    """How much moves on screen once the camera's own pan is cancelled out.

    A panning camera makes every pixel change; we only want players moving. Estimate the global
    shift between the two frames (phase correlation), undo it, then count changed pixels.
    Returns (motion fraction, pan size in pixels)."""
    import cv2
    a, b = prev.astype(np.float32), cur.astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = cur.shape
    warped = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    m = int(np.ceil(max(abs(dx), abs(dy)))) + 2
    diff = np.abs(warped - b)[m:h - m, m:w - m] > 14
    if mask is not None:
        diff &= mask[m:h - m, m:w - m]
    return float(diff.mean()) if diff.size else 0.0, float(np.hypot(dx, dy))


def motion_and_grass(video, duration):
    import cv2
    motion, grass, prev = [], [], None
    for f in read_frames(video, 1 / STEP, GRID_W, GRID_H):
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (3, 3), 0)
        gm = green_mask(f)
        # On a grass pitch, only count motion on/next to the grass (not crowds or scoreboards).
        mask = (cv2.dilate(gm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                if gm.mean() > 0.15 else None)
        motion.append(0.0 if prev is None else player_motion(prev, g, mask)[0])
        grass.append(float(gm.mean()))
        prev = g
        if len(motion) % 400 == 0:
            print(f"\r  motion scan {min(len(motion) * STEP / max(duration, 1), 1):.0%}",
                  end="", flush=True)
    print(f"\r  motion scan 100%")
    if len(motion) > 1:
        motion[0] = motion[1]
    return np.array(motion), np.array(grass)


def people_on_pitch(result, frame_rgb):
    """Split YOLO person boxes into players on the pitch vs. everyone (crowd, bench...)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    h, w = frame_rgb.shape[:2]
    gm = green_mask(frame_rgb)
    people = [(b, c) for b, k, c in zip(xyxy, cls, conf) if k == 0]
    on = []
    for b, c in people:
        # Look at the grass just under/around the feet.
        x0, x1 = int(max(0, b[0])), int(min(w, b[2]))
        y0, y1 = int(max(0, b[3] - (b[3] - b[1]) * 0.15)), int(min(h, b[3] + (b[3] - b[1]) * 0.15 + 2))
        patch = gm[y0:y1, x0:x1]
        on.append(patch.size > 0 and patch.mean() > 0.25)
    if gm.mean() < 0.15:      # not a grass pitch (indoor/gym): count everyone
        on = [True] * len(people)
    return [p for p, o in zip(people, on) if o], people


def load_yolo(cfg):
    try:
        from ultralytics import YOLO
        from common import ROOT
        models = ROOT / "models"
        models.mkdir(exist_ok=True)
        name = cfg["yolo_model"]
        path = models / name
        if not path.exists():
            # Ultralytics downloads into the current folder: fetch it, then move it.
            import os
            cwd = os.getcwd()
            os.chdir(models)
            try:
                YOLO(name)
            finally:
                os.chdir(cwd)
        return YOLO(str(path))
    except Exception as e:
        print(f"  (YOLO person detection unavailable: {e}; using the OpenCV fallback)")
        return None


def hog_detect(frame_rgb):
    """OpenCV fallback person detector (much weaker than YOLO on small, far-away players)."""
    import cv2
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame_rgb, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return [(np.array([x, y, x + w, y + h], float), float(s))
            for (x, y, w, h), s in zip(rects, np.ravel(weights))]


def people_scan(video, duration, cfg, model):
    """Every few seconds: players on the pitch, everyone, tallest person (fraction of height)."""
    every = cfg["analysis_yolo_every"]
    rows = []
    for i, f in enumerate(read_frames(video, 1 / every, 960, 540)):
        if model is not None:
            r = model.predict(f, imgsz=960, classes=[0], conf=0.3, verbose=False)[0]
            on, allp = people_on_pitch(r, f)
        else:
            allp = on = hog_detect(f)
        tallest = max(((b[3] - b[1]) / f.shape[0] for b, _ in allp), default=0.0)
        rows.append((i * every, len(on), len(allp), float(tallest)))
        if i % 50 == 0:
            print(f"\r  player scan {min(i * every / max(duration, 1), 1):.0%}", end="", flush=True)
    print("\r  player scan 100%")
    return rows


def analyse(folder, proxy, info, cfg):
    """Scene cuts + per-0.25s features, cached in the work folder."""
    path = folder / "analysis_yt.json"
    if path.exists():
        data = load_json(path)
        if data.get("proxy_mtime") == proxy.stat().st_mtime and data.get("cfg") == _cfg_key(cfg):
            print("Video already analysed, skipping.")
            return data
    dur = info["duration"]
    print("Finding camera cuts...")
    cuts = detect_cuts(proxy, fps=min(info["fps"], 30))
    print(f"  {len(cuts)} cuts")
    print("Measuring motion and pitch...")
    motion, grass = motion_and_grass(proxy, dur)
    print("Counting players...")
    model = load_yolo(cfg)
    rows = people_scan(proxy, dur, cfg, model)
    n = len(motion)
    t = np.arange(n) * STEP
    idx = np.clip(np.round(t / cfg["analysis_yolo_every"]).astype(int), 0, max(len(rows) - 1, 0))
    nearest = [rows[i] if rows else (0, 0, 0, 0.0) for i in idx]
    data = {
        "proxy_mtime": proxy.stat().st_mtime, "cfg": _cfg_key(cfg), "step": STEP,
        "cuts": cuts, "motion": motion.round(5).tolist(), "grass": grass.round(3).tolist(),
        "players": [r[1] for r in nearest], "people": [r[2] for r in nearest],
        "tallest": [round(r[3], 3) for r in nearest],
        "detector": "yolo" if model is not None else "opencv-hog",
    }
    save_json(path, data)
    return data


def _cfg_key(cfg):
    return {k: cfg[k] for k in ("yolo_model", "analysis_yolo_every")} | {"v": 2}


# ---------------------------------------------------------------- scoring
def classify(data, cfg):
    """Per-sample activity score (0 = skip) and a 'skip' flag with the reason."""
    m = np.array(data["motion"])
    grass = np.array(data["grass"])
    players = np.array(data["players"])
    people = np.array(data["people"])
    tallest = np.array(data["tallest"])
    cut_idx = [int(round(c / STEP)) for c in data["cuts"]]
    for i in cut_idx:            # a camera cut is not "motion"
        for j in (i, i + 1):
            if 0 < j < len(m):
                m[j] = m[j - 1]
    med = float(np.median(m[m > 0.003])) if (m > 0.003).any() else 0.01

    static = _smooth(m, 4) < 0.003
    graphics = (people == 0) & (grass < 0.15)
    talking = (tallest >= 0.45) & (people <= 3) & (_smooth(m, 8) < 2.5 * med)
    skip = static | graphics | talking
    # Ignore one-sample blips: a sample counts as skip only if most of the surrounding 2 s agree.
    skip = _smooth(skip.astype(float), 8) >= 0.5
    # Smoothing blurs where a skip section starts/ends; snap those edges onto a nearby camera cut.
    orig = skip.copy()
    for i in range(1, len(orig)):
        if orig[i] != orig[i - 1]:
            near = [c for c in cut_idx if abs(c - i) <= 6]
            if near:
                c = min(near, key=lambda c: abs(c - i))
                if c < i:
                    skip[c:i] = orig[i]
                else:
                    skip[i:c] = orig[i - 1]

    good = ~skip
    ref = float(np.percentile(m[good], 75)) if good.sum() > 10 else max(med, 1e-3)
    act = np.clip(_smooth(m, 4) / max(ref, 1e-4), 0, 1.5)
    pf = np.where(players >= cfg["min_people"], 1.0, np.where(players >= 1, 0.6, 0.3))
    act = act * pf * np.where(grass < 0.15, 0.7, 1.0)
    act[skip] = 0.0
    reason = np.where(static, "frozen", np.where(graphics, "graphics", np.where(talking, "talking", "")))
    return act, skip, reason, m


def _smooth(x, n):
    if n <= 1:
        return np.asarray(x, float)
    k = np.ones(n) / n
    return np.convolve(np.asarray(x, float), k, mode="same")


def boundaries(data, act, skip, m, lo, hi):
    """Places a clip may start or end, with a quality score (higher = more natural)."""
    n = len(act)
    T = n * STEP
    cuts = [c for c in data["cuts"] if lo < c < hi]
    # Edges of skip runs are boundaries too (e.g. a drill shot that ends in a talking head).
    edges = [i * STEP for i in range(1, n) if skip[i] != skip[i - 1] and lo < i * STEP < hi]
    pts = {}
    for c in cuts:
        pts[round(c, 2)] = ("cut", 1.0)
    for e in edges:
        if all(abs(e - c) > 1.0 for c in cuts):
            pts[round(e, 2)] = ("edge", 0.7)
    for p in (lo, hi):
        pts.setdefault(round(p, 2), ("edge", 0.7 if p in (0, T) else 0.2))
    # Pauses in the action: local motion minima (each at least 4 s apart).
    ms = _smooth(m, 8)
    med = float(np.median(ms[~skip])) if (~skip).any() else float(np.median(ms))
    i_lo, i_hi = int(lo / STEP), min(int(hi / STEP), n)
    last = -1e9
    for i in range(max(i_lo, 8), min(i_hi, n - 8)):
        if skip[i] or ms[i] > 0.7 * med:
            continue
        if ms[i] == ms[i - 8:i + 9].min() and i * STEP - last >= 4:
            t = round(i * STEP, 2)
            if all(abs(t - p) > 2 for p in pts):
                pts[t] = ("pause", 0.3 + 0.4 * (1 - ms[i] / max(med, 1e-6)))
                last = t
    # Last resort: every second, penalised by how much is moving (cutting mid-action is bad).
    for s in range(int(lo) + 1, int(hi)):
        i = min(int(s / STEP), n - 1)
        if not skip[i] and all(abs(s - p) > 1.5 for p in pts):
            pts[float(s)] = ("any", -0.4 * min(ms[i] / max(med, 1e-6), 2))
    return sorted((t, k, q) for t, (k, q) in pts.items())


def pick_clips(data, cfg, count, target, min_len, max_len, lo=0.0, hi=None):
    act, skip, reason, m = classify(data, cfg)
    n = len(act)
    hi = min(hi or n * STEP, n * STEP)
    pts = boundaries(data, act, skip, m, lo, hi)
    cs_act = np.concatenate([[0], np.cumsum(act)])
    cs_skip = np.concatenate([[0], np.cumsum(skip)])
    cs_dead = np.concatenate([[0], np.cumsum(act < 0.25)])
    cands = []
    for a in range(len(pts)):
        ta, ka, qa = pts[a]
        for b in range(a + 1, len(pts)):
            tb, kb, qb = pts[b]
            L = tb - ta
            if L < min_len:
                continue
            if L > max_len:
                break
            i0, i1 = int(round(ta / STEP)), int(round(tb / STEP))
            if cs_skip[i1] - cs_skip[i0] > 0:      # contains a title card/talking head/etc.
                continue
            mean_a = (cs_act[i1] - cs_act[i0]) / max(i1 - i0, 1)
            dead = (cs_dead[i1] - cs_dead[i0]) / max(i1 - i0, 1)
            if mean_a < 0.25 or dead > 0.5:
                continue
            score = mean_a - 0.6 * dead + 0.25 * (qa + qb) / 2 - 0.3 * abs(L - target) / target
            cands.append((score, ta, tb, ka, kb, mean_a, dead))
    cands.sort(reverse=True)
    chosen = []
    for c in cands:
        if len(chosen) >= count:
            break
        if all(c[2] <= o[1] or c[1] >= o[2] for o in chosen):
            chosen.append(c)
    clips = []
    for score, ta, tb, ka, kb, mean_a, dead in chosen:
        # Step just inside hard cuts so no frame of the neighbouring shot sneaks in.
        s = ta + (0.08 if ka == "cut" else 0)
        e = tb - (0.08 if kb == "cut" else 0)
        clips.append({"start": round(float(s), 2), "end": round(float(e), 2), "score": round(float(score), 3),
                      "activity": round(float(mean_a), 3), "dead_time": round(float(dead), 3),
                      "start_on": ka, "end_on": kb})
    skipped = _skip_summary(skip, reason)
    return clips, skipped


def _skip_summary(skip, reason):
    """Runs of skipped footage, for the summary printout."""
    out, i, n = [], 0, len(skip)
    while i < n:
        if skip[i]:
            j = i
            while j < n and skip[j]:
                j += 1
            r = [x for x in reason[i:j] if x]
            why = max(set(r), key=r.count) if r else "skip"
            if (j - i) * STEP >= 2:
                out.append((round(i * STEP, 1), round(j * STEP, 1), str(why)))
            i = j
        else:
            i += 1
    return out
