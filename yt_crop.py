"""YouTube pipeline, step 3: work out where the 9:16 window should sit, moment by moment.

  1. Sample the clip a few times a second and find the players (YOLO; OpenCV fallback).
  2. For each sample, slide a 9:16-wide window across the frame and keep the spot that covers the
     biggest group of players on the pitch (the ball counts extra when it's seen).
  3. Turn those spots into a camera path: fill gaps, remove outliers, ignore small shifts
     (dead zone), cap the pan speed and acceleration, then smooth. Each camera shot is handled
     separately so the window jumps cleanly at a cut instead of sliding across it.
  4. If too few players are found, follow the motion instead; if that fails too, centre crop.
"""
import numpy as np

from yt_select import green_mask, hog_detect, people_on_pitch, read_frames


def _targets(dets, cw, prev):
    """Best window centre (0..1 of frame width) for one sample, or nan."""
    if not dets:
        return np.nan
    xs = np.array([d[0] for d in dets])
    ws = np.array([d[1] for d in dets])
    lo, hi = cw / 2, 1 - cw / 2
    grid = np.linspace(lo, hi, 121)
    sig = 0.3 * cw
    cover = (ws[None, :] * np.exp(-0.5 * ((xs[None, :] - grid[:, None]) / sig) ** 2)).sum(1)
    if not np.isnan(prev):   # mild preference for staying put: avoids flip-flopping between groups
        cover = cover - 0.25 * cover.max() * np.abs(grid - prev) / max(cw, 1e-6)
    c = grid[int(np.argmax(cover))]
    k = ws * np.exp(-0.5 * ((xs - c) / sig) ** 2)
    return float(np.clip((k * xs).sum() / max(k.sum(), 1e-9), lo, hi))


def _median(x, n):
    out = x.copy()
    h = n // 2
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - h):i + h + 1])
    return out


def _gauss(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _fill(x):
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def camera_path(targets, fps, cw, cfg):
    """Smooth, slow, jitter-free path through the raw targets of ONE camera shot."""
    x = _fill(np.asarray(targets, float))
    if x is None:
        return None
    lo, hi = cw / 2, 1 - cw / 2
    x = _median(x, max(3, int(fps * 1.4) | 1))           # drop one-off outliers
    look = int(round(0.4 * fps))                          # offline, so we can look slightly ahead
    x = np.concatenate([x[look:], np.repeat(x[-1:], look)]) if look and len(x) > look else x
    dt = 1 / fps
    vmax = cfg["max_pan_speed"]
    amax = vmax * 1.5                                     # reach full pan speed in ~0.7 s
    dz = cfg["dead_zone"]
    pos = float(np.median(x[:int(fps * 2) + 1]))          # start already framed on the action
    v = 0.0
    out = np.empty_like(x)
    for i, t in enumerate(x):
        err = t - pos
        want = 0.0 if abs(err) <= dz else np.sign(err) * min((abs(err) - dz) * 1.5, vmax)
        v += np.clip(want - v, -amax * dt, amax * dt)
        pos = float(np.clip(pos + v * dt, lo, hi))
        if pos in (lo, hi):
            v = 0.0
        out[i] = pos
    return np.clip(_gauss(out, 0.5 * fps), lo, hi)


def track_clip(info, clip, cfg, model, cuts, area, debug_dir=None, name="clip"):
    """Returns {'t': sample times (s, from clip start), 'x': window centre 0..1, 'mode': ...,
    'thumb_t': a good thumbnail time, 'players': players per sample}."""
    bx, by, bw, bh = area
    cw = (bh * 9 / 16) / bw          # window width as a fraction of the picture width
    fps = cfg["track_fps"]
    length = clip["end"] - clip["start"]
    n_expected = int(length * fps)
    if cw >= 1:                       # source is already 9:16 or narrower: nothing to follow
        t = np.arange(max(n_expected, 1)) / fps
        return {"t": t.tolist(), "x": [0.5] * len(t), "mode": "center", "thumb_t": length / 2,
                "players": []}

    W = 960
    H = int(round(W * bh / bw / 2)) * 2
    src = info["source_path"]
    vf_area = f"crop={bw}:{bh}:{bx}:{by}," if (bw, bh) != (info["width"], info["height"]) else ""
    raw, players, frames_dbg, motion_c = [], [], [], []
    prev_t, prev_gray = np.nan, None
    import cv2
    for i, f in enumerate(_frames(src, clip["start"], length, fps, W, H, vf_area)):
        dets = []
        if model is not None:
            r = model.predict(f, imgsz=cfg["track_imgsz"], classes=[0, 32], conf=0.2, verbose=False)[0]
            on, _ = people_on_pitch(r, f)
            on = [(b, c) for b, c in on if c >= 0.3]
            n_people = len(on)
            for b, c in on:
                hgt = (b[3] - b[1]) / H
                dets.append((((b[0] + b[2]) / 2) / W, c * np.sqrt(max(hgt, 1e-3))))
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for b, k, c in zip(r.boxes.xyxy.cpu().numpy(), cls, r.boxes.conf.cpu().numpy()):
                    if k == 32 and dets:
                        dets.append((((b[0] + b[2]) / 2) / W, 1.5 * float(np.median([d[1] for d in dets]))))
        else:
            for b, c in hog_detect(f):
                dets.append((((b[0] + b[2]) / 2) / W, max(c, 0.1)))
            n_people = len(dets)
        players.append(n_people)
        tgt = _targets(dets, cw, prev_t)
        raw.append(tgt)
        if not np.isnan(tgt):
            prev_t = tgt
        # Motion centre of mass (fallback when detection finds too little).
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.int16)
        if prev_gray is not None:
            mv = (np.abs(g - prev_gray) > 15) & green_or_all(f)
            col = mv.sum(0).astype(float)
            motion_c.append(float((col * np.arange(W)).sum() / col.sum() / W) if col.sum() > 0.002 * W * H
                            else np.nan)
        else:
            motion_c.append(np.nan)
        prev_gray = g
        if debug_dir is not None and i % max(1, int(length * fps / 8)) == 0:
            frames_dbg.append((i, f.copy(), dets))

    raw = np.array(raw, float)
    if len(raw) == 0:
        return {"t": [0.0], "x": [0.5], "mode": "center", "thumb_t": length / 2, "players": []}
    found = (~np.isnan(raw)).mean()
    mode = "smart"
    if found < 0.4:
        mc = np.array(motion_c, float)
        if (~np.isnan(mc)).mean() >= 0.5:
            raw, mode = np.clip(mc, cw / 2, 1 - cw / 2), "smart-motion"
        else:
            raw, mode = np.full(len(raw), 0.5), "center"

    # Path per camera shot.
    t = np.arange(len(raw)) / fps
    seg_edges = [0] + [int(round(c * fps)) for c in cuts if 0 < c * fps < len(raw)] + [len(raw)]
    path = np.full(len(raw), 0.5)
    for a, b in zip(seg_edges[:-1], seg_edges[1:]):
        if b <= a:
            continue
        p = camera_path(raw[a:b], fps, cw, cfg)
        if p is None:   # no detections in this shot: hold the neighbouring shot's framing / centre
            p = np.full(b - a, path[a - 1] if a > 0 else 0.5)
        path[a:b] = p

    # Thumbnail: the most players in the middle 80% of the clip.
    pl = np.array(players, float)
    m0, m1 = int(len(pl) * 0.1), max(int(len(pl) * 0.9), 1)
    best = m0 + int(np.argmax(pl[m0:m1])) if m1 > m0 else len(pl) // 2
    result = {"t": t.tolist(), "x": path.tolist(), "mode": mode, "thumb_t": float(t[best]),
              "players": players, "found": round(float(found), 2), "cw": cw}
    if debug_dir is not None:
        _debug_sheet(debug_dir / f"{name}_track.jpg", frames_dbg, path, raw, cw, W, H)
    return result


def green_or_all(f):
    gm = green_mask(f)
    if gm.mean() < 0.15:
        return np.ones(gm.shape, bool)
    # Players stand on the grass: keep motion in and just above grass areas, drop the stands.
    import cv2
    return cv2.dilate(gm.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)


def _frames(src, start, length, fps, W, H, vf_area):
    import subprocess
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-an", "-vf", f"{vf_area}fps={fps},scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.wait()


def _debug_sheet(path, frames, path_x, raw, cw, W, H):
    """Contact sheet: sampled frames, detections (dots) and the chosen 9:16 window (box)."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i, f, dets in frames[:8]:
        img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        dark = (img * 0.45).astype(np.uint8)
        x0 = int((path_x[i] - cw / 2) * W)
        x1 = int((path_x[i] + cw / 2) * W)
        dark[:, x0:x1] = img[:, x0:x1]
        for x, _ in dets:
            cv2.circle(dark, (int(x * W), H - 12), 5, (0, 255, 255), -1)
        if not np.isnan(raw[i]):
            cv2.line(dark, (int(raw[i] * W), 0), (int(raw[i] * W), H), (0, 0, 255), 2)
        cv2.rectangle(dark, (x0, 0), (x1 - 1, H - 1), (0, 255, 0), 3)
        cv2.putText(dark, f"{i}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(dark, (480, int(480 * H / W))))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    cv2.imwrite(str(path), np.vstack(rows))
