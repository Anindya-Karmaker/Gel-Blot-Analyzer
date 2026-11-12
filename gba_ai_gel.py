"""Gel Blot Analyzer — image understanding for the AI bridge (numpy/scipy only, no Qt).

The AI tools in gba_ai_bridge.py call these functions to *understand* a gel or blot before
they drive the app: which way round the bands are (dark-on-light or light-on-dark), where
the lanes are, which bands are real (noise-gated), which lane is the molecular-weight
ladder, and whether some lanes look like a dilution series of a standard.

Everything here is a pure function of a 2-D grayscale array so it can be unit-tested
without the app. The lane finder is the same algorithm as the Auto Gel wizard
(AutoGelAnalysisDialog._compute_lane_profile / _run_lane_detection): column projection of
the band signal, log-compressed, padded find_peaks with an adaptive prominence sweep that
prefers regularly spaced lanes.

Conventions
-----------
* `gray`  : 2-D float array of the analysis image (what the densitometry measures).
* `sig`   : band signal, HIGH where a band is. For dark-band images sig = max - gray.
* Bands are reported top -> bottom with image-pixel y (already offset by the region origin).
"""

import itertools
import re

import numpy as np

try:
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, grey_opening, median_filter
    from scipy.signal import find_peaks, peak_widths
    SCIPY = True
except Exception:  # pragma: no cover - the app itself requires scipy
    SCIPY = False

# Defaults chosen so that "only true bands" are reported: a band must stand clearly above
# the lane's own noise (SNR), be wider than a speck, and (sample lanes) not be a ripple on
# the shoulder of a much stronger band.
DEFAULT_SNR_MIN = 5.0          # band height / noise sigma
DEFAULT_REL_MIN = 0.01         # sample lanes: >= 1 % of the lane's strongest band
DEFAULT_MIN_WIDTH = 2.0        # FWHM in px; narrower peaks are specks / dust


# ============================================================================ basics
def robust_noise_sigma(profile):
    """Noise sigma of a 1-D trace from the MAD of its first differences (robust to bands)."""
    p = np.asarray(profile, dtype=float)
    if p.size < 4:
        return 1e-9
    d = np.diff(p)
    mad = np.median(np.abs(d - np.median(d)))
    s = 1.4826 * mad / np.sqrt(2.0)
    if not np.isfinite(s) or s <= 0:
        s = max(1e-9, 1e-4 * float(np.ptp(p)))
    return float(s)


def background_sigma(net):
    """Robust spread of a baseline-subtracted trace: MAD about the median, which ignores the
    (sparse) bands and measures the background's own ripple/wander."""
    n = np.asarray(net, dtype=float)
    if n.size < 4:
        return 1e-9
    med = np.median(n)
    s = 1.4826 * np.median(np.abs(n - med))
    return float(s) if np.isfinite(s) and s > 0 else 1e-9


def baseline(profile, size):
    """Smooth background under a 1-D trace: grey opening (removes peaks narrower than
    `size`) then a light Gaussian, never above the trace."""
    p = np.asarray(profile, dtype=float)
    if p.size < 3:
        return np.zeros_like(p)
    size = int(max(3, min(size, p.size)))
    bg = grey_opening(p, size=size)
    bg = gaussian_filter1d(bg, sigma=max(1.0, size / 4.0))
    return np.minimum(bg, p)


def content_box(shape, margins):
    """(x0, y0, x1, y1) of the content inside transparent padding margins (l, r, t, b)."""
    H, W = shape[:2]
    l, r, t, b = (int(max(0, m)) for m in margins)
    x0, x1 = min(l, W - 1), max(min(l, W - 1) + 1, W - r)
    y0, y1 = min(t, H - 1), max(min(t, H - 1) + 1, H - b)
    return int(x0), int(y0), int(x1), int(y1)


# ============================================================================ polarity
def band_polarity(gray, max_val=None):
    """Decide whether bands are DARK on a light background or LIGHT on a dark background.

    Returns dict(polarity='dark_bands'|'light_bands'|'ambiguous', background_level 0..1,
    confidence 0..1, reason). The densitometry engine expects DARK bands on a LIGHT
    background (white lanes, black bands); 'light_bands' therefore means "invert first"."""
    g = np.asarray(gray, dtype=float)
    if g.size == 0:
        return {"polarity": "ambiguous", "confidence": 0.0, "background_level": None,
                "reason": "empty image"}
    if g.size > 4_000_000:                       # subsample big images (speed only)
        step = int(np.ceil(np.sqrt(g.size / 4_000_000)))
        g = g[::step, ::step]
    lo, hi = np.percentile(g, [0.5, 99.5])
    if hi - lo < 1e-9:
        return {"polarity": "ambiguous", "confidence": 0.0, "background_level": None,
                "reason": "image is flat"}
    n = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    bg = float(np.median(n))
    p2, p98 = np.percentile(n, [2, 98])
    dark_tail = max(1e-6, bg - p2)       # how far the darkest features stand out
    bright_tail = max(1e-6, p98 - bg)    # how far the brightest features stand out
    ratio = bright_tail / dark_tail
    if ratio > 1.25 and bg < 0.55:
        pol, conf = "light_bands", min(1.0, (ratio - 1.0) / 2.0 + (0.55 - bg))
        why = "background is dark (%.2f) and the bright features stand out most" % bg
    elif ratio < 0.8 and bg > 0.45:
        pol, conf = "dark_bands", min(1.0, (1.0 / ratio - 1.0) / 2.0 + (bg - 0.45))
        why = "background is light (%.2f) and the dark features stand out most" % bg
    elif bg <= 0.35:
        pol, conf, why = "light_bands", 0.4, "background is dark (%.2f)" % bg
    elif bg >= 0.65:
        pol, conf, why = "dark_bands", 0.4, "background is light (%.2f)" % bg
    else:
        pol, conf, why = "ambiguous", 0.0, "background mid-grey (%.2f), no clear tail" % bg
    return {"polarity": pol, "confidence": round(float(conf), 2),
            "background_level": round(bg, 3), "reason": why}


def band_signal(gray, max_val, dark_bands=True):
    g = np.asarray(gray, dtype=float)
    return (float(max_val) - g) if dark_bands else g.copy()


# ============================================================================ lanes
def _lane_column_profile(sig, smooth=5.0):
    """Same projection as AutoGelAnalysisDialog._compute_lane_profile."""
    inv = np.asarray(sig, dtype=float)
    H, W = inv.shape
    k = max(1, int(round(0.12 * H)))
    col_top = np.partition(inv, H - k, axis=0)[-k:, :].mean(axis=0)
    col = 0.6 * col_top + 0.4 * inv.mean(axis=0)
    if smooth > 0.05:
        col = gaussian_filter1d(col, sigma=smooth)
    col = np.log1p(np.maximum(col - float(np.min(col)), 0.0))
    pr = float(np.ptp(col))
    if pr < 1e-9:
        return np.zeros(W)
    return (col - float(np.min(col))) / pr


def _find_peaks_padded(col, prominence, distance, height=None):
    col = np.asarray(col, dtype=float)
    n = len(col)
    if n < 3:
        return np.array([], dtype=int)
    pad = min(max(2, int(distance)), n - 1)
    padded = np.pad(col, pad, mode="reflect")
    kw = dict(prominence=prominence, distance=max(1, int(distance)), width=1)
    if height is not None:
        kw["height"] = height
    try:
        pk, _ = find_peaks(padded, **kw)
    except Exception:
        return np.array([], dtype=int)
    pk = pk - pad
    return np.unique(pk[(pk >= 0) & (pk < n)]).astype(int)


def _regularity(pk):
    pk = np.asarray(pk, dtype=float)
    if len(pk) < 2:
        return len(pk) * 0.1
    sp = np.diff(np.sort(pk))
    sp = sp[sp > 0]
    if len(sp) == 0:
        return len(pk) * 0.1
    cv = float(np.std(sp) / (np.mean(sp) + 1e-9))
    return float(len(pk)) * float(np.exp(-2.0 * cv))


def _bounds_from_peaks(col, peaks, W, fallback_dist):
    peaks = np.unique(np.asarray(peaks, dtype=int))
    peaks = peaks[(peaks >= 0) & (peaks < W)]
    if len(peaks) == 0:
        return []
    if len(peaks) >= 2:
        sp = np.diff(peaks)
        sp = sp[sp > 0]
        med = float(np.median(sp)) if len(sp) else fallback_dist * 2.0
    else:
        med = fallback_dist * 2.0
    half = max(1.0, med / 2.0)
    b = []
    s0 = max(0, int(peaks[0] - half))
    seg = col[s0:peaks[0] + 1]
    b.append(s0 + int(np.argmin(seg)) if len(seg) else s0)
    for i in range(len(peaks) - 1):
        p0, p1 = int(peaks[i]), int(peaks[i + 1])
        seg = col[p0:p1 + 1]
        b.append(p0 + int(np.argmin(seg)) if len(seg) else (p0 + p1) // 2)
    e = min(W, int(peaks[-1] + half) + 1)
    seg = col[peaks[-1]:e]
    b.append(int(peaks[-1]) + int(np.argmin(seg)) if len(seg) else min(W - 1, e - 1))
    out = []
    for i in range(len(peaks)):
        x0, x1 = max(0, b[i]), min(W - 1, b[i + 1])
        if x1 > x0:
            out.append([int(x0), int(x1)])
    return out


def lane_pitch(col):
    """Dominant lane spacing (px) of a column profile from its autocorrelation, or None."""
    c = np.asarray(col, dtype=float)
    n = len(c)
    if n < 40:
        return None
    c = c - c.mean()
    if np.std(c) < 1e-9:
        return None
    ac = np.correlate(c, c, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = max(8, n // 60), max(9, n // 2)
    seg = ac[lo:hi]
    if seg.size < 3:
        return None
    pk, pr = find_peaks(seg, prominence=0.05)
    if len(pk) == 0:
        return None
    best = pk[int(np.argmax(seg[pk]))]
    # prefer the FIRST strong peak (the fundamental), not a harmonic
    strong = [p for p in pk if seg[p] >= 0.6 * seg[best]]
    return float(lo + (strong[0] if strong else best))


def _band_column_profile(sig):
    """Column profile that follows BANDS, not their darkest column: light 2-D blur, remove
    each row's background (its 10th percentile — NOT the median: on a band row the bands can
    cover most of the row), then the 97th percentile per column. A lane
    with one faint band still rises clearly above the empty gaps between lanes."""
    s2 = gaussian_filter(np.asarray(sig, dtype=float), (2, 2))
    s2 = np.clip(s2 - np.percentile(s2, 10, axis=1, keepdims=True), 0, None)
    col = gaussian_filter1d(np.percentile(s2, 97, axis=0), 3)
    col = np.log1p(col - col.min())
    pr = float(np.ptp(col))
    return (col - col.min()) / pr if pr > 1e-9 else np.zeros_like(col)


def _gap_lanes(col, min_frac=0.25):
    """Lanes as the runs of 'occupied' columns separated by empty gaps. Lane edges are put
    in the middle of each gap. Works when lanes are wide and unevenly stained (the darkest
    column is not the lane centre) — the classic failure of peak-based lane finding."""
    c = np.asarray(col, dtype=float)
    W = len(c)
    if W < 10 or np.ptp(c) < 1e-9:
        return []
    floor = float(np.percentile(c, 10))
    best = []
    for frac in (0.12, 0.18, 0.25, 0.35):
        on = c > floor + frac * (1.0 - floor)
        runs, start = [], None
        for x in range(W):
            if on[x] and start is None:
                start = x
            elif not on[x] and start is not None:
                runs.append([start, x - 1]); start = None
        if start is not None:
            runs.append([start, W - 1])
        if len(runs) < 2:
            continue
        # merge runs split by a tiny gap, drop slivers
        widths = np.array([r[1] - r[0] + 1 for r in runs])
        med = float(np.median(widths))
        merged = [runs[0]]
        for r in runs[1:]:
            if r[0] - merged[-1][1] < 0.15 * med:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        widths = np.array([r[1] - r[0] + 1 for r in merged])
        med = float(np.median(widths))
        merged = [r for r, w in zip(merged, widths) if w >= min_frac * med]
        if len(merged) < 2:
            continue
        cents = [(r[0] + r[1]) / 2.0 for r in merged]
        if not best or _regularity(cents) > _regularity([(r[0] + r[1]) / 2.0 for r in best]):
            best = merged
    if not best:
        return []
    bounds = []
    for i, r in enumerate(best):
        x0 = 0 if i == 0 else int((best[i - 1][1] + r[0]) // 2)
        x1 = W - 1 if i == len(best) - 1 else int((r[1] + best[i + 1][0]) // 2)
        if i == 0:
            x0 = max(0, r[0] - (best[1][0] - best[0][1]) // 2)
        if i == len(best) - 1:
            x1 = min(W - 1, r[1] + (best[-1][0] - best[-2][1]) // 2)
        bounds.append([int(x0), int(x1)])
    return bounds


def _grid_lanes(sig, n):
    """n lanes whose n-1 boundaries are placed JOINTLY (dynamic programming) on the emptiest
    columns — the gaps between lanes — with every lane 0.5-1.5x the average width and a mild
    preference for equal widths. Used when the user says how many lanes there are."""
    sig = np.asarray(sig, dtype=float)
    W = sig.shape[1]
    if n <= 1:
        return [[0, W - 1]]
    s2 = gaussian_filter(sig, (2, 2))
    s2 = np.clip(s2 - np.percentile(s2, 10, axis=1, keepdims=True), 0, None)
    col = gaussian_filter1d(np.percentile(s2, 99, axis=0), 3)
    col = np.clip(col - col.min(), 0, None)
    # log relative to the empty-gap level: a faint lane (10x the gap) then costs about as
    # much to cut through as a strong one (1000x), so cuts land in REAL gaps.
    s0 = max(float(np.percentile(col, 10)), 1e-3 * float(col.max()), 1e-9)
    col = np.log1p(col / s0)
    col = col / (col.max() + 1e-12)
    p = W / float(n)
    dmin, dmax = max(2, int(0.5 * p)), max(3, int(1.5 * p))
    INF = 1e18
    # cost[k][x]: best cost with the k-th cut (1-based) at column x
    prev = np.full(W, INF)
    for x in range(dmin, min(W, dmax + 1)):
        prev[x] = col[x] + 0.05 * ((x - p) / p) ** 2
    back = []
    for k in range(2, n):
        cur = np.full(W, INF)
        arg = np.zeros(W, dtype=int)
        for d in range(dmin, dmax + 1):
            cand = np.full(W, INF)
            cand[d:] = prev[:-d] + 0.05 * ((d - p) / p) ** 2
            better = cand < cur
            cur[better] = cand[better]
            arg[better] = (np.arange(W) - d)[better]
        cur = cur + col
        back.append(arg)
        prev = cur
    # last lane must also be 0.5-1.5x wide
    last = np.full(W, INF)
    for x in range(W):
        d = (W - 1) - x
        if dmin <= d <= dmax:
            last[x] = prev[x] + 0.05 * ((d - p) / p) ** 2
    if not np.isfinite(last).any() or last.min() >= INF:
        return [[int(round(i * p)), int(round((i + 1) * p)) - 1] for i in range(n)]
    x = int(np.argmin(last))
    cuts = [x]
    for arg in reversed(back):
        x = int(arg[x])
        cuts.append(x)
    cuts = [0] + sorted(cuts) + [W - 1]
    return [[int(cuts[i]), int(cuts[i + 1])] for i in range(n)]


def _gap_profile(sig):
    """Column profile in [0, 1] that is LOW in the empty gaps between lanes and HIGH on a
    lane, with faint and strong lanes on a comparable (log) scale: light 2-D blur, each row's
    background (10th percentile) removed, 99th percentile per column, log relative to the
    empty-gap level."""
    s2 = gaussian_filter(np.asarray(sig, dtype=float), (2, 2))
    s2 = np.clip(s2 - np.percentile(s2, 10, axis=1, keepdims=True), 0, None)
    col = gaussian_filter1d(np.percentile(s2, 99, axis=0), 3)
    col = np.clip(col - col.min(), 0, None)
    s0 = max(float(np.percentile(col, 10)), 1e-3 * float(col.max()), 1e-9)
    col = np.log1p(col / s0)
    return col / (col.max() + 1e-12)


def fit_target_lanes(sig, n, snap=0.3, stiffness=1.0, gap_profile=None):
    """Fit exactly `n` lanes (the user's Target Lanes) to a gel region.

    Two steps:

    1. COMB FIT. Lanes of one gel sit on an even pitch, so an n-lane comb -- n + 1 equally
       spaced borders, pitch p, left edge L -- is slid and stretched over the gap profile
       g(x). The comb kept is the one whose inner borders fall on the emptiest columns while
       (almost) no band signal is left outside it:

           J(p, L) = mean_k g(L + k p)  +  2 * (signal outside [L, L + n p]) / (all signal)
                     + 0.05 * |comb centre - image centre| / W

    2. BORDER SNAP. Each border may leave its comb position x_k for a nearby real gap:

           b_k = argmin_x  g(x) + stiffness * ((x - x_k) / p)^2 ,  |x - x_k| <= snap * p

       Next to a well-stained lane g has a clear minimum and the border snaps to it; between
       FAINT or EMPTY lanes g is flat, the quadratic term wins and the border stays at its
       equal-distance position -- which is the best estimate there is for such lanes.

    `gap_profile` lets a caller that re-fits often (the wizard's Target Lanes box) pass a
    cached _gap_profile(sig). Returns {"bounds": [[x0, x1], ...], "pitch": p, "left": L,
    "snapped": [bool, ...]} (snapped[k]: border k moved off its equal-distance position)."""
    sig = np.asarray(sig)
    W = sig.shape[1]
    n = int(n)
    if n <= 1 or W < 4 * max(1, n):
        return {"bounds": [[0, W - 1]] if n <= 1 else
                [[int(round(i * W / n)), int(round((i + 1) * W / n)) - 1] for i in range(n)],
                "pitch": float(W) / max(1, n), "left": 0.0, "snapped": []}
    g = _gap_profile(sig) if gap_profile is None else np.asarray(gap_profile, dtype=float)
    total = float(g.sum()) + 1e-12
    csum = np.concatenate([[0.0], np.cumsum(g)])
    xs = np.arange(W, dtype=float)
    best = None
    p_hi = W / float(n)
    for p in np.linspace(0.55 * p_hi, p_hi, 91):
        span = n * p
        offs = np.arange(0.0, W - span + 1e-9, max(1.0, p / 40.0))
        if offs.size == 0:
            offs = np.array([0.0])
        # inner borders L + k p, k = 1..n-1 (linear interpolation of g)
        teeth = offs[:, None] + p * np.arange(1, n)[None, :]
        gap_cost = np.interp(teeth.ravel(), xs, g).reshape(teeth.shape).mean(axis=1)
        a = np.clip(np.round(offs).astype(int), 0, W)
        b = np.clip(np.round(offs + span).astype(int), 0, W)
        outside = 1.0 - (csum[b] - csum[a]) / total
        centre = np.abs(offs + span / 2.0 - W / 2.0) / W
        cost = gap_cost + 2.0 * outside + 0.05 * centre
        i = int(np.argmin(cost))
        if best is None or cost[i] < best[0]:
            best = (float(cost[i]), float(p), float(offs[i]))
    _c, p, L = best
    borders, snapped = [], []
    half = max(1, int(round(snap * p)))
    for k in range(n + 1):
        xk = L + k * p
        lo = int(max(0, np.floor(xk - half)))
        hi = int(min(W - 1, np.ceil(xk + half)))
        if hi <= lo:
            borders.append(int(min(W - 1, max(0, round(xk))))); snapped.append(False)
            continue
        xx = np.arange(lo, hi + 1)
        j = int(np.argmin(g[lo:hi + 1] + stiffness * ((xx - xk) / p) ** 2))
        borders.append(int(xx[j]))
        snapped.append(abs(xx[j] - xk) > 0.04 * p)
    borders[0] = max(0, borders[0]); borders[-1] = min(W - 1, borders[-1])
    for k in range(1, n + 1):                       # keep them strictly increasing
        if borders[k] <= borders[k - 1]:
            borders[k] = min(W - 1, borders[k - 1] + 1)
    return {"bounds": [[int(borders[k]), int(borders[k + 1])] for k in range(n)],
            "pitch": p, "left": L, "snapped": snapped}


def lane_layout_warning(bounds):
    """Text when the lanes look unreliable (uneven spacing or sliver lanes), else None."""
    if len(bounds) < 3:
        return None
    c = np.array([(a + b) / 2.0 for a, b in bounds])
    w = np.array([b - a for a, b in bounds], dtype=float)
    sp = np.diff(c)
    cv = float(np.std(sp) / (np.mean(sp) + 1e-9))
    inner = w[1:-1] if len(w) > 3 else w
    if cv > 0.18 or float(inner.min()) < 0.5 * float(np.median(w)):
        return ("lane spacing is uneven (centre gaps %s px)" % ", ".join(str(int(x)) for x in sp))
    return None


def detect_lanes(sig, expected=None, smooth=5.0, prominence=0.05, distance=None):
    """Lane x-bounds [[x0, x1], ...] in the coordinates of `sig` (left -> right).

    Without `expected` this is the Auto Gel wizard's detector. With `expected` the
    prominence/spacing are swept until exactly that many lanes are found; if no setting
    gives the count, the closest result is returned (and, as a last resort for a clean
    gel, the width is split evenly)."""
    sig = np.asarray(sig, dtype=float)
    H, W = sig.shape
    col = _lane_column_profile(sig, smooth)
    if np.ptp(col) < 1e-9:
        return {"bounds": [], "method": "flat"}
    dist = int(distance) if distance else 15

    def run(prom, d):
        pk = _find_peaks_padded(col, prom, d)
        # adaptive sweep (wizard): adopt the most regularly spaced set if clearly better
        best, best_s = pk, _regularity(pk)
        for p in np.linspace(max(0.011, prom), 0.01, 9):
            cand = _find_peaks_padded(col, float(p), d)
            s = _regularity(cand)
            if s > best_s * 1.15:
                best, best_s = cand, s
        pk = best
        if len(pk) >= 3:                              # snap to the lane pitch
            xi = np.arange(len(pk), dtype=float)
            c = np.polyfit(xi, pk.astype(float), 1)
            if abs(c[0]) > 2:
                fit = np.polyval(c, xi)
                pk = np.array([int(round(f)) if abs(r - f) < 0.25 * abs(c[0]) else int(r)
                               for r, f in zip(pk, fit)], dtype=int)
        return _bounds_from_peaks(col, pk, W, d)

    if not expected:
        # Three detectors; the lane set with the most REGULAR spacing wins (real lanes sit
        # on an even pitch, artefacts don't).
        cands = [(run(prominence, dist), "Auto Gel peak detection")]
        pitch = lane_pitch(col)
        if pitch:
            cands.append((run(prominence, max(dist, int(0.6 * pitch))),
                          "peak detection at lane pitch %.0f px" % pitch))
        try:
            cands.append((_gap_lanes(_band_column_profile(sig)), "gap segmentation"))
        except Exception:
            pass

        def score(b):
            return _regularity([(x0 + x1) / 2.0 for x0, x1 in b]) if len(b) >= 2 else 0.0
        bounds, how = max(cands, key=lambda c: score(c[0]))
        warn = lane_layout_warning(bounds)
        if warn and pitch:
            # Uneven result: fit the lane count implied by the lane pitch instead, and have
            # the caller confirm that count with the user.
            n_est = int(round(W / pitch))
            if 2 <= n_est <= 40 and abs(n_est - len(bounds)) <= max(1, 0.3 * len(bounds)):
                return {"bounds": _grid_lanes(sig, n_est), "estimated_count": n_est,
                        "method": "%d lanes estimated from the lane pitch (%.0f px); peak "
                                  "detection gave %d uneven lanes" % (n_est, pitch, len(bounds))}
        return {"bounds": bounds, "method": how, "warning": warn}

    expected = int(expected)
    pitch = W / float(max(1, expected))
    try:
        grid = fit_target_lanes(sig, expected)["bounds"]
        grid_how = "%d-lane comb fit (equal spacing, borders snapped to the gaps between lanes)" % expected
    except Exception:
        grid = _grid_lanes(sig, expected)
        grid_how = "%d-lane grid snapped to the gaps between lanes" % expected
    reg = lambda b: _regularity([(x0 + x1) / 2.0 for x0, x1 in b]) if len(b) >= 2 else 0.0
    found = None
    for d in sorted({dist, max(3, int(pitch * 0.5)), max(3, int(pitch * 0.7))}):
        for prom in (0.25, 0.15, 0.1, 0.07, 0.05, 0.035, 0.025, 0.015, 0.01):
            bounds = run(prom, d)
            if len(bounds) == expected and (found is None or reg(bounds) > reg(found)):
                found = bounds
    def even_widths(b):
        # every lane (the outer ones too) within 0.6-1.6x the median lane width
        w = np.array([x1 - x0 for x0, x1 in b], dtype=float)
        med = float(np.median(w)) if len(w) else 0.0
        return med > 0 and float(w.min()) >= 0.6 * med and float(w.max()) <= 1.6 * med
    if found is not None and even_widths(found) and reg(found) >= 0.9 * reg(grid):
        return {"bounds": found, "method": "peak detection (matched expected count)"}
    return {"bounds": grid, "method": grid_how}


# ============================================================================ bands
def lane_profile(sig, x0, x1, y0=0, y1=None):
    """Mean band signal per row over the central 60 % of a lane (edges excluded so a
    neighbour's smile/bleed does not leak in)."""
    sig = np.asarray(sig, dtype=float)
    y1 = sig.shape[0] if y1 is None else y1
    w = max(1, x1 - x0)
    a = int(round(x0 + 0.2 * w))
    b = max(a + 1, int(round(x1 - 0.2 * w)))
    return sig[y0:y1, a:b].mean(axis=1)


def detect_bands(profile, snr_min=DEFAULT_SNR_MIN, rel_min=DEFAULT_REL_MIN,
                 min_width=DEFAULT_MIN_WIDTH, is_ladder=False, abs_floor=0.0):
    """True bands in one lane profile (high = band). Returns (bands, noise_sigma).

    A peak is kept only if it is (1) >= snr_min x the lane's noise sigma above the local
    baseline, (2) at least min_width px wide at half height, (3) for sample lanes at least
    rel_min of the lane's strongest band, and (4) above `abs_floor` (gel-wide floor)."""
    p = np.asarray(profile, dtype=float)
    n = p.size
    if n < 8:
        return [], 0.0
    raw = median_filter(p, size=3)
    sm = gaussian_filter1d(raw, sigma=max(0.8, n / 600.0))
    bg = baseline(sm, size=max(15, int(n * 0.12)))
    net = sm - bg
    sigma = max(robust_noise_sigma(raw), background_sigma(net))
    prom = max(snr_min * sigma, 1e-12)
    pk, props = find_peaks(net, prominence=prom, height=max(prom, abs_floor),
                           distance=max(2, n // 200), width=max(1.0, min_width * 0.5))
    if len(pk) == 0:
        return [], sigma
    w, _, lips, rips = peak_widths(net, pk, rel_height=0.5)
    heights = net[pk]
    keep = (w >= min_width) & (heights >= snr_min * sigma)
    if not is_ladder and keep.any():
        keep &= heights >= rel_min * float(np.max(heights[keep]))
    bands = []
    for i in np.nonzero(keep)[0]:
        lo = int(max(0, np.floor(lips[i] - 0.5 * w[i])))
        hi = int(min(n - 1, np.ceil(rips[i] + 0.5 * w[i])))
        bands.append({"y": int(pk[i]), "height": float(heights[i]),
                      "snr": float(heights[i] / sigma), "fwhm": float(w[i]),
                      "area": float(np.sum(np.clip(net[lo:hi + 1], 0, None)))})
    return bands, sigma


# ============================================================================ ladder & standards
def pick_ladder_lanes(lanes, ladder_count=None):
    """Indices (0-based) of lanes that look like a molecular-weight ladder: many bands
    spread over most of the lane, ideally at the gel's edge. Returns (indices, scores)."""
    if not lanes:
        return [], []
    H = max(1, max((ln["height_px"] for ln in lanes), default=1))
    scores = []
    for i, ln in enumerate(lanes):
        b = ln["bands"]
        if len(b) < 4:
            scores.append(0.0)
            continue
        ys = [x["y_rel"] for x in b]
        spread = (max(ys) - min(ys)) / float(H)
        s = len(b) * max(0.0, spread)
        if ladder_count:
            s *= np.exp(-abs(len(b) - ladder_count) / max(2.0, 0.3 * ladder_count))
        if i in (0, len(lanes) - 1):
            s *= 1.25                                   # ladders usually flank the gel
        scores.append(float(s))
    best = max(scores)
    if best <= 0:
        return [], scores
    # a ladder must clearly beat the typical sample lane
    med = float(np.median([s for s in scores])) if len(scores) > 2 else 0.0
    picks = [i for i, s in enumerate(scores) if s >= 0.7 * best and s > 1.5 * med]
    return picks, scores


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def find_standard_series(lanes, exclude=(), height_px=1):
    """Look for a band at the SAME position and of SIMILAR size in >= 3 neighbouring lanes
    whose INTENSITY differs (typically a dilution series of a known standard).

    Returns a list of candidates (best first): lanes (1-based), y (image px), areas,
    ratio_to_weakest, trend ('increasing'/'decreasing'/'mixed'), confidence."""
    pts = []
    edge = 0.04 * height_px          # wells / dye front are not standards
    for i, ln in enumerate(lanes):
        if i in exclude:
            continue
        for b in ln["bands"]:
            yr = b.get("y_rel", b["y"])
            if edge <= yr <= height_px - edge:
                pts.append((b["y"], i, b))
    if len(pts) < 3:
        return []
    fw = np.median([p[2]["fwhm"] for p in pts]) if pts else 5.0
    tol = max(0.012 * height_px, 0.6 * fw, 3.0)
    pts.sort(key=lambda t: t[0])
    clusters = []
    for y, i, b in pts:
        for c in clusters:
            if abs(y - c["y"]) <= tol and i not in c["lanes"]:
                c["lanes"][i] = b
                c["y"] = float(np.mean([bb["y"] for bb in c["lanes"].values()]))
                break
        else:
            clusters.append({"y": float(y), "lanes": {i: b}})
    cands = []
    order = [i for i in range(len(lanes)) if i not in exclude]
    for c in clusters:
        idx = [i for i in order if i in c["lanes"]]
        if len(idx) < 3:
            continue
        # maximal runs of neighbouring (non-ladder) lanes
        runs, cur = [], [idx[0]]
        for a, b in zip(idx, idx[1:]):
            if order.index(b) - order.index(a) == 1:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        for run in runs:
            if len(run) < 3:
                continue
            bands = [c["lanes"][i] for i in run]
            areas = np.array([b["area"] for b in bands])
            widths = np.array([b["fwhm"] for b in bands])
            if areas.min() <= 0:
                continue
            width_ok = widths.max() / max(1e-9, widths.min()) <= 2.0
            dyn = float(areas.max() / areas.min())
            rho = _spearman(np.arange(len(run)), areas)
            if not width_ok or dyn < 1.8:
                continue
            conf = min(1.0, 0.35 + 0.4 * abs(rho) + 0.05 * len(run) + 0.05 * min(dyn, 6.0))
            # Standards are usually purified protein / DNA: few bands per lane. A band
            # shared by busy multi-band lanes is more likely a common contaminant.
            busy = float(np.mean([len(lanes[i]["bands"]) for i in run]))
            if busy > 4:
                conf *= 0.6
            cands.append({
                "lanes": [i + 1 for i in run],
                "y": int(round(c["y"])),
                "areas": [round(float(a), 2) for a in areas],
                "ratio_to_weakest": [round(float(a / areas.min()), 2) for a in areas],
                "trend": "increasing" if rho >= 0.9 else "decreasing" if rho <= -0.9 else "mixed",
                "confidence": round(conf if abs(rho) >= 0.9 else conf * 0.6, 2),
            })
    cands.sort(key=lambda d: (-d["confidence"], -len(d["lanes"])))
    return cands[:3]


# ============================================================================ one call
def analyze(gray, max_val, origin=(0, 0), gel_type="auto", expected_lanes=None,
            ladder_count=None, snr_min=DEFAULT_SNR_MIN, rel_min=DEFAULT_REL_MIN,
            min_width=DEFAULT_MIN_WIDTH, assume_dark_bands=None):
    """Full read of a gel region. `gray` is the analysis image cropped to the region,
    `origin` its top-left in image pixels. Returns a JSON-able dict."""
    gray = np.asarray(gray, dtype=float)
    H, W = gray.shape
    ox, oy = origin
    pol = band_polarity(gray, max_val)
    if assume_dark_bands is None:
        dark = pol["polarity"] != "light_bands"
    else:
        dark = bool(assume_dark_bands)
    sig = band_signal(gray, max_val, dark_bands=dark)
    lane_res = detect_lanes(sig, expected=expected_lanes)
    lanes = []
    sigmas = []
    profiles = []
    for (x0, x1) in lane_res["bounds"]:
        prof = lane_profile(sig, x0, x1)
        profiles.append(prof)
    # gel-wide floor: 1 % of the strongest band anywhere (kills ripples in empty lanes)
    peak_max = 0.0
    for prof in profiles:
        raw = median_filter(prof, size=3)
        sm = gaussian_filter1d(raw, sigma=max(0.8, len(prof) / 600.0))
        peak_max = max(peak_max, float(np.max(sm - baseline(sm, max(15, int(len(sm) * 0.12))))))
    abs_floor = 0.01 * peak_max
    for li, ((x0, x1), prof) in enumerate(zip(lane_res["bounds"], profiles)):
        bands, s = detect_bands(prof, snr_min, rel_min, min_width, is_ladder=False,
                                abs_floor=abs_floor)
        sigmas.append(s)
        lanes.append({"lane": li + 1, "x0": int(ox + x0), "x1": int(ox + x1),
                      "center_x": int(ox + (x0 + x1) / 2), "height_px": H,
                      "bands": [dict(b, y_rel=b["y"], y=int(oy + b["y"])) for b in bands]})
    ladder_idx, scores = pick_ladder_lanes(lanes, ladder_count)
    # Re-detect ladder lanes without the relative-intensity gate (ladder bands vary a lot)
    for i in ladder_idx:
        x0, x1 = lane_res["bounds"][i]
        bands, _ = detect_bands(profiles[i], snr_min, rel_min, min_width, is_ladder=True,
                                abs_floor=abs_floor)
        lanes[i]["bands"] = [dict(b, y_rel=b["y"], y=int(oy + b["y"])) for b in bands]
    series = find_standard_series(lanes, exclude=set(ladder_idx), height_px=H)
    total = max([sum(b["area"] for b in ln["bands"]) for ln in lanes] or [0.0])
    out_lanes = []
    for i, ln in enumerate(lanes):
        lane_total = sum(b["area"] for b in ln["bands"])
        out_lanes.append({
            "lane": ln["lane"], "x0": ln["x0"], "x1": ln["x1"], "center_x": ln["center_x"],
            "role_guess": "ladder" if i in ladder_idx else ("empty" if not ln["bands"] else "sample"),
            "n_bands": len(ln["bands"]),
            "relative_load": round(lane_total / total, 3) if total > 0 else 0.0,
            "bands": [{"y": b["y"], "snr": round(b["snr"], 1), "fwhm_px": round(b["fwhm"], 1),
                       "rel_intensity": round(b["height"] / max(1e-9, max(bb["height"] for bb in ln["bands"])), 3)}
                      for b in ln["bands"]],
        })
    return {
        "region": {"x": int(ox), "y": int(oy), "width": int(W), "height": int(H)},
        "polarity": pol,
        "bands_are_dark": dark,
        "lane_detection": lane_res["method"],
        "lane_warning": None if expected_lanes else lane_res.get("warning"),
        "lane_count_estimated": lane_res.get("estimated_count"),
        "n_lanes": len(out_lanes),
        "lanes": out_lanes,
        "ladder_lanes": [i + 1 for i in ladder_idx],
        "standard_series_candidates": series,
        "noise": {"snr_min": snr_min, "min_relative_intensity": rel_min,
                  "min_band_width_px": min_width,
                  "median_lane_noise_sigma": round(float(np.median(sigmas)), 4) if sigmas else None},
    }


# ============================================================================ ladder fit
def _mw_number(v):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
    return float(m.group(0)) if m else None


def choose_ladder_bands(positions, heights, labels, max_combos=250_000):
    """More ladder bands detected than the preset has labels: choose the subset (in order)
    whose positions best follow a smooth log10(MW)-vs-migration curve (quadratic), with a
    small preference for strong bands. Falls back to the strongest bands when the labels
    are not numbers or there are too many combinations. Returns (chosen positions, method)."""
    pos = [int(p) for p in positions]
    n_lab = len(labels)
    if len(pos) <= n_lab:
        return pos, "all"
    mw = [_mw_number(v) for v in labels]
    h = np.array([float(heights.get(p, 0.0)) for p in pos])

    def strongest():
        idx = sorted(np.argsort(-h)[:n_lab])
        return [pos[i] for i in idx], "strongest bands"
    if any(m is None or m <= 0 for m in mw) or n_lab < 3:
        return strongest()
    try:
        from math import comb
        if comb(len(pos), n_lab) > max_combos:
            return strongest()
    except Exception:
        return strongest()
    lm = np.log10(np.array(mw, dtype=float))
    order = np.argsort(pos)
    ys = np.array(pos, dtype=float)[order]
    hs = h[order]
    hmax = max(1e-12, float(hs.max()))
    rng = max(1e-9, float(np.ptp(lm)))
    best, best_s = None, None
    for c in itertools.combinations(range(len(ys)), n_lab):
        yy = ys[list(c)]
        coef = np.polyfit(yy, lm, 2)
        r = float(np.sqrt(np.mean((np.polyval(coef, yy) - lm) ** 2))) / rng
        pen = float(np.mean(-np.log(np.maximum(hs[list(c)] / hmax, 1e-6))))
        sc = r + 0.02 * pen
        if best_s is None or sc < best_s:
            best, best_s = c, sc
    return [int(ys[i]) for i in best], "best fit to a smooth log(MW) curve"
