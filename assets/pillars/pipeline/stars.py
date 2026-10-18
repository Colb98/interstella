# Point-source detection and removal on sRGB float images (H,W,3).
import numpy as np
from scipy import ndimage as ndi

def ring_k(L, r, k, n=16):
    """k-th largest (0 = max) of n samples on a ring of radius r around every pixel"""
    ang = np.arange(n) * 2 * np.pi / n
    R = np.stack([ndi.shift(L, (-r * np.sin(a), -r * np.cos(a)), order=1, mode='nearest') for a in ang])
    R.sort(0)
    return R[-1 - k]

def spike_region(shape, cat, spikes, extra=0.0):
    """raster of each star's allowed removal region: disk(rr) + spikes(len sp)"""
    H, W = shape
    m = np.zeros(shape, np.float32)
    for cx, cy, rr, sp in cat:
        R = int(max(rr, sp) + 3)
        x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 2), max(0, int(cy) - R), min(H, int(cy) + R + 2)
        if x0 >= x1 or y0 >= y1: continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - cx, cy - yy
        v = np.clip(rr + 1 - np.hypot(dx, dy), 0, 1)
        if sp > rr:
            for t in spikes:
                c, s = np.cos(np.radians(t)), np.sin(np.radians(t))
                along, across = dx * c + dy * s, np.abs(-dx * s + dy * c)
                v = np.maximum(v, np.clip(2.2 + extra - across, 0, 1) * (along > 0) * (along < sp))
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], v)
    return m

def detect(img, passes, spikes, colour_test=None):
    """passes: list of (blur, median, ringR, ringSkip, win, thr). Returns catalog rows
       (x, y, peak, fr, fg, fb, level) — later passes skip what earlier ones found."""
    L = img.mean(-1)
    H, W = L.shape
    cat = []
    taken = np.zeros(L.shape, bool)
    for lvl, (blur, med, rr, skip, win, thr) in enumerate(passes):
        Lb = ndi.gaussian_filter(L, blur) if blur > 0 else L
        res = Lb - ndi.median_filter(L, size=med)
        score = Lb - ring_k(Lb, rr, skip)
        mx = ndi.maximum_filter(Lb, size=win)
        cand = np.argwhere((Lb == mx) & (res > thr) & (score > thr * 0.6) & ~taken)
        hw = max(2, int(rr))
        for (y, x) in cand:
            if y < hw + 3 or x < hw + 3 or y >= H - hw - 3 or x >= W - hw - 3: continue
            loc = img[y - 2 * hw:y + 2 * hw + 1, x - 2 * hw:x + 2 * hw + 1]
            if loc.shape[0] < 4 * hw + 1 or loc.shape[1] < 4 * hw + 1: continue
            b = np.median(np.concatenate([loc[0], loc[-1], loc[:, 0], loc[:, -1]]), 0)
            sl = (slice(y - hw, y + hw + 1), slice(x - hw, x + hw + 1))
            fl = np.clip(img[sl] - b, 0, None).sum((0, 1))
            if colour_test is not None and not colour_test(fl, res[y, x]): continue
            w = np.clip(res[sl], 0, None); yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1]; s = w.sum() + 1e-9
            cat.append((x + (w * xx).sum() / s, y + (w * yy).sum() / s, res[y, x], fl[0], fl[1], fl[2], lvl))
        # block the area around stars found so far (core + spikes) for the next passes
        if cat:
            c = np.array(cat)
            f = c[:, 3:6].sum(1)
            rr_ = 2.5 + 1.2 * np.sqrt(f); sp_ = np.where(f > 30, 6 * np.sqrt(f), 0)
            taken = spike_region(L.shape, np.stack([c[:, 0], c[:, 1], rr_, sp_], 1), spikes) > 0.5
    return np.array(cat)

def removal_mask(img, cat, spikes, t=0.03, rk=1.0, spk=6.0, spmin=30, dil=True):
    """mask = (pixels standing above the local median) ∩ (allowed region around a catalogued star)"""
    L = img.mean(-1)
    r1 = L - ndi.median_filter(L, size=9)
    small = ndi.median_filter(ndi.zoom(L, 0.25, order=1), size=11)      # coarse median for the big halos
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    up = ndi.map_coordinates(small, [yy * (small.shape[0] / L.shape[0]), xx * (small.shape[1] / L.shape[1])], order=1, mode='nearest')
    r2 = L - up
    f = cat[:, 3:6].sum(1)
    big = f > 25
    rr = np.minimum(3.0 + 1.5 * np.sqrt(f), 75) * rk
    sp = np.where(f > spmin, spk * np.sqrt(f), 0)
    allow_s = spike_region(L.shape, np.stack([cat[~big, 0], cat[~big, 1], rr[~big], sp[~big]], 1), spikes)
    allow_b = spike_region(L.shape, np.stack([cat[big, 0], cat[big, 1], rr[big], sp[big]], 1), spikes, extra=1.0)
    m = np.maximum((r1 > t) * allow_s, (np.maximum(r1, r2) > t) * allow_b)
    # always remove the innermost core
    core = spike_region(L.shape, np.stack([cat[:, 0], cat[:, 1], np.minimum(rr, 1.8 + 0.4 * np.sqrt(f)), np.zeros(len(cat))], 1), spikes)
    m = np.maximum(m, core)
    if dil: m = ndi.grey_dilation(m, size=3)
    return np.clip(ndi.gaussian_filter(m, 0.6) * 1.3, 0, 1)

def inpaint(img, mask, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0)):
    w = 1 - np.clip(mask * 1.5, 0, 1)
    out = img.copy(); filled = np.zeros(mask.shape, bool)
    target = mask > 0.01
    for s in sigmas:
        den = ndi.gaussian_filter(w, s)
        ok = (den > 0.3) & target & ~filled
        if ok.any():
            num = np.stack([ndi.gaussian_filter(img[..., c] * w, s) for c in range(img.shape[-1])], -1)
            out[ok] = (num / np.maximum(den, 1e-6)[..., None])[ok]
        filled |= ok
    rest = target & ~filled
    if rest.any():
        den = ndi.gaussian_filter(w, 64); num = np.stack([ndi.gaussian_filter(img[..., c] * w, 64) for c in range(img.shape[-1])], -1)
        out[rest] = (num / np.maximum(den, 1e-6)[..., None])[rest]
    a = mask[..., None]
    return img * (1 - a) + out * a

def halo_disks(img, cat, fmin=25, rmax=160):
    """for bright stars: radius where the azimuthal-median profile has merged into the local background"""
    L = img.mean(-1)
    out = []
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for row in cat:
        cx, cy = row[0], row[1]; f = row[3:6].sum()
        if f < fmin: continue
        rs = np.arange(1.0, rmax, 1.0)
        ys = cy + rs[:, None] * np.sin(ang)[None]; xs = cx + rs[:, None] * np.cos(ang)[None]
        v = ndi.map_coordinates(L, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rs), -1)
        p = np.median(v, 1)
        p = ndi.uniform_filter1d(p, 5)
        b = np.percentile(p[len(p) // 2:], 30)
        amp = p[0] - b
        rh = rs[-1]
        for i, r in enumerate(rs):
            if p[i] - b < max(0.012, 0.03 * amp): rh = r; break
        out.append((cx, cy, rh))
    return np.array(out)

def add_grain(clean, img, mask, seed=1):
    """re-add pixel noise inside inpainted areas so they don't look airbrushed"""
    rng = np.random.default_rng(seed)
    hp = img - np.stack([ndi.gaussian_filter(img[..., c], 1.5) for c in range(3)], -1)
    w = (mask < 0.01).astype(np.float32)
    var = np.stack([ndi.gaussian_filter(hp[..., c] ** 2 * w, 12) for c in range(3)], -1) / np.maximum(ndi.gaussian_filter(w, 12), 1e-3)[..., None]
    n = rng.standard_normal(clean.shape).astype(np.float32)
    n = np.stack([ndi.gaussian_filter(n[..., c], 0.6) for c in range(3)], -1) * 1.6
    return clean + n * np.sqrt(np.maximum(var, 0)) * mask[..., None]

def photometry(img, cat, r_ap=6.0, r_in=10.0, r_out=16.0):
    """uniform aperture photometry: rgb flux inside r_ap above the annulus median"""
    H, W = img.shape[:2]
    out = np.zeros((len(cat), 3))
    R = int(r_out + 1)
    for i, (cx, cy) in enumerate(cat[:, :2]):
        x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 1), max(0, int(cy) - R), min(H, int(cy) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        loc = img[y0:y1, x0:x1]
        ann = loc[(d >= r_in) & (d < r_out)]
        b = np.median(ann, 0) if len(ann) else 0
        out[i] = np.clip(loc[d < r_ap] - b, 0, None).sum(0)
    return out

def subtract_halos(img, cat, fmin=20, rmax=110, q=30):
    """Subtract each bright star's radially symmetric halo (q-th percentile over angle, so spikes,
       neighbours and nebular structure on one side don't bias it). Returns (image, core list)."""
    out = img.copy()
    H, W = img.shape[:2]
    cores = []
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    f = cat[:, 3:6].sum(1)
    order = np.argsort(-f)
    for i in order:
        if f[i] < fmin: continue
        cx, cy = cat[i, 0], cat[i, 1]
        rs = np.arange(0.0, rmax, 1.0)
        ys = cy + rs[:, None] * np.sin(ang)[None]; xs = cx + rs[:, None] * np.cos(ang)[None]
        prof = []
        for c in range(3):
            v = ndi.map_coordinates(out[..., c], [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rs), -1)
            prof.append(np.percentile(v, q, axis=1))
        prof = np.array(prof)                       # (3, R)
        ps = ndi.uniform_filter1d(prof.mean(0), 5)
        floor = ps[5:].min(); amp = ps[0] - floor
        rh = rs[-1]
        for k in range(3, len(rs)):
            if ps[k] - floor < max(0.008, 0.02 * amp): rh = rs[k]; break
        if rh < 4: continue
        # saturated core radius (no information left there)
        rc = 0.0
        for k in range(len(rs)):
            if prof[:, k].max() < 0.93: rc = rs[k]; break
        base = prof[:, min(int(rh) + 2, len(rs) - 1)]
        R = int(rh + 2)
        x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 2), max(0, int(cy) - R), min(H, int(cy) + R + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        taper = np.clip((rh - d) / max(3.0, 0.25 * rh), 0, 1)
        for c in range(3):
            halo = np.interp(d, rs, np.clip(prof[c] - base[c], 0, None)) * taper
            out[y0:y1, x0:x1, c] -= halo
        cores.append((cx, cy, rc, rh, i))
    return np.clip(out, 0, 1), np.array(cores)

def concentration(img, cat, r1=1.6, r2=6.0, ri=9.0, ro=14.0):
    L = img.mean(-1); H, W = L.shape
    out = np.zeros(len(cat))
    R = int(ro + 1)
    for i, (cx, cy) in enumerate(cat[:, :2]):
        x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 1), max(0, int(cy) - R), min(H, int(cy) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]; d = np.hypot(xx - cx, yy - cy); loc = L[y0:y1, x0:x1]
        ann = loc[(d >= ri) & (d < ro)]; b = np.median(ann) if len(ann) else 0
        a1 = np.clip(loc[d < r1] - b, 0, None).sum(); a2 = np.clip(loc[d < r2] - b, 0, None).sum()
        out[i] = a1 / (a2 + 1e-6)
    return out

def spike_score(img, cat, spikes, r0=6, r1=26):
    L = img.mean(-1)
    sp = np.radians(np.array(spikes[:6] if len(spikes) > 4 else spikes))
    inter = sp + (np.pi / len(sp))
    rs = np.arange(r0, r1, 1.0)
    out = np.zeros(len(cat))
    for i, (cx, cy) in enumerate(cat[:, :2]):
        def ring(angs):
            xs = cx + rs[:, None] * np.cos(angs)[None]; ys = cy - rs[:, None] * np.sin(angs)[None]
            return ndi.map_coordinates(L, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rs), -1)
        a, b = ring(sp), ring(inter)
        out[i] = np.median(a.mean(1) - b.mean(1)) / (np.median(b) + 1e-3)
    return out
