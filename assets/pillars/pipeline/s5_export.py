# Step 5: pack textures + star catalogue for js/scenes/nebulae/pillars.js
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from PIL import Image
from scipy import ndimage as ndi
O, DST = sys.argv[2], sys.argv[3]
rng = np.random.default_rng(16)
W, H = 2048, 2136
FW = 7.5131; LYPX = FW / W; FH = H * LYPX
D0 = 12.4
g = np.load(O + '/geo.npz')
alpha, zc, hh, front, dout = g['alpha'], g['zc'], g['hh'], g['front'], g['dout']
hst = np.load(O + '/hst_clean.npy'); jw = np.load(O + '/jw_clean.npy'); cov = np.load(O + '/jwcov_ext1.0.npy')
ONLY_STARS = '--stars' in sys.argv


def lin(disp):
    y = np.clip(disp, 0, 0.985) ** 2.2
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    A, B, C = y * c - a, y * d - b, y * e
    return (-B - np.sqrt(np.maximum(B * B - 4 * A * C, 0))) / (2 * A)


def u8(a): return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def pushpull(img, w):
    """pyramid inpainting: fill where w=0 from coarser averages of the known pixels"""
    if min(w.shape) <= 2:
        m = (img * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-6)
        return np.broadcast_to(m, img.shape).copy()
    Hh, Ww = w.shape; h2, w2 = (Hh + 1) // 2, (Ww + 1) // 2
    pi = np.pad(img * w[..., None], ((0, h2 * 2 - Hh), (0, w2 * 2 - Ww), (0, 0)), mode='edge')
    pw = np.pad(w, ((0, h2 * 2 - Hh), (0, w2 * 2 - Ww)), mode='edge')
    sw = pw.reshape(h2, 2, w2, 2).sum((1, 3)); si = pi.reshape(h2, 2, w2, 2, -1).sum((1, 3))
    ci = si / np.maximum(sw, 1e-6)[..., None]; cw = np.minimum(sw, 1.0)
    coarse = pushpull(ci, cw)
    up = np.repeat(np.repeat(coarse, 2, 0), 2, 1)[:Hh, :Ww]
    up = np.stack([ndi.gaussian_filter(up[..., c], 1.0) for c in range(up.shape[-1])], -1)
    return img * w[..., None] + up * (1 - w[..., None])


def normconv(img, w, sigmas):
    out = img.copy(); filled = w > 0.99
    for s in sigmas:
        den = ndi.gaussian_filter(w, s)
        num = np.stack([ndi.gaussian_filter(img[..., c] * w, s) for c in range(img.shape[-1])], -1)
        ok = (den > 0.25) & ~filled
        out[ok] = (num / np.maximum(den, 1e-6)[..., None])[ok]; filled |= ok
    if (~filled).any():
        m = img[w > 0.5].mean(0); out[~filled] = m
    return out


# ---- JWST: fill the three corners the NIRCam mosaic does not cover --------------------------
cs = ndi.gaussian_filter(ndi.binary_erosion(cov > 0.5, iterations=14).astype(np.float32), 5)   # the mosaic edge has a dark border
covin = ((cov > 0.5) & (ndi.distance_transform_edt(cov > 0.5) > 18)).astype(np.float32)
if not ONLY_STARS:
    fillj = pushpull(jw, covin)
    grain = rng.standard_normal(jw.shape).astype(np.float32) * 0.012
    jwf = jw * cs[..., None] + (fillj + grain) * (1 - cs[..., None])

    # ---- background sheets: pillars inpainted away, extended 3W x 2H, quarter resolution -------
    def sheet(img, name):
        m = ndi.binary_dilation(alpha > 0.03, iterations=10)
        w = (~m).astype(np.float32)
        f = pushpull(img, w)
        pad = np.pad(f, ((H // 2, H // 2), (W, W), (0, 0)), mode='reflect')
        yy, xx = np.mgrid[0:pad.shape[0], 0:pad.shape[1]]
        dx = np.maximum(np.maximum(W - xx, xx - 2 * W), 0); dy = np.maximum(np.maximum(H // 2 - yy, yy - (H // 2 + H)), 0)
        dist = np.hypot(dx, dy)[::4, ::4]
        small = pad[::4, ::4]
        bl = np.stack([ndi.gaussian_filter(small[..., c], 60) for c in range(3)], -1)
        k = np.clip(dist / (0.15 * W), 0, 1)[..., None]
        k = k * k * (3 - 2 * k)
        out = small * (1 - k) + bl * k
        # far out: drift toward the mean colour of the sheet (no mirror symmetry visible)
        mean = f[~m].mean(0)
        k2 = np.clip(dist / (0.8 * W), 0, 1)[..., None] ** 1.5
        out = out * (1 - k2) + mean * k2
        Image.fromarray(u8(out)).save(DST + f'/bg_{name}.jpg', quality=90)
        return out
    sheet(hst, 'hst'); sheet(jwf, 'jw')

    # ---- dust colour (for the sides of the pillars seen off-axis): the dust source function averaged over
    # dusty pixels only, so the background never bleeds into it; quarter resolution, display encoding
    def aces(x): return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    def dustcol(img, bgimg, name):
        P = lin(img); Bl = lin(bgimg)
        a = np.clip(alpha, 0, 0.993)[..., None]
        E = np.clip((P - Bl * (1 - a)) / np.maximum(a, 0.25), 0, 6)
        w = (alpha ** 2) * (alpha > 0.25)
        num = np.stack([ndi.gaussian_filter(E[..., c] * w, 6) for c in range(3)], -1)
        den = ndi.gaussian_filter(w, 6)
        ok = den > 0.02
        Ed = np.where(ok[..., None], num / np.maximum(den, 1e-6)[..., None], 0)
        Ed4 = Ed[::4, ::4]; w4 = ok[::4, ::4].astype(np.float32)
        Ed4 = pushpull(Ed4, w4)
        Image.fromarray(u8(aces(Ed4) ** (1 / 2.2)), 'RGB').save(DST + f'/dust_{name}.jpg', quality=92)
    def fullbg(img):
        m = ndi.binary_dilation(alpha > 0.03, iterations=10)
        return pushpull(img, (~m).astype(np.float32))
    dustcol(hst, fullbg(hst), 'hst'); dustcol(jwf, fullbg(jwf), 'jw')

    # ---- photo textures: RGB = star-free photo, A = canonical dust alpha (hst) / ionisation-front mask (jw)
    # no alpha channels anywhere: browsers decode RGBA images premultiplied, which would erase data
    Image.fromarray(u8(hst), 'RGB').save(DST + '/hst.webp', quality=92, method=6)
    Image.fromarray(u8(jwf), 'RGB').save(DST + '/jwst.webp', quality=92, method=6)
    Image.fromarray(np.dstack([u8(alpha), u8(front), u8(np.minimum(dout, 0.6) / 0.6)]), 'RGB').save(DST + '/matte.webp', lossless=True, method=6)

    # ---- geometry at half resolution: R = h/2.5 ly, G = (zc+4)/8, B = empty-space distance/0.6 ly ----
    dusty = alpha > 0.02
    _, (iy, ix) = ndi.distance_transform_edt(~dusty, return_indices=True)
    zce = zc[iy, ix]                               # extend depth outward so bilinear filtering stays clean
    def half(a): return a.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    # quarter resolution, 16 bit: h, zc, and the z-range [zlo, zhi] that any slab (incl. its noise and the glow
    # shell) occupies within 0.3 ly - a conservative bound for empty-space skipping of oblique rays
    def quarter(a): return a.reshape(H // 4, 4, W // 4, 4).mean((1, 3))
    q16 = lambda a, lo, hi: np.round(np.clip((a - lo) / (hi - lo), 0, 1) * 65535).astype('<u2')
    GZ = 0.3
    occ = (alpha > 0.002) | (dout < GZ)
    shell = np.where(dout < GZ, GZ, 0.0)
    lo = np.where(occ, zce - 1.6 * hh - shell - 0.05, 9.0)
    hi = np.where(occ, zce + 1.6 * hh + shell + 0.05, -9.0)
    R = int(0.3 / LYPX)
    fp = np.hypot(*np.mgrid[-R:R + 1, -R:R + 1]) <= R
    lo4 = ndi.minimum_filter(lo[::2, ::2], footprint=fp[::2, ::2])[::2, ::2]
    hi4 = ndi.maximum_filter(hi[::2, ::2], footprint=fp[::2, ::2])[::2, ::2]
    geo = np.dstack([q16(quarter(hh), 0, 2.5), q16(quarter(zce), -4, 4), q16(lo4, -9, 9), q16(hi4, -9, 9)])[::-1]   # rows bottom-up (GL)
    geo.tofile(DST + '/geo.bin')
    print('geo.bin', geo.shape)


# ---- stars --------------------------------------------------------------------------------------
# The releases render stars with a size that grows with brightness (stretched, saturated cores), so each
# star is described by what the photo shows: a Gaussian disc (peak A in scene radiance, width sigma in
# photo px, colour), a faint wing, and diffraction spikes of relative strength S.
SPK = {'hst': [46, 134, 224, 316], 'jw': [37, 97, 157, 217, 277, 337]}
LS = {'hst': 1.6, 'jw': 1.6}


def fit_star(disp, L, Lm, Dm, x, y, k, Dl=None):
    """Fit the light that star removal took out of the photo (Dl = lin(photo) - lin(star-free), per channel):
    background-free, so rims and nebulosity cannot inflate the fit."""
    rs = np.concatenate([[0.0, 0.5], np.arange(1.0, 72.0, 1.0)])
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False) + 0.21
    ys = y + rs[:, None] * np.sin(ang)[None]; xs = x + rs[:, None] * np.cos(ang)[None]
    DD = Dl.mean(-1)
    v = ndi.map_coordinates(DD, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rs), -1)
    dv = ndi.map_coordinates(Dm, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rs), -1)
    pr = np.percentile(v, 50, axis=1); pd = np.percentile(dv, 50, axis=1)
    unsat = pd < 0.95
    pk = max(pr[0], pr[1])
    use = unsat & (pr > max(0.01, 0.01 * pk)) & (rs <= 40)
    if use.sum() >= 2 and (use & (rs > 0)).any():
        r2 = rs[use] ** 2; lp = np.log(pr[use]); wts = pr[use]
        A_ = np.vstack([np.ones_like(r2), -r2]).T
        lnA, inv2s2 = np.linalg.lstsq(A_ * wts[:, None], lp * wts, rcond=None)[0]
        sig = np.sqrt(0.5 / inv2s2) if inv2s2 > 1e-4 else 25.0
        A = np.exp(lnA)
    else:
        sig, A = 0.9, max(pk, 0.005)
    sig = float(np.clip(sig, 0.75, 25.0))
    if unsat[0]:
        sig = min(sig, 2.2); A = min(A, max(pk, 1e-3) * 1.4)
    else:
        A = max(A, pk * 1.05)
    m = rs <= max(2.5 * sig, 2.0)
    col = np.array([max(ndi.map_coordinates(Dl[..., c], [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rs), -1)[m].clip(0).sum(), 1e-6) for c in range(3)])
    col = col / col.mean()
    S = 0.0
    a = np.arange(max(2.0, 1.5 * sig), max(6.0, 4.5 * sig), 1.0)
    vals = []
    for t in SPK[k]:
        sx = x + a * np.cos(np.radians(t)); sy = y - a * np.sin(np.radians(t))
        sv = ndi.map_coordinates(DD, [sy, sx], order=1, mode='nearest')
        halo = np.interp(a, rs, pr)
        ok = ndi.map_coordinates(Dm, [sy, sx], order=1, mode='nearest') < 0.95
        ex = (sv - halo)[ok]
        den = (A * np.exp(-a / (LS[k] * sig)))[ok]
        if len(ex): vals.append(np.median(ex / np.maximum(den, 1e-4)))
    if vals: S = float(np.clip(np.median(vals), 0, 4))
    return A, sig, S, col


def fit_all(k, src, rows_xy, removed=True):
    disp = np.load(O + '/' + src); L = lin(disp); Lm = L.mean(-1); Dm = disp.max(-1)
    if removed:
        clean = np.load(O + ('/hst_clean.npy' if k == 'hst' else '/jw_clean.npy'))
        Dl = np.clip(L - lin(clean), 0, None)
    else:   # faint stars left in the texture: subtract a local median background instead
        Dl = np.clip(L - np.stack([ndi.median_filter(L[..., c], 9) for c in range(3)], -1), 0, None)
    return [fit_star(disp, L, Lm, Dm, x, y, k, Dl) for x, y in rows_xy]


out = {}
for k, src in [('hst', 'hst.npy'), ('jw', 'jw_ext1.0.npy')]:
    cat = np.load(O + f'/catfinal_{k}.npy')
    rem = cat[cat[:, 7] > 0.5]
    fits = fit_all(k, src, rem[:, :2])
    # bound the disc by the scale at which the detector found the star (pass 0 = big saturated discs)
    lvl = rem[:, 6]
    smax = np.where(lvl <= 0, 25.0, np.where(lvl == 1, 6.0, 2.2))
    amax = np.where(lvl <= 1, 50.0, 2.5)
    fits = [(min(f[0], am), min(f[1], sm), f[2], f[3]) for f, sm, am in zip(fits, smax, amax)]
    # colours: the releases render stars in a narrow range of hues; faint fits are noisy, so pull the
    # colour toward the median of well-measured stars. Saturated stars always show diffraction spikes.
    A = np.array([f[0] for f in fits])
    good = (A > 0.3) & (A < 2.5)
    med = np.median(np.array([f[3] for f, g_ in zip(fits, good) if g_]), 0)
    # Hubble: the release's stars are peach-white (keep the median hue); NIRCam: white-yellow with real
    # colour differences (reddened and embedded stars), relative to the median
    target = med if k == 'hst' else np.array([1.0, 0.95, 0.82])
    keep = 0.35 if k == 'hst' else 0.7
    fits = [(a_, s_, max(S_, (0.6 if k == 'hst' else 0.8) * float(np.clip((a_ - 1.0) / 2.0, 0, 1))),
             target * np.clip(c_ / med, 0.3, 3.0) ** keep) for a_, s_, S_, c_ in fits]
    if k == 'jw': JW_MED = med
    print(k, 'median star colour', med.round(3))
    out[k] = (rem, fits)
    A = np.array([f[0] for f in fits]); sg = np.array([f[1] for f in fits]); S = np.array([f[2] for f in fits])
    print(k, 'sprites', len(rem), 'A pct', np.percentile(A, [10, 50, 90, 99]).round(3), 'sigma pct', np.percentile(sg, [10, 50, 90, 99]).round(2), 'S pct', np.percentile(S, [50, 90, 99]).round(3))

# faint JWST pool (kept in the texture): used to synthesise stars where there is no NIRCam data
catj = np.load(O + '/catfinal_jw.npy'); pmask = (catj[:, 7] < 0.5) & (catj[:, 6] >= 2)
pool = catj[pmask]
pool = pool[(pool[:, 0] > 80) & (pool[:, 0] < W - 80) & (pool[:, 1] > 80) & (pool[:, 1] < H - 80)]
pool = pool[rng.choice(len(pool), min(2500, len(pool)), replace=False)]
poolJ = fit_all('jw', 'jw_ext1.0.npy', pool[:, :2], removed=False)
poolJ = [(a_, min(s_, 1.3), S_, np.array([1.0, 0.95, 0.82]) * np.clip(c_ / JW_MED, 0.3, 3.0) ** 0.7) for a_, s_, S_, c_ in poolJ]
NPOOL_RATIO = pmask.sum() / len(pool)
print('pool', len(pool), 'of', pmask.sum())

(rh_, Fh), (rj_, Fj) = out['hst'], out['jw']
used = np.zeros(len(rj_), bool)
stars = []
def covI(x, y): return cov[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] > 0.5
NONE = (0.0, 1.0, 0.0, np.zeros(3))
for i, r in enumerate(rh_):
    d = np.hypot(rj_[:, 0] - r[0], rj_[:, 1] - r[1]); j = d.argmin()
    if d[j] < 2.5 and not used[j]:
        used[j] = True; stars.append([r[0], r[1], Fh[i], Fj[j], 0])
    elif covI(r[0], r[1]):
        stars.append([r[0], r[1], Fh[i], NONE, 0])          # JWST keeps it in the texture
    else:
        A, s, S, c = Fh[i]                                    # outside the NIRCam mosaic: borrow the Hubble disc
        stars.append([r[0], r[1], Fh[i], (A * 0.8, s * 0.9, S, np.array([0.95, 1.0, 1.05])), 0])
for j, r in enumerate(rj_):
    if not used[j]: stars.append([r[0], r[1], NONE, Fj[j], int(r[8] > 0.5)])
print('stars', len(stars))


def frame(x, y): return (x - W / 2) * LYPX, (H / 2 - y) * LYPX


def bbT(T):  # approximate blackbody chromaticity (linear sRGB, luminance 1) for the natural palette
    r = np.clip(1.0 if T < 6600 else 1.29 * ((T / 100 - 60) ** -0.133), 0, 1.3)
    gg = np.clip(0.39 * np.log(T / 100) - 0.63 if T < 6600 else 1.13 * ((T / 100 - 60) ** -0.0755), 0, 1.3)
    b = np.clip(0.0 if T < 1900 else (0.54 * np.log(T / 100 - 10) - 1.2 if T < 6600 else 1.0), 0, 1.3)
    c = np.array([r, gg, b]) ** 2.2; return c / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def enc(fit):
    A, s, S, c = fit
    return [float(f'{v:.4g}') for v in list(A * np.asarray(c))] + [round(float(s), 2), round(float(S), 3)]


rows = []
for x, y, fh, fj, emb in stars:
    X, Y = frame(x, y)
    xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    a = alpha[yi, xi]; z0, h0 = zc[yi, xi], hh[yi, xi]
    vis_h = fh[0] > 0
    fj = list(fj)
    if emb:
        z = z0 + rng.uniform(-0.4, 0.4) * h0; kind = 2
        fj[0] = fj[0] / max((1 - min(a, 0.99)) ** 0.15, 0.15)
    elif vis_h and a > 0.15:
        z = min(z0 + h0 + rng.uniform(0.3, 2.5), 6.0); kind = 1
    elif a > 0.15:
        z = z0 - h0 - rng.uniform(0.5, 40.0); kind = 0
        fj[0] = fj[0] / max((1 - min(a, 0.99)) ** 0.3, 0.1)
    else:
        if rng.random() < 0.3: z = rng.uniform(-4.5, 4.0); kind = 1
        else: z = -np.exp(rng.uniform(np.log(30), np.log(600))); kind = 0
    s = (D0 - z) / D0
    # natural colour: temperature from the JWST colour (F444W vs F090W) when available
    if fj[0] > 0:
        cj = fj[3]; T = float(np.clip(5800 * (cj[2] / max(cj[0], 1e-3)) ** 1.6, 2800, 30000))
    else: T = float(np.exp(rng.uniform(np.log(3200), np.log(12000))))
    fn = (fh[0], fh[1], fh[2], bbT(T) * 0.9) if vis_h else NONE
    rows.append([round(X * s, 4), round(Y * s, 4), round(float(z), 3)] + enc(fh) + enc(fn) + enc(tuple(fj)) + [kind])

# the ionising O stars of NGC 6611, ~7.5 ly above the image centre (HD 168076 at PA ~ 300 deg from the pillars)
OSTARS = [(2.75, 7.5, -0.3, 1.0, 40000), (2.2, 7.9, 0.4, 0.62, 38000), (3.6, 8.6, -0.8, 0.35, 34000), (1.4, 8.2, 0.9, 0.3, 33000),
          (3.1, 6.6, 0.2, 0.25, 31000), (4.4, 7.4, -0.2, 0.2, 30000), (2.0, 9.4, -1.1, 0.18, 30000), (0.6, 7.2, 0.3, 0.15, 25000),
          (5.2, 8.9, 0.6, 0.12, 22000), (3.9, 10.3, 0.0, 0.12, 22000), (1.1, 10.8, -0.4, 0.1, 20000), (4.9, 6.3, -0.9, 0.1, 18000)]
for X, Y, z, f, T in OSTARS:
    c = bbT(T)
    sg = 4.0 + 12.0 * f
    rows.append([X, Y, z] + enc((6.0, sg, 0.35, c * np.array([1.1, 0.9, 1.0]))) + enc((6.0, sg, 0.35, c)) + enc((6.0, sg * 0.9, 0.5, c * np.array([0.8, 0.95, 1.2]))) + [3])
poolH = [enc(f) for f in Fh]
json.dump({'D0': D0, 'FW': FW, 'FH': FH, 'LYPX': LYPX,
           'cols': 'x y z | hst: A*rgb, sigma (photo px), spike | nat: same | jw: same | kind (0 far/background, 1 local, 2 embedded protostar, 3 NGC 6611 O star)',
           'stars': rows, 'poolH': poolH, 'poolJ': [enc(f) for f in poolJ], 'poolRatio': round(float(NPOOL_RATIO), 3)},
          open(DST + '/stars.json', 'w'), separators=(',', ':'))
print('rows', len(rows))
