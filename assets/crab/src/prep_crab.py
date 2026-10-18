# Preprocess Crab Nebula reference images into Interstella assets.
# Common frame: centred on the Crab pulsar, north up, west (= image right) +X, L arcsec square, N px.
import sys, json, os, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
# reference downloads (see ../CREDITS.txt): large_heic0515a.jpg, large_weic2326a.jpg,
# cxc_2008_crab_crab_lg.jpg, gaia.tsv (4.5'), gaia8.tsv (9', G < 20.8). Usage:
#   CRAB_REF=<dir>/ N=4096 NH=4096 NA=2048 python -I prep_crab.py <outdir>
REF = os.environ.get('CRAB_REF', './ref/')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
os.makedirs(OUT, exist_ok=True)
N = int(os.environ.get('N', 2560)); L = 400.0
AS = L / N                       # arcsec per output px
# --- source geometry ------------------------------------------------------------
HST_SC = 6.41 * 60 / 3864        # arcsec/px (ESA/Hubble: FOV 6.41', north up)
HST_PSR = (2014.5, 1958.8)       # pulsar pixel (from RA/Dec offsets, checked by eye)
# complex similarity jwst px -> hst px, from match_stars.py (6 field stars, rms 1.7 Hubble px)
JW2HST = np.array([0.3140059868439497-0.011703080243992112j, 219.6944231076841+503.7398224477379j])
CXC_SC = 120.0 / 244 * 792 / 3300  # 2 arcmin bar = 244 px on the 792 px scale image
CXC_PSR = (1670.0, 1732.0)

def aces(x): return np.clip(x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14), 0, 1)
def inv_aces(y):
    y = np.clip(y, 0, 0.985)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    A = a - c*y; B = b - d*y; C = -e*y
    return (-B + np.sqrt(B*B - 4*A*C)) / (2*A)

def warp(img, src_of_out):
    """img: PIL RGB; src_of_out(X,Y arcsec) -> (sx, sy) affine; returns float array N x N x 3 and coverage."""
    # affine via three points
    xs = (np.arange(N) + 0.5 - N/2) * AS; ys = -(np.arange(N) + 0.5 - N/2) * AS
    p0 = np.array(src_of_out(xs[0], ys[0])); px = np.array(src_of_out(xs[0] + AS, ys[0])); py = np.array(src_of_out(xs[0], ys[0] - AS))
    a, d = px - p0; b, e = py - p0
    c0 = p0 - 0.5*np.array([a + b, d + e])   # PIL samples at pixel corners
    out = img.transform((N, N), Image.AFFINE, (a, b, c0[0], d, e, c0[1]), Image.BICUBIC)
    ones = Image.new('L', img.size, 255).transform((N, N), Image.AFFINE, (a, b, c0[0], d, e, c0[1]), Image.BILINEAR)
    return np.asarray(out).astype(np.float32) / 255, np.asarray(ones).astype(np.float32) / 255

def prescale(img, f):
    if f <= 1.05: return img, 1.0
    W, H = img.size
    return img.resize((round(W/f), round(H/f)), Image.LANCZOS), W / round(W/f)

# ---------------------------------------------------------------------------------
def load_hst():
    im = Image.open(REF + 'large_heic0515a.jpg').convert('RGB')
    im, f = prescale(im, AS / HST_SC)
    sc = HST_SC * f
    pc = (HST_PSR[0] / f, HST_PSR[1] / f)
    return warp(im, lambda X, Y: (pc[0] + X / sc, pc[1] - Y / sc))

def load_jw():
    im = Image.open(REF + 'large_weic2326a.jpg').convert('RGB')
    z, t = JW2HST
    jsc = HST_SC * abs(z)       # arcsec per jwst px
    im, f = prescale(im, AS / jsc)
    iz, it = 1 / z, -t / z      # hst px -> jwst px
    def m(X, Y):
        h = complex(HST_PSR[0] + X / HST_SC, HST_PSR[1] - Y / HST_SC)
        j = iz * h + it
        return (j.real / f, j.imag / f)
    return warp(im, m)

def load_cxc():
    im = Image.open(REF + 'cxc_2008_crab_crab_lg.jpg').convert('RGB')
    im, f = prescale(im, AS / CXC_SC)
    sc = CXC_SC * f; pc = (CXC_PSR[0] / f, CXC_PSR[1] / f)
    return warp(im, lambda X, Y: (pc[0] + X / sc, pc[1] - Y / sc))

# ---------------------------------------------------------------------------------
GAIA_SHIFT = (-0.3, -0.1)   # arcsec: Gaia (2016) -> this frame (median offset of matched stars)
def load_gaia(path):
    RA0, DE0 = 83.63322, 22.01446
    out = []
    for ln in open(path):
        p = ln.rstrip(chr(10)).split(chr(9))
        if len(p) < 3 or not p[0][:1].isdigit(): continue
        try: ra, de, g = float(p[0]), float(p[1]), float(p[2])
        except ValueError: continue
        try: br = float(p[3])
        except (ValueError, IndexError): br = 1.0
        X = -(ra - RA0) * np.cos(np.radians(DE0)) * 3600 + GAIA_SHIFT[0]; Y = (de - DE0) * 3600 + GAIA_SHIFT[1]
        out.append((X, Y, g, br))
    return out

def gaia_discs(stars, k=1.0, add=0.0):
    out = []
    for X, Y, g, br in stars:
        if abs(X) > L/2 + 5 or abs(Y) > L/2 + 5 or np.hypot(X, Y) < 1.0: continue
        r = min((0.33 + 0.09 * 10 ** (0.2 * (20.5 - g))) * k + add, 3.2 * k)
        out.append((N/2 + X / AS, N/2 - Y / AS, r / AS))
    return out

def patch_fill(a, discs, mask):
    """textured inpainting: each star disc is replaced by a nearby unmasked patch (feathered, level-matched)"""
    out = a.copy()
    rng = np.random.default_rng(3)
    for x, y, R in discs:
        P = int(R * 1.6) + 3
        x0, y0 = int(x) - P, int(y) - P
        if x0 < 0 or y0 < 0 or x0 + 2*P + 1 > N or y0 + 2*P + 1 > N: continue
        yy, xx = np.mgrid[y0:y0 + 2*P + 1, x0:x0 + 2*P + 1]
        d = np.hypot(xx + 0.5 - x, yy + 0.5 - y)
        w = np.clip((R * 1.35 - d) / (R * 0.35 + 1), 0, 1)          # 1 inside the disc, feathered edge
        ring = (d > R * 1.1) & (d < R * 1.6 + 2)
        best = None
        for ang in rng.permutation(8) * (np.pi / 4):
            for dist in (2.6 * R + 4, 3.6 * R + 6):
                sx, sy = int(round(x0 + np.cos(ang) * dist)), int(round(y0 + np.sin(ang) * dist))
                if sx < 0 or sy < 0 or sx + 2*P + 1 > N or sy + 2*P + 1 > N: continue
                if mask[sy:sy + 2*P + 1, sx:sx + 2*P + 1].any(): continue
                best = (sx, sy); break
            if best: break
        dst = out[y0:y0 + 2*P + 1, x0:x0 + 2*P + 1]
        if best is None: continue
        src = a[best[1]:best[1] + 2*P + 1, best[0]:best[0] + 2*P + 1]
        lvl = (dst[ring].mean(0) + 1e-4) / (src[ring].mean(0) + 1e-4)
        dst[:] = dst * (1 - w[..., None]) + src * np.clip(lvl, 0.3, 3.0) * w[..., None]
    return out

def gaia_mask(stars, k=1.0, add=0.0):
    """discs over Gaia stars, radius ~ sqrt(flux): 0.35'' at G = 20.5 up to 3'' for G < 13."""
    m = np.zeros((N, N), bool)
    for X, Y, g, br in stars:
        if abs(X) > L/2 + 5 or abs(Y) > L/2 + 5: continue
        if np.hypot(X, Y) < 1.0: continue                     # the pulsar has its own mask
        r = min((0.33 + 0.09 * 10 ** (0.2 * (20.5 - g))) * k + add, 3.2 * k)   # arcsec
        x = N/2 + X / AS; y = N/2 - Y / AS; R = r / AS
        x0, x1 = int(max(x - R - 1, 0)), int(min(x + R + 2, N)); y0, y1 = int(max(y - R - 1, 0)), int(min(y + R + 2, N))
        if x0 >= x1 or y0 >= y1: continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= R * R
    return m

def find_stars(g, thr, r=3, sep=1.0):
    """compact point sources in radiance image g: local max of top-hat, ring test."""
    bg = ndi.grey_opening(g, size=(2*r+5, 2*r+5))
    th = ndi.gaussian_filter(g - bg, 0.7)
    mx = ndi.maximum_filter(th, size=2*r+1)
    ys, xs = np.nonzero((th == mx) & (th > thr))
    keep = []
    for y, x in zip(ys, xs):
        if y < 8 or x < 8 or y >= g.shape[0]-8 or x >= g.shape[1]-8: continue
        v = th[y, x]; R = r + 2
        ring = np.concatenate([th[y-R, x-R:x+R+1], th[y+R, x-R:x+R+1], th[y-R:y+R+1, x-R], th[y-R:y+R+1, x+R]])
        if np.percentile(ring, 80) > 0.25 * v: continue          # elongated (a filament), not a star
        # compact: at 2 px the profile must already have dropped well below the peak (knots are broader)
        r2 = np.concatenate([th[y-2, x-2:x+3], th[y+2, x-2:x+3], th[y-1:y+2, x-2], th[y-1:y+2, x+2]])
        if r2.mean() > 0.45 * v: continue
        keep.append((x, y, v))
    return keep

def inpaint(a, mask, sigma=3.0, it=5):
    """fill masked pixels with normalised-convolution of the unmasked neighbourhood."""
    w = (~mask).astype(np.float32)
    out = a.copy()
    for s in [sigma * (1.6 ** k) for k in range(it)]:
        num = ndi.gaussian_filter(out * w[..., None], (s, s, 0)) if a.ndim == 3 else ndi.gaussian_filter(out * w, s)
        den = ndi.gaussian_filter(w, s) + 1e-6
        fill = num / (den[..., None] if a.ndim == 3 else den)
        sel = mask & (den > 0.05)
        out[sel] = fill[sel]; w = np.maximum(w, sel.astype(np.float32))
    return out

def star_mask(stars, flux_px, base=2.0, img=None):
    """discs over the stars; the radius grows with the star's aperture flux (halo of saturated stars)."""
    m = np.zeros((N, N), bool)
    W = 40
    yy, xx = np.mgrid[-W:W+1, -W:W+1]
    for x, y, v in stars:
        if img is not None:
            win = img[max(y-4,0):y+5, max(x-4,0):x+5]
            ring = np.concatenate([img[max(y-7,0), max(x-7,0):x+8], img[min(y+7,N-1), max(x-7,0):x+8]])
            fl = np.clip(win - np.median(ring), 0, None).sum() / flux_px
            r = base + 0.9 * np.sqrt(fl)
        else:
            r = base + 2.2 * np.sqrt(v / flux_px)
        r = min(r, 22)
        disc = (xx*xx + yy*yy) <= r*r
        y0, x0 = y - W, x - W
        ys = slice(max(y0, 0), min(y0 + 2*W+1, N)); xs = slice(max(x0, 0), min(x0 + 2*W+1, N))
        m[ys, xs] |= disc[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]
    return m

def low_envelope(ch, size_as=6.0, pct=25, ds=4):
    """smooth lower envelope (percentile filter on a downsampled copy): the diffuse synchrotron."""
    small = ndi.zoom(ch, 1/ds, order=1)
    k = max(3, int(round(size_as / (AS*ds))) | 1)
    env = ndi.percentile_filter(small, pct, size=k)
    env = ndi.gaussian_filter(env, k / 2.5)
    return np.clip(ndi.zoom(env, (ch.shape[0]/small.shape[0], ch.shape[1]/small.shape[1]), order=1), 0, None)

PCT = float(os.environ.get('PCT', 40))
def synch_rgb(lin, envB, c, nm, nsoft):
    """per-channel lower envelope; R and G capped near the synchrotron colour so filament haze stays in F."""
    # one fixed colour c (the shader reconstructs RGB = S * c), scaled to stay below every channel's envelope
    S = low_envelope(lin[..., 2], 7.0, PCT) / c[2]
    for k in (0, 1):
        S = np.minimum(S, low_envelope(lin[..., k], 7.0, PCT) / c[k])
    S = ndi.gaussian_filter(S * nm, 1.5 / AS) * nsoft
    return S[..., None] * c[None, None, :]

def deficit(lin, syn):
    """absorption A so that lin - syn*(1-A) >= 0 in every channel (keeps filament hues clean)"""
    k = np.min(lin / (syn + 1e-5), axis=2)
    return np.clip(1 - k, 0, 0.92)

def blue_to_syn(F, c):
    """fine synchrotron wisps left in F (blue-dominant) go back to the synchrotron map, so the
    filament shell carries only line / dust emission; exact: F' + S'*c == F + S*c."""
    m = np.min(F / c[None, None, :], axis=2)
    blu = F[..., 2] / (np.maximum(F[..., 0], F[..., 1]) + 1e-4)
    w = np.clip((blu - 0.85) / 0.4, 0, 1)
    mv = m * w
    return F - mv[..., None] * c[None, None, :], mv

def enc(x):       # HDR radiance -> 8-bit: (x/(1+x))^(1/2.2)
    x = np.clip(x, 0, None); return np.clip((x/(1+x)) ** (1/2.2), 0, 1)

def save(a, name, q=88, mode='RGB', size=None):
    u8 = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8, mode if u8.ndim == 3 else 'L')
    if size and size != im.size[0]: im = im.resize((size, size), Image.LANCZOS)
    im.save(os.path.join(OUT, name), quality=q, method=6)
    print(name, os.path.getsize(os.path.join(OUT, name)) // 1024, 'KB')

def nebula_mask(g, thr_frac=0.035, sig_as=4.0):
    s = ndi.gaussian_filter(g, sig_as / AS)
    m = s > thr_frac * np.percentile(s, 99.5)
    lab, n = ndi.label(m)
    c = lab[N//2, N//2]
    m = lab == c
    m = ndi.binary_fill_holes(m)
    soft = ndi.gaussian_filter(m.astype(np.float32), 4.0 / AS)
    return m, np.clip(soft * 1.6 - 0.3, 0, 1)

# =================================================================================
if __name__ == '__main__':
    yy, xx = np.mgrid[0:N, 0:N]
    Xas = (xx + 0.5 - N/2) * AS; Yas = -(yy + 0.5 - N/2) * AS

    # ---------------- HST ----------------
    hst, hcov = load_hst()
    hl = inv_aces(hst ** 2.2)                      # display -> radiance under the host's ACES
    g = hl.mean(2)
    fpx = np.percentile(g, 99.0)
    gaia = load_gaia(REF + 'gaia.tsv')
    psr_m = (Xas**2 + Yas**2) < 1.6**2
    sm = gaia_mask(gaia) | psr_m
    hl_c = patch_fill(inpaint(hl, sm), gaia_discs(gaia), sm)
    nm, nsoft = nebula_mask(hl_c.mean(2))
    # star catalogue for the sprite overlay: Gaia DR3 within 9' (G < 20.8): X, Y (arcsec), G, BP-RP
    cat = [[round(X, 2), round(Y, 2), round(g, 2), round(br, 2)] for X, Y, g, br in load_gaia(REF + 'gaia8.tsv') if np.hypot(X, Y) > 1.0]
    cat.sort(key=lambda r: r[2])
    print('catalogue', len(cat))
    # synchrotron: blue lower envelope; colour from pixels where the envelope explains most of the light
    envB = low_envelope(hl_c[..., 2], 7.0, 30)
    ratio = hl_c[..., 2] / (envB + 1e-4)
    pure = nm & (ratio < 1.15) & (envB > np.percentile(envB[nm], 60))
    csyn = np.median(hl_c[pure] / hl_c[pure][:, 2:3], axis=0)
    print('hst synchrotron colour (per unit B)', csyn, pure.sum())
    syn = synch_rgb(hl_c, envB, csyn, nm, nsoft)
    S = syn[..., 2]
    # dark lanes: light missing below the synchrotron envelope (grey, synchrotron-weighted), so that
    # F + syn * (1 - A) reproduces the image exactly
    A_h = deficit(hl_c, syn) * nsoft
    F = np.clip(hl_c - syn * (1 - A_h[..., None]), 0, None) * nsoft[..., None]
    F, S_h_add = blue_to_syn(F, csyn)
    # saturated-blue JPEG specks (B clipped at 255): keep their blue on the synchrotron side
    exc = np.clip(F[..., 2] - 1.5 * np.maximum(F[..., 0], F[..., 1]) - 0.02, 0, None)
    F[..., 2] -= exc; S_h_add = S_h_add + exc
    syn = syn + S_h_add[..., None] * csyn; S = syn[..., 2]

    # ---------------- JWST ----------------
    jw, jcov = load_jw()
    jl = inv_aces(jw ** 2.2)
    gj = jl.mean(2)
    fpj = np.percentile(gj[jcov > 0.99], 99.0)
    smj = gaia_mask(gaia, k=1.5, add=0.25) | psr_m
    jl_c = patch_fill(inpaint(jl, smj), gaia_discs(gaia, k=1.5, add=0.25), smj)
    envBj = low_envelope(jl_c[..., 2], 7.0, 30)
    ratio = jl_c[..., 2] / (envBj + 1e-4)
    purej = nm & (jcov > 0.99) & (ratio < 1.15) & (envBj > np.percentile(envBj[nm & (jcov > 0.99)], 60))
    csynj = np.median(jl_c[purej] / jl_c[purej][:, 2:3], axis=0)
    print('jwst synchrotron colour', csynj, purej.sum())
    synj = synch_rgb(jl_c, envBj, csynj, nm, nsoft)
    Sj = synj[..., 2]
    A_j = deficit(jl_c, synj) * nsoft
    Fj = np.clip(jl_c - synj * (1 - A_j[..., None]), 0, None) * nsoft[..., None]
    Fj, S_j_add = blue_to_syn(Fj, csynj)
    exc = np.clip(Fj[..., 2] - 1.5 * np.maximum(Fj[..., 0], Fj[..., 1]) - 0.02, 0, None)
    Fj[..., 2] -= exc; S_j_add = S_j_add + exc
    synj = synj + S_j_add[..., None] * csynj; Sj = synj[..., 2]
    # outside the Webb field: recolour Hubble (filament luminance -> Webb's median filament colour)
    cov = np.clip((ndi.distance_transform_edt(jcov > 0.99) * AS - 1.0) / 6.0, 0, 1)
    both = nm & (cov > 0.99)
    k_f = np.median(Fj[both].sum(1)) / max(np.median(F[both].sum(1)), 1e-4)
    cf = np.median(Fj[both & (Fj.sum(2) > np.percentile(Fj[both].sum(1), 70))], axis=0); cf = cf / cf.sum()
    k_s = np.median(Sj[both]) / max(np.median(S[both]), 1e-4)
    print('jwst fill gains', k_f, k_s, cf)
    Ffill = F.sum(2)[..., None] * k_f * cf[None, None, :]
    Fj = Fj * cov[..., None] + Ffill * (1 - cov[..., None])
    Sj = Sj * cov + S * k_s * (1 - cov)
    A_j = A_j * cov + A_h * (1 - cov)

    # ---------------- Chandra ----------------
    cx, ccov = load_cxc()
    v = cx.mean(2)
    floor = np.percentile(v[ccov > 0.99], 20)
    v = np.clip((v - floor) / (1 - floor), 0, 1)
    xl = inv_aces(v ** 2.2)
    xl = inpaint(xl, (Xas**2 + Yas**2) < 1.8**2)
    xl = ndi.gaussian_filter(xl, 0.35 / AS)                 # Chandra counts are noisy at this sampling
    xm = ndi.gaussian_filter((xl > 0.02).astype(np.float32), 6 / AS)
    xl *= np.clip(xm * 2, 0, 1)

    # ---------------- outline R(theta) ----------------
    nb = 180
    th = np.arctan2(Yas, Xas)
    rr = np.hypot(Xas, Yas)
    b = ((th + np.pi) / (2*np.pi) * nb).astype(int) % nb
    R = np.zeros(nb)
    for i in range(nb):
        sel = (b == i) & nm
        R[i] = np.percentile(rr[sel], 99.5) if sel.any() else 0
    R = ndi.gaussian_filter1d(R, 1.0, mode='wrap')
    print('outline R (arcsec) min/max', R.min(), R.max())

    # ---------------- write ----------------
    save(enc(F), 'crab_hst.webp', 90, size=int(os.environ.get('NH', N)))
    save(enc(Fj), 'crab_jwst.webp', 90)
    aux = np.stack([enc(S), enc(Sj), enc(xl)], -1)
    save(aux, 'crab_aux.webp', 92, size=int(os.environ.get('NA', N)))
    dust = np.stack([A_h, A_j, np.zeros_like(A_h)], -1)
    NA = int(os.environ.get('NA', N // 2))
    small = lambda a: np.asarray(Image.fromarray((np.clip(a, 0, 1)*255+.5).astype(np.uint8)).resize((NA, NA), Image.LANCZOS))
    Image.fromarray(small(dust)).save(os.path.join(OUT, 'crab_dust.webp'), quality=90, method=6)
    print('crab_dust.webp', os.path.getsize(os.path.join(OUT, 'crab_dust.webp')) // 1024, 'KB')
    meta = dict(L=L, N=N, arcsec_per_px=AS, synHST=[round(float(c), 4) for c in csyn], synJWST=[round(float(c), 4) for c in csynj],
                R=[round(float(r), 1) for r in R], stars=cat)
    json.dump(meta, open(os.path.join(OUT, 'crab_meta.json'), 'w'), separators=(',', ':'))
    print('stars kept', len(cat))
    # previews (what the canonical view should reproduce)
    prev = lambda a: (np.clip(aces(a), 0, 1) ** (1/2.2) * 255).astype(np.uint8)
    Image.fromarray(prev(F + syn * (1 - A_h[..., None] * 0))).resize((1024, 1024)).save('pv_hst_recon.jpg')
    Image.fromarray(prev(F)).resize((1024, 1024)).save('pv_hst_F.jpg')
    Image.fromarray(prev(syn)).resize((1024, 1024)).save('pv_hst_S.jpg')
    Image.fromarray(prev(Fj)).resize((1024, 1024)).save('pv_jw_F.jpg')
    Image.fromarray(prev(synj)).resize((1024, 1024)).save('pv_jw_S.jpg')
    Image.fromarray(prev(np.repeat(xl[..., None], 3, 2))).resize((1024, 1024)).save('pv_x.jpg')
    Image.fromarray((np.clip(dust, 0, 1) * 255).astype(np.uint8)).resize((1024, 1024)).save('pv_dust.jpg')
