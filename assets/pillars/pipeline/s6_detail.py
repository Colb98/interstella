# Step 6: native-resolution detail crops of the head of the left pillar (P1, with the EGGs) for close fly-ins.
# Hubble heic1501a at its native 0.040"/px and JWST weic2216b resampled onto the same grid; stars removed
# with the 2048-px star masks (upsampled) + push-pull inpainting.
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
REF, O, DST = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 2048, 2136
X0N, Y0N, WN, HN = 1840, 750, 1536, 912          # native Hubble pixels
hst = Image.open(REF + '/heic1501a.jpg'); HWn, HHn = hst.size
sx, sy = HWn / W, HHn / H
crop = np.asarray(hst.crop((X0N, Y0N, X0N + WN, Y0N + HN)), dtype=np.float32) / 255
print('rect (working px):', X0N / sx, Y0N / sy, (X0N + WN) / sx, (Y0N + HN) / sy)


def pushpull(img, w):
    if min(w.shape) <= 2:
        m = (img * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-6)
        return np.broadcast_to(m, img.shape).copy()
    Hh, Ww = w.shape; h2, w2 = (Hh + 1) // 2, (Ww + 1) // 2
    pi = np.pad(img * w[..., None], ((0, h2 * 2 - Hh), (0, w2 * 2 - Ww), (0, 0)), mode='edge')
    pw = np.pad(w, ((0, h2 * 2 - Hh), (0, w2 * 2 - Ww)), mode='edge')
    sw = pw.reshape(h2, 2, w2, 2).sum((1, 3)); si = pi.reshape(h2, 2, w2, 2, -1).sum((1, 3))
    coarse = pushpull(si / np.maximum(sw, 1e-6)[..., None], np.minimum(sw, 1.0))
    up = np.repeat(np.repeat(coarse, 2, 0), 2, 1)[:Hh, :Ww]
    up = np.stack([ndi.gaussian_filter(up[..., c], 1.0) for c in range(up.shape[-1])], -1)
    return img * w[..., None] + up * (1 - w[..., None])


def upmask(mname):
    m = np.load(O + '/' + mname)
    yy, xx = np.mgrid[0:HN, 0:WN]
    mu = ndi.map_coordinates(m, [(yy + Y0N + 0.5) / sy - 0.5, (xx + X0N + 0.5) / sx - 0.5], order=1)
    mu = ndi.grey_dilation(mu, size=5)
    return np.clip(mu * 1.5, 0, 1)


def clean(img, mask):
    w = (mask < 0.05).astype(np.float32)
    f = pushpull(img, w)
    rng = np.random.default_rng(3)
    hp = img - np.stack([ndi.gaussian_filter(img[..., c], 1.5) for c in range(3)], -1)
    sd = np.sqrt(np.stack([ndi.gaussian_filter(hp[..., c] ** 2 * w, 20) for c in range(3)], -1) / np.maximum(ndi.gaussian_filter(w, 20), 1e-3)[..., None])
    f = f + rng.standard_normal(f.shape).astype(np.float32) * sd * (1 - w)[..., None]
    a = np.clip(mask, 0, 1)[..., None]
    return np.clip(img * (1 - a) + f * a, 0, 1)


hc = clean(crop, upmask('hst_starmask.npy'))
Image.fromarray((hc * 255 + 0.5).astype(np.uint8)).save(DST + '/hst_head.webp', quality=92, method=6)

# JWST onto the same native grid (same registration as step 1)
def NE(theta):
    t = np.radians(theta)
    return np.array([-np.sin(t), np.cos(t)]), np.array([-np.cos(t), -np.sin(t)])
Nh, Eh = NE(34.98); Nj, Ej = NE(-2.5646)
jw = Image.open(REF + '/weic2216b.jpg'); JW, JH = jw.size
sJ = (4.21 * 60 / JW + 7.30 * 60 / JH) / 2
sH = (4.53 * 60 / HWn + 4.72 * 60 / HHn) / 2
dE = 2.46 * 15 * np.cos(np.radians(13.85)); dN = -(51 * 60 + 6.26 - (50 * 60 + 21.97))
yy, xx = np.mgrid[0:HN, 0:WN].astype(np.float64)
# + star-matched refinement of the published WCS: NIRCam sits (20.6, 21.6) working px (2.73", 2.87") off
right = (xx + X0N + 0.5 - HWn / 2) * sH + 20.6 * sH * sx; up = -(yy + Y0N + 0.5 - HHn / 2) * sH - 21.6 * sH * sy
vN = right * Nh[0] + up * Nh[1] - dN; vE = right * Eh[0] + up * Eh[1] - dE
r2 = vN * Nj[0] + vE * Ej[0]; u2 = vN * Nj[1] + vE * Ej[1]
cj = r2 / sJ + JW / 2 - 0.5; rj = -u2 / sJ + JH / 2 - 0.5
x0, x1, y0, y1 = int(cj.min()) - 8, int(cj.max()) + 8, int(rj.min()) - 8, int(rj.max()) + 8
sub = np.asarray(jw.crop((x0, y0, x1, y1)), dtype=np.float32) / 255
sub = np.stack([ndi.gaussian_filter(sub[..., c], 0.45) for c in range(3)], -1)    # 0.030 → 0.040"/px: light prefilter
jc = np.stack([ndi.map_coordinates(sub[..., c], [rj - y0, cj - x0], order=3, mode='nearest') for c in range(3)], -1).clip(0, 1)
jcc = clean(jc, upmask('jw_starmask.npy'))
Image.fromarray((jcc * 255 + 0.5).astype(np.uint8)).save(DST + '/jwst_head.webp', quality=92, method=6)
print('done', hc.shape, jcc.shape)
