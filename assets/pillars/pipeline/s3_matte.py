# Step 3: dust opacity matte (canonical-view alpha) from Hubble + JWST colour.
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
O = sys.argv[1]
h = np.load(O + '/hst_clean.npy')
j = np.load(O + '/jw_ext1.0.npy')
cov = np.load(O + '/jwcov_ext1.0.npy')
H, W = h.shape[:2]
eps = 0.02
def bgest(f, mask=None, q=15, win=75, s=40):
    small = ndi.zoom(f, 0.25, order=1)
    b = ndi.percentile_filter(small, q, size=win)
    b = ndi.gaussian_filter(b, s / 4)
    yy, xx = np.mgrid[0:H, 0:W]
    return ndi.map_coordinates(b, [yy * b.shape[0] / H, xx * b.shape[1] / W], order=1, mode='nearest')
hs = np.stack([ndi.gaussian_filter(h[..., c], 1.0) for c in range(3)], -1)
f1 = np.log((hs[..., 0] + eps) / (hs[..., 2] + eps))
d1 = f1 - bgest(f1)
js = np.stack([ndi.median_filter(j[..., c], 5) for c in range(3)], -1)
f2 = np.log((js[..., 0] + eps) / (js[..., 2] + eps))
f2f = np.where(cov > 0.5, f2, np.nan)
# fill outside coverage with a smooth version before estimating the background
fill = np.where(cov > 0.5, f2, 0); w = ndi.gaussian_filter(cov, 30); f2s = ndi.gaussian_filter(fill, 30) / np.maximum(w, 1e-3)
f2 = np.where(cov > 0.5, f2, f2s)
d2 = f2 - bgest(f2)
np.save(O + '/d1.npy', d1.astype(np.float32)); np.save(O + '/d2.npy', d2.astype(np.float32))
def save(a, n, lo, hi):
    Image.fromarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)).resize((1024, 1068)).save(O + '/m_' + n + '.png')
save(d1, 'd1', 0, 1.2); save(d2, 'd2', 0, 2.0)
print('d1 pct', np.percentile(d1, [5, 25, 50, 75, 95]), 'd2', np.percentile(d2[cov > 0.5], [5, 25, 50, 75, 95]))
