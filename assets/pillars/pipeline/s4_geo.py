# Step 4: canonical alpha, column optical depth, per-pixel layer depth (zc) and half-thickness (h).
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from polys import *
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
O = sys.argv[2]
h = np.load(O + '/hst_clean.npy'); H, W = h.shape[:2]
score = np.load(O + '/score.npy')
LYPX = 7.5131 / W            # ly per working pixel (4.53' at 5700 ly over 2048 px)
def ys(y): return (H / 2 - y) * LYPX            # pixel row -> ly (up positive)
def rast(poly):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(poly, fill=1); return np.asarray(im, bool)
def ss(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

# structures, back to front precedence handled by painting order (later wins)
S = {'P1A': rast(P1A) | rast(FING) | rast(LOBE), 'BASE': rast(BASE), 'P1B': rast(P1B), 'P2': rast(P2), 'P3': rast(P3)}
veil = rast(VEIL3)
ORDER = ['P1A', 'BASE', 'P1B', 'P2', 'P3']
lab = np.zeros((H, W), np.int8)          # 0 none, 1.. structures, 6 veil, 7 globule
for i, k in enumerate(ORDER): lab[S[k]] = i + 1
U = lab > 0
din = ndi.distance_transform_edt(U)
near = ndi.distance_transform_edt(~U) < 40
gbox = np.zeros((H, W), bool)
for (x0, y0, x1, y1) in GLOB: gbox[y0:y1, x0:x1] = True
Bf = ((ndi.gaussian_filter(score, 1.0) > 0.9) & (near | gbox)) | (din > 10)
Bf = ndi.binary_opening(Bf, iterations=1)
Bf = ndi.binary_fill_holes(ndi.binary_closing(Bf, iterations=4))
lb, nb = ndi.label(Bf); szb = ndi.sum(Bf, lb, range(1, nb + 1)); kb = np.zeros(nb + 1, bool); kb[1:] = szb > 60; Bf = kb[lb]
rng = np.random.default_rng(7)
warp = sum(ndi.gaussian_filter(rng.standard_normal((H, W)), s) * s * 0.9 for s in (3, 8, 20))   # ragged, multi-scale edge
sd = ndi.distance_transform_edt(Bf) - ndi.distance_transform_edt(~Bf)
a_core = ss(0, 6, sd + warp * (din < 40) * (din > 0))
sc2 = ndi.gaussian_filter(score, 2.0)
a_edge = 0.8 * ss(0.6, 1.2, sc2) * (near | gbox) * ss(0, 1, ndi.gaussian_filter((near | gbox).astype(float), 6) * 2 - 0.2)
a_veil = 0.55 * ss(0.3, 1.3, score) * veil
alpha = np.maximum.reduce([a_core, a_edge, a_veil])
alpha = ndi.gaussian_filter(alpha, 0.7)
alpha[alpha < 0.02] = 0
tau = -np.log(1 - np.minimum(alpha, 0.993))

# label every dusty pixel: polygon label, else veil, else nearest structure within 40 px, else globule
dusty = alpha > 0.02
lab2 = lab.copy()
lab2[(lab2 == 0) & veil & dusty] = 6
dist, (iy, ix) = ndi.distance_transform_edt(lab2 == 0, return_indices=True)
fillable = (lab2 == 0) & dusty
lab2[fillable & (dist < 40)] = lab2[iy, ix][fillable & (dist < 40)]
lab2[(lab2 == 0) & dusty] = 7
lab2[~dusty & (lab2 > 0) & (din == 0)] = 0

# depth centre per structure (ly, +z toward the canonical camera) — after McLeod et al. 2015 §4.2:
# P1a behind NGC 6611 with its tip toward us, P1b / P2 / P3 in front, P3 nearest; P2 tail toward us.
yy = ys(np.arange(H))[:, None] * np.ones((1, W))
ZC = {1: -2.0 + 1.0 * np.clip(yy / 2.9, 0, 1.2),           # P1a: base -2.0 → tip -1.0
      2: 0.6 + 0.0 * yy,                                  # base cloud
      3: 1.0 - 0.8 * np.clip((yy + 2.6) / 2.67, 0, 1),     # P1b: bottom +1.0 → top +0.2
      4: 2.0 - 1.4 * np.clip((yy + 2.54) / 4.38, 0, 1),    # P2: bottom +2.0 → tip +0.6
      5: 1.8 + 0.6 * np.clip((yy + 1.07) / 1.77, 0, 1),    # P3: bottom +1.8 → tip +2.4
      6: 1.6 + 0.0 * yy, 7: 1.4 + 0.0 * yy}
zc = np.zeros((H, W), np.float32)
for k, f in ZC.items(): zc[lab2 == k] = f[lab2 == k]
# smooth depth across label seams (weighted by dust): ~0.45 ly so junctions (P1a/P1b, pillars/base) ramp
# instead of forming ledges when seen off-axis
w = dusty.astype(np.float32)
zc_s = ndi.gaussian_filter(zc * w, 120) / np.maximum(ndi.gaussian_filter(w, 120), 1e-4)
zc = np.where(dusty, zc_s, 0)

# half-thickness from the in-plane distance to the structure's edge: round cross-sections. The cap R and the
# aspect factor vary smoothly (no thickness step where a pillar joins the base); the base cloud thickens
# gradually downward from its top (~row 1640) instead of forming a plinth
hh = np.zeros((H, W), np.float32)
dG = ndi.distance_transform_edt(alpha > 0.3) * LYPX
RG = ndi.gaussian_filter(ndi.maximum_filter(dG, size=301), 25)
rows = np.arange(H)[:, None] * np.ones((1, W))
isbase = (lab2 == 2).astype(np.float32)
capL = np.where(lab2 == 2, 1.05 + 0.75 * np.clip((rows - 1640) / 400.0, 0, 1) ** 1.5, 1.3)
facL = np.where(lab2 == 2, 0.9 + 0.25 * np.clip((rows - 1640) / 400.0, 0, 1), 0.9)
wd = dusty.astype(np.float32); den = np.maximum(ndi.gaussian_filter(wd, 100), 1e-4)
cap = ndi.gaussian_filter(capL * wd, 100) / den
fac = ndi.gaussian_filter(facL * wd, 100) / den
R = np.minimum(RG, cap)
dd = np.minimum(dG, R)
hk = np.sqrt(np.maximum(dd * (2 * R - dd), 0)) * fac
hh = np.where(dusty, np.maximum(hk, 0.025), 0)
hh = np.where(dusty, ndi.gaussian_filter(hh, 1.5), 0)
hh[dusty] = np.maximum(hh[dusty], 0.02)

# ionisation-front mask: thin bright rims (high-pass luminance) on dusty pixels
L = h.mean(-1)
hp = L - ndi.gaussian_filter(L, 6)
front = np.clip(hp / 0.10, 0, 1) * np.clip(alpha * 3, 0, 1)
front = np.maximum(front, np.clip((L - ndi.gaussian_filter(L, 40) - 0.05) / 0.25, 0, 1) * np.clip(alpha * 3, 0, 1))
# empty-space distance (ly) to the nearest dusty pixel
dout = ndi.distance_transform_edt(~dusty) * LYPX

np.savez_compressed(O + '/geo.npz', alpha=alpha.astype(np.float32), tau=tau.astype(np.float32), lab=lab2, zc=zc.astype(np.float32),
                    hh=hh.astype(np.float32), front=front.astype(np.float32), dout=dout.astype(np.float32))
def save(a, n, lo, hi):
    Image.fromarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)).resize((1024, 1068)).save(O + '/geo_' + n + '.png')
save(alpha, 'alpha', 0, 1); save(hh, 'h', 0, 1.6); save(np.where(dusty, zc, -4), 'zc', -2.5, 2.5); save(front, 'front', 0, 1); save(lab2, 'lab', 0, 7)
print('h max', hh.max(), 'zc range', zc[dusty].min(), zc[dusty].max(), 'alpha mean', alpha.mean())
