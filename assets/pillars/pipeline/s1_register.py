# Step 1: downsample Hubble heic1501a to the working grid and resample JWST weic2216b onto it via the published WCS.
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
REF, OUT = sys.argv[1], sys.argv[2]
W = 2048
hst = Image.open(REF + '/heic1501a.jpg'); HW, HH = hst.size
H = round(W * HH / HW)
sH = (4.53 * 60 / HW + 4.72 * 60 / HH) / 2       # arcsec / px (native)
print('hubble', HW, HH, 'scale', sH, 'working', W, H, 'scale', sH * HW / W)
hs = np.asarray(hst.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255
np.save(OUT + '/hst.npy', hs)

def NE(theta):
    t = np.radians(theta)
    return np.array([-np.sin(t), np.cos(t)]), np.array([-np.cos(t), -np.sin(t)])
Nh, Eh = NE(34.98); Nj, Ej = NE(-2.5646)
jw = Image.open(REF + '/weic2216b.jpg'); JW, JH = jw.size
sJ = (4.21 * 60 / JW + 7.30 * 60 / JH) / 2
F = 4
jws = np.asarray(jw.resize((JW // F, JH // F), Image.LANCZOS), dtype=np.float32) / 255
sJF = sJ * JW / (JW // F)
dE = 2.46 * 15 * np.cos(np.radians(13.85)); dN = -(51 * 60 + 6.26 - (50 * 60 + 21.97))
print('jwst offset E,N arcsec', dE, dN, 'scale', sJ)
# star-matched refinement of the published WCS (applied afterwards as a shift of (-20.6, -21.6) px)
dx = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
dy = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
ext = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
EWd, EHd = int(round(W * ext)), int(round(H * ext))
sw = sH * HW / W
yy, xx = np.mgrid[0:EHd, 0:EWd].astype(np.float64)
right = (xx + 0.5 - EWd / 2) * sw + dx; up = -(yy + 0.5 - EHd / 2) * sw + dy
vN = right * Nh[0] + up * Nh[1]; vE = right * Eh[0] + up * Eh[1]
vN2 = vN - dN; vE2 = vE - dE
r2 = vN2 * Nj[0] + vE2 * Ej[0]; u2 = vN2 * Nj[1] + vE2 * Ej[1]
cj = r2 / sJF + jws.shape[1] / 2 - 0.5; rj = -u2 / sJF + jws.shape[0] / 2 - 0.5
cov = ((cj >= 0) & (cj <= jws.shape[1] - 1) & (rj >= 0) & (rj <= jws.shape[0] - 1)).astype(np.float32)
out = np.stack([ndi.map_coordinates(jws[..., c], [rj, cj], order=3, mode='constant') for c in range(3)], -1)
out = np.clip(out, 0, 1)
np.save(OUT + f'/jw_ext{ext}.npy', out.astype(np.float32)); np.save(OUT + f'/jwcov_ext{ext}.npy', cov)
Image.fromarray((out * 255).astype(np.uint8)).resize((EWd // 2, EHd // 2)).save(OUT + f'/jw_reg_ext{ext}.jpg', quality=88)
print('coverage', cov.mean())
