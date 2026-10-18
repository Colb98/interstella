# Step 2b: refine the JWST registration. The published WCS centres leave NIRCam (20.6, 21.6) working px
# (2.73", 2.87") off; measured by matching the brightest stars common to both catalogues
# (median offset, 79 pairs, scatter 1.3 px). Shifts every JWST product of steps 1-2 in place.
import sys, numpy as np
from scipy import ndimage as ndi
O = sys.argv[1] + '/'; DX, DY = 20.6, 21.6
for f in ['jw_ext1.0.npy', 'jw_clean.npy']:
    a = np.load(O + f); a = np.stack([ndi.shift(a[..., c], (-DY, -DX), order=3, mode='nearest') for c in range(3)], -1)
    np.save(O + f, a.clip(0, 1).astype(np.float32))
for f in ['jwcov_ext1.0.npy', 'jw_starmask.npy']:
    np.save(O + f, ndi.shift(np.load(O + f), (-DY, -DX), order=1, mode='constant', cval=0).astype(np.float32))
for f in ['cat_jw_all.npy', 'catfinal_jw.npy', 'cat_jw.npy', 'cores_jw.npy']:
    a = np.load(O + f); a[:, 0] -= DX; a[:, 1] -= DY; np.save(O + f, a)
