import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
O = sys.argv[1]
h = np.load(O + '/hst_clean.npy'); H, W = h.shape[:2]
d1 = np.load(O + '/d1.npy'); d2 = np.load(O + '/d2.npy'); cov = np.load(O + '/jwcov_ext1.0.npy')
L = ndi.gaussian_filter(h.mean(-1), 1.5)
small = ndi.zoom(L, 0.25, order=1)
bgL = ndi.gaussian_filter(ndi.percentile_filter(small, 60, size=75), 10)
yy, xx = np.mgrid[0:H, 0:W]
bgL = ndi.map_coordinates(bgL, [yy * bgL.shape[0] / H, xx * bgL.shape[1] / W], order=1, mode='nearest')
dark = bgL - L
d2s = ndi.gaussian_filter(d2, 1.5) * (cov > 0.5)
score = np.maximum.reduce([ndi.gaussian_filter(d1, 1.5) / float(sys.argv[2]), d2s / float(sys.argv[3]), dark / float(sys.argv[4])])
np.save(O + '/score.npy', score.astype(np.float32))
B = score > 1
B = ndi.binary_opening(B, iterations=2)
B = ndi.binary_closing(B, structure=np.ones((3, 3)), iterations=6)
B = ndi.binary_fill_holes(B)
lab, n = ndi.label(B); sz = ndi.sum(B, lab, range(1, n + 1))
keep = np.zeros(n + 1, bool); keep[1:] = sz > 150
B = keep[lab]
np.save(O + '/maskB.npy', B)
v = (h * 255).astype(np.uint8).copy()
edge = B ^ ndi.binary_erosion(B)
v[ndi.binary_dilation(edge)] = (0, 255, 0)
Image.fromarray(v).resize((1024, 1068)).save(O + '/maskB.jpg', quality=90)
Image.fromarray((np.clip(score / 2, 0, 1) * 255).astype(np.uint8)).resize((1024, 1068)).save(O + '/score.png')
print(n, B.mean())
