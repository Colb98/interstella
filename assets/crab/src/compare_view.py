# side-by-side: snapshot vs reference warped into the same Earth-view framing (D0=36, fov 19)
import sys, os, math, numpy as np
REF = os.environ.get('CRAB_REF', './ref/')
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
snap, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
fov = float(sys.argv[4]) if len(sys.argv) > 4 else 19.0
D0 = 36.0
LY = 6500*math.pi/648000
pxas = 720 / (2*D0*math.tan(math.radians(fov/2))) * LY      # screen px per arcsec
REFS = {'hst': (REF + 'large_heic0515a.jpg', 6.41*60/3864, (2014.5,1958.8), None),
        'jwst': (REF + 'large_weic2326a.jpg', None, None, 'jw'),
        'xray': (REF + 'cxc_2008_crab_crab_lg.jpg', 120/244*792/3300, (1670,1732), None)}
f, sc, psr, kind = REFS[ref]
im = Image.open(f).convert('RGB')
if kind == 'jw':
    z, t = 0.3140059868439497-0.011703080243992112j, 219.6944231076841+503.7398224477379j
    hsc = 6.41*60/3864; hp = complex(2014.5,1958.8)
    jp = (hp - t)/z; sc = hsc*abs(z); psr = (jp.real, jp.imag)
s = 1/(pxas*sc)   # src px per screen px
a, b, c = s, 0, psr[0] - 640*s
d, e, g = 0, s, psr[1] - 360*s
r = im.transform((1280,720), Image.AFFINE, (a,b,c,d,e,g), Image.BICUBIC)
sn = Image.open(snap).convert('RGB')
W = Image.new('RGB', (1280, 1440)); W.paste(sn, (0,0)); W.paste(r, (0,720))
W.resize((960,1080)).save(out)
