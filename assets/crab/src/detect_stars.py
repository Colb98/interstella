import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS=None
def load_gray(f, ds):
    im=Image.open(f).convert('RGB')
    W,H=im.size
    im=im.resize((W//ds,H//ds), Image.LANCZOS)
    a=np.asarray(im).astype(np.float32)/255
    return a, (W,H)
def detect(a, n=80, r=3):
    g=a.mean(2)
    bg=ndi.median_filter(g, size=15)
    th=g-bg
    mx=ndi.maximum_filter(th, size=2*r+1)
    pk=(th==mx)&(th>0.12)
    ys,xs=np.nonzero(pk)
    v=th[ys,xs]
    # compactness: value vs ring at radius 4
    o=np.argsort(-v)
    pts=[]
    for i in o:
        y,x=ys[i],xs[i]
        if y<6 or x<6 or y>=g.shape[0]-6 or x>=g.shape[1]-6: continue
        ring=np.concatenate([th[y-5,x-5:x+6],th[y+5,x-5:x+6],th[y-5:y+6,x-5],th[y-5:y+6,x+5]]).mean()
        if ring>0.3*v[i]: continue
        pts.append((x,y,v[i]))
        if len(pts)>=n: break
    return np.array(pts)
if __name__=='__main__':
    f,ds=sys.argv[1],int(sys.argv[2])
    a,sz=load_gray(f,ds)
    p=detect(a)
    np.save(sys.argv[3], np.column_stack([p[:,0]*ds,p[:,1]*ds,p[:,2]]))
    print(sz, len(p)); print(np.round(p[:15]*[ds,ds,1],2))
