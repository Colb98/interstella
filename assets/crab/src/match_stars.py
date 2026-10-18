import sys, numpy as np, itertools
A=np.load(sys.argv[1])[:,:2]  # source (to map)
B=np.load(sys.argv[2])[:,:2]  # target (HST)
smin,smax=float(sys.argv[3]),float(sys.argv[4])
n=35; A=A[:n]; B=B[:n]
Ac=A[:,0]+1j*A[:,1]; Bc=B[:,0]+1j*B[:,1]
best=(0,None)
pa=[(i,j) for i in range(n) for j in range(n) if i!=j]
for (i,j) in pa:
    da=Ac[j]-Ac[i]
    if abs(da)<200: continue
    # all target pairs
    k,l=np.meshgrid(np.arange(n),np.arange(n),indexing='ij'); k=k.ravel(); l=l.ravel(); m=k!=l; k=k[m]; l=l[m]
    z=(Bc[l]-Bc[k])/da
    s=np.abs(z); ang=np.angle(z,deg=True)
    ok=(s>smin)&(s<smax)&(np.abs(ang)<40)
    for kk,ll,zz in zip(k[ok],l[ok],z[ok]):
        t=Bc[kk]-zz*Ac[i]
        P=zz*Ac+t
        d=np.abs(P[:,None]-Bc[None,:]).min(1)
        c=(d<8).sum()
        if c>best[0]: best=(c,(zz,t))
c,(z,t)=best
# refine least squares on inliers
P=z*Ac+t; d=np.abs(P[:,None]-Bc[None,:]); j=d.argmin(1); m=d.min(1)<8
X=np.column_stack([Ac[m],np.ones(m.sum())]); sol=np.linalg.lstsq(X,Bc[j[m]],rcond=None)[0]
print('inliers',c, 'scale',abs(sol[0]),'rot',np.angle(sol[0],deg=True),'t',sol[1])
res=np.abs(X@sol-Bc[j[m]]); print('rms',np.sqrt((res**2).mean()))
np.save(sys.argv[5], np.array([sol[0],sol[1]]))
