import sys, os, numpy as np
H, W = 2136, 2048
sys.path.insert(0, sys.argv[1])
from stars import *
from PIL import Image, ImageDraw
O = sys.argv[2]
SP = {'hst': [46, 134, 224, 316], 'jw': [37, 97, 157, 217, 277, 337, 127, 307]}
def hst_colour(fl, pk):
    s = fl.sum() + 1e-9
    return fl[2] / s > 0.185 and fl[1] / s < 0.37
cat_cache = '--cache' in sys.argv
which = [a for a in sys.argv[3:] if not a.startswith('--')] or ['hst', 'jw']
for name, f, passes, ct, t in [
        ('hst', 'hst.npy', [(4.0, 101, 34, 6, 61, 0.3), (3.0, 61, 20, 6, 41, 0.3), (1.0, 31, 7, 4, 15, 0.2), (0, 7, 3, 4, 5, 0.08)], hst_colour, 0.035),
        ('jw', 'jw_ext1.0.npy', [(4.0, 101, 34, 10, 61, 0.3), (3.0, 61, 20, 10, 41, 0.3), (1.0, 31, 7, 8, 15, 0.22), (0, 7, 3, 3, 5, 0.07)], None, 0.03)]:
    if name not in which: continue
    img = np.load(O + '/' + f)
    import os
    cat = np.load(O + f'/cat_{name}.npy') if os.path.exists(O + f'/cat_{name}.npy') and cat_cache else detect(img, passes, SP[name], ct)
    print(name, [int((cat[:, 6] == l).sum()) for l in range(4)])
    REJECT = {'hst': [(1110, 1713), (1346, 1028), (634, 371), (1061, 819), (747, 311), (447, 946), (1821, 1996)], 'jw': []}[name]
    for (mx, my) in REJECT: cat = cat[np.hypot(cat[:, 0] - mx, cat[:, 1] - my) > 4]
    # de-duplicate (two passes can find the same star): keep the brighter
    f_ = cat[:, 3:6].sum(1); order = np.argsort(-f_); keep = []
    for i in order:
        if all(np.hypot(cat[i, 0] - cat[j, 0], cat[i, 1] - cat[j, 1]) > 3 for j in keep[-400:]): keep.append(i)
    cat = cat[np.sort(keep)]
    if name == 'jw':
        base = np.load(O + '/cat_jw_all.npy') if os.path.exists(O + '/cat_jw_all.npy') else cat
        cat = base
    MANUAL = {'hst': [(842, 792), (540, 201), (1350, 255)],
              'jw': [(1080, 808), (742, 855), (1410, 1435), (1325, 1033), (1215, 1063), (703, 622), (1337, 1462), (1387, 1472), (1355, 1425)]}[name]
    for (mx, my) in MANUAL:
        if len(cat): cat = cat[np.hypot(cat[:, 0] - mx, cat[:, 1] - my) > 4]
        loc = img[my - 6:my + 7, mx - 6:mx + 7]; b = np.median(np.concatenate([loc[0], loc[-1], loc[:, 0], loc[:, -1]]), 0)
        fl = np.clip(img[my - 3:my + 4, mx - 3:mx + 4] - b, 0, None).sum((0, 1)) * 4 + 10
        cat = np.concatenate([cat, [[mx, my, 0.5, fl[0], fl[1], fl[2], -1]]])     # level -1 = manual
    if name == 'jw': np.save(O + '/cat_jw_all.npy', cat[cat[:, 6] >= 0])
    if name == 'hst':
        # knots on the ionisation fronts mimic faint stars: drop detections sitting on strong edges
        Lmed = ndi.gaussian_filter(ndi.median_filter(img.mean(-1), size=9), 2.0)
        gmag = np.hypot(ndi.sobel(Lmed, 0), ndi.sobel(Lmed, 1)) / 8
        gmax = ndi.maximum_filter(gmag, size=7)
        gi = gmax[np.clip(cat[:, 1].round().astype(int), 0, H - 1), np.clip(cat[:, 0].round().astype(int), 0, W - 1)]
        edge = (gi > 0.018) & (cat[:, 6] >= 1)
        print('hst: dropped', edge.sum(), 'detections on ionisation fronts / edges')
        cat = cat[~edge]
    al = np.load(O + '/geo.npz')['alpha']
    ai = al[np.clip(cat[:, 1].astype(int), 0, al.shape[0] - 1), np.clip(cat[:, 0].astype(int), 0, al.shape[1] - 1)]
    fsum = cat[:, 3:6].sum(1)
    if name == 'jw':
        ssc = np.zeros(len(cat)); b01 = cat[:, 6] <= 1
        ssc[b01] = spike_score(img, cat[b01], SP[name])
        spiky = (cat[:, 6] < 0) | (b01 & ((ssc >= 0.05) | (ai < 0.2)))
        red = cat[:, 3] / (cat[:, 4] + cat[:, 5] + 1e-6)
        emb = (cat[:, 6] >= 2) & (ai > 0.4) & (red > 1.6) & (red < 3.5) & (fsum > 6.0)
        removed = spiky | ((cat[:, 6] == 2) & (ai < 0.2)) | emb
        halo_ok = spiky
    else:
        removed = np.ones(len(cat), bool); emb = np.zeros(len(cat), bool); halo_ok = removed
    print(name, 'catalogue', len(cat), 'removed', removed.sum(), 'embedded', emb.sum())
    full = np.concatenate([cat, removed[:, None], emb[:, None]], 1)
    np.save(O + f'/catfinal_{name}.npy', full)
    sub, cores = subtract_halos(img, np.where(halo_ok[:, None], cat, np.concatenate([cat[:, :3], np.zeros((len(cat), 3)), cat[:, 6:]], 1)), rmax=26 if name == 'jw' else 110)
    np.save(O + f'/cores_{name}.npy', cores)
    cr = cat[removed]
    m = removal_mask(sub, cr, SP[name], t=t, spk=9.0, rk=0.7 if name == 'jw' else 1.0, dil=(name != 'jw'))
    if len(cores):
        disk = spike_region(m.shape, np.stack([cores[:, 0], cores[:, 1], cores[:, 2] * 1.2 + 2, np.zeros(len(cores))], 1), SP[name])
        m = np.maximum(m, np.clip(ndi.gaussian_filter(disk, 1.0) * 1.3, 0, 1))
    img = sub
    np.save(O + f'/{name}_starmask.npy', m.astype(np.float32))
    clean = add_grain(inpaint(img, m), img, m).clip(0, 1)
    np.save(O + f'/{name}_clean.npy', clean.astype(np.float32))
    Image.fromarray((clean * 255).astype(np.uint8)).save(O + f'/{name}_clean.jpg', quality=92)
    Image.fromarray((clean * 255).astype(np.uint8)).crop((0, 0, 1024, 1024)).save(O + f'/{name}_clean_tl.jpg', quality=90)
    Image.fromarray((clean * 255).astype(np.uint8)).crop((900, 900, 1600, 1600)).save(O + f'/{name}_clean_mid.jpg', quality=90)
    im = Image.fromarray((img * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    for x, y, pk, fr, fg, fb, lvl in cat:
        r = 2 + np.sqrt(fr + fg + fb)
        d.ellipse([x - r, y - r, x + r, y + r], outline=[(255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 0)][int(lvl)])
    im.crop((0, 0, 1024, 1024)).save(O + f'/det_{name}_tl.jpg', quality=88)
