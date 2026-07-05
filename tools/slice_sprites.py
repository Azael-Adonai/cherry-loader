import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "source", "CherryCycle.png")
OUT = ROOT
SPR = os.path.join(OUT, "sprites")
os.makedirs(SPR, exist_ok=True)

SCALE = 0.5      # output sprite resolution vs source
OBJ_A = 150      # alpha >= this is solid object (stems dip to ~165)
SOFT_REACH = 60  # max px a soft shadow pixel may sit from its object

img = np.asarray(Image.open(SRC)).copy()
H, W = img.shape[:2]
alpha = img[:, :, 3]

hist = [((alpha >= a) & (alpha < b)).sum() for a, b in [(1, 40), (40, 100), (100, 150), (150, 256)]]
print("alpha buckets 1-40/40-100/100-150/150+:", hist)

obj = alpha >= OBJ_A
blob, nb = ndi.label(obj, structure=np.ones((3, 3)))
print("blobs:", nb)

# per-blob stats
areas = ndi.sum(np.ones_like(blob), blob, index=np.arange(1, nb + 1))
dt_full = ndi.distance_transform_edt(obj)
dtmax = ndi.maximum(dt_full, blob, index=np.arange(1, nb + 1))

# --- reattach skinny fragments (severed stem tips) & drop dust ---
keep = np.ones(nb + 1, bool); keep[0] = False
frag_ids = [i + 1 for i in range(nb) if dtmax[i] < 13 and areas[i] < 3000]
solid_ids = [i + 1 for i in range(nb) if i + 1 not in frag_ids]
solid_mask = np.isin(blob, solid_ids)
dist_s, (iy, ix) = ndi.distance_transform_edt(~solid_mask, return_indices=True)
relabel = np.arange(nb + 1)
for fid in frag_ids:
    m = blob == fid
    dmin = dist_s[m].min()
    if dmin <= 80:
        # merge into the nearest solid blob
        yx = np.argmin(np.where(m, dist_s, np.inf).ravel())
        y, x = np.unravel_index(yx, m.shape)
        relabel[fid] = blob[iy[y, x], ix[y, x]]
        print(f"  fragment {fid} (area {int(areas[fid-1])}) -> blob {relabel[fid]} (gap {dmin:.0f}px)")
    elif areas[fid - 1] < 600:
        relabel[fid] = 0
        print(f"  fragment {fid} dropped (area {int(areas[fid-1])}, gap {dmin:.0f}px)")
for i in range(1, nb + 1):
    if areas[i - 1] < 600 and relabel[i] == i and i not in frag_ids:
        relabel[i] = 0
blob = relabel[blob]

# --- split blobs holding several fruits (touching cherries) ---
ids = sorted(set(blob.ravel()) - {0})
next_id = max(ids) + 1
for bid in list(ids):
    sl = ndi.find_objects((blob == bid).astype(np.int8))[0]
    bm = blob[sl] == bid
    dt = dt_full[sl] * bm
    m_dt = dt.max()
    if m_dt < 30:
        continue  # thin structures (spent flowers) never split
    marks, nm = ndi.label(dt >= 0.7 * m_dt, structure=np.ones((3, 3)))
    if nm < 2:
        continue
    # merge marker regions closer than 1.1 * dtmax (lumpy single fruit)
    cents = np.array(ndi.center_of_mass(np.ones_like(marks), marks, np.arange(1, nm + 1)))
    groups = list(range(nm))
    for a in range(nm):
        for b in range(a + 1, nm):
            if np.hypot(*(cents[a] - cents[b])) < 1.1 * m_dt:
                gb = groups[b]
                groups = [groups[a] if g == gb else g for g in groups]
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        continue
    print(f"  splitting blob {bid} into {len(uniq)} fruits")
    seeds = np.zeros_like(marks)
    for gi, g in enumerate(uniq):
        for mi in range(nm):
            if groups[mi] == g:
                seeds[marks == mi + 1] = gi + 1
    # geodesic label propagation inside the blob
    lab = seeds.copy()
    st = np.ones((3, 3))
    for _ in range(4000):
        grown = ndi.grey_dilation(lab, footprint=st)
        upd = bm & (lab == 0) & (grown > 0)
        if not upd.any():
            break
        lab[upd] = grown[upd]
    out_ids = [bid] + [next_id + i for i in range(len(uniq) - 1)]
    next_id += len(uniq) - 1
    sub = blob[sl]
    for gi in range(len(uniq)):
        sub[(lab == gi + 1) & bm] = out_ids[gi]
    blob[sl] = sub

# compact relabel
ids = sorted(set(blob.ravel()) - {0})
lab = np.zeros_like(blob)
for i, old in enumerate(ids):
    lab[blob == old] = i + 1
n_items = len(ids)
print("items:", n_items)

# --- assign soft pixels (shadows, feathered edges) to nearest item ---
dist_l, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
nearest = lab[iy, ix]
soft = (alpha > 0) & (lab == 0) & (dist_l <= SOFT_REACH)
assign = lab + np.where(soft, nearest, 0)

PAD = 30           # room for the synthetic ambient shadow
SH_SIGMA = 11      # blur of the ambient shadow (full-res px)
SH_ALPHA = 0.34
SH_RGB = (55, 30, 22)

items = []
for k in range(1, n_items + 1):
    m = assign == k
    mf = ndi.gaussian_filter(m.astype(np.float32), 1.5)
    a_k = (alpha.astype(np.float32) * np.clip(mf, 0, 1)).astype(np.uint8)
    ys, xs = np.where((lab == k))
    y0 = max(0, ys.min() - PAD); y1 = min(H, ys.max() + 1 + PAD)
    x0 = max(0, xs.min() - PAD); x1 = min(W, xs.max() + 1 + PAD)

    crop = img[y0:y1, x0:x1].copy().astype(np.float32)
    core_k = (lab == k)[y0:y1, x0:x1]

    # strip the photo's baked shadow: semi-transparent pixels far from the
    # solid object are shadow, not fruit edge — they were cut apart between
    # neighbours and no longer line up after the ring snap
    d_out = ndi.distance_transform_edt(~core_k)
    keep = np.where(d_out <= 5, 1.0, np.exp(-(d_out - 5) / 3.0))
    a2 = a_k[y0:y1, x0:x1].astype(np.float32) * keep

    # bake a fresh ambient shadow from this sprite's own silhouette: soft
    # halos overlap gracefully and never show a cut edge
    sh_a = np.clip(ndi.gaussian_filter(core_k.astype(np.float32), SH_SIGMA)
                   * 1.6, 0, 1) * SH_ALPHA * 255.0

    out_a = a2 + sh_a * (1 - a2 / 255.0)
    safe = np.maximum(out_a, 1e-3)
    for c in range(3):
        crop[:, :, c] = (crop[:, :, c] * a2 +
                         SH_RGB[c] * sh_a * (1 - a2 / 255.0)) / safe
    crop[:, :, 3] = out_a
    crop = np.clip(crop, 0, 255).astype(np.uint8)

    dt = ndi.distance_transform_edt(core_k)
    py, px = np.unravel_index(np.argmax(dt), dt.shape)
    fruit_r = float(dt.max())

    cys, cxs = np.where(core_k)
    d2 = (cys - py) ** 2 + (cxs - px) ** 2
    j = int(np.argmax(d2))
    sy, sx = int(cys[j]), int(cxs[j])
    stem_len = float(np.sqrt(d2[j]))

    w, h = x1 - x0, y1 - y0
    sw, sh = max(1, round(w * SCALE)), max(1, round(h * SCALE))
    spr = Image.fromarray(crop).resize((sw, sh), Image.LANCZOS)
    fn = f"item_{k:02d}.webp"
    spr.save(os.path.join(SPR, fn), "WEBP", quality=92, method=6)

    items.append(dict(
        file=fn,
        x=x0 * SCALE, y=y0 * SCALE, w=w * SCALE, h=h * SCALE,
        px=px * SCALE, py=py * SCALE,
        gx=(x0 + px) * SCALE, gy=(y0 + py) * SCALE,
        stemLen=stem_len * SCALE, fruitR=fruit_r * SCALE,
        stemDx=(sx - px) * SCALE, stemDy=(sy - py) * SCALE,
        area=float(core_k.sum()) * SCALE * SCALE,
    ))

# least-squares (Kasa) circle fit through the pivots: balanced rotation axis
P = np.array([[it["gx"], it["gy"]] for it in items])
A = np.stack([P[:, 0], P[:, 1], np.ones(len(P))], axis=1)
b = (P ** 2).sum(axis=1)
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
cx, cy = float(sol[0] / 2), float(sol[1] / 2)

# true inner clearance: nearest solid pixel to the centre (stems reach inward)
ys_all, xs_all = np.where(lab > 0)
inner_r = float(np.hypot(xs_all * SCALE - cx, ys_all * SCALE - cy).min())

for it in items:
    dx, dy = it["gx"] - cx, it["gy"] - cy
    it["rad"] = float(np.hypot(dx, dy))
    it["ang"] = float((np.degrees(np.arctan2(dx, -dy))) % 360)  # 0 = top, cw+
items.sort(key=lambda it: it["ang"])

# --- snap to a perfect ring: uniform radius, even angular grid ---
# each item keeps its orientation relative to the ring via a constant
# rotation delta applied around its own pivot
n_it = len(items)
step = 360.0 / n_it
rads = np.array([it["rad"] for it in items])
print("pre-snap radii:", np.round(np.sort(rads), 0).tolist())
# uniform radius biased toward the outer (large-fruit) placements so the
# big cherries keep the photo's gentle overlap instead of bunching up
R = float(np.percentile(rads, 78))
a0 = float(np.mean([it["ang"] - i * step for i, it in enumerate(items)]))
max_out = max(it["rad"] - R for it in items)
for i, it in enumerate(items):
    target = (a0 + i * step) % 360
    delta = target - it["ang"]
    if delta > 180: delta -= 360
    if delta < -180: delta += 360
    t = np.radians(target)
    gx2 = cx + R * np.sin(t)
    gy2 = cy - R * np.cos(t)
    it["x"] += gx2 - it["gx"]
    it["y"] += gy2 - it["gy"]
    it["gx"], it["gy"] = float(gx2), float(gy2)
    it["ang"], it["rad"] = float(target), R
    it["rot"] = float(delta)
# items that moved inward shrink the clear centre; keep the arc safe
inner_r = inner_r - max(0.0, max_out) - 15.0
print(f"ring snap: R={R:.1f} grid={step:.1f} deg  max delta="
      f"{max(abs(it['rot']) for it in items):.1f} deg  innerR={inner_r:.1f}")

meta = dict(W=W * SCALE, H=H * SCALE, cx=cx, cy=cy, innerR=inner_r, items=items)
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)

dbg = Image.new("RGB", (round(W * SCALE), round(H * SCALE)), (250, 247, 243))
for it in items:
    spr = Image.open(os.path.join(SPR, it["file"])).convert("RGBA")
    spr = spr.rotate(-it["rot"], resample=Image.BICUBIC, expand=False,
                     center=(it["px"], it["py"]))
    dbg.paste(spr, (round(it["x"]), round(it["y"])), spr)
dr = ImageDraw.Draw(dbg)
for i, it in enumerate(items):
    x, y, w, h = it["x"], it["y"], it["w"], it["h"]
    dr.rectangle([x, y, x + w, y + h], outline=(70, 160, 255), width=2)
    gx, gy = it["gx"], it["gy"]
    dr.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(220, 30, 60))
    dr.line([gx, gy, gx + it["stemDx"], gy + it["stemDy"]], fill=(30, 180, 90), width=2)
    dr.text((x + 3, y + 2), str(i), fill=(20, 20, 200))
dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 140, 0))
dbg.save(os.path.join(OUT, "debug_slice.png"))

total = sum(os.path.getsize(os.path.join(SPR, it["file"])) for it in items)
print(f"{len(items)} sprites, {total // 1024} KB total")
