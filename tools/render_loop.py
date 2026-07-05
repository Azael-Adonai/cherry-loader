"""Render the cherry loader's steady-state master loop to a seamless video.

Replicates the exact motion math of index.html (same prand hash, same spring
physics, same periodic forcing), pre-settles the springs, then renders one
full LOOP_T-second cycle. Frame 0 and frame N are identical by construction.

Outputs (in exports/):
  cherry_loop.webp      - transparent animated WebP, 16 s @ 25 fps, loops forever
  cherry_loop.mp4       - opaque H.264 on the warm background (if ffmpeg exists)
  cherry_loop_alpha.mov - ProRes 4444 with alpha for editing (if ffmpeg exists)
"""
import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "exports")
os.makedirs(D, exist_ok=True)
meta = json.load(open(os.path.join(ROOT, "meta.json")))

LOOP_T = 16.0
FPS = 25
NF = int(LOOP_T * FPS)          # 400 frames
H_STEP = 1.0 / 100              # integrator step; 4 steps per frame
TAU = math.pi * 2

OMEGA = 360.0 / LOOP_T
SURGES = [(4.0, LOOP_T / 6, 0.0), (2.2, LOOP_T / 2, 2.1)]
WAVE_SPEED = OMEGA + 360.0 / (2 * SURGES[0][1])
WAVE_SIGMA = 26.0
ENTRY_DUR, EXIT_DUR = 0.55, 0.5


def exist_at(n, tau):
    e_in = tau - n["delay"]
    e_out = tau - n["exitAt"]
    if e_in < 0 or e_out >= EXIT_DUR:
        return 0.0
    if e_in < ENTRY_DUR:
        return e_in / ENTRY_DUR
    if e_out >= 0:
        return 1 - e_out / EXIT_DUR
    return 1.0
OUT_SIZE = 640
SQ = 1700                        # square working canvas, ring centred
CROP = 1450                      # central crop before downscale (framing)
BG = (247, 242, 234)


def prand(i, salt):
    x = (i * 374761393 + salt * 668265263 + 1013904223) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 1274126177) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x / 4294967296.0


def theta(t):
    return OMEGA * t + sum(A * math.sin(TAU * t / T + p) for A, T, p in SURGES)


def theta_acc(t):
    return sum(-A * (TAU / T) ** 2 * math.sin(TAU * t / T + p) for A, T, p in SURGES)


def wrap180(a):
    return ((a + 180) % 360 + 360) % 360 - 180


cx, cy = meta["cx"], meta["cy"]
items = meta["items"]

# per-item params, identical to index.html
nodes = []
for idx, it in enumerate(items):
    L = max(it["stemLen"], 30)
    spr = Image.open(os.path.join(ROOT, "sprites", it["file"])).convert("RGBA")
    # pad the canvas so rotation never clips (CSS transforms don't clip;
    # PIL's affine does, and long stems sweep far outside their bbox)
    pad = int(0.45 * max(spr.size)) + 24
    big = Image.new("RGBA", (spr.size[0] + 2 * pad, spr.size[1] + 2 * pad),
                    (0, 0, 0, 0))
    big.paste(spr, (pad, pad))
    nodes.append(dict(
        idx=idx, it=it,
        sprite=big, pad=pad,
        ang=it["ang"], rot=it.get("rot", 0.0),
        f0=min(2.6, max(0.9, 1.9 * math.sqrt(80 / L))),
        zeta=0.10 + prand(idx, 1) * 0.08,
        gain=13 * min(1.6, max(0.85, L / 140)) * (0.85 + 0.3 * prand(idx, 2)),
        breezeA=0.7 + prand(idx, 3) * 0.5,
        breezeF=(4 + math.floor(prand(idx, 4) * 7)) / LOOP_T,
        breezeP=prand(idx, 5) * TAU,
        gust=105 * (0.7 + 0.6 * prand(idx, 8)),
        isFlower=it["stemLen"] < 1.9 * it["fruitR"],
        m=max(0.15, it["area"] / 9000),
        wHalf=it["fruitR"] * 1.2,
        radPx=it["rad"],
        fu=TAU * (0.75 + prand(idx, 9) * 0.6),
        gustT=175 * (0.7 + 0.6 * prand(idx, 10)),
        kick=(-1 if prand(idx, 6) < 0.5 else 1) * (35 + prand(idx, 7) * 30),
        tanX=math.cos(math.radians(it["ang"])),
        tanY=math.sin(math.radians(it["ang"])),
        ux=math.sin(math.radians(it["ang"])),
        uy=-math.cos(math.radians(it["ang"])),
        u=0.0, uV=0.0,
        spin=0.0, sway=0.0, swayV=0.0,
        exist=0.0, cyc=-1,
    ))
# spinner parity follows the JS creation order (sorted by pivot y)
spinner_count = 0
for n in sorted(nodes, key=lambda n: n["it"]["gy"]):
    if n["isFlower"]:
        n["spin"] = (1 if spinner_count % 2 == 0 else -1) * 360.0 / LOOP_T
        spinner_count += 1

# cycle choreography, identical to index.html: reveal order starts at the
# bud just before the blossoms, exits leave in the same clockwise order
spin_idx = [i for i, n in enumerate(nodes) if n["spin"]]
start_idx = (min(spin_idx) - 1 + len(nodes)) % len(nodes) if spin_idx else 0
for i, n in enumerate(nodes):
    k = (i - start_idx + len(nodes)) % len(nodes)
    n["delay"] = 0.3 + 0.5 * k if k < 3 else 1.55 + 0.088 * (k - 3)
    n["exitAt"] = 13.2 + 0.075 * k

draw_order = sorted(range(len(nodes)), key=lambda i: nodes[i]["it"]["gy"])

# ring-adjacent contact pairs (identical to index.html)
RESTITUTION = 0.35
pairs = []
for i in range(len(nodes)):
    a, b = nodes[i], nodes[(i + 1) % len(nodes)]
    arc = a["radPx"] * (2 * math.pi / len(nodes))
    pairs.append((a, b, arc))


def collide():
    for a, b, arc in pairs:
        slack = max(0.0, arc - a["wHalf"] * a["exist"] - b["wHalf"] * b["exist"])
        pen = -(slack + b["u"] - a["u"])
        if pen <= 0:
            continue
        ms = a["m"] + b["m"]
        a["u"] -= pen * b["m"] / ms
        b["u"] += pen * a["m"] / ms
        relV = b["uV"] - a["uV"]
        if relV < 0:
            hard = relV < -4
            J = -(1 + (RESTITUTION if hard else 0)) * relV * a["m"] * b["m"] / ms
            a["uV"] -= J / a["m"]
            b["uV"] += J / b["m"]
            if hard:
                a["swayV"] += (J / a["m"]) * 1.4
                b["swayV"] -= (J / b["m"]) * 1.4

# ---- integrate physics: settle for 3 loops, then record one loop ----
SETTLE = 3
states = []                      # per frame: list of (sway, u) per item
state0 = None
state16 = None
steps_total = int(round((LOOP_T * (SETTLE + 1)) / H_STEP))
for step in range(steps_total + 1):
    t = -SETTLE * LOOP_T + step * H_STEP
    if t >= -1e-9:
        k = step - int(round(SETTLE * LOOP_T / H_STEP))
        if k % int(round(1 / (FPS * H_STEP))) == 0 and len(states) < NF:
            states.append([(n["sway"], n["u"]) for n in nodes])
        if state0 is None and abs(t) < 1e-9:
            state0 = [(n["sway"], n["u"]) for n in nodes]
        if abs(t - LOOP_T) < 1e-9:
            state16 = [(n["sway"], n["u"]) for n in nodes]
    acc = theta_acc(t)
    th = theta(t)
    wh = (t * WAVE_SPEED) % 360
    taup = (t % LOOP_T + LOOP_T) % LOOP_T
    cycp = math.floor(t / LOOP_T)
    for n in nodes:
        n["exist"] = exist_at(n, taup)
        if taup < n["delay"] or taup - n["exitAt"] >= EXIT_DUR:
            continue                     # resting between cycles: frozen
        if n["cyc"] != cycp:             # fresh entry: land swinging
            n["cyc"] = cycp
            n["swayV"] = max(-160, min(160, n["swayV"] + n["kick"]))
            n["uV"] -= 20
        d = wrap180(n["ang"] + th - wh)
        w = math.exp(-(d * d) / (WAVE_SIGMA * WAVE_SIGMA))
        w0 = TAU * n["f0"]
        a = (-w0 * w0 * n["sway"] - 2 * n["zeta"] * w0 * n["swayV"]
             - n["gain"] * acc + n["gust"] * w)
        n["swayV"] += a * H_STEP
        n["sway"] += n["swayV"] * H_STEP
        aU = (-n["fu"] * n["fu"] * n["u"] - 2 * 0.22 * n["fu"] * n["uV"]
              - acc * 0.017453 * 1.35 * n["radPx"] + n["gustT"] * w)
        n["uV"] += aU * H_STEP
        n["u"] += n["uV"] * H_STEP
    collide()

closure = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
              for a, b in zip(state0, state16))
print(f"loop closure: max state diff over one loop = {closure:.4f}", flush=True)

# ---- render ----
W = int(round(meta["W"]))
Hh = int(round(meta["H"]))
ox, oy = SQ / 2 - cx, SQ / 2 - cy   # paste offset to centre the ring


def sprite_affine(spr, angle_deg, sx, sy, px, py, tx=0.0, ty=0.0):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    ia, ib = c / sx, s / sx             # inverse of rot + (sx, sy) scale
    id_, ie = -s / sy, c / sy
    ic = px - ia * (px + tx) - ib * (py + ty)
    if_ = py - id_ * (px + tx) - ie * (py + ty)
    return spr.transform(spr.size, Image.AFFINE, (ia, ib, ic, id_, ie, if_),
                         resample=Image.BICUBIC)


frames = []
for k in range(NF):
    t = k / FPS
    th = theta(t)
    wh = (t * WAVE_SPEED) % 360
    breath = 1 + 0.006 * math.sin(TAU * t / (LOOP_T / 2))

    canvas = Image.new("RGBA", (SQ, SQ), (0, 0, 0, 0))
    for i in draw_order:
        n = nodes[i]
        it = n["it"]
        d = wrap180(n["ang"] + th - wh)
        w = math.exp(-(d * d) / (WAVE_SIGMA * WAVE_SIGMA))
        s = 1 + 0.05 * w
        breeze = n["breezeA"] * math.sin(TAU * n["breezeF"] * t + n["breezeP"])
        sway_k, u_k = states[k][i]
        angle = n["rot"] + sway_k + breeze + n["spin"] * t

        # cycle envelope: reveal pop -> alive -> farewell (matches index.html)
        tau = t % LOOP_T
        e_in = tau - n["delay"]
        e_out = tau - n["exitAt"]
        if e_in < 0 or e_out >= EXIT_DUR:
            continue
        scE, opE, txE, tyE, sq = 1.0, 1.0, 0.0, 0.0, 0.0
        if e_in < ENTRY_DUR:                        # element-by-element reveal
            r = e_in / ENTRY_DUR
            qq = r - 1
            scE = 0.45 + 0.55 * (1 + 2.70158 * qq ** 3 + 1.70158 * qq ** 2)
            e = 1 - (1 - r) ** 3
            txE = n["ux"] * 24 * (1 - e) + n["uy"] * 12 * (1 - e) ** 2
            tyE = n["uy"] * 24 * (1 - e) - n["ux"] * 12 * (1 - e) ** 2
            sq = 0.045 * max(0.0, math.sin((r - 0.3) / 0.7 * math.pi))
            opE = min(1.0, r / 0.5)
        elif e_out >= 0:                            # farewell: lift off, fade
            q = e_out / EXIT_DUR
            scE = 1 - 0.45 * q * q
            opE = max(0.0, 1 - 1.2 * q * q)
            drift = 20 * q * q
            txE, tyE = n["ux"] * drift, n["uy"] * drift
        if opE <= 0:
            continue

        S = s * scE
        spr = sprite_affine(n["sprite"], angle, S * (1 + sq), S * (1 - sq),
                            it["px"] + n["pad"], it["py"] + n["pad"],
                            n["tanX"] * u_k + txE, n["tanY"] * u_k + tyE)
        if opE < 1:
            al = spr.getchannel("A").point(lambda v: int(v * opE))
            spr.putalpha(al)
        canvas.alpha_composite(spr, (int(round(it["x"] + ox - n["pad"])),
                                     int(round(it["y"] + oy - n["pad"]))))

    # whole-wheel rotation + breath about the ring centre
    a = math.radians(th)
    c, s_ = math.cos(a), math.sin(a)
    ia, ib = c / breath, s_ / breath
    id_, ie = -s_ / breath, c / breath
    pcx = pcy = SQ / 2
    ic = pcx - ia * pcx - ib * pcy
    if_ = pcy - id_ * pcx - ie * pcy
    frame = canvas.transform((SQ, SQ), Image.AFFINE, (ia, ib, ic, id_, ie, if_),
                             resample=Image.BICUBIC)
    m = (SQ - CROP) // 2
    frame = frame.crop((m, m, m + CROP, m + CROP))
    frames.append(frame.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS))
    if k % 50 == 0:
        print(f"frame {k}/{NF}", flush=True)

webp_path = os.path.join(D, "cherry_loop.webp")
frames[0].save(webp_path, "WEBP", save_all=True, append_images=frames[1:],
               duration=int(1000 / FPS), loop=0, quality=82, method=4)
print("cherry_loop.webp:", os.path.getsize(webp_path) // 1024, "KB")

if shutil.which("ffmpeg"):
    with tempfile.TemporaryDirectory() as td:
        for k, fr in enumerate(frames):
            fr.save(os.path.join(td, f"a{k:04d}.png"))      # RGBA, exact frames
            solid = Image.new("RGB", fr.size, BG)
            solid.paste(fr, (0, 0), fr)
            solid.save(os.path.join(td, f"s{k:04d}.png"))
        mp4 = os.path.join(D, "cherry_loop.mp4")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS),
                        "-i", os.path.join(td, "s%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        mp4], check=True, capture_output=True)
        print("cherry_loop.mp4:", os.path.getsize(mp4) // 1024, "KB", flush=True)
        # ProRes 4444 with alpha, built from the exact frame sequence (the
        # animated webp merges identical blank frames, so never build from it)
        mov = os.path.join(D, "cherry_loop_alpha.mov")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS),
                        "-i", os.path.join(td, "a%04d.png"),
                        "-c:v", "prores_ks", "-profile:v", "4444",
                        "-pix_fmt", "yuva444p10le", mov],
                       check=True, capture_output=True)
        print("cherry_loop_alpha.mov:", os.path.getsize(mov) // 1024, "KB")
else:
    print("ffmpeg not found - skipped mp4/mov (the webp loops seamlessly on its own)")
