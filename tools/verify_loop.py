"""Verify the rendered loop's wrap is seamless (frame N -> frame 0).

Note: the ProRes/mp4 exports live in render_loop.py — they are built from
the exact frame sequence there. The animated webp merges identical blank
frames (the empty beat), so it reports fewer frames than rendered; that is
harmless for playback but never re-encode video from the webp.
"""
import os

import numpy as np
from PIL import Image, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

im = Image.open(os.path.join(ROOT, "exports", "cherry_loop.webp"))
frames = [f.convert("RGBA") for f in ImageSequence.Iterator(im)]
print("frames:", len(frames), "size:", frames[0].size)

a0 = np.asarray(frames[0]).astype(np.int16)
a1 = np.asarray(frames[1]).astype(np.int16)
aN = np.asarray(frames[-1]).astype(np.int16)
step = np.abs(a1 - a0).mean()
wrap = np.abs(a0 - aN).mean()
print(f"mean px diff consecutive (0->1): {step:.3f}")
print(f"mean px diff at the wrap (last->0): {wrap:.3f}")
print("seam ratio (wrap/step, ~1.0 = seamless):", round(wrap / max(step, 1e-6), 2))

