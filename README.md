# 🍒 Cherry Season Loader

A loading animation built from a real photograph — the full life of a cherry,
from winter bud to blossom to ripe fruit to dried, arranged in a ring and
brought to life with spring physics.

**[▶ Live demo](https://azael-adonai.github.io/cherry-loader/)** · [seamless-loop capture mode](https://azael-adonai.github.io/cherry-loader/?loop) · [progress demo](https://azael-adonai.github.io/cherry-loader/?progress)

![Cherry season loop](exports/cherry_loop.webp)

Every element is a sprite cut from one photo. Nothing is vector-redrawn:
the stems that swing, the blossoms that pinwheel, the cherries that bump
into each other — that's the original fruit, animated.

## What it does

- **Real physics.** The ring spins with gentle surges; every stem is a damped
  spring pendulum driven by the wheel's angular acceleration, with its
  frequency derived from the *actual stem length* measured in the photo.
- **Collisions.** Each fruit can slide along the ring arc. Spin surges and a
  travelling gust shunt neighbours into each other — they collide with
  mass-weighted impulses, and every knock shivers the stems.
- **A living loop.** Each 16-second cycle: the bud appears… the blossoms
  bloom… the season cascades clockwise into the full ring, lives for a while,
  then departs the same way, breathes for a beat, and begins again.
- **Mathematically seamless.** Every period divides the 16 s master loop and
  the physics runs on a fixed 120 Hz grid, so the cycle repeats *exactly* —
  state drift across the loop is 0.0000.

## Use it

**Drop-in (zero dependencies):** copy
[`dist/cherry_loader_single.html`](dist/cherry_loader_single.html) — one
self-contained ~255 KB file — and open it, iframe it, or lift its markup
into your page.

**From source:** serve this folder (`python -m http.server`) and open
`index.html`. It loads `meta.json` + `sprites/`.

**Progress API** (optional — indeterminate by default):

```js
CherryLoader.set(0.42);  // shows a cherry-red arc + percentage
CherryLoader.done();     // farewell ripple of stem-swings, then fade out
```

Add `?progress` to the URL for a simulated-progress demo.

**Capture a video loop:** add `?loop` — the physics is pre-settled into its
periodic steady state, so **any** screen recording of exactly 16 seconds is a
perfect seamless loop. Or just take the pre-rendered files in
[`exports/`](exports/): transparent animated WebP and H.264 MP4 (a ProRes
4444 `.mov` with alpha can be rendered locally, see below).

Respects `prefers-reduced-motion` (static ring, no animation).

## Rebuild it

Requires Python 3 with `numpy`, `scipy`, `pillow` (and `ffmpeg` on PATH for
the video exports).

```sh
python tools/slice_sprites.py   # photo -> sprites/ + meta.json
                                #   segments each fruit, repairs matting-severed
                                #   stems, splits touching cherries, rebakes
                                #   ambient shadows, snaps the ring to a perfect
                                #   circle at 15 degree spacing
python tools/build_single.py    # -> dist/cherry_loader_single.html
python tools/render_loop.py     # -> exports/ (webp + mp4 + ProRes alpha .mov)
python tools/verify_loop.py     # checks the rendered wrap is seamless
```

`tools/render_loop.py` reproduces the browser animation exactly — both sides
derive per-item randomness from the same index-hashed PRNG — and prints the
loop-closure metric.

## Credits

Photo and concept by **Salah Eddine El Moukhtari** — one cherry tree,
photographed across a season. Animation engineered with
[Claude Code](https://claude.com/claude-code).

## License

[MIT](LICENSE)
