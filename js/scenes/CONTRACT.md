# Scene contract

Interstella is a catalog of astrophysical objects. The host (`index.html`) owns the renderer,
camera, post-processing (bloom → ACES tone map → grain), UI chrome and audio. Each catalog entry
is a **scene** defined in one module under `js/scenes/`. `js/scenes/moon.js` is a complete,
working reference — read it first and copy its structure.

## Module shape

```js
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

export default [ sceneDef, sceneDef, ... ];   // one module may export several scenes
```

The module is already listed in `js/scenes/manifest.js`. Do not edit the manifest, `index.html`,
`js/glsl/*`, or any other scene's file. If you genuinely need something from the host, say so in
your final report instead of editing it.

## Scene definition

| field | meaning |
|---|---|
| `id` | unique kebab-case id, also used in `#hash` deep links and `tools/snap.sh` |
| `name` | Vietnamese display name (e.g. `'Mặt Trời'`) |
| `category` | `'compact'` · `'stars'` · `'planets'` · `'small'` · `'galaxies'` · `'nebulae'` |
| `order` | sort key inside the category (lower first) |
| `subtitle` | one short line: type / classification |
| `swatch` | `[innerColor, outerColor]` CSS colors for the catalog card orb |
| `blurb` | 2–3 Vietnamese sentences: what it is and the one physical idea that makes it interesting (HTML allowed) |
| `intro` | optional toast shown after arriving (HTML) — a "try this" hint is ideal |
| `facts` | `[[label, value], ...]` real measured data, Vietnamese labels |
| `unitKm` | kilometres per scene unit (used for the live distance row); omit for galaxies/nebulae if meaningless and use your own `hud` rows |
| `camera` | `{ distance, polar°, azimuth°, min, max, fov, target:[x,y,z] }` — OrbitControls orbit `target` |
| `exposure`, `bloom` | defaults applied on entry (user can change them) |
| `params` | panel controls, see below |
| `actions` | dock buttons `{ label, key?, title?, run(ctx) }` |
| `usesBH` | `true` only if the scene renders the host's accretion disk (shows the black-hole panel sections) |
| `create(ctx)` | builds and returns the **instance** (called once, lazily, on first visit) |

### params

```js
{ key, label, type: 'range', min, max, step, default, fmt?: v => string, note? }
{ key, label, type: 'toggle', default: bool, note? }
{ key, label, type: 'seg', options: [[value, 'label'], ...], default, note? }
```
Values live in `ctx.params[key]` (read them every frame in `update`). Optional
`onChange(value, ctx)` on the param, or `instance.onParam(key, value, ctx)`.

Reserved keys (do not use for actions): `u [ ] k h s m f ? /` and Escape.

## Instance (returned by `create`)

| field | meaning |
|---|---|
| `material` | **required.** A `THREE.ShaderMaterial` drawn as a fullscreen quad into the linear HDR target |
| `overlay` | optional `THREE.Object3D` drawn after the fullscreen pass with the perspective camera (no depth buffer — use additive blending, `depthTest:false, depthWrite:false`) |
| `update(dt, ctx)` | per frame; set uniforms from `ctx.params`, `ctx.time`, `ctx.camera` |
| `hud(ctx)` | optional; returns live `[[label, value], ...]` rows for the info card (called at 10 Hz) |
| `onClick(info, ctx)` | optional; `info = { x, y, ndcX, ndcY, dir, origin, shift }` (`dir`/`origin` are world-space Vector3) |
| `enter(ctx)` / `exit(ctx)` | optional hooks when the scene becomes active / inactive |
| `ready` | optional Promise that resolves when large assets (textures, data) have loaded; the host keeps the transition (and snapshots) waiting on it, up to 15 s (60 s in snap mode). Put assets under `assets/<scene>/` |
| `setQuality(q)` | optional; `q = { scale, steps, k }` when the user changes quality (`steps` is the black-hole march budget: 200 / 320 / 480 / 720) |

## ctx

```
THREE, camera, controls (OrbitControls), renderer, overlayScene
shared        uniforms every shader needs — spread into yours: uniforms: { ...ctx.shared, ...own }
bhUniforms    the black hole's uniforms (already include shared); for scenes reusing js/glsl/blackhole.js
bbTexture, bbColor(T) → [r,g,b]  blackbody colour on the CPU (same table as the GLSL bb())
params        this scene's param values
time          { sim, wall, dt } seconds; sim = wall × global time-speed slider (default 6×)
quality       { scale, steps, k } at creation time
audio         ping(freq, gain, pan) · click(gain, pan, brightness0to1) · tone(freq, dur, gain, pan, type) · ringdown() · started
toast(html, ms), fmt.{dist(m), time(s), sci(x), num(x)}
syncUI()      refresh the panel after changing ctx.params from code (the host also calls it after actions and at 10 Hz)
bhGlobals     the host's global settings object (read-only for scenes, e.g. timeScale, exposure)
```

## Shaders

* Start the fragment shader with `COMMON_GLSL` (declares `uRes uTime uWall uCamPos uCamRight uCamUp
  uCamFwd uTanHalfFov uBB uSkyMode uStarBright uSkyBright`). It gives you `camRay()`, `snoise`,
  `fbm(p, octaves)`, `hash33`, `hash13`, `worley(p)`, `sphereHit(ro, rd, c, r)`, `bb(T)`
  (blackbody chromaticity, luminance 1), `sky(dir, gs)` (procedural Milky Way + stars; `gs` = frequency
  shift, use 1.0), `gridLine`, `erfA`, constants `PI`, `TAU`.
* Use `FULLSCREEN_VERT` as the vertex shader, `depthTest:false, depthWrite:false`.
* Write **linear HDR** radiance to `gl_FragColor` (alpha 1). Tone mapping, exposure and gamma are done
  later. Bloom picks up values above ~1.25. A sunlit surface should sit around 0.2–1.0; reserve
  > 2 for things that genuinely glow.
* The fullscreen pass renders at 40–100 % resolution (adaptive). Budget: the scene should stay
  comfortably above 60 fps at 1080p on a mid-range GPU (RTX 3060 class). Keep loops bounded with
  constants; volumetric marches ≤ ~96 steps with early exit; avoid >4 octave noise inside march loops
  where you can.
* WebGL2 / GLSL ES 3.0 (three.js adds the version line); `transpose`, `inverse`, `tanh`, bit ops OK.

## Quality bar

* **Physically grounded.** Use real parameters (radii, temperatures, rotation, composition) and real
  mechanisms (limb darkening laws, Rayleigh scattering, blackbody colour, Keplerian/flat rotation
  curves, ionisation layers, …). When you approximate, keep it plausible and note it in a comment.
* **Beautiful.** This is a showcase; aim for the image a planetarium would project.
* **Interactive.** Each scene needs ≥ 2 meaningful params and ideally an action or click behaviour
  that teaches something ("make me surprise").
* All user-facing text in Vietnamese with correct diacritics.

## Verify your work (required)

The host has a snapshot mode. From the project root:

```bash
tools/snap.sh <sceneId> "$TEMP/snaps/<name>.png" "cam=3,80,30&params=key:value&t=40&ui=0"
```

It prints `{"ok":..., "errors":[...], "hud":[...]}` (shader compile errors include the offending
source lines) and writes a 1280×720 screenshot you can look at with the Read tool. Iterate until
`ok` is `true` and the images look right from several camera angles and parameter settings, with the
UI on (`ui=1`, default) at least once to check the info card and panel.

Measure real GPU cost with `tools/bench.sh med <sceneId>` (prints `benchMs` per frame from a GPU timer
query at 1920×1080, 75 % render scale; rerun if it prints 0). Aim for ≤ 8 ms; anything above ~12 ms
needs optimising. `index.html?snap=1&cycle=1` visits every scene in one page load.

Do **not** use the shared browser preview tools — several authors work at once. Do not commit to git.

## Final report

List each scene id, what it shows, its params/actions, the physics used and any approximations,
the snapshot commands you used to verify, and anything you need from the host.
