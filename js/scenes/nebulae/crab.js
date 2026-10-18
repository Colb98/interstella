// Tinh vân Con Cua (M1) — the remnant of SN 1054, rebuilt from the real Hubble / Webb / Chandra
// images (assets/crab/, see CREDITS.txt; preprocessing scripts in assets/crab/src/).
// One scene unit = one light-year; distance 6 500 ly (1″ = 0.0315 ly).
//
// Approach: "image-based 3D". The reference images were registered to one sky frame centred on the
// pulsar (north up, west = +x; Earth looks along −z; Webb matched to Hubble on field stars, Chandra by
// its scale bar) and split, as radiance under the host's ACES curve (inverse tone mapped), into
//   · line / dust emission of the filaments, field stars removed (Gaia DR3)   crab_hst / crab_jwst.webp
//   · the synchrotron nebula (Hubble, Webb) and the Chandra image               crab_aux.webp (R, G, B)
//   · dark lanes: light missing below the synchrotron envelope                  crab_dust.webp
// with F + S·(1 − A) = photograph. In 3D each pixel's light is placed along the Earth-view ray:
//   filaments   → two thin, lumpy shell surfaces (Martin, Milisavljevic & Drissen 2021: the SITELLE 3D
//                 map shows the filaments as a honeycomb cage on a shell, not an ellipsoid; here the
//                 outline comes from the image and the line-of-sight depth ≈ the short projected radius),
//                 each honeycomb cell on the near or the far side; shaded as optically thin sheets;
//   synchrotron → smooth part fills the volume, fine structure on an inner sheet; dust on the near
//                 side absorbs it;
//   inner PWN   → part of its synchrotron light in the pulsar's equatorial plane (torus, wisps) and
//                 on the spin axis (jet, counter-jet).
// Every weight integrates to 1 along the Earth-view rays, so the canonical view reproduces the
// photograph; orbiting reveals the cage (parallax between near and far filaments, limb brightening).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';
import { t, dec, locale } from '../../i18n.js';

const ASSET = f => new URL(`../../../assets/crab/${f}`, import.meta.url).href;
const LY_PER_AS = 6500 * Math.PI / 648000;      // ly per arcsec at 6 500 ly (1″ = 0.0315 ly)
const FRAME_AS = 400;                            // texture frame: 400″ square centred on the pulsar
const FRAME = FRAME_AS * LY_PER_AS;              // 12.6 ly
const D0 = 36;                                   // canonical camera distance (Earth view, a 'telescope' field of 19°)
const S0A = 0.70, S0B = 0.98;                    // range of the filament shell radius (outline units)
const SYN_EDGE = 1.1;                            // synchrotron volume edge (outline units; > 1 keeps every chord finite)
// pulsar spin axis: projected PA 126° (Ng & Romani 2004 torus fit), inclined 63° to the line of sight,
// south-east jet approaching (+z = toward Earth). World: x = west, y = north.
const PA = 126 * Math.PI / 180, INC = 63 * Math.PI / 180;
// close-up: the eastern limb of the cage, seen from just outside, a little off the Earth line
const CLOSE_T = new THREE.Vector3(-4.6, -1.6, 0.6), CLOSE_V = new THREE.Vector3(-0.9, 0.15, 2.5).normalize().multiplyScalar(2.8);
const SPIN = new THREE.Vector3(-Math.sin(PA) * Math.sin(INC), Math.cos(PA) * Math.sin(INC), Math.cos(INC)).normalize();

const FRAG = COMMON_GLSL + /* glsl */`
#ifndef STEPS
#define STEPS 80
#endif
uniform sampler2D uFil, uAux, uDust, uGeo, uOut, uLut;
uniform float uMode;          // 0 Hubble, 1 Webb, 2 Chandra
uniform vec3 uSynCol;         // synchrotron colour per unit of the stored channel
uniform vec3 uAuxSel, uDustSel;
uniform float uHMean;
uniform float uSc, uFilK, uSynK, uBright, uPulse, uWisp, uDetail;
uniform vec3 uSpin;
const float D0 = ${D0.toFixed(1)};
const float FRAME = ${FRAME.toFixed(5)};
const float S0A = ${S0A.toFixed(3)}, S0B = ${S0B.toFixed(3)};
const float SYN_EDGE = ${SYN_EDGE.toFixed(3)};
const float TEXEL = FRAME / 4096.0;

float ign(){ return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))); }
vec3 dec(vec3 e){ vec3 y = pow(e, vec3(2.2)); return y / max(1.0 - y, vec3(4e-3)); }
float dec1(float e){ float y = pow(e, 2.2); return y / max(1.0 - y, 4e-3); }
vec3 starPSF(vec3 rd, vec3 v, float tc, vec3 col, float lum){
  if (tc <= 0.0) return vec3(0.0);
  float a = length(v - rd * tc) / tc;
  float x = a / (2.0 * uTanHalfFov / uRes.y);
  return col * lum * (exp(-0.35 * x * x) + 0.012 / (1.0 + 0.015 * x * x));
}
// --- inner pulsar-wind nebula routing (functions of the sky position only, so the Earth view stays exact)
float zEq(vec2 xy){ return -dot(uSpin.xy, xy) / uSpin.z; }                 // equatorial plane depth
float rhoEq(vec2 xy){ return length(vec3(xy, zEq(xy))); }                     // radius in that plane
// share of the inner light placed in the equatorial plane: inner ring ≈ 14″ (0.45 ly), torus ≈ 40″
// (1.3 ly) semi-major axis (Chandra); the rest stays volumetric
float mTorus(vec2 xy){ return 0.65 * smoothstep(1.45, 0.85, rhoEq(xy)); }
float jetWin(float ta){                                                       // along the spin axis, ly
  return ta > 0.0 ? smoothstep(0.12, 0.3, ta) * smoothstep(2.1, 1.2, ta)      // SE jet (approaching)
                  : 0.35 * smoothstep(-0.1, -0.25, ta) * smoothstep(-0.9, -0.45, ta); // NW counter-jet
}
const float JW = 0.09;                                                        // jet radius, ly
float mJet(vec2 xy){
  float pl = length(uSpin.xy);
  vec2 a2 = uSpin.xy / pl;
  float t2 = dot(xy, a2);
  float e = length(xy - a2 * t2);
  return 0.6 * jetWin(t2 / pl) * exp(-e * e / (JW * JW));   // 60 % on the axis, the rest stays diffuse
}
// sky-plane sample of the source maps; lod from the pixel footprint
vec2 skyUV(vec2 xy){ return xy / FRAME + 0.5; }
// sky position of a 3D point as seen from the Earth-view camera at (0, 0, D0): the photographs are
// projections along these rays, so every per-pixel quantity is a function of skyXY
vec2 skyXY(vec3 p){ return p.xy * (D0 / (D0 - p.z)); }
float ridge3(vec3 x){ float n = 1.0 - abs(snoise(x)); return n * n; }
// synchrotron split: a smooth share (mip-blurred, capped so it never spills past the nebula's edge)
// fills the volume; the remainder (fine structure) goes on the inner sheet
float synVol(vec2 uv, float lod, vec3 sel, float isX){
  float Ss = dec1(dot(textureLod(uAux, uv, lod + 5.0).rgb, sel));
  float Sm = dec1(dot(textureLod(uAux, uv, lod + 2.0).rgb, sel));
  return min(mix(0.7, 0.55, isX) * Ss, Sm);
}
float synSplit(vec2 uv, float lod, vec3 sel, float isX, out float vol){
  vol = synVol(uv, lod, sel, isX);
  return max(dec1(dot(textureLod(uAux, uv, lod).rgb, sel)) - vol, 0.0);
}

void main(){
  vec3 rd = camRay();
  float sc = max(uSc, 1e-3);
  vec3 ro = uCamPos / sc;                      // nebula frame (today's size)
  float pxAng = 2.0 * uTanHalfFov / uRes.y;    // radians per pixel
  vec3 I = vec3(0.0); float T = 1.0;

  // the Crab pulsar (PSR B0531+21): ~16.5 mag optical, pulses twice per 33.7 ms turn
  vec3 v = -uCamPos; float tcS = dot(v, rd);
  vec3 psrCol = uMode > 1.5 ? vec3(0.75, 0.88, 1.0) : vec3(0.85, 0.9, 1.0);
  vec3 star = starPSF(rd, v, tcS, psrCol, (0.6 + 9.0 * uPulse) * (uMode > 1.5 ? 2.5 : 1.0));
  float tStar = tcS / sc; bool starDone = false;

  // equatorial-plane hit (torus + wisps) and closest approach to the spin axis (jet)
  float dn = dot(rd, uSpin);
  float tP = abs(dn) > 1e-4 ? -dot(ro, uSpin) / dn : -1.0;
  bool planeDone = tP <= 0.0;
  float bq = dn, dq = dot(rd, ro), eq = dot(uSpin, ro), den = max(1.0 - bq * bq, 1e-4);
  float tJr = (bq * eq - dq) / den, tJa = (eq - bq * dq) / den;
  vec3 pr = ro + rd * tJr, pa = uSpin * tJa;
  float dJ = length(pr - pa);
  bool jetDone = tJr <= 0.0 || dJ > 4.0 * JW || jetWin(tJa) <= 0.0;

  vec3 auxW = uAuxSel;
  float isX = step(1.5, uMode);
  vec3 xCol = vec3(0.12, 0.38, 1.0) * 1.8;   // Chandra palette: blue, saturating to white where bright

  // main march through the bounding ellipsoid (pulsar-centred, nebula units).
  // Filaments live on thin, lumpy shell surfaces (two radii, near and far side): between samples we
  // look for sign changes of g = s − s₀(sky) and shade each crossing like an optically thin sheet.
  // Synchrotron light fills the volume.
  vec3 BX = vec3(6.3, 6.3, 6.0);
  vec3 o = ro / BX, d = rd / BX;
  float qa = dot(d, d), qb = dot(o, d), qc = dot(o, o) - 1.0, h = qb * qb - qa * qc;
  float t0 = 1e9, t1 = -1e9;
  if (h > 0.0){ h = sqrt(h); t0 = max((-qb - h) / qa, 0.0); t1 = (-qb + h) / qa; }
  if (t1 > t0){
    float dt = (t1 - t0) / float(STEPS);
    float t = t0 + dt * ign();
    vec3 gPrev = vec3(0.0); vec3 pPrev = vec3(0.0); bool first = true;
    for (int i = 0; i < STEPS + 1; i++){
      if (t > t1 + dt) break;
      vec3 p = ro + rd * t;
      // point-like features in front of this sample are composited in depth order
      if (!starDone && tStar < t){ I += star; starDone = true; }
      if (!planeDone && tP < t){
        planeDone = true;
        vec3 ph = ro + rd * tP;
        float rho = length(ph);
        vec2 sxy = skyXY(ph);
        float mt = mTorus(sxy) * (1.0 - mJet(sxy));
        if (mt > 1e-3){
          vec2 uvp = skyUV(sxy);
          float lodP = max(log2(max(tP * pxAng / TEXEL, 1e-3)) - 0.6, 0.0);
          vec3 aux = textureLod(uAux, uvp, lodP).rgb;
          float S = dec1(dot(aux, auxW));
          // wisps: synchrotron ripples moving out through the torus at ~0.5 c (time-lapse)
          vec3 e1 = normalize(cross(uSpin, vec3(0.0, 0.0, 1.0))), e2 = cross(uSpin, e1);
          float ang = atan(dot(ph, e2), dot(ph, e1));
          // a few discrete arcs born at the inner ring (~0.45 ly) drifting out through the torus
          float wsum = 0.0;
          for (int k = 0; k < 3; k++){
            float ph = fract(uWisp * 0.35 + float(k) / 3.0);
            float rk = 0.42 + ph * 0.95;
            float arc = 0.55 + 0.45 * sin(ang + float(k) * 2.1 + rk * 3.0);
            wsum += exp(-(rho - rk) * (rho - rk) / (0.045 * 0.045)) * arc * (1.0 - ph);
          }
          float wisp = 1.0 + 0.9 * (wsum - 0.07) * smoothstep(0.3, 0.45, rho);
          // a thick torus, not a sheet: limit the edge-on brightening
          float geo = abs(uSpin.z) / max(abs(dn), 0.6);
          I += mt * geo * T * S * mix(uSynCol, xCol, isX) * uSynK * wisp * uBright;
        }
      }
      if (!jetDone && tJr < t){
        jetDone = true;
        vec2 uvj = skyUV(skyXY(pr));
        float S = dec1(dot(textureLod(uAux, uvj, 1.0).rgb, auxW));
        float sinC = length(uSpin.xy), sinR = sqrt(den);
        float kink = 1.0 + 0.35 * sin(tJa * 9.0 - uWisp * 2.0);              // the jet's knotty, kinking flow
        I += T * S * 0.6 * jetWin(tJa) * exp(-dJ * dJ / (JW * JW)) * sinC / max(sinR, 0.3) * kink
               * mix(uSynCol, xCol, isX) * uSynK * uBright;
      }

      float wz = D0 - p.z;
      vec2 xy = p.xy * (D0 / wz);
      float cosA = wz / length(vec3(p.xy, wz));     // Earth-view rays are not parallel: dl = dz / cos α
      float th = atan(xy.y, xy.x);
      vec2 RH = texture(uOut, vec2(th / TAU + 0.5, 0.5)).rg;     // outline radius R(θ), depth scale H(θ)
      float rN = length(xy) / RH.x;
      RH.y = mix(uHMean, RH.y, smoothstep(0.0, 0.6, rN));        // no cone point on the axis
      float zz = p.z / RH.y;
      float s = sqrt(rN * rN + zz * zz);
      vec2 uv = skyUV(xy);
      vec4 g = texture(uGeo, uv);
      // shell radii of the two filament layers; never inside the sky radius, so every Earth-view
      // ray crosses each layer twice (near the rim the layers fold into a skirt close to z = 0)
      // towards the outline the shell radius rises to ~1, so the silhouette is the shell's limb
      float sA = mix(mix(S0A, S0B, g.g), 1.02, smoothstep(0.45, 1.0, rN));
      float sB = sA - 0.09 * (1.0 - smoothstep(0.75, 1.0, rN));
      sA = max(sA, rN + 0.01); sB = max(sB, rN + 0.005);
      // inner sheet that carries the fine structure of the synchrotron (wisps, fibres, bays); the
      // smooth part fills the volume. Seen off the Earth line, extruded fine structure would streak.
      float sF = mix(mix(0.40, 0.62, g.g), 1.0, smoothstep(0.35, 1.0, rN));
      sF = max(sF, rN + 0.02);
      vec3 gg = vec3(s - sA, s - sB, s - sF);
      vec3 cr = gg * gPrev;
      if (!first && min(cr.x, min(cr.y, cr.z)) < 0.0){
        for (int L = 0; L < 3; L++){
          float a = L == 0 ? gPrev.x : L == 1 ? gPrev.y : gPrev.z, b = L == 0 ? gg.x : L == 1 ? gg.y : gg.z;
          if (a * b >= 0.0 || (L < 2 && isX > 0.5)) continue;
          vec3 ph = mix(pPrev, p, a / (a - b));
          vec2 xyh = skyXY(ph);
          vec2 uvh = skyUV(xyh);
          float th2 = max(t - dt, 0.0) + dt * (a / (a - b));
          float lodS = log2(th2 * pxAng / TEXEL) - 0.6;
          // sheet shading: normal of s(p), obliquity relative to the Earth-view ray (exact there)
          vec3 nrm = normalize(vec3(xyh / (RH.x * RH.x), ph.z / (RH.y * RH.y)));
          vec3 ec = normalize(vec3(-ph.xy, D0 - ph.z));
          float amp = max(abs(dot(nrm, ec)), 0.7) / max(abs(dot(nrm, rd)), 0.7);
          if (L == 2){
            float vol;
            float fine = synSplit(uvh, max(lodS - 1.0, 0.0), auxW, isX, vol);
            float innerS = 1.0 - mTorus(xyh) * (1.0 - mJet(xyh)) - mJet(xyh);
            I += T * fine * 0.5 * amp * innerS * mix(uSynCol, xCol, isX) * uSynK * uBright;
            continue;
          }
          vec3 F = dec(textureLod(uFil, uvh, max(lodS, 0.0)).rgb);
          // near / far split per honeycomb cell, blended to ½ where the two sides meet at the rim
          float fz = mix(0.5, g.r, smoothstep(0.0, 0.4, abs(ph.z) / RH.y));
          float side = ph.z > 0.0 ? fz : 1.0 - fz;
          float wl = L == 0 ? g.b : 1.0 - g.b;
          // close-up: fibrous sub-texel structure (radially stretched Rayleigh–Taylor fingers)
          if (uDetail > 0.0 && lodS < 0.0){
            float k = clamp(-lodS * 0.5, 0.0, 1.0) * uDetail;
            vec3 rad = normalize(ph);
            vec3 qd = ph * 14.0; qd -= rad * dot(qd, rad) * 0.8;     // stretched along the radius
            float fib = ridge3(qd) * 0.6 + ridge3(qd * 2.9 + 5.0) * 0.4;
            F *= mix(1.0, 0.2 + 2.0 * fib * fib, k);
          }
          I += F * side * wl * amp * uFilK * uBright;
          // dust in the near-side filaments dims the synchrotron light behind them
          // (the decomposition F + S·(1 − A) is exact in the Earth view)
          if (ph.z > 0.0){
            float A = dot(textureLod(uDust, uvh, max(lodS - 1.0, 0.0)).rgb, uDustSel);
            T *= pow(1.0 - min(A, 0.65), wl * min(amp, 2.0));   // capped: deep lanes would read as holes off-axis
          }
        }
      }
      gPrev = gg; pPrev = p; first = false;
      if (s < SYN_EDGE && t < t1){
        float nSy = exp(texture(uLut, vec2((min(rN, 1.25) / 1.25 * 127.0 + 0.5) / 128.0, 0.5)).g) * RH.y / cosA;
        float wsy = pow(1.0 - (s * s) / (SYN_EDGE * SYN_EDGE), 1.5);
        float inner = 1.0 - mTorus(xy) * (1.0 - mJet(xy)) - mJet(xy);
        float lod = max(log2(t * pxAng / TEXEL) - 1.0, 0.0);
        float vol = synVol(uv, lod, auxW, isX);
        I += T * vol * wsy / (2.0 * nSy) * inner * mix(uSynCol, xCol, isX) * uSynK * uBright * dt;
      }
      t += dt;
    }
  }
  if (!starDone) I += star;
  I += sky(rd, 1.0) * 0.15 * (1.0 - 0.7 * isX);
  // the host darkens the frame edges (lens vignette after tone mapping); the telescope mosaics are
  // flat-fielded, so undo most of it: x' = ACES⁻¹(ACES(x) / vignette)
  vec2 dc = gl_FragCoord.xy / uRes - 0.5;
  float vig = 1.0 - 0.55 * 0.85 * smoothstep(0.05, 0.75, dot(dc, dc) * 2.0);
  vec3 y = clamp(I * (2.51 * I + 0.03) / (I * (2.43 * I + 0.59) + 0.14), 0.0, 1.0) / vig;
  y = min(y, vec3(0.985));
  vec3 A = 2.51 - 2.43 * y, B = 0.03 - 0.59 * y, C = -0.14 * y;
  I = (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
  gl_FragColor = vec4(I, 1.0);
}`;

// background / foreground field stars from the Hubble frame, as sprites fixed in angle around the
// Earth line of sight (placed far behind the nebula so they show parallax when orbiting)
const STAR_VERT = /* glsl */`
attribute vec4 aCol;          // rgb, sigma (radians)
uniform float uPxScale, uMode, uGain;
varying vec3 vCol; varying float vSpk; varying float vPs;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float sigPx0 = aCol.w * uPxScale;                    // Gaussian σ in device pixels
  float sigPx = max(sigPx0, 0.65);                     // never below a pixel: keep the flux, widen the PSF
  float spk = uMode > 0.5 && uMode < 1.5 ? 1.0 : 0.0;   // Webb: hexagonal diffraction spikes
  float ps = clamp(sigPx * (spk > 0.5 ? 16.0 : 6.0), 3.0, 96.0);
  gl_PointSize = ps;
  float lum = dot(aCol.rgb, vec3(0.3, 0.5, 0.2));
  // Webb: near-IR stars are bluish-white in the release palette, fainter relative to the nebula
  vCol = (spk > 0.5 ? lum * vec3(0.8, 0.9, 1.1) * 0.35 : aCol.rgb) * uGain * (sigPx0 * sigPx0) / (sigPx * sigPx);
  vSpk = spk; vPs = ps / sigPx;
}`;
const STAR_FRAG = /* glsl */`
varying vec3 vCol; varying float vSpk; varying float vPs;
void main(){
  vec2 c = (gl_PointCoord - 0.5) * vPs;                // in units of σ
  float r2 = dot(c, c);
  float core = exp(-0.5 * r2) + 0.03 / (1.0 + 0.4 * r2);
  float sp = 0.0;
  if (vSpk > 0.5){
    for (int k = 0; k < 3; k++){                       // 6 spikes at 60° (+ the faint horizontal pair)
      float a = float(k) * 1.0472 + 1.5708;
      vec2 dir = vec2(cos(a), sin(a));
      float along = abs(dot(c, dir)), perp = abs(dot(c, vec2(-dir.y, dir.x)));
      sp += exp(-perp * perp * 1.5) * 0.22 / (1.0 + along * 0.5);
    }
    sp += exp(-c.y * c.y * 2.0) * 0.12 / (1.0 + abs(c.x) * 0.6);
  }
  float edge = 1.0 - smoothstep(0.35, 0.5, length(gl_PointCoord - 0.5));
  gl_FragColor = vec4(vCol * (core + sp) * edge, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------
const stepsFor = q => (q.steps <= 200 ? 56 : q.steps <= 320 ? 72 : q.steps <= 480 ? 88 : 104);
const vi = (x, d = 1) => dec(x, d);
const ly = (x, d = 1) => vi(x, d) + t(' năm ánh sáng', ' light-years');

function loadTex(url, { mip = true } = {}) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    tex.colorSpace = THREE.NoColorSpace;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = mip ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = mip;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(t('không tải được ', 'could not load ') + url))));
}
const retry = (fn, ...a) => fn(...a).catch(() => new Promise(r => setTimeout(r, 400)).then(() => fn(...a)));

// half-chord integral of the synchrotron volume profile, ∫₀^∞ w(√(r² + u²)) du, stored as a log
function makeLut() {
  const NR = 128, data = new Float32Array(NR * 4);
  for (let i = 0; i < NR; i++) {
    const r = 1.25 * i / (NR - 1);
    let b = 0; const U = 1.7, n = 600, du = U / n;
    for (let k = 0; k < n; k++) {
      const s = Math.hypot(r, (k + 0.5) * du);
      if (s < SYN_EDGE) b += Math.pow(1 - (s * s) / (SYN_EDGE * SYN_EDGE), 1.5) * du;
    }
    data[i * 4 + 1] = Math.log(Math.max(b, 1e-12));
  }
  const t = new THREE.DataTexture(data, NR, 1, THREE.RGBAFormat, THREE.FloatType);
  t.needsUpdate = true;
  return halfify(t);
}
function halfify(t) {   // convert a float RGBA DataTexture to half floats (linear filtering everywhere)
  const src = t.image.data, dst = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) dst[i] = THREE.DataUtils.toHalfFloat(src[i]);
  const h = new THREE.DataTexture(dst, t.image.width, t.image.height, THREE.RGBAFormat, THREE.HalfFloatType);
  h.minFilter = h.magFilter = THREE.LinearFilter; h.wrapS = t.wrapS; h.wrapT = t.wrapT; h.needsUpdate = true;
  return h;
}
// outline radius R(θ) from the Hubble image, and the line-of-sight depth scale H(θ)
function makeOutline(Ras) {
  const n = Ras.length, data = new Float32Array(n * 4);
  const R = Ras.map(r => r * LY_PER_AS);
  const Rs = [...R].sort((a, b) => a - b), Rmin = Rs[Math.floor(n * 0.1)];
  for (let i = 0; i < n; i++) {
    data[i * 4] = R[i];
    // prolate remnant: depth along the line of sight ~ the short projected radius, a little deeper
    // where the outline reaches further (Martin et al. 2021: not an ellipsoid, a lobed "heart")
    data[i * 4 + 1] = 0.55 * Rmin * 1.08 + 0.45 * R[i] * 0.92;
  }
  const t = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat, THREE.FloatType);
  t.wrapS = THREE.RepeatWrapping; t.needsUpdate = true;
  const h = halfify(t); h.wrapS = THREE.RepeatWrapping; h.wrapT = THREE.ClampToEdgeWrapping;
  let Hm = 0; for (let i = 0; i < n; i++) Hm += data[i * 4 + 1] / n;
  return { tex: h, R, Hmean: Hm };
}
// per-cell fields on the sky plane: near/far split (bimodal, ~1 ly honeycomb cells) and shell radius
function makeGeo() {
  const N = 256, data = new Uint8Array(N * N * 4);
  const hash = (x, y, s) => { const h = Math.sin(x * 127.1 + y * 311.7 + s * 74.7) * 43758.5453; return h - Math.floor(h); };
  const vnoise = (x, y, s) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), w = yf * yf * (3 - 2 * yf);
    const a = hash(xi, yi, s), b = hash(xi + 1, yi, s), c = hash(xi, yi + 1, s), d = hash(xi + 1, yi + 1, s);
    return a + (b - a) * u + (c - a) * w + (a - b - c + d) * u * w;
  };
  const fbm = (x, y, s) => vnoise(x, y, s) * 0.6 + vnoise(x * 2.1, y * 2.1, s + 3) * 0.28 + vnoise(x * 4.3, y * 4.3, s + 7) * 0.12;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const X = ((i + 0.5) / N - 0.5) * FRAME, Y = ((j + 0.5) / N - 0.5) * FRAME;
    const f = fbm(X / 1.1, Y / 1.1, 1) - 0.5;
    const near = 1 / (1 + Math.exp(-f / 0.035));              // mostly 0 or 1: each filament on one side
    const s0 = Math.min(1, Math.max(0, fbm(X / 1.8 + 9, Y / 1.8, 5) * 1.25 - 0.12));
    const o = (j * N + i) * 4;
    const lq = fbm(X / 0.8 + 21, Y / 0.8 - 4, 9) - 0.5;
    const outer = 1 / (1 + Math.exp(-lq / 0.04));             // which of the two shell layers
    data[o] = Math.round((0.06 + 0.88 * near) * 255); data[o + 1] = Math.round(s0 * 255);
    data[o + 2] = Math.round((0.1 + 0.8 * outer) * 255); data[o + 3] = 255;
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.minFilter = t.magFilter = THREE.LinearFilter; t.needsUpdate = true;
  return t;
}

function makeFlyer() {
  let fly = null;
  const sph = new THREE.Spherical(), off = new THREE.Vector3();
  return {
    to(ctx, dist, polar, azim, dur = 2.4) {
      off.copy(ctx.camera.position).sub(ctx.controls.target); sph.setFromVector3(off);
      const from = { r: sph.radius, p: sph.phi, a: sph.theta };
      const to = { r: dist ?? sph.radius, p: THREE.MathUtils.degToRad(polar), a: THREE.MathUtils.degToRad(azim) };
      let da = to.a - from.a; da = Math.atan2(Math.sin(da), Math.cos(da)); to.a = from.a + da;
      fly = { from, to, t: 0, dur };
    },
    update(ctx, dt) {
      if (!fly) return;
      fly.t += dt; const k = Math.min(1, fly.t / fly.dur), e = k * k * (3 - 2 * k);
      sph.set(fly.from.r + (fly.to.r - fly.from.r) * e, fly.from.p + (fly.to.p - fly.from.p) * e, fly.from.a + (fly.to.a - fly.from.a) * e);
      ctx.camera.position.setFromSpherical(sph).add(ctx.controls.target);
      ctx.camera.lookAt(ctx.controls.target);
      if (k >= 1) fly = null;
    },
  };
}

const MODES = [['hst', 'Hubble (2005)'], ['jwst', 'JWST (2023)'], ['xray', 'Tia X Chandra']];
const MODE_TOAST = {
  hst: () => t('Hubble 2005 (24 ảnh WFPC2 ghép): cam = hydro, đỏ = nitơ, hồng = lưu huỳnh, lục = oxy; quầng lam = synchrotron.',
    'Hubble 2005 (mosaic of 24 WFPC2 frames): orange = hydrogen, red = nitrogen, pink = sulfur, green = oxygen; blue haze = synchrotron.'),
  jwst: () => t('Webb 2023 (NIRCam + MIRI, 1,6–21 µm): sợi cam-đỏ là [S III] và bụi ấm; màn sữa lam-trắng là synchrotron với các thớ từ trường.',
    'Webb 2023 (NIRCam + MIRI, 1.6–21 µm): the orange-red filaments are [S III] and warm dust; the milky blue-white haze is synchrotron, streaked along the magnetic field.'),
  xray: () => t('Chandra (tia X): chỉ thấy tinh vân gió pulsar — vành xuyến, các gợn sóng và hai tia phản lực dọc trục quay.',
    'Chandra (X-rays): only the pulsar wind nebula shows — the torus, the wisps and the two jets along the spin axis.'),
};

const crab = {
  id: 'supernova-remnant',
  name: 'Tàn dư siêu tân tinh',
  category: 'nebulae',
  order: 30,
  subtitle: 'Tinh vân Con Cua (M1) · tàn dư của SN 1054',
  swatch: ['#b8d4ff', '#d8642a'],
  blurb: 'Năm 1054, các nhà thiên văn Trung Quốc ghi lại một “khách tinh” sáng đến mức thấy được giữa ban ngày suốt 23 ngày. Ngày nay mảnh vỡ là một <b>lồng sợi khí</b> Hα / [S II] / [O III] giãn nở ~1 500 km/s, bên trong lấp đầy ánh sáng lam trắng — <b>bức xạ synchrotron</b> của electron siêu tương đối tính do <b>pulsar Con Cua</b> (quay 30 vòng/giây) thổi ra. Hình ảnh dựng lại từ ảnh thật của Hubble, Webb và Chandra trên mô hình 3D của Martin và cộng sự (2021).',
  intro: 'Đây là góc nhìn từ Trái Đất — xoay camera để thấy lồng sợi 3D. Đổi <b>ảnh</b> Hubble / Webb / tia X, bấm <b>R</b> để tua lại từ năm 1054.',
  facts: [
    ['Khoảng cách', '≈ 6 500 năm ánh sáng (2,0 kpc)'], ['Chòm sao', 'Kim Ngưu (Taurus)'],
    ['Kích thước', '≈ 12 × 8 năm ánh sáng (6,5′ × 4,2′ trên ảnh)'], ['Tốc độ giãn nở', '≈ 1 500 km/s; lùi ngược hội tụ về năm ~1130 — gió pulsar đang đẩy lớp vỏ nhanh dần'],
    ['Bùng nổ', '4/7/1054'], ['Thấy bằng mắt thường', 'ngày: 23 ngày · đêm: 653 ngày'],
    ['Pulsar', 'PSR B0531+21 · chu kỳ 33,7 ms (29,6 Hz)'], ['Công suất hãm quay', '≈ 4,5×10³¹ W (~10⁵ L☉)'],
    ['Khối lượng sợi khí', '≈ 7 M☉ (khí + ~0,03 M☉ bụi)'],
    ['Ảnh gốc', 'Hubble 2005: NASA, ESA, J. Hester & A. Loll · Webb 2023: NASA, ESA, CSA, STScI, T. Temim (CC BY 4.0) · Chandra: NASA/CXC/SAO/F. Seward et al. · sao nền: ESA/Gaia DR3'],
    ['Mô hình 3D', 'Martin, Milisavljevic & Drissen 2021, MNRAS 502, 1864'],
  ],
  en: {
    name: 'Supernova remnant',
    subtitle: 'Crab Nebula (M1) · remnant of SN 1054',
    blurb: 'In 1054, Chinese astronomers recorded a “guest star” bright enough to see in broad daylight for 23 days. Today the debris is a <b>cage of filaments</b> glowing in Hα / [S II] / [O III], expanding at ~1,500 km/s and filled with blue-white light — <b>synchrotron radiation</b> from ultra-relativistic electrons blown out by the <b>Crab Pulsar</b>, which spins 30 times a second. The view is rebuilt from real Hubble, Webb and Chandra images on the 3D model of Martin et al. (2021).',
    intro: 'This is the view from Earth — orbit the camera to see the 3D filament cage. Switch the <b>image</b> between Hubble / Webb / X-ray, and press <b>R</b> to replay it from 1054.',
    facts: [
      ['Distance', '≈ 6,500 light-years (2.0 kpc)'], ['Constellation', 'Taurus'],
      ['Size', '≈ 12 × 8 light-years (6.5′ × 4.2′ on the sky)'], ['Expansion speed', '≈ 1,500 km/s; traced back, it converges on ~1130 — the pulsar wind is accelerating the shell'],
      ['Explosion', 'July 4, 1054'], ['Naked-eye visibility', 'by day: 23 days · by night: 653 days'],
      ['Pulsar', 'PSR B0531+21 · period 33.7 ms (29.6 Hz)'], ['Spin-down power', '≈ 4.5×10³¹ W (~10⁵ L☉)'],
      ['Filament mass', '≈ 7 M☉ (gas + ~0.03 M☉ of dust)'],
      ['Source images', 'Hubble 2005: NASA, ESA, J. Hester & A. Loll · Webb 2023: NASA, ESA, CSA, STScI, T. Temim (CC BY 4.0) · Chandra: NASA/CXC/SAO/F. Seward et al. · background stars: ESA/Gaia DR3'],
      ['3D model', 'Martin, Milisavljevic & Drissen 2021, MNRAS 502, 1864'],
    ],
  },
  camera: { distance: D0, polar: 90, azimuth: 0, min: 0.4, max: 90, fov: 19, target: [0, 0, 0] },
  exposure: 1.0,
  bloom: 0.15,
  params: [
    { key: 'mode', label: 'Ảnh', type: 'seg', options: MODES, default: 'hst',
      note: 'Hubble: ánh sáng khả kiến · Webb: hồng ngoại · Chandra: tia X của tinh vân gió pulsar.',
      onChange: (v, ctx) => ctx.toast(MODE_TOAST[v](), 6000),
      en: { label: 'Image', note: 'Hubble: visible light · Webb: infrared · Chandra: X-rays from the pulsar wind nebula.', options: ['Hubble (2005)', 'JWST (2023)', 'Chandra X-ray'] } },
    { key: 'year', label: 'Năm quan sát', type: 'range', min: 1060, max: 3000, step: 1, default: 2026, fmt: v => String(Math.round(v)),
      note: 'Kích thước ∝ tuổi (giãn nở gần như tự do). Pulsar chậm dần 4,2×10⁻¹³ s mỗi giây.',
      en: { label: 'Year observed', note: 'Size ∝ age (nearly free expansion). The pulsar slows by 4.2×10⁻¹³ s every second.' } },
    { key: 'fil', label: 'Sợi khí', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Khí từ lõi ngôi sao, bị gió pulsar đẩy thành ngón Rayleigh–Taylor.',
      en: { label: 'Filaments', note: 'Gas from the star’s core, pushed by the pulsar wind into Rayleigh–Taylor fingers.' } },
    { key: 'synch', label: 'Synchrotron', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Electron siêu tương đối tính xoắn quanh từ trường của pulsar.',
      en: { label: 'Synchrotron', note: 'Ultra-relativistic electrons spiralling around the pulsar’s magnetic field.' } },
    { key: 'rate', label: 'Nhịp pulsar (hiển thị)', type: 'range', min: 0.25, max: 30, step: 0.25, default: 1, fmt: v => vi(v, 2) + ' Hz',
      note: 'Thực tế 29,6 Hz — mỗi vòng quay có xung chính và xung phụ (cách 0,4 chu kỳ).',
      en: { label: 'Pulsar rate (display)', note: 'Really 29.6 Hz — each rotation gives a main pulse and an interpulse (0.4 of a period later).' } },
    { key: 'click', label: 'Âm thanh pulsar', type: 'toggle', default: false, note: 'Mỗi xung chính một tiếng tách.',
      en: { label: 'Pulsar sound', note: 'One click per main pulse.' } },
  ],
  actions: [
    { label: 'Tua lại từ 1054', key: 'r', run: ctx => { ctx.scene._replay = { t: 0 }; ctx.toast(t('Năm 1054 → nay: mảnh vỡ bay ~1 500 km/s, mỗi thế kỷ rộng thêm ~1 năm ánh sáng.', '1054 → today: the debris flies out at ~1,500 km/s, growing ~1 light-year wider every century.')); },
      en: { label: 'Replay from 1054' } },
    { label: 'Nhịp thật 30 Hz', key: 'j', run: ctx => {
      ctx.params.rate = ctx.params.rate >= 29 ? 1 : 29.6;
      ctx.toast(ctx.params.rate > 29 ? t('Tốc độ thật: 29,6 vòng/giây. Sao neutron nặng ~1,4 M☉, đường kính ~20 km.', 'Real speed: 29.6 rotations per second. The neutron star holds ~1.4 M☉ in a ball ~20 km across.') : t('Đã làm chậm lại để dễ quan sát.', 'Slowed down so it is easier to follow.'));
    }, en: { label: 'Real 30 Hz rate' } },
    { label: 'Nhìn từ Trái Đất', key: 'e', run: ctx => { ctx.scene._go = 'earth'; }, en: { label: 'View from Earth' } },
    { label: 'Cận cảnh sợi khí', key: 'c', run: ctx => {
      ctx.scene._go = 'close';
      ctx.toast(t('Cận cảnh rìa phía đông: mỗi sợi dày ~0,1–0,3 năm ánh sáng, là ngón Rayleigh–Taylor khi bong bóng gió pulsar nhẹ đẩy vào lớp vỏ nặng hơn.', 'Close-up of the eastern edge: each filament is ~0.1–0.3 light-years thick, a Rayleigh–Taylor finger formed where the light pulsar-wind bubble pushes into the heavier shell.'), 7000);
    }, en: { label: 'Filament close-up' } },
    { label: 'Nhìn ngang', key: 'n', run: ctx => { ctx.scene._fly = [D0, 80, 90]; ctx.toast(t('Nhìn vuông góc với hướng Trái Đất: lồng sợi là một vỏ dày, sợi gần và xa tách nhau.', 'Seen at right angles to our line of sight: the filament cage is a thick shell, with near and far filaments pulled apart.')); },
      en: { label: 'Side view' } },
  ],
  create(ctx) {
    const lut = makeLut(), geo = makeGeo();
    const uniforms = {
      ...ctx.shared,
      uFil: { value: null }, uAux: { value: null }, uDust: { value: null }, uGeo: { value: geo }, uOut: { value: null }, uLut: { value: lut },
      uMode: { value: 0 }, uSynCol: { value: new THREE.Vector3(0.52, 0.88, 1) },
      uAuxSel: { value: new THREE.Vector3(1, 0, 0) }, uDustSel: { value: new THREE.Vector3(1, 0, 0) },
      uSc: { value: 1 }, uFilK: { value: 1 }, uSynK: { value: 1 }, uBright: { value: 1 }, uPulse: { value: 0 }, uWisp: { value: 0 },
      uDetail: { value: 1 }, uHMean: { value: 4 }, uSpin: { value: SPIN.clone() },
    };
    const material = new THREE.ShaderMaterial({
      uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
      defines: { STEPS: stepsFor(ctx.quality) },
    });
    const setQuality = q => { material.defines.STEPS = stepsFor(q); material.needsUpdate = true; };

    // star sprites
    const starU = { uPxScale: { value: 500 }, uMode: { value: 0 }, uGain: { value: 1 } };
    const starGeo = new THREE.BufferGeometry();
    const stars = new THREE.Points(starGeo, new THREE.ShaderMaterial({
      uniforms: starU, vertexShader: STAR_VERT, fragmentShader: STAR_FRAG,
      blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
    }));
    stars.frustumCulled = false;
    const overlay = new THREE.Group(); overlay.add(stars);

    let tex = {}, meta = null;
    const ready = Promise.all([
      retry(loadTex, ASSET('crab_hst.webp')), retry(loadTex, ASSET('crab_jwst.webp')),
      retry(loadTex, ASSET('crab_aux.webp')), retry(loadTex, ASSET('crab_dust.webp')),
      fetch(ASSET('crab_meta.json')).then(r => r.json()),
    ]).then(([hst, jwst, aux, dust, m]) => {
      tex = { hst, jwst }; meta = m;
      uniforms.uAux.value = aux; uniforms.uDust.value = dust;
      const ol = makeOutline(m.R);
      uniforms.uOut.value = ol.tex; uniforms.uHMean.value = ol.Hmean;
      // star sprites: Gaia DR3 field stars (G < 20.8 within 9′): colour from BP−RP (observed, i.e.
      // reddened like in the photographs), flux ∝ 10^(−0.4 G), slightly wider PSF for bright stars
      const n = m.stars.length, pos = new Float32Array(n * 3), col = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        const [X, Y, G, c] = m.stars[i];
        const x = X * LY_PER_AS, y = Y * LY_PER_AS;
        const k = 2.5 + 8 * ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1;   // 50–340 ly behind
        pos.set([x * k, y * k, D0 * (1 - k)], i * 3);
        const T = Math.min(Math.max(5040 / (0.4929 + 0.5092 * c - 0.0353 * c * c), 2600), 25000);
        const rgb = ctx.bbColor(T);
        const f = Math.pow(10, -0.4 * (G - 18));
        const sigAs = 0.11 * (1 + 0.25 * Math.max(0, 18.5 - G));
        const peak = Math.min(10 * f * (0.11 / sigAs) ** 2, 40);
        const w = 0.45;                                                         // photographs: whitish cores
        col.set([(rgb[0] * (1 - w) + w) * peak, (rgb[1] * (1 - w) + w) * peak, (rgb[2] * (1 - w) + w) * peak, sigAs * LY_PER_AS / D0], i * 4);
      }
      starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      starGeo.setAttribute('aCol', new THREE.BufferAttribute(col, 4));
    });

    const flyer = makeFlyer();
    // camera moves that also shift the orbit target (the close-up looks at a patch of the shell)
    let move = null, dbgView = new URLSearchParams(location.search).get('crabView');
    const goTo = (ctx, where, instant = false) => {
      const t1 = where === 'close' ? CLOSE_T.clone() : new THREE.Vector3();
      const c1 = where === 'close' ? CLOSE_T.clone().add(CLOSE_V) : new THREE.Vector3(0, 0, D0);
      ctx.setFov(where === 'close' ? 34 : 19);
      if (instant) { ctx.controls.target.copy(t1); ctx.camera.position.copy(c1); ctx.camera.lookAt(t1); return; }
      move = { t: 0, t0: ctx.controls.target.clone(), t1, c0: ctx.camera.position.clone(), c1 };
    };
    const sz = new THREE.Vector2();
    let phase = 0.37, wisp = 0, mode = null;
    return {
      material, overlay, ready, setQuality,

      update(dt, ctx) {
        const p = ctx.params, def = ctx.scene;
        if (dbgView) { goTo(ctx, dbgView, true); dbgView = null; }      // ?crabView=close (snapshot checks)
        if (def._go) { goTo(ctx, def._go); def._go = null; }
        if (def._fly) { flyer.to(ctx, ...def._fly); def._fly = null; }
        if (move) {
          move.t += dt; const k = Math.min(1, move.t / 2.6), e = k * k * (3 - 2 * k);
          ctx.controls.target.lerpVectors(move.t0, move.t1, e);
          ctx.camera.position.lerpVectors(move.c0, move.c1, e);
          ctx.camera.lookAt(ctx.controls.target);
          if (k >= 1) move = null;
        } else flyer.update(ctx, dt);
        const rp = def._replay;
        if (rp) {
          rp.t += dt;
          const k = Math.min(1, rp.t / 9);
          p.year = Math.round(1060 + (2026 - 1060) * (1 - Math.pow(1 - k, 2)));
          if (k >= 1) def._replay = null;
        }
        if (p.mode !== mode && tex.hst) {
          mode = p.mode;
          const m = mode === 'jwst' ? 1 : mode === 'xray' ? 2 : 0;
          uniforms.uMode.value = m;
          uniforms.uFil.value = m === 1 ? tex.jwst : tex.hst;
          uniforms.uAuxSel.value.set(m === 0 ? 1 : 0, m === 1 ? 1 : 0, m === 2 ? 1 : 0);
          uniforms.uDustSel.value.set(m === 0 ? 1 : 0, m === 1 ? 1 : 0, 0);
          const c = m === 1 ? meta.synJWST : meta.synHST;
          uniforms.uSynCol.value.set(c[0], c[1], c[2]);
          starU.uMode.value = m;
          stars.visible = m !== 2;
        }
        const age = Math.max(p.year - 1054, 6);
        uniforms.uSc.value = age / 972;
        // younger: denser filaments and a more powerful pulsar wind (spin-down L ∝ (1 + t/τ)⁻², τ ≈ 700 yr)
        const tau = 700, Lw = ((1 + 972 / tau) / (1 + age / tau)) ** 2;
        uniforms.uBright.value = Math.min(Math.pow(972 / age, 0.5), 4);
        uniforms.uFilK.value = p.fil;
        uniforms.uSynK.value = p.synch * Math.pow(Lw, 0.5);
        // pulse profile: main pulse + interpulse 0.4 cycles later (Crab optical / X-ray profile)
        const prev = phase;
        phase = (phase + dt * p.rate) % 1;
        const g = (x, w) => Math.exp(-0.5 * (x / w) ** 2);
        const dph = x => { const d = phase - x; return d - Math.round(d); };
        const w = p.rate > 10 ? 0.06 : 0.025;      // at the real rate a frame averages over the pulse
        uniforms.uPulse.value = p.rate > 10 ? 0.35 + 0.3 * Math.random() : g(dph(0), w) + 0.55 * g(dph(0.4), w * 1.4);
        if (p.click && ctx.audio.started && (phase < prev || (p.rate > 10 && Math.random() < 0.5))) ctx.audio.click(0.22, 0, 0.8);
        // wisps really move outward at ~0.5 c (a few months to cross the torus); here a time-lapse
        wisp = (wisp + dt * 0.18) % 1000;
        uniforms.uWisp.value = wisp;
        ctx.renderer.getDrawingBufferSize(sz);
        starU.uPxScale.value = sz.y * ctx.camera.projectionMatrix.elements[5] / 2;
      },
      hud(ctx) {
        const p = ctx.params, age = Math.max(p.year - 1054, 0);
        const d = ctx.camera.position.distanceTo(ctx.controls.target);
        const P = 33.7 + 4.21e-13 * (p.year - 2026) * 3.156e7 * 1e3;          // ms
        const Edot = 4 * Math.PI ** 2 * 1e38 * 4.21e-13 / (P * 1e-3) ** 3;     // W, I = 10³⁸ kg m²
        return [
          [t('Camera cách pulsar', 'Camera to pulsar'), ly(d)],
          [t('Tuổi tàn dư', 'Remnant age'), Math.round(age).toLocaleString(locale()) + t(' năm', ' years')],
          [t('Kích thước (dài × ngắn)', 'Size (long × short)'), vi(12.2 * age / 972) + ' × ' + ly(7.9 * age / 972)],
          [t('Chu kỳ pulsar năm ', 'Pulsar period in ') + Math.round(p.year), vi(P, 2) + ' ms (' + vi(1000 / P, 1) + ' Hz)'],
          [t('Công suất hãm quay', 'Spin-down power'), vi(Edot / 1e31, 1) + '×10³¹ W'],
        ];
      },
    };
  },
};

export default [crab];
