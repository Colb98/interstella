// Thiên hà — galaxies.
//
// spiral-galaxy: a Milky-Way-like barred spiral (SBbc). One scene unit = 1 kpc.
//
// Construction (Lindblad / Kalnajs kinematic density wave, held rigid as a Lin–Shu quasi-stationary
// wave when the dark halo is present):
//   * every disk star has a guiding radius r0 and moves on a slightly non-circular orbit
//       r = r0 (1 − ε cos Ψ),   Ψ = m (θ − Φ(r0)),   θ = θ0 + Ω(r0) t
//     whose orientation Φ(r0) turns with radius as a logarithmic spiral of pitch p. Neighbouring
//     orbits crowd where dr/dr0 is smallest, which is where the arms appear, while each star keeps
//     circulating at its own Ω(r0): stars flow *through* the arms (density wave, not material arms).
//   * the pattern turns rigidly at Ω_p (one pattern speed for bar and arms, corotation ≈ 7.9 kpc).
//   * young stars and H II regions are born at the arm shock (just upstream of the stellar crest,
//     where the dust lane is) and drift downstream at Ω − Ω_p while they age and fade.
//   * rotation curve: visible mass (Hernquist bulge 0.9×10¹⁰ M☉, a = 0.5 kpc + exponential disk
//     5×10¹⁰ M☉, R_d = 2.6 kpc, enclosed mass treated spherically) ± a pseudo-isothermal dark halo
//     (v∞ = 245 km/s, r_c = 5 kpc) → flat ≈ 220 km/s at the Sun with it, Keplerian fall-off without.
//   * WITHOUT the dark halo we drop the self-gravity that keeps the wave rigid and let each orbit's
//     alignment precess at the kinematic rate Ω − κ/2 of the visible-mass curve (Lindblad). That rate
//     varies strongly with radius, so the arms wind up — a pedagogical caricature (a halo-less disk
//     is in reality bar-unstable, Ostriker & Peebles 1973), noted as such.
// All motion is evaluated in the vertex shader from two time accumulators (time spent with / without
// dark matter), so switching the rotation curve is continuous and the CPU does nothing per star.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

// ---------------------------------------------------------------------------------------------
// physics constants shared by JS and GLSL
const KMS = 1.0227e-3;          // 1 km/s/kpc in rad/Myr
const GN = 4.30091e-6;          // G in kpc (km/s)² / M☉
const RB = 4.5;                 // bar half-length, kpc (MW long bar ≈ 5 kpc)
const OMP = 28 * KMS;           // pattern speed, rad/Myr
const R_SUN = 8.2;
// headless snapshot mode (tools/snap.sh): start at the requested t= instead of t = 0
const SNAP = typeof location !== 'undefined' && /[?&]snap=1/.test(location.search);
const TAB_N = 64, TAB_DR = 0.4; // lookup table r = 0 … 25.6 kpc

function vVis2(r) {
  r = Math.max(r, 0.02);
  const mb = 0.9e10 * r * r / ((r + 0.5) * (r + 0.5));
  const x = r / 2.6, md = 5e10 * (1 - (1 + x) * Math.exp(-x));
  return GN * (mb + md) / r;
}
function vHalo2(r) { r = Math.max(r, 0.02); return 245 * 245 * (1 - 5 / r * Math.atan(r / 5)); }
const vDM = r => Math.sqrt(vVis2(r) + vHalo2(r));
const vKep = r => Math.sqrt(vVis2(r));
const omDM = r => vDM(r) / Math.max(r, 0.05) * KMS;
const omKep = r => vKep(r) / Math.max(r, 0.05) * KMS;
// Ω − κ/2 of the visible-mass curve (rad/Myr): precession rate of a nearly circular orbit
function precK(r) {
  r = Math.max(r, 0.05);
  const h = 0.01 * r, v = vKep(r), dv = (vKep(r + h) - vKep(r - h)) / (2 * h), om = v / r;
  return (om - 0.5 * Math.sqrt(Math.max(2 * om * (om + dv), 0))) * KMS;
}
const dPrec = r => precK(Math.max(r, RB)) - precK(RB);

// small deterministic PRNG so every load (and every snapshot) is the same galaxy
function rng(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

// ---------------------------------------------------------------------------------------------
// GLSL: rotation curve + pattern, shared by the particle shaders and the fullscreen glow
const ROT_GLSL = /* glsl */`
#ifndef TAU
#define PI 3.14159265359
#define TAU 6.28318530718
#endif
uniform float uTdm, uTkep, uMixK, uM, uTanP, uOmP, uBar0, uRb, uDelta;
uniform vec4 uTab[${TAB_N + 1}];   // (Ω − κ/2)_vis(r) − same at R_bar, r·d/dr of that, Ω_dm, Ω_kep  [rad/Myr]
#define TAB_DR ${TAB_DR.toFixed(3)}
vec4 tab(float r){
  float x = clamp(r / TAB_DR, 0.0, ${TAB_N - 1}.999);
  int i = int(x);
  return mix(uTab[i], uTab[i + 1], x - float(i));
}
float totalT(){ return uTdm + uTkep; }
float barAngle(){ return uBar0 + uOmP * totalT(); }
float psiStar0(){ return atan(-uM / uTanP, 1.0); }
// orientation of the orbit family with guiding radius r (log spiral outside the bar, aligned inside)
// smooth max(0, ln(r/R_bar)) so the orbit family turns from bar-aligned to spiral without a kink
float lnSoft(float r){ float l = log(max(r, 0.05) / uRb); return 0.5 * (l + sqrt(l * l + 0.02)); }
float rLnSoft(float r){ float l = log(max(r, 0.05) / uRb); return 0.5 * (1.0 + l / sqrt(l * l + 0.02)); }
float phiPat(float r, vec4 tb){
  return barAngle() - psiStar0() / uM - lnSoft(r) / uTanP + tb.x * uTkep;
}
// r dΦ/dr — sets where neighbouring orbits crowd (the arm crest) and how tightly the arms are wound
float rPhiP(float r, vec4 tb){ return -rLnSoft(r) / uTanP + tb.y * uTkep; }
float psiStar(float r, vec4 tb){ return atan(uM * rPhiP(r, tb), 1.0); }
float epsOrb(float r){
  return 0.75 * uTanP / uM * smoothstep(0.75 * uRb, 1.05 * uRb, r) * (1.0 - 0.6 * smoothstep(13.0, 20.0, r));
}
mat2 rot2(float a){ float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
`;

// ---------------------------------------------------------------------------------------------
// star particles: position fully computed on the GPU
const STAR_VERT = /* glsl */`
precision highp float;
${ROT_GLSL}
attribute vec4 aOrb;   // r0 (kpc), θ0 (or birth phase scatter for young), z amplitude, z phase
attribute vec4 aPop;   // type, lifetime (Myr), age offset 0..1, arm index 0..3
attribute vec4 aCol;   // linear rgb × luminance, sprite size (kpc)
uniform float uPxScale, uMinPx, uGain;
varying vec3 vCol;
varying float vSharp;
void main(){
  float T = totalT();
  float r0 = aOrb.x;
  vec4 tb = tab(r0);
  float AO = tb.z * uTdm + tb.w * uTkep;           // angle swept since t = 0
  int type = int(aPop.x + 0.5);
  vec3 p; float b = 1.0, sz = aCol.w;
  if (type == 0){                                   // old disk
    float th = aOrb.y + AO;
    float psi = uM * (th - phiPat(r0, tb));
    float r = r0 * (1.0 - 0.7 * epsOrb(r0) * cos(psi));
    p = vec3(r * cos(th), aOrb.z * sin(aOrb.w + 0.085 * T), r * sin(th));
  } else if (type == 1){                            // bar: x1 orbits, closed in the bar frame
    float bar = barAngle();
    float t = aOrb.y + AO - uOmP * T;
    vec2 q = vec2(r0 * cos(t), r0 * 0.40 * sin(t));
    q = rot2(-bar) * q;
    p = vec3(q.x, aOrb.z * sin(aOrb.w + 0.2 * T), q.y);
  } else if (type == 2){                            // bulge: slowly rotating spheroid
    float th = aOrb.y + 0.6 * AO;
    p = vec3(r0 * cos(th), aOrb.z * sin(aOrb.w + 0.25 * T), r0 * sin(th));
  } else if (type == 3){                            // halo & globular clusters: essentially non-rotating
    p = vec3(r0 * cos(aOrb.y), aOrb.z, r0 * sin(aOrb.y));
  } else {                                          // 4 young OB stars, 5 H II regions
    float tau = aPop.y;
    float age = mod(T + aPop.z * tau, tau);
    float om = mix(tb.z, tb.w, uMixK);
    float drift = om - (uOmP + uMixK * tb.x);       // angular speed relative to the local pattern
    float side = tanh(drift / (3.0e-3));            // upstream side flips at corotation
    float psi = psiStar(r0, tb) + side * 0.15 + aOrb.y + uM * drift * age;
    float th = phiPat(r0, tb) + (psi + TAU * mod(aPop.w, uM)) / uM;
    float r = r0 * (1.0 - 0.45 * epsOrb(r0) * cos(psi));
    p = vec3(r * cos(th), aOrb.z * sin(aOrb.w + 0.1 * T), r * sin(th));
    float a = age / tau;
    if (type == 4) b = smoothstep(0.0, 0.05, a) * pow(1.0 - a, 1.3);
    else { b = smoothstep(0.0, 0.12, a) * (1.0 - a) * (1.0 - a); sz *= 0.55 + 0.9 * a; }
  }
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float d = max(-mv.z, 1e-3);
  float px = sz * uPxScale / d;
  float ps = clamp(px, uMinPx, 110.0);
  gl_PointSize = ps;
  vCol = aCol.rgb * b * uGain * min(px * px / (ps * ps), 4.0);
  vSharp = type == 5 ? 2.2 : 3.6;
}`;
const STAR_FRAG = /* glsl */`
precision highp float;
varying vec3 vCol;
varying float vSharp;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = min(dot(c, c), 1.0);
  gl_FragColor = vec4(vCol * exp(-r2 * vSharp) * (1.0 - r2) * (1.0 - r2), 1.0);
}`;

// dust clouds: multiplicative absorption sprites (transmission = exp(−τ), reddening ∝ λ⁻¹)
const DUST_VERT = /* glsl */`
precision highp float;
${ROT_GLSL}
attribute vec4 aOrb;   // r0 | bar-frame x;  phase scatter | bar-frame y;  z;  arm index
attribute vec2 aDust;  // type (0 arm lane, 1 bar lane, 2 flocculent), sprite size kpc
attribute float aTau;
uniform float uPxScale, uMinPx, uDust;
varying float vTau;
void main(){
  int type = int(aDust.x + 0.5);
  float r0 = aOrb.x;
  vec3 p;
  if (type == 1){
    vec2 q = rot2(-barAngle()) * aOrb.xy;
    p = vec3(q.x, aOrb.z, q.y);
  } else {
    vec4 tb = tab(r0);
    float th, psi;
    if (type == 0){
      float om = mix(tb.z, tb.w, uMixK);
      float side = tanh((om - (uOmP + uMixK * tb.x)) / 3.0e-3);
      psi = psiStar(r0, tb) - side * uDelta + aOrb.y;
      th = phiPat(r0, tb) + (psi + TAU * mod(aOrb.w, uM)) / uM;
    } else {
      th = aOrb.y + tb.z * uTdm + tb.w * uTkep;
      psi = uM * (th - phiPat(r0, tb));
    }
    float r = r0 * (1.0 - 0.45 * epsOrb(r0) * cos(psi));
    p = vec3(r * cos(th), aOrb.z, r * sin(th));
  }
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float d = max(-mv.z, 1e-3);
  float px = aDust.y * uPxScale / d;
  float ps = clamp(px, uMinPx, 160.0);
  gl_PointSize = ps;
  // fade sprites that fill the screen (camera inside the cloud) instead of blacking out the view
  // Dust is mixed with the stars, not a foreground screen: seen edge-on, half of the light comes from
  // stars in front of the lane. With no depth buffer we mimic that by softening grazing sightlines.
  float mu = abs(normalize(cameraPosition - p).y);
  vTau = aTau * uDust * min(px * px / (ps * ps), 1.0) * (1.0 - smoothstep(60.0, 160.0, px)) * mix(0.45, 1.0, sqrt(mu));
}`;
const DUST_FRAG = /* glsl */`
precision highp float;
varying float vTau;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = min(dot(c, c), 1.0);
  float t = vTau * exp(-r2 * 2.0) * (1.0 - r2);
  gl_FragColor = vec4(1.0 - exp(-t * vec3(0.62, 0.80, 1.0)), 1.0);
}`;

// fullscreen pass: extragalactic background + the smooth light of unresolved stars
const GAL_FRAG = COMMON_GLSL + ROT_GLSL + /* glsl */`
uniform float uGlow;
// faint distant galaxies on the cube-mapped sky
vec3 farGalaxies(vec3 dir){
  vec3 a = abs(dir); float face; vec2 uv;
  if (a.x >= a.y && a.x >= a.z){ face = dir.x > 0.0 ? 0.0 : 1.0; uv = dir.yz / a.x; }
  else if (a.y >= a.z){ face = dir.y > 0.0 ? 2.0 : 3.0; uv = dir.xz / a.y; }
  else { face = dir.z > 0.0 ? 4.0 : 5.0; uv = dir.xy / a.z; }
  vec2 g = uv * 34.0; vec2 id = floor(g); vec2 f = g - id;
  vec3 h = hash33(vec3(id, face * 7.3 + 91.0));
  if (h.x > 0.16) return vec3(0.0);
  vec3 h2 = hash33(vec3(id.yx, face + 5.0));
  vec2 dd = (f - 0.3 - 0.4 * h.yz) * rot2(h2.x * PI);
  float e = 0.25 + 0.7 * h2.y;
  float r2 = (dd.x * dd.x + dd.y * dd.y / (e * e)) / (0.0009 + 0.002 * h2.z);
  return mix(vec3(1.0, 0.8, 0.6), vec3(0.7, 0.8, 1.0), step(0.6, h2.z)) * exp(-r2) * 0.05;
}
vec2 slab(float oy, float dy, float H){
  if (abs(dy) < 1e-5) return abs(oy) < H ? vec2(-1e9, 1e9) : vec2(1e9, -1e9);
  float t0 = (-H - oy) / dy, t1 = (H - oy) / dy;
  return vec2(min(t0, t1), max(t0, t1));
}
vec2 cylinder(vec3 ro, vec3 rd, float R){
  float a = dot(rd.xz, rd.xz), b = dot(ro.xz, rd.xz), c = dot(ro.xz, ro.xz) - R * R;
  if (a < 1e-8) return c < 0.0 ? vec2(-1e9, 1e9) : vec2(1e9, -1e9);
  float h = b * b - a * c;
  if (h < 0.0) return vec2(1e9, -1e9);
  h = sqrt(h); return vec2((-b - h) / a, (-b + h) / a);
}
void main(){
  vec3 rd = camRay(), ro = uCamPos;
  vec3 col = uSkyMode < 0.5
    ? (starLayer(rd, 40.0, 0.05, 0.07, 1.0, 1.0) * 0.45 + starLayer(rd, 110.0, 0.06, 0.09, 2.0, 1.0) * 0.18) * uStarBright + farGalaxies(rd) * uSkyBright
    : sky(rd, 1.0) * 0.5;
  float jit = hash13(vec3(gl_FragCoord.xy, 3.1));
  vec3 warm = bb(4600.0), cool = bb(9500.0), core = bb(3900.0);

  // --- thin + thick disk with the old-star arm response and the blue light of young stars ---
  vec2 ts = slab(ro.y, rd.y, 1.6), tc = cylinder(ro, rd, 21.0);
  float t0 = max(max(ts.x, tc.x), 0.0), t1 = min(ts.y, tc.y);
  vec3 disk = vec3(0.0);
  if (t1 > t0){
    float dl = (t1 - t0) / 28.0;
    for (int i = 0; i < 28; i++){
      vec3 p = ro + rd * (t0 + (float(i) + jit) * dl);
      float R = length(p.xz);
      vec4 tb = tab(R);
      float vert = exp(-abs(p.y) / 0.3) / 0.6 + 0.12 * exp(-abs(p.y) / 0.9) / 1.8;
      float sig = exp(-R / 2.6) * (1.0 - 0.85 * exp(-R * R / 16.0));
      float psi = uM * (atan(p.z, p.x) - phiPat(R, tb));
      float e = epsOrb(R);
      float crowd = 1.0 / max(1.0 - 0.7 * e * (cos(psi) + uM * rPhiP(R, tb) * sin(psi)), 0.25);
      float dpsi = mod(psi - psiStar(R, tb) - 0.6 + PI, TAU) - PI;
      float young = exp(-dpsi * dpsi * 2.0) * smoothstep(0.8 * uRb, 1.2 * uRb, R) * exp(-R / 4.0) * (1.0 - smoothstep(11.0, 16.0, R));
      disk += (warm * sig * crowd + cool * young * 1.1) * vert * dl;
    }
  }
  // --- bulge + bar ---
  vec3 bul = vec3(0.0);
  vec2 tbs = sphereHit(ro, rd, vec3(0.0), 6.0);
  t0 = max(tbs.x, 0.0); t1 = tbs.y;
  if (t1 > t0){
    float dl = (t1 - t0) / 36.0;
    mat2 rb = rot2(barAngle());
    for (int i = 0; i < 36; i++){
      vec3 p = ro + rd * (t0 + (float(i) + jit) * dl);
      vec2 q = rb * p.xz;                         // bar frame
      float s = length(vec3(p.x, p.y / 0.7, p.z));
      float rhoB = exp(-s / 0.3) / (s + 0.06) * 0.3 + 0.2 * exp(-s / 0.8);   // nucleus + classical bulge
      vec2 qa = abs(q) / vec2(uRb, 0.36 * uRb);
      float sxy = pow(pow(qa.x, 2.5) + pow(qa.y, 2.5), 0.4);   // mildly boxy isophotes
      float hz = 0.3 + 0.5 * smoothstep(0.1, 0.7, qa.x) * (1.0 - 0.5 * smoothstep(0.7, 1.1, qa.x));   // boxy/peanut
      float rhoBar = exp(-sxy * sxy * 2.0) * exp(-abs(p.y) / hz) * 1.1;
      bul += (core * rhoB + warm * rhoBar) * dl;
    }
  }
  col += (disk * 0.55 + bul * 0.45) * uGlow;
  gl_FragColor = vec4(col, 1.0);
}`;

// ---------------------------------------------------------------------------------------------
function buildStars(ctx) {
  const R = rng(20251017);
  const bb = T => ctx.bbColor(T);
  const old = [], young = [];
  const push = (arr, r0, a, zA, zP, type, tau, off, k, col, lum, size) =>
    arr.push(r0, a, zA, zP, type, tau, off, k, col[0] * lum, col[1] * lum, col[2] * lum, size);
  const gauss = () => { let u = 0; for (let i = 0; i < 4; i++) u += R(); return (u - 2) * 1.732; };
  // old thin + thick disk: exponential surface density R e^(−R/R_d)
  for (let i = 0; i < 140000; i++) {
    let r; do r = -2.6 * Math.log(R() * R() + 1e-9); while (r > 19.5 || r < 0.4 || (r < RB && R() < 0.55 * (1 - r / RB)));  // the bar has swept up the inner disk
    const thick = R() < 0.12;
    const zA = (thick ? 0.9 : 0.28) * Math.abs(gauss()) * (0.7 + 0.04 * r);
    const u = R();
    const T = 3500 + 4200 * Math.pow(u, 2.2);
    const lum = (u > 0.985 ? 6 : 1) * (0.5 + R()) * 0.3;
    push(old, r, R() * 6.2832, zA, R() * 6.2832, 0, 0, 0, 0, bb(T), lum, 0.022);
  }
  // bar
  for (let i = 0; i < 32000; i++) {
    const r = RB * Math.pow(R(), 0.8) * 0.97 + 0.15;
    const zA = (0.12 + 0.32 * (r / RB)) * Math.abs(gauss()) * 0.8;   // boxy / peanut thickening toward the ends
    push(old, r, R() * 6.2832, zA, R() * 6.2832, 1, 0, 0, 0, bb(3700 + 1300 * R()), (0.6 + R()) * 0.35, 0.022);
  }
  // classical bulge
  for (let i = 0; i < 26000; i++) {
    let s; do s = -0.45 * Math.log(R() * R() + 1e-9); while (s > 3.2);
    const ct = 2 * R() - 1, st = Math.sqrt(1 - ct * ct);
    push(old, s * st, R() * 6.2832, s * Math.abs(ct) * 0.7 + 0.03, R() * 6.2832, 2, 0, 0, 0, bb(3600 + 1200 * R()), (0.6 + R()) * 0.3, 0.02);
  }
  // stellar halo (ρ ∝ r^-3.5) and globular clusters
  for (let i = 0; i < 6000; i++) {
    const s = 1.5 / Math.pow(Math.max(R(), 0.002), 1 / 2.5) - 0.5;
    if (s > 45) { i--; continue; }
    const ct = 2 * R() - 1, st = Math.sqrt(1 - ct * ct);
    push(old, s * st, R() * 6.2832, s * ct * 0.8, 0, 3, 0, 0, 0, bb(4300 + 1500 * R()), (0.5 + R()) * 1.5, 0.03);
  }
  for (let i = 0; i < 160; i++) {
    const s = 1 + 30 * Math.pow(R(), 2.2);
    const ct = 2 * R() - 1, st = Math.sqrt(1 - ct * ct);
    push(old, s * st, R() * 6.2832, s * ct, 0, 3, 0, 0, 0, bb(5200), 6, 0.05);
  }
  // shuffle so that the "star count" draw range is a fair subsample
  shuffle(old, 12, R);
  // young OB associations and H II complexes, born on the arm shock
  const youngRadius = () => { let r; do r = RB * 0.95 + 11 * R(); while (R() > (r / 6.5) * Math.exp(1 - r / 6.5) * (1 - smoothstep(11, 15.5, r))); return r; };
  for (let c = 0; c < 2400; c++) {
    const r = youngRadius(), off = R(), k = Math.floor(R() * 4), sc = 0.5 * gauss(), tau = 20 + 60 * R();
    const n = 5 + Math.floor(Math.pow(R(), 2) * 30), spread = 0.06 + 0.12 * R();
    for (let j = 0; j < n; j++) {
      const T = 12000 + 28000 * Math.pow(R(), 2);
      push(young, r + spread * gauss(), sc + spread * 0.3 * gauss(), 0.1 * Math.abs(gauss()), R() * 6.28, 4, tau, off, k, bb(T), (0.5 + 3 * Math.pow(R(), 3)) * 6, 0.022);
    }
  }
  // intermediate-age A/F stars (0.2–0.5 Gyr): born in the arms long ago, now spread far downstream
  for (let i = 0; i < 40000; i++) {
    const r = youngRadius();
    push(young, r, 0.6 * gauss(), 0.2 * Math.abs(gauss()), R() * 6.28, 4, 150 + 300 * R(), R(), Math.floor(R() * 4), bb(7000 + 6000 * R()), (0.5 + R()) * 1.6, 0.022);
  }
  const hii = [1.0, 0.24, 0.42];   // Hα 656 nm + Hβ / [O II]: the pink of star-forming regions
  for (let c = 0; c < 700; c++) {
    const r = youngRadius(), off = R(), k = Math.floor(R() * 4), sc = 0.4 * gauss(), tau = 8 + 10 * R();
    const n = 4 + Math.floor(Math.pow(R(), 2) * 22), spread = 0.04 + 0.08 * R();
    for (let j = 0; j < n; j++)
      push(young, r + spread * gauss(), sc + spread * 0.25 * gauss(), 0.05 * Math.abs(gauss()), R() * 6.28, 5, tau, off, k, hii, (0.6 + R()) * 2.2, 0.05 + 0.07 * R());
    // the exciting O star in the middle of the nebula
    push(young, r, sc, 0.02, 0, 4, tau, off, k, bb(38000), 12, 0.02);
  }
  shuffle(young, 12, R);
  return { old: toGeom(old), young: toGeom(young), nOld: old.length / 12, nYoung: young.length / 12 };
}
function smoothstep(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function shuffle(arr, stride, R) {
  const n = arr.length / stride, tmp = new Array(stride);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(R() * (i + 1));
    for (let s = 0; s < stride; s++) { tmp[s] = arr[i * stride + s]; arr[i * stride + s] = arr[j * stride + s]; arr[j * stride + s] = tmp[s]; }
  }
}
function toGeom(arr) {
  const n = arr.length / 12;
  const orb = new Float32Array(n * 4), pop = new Float32Array(n * 4), col = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) for (let s = 0; s < 4; s++) {
    orb[i * 4 + s] = arr[i * 12 + s]; pop[i * 4 + s] = arr[i * 12 + 4 + s]; col[i * 4 + s] = arr[i * 12 + 8 + s];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));   // unused, required by three
  g.setAttribute('aOrb', new THREE.BufferAttribute(orb, 4));
  g.setAttribute('aPop', new THREE.BufferAttribute(pop, 4));
  g.setAttribute('aCol', new THREE.BufferAttribute(col, 4));
  return g;
}
function buildDust() {
  const R = rng(777);
  const gauss = () => { let u = 0; for (let i = 0; i < 4; i++) u += R(); return (u - 2) * 1.732; };
  const orb = [], dd = [], tau = [];
  const expz = h => (R() < 0.5 ? -1 : 1) * h * -Math.log(1 - 0.98 * R());   // exponential vertical profile: soft lane edges
  // arm lanes: clumpy clouds along the upstream (shocked) edge of each arm, with feathers / spurs
  for (let c = 0; c < 2400; c++) {
    let r; do r = RB * 0.9 + 13 * R(); while (R() > Math.exp(-(r - RB) / 7));
    const k = Math.floor(R() * 4), sc = 0.22 * gauss(), n = 3 + Math.floor(R() * 9);
    const str = Math.pow(R(), 1.5) * 1.6 + 0.2;
    const spur = R() < 0.3 ? (0.5 + R()) : 0;   // a feather sheared off downstream
    for (let j = 0; j < n; j++) {
      const f = R(), dr = 0.3 * gauss() + spur * f * 0.6;
      orb.push(r + dr, sc + 0.05 * gauss() + spur * f * 0.6, expz(R() < 0.1 ? 0.22 : 0.06), k);
      dd.push(0, 0.15 + 0.25 * R()); tau.push(0.42 * str * smoothstep(RB * 0.9, RB * 1.4, r) * Math.exp(-r / 12) * (spur ? 1 - 0.6 * f : 1));
    }
  }
  // straight dust lanes on the leading edges of the bar (gas flowing in toward the nuclear ring)
  for (let i = 0; i < 2600; i++) {
    const s = 0.12 + 0.88 * Math.pow(R(), 0.8), sign = R() < 0.5 ? -1 : 1;
    const x = s * RB * 0.95, y = 0.35 + 0.8 * s * s + 0.08 * gauss();
    orb.push(sign * x + 0.05 * gauss(), sign * y, expz(0.05), 0);
    dd.push(1, 0.07 + 0.08 * R()); tau.push(0.22 * (0.6 + R()));
  }
  // nuclear ring
  for (let i = 0; i < 500; i++) {
    const a = R() * 6.2832, r = 0.35 + 0.05 * gauss();
    orb.push(r * Math.cos(a), r * Math.sin(a) * 0.8, 0.03 * gauss(), 0); dd.push(1, 0.06 + 0.06 * R()); tau.push(0.12 * R());
  }
  // flocculent inter-arm dust (follows the stars, so it shears)
  for (let i = 0; i < 5000; i++) {
    let r; do r = 2.5 + 15 * R(); while (R() > Math.exp(-(r - 2.5) / 6));
    orb.push(r, R() * 6.2832, expz(R() < 0.15 ? 0.3 : 0.1), 0);
    dd.push(2, 0.15 + 0.3 * R()); tau.push(0.18 * R());
  }
  const n = tau.length;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute('aOrb', new THREE.BufferAttribute(new Float32Array(orb), 4));
  g.setAttribute('aDust', new THREE.BufferAttribute(new Float32Array(dd), 2));
  g.setAttribute('aTau', new THREE.BufferAttribute(new Float32Array(tau), 1));
  return g;
}
function labelTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 256;
  const g = c.getContext('2d');
  g.textAlign = 'center';
  g.fillStyle = 'rgba(4,6,14,0.8)';
  g.beginPath(); g.roundRect(106, 40, 300, 106, 18); g.fill();
  g.shadowColor = 'rgba(0,0,0,0.9)'; g.shadowBlur = 8;
  g.fillStyle = '#ffe9a8';
  g.font = '600 44px system-ui, "Segoe UI", sans-serif';
  g.fillText('Bạn ở đây', 256, 92);
  g.fillStyle = '#cfe0ff';
  g.font = '26px system-ui, "Segoe UI", sans-serif';
  g.fillText('Mặt Trời · 8,2 kpc', 256, 132);
  g.strokeStyle = '#ffe9a8'; g.lineWidth = 3;
  g.beginPath(); g.moveTo(256, 146); g.lineTo(256, 178); g.stroke();
  g.beginPath(); g.arc(256, 196, 14, 0, Math.PI * 2); g.stroke();
  g.fillStyle = '#ffffff'; g.beginPath(); g.arc(256, 196, 3.5, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ---------------------------------------------------------------------------------------------
const VIEWS = [['Nhìn chính diện', 2.5], ['Nghiêng 60°', 60], ['Nhìn cạnh', 89.4]];

const spiral = {
  id: 'spiral-galaxy',
  name: 'Thiên hà xoắn ốc',
  category: 'galaxies',
  order: 10,
  subtitle: 'Xoắn ốc có thanh ngang SBbc · kiểu Ngân Hà',
  swatch: ['#ffe2a6', '#20306a'],
  blurb: 'Nhánh xoắn không phải dải sao cố định mà là <b>sóng mật độ</b>: quỹ đạo elip của các sao xếp lệch dần theo bán kính tạo nên vùng dồn ứ mà sao và khí chảy xuyên qua như xe qua đoạn tắc đường — khí bị nén ở đó sinh ra sao xanh và vùng H II màu hồng. Vera Rubin đo được vận tốc quay <b>không giảm</b> ra tới tận rìa: phần lớn khối lượng là <b>vật chất tối</b> vô hình.',
  intro: 'Thử đổi <b>Đường cong quay</b> sang <b>Không có vật chất tối</b>: chỉ còn khối lượng nhìn thấy, sao ở rìa chậm hẳn lại theo kiểu Kepler và các nhánh bị xoắn chặt dần. Nhấn <b>V</b> để đổi góc nhìn, <b>G</b> để bay tới Mặt Trời.',
  facts: [
    ['Phân loại', 'SBbc (xoắn ốc có thanh)'],
    ['Đường kính đĩa', '≈ 30 kpc · 100 000 năm AS'],
    ['Số sao', '100–400 tỷ'],
    ['Khối lượng', '≈ 1,5×10¹² M☉ (≈ 95% tối)'],
    ['Mặt Trời cách tâm', '8,2 kpc'],
    ['Hố đen trung tâm', 'Sgr A* · 4,3×10⁶ M☉'],
  ],
  camera: { distance: 40, polar: 55, azimuth: 30, min: 3, max: 150, fov: 50 },
  exposure: 1.0,
  bloom: 0.55,
  params: [
    { key: 'darkMatter', label: 'Đường cong quay', type: 'seg', options: [[1, 'Có vật chất tối'], [0, 'Không có vật chất tối']], default: 1,
      note: 'Có quầng tối: v ≈ 220 km/s gần như không đổi tới rìa. Không có: v giảm dần theo Kepler từ khối lượng nhìn thấy (sao, khí).' },
    { key: 'speed', label: 'Tốc độ thời gian', type: 'range', min: 0, max: 40, step: 0.5, default: 6, fmt: v => v.toFixed(1) + ' triệu năm/s',
      note: 'Ở tốc độ thời gian chung mặc định. Một vòng quay của Mặt Trời (năm thiên hà) ≈ 230 triệu năm.' },
    { key: 'arms', label: 'Số nhánh', type: 'seg', options: [[2, '2 nhánh'], [4, '4 nhánh']], default: 2 },
    { key: 'pitch', label: 'Góc mở nhánh', type: 'range', min: 7, max: 25, step: 0.5, default: 15, fmt: v => v.toFixed(1) + '°',
      note: 'Ngân Hà ≈ 12–13°. Nhánh mở rộng là đặc trưng của loại Sc, nhánh cuốn chặt của loại Sa.' },
    { key: 'dust', label: 'Bụi', type: 'range', min: 0, max: 2, step: 0.05, default: 1 },
    { key: 'count', label: 'Số hạt sao', type: 'range', min: 0.15, max: 1, step: 0.05, default: 1, fmt: v => Math.round(v * 100) + '%' },
    { key: 'bright', label: 'Độ sáng sao', type: 'range', min: 0.2, max: 3, step: 0.05, default: 1 },
    { key: 'glow', label: 'Ánh sáng khuếch tán', type: 'range', min: 0, max: 2, step: 0.05, default: 1, note: 'Ánh sáng gộp của hàng tỷ sao không phân giải được.' },
    { key: 'marker', label: 'Đánh dấu Mặt Trời', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Đổi góc nhìn', key: 'v', title: 'Chính diện → nghiêng 60° → nhìn cạnh', run: ctx => ctx.scene._inst?.cycleView(ctx) },
    { label: 'Tới Mặt Trời', key: 'g', title: 'Bay tới vị trí Mặt Trời và bám theo quỹ đạo', run: ctx => ctx.scene._inst?.toggleSun(ctx) },
    { label: 'Đặt lại thời gian', key: 'r', title: 'Trở về t = 0 (các nhánh chưa bị xoắn)', run: ctx => ctx.scene._inst?.resetTime(ctx) },
  ],
  create(ctx) {
    const def = this;
    const camera = ctx.camera, controls = ctx.controls;
    // lookup table: (Ω−κ/2)_vis − same at R_bar, r·d/dr of it, Ω_dm, Ω_kep
    const tabArr = [];
    for (let i = 0; i <= TAB_N; i++) {
      const r = Math.max(i * TAB_DR, 0.05), h = 0.02 * Math.max(r, RB);
      tabArr.push(new THREE.Vector4(dPrec(r), r > RB ? r * (dPrec(r + h) - dPrec(r - h)) / (2 * h) : 0, omDM(r), omKep(r)));
    }
    const rotU = {
      uTdm: { value: 0 }, uTkep: { value: 0 }, uMixK: { value: 0 }, uM: { value: 2 }, uTanP: { value: Math.tan(15 * Math.PI / 180) },
      uOmP: { value: OMP }, uBar0: { value: 0 }, uRb: { value: RB }, uDelta: { value: 0.75 }, uTab: { value: tabArr },
    };
    const uniforms = { ...ctx.shared, ...rotU, uGlow: { value: 1 } };
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: GAL_FRAG, depthTest: false, depthWrite: false });

    const S = buildStars(ctx);
    const ptU = { ...rotU, uPxScale: { value: 500 }, uMinPx: { value: 2 }, uGain: { value: 1 } };
    const starMat = new THREE.ShaderMaterial({
      uniforms: ptU, vertexShader: STAR_VERT, fragmentShader: STAR_FRAG,
      blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
    });
    const oldPts = new THREE.Points(S.old, starMat);
    const youngPts = new THREE.Points(S.young, starMat);
    const dustU = { ...rotU, uPxScale: ptU.uPxScale, uMinPx: { value: 2.5 }, uDust: { value: 1 } };
    const dustMat = new THREE.ShaderMaterial({
      uniforms: dustU, vertexShader: DUST_VERT, fragmentShader: DUST_FRAG, transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.ZeroFactor, blendDst: THREE.OneMinusSrcColorFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
    });
    const dustPts = new THREE.Points(buildDust(), dustMat);
    oldPts.renderOrder = 1; youngPts.renderOrder = 2; dustPts.renderOrder = 3;

    // Sun marker + its orbit
    const marker = new THREE.Group();
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture(), sizeAttenuation: false, depthTest: false, depthWrite: false, transparent: true, toneMapped: false }));
    sprite.center.set(0.5, 60 / 256);
    sprite.scale.set(0.32, 0.16, 1);
    sprite.material.color.setScalar(1.5);
    const ringPts = [];
    for (let i = 0; i <= 256; i++) { const a = i / 256 * Math.PI * 2; ringPts.push(new THREE.Vector3(R_SUN * Math.cos(a), 0, R_SUN * Math.sin(a))); }
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: new THREE.Color(0.12, 0.1, 0.04), blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false }));
    marker.add(ring, sprite);
    marker.renderOrder = 4; sprite.renderOrder = 4; ring.renderOrder = 4;

    const overlay = new THREE.Group();
    for (const o of [oldPts, youngPts, dustPts, marker]) { o.frustumCulled = false; overlay.add(o); }

    // --- time bookkeeping: Myr spent with / without the dark halo ---
    const p0 = ctx.params;
    let Tdm = 0, Tkep = 0, mixK = p0.darkMatter ? 0 : 1, lastSim = 0, inited = false;
    const THETA_SUN0 = -0.47;   // Sun–centre line ≈ 27° from the bar's major axis
    const sunPos = new THREE.Vector3(), prevSun = new THREE.Vector3();
    const sunAngle = () => THETA_SUN0 + omDM(R_SUN) * Tdm + omKep(R_SUN) * Tkep;
    const updateSun = () => { const a = sunAngle(); sunPos.set(R_SUN * Math.cos(a), 0, R_SUN * Math.sin(a)); };
    updateSun(); prevSun.copy(sunPos);

    // --- camera helpers (spherical tween around a possibly moving target) ---
    let fly = null, follow = false, viewIdx = 0;
    const sph = new THREE.Spherical(), tmp = new THREE.Vector3(), sz = new THREE.Vector2(), ORIGIN = new THREE.Vector3();
    function startFly(dist, polarDeg, azRad, tgtFn, dur = 2.4) {
      sph.setFromVector3(tmp.copy(camera.position).sub(controls.target));
      fly = { r0: sph.radius, p0: sph.phi, a0: sph.theta, r1: dist, p1: THREE.MathUtils.degToRad(polarDeg), a1: azRad ?? sph.theta,
              tg0: controls.target.clone(), tgtFn, t0: performance.now(), dur };
      let da = fly.a1 - fly.a0; da = Math.atan2(Math.sin(da), Math.cos(da)); fly.a1 = fly.a0 + da;
    }
    const inst = {
      material, overlay,
      cycleView(c) {
        viewIdx = (viewIdx + 1) % VIEWS.length;
        sph.setFromVector3(tmp.copy(camera.position).sub(controls.target));
        const dist = follow ? Math.max(sph.radius, 4) : (sph.radius < 25 ? 40 : sph.radius);
        startFly(dist, VIEWS[viewIdx][1], null, null, 2.2);
        c.toast(VIEWS[viewIdx][0] + (viewIdx === 2 ? ' — làn bụi ở mặt phẳng giữa cắt ngang đĩa sáng, khối phình hiện rõ hình hộp/hạt lạc của thanh ngang.' : viewIdx === 0 ? ' — nhìn thẳng xuống đĩa: thanh ngang, các nhánh và làn bụi ở mép trong của nhánh.' : ''), 4500);
      },
      toggleSun(c) {
        if (!follow) {
          follow = true;
          controls.minDistance = 0.6;
          const az = Math.atan2(sunPos.x, sunPos.z);   // look inward from behind the Sun
          startFly(4.5, 68, az, () => sunPos, 3.0);
          c.toast('Mặt Trời ở <b>8,2 kpc</b> từ tâm, giữa nhánh Perseus và nhánh Nhân Mã, quay ~220 km/s — mỗi vòng ≈ 230 triệu năm. Nhấn <b>G</b> lần nữa để quay về.', 7000);
        } else {
          follow = false;
          controls.minDistance = def.camera.min;
          startFly(40, 55, null, () => ORIGIN, 2.6);
        }
      },
      resetTime(c) {
        Tdm = 0; Tkep = 0; updateSun(); prevSun.copy(sunPos);
        c.toast('t = 0: các nhánh trở lại hình dạng ban đầu.', 2500);
      },
      update(dt, c) {
        const p = c.params;
        // time: `speed` Myr per wall-second at the default global time speed (6×)
        if (!inited) {   // first frame: t = 0 (snapshot mode: start at t= so the epoch is reproducible)
          inited = true; lastSim = c.time.sim; mixK = p.darkMatter ? 0 : 1;
          const T0 = SNAP ? c.time.sim / 6 * p.speed : 0;
          if (p.darkMatter) Tdm = T0; else Tkep = T0;
        }
        const dSim = Math.max(0, c.time.sim - lastSim); lastSim = c.time.sim;
        const dT = Math.min(dSim, 0.5) / 6 * p.speed;
        const target = p.darkMatter ? 0 : 1;
        mixK += Math.sign(target - mixK) * Math.min(Math.abs(target - mixK), dt / 1.2);
        Tdm += dT * (1 - mixK); Tkep += dT * mixK;
        rotU.uTdm.value = Tdm; rotU.uTkep.value = Tkep; rotU.uMixK.value = mixK;
        rotU.uM.value = p.arms; rotU.uTanP.value = Math.tan(THREE.MathUtils.degToRad(p.pitch));
        uniforms.uGlow.value = p.glow;
        c.renderer.getDrawingBufferSize(sz);
        ptU.uPxScale.value = sz.y * camera.projectionMatrix.elements[5] / 2;
        ptU.uGain.value = p.bright;
        dustU.uDust.value = p.dust;
        dustPts.visible = p.dust > 0.001;
        S.old.setDrawRange(0, Math.round(S.nOld * p.count));
        S.young.setDrawRange(0, Math.round(S.nYoung * Math.max(p.count, 0.4)));
        // Sun
        prevSun.copy(sunPos); updateSun();
        sprite.position.copy(sunPos);
        marker.visible = !!p.marker;
        // camera
        if (fly) {
          const k = Math.min(1, (performance.now() - fly.t0) / (fly.dur * 1000)), e = k * k * (3 - 2 * k);
          const tg = fly.tgtFn ? fly.tgtFn() : fly.tg0;
          controls.target.lerpVectors(fly.tg0, tg, e);
          sph.set(THREE.MathUtils.lerp(fly.r0, fly.r1, e), THREE.MathUtils.lerp(fly.p0, fly.p1, e), THREE.MathUtils.lerp(fly.a0, fly.a1, e));
          camera.position.setFromSpherical(sph).add(controls.target);
          camera.lookAt(controls.target);
          if (k >= 1) fly = null;
        } else if (follow) {
          tmp.copy(sunPos).sub(prevSun);
          controls.target.add(tmp); camera.position.add(tmp);
        }
      },
      hud(c) {
        const cp = camera.position, rc = Math.hypot(cp.x, cp.z);
        const v = mixK < 0.5 ? vDM(R_SUN) : vKep(R_SUN);
        const vMix = vDM(R_SUN) * (1 - mixK) + vKep(R_SUN) * mixK;
        const T = Tdm + Tkep;
        const year = 2 * Math.PI * R_SUN / (vMix * KMS);
        const laps = (omDM(R_SUN) * Tdm + omKep(R_SUN) * Tkep) / (2 * Math.PI);
        const fT = T >= 1000 ? (T / 1000).toFixed(2).replace('.', ',') + ' tỷ năm' : Math.round(T) + ' tr. năm';
        return [
          ['Bán kính dưới camera', `${rc.toFixed(1).replace('.', ',')} kpc`],
          ['Tốc độ Mặt Trời', `${Math.round(vMix)} km/s` + (v < 200 ? ' · Kepler' : '')],
          ['Năm thiên hà', `${Math.round(year)} tr. năm`],
          ['Thời gian', `${fT} · ${laps.toFixed(2).replace('.', ',')} vòng`],
        ];
      },
      enter(c) { if (inited) lastSim = c.time.sim; fly = null; follow = false; controls.minDistance = def.camera.min; },
      exit() { fly = null; follow = false; },
    };
    def._inst = inst;
    return inst;
  },
};

// =============================================================================================
// galaxy-collision: Toomre & Toomre (1972) restricted three-body encounter.
// Two massive cores (Plummer-softened point masses standing in for bulge + dark halo) fall toward
// each other on a parabolic orbit; each carries a disk of massless test stars integrated on the CPU
// (drift–kick–drift leapfrog, ≤ 0.5 Myr steps). Units: kpc, Myr, 10¹⁰ M☉ → G = 0.04498.
// Optional dynamical friction (Chandrasekhar drag of the overlapping dark halos, simple
// −k v_rel / (1 + (r/12 kpc)²) law) lets the cores sink and merge after the second passage.
const GC = 0.04498;
const EPS2 = 1.0;               // core softening² (kpc²)
const N_COLL = 36000;

const COLL_VERT = /* glsl */`
precision highp float;
attribute vec4 aCol;            // linear rgb × luminance, sprite size (kpc)
uniform float uPxScale, uMinPx, uGain;
varying vec3 vCol;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float px = aCol.w * uPxScale / max(-mv.z, 1e-3);
  float ps = clamp(px, uMinPx, 64.0);
  gl_PointSize = ps;
  vCol = aCol.rgb * uGain * min(px * px / (ps * ps), 4.0);
}`;
const COLL_FRAG = /* glsl */`
precision highp float;
varying vec3 vCol;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = min(dot(c, c), 1.0);
  gl_FragColor = vec4(vCol * exp(-r2 * 3.6) * (1.0 - r2) * (1.0 - r2), 1.0);
}`;
const COLL_FRAG_BG = COMMON_GLSL + /* glsl */`
uniform vec3 uC1, uC2;
uniform vec2 uM;                // core masses (10¹⁰ M☉)
// column density of a Plummer-like bulge ρ ∝ (r² + a²)^-2 along the ray: ∝ (b² + a²)^-3/2
float column(vec3 ro, vec3 rd, vec3 c, float a){
  vec3 oc = c - ro; float t = max(dot(oc, rd), 0.0);
  float b2 = dot(oc, oc) - t * t;
  return a * a * a / pow(b2 + a * a, 1.5);
}
void main(){
  vec3 rd = camRay(), ro = uCamPos;
  vec3 col = uSkyMode < 0.5
    ? (starLayer(rd, 40.0, 0.05, 0.07, 1.0, 1.0) * 0.45 + starLayer(rd, 110.0, 0.06, 0.09, 2.0, 1.0) * 0.18) * uStarBright
    : sky(rd, 1.0) * 0.5;
  vec3 warm = bb(4300.0);
  col += warm * (uM.x * (0.08 * column(ro, rd, uC1, 0.8) + 0.03 * column(ro, rd, uC1, 3.0))
               + uM.y * (0.08 * column(ro, rd, uC2, 0.8) + 0.03 * column(ro, rd, uC2, 3.0)));
  gl_FragColor = vec4(col, 1.0);
}`;

const collision = {
  id: 'galaxy-collision',
  name: 'Thiên hà va chạm',
  category: 'galaxies',
  order: 20,
  subtitle: 'Tương tác thủy triều · mô hình Toomre & Toomre (1972)',
  swatch: ['#cfe0ff', '#40204e'],
  blurb: 'Khi hai thiên hà lướt qua nhau, lực thủy triều kéo các sao ở rìa thành những <b>đuôi</b> dài và <b>cầu nối</b> — như cặp Ăng-ten (NGC 4038/39) hay Con Chuột (NGC 4676). Năm 1972, anh em Toomre tái tạo những hình dạng ấy chỉ với hai khối lượng điểm và các sao “thử” không tự hút nhau: đuôi dài nhất khi đĩa quay <b>cùng chiều</b> quỹ đạo, vì sao ở rìa cộng hưởng với lực kéo.',
  intro: 'Đổi <b>Chiều quay đĩa A</b> sang <b>Nghịch</b>: đĩa A gần như không có đuôi thủy triều — cộng hưởng giữa chuyển động quỹ đạo và chuyển động quay của đĩa mới là chìa khóa. Nhấn <b>R</b> để chạy lại.',
  facts: [
    ['Mô hình', 'Bài toán ba vật hạn chế'],
    ['Quỹ đạo', 'Parabol, cận điểm 8–30 kpc'],
    ['Ví dụ', 'Ăng-ten · Con Chuột · Bánh Xe'],
    ['Đuôi thủy triều Ăng-ten', '≈ 100 kpc'],
    ['Một lần lướt qua', 'vài trăm triệu năm'],
    ['Ngân Hà + Andromeda', 'đang tiến lại gần ở 110 km/s'],
  ],
  camera: { distance: 150, polar: 38, azimuth: 20, min: 12, max: 400, fov: 50 },
  exposure: 1.0,
  bloom: 0.6,
  params: [
    { key: 'speed', label: 'Tốc độ thời gian', type: 'range', min: 0, max: 40, step: 0.5, default: 10, fmt: v => v.toFixed(1) + ' triệu năm/s', note: 'Ở tốc độ thời gian chung mặc định.' },
    { key: 'spinA', label: 'Chiều quay đĩa A', type: 'seg', options: [[1, 'Thuận'], [-1, 'Nghịch']], default: 1, note: 'So với chiều chuyển động trên quỹ đạo. Đổi sẽ chạy lại mô phỏng.' },
    { key: 'incB', label: 'Độ nghiêng đĩa B', type: 'range', min: 0, max: 180, step: 5, default: 60, fmt: v => v + '°', note: '> 90° nghĩa là đĩa B quay nghịch.' },
    { key: 'ratio', label: 'Tỉ lệ khối lượng B/A', type: 'range', min: 0.2, max: 1, step: 0.05, default: 1, fmt: v => v.toFixed(2).replace('.', ',') },
    { key: 'peri', label: 'Khoảng cách cận điểm', type: 'range', min: 8, max: 30, step: 1, default: 14, fmt: v => v + ' kpc' },
    { key: 'friction', label: 'Ma sát động lực (quầng tối)', type: 'toggle', default: true, note: 'Quầng vật chất tối hãm chuyển động của hai nhân, khiến chúng quay lại và sáp nhập. Mô hình Toomre gốc không có hiệu ứng này.' },
    { key: 'bright', label: 'Độ sáng sao', type: 'range', min: 0.2, max: 3, step: 0.05, default: 1 },
  ],
  actions: [
    { label: 'Chạy lại', key: 'r', title: 'Đặt lại hai thiên hà ở vị trí ban đầu', run: ctx => ctx.scene._inst?.restart(ctx, true) },
  ],
  create(ctx) {
    const def = this;
    const RESTART_KEYS = ['spinA', 'incB', 'ratio', 'peri'];
    const T_PERI = 250;   // Myr from start to first pericentre
    const pos = new Float32Array(N_COLL * 3), vel = new Float32Array(N_COLL * 3), col = new Float32Array(N_COLL * 4);
    const geom = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(pos, 3); posAttr.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('position', posAttr);
    geom.setAttribute('aCol', new THREE.BufferAttribute(col, 4));
    const ptU = { uPxScale: { value: 500 }, uMinPx: { value: 2 }, uGain: { value: 1 } };
    const pts = new THREE.Points(geom, new THREE.ShaderMaterial({
      uniforms: ptU, vertexShader: COLL_VERT, fragmentShader: COLL_FRAG,
      blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
    }));
    pts.frustumCulled = false;
    const uniforms = { ...ctx.shared, uC1: { value: new THREE.Vector3() }, uC2: { value: new THREE.Vector3() }, uM: { value: new THREE.Vector2(10, 10) } };
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: COLL_FRAG_BG, depthTest: false, depthWrite: false });

    // cores: positions / velocities (relative to the centre of mass at the origin)
    const C = { p: [new THREE.Vector3(), new THREE.Vector3()], v: [new THREE.Vector3(), new THREE.Vector3()], m: [10, 10] };
    let T = 0, lastSim = 0, inited = false, friction = true, merged = false;
    const sz = new THREE.Vector2();

    function setup(p) {
      const R = rng(4242);
      const gauss = () => { let u = 0; for (let i = 0; i < 4; i++) u += R(); return (u - 2) * 1.732; };
      C.m[0] = 10; C.m[1] = 10 * p.ratio;
      const M = C.m[0] + C.m[1], q = p.peri;
      // parabolic two-body orbit, T_PERI before pericentre: t = sqrt(2q³/GM)(D + D³/3), D = tan(ν/2)
      const tau = Math.sqrt(2 * q * q * q / (GC * M)), target = -T_PERI / tau;
      let D = -1; for (let i = 0; i < 60; i++) D -= (D + D * D * D / 3 - target) / (1 + D * D);
      const nu = 2 * Math.atan(D), r = q * (1 + D * D), h = Math.sqrt(2 * GC * M * q);
      const rel = new THREE.Vector3(r * Math.cos(nu), 0, r * Math.sin(nu));
      const vr = GC * M / h * Math.sin(nu), vt = GC * M / h * (1 + Math.cos(nu));
      const vrel = new THREE.Vector3(vr * Math.cos(nu) - vt * Math.sin(nu), 0, vr * Math.sin(nu) + vt * Math.cos(nu));
      C.p[0].copy(rel).multiplyScalar(-C.m[1] / M); C.p[1].copy(rel).multiplyScalar(C.m[0] / M);
      C.v[0].copy(vrel).multiplyScalar(-C.m[1] / M); C.v[1].copy(vrel).multiplyScalar(C.m[0] / M);
      const L = new THREE.Vector3().crossVectors(rel, vrel).normalize();   // orbital angular momentum
      // disk normals: A tilted 15° from the orbital plane, B by incB; spin sign flips the normal
      const nA = L.clone().applyAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(15)).multiplyScalar(p.spinA);
      const nB = L.clone().applyAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(p.incB)).applyAxisAngle(L, 0.6);
      const nB0 = Math.round(N_COLL * Math.sqrt(p.ratio) / (1 + Math.sqrt(p.ratio)));
      const bbW = ctx.bbColor(4600), bbB = ctx.bbColor(14000), bbM = ctx.bbColor(7500);
      const e1 = new THREE.Vector3(), e2 = new THREE.Vector3();
      for (let g = 0; g < 2; g++) {
        const n = g === 0 ? nA : nB, m = C.m[g], rs = Math.sqrt(m / 10);
        e1.crossVectors(n, Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)).normalize();
        e2.crossVectors(n, e1);
        const i0 = g === 0 ? 0 : N_COLL - nB0, i1 = g === 0 ? N_COLL - nB0 : N_COLL;
        for (let i = i0; i < i1; i++) {
          let rr; do rr = R() < 0.45 ? 1.2 + 11.8 * Math.sqrt(R()) : -3.2 * Math.log(R() * R() + 1e-9); while (rr > 13 || rr < 1.2);   // exponential disk + extended outer H I-like disk (feeds the tails)
          rr *= rs;
          const ph = R() * Math.PI * 2, z = 0.12 * gauss() * rs;
          const vc = Math.sqrt(GC * m * rr * rr / Math.pow(rr * rr + EPS2, 1.5));
          const cp = Math.cos(ph), sp = Math.sin(ph);
          pos[i * 3] = C.p[g].x + rr * (cp * e1.x + sp * e2.x) + z * n.x;
          pos[i * 3 + 1] = C.p[g].y + rr * (cp * e1.y + sp * e2.y) + z * n.y;
          pos[i * 3 + 2] = C.p[g].z + rr * (cp * e1.z + sp * e2.z) + z * n.z;
          vel[i * 3] = C.v[g].x + vc * (-sp * e1.x + cp * e2.x);
          vel[i * 3 + 1] = C.v[g].y + vc * (-sp * e1.y + cp * e2.y);
          vel[i * 3 + 2] = C.v[g].z + vc * (-sp * e1.z + cp * e2.z);
          // colour: old yellow centre → blue star-forming outskirts
          const k = Math.min(1, Math.max(0, (rr / rs - 2) / 8));
          const c0 = k < 0.5 ? bbW : bbM, c1 = k < 0.5 ? bbM : bbB, kk = k < 0.5 ? k * 2 : k * 2 - 1;
          const lum = (0.6 + R()) * (R() < 0.02 ? 4 : 1) * (1.2 + 1.2 * k);
          for (let s = 0; s < 3; s++) col[i * 4 + s] = (c0[s] + (c1[s] - c0[s]) * kk) * lum;
          col[i * 4 + 3] = 0.11;
        }
      }
      geom.attributes.aCol.needsUpdate = true;
      posAttr.needsUpdate = true;
      T = 0; merged = false;
    }
    const a0 = new THREE.Vector3(), d = new THREE.Vector3();
    function step(dt) {
      const P = C.p, V = C.v, m0 = C.m[0], m1 = C.m[1], h = dt / 2, n3 = N_COLL * 3;
      // drift ½
      P[0].addScaledVector(V[0], h); P[1].addScaledVector(V[1], h);
      for (let i = 0; i < n3; i++) pos[i] += vel[i] * h;
      // kick: cores (mutual softened gravity + dynamical friction on the relative motion)
      d.subVectors(P[1], P[0]);
      const r2 = d.lengthSq() + EPS2, r = Math.sqrt(r2);
      a0.copy(d).multiplyScalar(GC / (r2 * r));
      V[0].addScaledVector(a0, m1 * dt); V[1].addScaledVector(a0, -m0 * dt);
      if (friction) {
        const M = m0 + m1, k = 0.0028 / (1 + (r / 12) * (r / 12)) * dt;
        a0.subVectors(V[1], V[0]);
        V[0].addScaledVector(a0, k * m1 / M); V[1].addScaledVector(a0, -k * m0 / M);
      }
      // kick: test stars feel both cores
      const x0 = P[0].x, y0 = P[0].y, z0 = P[0].z, x1 = P[1].x, y1 = P[1].y, z1 = P[1].z, g0 = GC * m0 * dt, g1 = GC * m1 * dt;
      for (let j = 0; j < n3; j += 3) {
        const px = pos[j], py = pos[j + 1], pz = pos[j + 2];
        let dx = x0 - px, dy = y0 - py, dz = z0 - pz, q2 = dx * dx + dy * dy + dz * dz + EPS2, f = g0 / (q2 * Math.sqrt(q2));
        const ax = dx * f, ay = dy * f, az = dz * f;
        dx = x1 - px; dy = y1 - py; dz = z1 - pz; q2 = dx * dx + dy * dy + dz * dz + EPS2; f = g1 / (q2 * Math.sqrt(q2));
        vel[j] += ax + dx * f; vel[j + 1] += ay + dy * f; vel[j + 2] += az + dz * f;
      }
      // drift ½
      P[0].addScaledVector(V[0], h); P[1].addScaledVector(V[1], h);
      for (let i = 0; i < n3; i++) pos[i] += vel[i] * h;
      T += dt;
      if (P[0].distanceTo(P[1]) < 1.5 && T > T_PERI) merged = true;
    }
    function advance(dT, maxSteps) {
      const n = Math.min(maxSteps, Math.ceil(dT / 0.5 - 1e-9));
      if (n <= 0) return;
      const dt = Math.min(0.5, dT / n);
      for (let s = 0; s < n; s++) step(dt);
      posAttr.needsUpdate = true;
    }
    const inst = {
      material, overlay: pts,
      restart(c, toast) {
        setup(c.params);
        if (toast) c.toast(`Hai thiên hà trở lại vị trí xuất phát, còn ${T_PERI} triệu năm nữa tới cận điểm.`, 3000);
      },
      onParam(key, v, c) { if (RESTART_KEYS.includes(key)) inst.restart(c, false); },
      update(dt, c) {
        const p = c.params;
        friction = !!p.friction;
        if (!inited) {   // first frame (snapshot mode: catch up to t=)
          inited = true; setup(p); lastSim = c.time.sim;
          if (SNAP) advance(Math.min(c.time.sim / 6 * p.speed, 2500), 6000);
        }
        const dSim = Math.max(0, c.time.sim - lastSim); lastSim = c.time.sim;
        advance(Math.min(dSim, 0.5) / 6 * p.speed, 24);
        c.renderer.getDrawingBufferSize(sz);
        ptU.uPxScale.value = sz.y * c.camera.projectionMatrix.elements[5] / 2;
        ptU.uGain.value = p.bright;
        uniforms.uC1.value.copy(C.p[0]); uniforms.uC2.value.copy(C.p[1]);
        uniforms.uM.value.set(C.m[0], C.m[1]);
      },
      hud() {
        const sep = C.p[0].distanceTo(C.p[1]);
        const vrel = C.v[0].distanceTo(C.v[1]) / KMS;   // kpc/Myr → km/s
        const tp = T - T_PERI;
        const state = merged ? 'Đã sáp nhập' : T < T_PERI - 20 ? 'Đang tiến lại gần' : Math.abs(tp) <= 20 ? 'Cận điểm!' : 'Đang tách ra / quay lại';
        return [
          ['Khoảng cách hai nhân', `${sep.toFixed(1).replace('.', ',')} kpc`],
          ['Vận tốc tương đối', `${Math.round(vrel)} km/s`],
          ['So với cận điểm', `${tp < 0 ? '−' : '+'}${Math.abs(Math.round(tp))} triệu năm`],
          ['Trạng thái', state],
        ];
      },
      enter(c) { if (inited) lastSim = c.time.sim; },
    };
    def._inst = inst;
    return inst;
  },
};

export default [spiral, collision];
