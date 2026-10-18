// Tinh vân Đầu Ngựa (Barnard 33 trong IC 434) — dựng lại từ ảnh thật.
//
// One scene unit = one light-year. x/y are the sky plane in the orientation of the ESO VLT image
// (σ Orionis is "up"), +z points toward the observer. The canonical camera (0, 0, D0) sees exactly
// the ESO FORS2 frame (6.6′ × 6.8′ ≈ 2.63 × 2.71 ly at 1 375 ly).
//
// How the 3-D nebula is built from the photographs (offline, see assets/horsehead/CREDITS.txt):
//   * each image is converted back to linear radiance (inverse of the host's ACES + gamma), the
//     stars are measured and removed;
//   * the emission behind the dust B(x,y) is the star-free image where it is not darkened and a smooth
//     extrapolation of the surrounding IC 434 glow behind the dust;
//   * the dust optical depth τ_V(x,y) = −ln(I_R / B_R) / (A_R/A_V) (screen model, R ≈ Hα band);
//   * the residual L = I − B·e^(−τ) is the dust's own light (scattered starlight in the visible,
//     scattered + thermal/PAH light in the infrared, ionisation-front Hα on the rim);
//   * depth: the silhouette is "inflated" — half-thickness h(x,y) = √(d(2r − d)) from the distance d to
//     the cloud edge and the local half-width r, i.e. the pillar is assumed roughly as deep as it is wide.
// The shader distributes τ and L along z with the normalised profile (15/16)(1 − u²)²/h, u = z/h, so
// that the canonical view reproduces the photograph exactly (emission and absorption are integrated
// analytically per step), while any other viewpoint shows real parallax: IC 434 is a sheet ~2–3.5 ly
// behind the head, the stars sit at random depths in front of or behind the cloud and are dimmed by
// the dust column they actually cross.
// Texture lookups use the canonical projection (x,y)·D0/(D0 − z) so the canonical image stays sharp.
//
// Views: visible (ESO VLT/FORS2, B·V·R), infrared (Hubble WFC3/IR 2013 inside its footprint, Euclid
// ERO 2023 around it), JWST NIRCam 2024 on the mane; the wide view uses a CC0 Hα mosaic (Gianni
// Lacroce) with σ Ori, Alnitak, NGC 2023 and the Flame Nebula.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';
import { t, dec } from '../../i18n.js';

const ASSET = f => new URL(`../../../assets/horsehead/${f}`, import.meta.url).href;

// ---- registration (world ly → texture uv), from star matching between the images ----------------
// rows: [a00, a01, b0], [a10, a11, b1];  uv = A·(x, y) + b   (uv.y up, as three.js flips images)
const XF = {
  C: [[0.3802424, 0.0, 0.5], [0.0, 0.3695949, 0.5]],                   // ESO VLT/FORS2 eso0202a
  W: [[0.0366329, 0.0004796, 0.6925163], [-0.0004796, 0.0366329, 0.3162625]],   // Hα mosaic crop (27 ly)
  F: [[0.0134253, 0.0001757, 0.4703541], [-0.0002425, 0.0185227, 0.3095328]],   // whole Hα mosaic (75 ly)
  H: [[0.4183031, -0.0974876, 0.4693601], [0.0932348, 0.4003203, 0.4878385]],   // Hubble heic1307a
  E: [[0.0605889, -0.0017202, 0.3835025], [0.0017202, 0.0605889, 0.36692]],     // Euclid ERO
  J: [[1.0803023, -0.4422911, 0.1005675], [0.4397918, 1.0725768, -0.7313639]],  // JWST weic2411a
};
// texture decode scales (texel^2.2 × scale = linear radiance)
const SC = { CL: 0.89875, WL: 3.49779, HB: 0.17041, HL: 0.68000, EB: 0.57031, EL: 0.33162, JB: 0.76720, JL: 1.22996 };
const D0 = 7.0;          // canonical camera distance (ly) for fov 22°: frame height 2.71 ly
const FOV0 = 22;
const TAU_HEAD = 6.2;    // 95th percentile of τ_V across the head, measured from the ESO image (A_V ≈ 1.086 τ_V)
// relative extinction A_λ/A_V per colour channel (R, G, B) of each image (Cardelli et al. 1989, R_V ≈ 3.1)
const KEXT = { vis: [0.8, 1.0, 1.3], ir: [0.18, 0.23, 0.28], jwst: [0.05, 0.07, 0.1] };
// landmarks in world coordinates (projected on the sky plane)
const SIGMA_ORI = [3.556, 13.663, -3.0];
const ALNITAK = [-12.235, 1.616, -2.5];
const FLAME = [-14.88, -3.145];
const NGC2023 = [-4.451, -4.584];

const xfMat = r => new THREE.Matrix3().set(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], 0, 0, 1);
const uvOf = (r, x, y) => [r[0][0] * x + r[0][1] * y + r[0][2], r[1][0] * x + r[1][1] * y + r[1][2]];
function worldOfUv(r, u, v) {
  const a = r[0][0], b = r[0][1], c = r[1][0], d = r[1][1], det = a * d - b * c;
  const du = u - r[0][2], dv = v - r[1][2];
  return [(d * du - b * dv) / det, (-c * du + a * dv) / det];
}
// the JWST footprint (0.86 ly square, rotated 22° on our sky plane): framed like the press image by
// rolling the camera so the NIRCam x axis is horizontal
const J_CENTER = worldOfUv(XF.J, 0.5, 0.4);
const J_FOV = 10;
const J_D = 0.23 / Math.tan(THREE.MathUtils.degToRad(J_FOV / 2));   // 0.46 ly visible height
const J_ROLL = Math.atan2(-(XF.J[1][0]), XF.J[1][1]);  // world angle of the image's +x axis (≈ −22°)
const UP_J = [-Math.sin(J_ROLL), Math.cos(J_ROLL), 0];
// A photograph only constrains the cloud along one line of sight: the orbit is limited to a cone
// around it (wider in the wide field, where everything sits far behind the camera)
const ORBIT_AZ = 32, ORBIT_POL = 24, ORBIT_AZ_WIDE = 50, ORBIT_POL_WIDE = 35;
// the JWST frame is only 0.86 ly across, a few ly in front of IC 434: a narrower cone keeps it in view
const ORBIT_AZ_J = 12, ORBIT_POL_J = 9;
const DEG = Math.PI / 180;

// ------------------------------------------------------------------------------------------
// shaders
// ------------------------------------------------------------------------------------------
const HEAD_GLSL = /* glsl */`
#ifndef STEPS
#define STEPS 64
#endif
#ifndef MODE
#define MODE 0
#endif
uniform sampler2D tCD, tCL, tCB, tWD, tWL, tWB, tWG, tF;
uniform sampler2D tHB, tHL, tEB, tEL, tJB, tJL, tEG, tHG;
#define FC 0.1
uniform mat3 uMC, uMW, uMF, uMH, uME, uMJ;
uniform float uDust, uIon, uDepth, uPal, uReady, uCvis;
uniform vec3 uPC;          // extrusion centre: the camera while it looks along the line of sight
uniform float uOrtho;      // 0: perspective extrusion from uPC, 1: orthographic (oblique views)
uniform float uOrthoD;     // the same for the dust volume (kept partly camera-centred: crisper, less smear)
uniform float uObl;        // 0: looking along the line of sight of the photographs, 1: clearly oblique
uniform vec3 uKext;
float ign(){ return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))); }
vec2 uvOf(mat3 M, vec2 p){ return (M * vec3(p, 1.0)).xy; }
float ew(vec2 uv, float f){ vec2 e = min(uv, 1.0 - uv); return smoothstep(0.0, f, min(e.x, e.y)); }
vec3 dec(vec3 t){ return t * t * sqrt(t); }   // ≈ t^2.5, close to the 1/2.2 encoding and cheaper than pow
vec3 decB(vec3 t){ return pow(t, vec3(2.2)); }
// palette: the ESO frame is B·V·R (pink IC 434, teal dust); the wide mosaic is Hα-dominated (deep red)
vec3 w2e(vec3 c){ return vec3(2.0 * c.r, c.g + 0.185 * c.r, c.b + 0.284 * c.r); }
vec3 e2w(vec3 c){ return max(vec3(0.5 * c.r, c.g - 0.0925 * c.r, c.b - 0.142 * c.r), 0.0); }
// low-frequency gain (offline fit) that matches the mosaic to the ESO frame around it
vec3 palW(vec3 c, vec2 uw){ vec3 g = exp((texture(tWG, uw).rgb - 0.5) * 6.4); return mix(w2e(c) * g, c, uPal); }
vec3 palC(vec3 c, vec2 uw){ vec3 g = exp((texture(tWG, uw).rgb - 0.5) * 6.4); return mix(c, e2w(c / g), uPal); }
// infrared: Euclid matched to Hubble around its footprint, Hubble matched to JWST around the mane
vec3 gE(vec2 ue){ return exp((texture(tEG, ue).rgb - 0.5) * 6.4); }
#if MODE == 2
vec3 gH(vec2 uh){ return exp((texture(tHG, uh).rgb - 0.5) * 6.4); }
#else
vec3 gH(vec2 uh){ return vec3(1.0); }
#endif
// canonical projection of a world point onto the z = 0 sky plane
// The photographs are projections onto the sky plane z = 0. While the camera looks roughly along the
// line of sight, the volume is extruded along the camera's own rays, so every view along it (the
// canonical frames, zooms, the wide field) shows the photograph pixel-exact; as the camera swings to
// an oblique angle the extrusion relaxes to plain orthographic (along z) and the depth shows.
vec2 projO(vec3 p, float o){ float dz = uPC.z - p.z * (1.0 - o); return uPC.xy + (p.xy - uPC.xy) * (uPC.z / (abs(dz) < 0.05 ? (dz < 0.0 ? -0.05 : 0.05) : dz)); }
vec2 proj(vec3 p){ return projO(p, uOrtho); }
vec2 projD(vec3 p){ return projO(p, uOrthoD); }

// dust column τ_V, half-thickness h, ionised-gas fraction of L, and the dust light L (radiance·ly)
void dustAt(vec2 q, out float tau, out float h, out float rim, out vec3 L){
  tau = 0.0; h = 0.05; rim = 0.0; L = vec3(0.0);
  vec2 uc = uvOf(uMC, q); float wc = ew(uc, FC) * uCvis;
  if (wc < 1.0){
    vec2 uw = uvOf(uMW, q); float ww = ew(uw, 0.015);
    if (ww > 0.0){
      vec3 d = texture(tWD, uw).rgb;
      tau = d.r * d.r * 10.0 * ww; h = max(d.g * 1.4, 0.05); rim = d.b;
#if MODE == 0
      L = palW(dec(texture(tWL, uw).rgb) * ${SC.WL.toFixed(5)}, uw) * ww;
#endif
    }
  }
  if (wc > 0.0){
    vec3 d = texture(tCD, uc).rgb;
    tau = mix(tau, d.r * d.r * 10.0, wc); h = mix(h, max(d.g * 2.0, 0.04), wc); rim = mix(rim, d.b, wc);
#if MODE == 0
    L = mix(L, palC(dec(texture(tCL, uc).rgb) * ${SC.CL.toFixed(5)}, uvOf(uMW, q)), wc);
#endif
  }
#if MODE >= 1
  // infrared dust light: Euclid around, Hubble inside its footprint, JWST on the mane
  vec2 uh = uvOf(uMH, q); float wh = ew(uh, 0.04);
#if MODE == 2
  vec2 uj = uvOf(uMJ, q); float wj = ew(uj, 0.08);
#else
  float wj = 0.0; vec2 uj = vec2(0.0);
#endif
  if (wh < 1.0 && wj < 1.0){
    vec2 ue = uvOf(uME, q); float we = ew(ue, 0.02);
    if (we > 0.0) L = dec(texture(tEL, ue).rgb) * gE(ue) * (${SC.EL.toFixed(5)} * we);
  }
  if (wh > 0.0 && wj < 1.0) L = mix(L, dec(texture(tHL, uh).rgb) * gH(uh) * ${SC.HL.toFixed(5)}, wh);
#if MODE == 2
  if (wj > 0.0) L = mix(L, dec(texture(tJL, uj).rgb) * ${SC.JL.toFixed(5)}, wj);
#endif
#endif
}

// emission behind the dust (IC 434 / Orion B surface): radiance seen at canonical projection q
vec3 backAt(vec2 q, out float cover, float lb){
  vec3 B = vec3(0.0); cover = 0.0;
#if MODE == 0
  vec2 uf = uvOf(uMF, q); float wf = ew(uf, 0.04);
  if (wf > 0.0){ B = mix(w2e(decB(texture(tF, uf, lb).rgb) * 4.0), decB(texture(tF, uf, lb).rgb) * 4.0, uPal) * wf; cover = wf; }
  vec2 uw = uvOf(uMW, q); float ww = ew(uw, 0.015);
  vec2 uc = uvOf(uMC, q); float wc = ew(uc, FC) * uCvis;
  if (ww > 0.0){
    B = mix(B, palW(decB(texture(tWB, uw, lb).rgb) * 4.0, uw), ww);
  }
  if (wc > 0.0) B = mix(B, palC(decB(texture(tCB, uc, lb).rgb) * 4.0, uw), wc);
#else
  vec2 ue = uvOf(uME, q); float we = ew(ue, 0.03);
  if (we > 0.0){ B = decB(texture(tEB, ue, lb).rgb) * gE(ue) * (${SC.EB.toFixed(5)} * we); cover = we; }
  vec2 uh = uvOf(uMH, q); float wh = ew(uh, 0.1);
  if (wh > 0.0) B = mix(B, decB(texture(tHB, uh, lb).rgb) * gH(uh) * ${SC.HB.toFixed(5)}, wh);
#if MODE == 2
  vec2 uj = uvOf(uMJ, q); float wj = ew(uj, 0.08);
  if (wj > 0.0) B = mix(B, decB(texture(tJB, uj, lb).rgb) * ${SC.JB.toFixed(5)}, wj);
#endif
#endif
  return B;
}
`;

const FRAG = COMMON_GLSL + HEAD_GLSL + /* glsl */`
// IC 434 seen behind the head: one main emitting sheet carries the photographed image; two fainter
// layers in front of / behind it carry only a smooth (mip-blurred) zero-sum modulation along the σ Ori
// direction. The canonical view stays exact, oblique views get depth structure without ghost copies
// of the photographed detail.
const float ZB0 = -2.45, ZB1 = -2.0, ZB2 = -3.1;
const float ZS = 1.95;                               // dust slab |z| < ZS·depth
const float SA = 0.08;                               // amplitude of the layered modulation
float stri(vec2 q, float k){ return k < 0.5 ? snoise(vec3(q.x * 2.2, q.y * 0.22, 1.3)) : snoise(vec3(q.x * 1.3, q.y * 0.15, 7.1)); }

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  if (uReady < 0.5){ gl_FragColor = vec4(sky(rd, 1.0) * 0.3, 1.0); return; }
  vec3 I = vec3(0.0), T = vec3(1.0);
  float rz = abs(rd.z) < 1e-4 ? (rd.z < 0.0 ? -1e-4 : 1e-4) : rd.z;
  vec3 rdz = vec3(rd.xy, rz);
  // background planes
  float tp0 = (ZB0 - ro.z) / rz, tp1 = (ZB1 - ro.z) / rz, tp2 = (ZB2 - ro.z) / rz, c0 = 0.0, c1 = 0.0, c2 = 0.0, cx;
  vec3 P0 = vec3(0.0), P1 = vec3(0.0), P2 = vec3(0.0);
  float ib = uIon;
  if (tp0 > 0.0){
    vec2 q = proj(ro + rdz * tp0);
    P0 = (backAt(q, c0, 0.0) - SA * (stri(q, 0.0) + stri(q, 1.0)) * backAt(q, cx, 4.0)) * ib;
  }
  if (tp1 > 0.0){ vec2 q = proj(ro + rdz * tp1); P1 = SA * stri(q, 0.0) * backAt(q, c1, 4.0) * ib; }
  if (tp2 > 0.0){ vec2 q = proj(ro + rdz * tp2); P2 = SA * stri(q, 1.0) * backAt(q, c2, 4.0) * ib; }
  // dust slab
  float zs = ZS * uDepth;
  {
    // near the canonical direction the texture lookups hardly move along the ray: march only the
    // local thickness of the cloud (exact sampling of thin edges, ~3× finer steps)
    vec3 pa = ro + rdz * ((zs - ro.z) / rz), pb = ro + rdz * ((-zs - ro.z) / rz);
    vec2 qa = projD(pa), qb = projD(pb), q0 = 0.5 * (qa + qb);
    float drift = length(qa - qb);
    if (drift < 0.25){
      vec2 uc = uvOf(uMC, q0), uw = uvOf(uMW, q0);
      float hc = ew(uc, 0.0) > 0.0 ? textureLod(tCD, uc, 3.0).g : 0.0;
      float hw = ew(uw, 0.0) > 0.0 ? textureLod(tWD, uw, 3.0).g : 0.0;
      float h0 = max(hc, hw) * 2.0 * uDepth;
      zs = min(zs, h0 * 1.35 + 0.06 + 2.0 * drift);
    }
  }
  float ta = (-zs - ro.z) / rz, tb = (zs - ro.z) / rz;
  float tn = max(min(ta, tb), 0.0), tf = max(ta, tb);
  // planes in front of the slab (camera behind the sheet) are not dimmed by the dust
  if (tp0 > 0.0 && tp0 < tn) { I += P0; P0 = vec3(0.0); }
  if (tp1 > 0.0 && tp1 < tn) { I += P1; P1 = vec3(0.0); }
  if (tp2 > 0.0 && tp2 < tn) { I += P2; P2 = vec3(0.0); }
  if (tf > tn){
    // a very oblique ray crosses kilometres of slab: cap the marched length (far parts are faint anyway)
    tf = min(tf, tn + 60.0);
    float dt = (tf - tn) / float(STEPS);
    float t = tn + dt * ign();
    vec3 kx = uKext * uDust;
    for (int i = 0; i < STEPS; i++){
      vec3 p = ro + rd * t;
      vec2 q = projD(p);
      float tau, h, rim; vec3 L;
      dustAt(q, tau, h, rim, L);
      h *= uDepth;
      // 3-D structure: the mid-plane wanders and the density is modulated by clumps that are small
      // along z (they average out along the canonical line of sight, which stays photometric)
      float zc = 0.18 * h * sin(q.x * 2.3 + 1.7 * sin(q.y * 1.9));
      float u = (p.z - zc) / h;
      if (abs(u) < 1.0 && (tau > 0.003 || dot(L, L) > 1e-8)){
        float n3 = 0.62 * snoise(vec3(q * 5.0, p.z * 8.0 + 5.0));
        float w = (1.0 - u * u); w = w * w * 0.9375 / h * (1.0 + (0.35 + 0.25 * uObl) * n3);
        vec3 dtau = kx * (tau * w * dt);
        vec3 em = L * (w * dt);
#if MODE == 0
        em *= mix(1.0, uIon, rim) * uDust;
#else
        em *= mix(1.0, uIon, 0.6) * uDust;
#endif
        vec3 a = exp(-dtau);
        I += T * em * mix(vec3(1.0), (1.0 - a) / max(dtau, vec3(1e-5)), step(vec3(1e-4), dtau));
        T *= a;
        if (max(T.r, max(T.g, T.b)) < 0.003) break;
      }
      t += dt;
    }
  }
  I += (P0 + P1 + P2) * T;
  // beyond the mosaics: the ordinary sky, faint
  float cov = max(c0, max(c1, c2));
  I += sky(rd, 1.0) * 0.35 * (1.0 - cov) * T;
  // outside the sharp frames (ESO / Hubble) the mosaics have lost their faint stars: a distant
  // procedural field keeps the star density continuous
  {
    vec2 qb = proj(ro + rdz * max((ZB1 - ro.z) / rz, 0.0));
#if MODE == 0
    float inner = ew(uvOf(uMC, qb), FC) * uCvis;
#else
    float inner = ew(uvOf(uMH, qb), 0.04);
#endif
    vec3 st = starLayer(rd, 260.0, 0.22, 0.10, 5.0, 1.0) * 0.5 + starLayer(rd, 600.0, 0.3, 0.12, 6.0, 1.0) * 0.18;
    I += st * (1.0 - inner) * cov * T * (MODE == 0 ? vec3(1.0) : vec3(0.9, 0.95, 1.1));
  }
  gl_FragColor = vec4(I, 1.0);
}`;

// stars: point sprites with the image's photometry; dimmed by the dust column they actually cross
const STAR_VERT = /* glsl */`
uniform sampler2D tCD, tWD;
uniform mat3 uMC, uMW;
uniform vec3 uKext;
uniform float uDust, uPixAng, uSpikes, uGain, uOrtho, uCvis;
uniform vec3 uPC;
attribute vec3 aE;
varying vec3 vCol;
varying float vSig, vSize, vSpk;
float ew(vec2 uv, float f){ vec2 e = min(uv, 1.0 - uv); return smoothstep(0.0, f, min(e.x, e.y)); }
float tauAt(vec2 q){
  vec2 uc = (uMC * vec3(q, 1.0)).xy, uw = (uMW * vec3(q, 1.0)).xy;
  float wc = ew(uc, 0.1) * uCvis, ww = ew(uw, 0.015);
  vec3 dw = textureLod(tWD, uw, 0.0).rgb, dc = textureLod(tCD, uc, 0.0).rgb;
  return mix(dw.r * dw.r * 10.0 * ww, dc.r * dc.r * 10.0, wc);
}
void main(){
  // position = (x, y) where the photograph shows the star, z = its depth; placed on the same
  // (view-dependent) extrusion rays as the nebula so stars and gas move together
  vec3 P = vec3(uPC.xy + (position.xy - uPC.xy) * ((uPC.z - position.z * (1.0 - uOrtho)) / (abs(uPC.z) < 0.05 ? 0.05 : uPC.z)), position.z);
  vec4 mv = modelViewMatrix * vec4(P, 1.0);
  float r = length(mv.xyz);
  vec3 phi = aE / (r * r);
  // dust between camera and star: the column τ_V where the segment crosses the mid-plane z = 0
  vec3 c = cameraPosition;
  float tau = 0.0;
  if (c.z * P.z < 0.0){
    float s = c.z / (c.z - P.z);
    tau = tauAt(mix(c.xy, P.xy, s));
  }
  phi *= exp(-uKext * tau * uDust);
  float l = dot(phi, vec3(0.3, 0.5, 0.2)) * uGain;
  float sig = clamp(0.9 + 0.5 * log2(1.0 + l * 1.0e4), 0.9, 5.5);
  float spk = uSpikes > 0.5 ? clamp(sqrt(l * 3.0e4), 0.0, 220.0) : 0.0;
  vSig = sig; vSpk = spk;
  vSize = min(2.0 * (3.2 * sig + 1.5 + spk), 600.0);
  float sa = sig * uPixAng;
  vCol = phi * uGain / (6.2831853 * sa * sa);
  gl_PointSize = vSize;
  gl_Position = projectionMatrix * mv;
  if (l < 1e-9) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;
const STAR_FRAG = /* glsl */`
uniform float uSpikes;
varying vec3 vCol;
varying float vSig, vSize, vSpk;
void main(){
  vec2 d = (gl_PointCoord - 0.5) * vSize;
  float r2 = dot(d, d);
  float f = exp(-0.5 * r2 / (vSig * vSig)) + 0.012 / (1.0 + r2 / (vSig * vSig * 3.0));
  if (vSpk > 0.0){
    // diffraction spikes: Hubble (4, ×-shaped) or JWST (6 + 2 short horizontal)
    float s = 0.0;
    float L = vSpk;
    if (uSpikes < 1.5){
      for (int k = 0; k < 2; k++){
        float a = 0.785398 + float(k) * 1.570796;
        vec2 e = vec2(cos(a), sin(a));
        float al = abs(dot(d, e)), pr = abs(d.x * e.y - d.y * e.x);
        s += exp(-pr * pr / 0.9) * exp(-al / (0.35 * L)) ;
      }
    } else {
      for (int k = 0; k < 4; k++){
        float a = k == 3 ? 0.0 : 1.570796 + float(k) * 1.047198;
        vec2 e = vec2(cos(a), sin(a));
        float al = abs(dot(d, e)), pr = abs(d.x * e.y - d.y * e.x);
        s += exp(-pr * pr / 0.8) * exp(-al / ((k == 3 ? 0.12 : 0.4) * L));
      }
    }
    f += s * 0.06 / (vSig * vSig);
  }
  // fade to zero at the sprite's edge so bright stars never show its square outline
  f *= 1.0 - smoothstep(0.75, 1.0, 2.0 * length(gl_PointCoord - 0.5));
  gl_FragColor = vec4(vCol * f, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------
const stepsFor = q => (q.steps <= 200 ? 40 : q.steps <= 320 ? 52 : q.steps <= 480 ? 64 : 80);
const vi = (x, d = 1) => dec(x, d);
const ly = (x, d = 1) => vi(x, d) + t(' năm ánh sáng', ' light-years');
const MODES = { vis: 0, ir: 1, jwst: 2 };
const hash = i => { const s = Math.sin(i * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

function loadTex(url) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    tex.colorSpace = THREE.NoColorSpace;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = 4;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(t('không tải được ', 'could not load ') + url))));
}
async function loadStars(name) {
  const r = await fetch(ASSET(`stars_${name}.bin`));
  if (!r.ok) throw new Error(t('không tải được stars_', 'could not load stars_') + name);
  return new Float32Array(await r.arrayBuffer());
}
// keep a CPU copy of a dust texture (τ_V in R) for click measurements
function cpuImage(tex) {
  const img = tex.image, cv = document.createElement('canvas');
  cv.width = Math.min(img.width, 1024); cv.height = Math.round(img.height * cv.width / img.width);
  const g = cv.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0, cv.width, cv.height);
  return { w: cv.width, h: cv.height, data: g.getImageData(0, 0, cv.width, cv.height).data };
}

// star catalogue → points. Each star: x, y (sky plane, ly), F rgb (radiance·ly² as photographed), τ_V.
// Depth is random (σ Ori cluster members and field stars within a few ly); a star seen through thick
// dust must be in front of it. Behind-the-dust stars get their extinction undone so the canonical view
// matches the photograph and the star brightens when you move to look around the cloud.
function buildStars(arr, kext, pc) {
  const n = arr.length / 6, pos = new Float32Array(n * 3), E = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = arr[i * 6], y = arr[i * 6 + 1], tv = arr[i * 6 + 5];
    const h1 = hash(i * 3.1 + x * 7.7), h2 = hash(i * 1.7 + y * 3.3);
    let z;
    if (tv > 2.0 || h1 < 0.42) z = 2.1 + 2.6 * h2;      // in front of the cloud
    else z = -3.7 - 4.0 * h2;                           // behind IC 434's emitting layer
    const s = (pc[2] - z) / pc[2];
    const px = pc[0] + (x - pc[0]) * s, py = pc[1] + (y - pc[1]) * s;
    pos.set([x, y, z], i * 3);
    const rc2 = (px - pc[0]) ** 2 + (py - pc[1]) ** 2 + (z - pc[2]) ** 2;
    const k = rc2 / (pc[2] * pc[2]);
    for (let c = 0; c < 3; c++) {
      const un = z < 0 ? Math.exp(Math.min(tv * kext[c], 3.5)) : 1;
      E[i * 3 + c] = arr[i * 6 + 2 + c] * k * un;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aE', new THREE.BufferAttribute(E, 3));
  return g;
}
// a named bright star as a 1-point geometry (energy E in radiance·ly²)
function brightStar(p, rgb, e) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(p), 3));
  g.setAttribute('aE', new THREE.BufferAttribute(new Float32Array(rgb.map(c => c * e)), 3));
  return g;
}

// camera flights: target, distance, polar, azimuth and fov interpolated together
function makeFlyer() {
  let fly = null;
  const sph = new THREE.Spherical(), off = new THREE.Vector3();
  return {
    to(ctx, { target, dist, polar, azim, fov, up, dur = 2.6 }) {
      off.copy(ctx.camera.position).sub(ctx.controls.target); sph.setFromVector3(off);
      const from = { r: sph.radius, p: sph.phi, a: sph.theta, t: ctx.controls.target.clone(), f: ctx.camera.fov, u: ctx.camera.up.clone() };
      const to = { r: dist ?? sph.radius, p: THREE.MathUtils.degToRad(polar ?? THREE.MathUtils.radToDeg(sph.phi)),
        a: THREE.MathUtils.degToRad(azim ?? THREE.MathUtils.radToDeg(sph.theta)),
        t: target ? new THREE.Vector3(...target) : from.t.clone(), f: fov ?? from.f, u: new THREE.Vector3(...(up || [0, 1, 0])) };
      let da = to.a - from.a; da = Math.atan2(Math.sin(da), Math.cos(da)); to.a = from.a + da;
      fly = { from, to, t: 0, dur };
    },
    active: () => !!fly,
    update(ctx, dt) {
      if (!fly) return;
      fly.t += dt; const k = Math.min(1, fly.t / fly.dur), e = k * k * (3 - 2 * k);
      const f = fly.from, t = fly.to;
      // fly distance in log space so long zooms feel even
      const r = Math.exp(Math.log(f.r) + (Math.log(t.r) - Math.log(f.r)) * e);
      sph.set(r, f.p + (t.p - f.p) * e, f.a + (t.a - f.a) * e);
      ctx.controls.target.lerpVectors(f.t, t.t, e);
      ctx.camera.up.lerpVectors(f.u, t.u, e).normalize();
      ctx.camera.position.setFromSpherical(sph).add(ctx.controls.target);
      ctx.camera.lookAt(ctx.controls.target);
      ctx.setFov(f.f + (t.f - f.f) * e);
      if (k >= 1) fly = null;
    },
  };
}

// ------------------------------------------------------------------------------------------
// scene
// ------------------------------------------------------------------------------------------
const VIEW_LABEL = {
  vis: () => t('khả kiến B·V·R (0,44–0,66 µm) · ESO VLT/FORS2', 'visible B·V·R (0.44–0.66 µm) · ESO VLT/FORS2'),
  ir: () => t('hồng ngoại gần 1,1–1,6 µm · Hubble WFC3/IR + Euclid', 'near-infrared 1.1–1.6 µm · Hubble WFC3/IR + Euclid'),
  jwst: () => t('hồng ngoại 1,9–4,4 µm · JWST NIRCam', 'infrared 1.9–4.4 µm · JWST NIRCam'),
};
const VIEWS = {
  vis: { target: [0, 0, 0], dist: D0, polar: 90, azim: 0, fov: FOV0 },
  ir: { target: [0, 0, 0], dist: D0, polar: 90, azim: 0, fov: FOV0 },
  jwst: { target: [J_CENTER[0], J_CENTER[1], 0], dist: J_D, polar: 90, azim: 0, fov: J_FOV, up: UP_J },
  wide: { target: [-5.5, 5.0, 0], dist: 36, polar: 90, azim: 0, fov: 40 },
};

const horsehead = {
  id: 'dark-nebula',
  name: 'Tinh vân Đầu Ngựa',
  category: 'nebulae',
  order: 40,
  subtitle: 'Tinh vân tối Barnard 33 trước nền IC 434 (Lạp Hộ)',
  swatch: ['#e0607a', '#160a0e'],
  blurb: 'Một cột bụi và khí phân tử lạnh (~10–20 K) nhô ra từ đám mây Orion B. Bản thân nó gần như không phát sáng: ta thấy <b>bóng</b> của nó in trên tấm khí hydro IC 434 đang phát Hα do sao σ Orionis ion hóa. Hình 3-D ở đây được dựng trực tiếp từ ảnh thật: độ sâu quang học của bụi đo từ độ tối của bóng, nên góc nhìn chuẩn khớp với ảnh, còn khi xoay quanh bạn thấy thị sai thật.',
  intro: 'Nhấn <b>I</b> để xem bằng hồng ngoại (Hubble): bụi trở nên trong suốt. <b>J</b>: cận cảnh JWST. <b>W</b>: toàn cảnh với σ Orionis, Alnitak và tinh vân Ngọn Lửa. Click vào bụi để đo A<sub>V</sub>.',
  facts: [
    ['Khoảng cách', '≈ 1 375 năm ánh sáng'],
    ['Chòm sao', 'Lạp Hộ, cạnh sao Alnitak'],
    ['Cao (đầu + cổ)', '≈ 3,5 năm ánh sáng (≈ 8′)'],
    ['Sao ion hóa', 'σ Orionis (O9.5 V, ≈ 32 000 K), cách ≈ 14 ly'],
    ['Nhiệt độ lõi', '≈ 10–20 K · n<sub>H</sub> ~ 10⁵ cm⁻³'],
    ['Tuổi thọ còn lại', '≈ 5 triệu năm (bị UV bào mòn)'],
    ['Ảnh gốc', 'ESO · ESA/Hubble · ESA/Webb · ESA/Euclid · G. Lacroce (Hα)'],
  ],
  en: {
    name: 'Horsehead Nebula',
    subtitle: 'Dark nebula Barnard 33 against IC 434 (Orion)',
    blurb: 'A pillar of cold (~10–20 K) dust and molecular gas jutting out of the Orion B cloud. It gives off almost no light of its own: what we see is its <b>silhouette</b> against the sheet of hydrogen in IC 434, glowing in Hα because σ Orionis ionizes it. The 3D shape here is built directly from real images: the dust’s optical depth is measured from how dark the silhouette is, so the reference view matches the photographs, while orbiting around reveals real parallax.',
    intro: 'Press <b>I</b> to see it in infrared (Hubble): the dust turns transparent. <b>J</b>: JWST close-up. <b>W</b>: wide view with σ Orionis, Alnitak and the Flame Nebula. Click on the dust to measure A<sub>V</sub>.',
    facts: [
      ['Distance', '≈ 1,375 light-years'],
      ['Constellation', 'Orion, next to Alnitak'],
      ['Height (head + neck)', '≈ 3.5 light-years (≈ 8′)'],
      ['Ionizing star', 'σ Orionis (O9.5 V, ≈ 32,000 K), ≈ 14 ly away'],
      ['Core temperature', '≈ 10–20 K · n<sub>H</sub> ~ 10⁵ cm⁻³'],
      ['Remaining lifetime', '≈ 5 million years (eroded by UV)'],
      ['Source images', 'ESO · ESA/Hubble · ESA/Webb · ESA/Euclid · G. Lacroce (Hα)'],
    ],
  },
  camera: { distance: D0, polar: 90, azimuth: 0, min: 0.6, max: 120, fov: FOV0, target: [0, 0, 0] },
  exposure: 1.0,
  bloom: 0.22,
  params: [
    { key: 'view', label: 'Ảnh / bước sóng', type: 'seg', options: [['vis', 'Khả kiến (ESO VLT)'], ['ir', 'Hồng ngoại (Hubble 2013)'], ['jwst', 'JWST (2024)']], default: 'vis',
      note: 'Hồng ngoại: A<sub>H</sub>/A<sub>V</sub> ≈ 0,18 — bụi gần trong suốt và tự phát sáng (ánh sao tán xạ + PAH).',
      en: { label: 'Image / wavelength', note: 'Infrared: A<sub>H</sub>/A<sub>V</sub> ≈ 0.18 — the dust is nearly transparent and glows on its own (scattered starlight + PAHs).', options: ['Visible (ESO VLT)', 'Infrared (Hubble 2013)', 'JWST (2024)'] } },
    { key: 'dust', label: 'Mật độ bụi', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      en: { label: 'Dust density' } },
    { key: 'ion', label: 'Độ sáng IC 434 (σ Ori)', type: 'range', min: 0, max: 2.5, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Bức xạ UV của σ Orionis ion hóa mặt đám mây: tấm IC 434 và viền sáng trên bờm.',
      en: { label: 'IC 434 brightness (σ Ori)', note: 'UV light from σ Orionis ionizes the face of the cloud: the IC 434 sheet and the bright rim along the mane.' } },
    { key: 'depth', label: 'Độ dày theo hướng nhìn', type: 'range', min: 0.2, max: 2, step: 0.01, default: 0.6, fmt: v => '×' + v.toFixed(2),
      note: 'Ảnh chỉ cho biết tổng lượng bụi dọc tia nhìn, không cho độ sâu. ×1 = sâu bằng bề rộng; mặc định ×0,6 (mô hình vùng PDR của Habart và cs. 2005 cho mép sáng dày ~0,1 pc theo hướng nhìn). Góc nhìn chuẩn không đổi — hãy xoay để thấy khác biệt.',
      en: { label: 'Depth along the line of sight', note: 'The images only give the total dust column along each sightline, not its depth. ×1 = as deep as it is wide; the default ×0.6 follows the PDR model of Habart et al. 2005, where the bright edge is ~0.1 pc thick along the line of sight. The reference view doesn’t change — orbit around to see the difference.' } },
    { key: 'pal', label: 'Màu (khả kiến)', type: 'seg', options: [['auto', 'Tự động'], ['eso', 'ESO B·V·R'], ['ha', 'Hα thật']], default: 'auto',
      en: { label: 'Colour (visible)', options: ['Auto', 'ESO B·V·R', 'True Hα'] } },
  ],
  actions: [
    { label: 'Khả kiến', key: 'v', title: 'Ảnh ESO VLT (B, V, R)', run: ctx => ctx.scene._go?.(ctx, 'vis'),
      en: { label: 'Visible', title: 'ESO VLT image (B, V, R)' } },
    { label: 'Hồng ngoại', key: 'i', title: 'Hubble WFC3/IR 2013 + Euclid 2023', run: ctx => ctx.scene._go?.(ctx, 'ir'),
      en: { label: 'Infrared', title: 'Hubble WFC3/IR 2013 + Euclid 2023' } },
    { label: 'JWST', key: 'j', title: 'Cận cảnh bờm ngựa — JWST NIRCam 2024', run: ctx => ctx.scene._go?.(ctx, 'jwst'),
      en: { label: 'JWST', title: 'Close-up of the mane — JWST NIRCam 2024' } },
    { label: 'Toàn cảnh', key: 'w', title: 'σ Orionis, Alnitak, NGC 2023, tinh vân Ngọn Lửa', run: ctx => ctx.scene._go?.(ctx, 'wide'),
      en: { label: 'Wide view', title: 'σ Orionis, Alnitak, NGC 2023, Flame Nebula' } },
    { label: 'Nhìn nghiêng', key: 'o', title: 'Bay tới mép góc nhìn cho phép (±32°, JWST ±12°) để thấy chiều sâu; nhấn lần nữa để quay về', run: ctx => ctx.scene._go?.(ctx, 'oblique'),
      en: { label: 'Oblique view', title: 'Fly to the edge of the allowed viewing angle (±32°, JWST ±12°) to see the depth; press again to go back' } },
  ],
  create(ctx) {
    const P = ctx.params;
    const own = {
      tCD: { value: null }, tCL: { value: null }, tCB: { value: null }, tWD: { value: null }, tWL: { value: null }, tWB: { value: null }, tWG: { value: null }, tF: { value: null },
      tHB: { value: null }, tHL: { value: null }, tEB: { value: null }, tEL: { value: null }, tJB: { value: null }, tJL: { value: null }, tEG: { value: null }, tHG: { value: null },
      uMC: { value: xfMat(XF.C) }, uMW: { value: xfMat(XF.W) }, uMF: { value: xfMat(XF.F) },
      uMH: { value: xfMat(XF.H) }, uME: { value: xfMat(XF.E) }, uMJ: { value: xfMat(XF.J) },
      uDust: { value: 1 }, uIon: { value: 1 }, uDepth: { value: 1 }, uPal: { value: 0 }, uReady: { value: 0 }, uCvis: { value: 1 },
      uPC: { value: new THREE.Vector3(0, 0, D0) }, uOrtho: { value: 0 }, uOrthoD: { value: 0 }, uObl: { value: 0 }, uKext: { value: new THREE.Vector3(...KEXT.vis) },
    };
    const uniforms = { ...ctx.shared, ...own };
    const modeOf = v => MODES[v] ?? 0;
    const material = new THREE.ShaderMaterial({
      uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
      defines: { STEPS: stepsFor(ctx.quality), MODE: modeOf(P.view) },
    });
    const setQuality = q => { material.defines.STEPS = stepsFor(q); material.needsUpdate = true; };

    // stars
    const starU = {
      tCD: own.tCD, tWD: own.tWD, uMC: own.uMC, uMW: own.uMW, uKext: own.uKext, uDust: own.uDust,
      uPixAng: { value: 0.001 }, uSpikes: { value: 0 }, uGain: { value: 1 }, uPC: own.uPC, uOrtho: own.uOrtho, uCvis: own.uCvis,
    };
    const starMat = new THREE.ShaderMaterial({
      uniforms: starU, vertexShader: STAR_VERT, fragmentShader: STAR_FRAG,
      blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, transparent: true,
    });
    const overlay = new THREE.Group();
    const sets = {};
    const PC_J = [J_CENTER[0], J_CENTER[1], J_D];
    // σ Orionis (V = 3.8, O9.5 V) and Alnitak (V = 1.8, O9.5 Iab): energies tuned so that, like in
    // the photographs, they saturate into glare; the mosaics already contain their reflection halos
    const sOri = new THREE.Points(brightStar(SIGMA_ORI, [0.78, 0.86, 1.0], 6.0), starMat);
    const alnitak = new THREE.Points(brightStar(ALNITAK, [0.76, 0.84, 1.0], 26.0), starMat);
    for (const o of [sOri, alnitak]) { o.frustumCulled = false; overlay.add(o); }

    // assets
    const T = {};
    let cpuC = null, cpuW = null;
    const files = {
      tCD: 'vis_c_d.jpg', tCL: 'vis_c_l.jpg', tCB: 'vis_c_b.jpg', tWD: 'vis_w_d.jpg', tWL: 'vis_w_l.jpg', tWB: 'vis_w_b.jpg', tWG: 'vis_w_gain.png', tF: 'vis_far.jpg',
      tHB: 'ir_h_b.jpg', tHL: 'ir_h_l.jpg', tEB: 'ir_e_b.jpg', tEL: 'ir_e_l.jpg', tJB: 'ir_j_b.jpg', tJL: 'ir_j_l.jpg', tEG: 'ir_e_gain.png', tHG: 'ir_h_gain.png',
    };
    const ready = Promise.all([
      ...Object.entries(files).map(([k, f]) => loadTex(ASSET(f)).then(t => { own[k].value = t; T[k] = t; })),
      ...['vis', 'ir', 'jwst'].map(n => loadStars(n).then(a => { sets[n] = a; })),
    ]).then(() => {
      for (const [n, a] of Object.entries(sets)) {
        const pts = new THREE.Points(buildStars(a, KEXT[n], n === 'jwst' ? PC_J : [0, 0, D0]), starMat);
        pts.frustumCulled = false; pts.visible = false; sets[n] = pts; overlay.add(pts);
      }
      try { cpuC = cpuImage(T.tCD); cpuW = cpuImage(T.tWD); } catch (e) { /* click measurement unavailable */ }
      own.uReady.value = 1;
    });

    // camera
    const flyer = makeFlyer();
    let curMode = -1, firstFrame = true, lastView = P.view;
    const scene = ctx.scene;
    const setView = (c, v) => { c.params.view = v; c.syncUI?.(); };
    scene._go = (c, which) => {
      if (which === 'oblique') {
        const off = c.camera.position.clone().sub(c.controls.target);
        const az = THREE.MathUtils.radToDeg(Math.atan2(off.x, off.z));
        // swing to the edge of the allowed cone (alternating sides), or back to the line of sight
        const side = az > 1 ? -1 : 1, edge = (c.params.view === 'jwst' ? ORBIT_AZ_J : ORBIT_AZ) - 2;
        const atEdge = Math.abs(az) > edge * 0.6;
        flyer.to(c, { polar: atEdge ? 90 : (c.params.view === 'jwst' ? 87 : 84), azim: atEdge ? 0 : side * edge, up: c.camera.up.toArray(), dur: 3.2 });
        return;
      }
      // the wide field exists only as an Hα (visible) mosaic
      setView(c, which === 'wide' ? 'vis' : which);
      const v = VIEWS[which];
      flyer.to(c, { ...v, dur: which === 'wide' ? 3.6 : 2.8 });
      lastView = c.params.view;
    };
    const tmp = new THREE.Vector3(), sz = new THREE.Vector2();
    let saved = null;
    const setLimits = (c, az, pol) => {
      const k = c.controls;
      k.minAzimuthAngle = -az * DEG; k.maxAzimuthAngle = az * DEG;
      k.minPolarAngle = (90 - pol) * DEG; k.maxPolarAngle = (90 + pol) * DEG;
    };

    return {
      material, overlay, setQuality, ready,
      enter(c) {
        firstFrame = true;
        const k = c.controls;
        saved = [k.minAzimuthAngle, k.maxAzimuthAngle, k.minPolarAngle, k.maxPolarAngle];
        setLimits(c, ORBIT_AZ, ORBIT_POL);
      },
      exit(c) {
        c.camera.up.set(0, 1, 0);
        if (saved) [c.controls.minAzimuthAngle, c.controls.maxAzimuthAngle, c.controls.minPolarAngle, c.controls.maxPolarAngle] = saved;
      },
      update(dt, c) {
        const p = c.params;
        // a view picked in the panel (or by a deep link / snapshot): move the orbit target to it,
        // keeping the current viewing offset
        if (firstFrame || p.view !== lastView) {
          const want = p.view === 'jwst' ? VIEWS.jwst : VIEWS.vis;
          const wasJ = lastView === 'jwst', isJ = p.view === 'jwst';
          if (firstFrame && isJ) {
            tmp.copy(c.camera.position).sub(c.controls.target).multiplyScalar(J_D / D0);
            c.controls.target.set(...want.target); c.camera.position.copy(c.controls.target).add(tmp);
            c.camera.up.set(...UP_J); c.camera.lookAt(c.controls.target); c.setFov(J_FOV);
          } else if (!firstFrame && wasJ !== isJ && !flyer.active()) {
            flyer.to(c, { ...want, polar: undefined, azim: undefined });
          }
          lastView = p.view; firstFrame = false;
        }
        flyer.update(c, dt);
        {
          // the allowed cone widens smoothly as the camera backs off toward the wide field
          const w = THREE.MathUtils.smoothstep(c.camera.position.distanceTo(c.controls.target), 12, 30);
          const [a0, p0] = p.view === 'jwst' ? [ORBIT_AZ_J, ORBIT_POL_J] : [ORBIT_AZ, ORBIT_POL];
          setLimits(c, a0 + (ORBIT_AZ_WIDE - a0) * w, p0 + (ORBIT_POL_WIDE - p0) * w);
        }
        const mode = modeOf(p.view);
        if (mode !== curMode) {
          curMode = mode; material.defines.MODE = mode; material.needsUpdate = true;
          own.uKext.value.set(...KEXT[p.view]);
          starU.uSpikes.value = mode === 0 ? 0 : mode === 1 ? 1 : 2;
          sOri.visible = alnitak.visible = mode === 0;
        }
        for (const n of ['vis', 'ir', 'jwst']) if (sets[n] && sets[n].isPoints) sets[n].visible = n === p.view;
        // obliqueness of the view w.r.t. the line of sight of the photographs
        tmp.copy(c.camera.position).sub(c.controls.target).normalize();
        own.uObl.value = THREE.MathUtils.smoothstep(1 - Math.abs(tmp.z), 0.015, 0.2);
        own.uPC.value.copy(c.camera.position);
        own.uOrtho.value = Math.max(THREE.MathUtils.smoothstep(1 - Math.abs(tmp.z), 0.004, 0.12),
          // (inside or next to the dust slab the camera-centred extrusion degenerates)
          1 - THREE.MathUtils.smoothstep(Math.abs(c.camera.position.z), 1.95 * p.depth + 0.15, 1.95 * p.depth + 0.8));
        own.uOrthoD.value = 0.5 * own.uOrtho.value;
        own.uDust.value = p.dust; own.uIon.value = p.ion; own.uDepth.value = p.depth;
        const d = c.camera.position.distanceTo(c.controls.target);
        const autoPal = THREE.MathUtils.smoothstep(d, 12, 30);
        own.uPal.value = p.pal === 'eso' ? 0 : p.pal === 'ha' ? 1 : autoPal;
        // in wide views the mosaic's own (consistent) horse replaces the ESO frame, which is then a few pixels
        own.uCvis.value = 1 - THREE.MathUtils.smoothstep(d, 14, 26);
        c.renderer.getDrawingBufferSize(sz);
        starU.uPixAng.value = 2 * Math.tan(THREE.MathUtils.degToRad(c.camera.fov) / 2) / Math.max(sz.y, 1);
      },
      onClick(info, c) {
        // measure the dust column where the click ray meets the sky plane z = 0
        if (!cpuC || Math.abs(info.dir.z) < 1e-3) return;
        const t = -info.origin.z / info.dir.z;
        if (t <= 0) return;
        const x = info.origin.x + info.dir.x * t, y = info.origin.y + info.dir.y * t;
        const samp = (img, r) => {
          const [u, v] = uvOf(r, x, y);
          if (u < 0 || u > 1 || v < 0 || v > 1) return null;
          const i = (Math.min(img.h - 1, Math.floor((1 - v) * img.h)) * img.w + Math.min(img.w - 1, Math.floor(u * img.w))) * 4;
          const e = img.data[i] / 255; return e * e * 10;
        };
        let tau = samp(cpuC, XF.C); if (tau === null) tau = samp(cpuW, XF.W);
        if (tau === null) { c.toast(t('Ngoài vùng có dữ liệu bụi.', 'Outside the area with dust data.'), 2500); return; }
        tau *= c.params.dust;
        const pct = k => Math.round(100 * Math.exp(-tau * k));
        c.toast(t(`Tại (${vi(x, 2)}; ${vi(y, 2)}) ly: τ<sub>V</sub> ≈ ${vi(tau, 1)} → A<sub>V</sub> ≈ ${vi(1.086 * tau, 1)} mag.<br>Ánh sáng nền lọt qua: Hα ${pct(0.8)} % · 1,6 µm ${pct(0.18)} % · 4 µm ${pct(0.05)} %`,
          `At (${vi(x, 2)}, ${vi(y, 2)}) ly: τ<sub>V</sub> ≈ ${vi(tau, 1)} → A<sub>V</sub> ≈ ${vi(1.086 * tau, 1)} mag.<br>Background light getting through: Hα ${pct(0.8)} % · 1.6 µm ${pct(0.18)} % · 4 µm ${pct(0.05)} %`), 6000);
        c.audio?.click?.(0.3, info.ndcX, Math.min(1, tau / 6));
      },
      hud(c) {
        const d = c.camera.position.distanceTo(c.controls.target);
        const asp = c.renderer.domElement.width / Math.max(c.renderer.domElement.height, 1);
        const wv = 2 * d * Math.tan(THREE.MathUtils.degToRad(c.camera.fov) / 2) * asp;
        const k = KEXT[c.params.view][0];
        const tau = TAU_HEAD * c.params.dust;
        return [
          [t('Ánh sáng', 'Light'), VIEW_LABEL[c.params.view]()],
          [t('Camera cách đầu ngựa', 'Camera to the Horsehead'), ly(c.camera.position.length())],
          [t('Bề rộng khung nhìn', 'Field of view width'), ly(wv)],
          [t('A<sub>V</sub> qua đầu ngựa', 'A<sub>V</sub> through the head'), '≳ ' + vi(1.086 * tau) + t(' mag (ảnh bão hòa)', ' mag (image saturated)')],
          [t('Ánh nền xuyên qua đầu', 'Background light through the head'), '≈ ' + vi(100 * Math.exp(-tau * k), 1) + ' %'],
        ];
      },
    };
  },
};

export default [horsehead];
