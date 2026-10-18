// Sao Thiên Vương — Uranus, from measured data.
//
// One scene unit = Uranus' equatorial radius (25 559 km at 1 bar). Supersedes the 'uranus' entry in giants.js.
//
// What is real here, and where it comes from (see assets/uranus/CREDITS.txt):
//   * Colour, brightness and the seasonal polar cap: zonal I/F profiles measured on the Hubble OPAL global maps
//     of Oct 2025 (WFC3/UVIS F467M, F547M, F657N, Minnaert-corrected by the OPAL team; CC BY 4.0). Uranus is
//     nearly featureless at HST resolution, so the maps reduce to a latitude profile per filter (variance along
//     longitude < 0.6 %); that profile is split into a 'base' (≤ 27°) and the 2025 north polar cap excess, and the
//     cap is re-scaled per hemisphere with the season (see capAmp()). Per-filter Minnaert exponents k = 0.57 /
//     0.80 / 0.85 (red / green / blue) are the ones OPAL measured for those filters.
//   * Discrete storms: positions, latitudes and sizes of the bright features found in the OPAL F845M maps of
//     Nov 2014 and Oct 2025 (and the Keck 2014 / HST 2006 / JWST 2023 features named in the literature).
//     The 2025 storm at 35.6°N moved 20° in longitude between the two OPAL rotations (20.4 h apart) — the same
//     rate the wind profile below predicts, which also fixes the sign convention used here.
//   * Zonal winds: the 18th-order symmetric Legendre fit of Sromovsky et al. (2015, Icarus 258, 192), drift rate
//     in °/h relative to the interior, from 846 Keck/Gemini/HST cloud features: −40 m/s at the equator, +258 m/s
//     near 55°, and solid-body rotation (4.1°/h faster than the interior) from 62° to the pole.
//   * Rotation: 17 h 14 min 52 s (Lamy et al. 2025, HST auroral tracking), retrograde about the IAU north pole
//     (RA 257.311°, Dec −15.175°). Season from the planet's real orbit (JPL mean elements): the Sun direction
//     and the sub-solar latitude are computed for the chosen year.
//   * Rings: semi-major axes, eccentricities, widths of the nine narrow rings (French et al. 1991); ε, α, β, 4, 5, 6
//     are eccentric and their apses precess at the rate set by Uranus' J2 (ε: 1.36°/day). Dusty ζ, ν, μ rings.
//   * Moons: Miranda, Ariel, Umbriel, Titania, Oberon with real orbits, radii and albedos.
// Approximations are marked 'approx.' in the comments.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';

const V3 = THREE.Vector3;
const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAU = Math.PI * 2;
const RE = 25559, RP = 24973;                 // km (1 bar)
const FLAT = 1 - RP / RE;                      // 0.0229
const PROT = 17.247864 * 3600;                 // s (Lamy et al. 2025)
const OMEGA = TAU / PROT;                      // rad/s, retrograde about the IAU north pole
const GM = 5793939;                            // km³/s²
const J2 = 3510.7e-6;
const PHASE_P = 40 * 3600;                     // two-phase advection period (planet seconds)
const DAY = 86400;

const vn = (x, d = 1) => x.toFixed(d).replace('.', ',').replace('-', '−');
const fmtInt = n => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
const latStr = d => `${vn(Math.abs(d))}° ${d >= 0 ? 'B' : 'N'}`;
const wrap = a => ((a % TAU) + TAU) % TAU;
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const NOW_YEAR = 1970 + Date.now() / 31557600000;

/* ------------------------------------------------------------------------------------------------ */
/* winds: Sromovsky et al. 2015, Table 5 'Fit 1' (symmetric, planetographic latitude).               */
/* drift(x) in °/h, x = sin(lat); positive = retrograde relative to the interior. In body-fixed east  */
/* longitude (right-handed about the IAU north pole) features move at +drift °/h.                    */
/* ------------------------------------------------------------------------------------------------ */
const LEG = [-1.245225, -3.582487, -0.118185, 0.848593, 0.315199, -0.188857, -0.263077, -0.026728, 0.104192, 0.059944];
function driftDegH(x) {
  let p0 = 1, p1 = x, s = LEG[0];
  for (let n = 1; n < 18; n++) {
    const p2 = ((2 * n + 1) * x * p1 - n * p0) / (n + 1); p0 = p1; p1 = p2;
    if ((n + 1) % 2 === 0) s += LEG[(n + 1) / 2] * p2;
  }
  return s;
}
const driftRad = latD => driftDegH(Math.sin(latD * D2R)) * D2R / 3600;        // rad/s of east longitude
// wind speed (m/s, + = prograde = with the planet's rotation) from the drift rate (their Eq. 3–4)
function windMS(latD) {
  const th = latD * D2R;
  const R = Math.abs(latD) > 89.99 ? 0 : RE / Math.sqrt(1 + (RP / RE) ** 2 * Math.tan(th) ** 2);
  return -4.8481e-3 * R * driftDegH(Math.sin(th));
}

/* ------------------------------------------------------------------------------------------------ */
/* ephemeris: JPL approximate Keplerian elements (Standish, 1800–2050) → Sun and Earth seen from Uranus */
/* ------------------------------------------------------------------------------------------------ */
function kepler(el, year) {
  const T = (year - 2000) / 100;
  const [a, e, I, L, w, O] = el.map(([v, r]) => v + r * T);
  const M = ((L - w) % 360) * D2R, om = (w - O) * D2R;
  let E = M; for (let k = 0; k < 20; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const xp = a * (Math.cos(E) - e), yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
  const co = Math.cos(om), so = Math.sin(om), cO = Math.cos(O * D2R), sO = Math.sin(O * D2R), cI = Math.cos(I * D2R), sI = Math.sin(I * D2R);
  return new V3((co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp,
                (co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp,
                so * sI * xp + co * sI * yp);   // ecliptic J2000, AU
}
const EL_URANUS = [[19.18916464, -0.00196176], [0.04725744, -0.00004397], [0.77263783, -0.00242939], [313.23810451, 428.48202785], [170.95427630, 0.40805281], [74.01692503, 0.04240589]];
const EL_EARTH = [[1.00000261, 0.00000562], [0.01671123, -0.00004392], [-0.00001531, -0.01294668], [100.46457166, 35999.37244981], [102.93768193, 0.32327364], [0, 0]];
// IAU north pole of Uranus (RA 257.311°, Dec −15.175°) in ecliptic J2000 coordinates
const POLE_E = (() => {
  const ra = 257.311 * D2R, de = -15.175 * D2R, eps = 23.43928 * D2R;
  const x = Math.cos(de) * Math.cos(ra), y = Math.cos(de) * Math.sin(ra), z = Math.sin(de);
  return new V3(x, Math.cos(eps) * y + Math.sin(eps) * z, -Math.sin(eps) * y + Math.cos(eps) * z);
})();
function geometry(year) {
  const rU = kepler(EL_URANUS, year), rE = kepler(EL_EARTH, year);
  const sun = rU.clone().negate().normalize();
  const earth = rE.clone().sub(rU).normalize();
  return { rAU: rU.length(), sun, earth, B: Math.asin(POLE_E.dot(sun)) * R2D, BE: Math.asin(POLE_E.dot(earth)) * R2D, dE: rE.distanceTo(rU) };
}
// Seasonal polar cap: brightens on the hemisphere coming out of winter, with a lag of a few years, and fades
// after its summer (approx. — calibrated on: north cap absent at the 2007 equinox, faint in 2014, strong 2020s
// (OPAL, JWST); south cap bright for Voyager 1986, faded by the early 2000s, leaving the 45°S collar).
function capAmp(year) {
  const B = geometry(year - 3).B;
  return [smooth(5, 55, B), smooth(5, 55, -B)];
}
// small convective 'popcorn' clouds of the polar region: seen in the north from 2012 on (Keck/Gemini), approx.
function popAmp(year) {
  const B = geometry(year - 1).B;
  return [smooth(-10, 15, B), smooth(-10, 15, -B)];
}

/* ------------------------------------------------------------------------------------------------ */
/* discrete storms (planetographic lat °, east lon ° at t = 0, size E–W × N–S km, visible / near-IR    */
/* amplitude, life cycle h, phase, max age of the sheared plume h)                                     */
/* ------------------------------------------------------------------------------------------------ */
const STORMS = [
  // OPAL F845M, 23 Oct 2025: the bright pair at 35.6°N and two faint features
  { lat: 35.6, lon: 301.5, ax: 1700, ay: 1200, vis: 0.10, ir: 1.6, P: 70, ph: 0.35, tail: 30 },
  { lat: 35.4, lon: 289.1, ax: 1400, ay: 1000, vis: 0.08, ir: 1.3, P: 55, ph: 0.6, tail: 25 },
  { lat: 29.1, lon: 207.5, ax: 900, ay: 700, vis: 0.04, ir: 0.6, P: 40, ph: 0.1, tail: 18 },
  { lat: 43.6, lon: 198.6, ax: 1000, ay: 750, vis: 0.04, ir: 0.7, P: 45, ph: 0.8, tail: 20 },
  // OPAL F845M, 8 Nov 2014 (the 'record' storm season, de Pater et al. 2015)
  { lat: 33.4, lon: 255.2, ax: 4200, ay: 1700, vis: 0.12, ir: 2.2, P: 400, ph: 0.3, tail: 40, long: 1 },
  { lat: 37.2, lon: 348.2, ax: 2400, ay: 1100, vis: 0.06, ir: 1.0, P: 90, ph: 0.5, tail: 30 },
  { lat: 18.9, lon: 94.4, ax: 1100, ay: 900, vis: 0.04, ir: 0.7, P: 36, ph: 0.2, tail: 16 },
  { lat: 16.7, lon: 160.1, ax: 900, ay: 700, vis: 0.03, ir: 0.6, P: 30, ph: 0.7, tail: 14 },
  { lat: 22.9, lon: 173.6, ax: 1000, ay: 800, vis: 0.03, ir: 0.5, P: 48, ph: 0.45, tail: 18 },
  { lat: 2.5, lon: 231.8, ax: 1600, ay: 1200, vis: 0.05, ir: 1.0, P: 60, ph: 0.9, tail: 24 },
  // Keck, Aug 2014: the brightest 2.2 µm storm ever seen on Uranus, ~15°N
  { lat: 15.0, lon: 30.0, ax: 1700, ay: 1300, vis: 0.07, ir: 2.0, P: 80, ph: 0.55, tail: 30 },
  // HST/Keck 2006: the Uranus Dark Spot (28°N, 1 300 × 2 700 km) and its bright companion
  { lat: 28.0, lon: 60.0, ax: 2700, ay: 1300, vis: -0.06, ir: -0.35, P: 600, ph: 0.4, tail: 0, long: 1 },
  { lat: 30.4, lon: 61.5, ax: 1300, ay: 650, vis: 0.06, ir: 1.1, P: 30, ph: 0.15, tail: 12 },
  // 'Berg' (34°S, 1994–2009) and a Voyager-era southern feature (−27°)
  { lat: -34.0, lon: 140.0, ax: 2500, ay: 1200, vis: 0.06, ir: 1.2, P: 500, ph: 0.1, tail: 30, long: 1 },
  { lat: -27.0, lon: 300.0, ax: 1200, ay: 900, vis: 0.03, ir: 0.6, P: 50, ph: 0.65, tail: 20 },
  // JWST NIRCam, Feb 2023: bright cloud at the edge of the north polar cap
  { lat: 56.0, lon: 120.0, ax: 1500, ay: 1100, vis: 0.07, ir: 1.5, P: 60, ph: 0.25, tail: 22 },
];
const NS = STORMS.length;

/* ------------------------------------------------------------------------------------------------ */
/* rings (French et al. 1991; Nicholson et al. 2018): a km, e, width at periapse / apoapse km,         */
/* equivalent width τ·w (km), apse longitude at t = 0 (deg, arbitrary)                                */
/* ------------------------------------------------------------------------------------------------ */
const RINGS = [
  ['6', 41837.2, 1.013e-3, 1.5, 2.2, 0.4, 20], ['5', 42234.8, 1.899e-3, 1.9, 4.6, 1.1, 110],
  ['4', 42570.9, 1.059e-3, 2.3, 3.0, 0.8, 250], ['α', 44718.5, 0.761e-3, 4.8, 10.0, 2.6, 330],
  ['β', 45661.0, 0.442e-3, 6.1, 11.4, 1.6, 200], ['η', 47175.9, 0.004e-3, 1.6, 1.6, 0.35, 0],
  ['γ', 47626.9, 0.109e-3, 3.6, 4.7, 5.0, 60], ['δ', 48300.1, 0.004e-3, 4.1, 6.1, 1.9, 0],
  ['λ', 50023.9, 0.0, 1.5, 2.5, 0.25, 0], ['ε', 51149.3, 7.936e-3, 19.7, 96.4, 70.0, 150],
];
// apsidal precession from J2 (the ε value, 1.354°/d, agrees with the measured 1.3616°/d to 0.6 %)
const ringPrec = a => Math.sqrt(GM / (a * a * a)) * 1.5 * J2 * (RE / a) ** 2;   // rad/s

/* the five major moons: a km, period d, radius km, geometric albedo, tint */
const MOONS = [
  ['Miranda', 129390, 1.413479, 235.8, 0.32, [0.95, 0.95, 0.95]],
  ['Ariel', 191020, 2.520379, 578.9, 0.53, [0.97, 0.96, 0.95]],
  ['Umbriel', 266300, 4.144177, 584.7, 0.26, [0.93, 0.93, 0.95]],
  ['Titania', 435910, 8.705872, 788.4, 0.35, [1.0, 0.95, 0.9]],
  ['Oberon', 583520, 13.463239, 761.4, 0.31, [1.0, 0.94, 0.88]],
];

/* ------------------------------------------------------------------------------------------------ */
/* zonal albedo profiles measured on the OPAL 2025 maps (normal albedo, mean of both rotations)       */
/* columns: |lat|°, F657N, F547M, F467M                                                              */
/* ------------------------------------------------------------------------------------------------ */
const BASE_VIS = [[0, 0.267, 0.604, 0.749], [5, 0.278, 0.6033, 0.7449], [10, 0.2811, 0.6015, 0.7461], [15, 0.2857, 0.6003, 0.7474],
  [20, 0.2949, 0.6029, 0.7502], [25, 0.3095, 0.6086, 0.7528], [27.5, 0.312, 0.609, 0.753], [90, 0.312, 0.609, 0.753]];
// 2025 north polar cap = profile − base
const CAP_VIS = [[25, 0, 0, 0], [30, 0.013, 0.005, 0.001], [35, 0.027, 0.009, 0.001], [40, 0.0477, 0.0149, 0.0008], [45, 0.0834, 0.0302, 0.008],
  [50, 0.1166, 0.0443, 0.0109], [55, 0.1423, 0.0533, 0.0081], [60, 0.1579, 0.0569, 0.0041], [65, 0.1668, 0.0617, 0.0057], [70, 0.1721, 0.0626, 0.0047],
  [75, 0.1741, 0.0619, 0.003], [80, 0.1761, 0.0611, 0.0024], [90, 0.1741, 0.0636, 0.0014]];
const ff = v => (+v).toFixed(5);
function tab3(name, pts) {
  const n = pts.length;
  return `vec3 ${name}(float x){
  const float X[${n}] = float[${n}](${pts.map(p => ff(p[0])).join(',')});
  const vec3 Y[${n}] = vec3[${n}](${pts.map(p => `vec3(${ff(p[1])},${ff(p[2])},${ff(p[3])})`).join(',')});
  if (x <= X[0]) return Y[0];
  for (int i = 1; i < ${n}; i++){ if (x < X[i]){ float u = (x - X[i-1]) / (X[i] - X[i-1]); return mix(Y[i-1], Y[i], u * u * (3.0 - 2.0 * u)); } }
  return Y[${n - 1}];
}`;
}

/* ------------------------------------------------------------------------------------------------ */
/* shader                                                                                             */
/* ------------------------------------------------------------------------------------------------ */
const FRAG = COMMON_GLSL + /* glsl */`
#define R2D 57.2957795
#define D2R 0.0174532925
#define NS ${NS}
#define NR ${RINGS.length}
#define RING_IN 1.29
#define RING_OUT 4.05
uniform vec3 uSunDir;            // world, toward the Sun
uniform mat3 uW2B;               // world → body frame (y = IAU north pole, not spinning)
uniform float uFlat, uSpin, uSunI;
uniform vec3 uPh;                // two-phase advection clock: t1, t2 (s), weight of phase 1
uniform vec2 uSeed;
uniform float uPolRot, uPopT, uWaveRot;
uniform vec2 uCap, uPop;         // polar cap / polar convection amplitude, (north, south)
uniform float uView, uContrast, uIRGain;
uniform float uRingOn, uRingGain, uDustGain, uAxis, uMoonOn;
uniform vec4 uStA[NS];           // storm: lat, lon (rad), half-size E–W, N–S (rad)
uniform vec4 uStB[NS];           // storm: visible amp, IR amp, centre drift (rad/s), plume age (s)
uniform vec4 uRing[NR];          // a (R_U), e, apse longitude (rad), equivalent width (R_U)
uniform vec2 uRingW[NR];         // width at periapse / apoapse (R_U)
uniform vec4 uMoon[5];           // body-frame position (R_U), radius (R_U)
uniform vec4 uMoonA;             // orbital angles (tidal locking) of moons 0–3
uniform float uMoonA4;
float gMu, gMu0;

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
float bump(float x, float c, float w){ float t = (x - c) / w; return exp(-t * t); }
vec3 seedOff(float s){ return vec3(s * 13.7, s * 5.3, s * 9.1); }
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * c, 1.5); }

${tab3('baseVis', BASE_VIS)}
${tab3('capVis', CAP_VIS)}

// Sromovsky et al. 2015 symmetric Legendre fit: drift rate (°/h) relative to the interior; + = retrograde.
const float LC[10] = float[10](${LEG.map(ff).join(',')});
float driftDegH(float x){
  float p0 = 1.0, p1 = x, s = LC[0];
  for (int n = 1; n < 18; n++){
    float fn = float(n);
    float p2 = ((2.0 * fn + 1.0) * x * p1 - fn * p0) / (fn + 1.0); p0 = p1; p1 = p2;
    if (n % 2 == 1) s += LC[(n + 1) / 2] * p2;
  }
  return s;
}

// advected haze/cloud texture: zonally stretched noise (features ~4× longer than wide), two scales
vec2 flowField(vec3 p, float seed){
  vec3 Q = vec3(p.x, p.y * 3.0, p.z) * 2.4 + seedOff(seed);
  float a = fbm(Q, 3);
  float b = fbm(Q * 2.6 + vec3(a * 1.6, 0.0, a * 1.2), 3);
  return vec2(a, b);
}

// Ring structure: nine narrow rings (eccentric ones follow precessing ellipses, ε 20–96 km wide), the dusty ζ ring
// inside them and the faint outer ν and μ rings (dust optical depths ×uDustGain).
float rbox(float r, float r0, float w, float fw){ float we = max(w, fw); return (w / we) * (1.0 - smoothstep(0.35 * we, 0.65 * we, abs(r - r0))); }
void ringProfile(float r, float az, float fw, out float tau, out vec3 alb, out float dust){
  float t = 0.0;
  if (r > 1.6 && r < 2.06){
    for (int i = 0; i < NR; i++){
      vec4 R = uRing[i];
      float c = cos(az - R.z);
      float w = mix(uRingW[i].x, uRingW[i].y, 0.5 - 0.5 * c);
      t += (R.w / w) * rbox(r, R.x * (1.0 - R.y * c), w, fw);   // τ ∝ 1/width: the same particles, squeezed at periapse
    }
    t += 0.002 * rbox(r, 1.8458, 55.0 / 25559.0, fw);                // η's broad low-τ shoulder
  }
  float dz = 2e-4 * smoothstep(1.30, 1.44, r) * (1.0 - smoothstep(1.56, 1.62, r))               // ζ (1986U2R)
           + 5.4e-6 * bump(r, 2.633, 0.05) + 8.5e-6 * bump(r, 3.822, 0.23);                       // ν, μ
  dz *= uDustGain;
  tau = t + dz;
  dust = dz / max(tau, 1e-9);
  // ring particles are as dark as charcoal (Bond albedo ~0.02); μ ring dust is blue (small ice grains), ν reddish
  alb = mix(vec3(0.045), r > 3.2 ? vec3(0.25, 0.45, 0.9) : vec3(0.42, 0.37, 0.33), dust);
}
float ringTau(float r, float az, float fw){ float t, d; vec3 a; ringProfile(r, az, fw, t, a, d); return t; }

// shadow of the oblate planet on point p (body frame) toward the Sun s
float bodyShadow(vec3 p, vec3 s, vec3 E){
  vec3 o = p * E, d = s * E;
  float tc = -dot(o, d) / dot(d, d);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.996, 1.004, length(o + d * tc));
}

/* ---- moons ---- */
float moonsHit(vec3 ro, vec3 rd, out int id){
  float tb = 1e9; id = -1;
  if (uMoonOn < 0.5) return tb;
  for (int i = 0; i < 5; i++){
    // (perpendicular-distance form: a 300 km moon seen from 10⁶ km is below float precision in b² − c)
    vec3 oc = ro - uMoon[i].xyz; float b = dot(oc, rd);
    vec3 pp = oc - b * rd; float h = uMoon[i].w * uMoon[i].w - dot(pp, pp);
    if (h < 0.0) continue;
    float t = -b - sqrt(h);
    if (t > 0.0 && t < tb){ tb = t; id = i; }
  }
  return tb;
}
const float MOON_ALB[5] = float[5](0.32, 0.53, 0.26, 0.35, 0.31);
vec3 moonGlow(vec3 ro, vec3 rd, float pixAng, vec3 s){
  vec3 g = vec3(0.0);
  if (uMoonOn < 0.5) return g;
  for (int i = 0; i < 5; i++){
    vec3 v = uMoon[i].xyz - ro; float t = dot(v, rd);
    if (t <= 0.0) continue;
    float d = length(v - rd * t), px = t * pixAng;
    float ph = 0.5 + 0.5 * dot(normalize(ro - uMoon[i].xyz), s);
    float area = (uMoon[i].w / px) * (uMoon[i].w / px);               // flux ∝ disc area in pixels, spread over the PSF
    // (point sources get a display floor so the moons stay findable when zoomed out)
    g += vec3(0.85, 0.87, 0.9) * MOON_ALB[i] * uSunI * ph * max(min(area, 1.0), 0.05) * 6.0 * (1.0 - smoothstep(0.4, 1.2, uMoon[i].w / px)) * exp(-d * d / (1.2 * px * px));
  }
  return g;
}
float moonShadows(vec3 p, vec3 s){
  float sh = 1.0;
  if (uMoonOn < 0.5) return sh;
  for (int i = 0; i < 5; i++){
    vec3 v = uMoon[i].xyz - p; float tca = dot(v, s);
    if (tca > 0.0){
      float d = length(v - s * tca), pen = tca * 2.4e-4 + 0.001;      // the Sun is 0.014° in radius at 19 AU
      sh *= mix(0.0, 1.0, smoothstep(uMoon[i].w - pen, uMoon[i].w + pen, d));
    }
  }
  return sh;
}
vec3 moonShade(int i, vec3 p, vec3 rd, vec3 s, vec3 E){
  vec3 n = normalize(p - uMoon[i].xyz);
  float ang = i == 0 ? uMoonA.x : i == 1 ? uMoonA.y : i == 2 ? uMoonA.z : i == 3 ? uMoonA.w : uMoonA4;
  vec3 m = rotY(-ang) * n;                                              // tidally locked
  float n1 = fbm(m * 2.5 + float(i) * 7.0, 4), n2 = fbm(m * 9.0 + float(i) * 3.0, 3);
  float a = MOON_ALB[i];
  float v;
  if (i == 0) v = 1.0 + 0.35 * sign(n1) * smoothstep(0.0, 0.25, abs(n1)) * (0.5 + 0.5 * sin(m.x * 18.0)); // Miranda: patchwork coronae
  else if (i == 1) v = 1.0 + 0.25 * n1 - 0.3 * (1.0 - smoothstep(0.0, 0.05, abs(n2)));                 // Ariel: bright, graben
  else if (i == 2) v = 1.0 + 0.15 * n1 + 1.6 * smoothstep(0.92, 0.97, dot(m, normalize(vec3(0.3, -0.1, 0.95)))); // Umbriel: dark, bright Wunda ring
  else v = 1.0 + 0.3 * n1 + 0.5 * smoothstep(0.45, 0.7, n2);                                           // Titania/Oberon: craters, rays
  float mu0 = max(dot(n, s), 0.0), mu = max(dot(n, -rd), 1e-3);
  vec3 tint = i >= 3 ? vec3(1.0, 0.95, 0.89) : vec3(0.96, 0.96, 0.97);
  return tint * a * v * 2.0 * mu0 / (mu0 + mu) * uSunI * bodyShadow(p, s, E);     // Lommel–Seeliger
}

/* ---- the atmosphere ---- */
// Returns the visible normal albedo (linear RGB), its zonally smooth part, and the near-IR (methane-band)
// reflectivity + high-cloud fraction. q = body-fixed unit vector, lat = planetographic latitude (rad).
void surface(vec3 q, float lat, out vec3 vis, out vec3 visSm, out float ir, out float hc){
  float latD = lat * R2D, a = abs(latD);
  bool north = latD > 0.0;
  float capA = north ? uCap.x : uCap.y, popA = north ? uPop.x : uPop.y;
  float lon = atan(-q.z, q.x);
  float drP = driftDegH(sin(lat)) * (D2R / 3600.0);           // rad/s of east longitude at this latitude
  // zonal winds shear the haze: two copies advected over [0, P) and cross-faded (each reset while invisible)
  vec3 qa = rotY(-drP * uPh.x) * q, qb = rotY(-drP * uPh.y) * q;
  float w1 = uPh.z, w2 = 1.0 - w1;
  vec2 f = (flowField(qa, uSeed.x) * w1 + flowField(qb, uSeed.y) * w2) * inversesqrt(w1 * w1 + w2 * w2);
  float lw = latD + 1.3 * f.x;                                 // band edges ripple with the flow
  float aw = abs(lw);

  // ---- zonal structure (OPAL 2025 profile, re-scaled cap) -----------------------------------
  vec3 filt = baseVis(a) + capVis(a) * capA;                   // (F657N, F547M, F467M) normal albedo
  visSm = vec3(0.92 * filt.x, filt.y, 0.93 * filt.z);          // → linear sRGB (657 nm sits in a methane window: ×0.92)

  // ---- fine banding seen in Keck H-band images 2012–2014 (Sromovsky et al. 2015), approx. amplitudes ----
  float bands = -0.35 * bump(lw, 8.5, 1.6)                                                   // dark band at 8.5°N
              + 0.25 * bump(aw, 24.0, 6.0)                                                   // brighter storm latitudes
              - 0.25 * bump(aw, 43.0, 3.5) * capA                                            // dark ring outside the cap (JWST 2023)
              + 0.3 * (0.3 + 0.7 * capA) * (bump(aw, 48.0, 0.7) + bump(aw, 50.5, 0.7) + bump(aw, 53.0, 0.7))  // narrow bands 2.5° apart
              + 0.35 * bump(lw, -45.0, 2.5);                                                 // the southern collar
  // equatorial scalloped wave near 4.5°S (~36° wavelength) and the 'ribbon' at 6°S: a pattern that propagates
  // relative to the flow (uWaveRot), unlike passive clouds
  float wph = lon * 10.0 - uWaveRot;
  bands += 0.3 * bump(latD, -4.5 + 0.9 * sin(wph), 0.8) + 0.18 * bump(latD, -6.2 + 0.5 * sin(wph * 2.0 + 1.0), 0.45);

  // ---- polar 'popcorn': ~650 km convective cells of the solid-body polar region (62°–pole) -------------
  float pop = 0.0;
  if (a > 52.0 && popA > 0.0){
    vec3 qp = rotY(-uPolRot) * q;                              // rigid rotation 4.1°/h faster than the interior
    vec2 w = worley(qp * 38.0 + vec3(0.0, uPopT, 0.0));
    pop = smoothstep(0.5, 0.1, w.x) * smoothstep(0.0, 0.45, fbm(qp * 7.0 + uPopT * 0.05, 2) + 0.1);   // clustered
    pop *= smoothstep(52.0, 60.0, a) * popA;
  }

  // ---- discrete storms: coherent cores drifting with the wind at their latitude, plus a plume of
  //      condensate released at the core and sheared by the differential rotation since it formed ----
  float sv = 0.0, si = 0.0;
  float cl = cos(lat);
  for (int i = 0; i < NS; i++){
    vec4 A = uStA[i], B = uStB[i];
    float dy = (lat - A.x) / A.w;
    if (abs(dy) > 3.5) continue;
    float x = (mod(lon - A.y + PI, TAU) - PI) * cl / A.z;
    // plume: condensate released at the vortex drifts away at the local wind minus the vortex's own drift
    // (B.z = core drift − its offset from the zonal wind), so it streaks downstream and tilts with the shear
    float L = (drP - B.z) * B.w * cl / A.z;
    float xs = x - clamp(x, min(0.0, L), max(0.0, L));
    float along = abs(L) > 1e-3 ? clamp(x / L, 0.0, 1.0) : 0.0;
    float g = max(exp(-(x * x + dy * dy)), exp(-(4.0 * xs * xs + 2.2 * dy * dy)) * 0.6 * (1.0 - along));
    sv += B.x * g; si += B.y * g;
  }

  // ---- compose -------------------------------------------------------------------------------
  vis = visSm * (1.0 + 0.015 * bands + 0.008 * f.y);
  vis = mix(vis, vec3(0.86, 0.91, 0.95), clamp(sv + 0.03 * pop, 0.0, 0.7));   // white methane-ice cloud tops
  vis *= 1.0 + min(sv, 0.0) * 1.5;                                               // dark spot
  // near-IR reflectivity (I/F-like, H band / F140M): the deep atmosphere is dark in methane bands, so only
  // aerosols and high clouds show; the polar cap is the brightest region (JWST 2023), approx. profile
  float ir0 = 0.085 + 0.02 * smoothstep(12.0, 25.0, a) - 0.012 * smoothstep(33.0, 45.0, a)
            + 0.02 * smoothstep(55.0, 65.0, a) + 0.2 * capA * smoothstep(50.0, 62.0, a);
  ir = ir0 * (1.0 + 0.55 * bands + 0.07 * f.y + 0.05 * f.x) + 0.07 * pop * (0.6 + 0.4 * capA);
  ir = max(ir + 0.1 * si, 0.012);
  // high, bright haze/cloud: storm tops and the polar cap also show at 3 µm (whiter in the JWST colours)
  hc = clamp(si * 0.6 + 0.6 * capA * smoothstep(52.0, 66.0, a), 0.0, 1.0);
}

vec3 sunDisc(vec3 rdW){
  float c = dot(rdW, uSunDir);
  return bb(5772.0) * (pow(max(c, 0.0), 40000.0) * 6.0 + pow(max(c, 0.0), 2000.0) * 0.3);
}

void main(){
  vec3 rdW = camRay(), roW = uCamPos;
  vec3 ro = uW2B * roW, rd = uW2B * rdW, s = uW2B * uSunDir;
  vec3 E = vec3(1.0, 1.0 / (1.0 - uFlat), 1.0);
  vec3 roE = ro * E, rdE = rd * E;
  float A = dot(rdE, rdE), B = dot(roE, rdE), C = dot(roE, roE) - 1.0, H = B * B - A * C;
  float tP = 1e9;
  if (H > 0.0){ float t0 = (-B - sqrt(H)) / A; if (t0 > 0.0) tP = t0; }
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  mat3 Rs = rotY(-uSpin);
  int mid = -1; float tM = moonsHit(ro, rd, mid);
  float tN = min(tP, tM);
  float irMode = step(1.5, uView);
  float gain = mix(1.0, uIRGain, irMode);
  vec3 col;
  if (tP < 1e8 && tP <= tN){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = Rs * normalize(p * E);
    float lat = asin(clamp(n.y, -1.0, 1.0));
    gMu0 = dot(n, s); gMu = max(dot(n, -rd), 0.0);
    vec3 vis, visSm; float ir, hc;
    surface(q, lat, vis, visSm, ir, hc);
    float sh = 1.0;
    if (uRingOn > 0.5 && p.y * s.y < 0.0){
      vec3 pr = p - s * (p.y / s.y);
      sh = exp(-ringTau(length(pr.xz), atan(-pr.z, pr.x), tP * pixAng / max(abs(s.y), 0.03)) / max(abs(s.y), 0.004));
    }
    sh *= moonShadows(p, s);
    float mu0 = max(gMu0, 0.0), mu = max(gMu, 0.1);
    float limb = mix(0.55, 1.0, smoothstep(0.0, 0.35, gMu));        // Minnaert fits break down at grazing angles (approx.)
    float twi = 0.004 * smoothstep(-0.1, 0.03, gMu0) * (1.0 - smoothstep(0.0, 0.15, gMu0));   // twilight through the haze
    if (irMode > 0.5){
      // methane-band photometry: limb brightening from the stratospheric haze (Minnaert k ≈ 0.42), approx.
      float lit = pow(mu0, 0.42) * pow(mu, -0.58) + twi;
      vec3 c140 = vec3(0.3, 0.72, 1.0) * ir;                         // F140M shown in blue (as in the JWST release)
      vec3 cHigh = vec3(1.0, 0.93, 0.85) * ir * 1.15;                // + cloud tops reaching the 3 µm image (orange)
      col = mix(c140, cHigh, hc * 0.8) * lit * uSunI * sh * gain;
      col += vec3(0.02, 0.05, 0.1) * gain * uSunI * pow(1.0 - gMu, 3.0) * smoothstep(-0.2, 0.2, gMu0);   // limb glow
    } else {
      vec3 alb = vis;
      if (uView > 0.5){
        // enhanced, like the processed Voyager 2 colour images: stretch the departure from the equatorial colour
        // (the smooth latitude profile less than the small-scale detail, so neither saturates), then partly
        // colour-balance on that reference so the methane-depleted, hazy polar region shows up warm
        vec3 ref = vec3(0.92 * 0.267, 0.604, 0.93 * 0.749);
        vec3 stv = max(ref + (visSm - ref) * (1.0 + 0.1 * (uContrast - 1.0)) + (vis - visSm) * uContrast, 0.0);
        alb = mix(stv * 0.62, stv / ref * 0.38, 0.35);
      }
      // per-filter Minnaert law measured by OPAL: k = 0.57 (657 nm), 0.80 (547 nm), 0.85 (467 nm)
      // (blended toward a common k at grazing angles, where the single-k fits no longer hold, approx.)
      vec3 k = mix(vec3(0.78), vec3(0.57, 0.80, 0.85), smoothstep(0.1, 0.45, gMu));
      vec3 lit = pow(vec3(mu0), k) * pow(vec3(mu), k - 1.0) + twi;
      col = alb * lit * limb * uSunI * sh;
      // forward-scattering haze and Rayleigh limb (blue-green), very faint
      col += vec3(0.03, 0.07, 0.1) * uSunI * pow(1.0 - gMu, 5.0) * smoothstep(-0.25, 0.2, gMu0) * (0.4 + 0.6 * hg(dot(rd, s), 0.6)) * sh;
      // pre-compensate the host's ACES shoulder, which greys out bright pastel colours (display only)
      if (uView < 0.5) col = max(mix(vec3(dot(col, vec3(0.2126, 0.7152, 0.0722))), col, 1.15), 0.0);
    }
  } else if (tM < 1e8){
    col = moonShade(mid, ro + rd * tM, rd, s, E) * gain;
  } else {
    col = sky(rdW, 1.0) * 0.6 + sunDisc(rdW);
    col += moonGlow(ro, rd, pixAng, s) * gain;
    // the upper atmosphere's haze above the limb (scale height ~ 50 km), backlit
    float tc = -B / A;
    if (tc > 0.0){
      vec3 pc = roE + rdE * tc;
      float hgt = max(length(pc) - 1.0, 0.0);
      float il = smoothstep(-0.3, 0.2, dot(normalize(pc), s));
      col += vec3(0.04, 0.09, 0.12) * il * (0.25 + 0.75 * hg(dot(rd, s), 0.7)) * uSunI * exp(-hgt / 0.004) * gain;
    }
  }
  // ---- rings ----
  if (uRingOn > 0.5 && abs(rd.y) > 1e-6){
    float tR = -ro.y / rd.y;
    if (tR > 0.0 && tR < tN){
      vec3 pr = ro + rd * tR;
      float r = length(pr.xz);
      if (r > RING_IN && r < RING_OUT){
        float mu = abs(rd.y), mu0 = max(abs(s.y), 1e-4);
        vec2 vh = normalize(rd.xz + 1e-7);
        float cr = dot(pr.xz / r, vh);
        float fw = min(tR * pixAng * sqrt(cr * cr / (mu * mu) + 1.0 - cr * cr), 0.3);
        float tau, dust; vec3 alb;
        ringProfile(r, atan(-pr.z, pr.x), fw, tau, alb, dust);
        if (tau > 0.0){
          float shd = bodyShadow(pr, s, E);
          float c = dot(s, rd);
          float P = mix(hg(c, -0.3), hg(c, 0.7), dust);
          float IF;
          if (s.y * rd.y < 0.0) IF = 0.25 * P * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu)));
          else {
            float dm = mu - mu0;
            IF = abs(dm) > 1e-3 ? 0.25 * P * mu0 / dm * (exp(-tau / mu) - exp(-tau / mu0)) : 0.25 * P * mu0 * tau / (mu * mu) * exp(-tau / mu);
          }
          col = col * exp(-tau / mu) + alb * max(IF, 0.0) * uSunI * uRingGain * shd * gain;
        }
      }
    }
  }
  // ---- spin axis (IAU north = orange) ----
  if (uAxis > 0.5){
    float b = rd.y, dd = dot(rd, ro), den = 1.0 - b * b;
    if (den > 1e-6){
      float sc = (b * ro.y - dd) / den, ta = ro.y + b * sc;
      if (sc > 0.0 && sc < tN){
        float dist = length(ro + rd * sc - vec3(0.0, ta, 0.0));
        float line = exp(-pow(dist / (1.4 * sc * pixAng), 2.0)) * step(0.985, abs(ta)) * (1.0 - smoothstep(1.75, 2.0, abs(ta)));
        col += (ta > 0.0 ? vec3(1.0, 0.42, 0.22) : vec3(0.3, 0.6, 1.0)) * line * 0.7;
      }
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ------------------------------------------------------------------------------------------------ */
/* JS: orientation, clocks, picking                                                                   */
/* ------------------------------------------------------------------------------------------------ */
// World: the Sun is fixed at azimuth sunAz in the xz-plane (= Uranus' orbital plane, world y = ecliptic north);
// the ecliptic frame is rotated about its pole so that the real Sun direction for 'year' lands there. The pole
// therefore swings around as the year changes — Uranus 'rolling' along its orbit, seen from the Sun's side.
function orient(st, year, sunAzDeg) {
  const g = geometry(year);
  const az = sunAzDeg * D2R;
  const e1 = new V3(g.sun.x, g.sun.y, 0).normalize(), e3 = new V3(0, 0, 1), e2 = new V3().crossVectors(e3, e1);
  const w1 = new V3(Math.sin(az), 0, Math.cos(az)), w2 = new V3(Math.cos(az), 0, -Math.sin(az)), w3 = new V3(0, 1, 0);
  const toW = v => w1.clone().multiplyScalar(v.dot(e1)).addScaledVector(w2, v.dot(e2)).addScaledVector(w3, v.dot(e3));
  st.sun.copy(toW(g.sun)).normalize();
  st.earth = toW(g.earth).normalize();
  const P = st.pole.copy(toW(POLE_E)).normalize();
  const xb = new V3().crossVectors(new V3(0, 1, 0), P).normalize(), zb = new V3().crossVectors(xb, P);
  st.W2B.set(xb.x, xb.y, xb.z, P.x, P.y, P.z, zb.x, zb.y, zb.z);
  st.geo = g;
  st.B = Math.asin(THREE.MathUtils.clamp(P.dot(st.sun), -1, 1)) * R2D;
}
function hitBody(o, d, W2B) {
  const ro = o.clone().applyMatrix3(W2B), rd = d.clone().applyMatrix3(W2B);
  const k = 1 / (1 - FLAT);
  const A = rd.x * rd.x + rd.y * rd.y * k * k + rd.z * rd.z, B = ro.x * rd.x + ro.y * rd.y * k * k + ro.z * rd.z;
  const C = ro.x * ro.x + ro.y * ro.y * k * k + ro.z * ro.z - 1, H = B * B - A * C;
  let t = null;
  if (H >= 0) { const t0 = (-B - Math.sqrt(H)) / A; if (t0 > 0) t = t0; }
  return { t, ro, rd, k };
}
// body point → planetographic latitude / east longitude (body-fixed)
function bodyLatLon(p, k, spin) {
  const n = new V3(p.x, p.y * k * k, p.z).normalize();
  const c = Math.cos(spin), s = Math.sin(spin);
  const qx = c * p.x - s * p.z, qz = s * p.x + c * p.z;     // Ry(−spin)
  return { lat: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) * R2D, lonE: Math.atan2(-qz, qx), n };
}
function seasonText(B, year) {
  const g1 = geometry(year + 0.5).B;
  const rising = g1 > B;
  if (Math.abs(B) < 3) return rising ? 'điểm xuân phân của bán cầu Bắc' : 'điểm thu phân của bán cầu Bắc';
  if (Math.abs(B) > 80) return B > 0 ? 'hạ chí Bắc — cực Bắc hướng thẳng vào Mặt Trời' : 'đông chí Bắc — cực Nam hướng vào Mặt Trời';
  if (B > 0) return rising ? 'xuân ở bán cầu Bắc' : 'hạ ở bán cầu Bắc';
  return rising ? 'đông ở bán cầu Bắc' : 'thu ở bán cầu Bắc';
}
const windStr = u => { const r = Math.round(u); return r === 0 ? '≈ 0 m/s (quay cùng lõi)' : `${r > 0 ? '+' : '−'}${Math.abs(r)} m/s ${r > 0 ? '(thuận chiều quay)' : '(ngược chiều quay)'}`; };
// inertial rotation period of the clouds at a latitude: interior rate + prograde drift (drift > 0 = retrograde)
const cloudPeriod = latD => TAU / (OMEGA - driftRad(latD)) / 3600;

function instOf(ctx) { return ctx.scene && ctx.scene.__inst; }
// put the camera on the Earth's side (Uranus is always seen at < 3° phase from Earth)
function earthView(ctx, dist) {
  const inst = instOf(ctx); if (!inst) return;
  const st = inst.st; st.clock(ctx);
  const d = dist || Math.max(3.2, Math.min(6, ctx.camera.position.length()));
  ctx.camera.position.copy(st.earth.clone().multiplyScalar(d)); ctx.camera.up.set(0, 1, 0);
  ctx.camera.lookAt(0, 0, 0); ctx.controls.target.set(0, 0, 0); ctx.controls.update();
}
function epoch(ctx, year, view, msg) {
  ctx.params.year = year;
  if (view) ctx.params.view = view;
  const inst = instOf(ctx);
  if (inst) { orient(inst.st, year, ctx.params.sunAz); earthView(ctx); }
  ctx.toast(msg, 9000);
}

const DEF = {
  id: 'uranus', name: 'Sao Thiên Vương', category: 'planets', order: 42,
  subtitle: 'Hành tinh băng khổng lồ · lăn nghiêng trên quỹ đạo',
  swatch: ['#d4f3f4', '#3f8a96'],
  blurb: 'Trục quay nghiêng <b>97,8°</b>: Sao Thiên Vương lăn trên quỹ đạo 84 năm như một quả bóng, mỗi cực có 42 năm ngày rồi 42 năm đêm. Dưới ánh sáng nhìn thấy, mê-tan hấp thụ ánh đỏ nên hành tinh gần như một quả cầu lục lam trơn; ở cận hồng ngoại các cơn bão, các dải mây và <b>chỏm cực theo mùa</b> hiện rõ, trôi theo những luồng gió từ −40 m/s ở xích đạo tới +258 m/s ở vĩ độ 55°.',
  intro: 'Kéo <b>Năm</b> để Sao Thiên Vương "lăn" qua các mùa (1986 Voyager 2 → 2007 điểm phân → 2030 hạ chí). Chuyển <b>Ảnh</b> sang <b>Hồng ngoại</b> và đặt <b>Khung</b> = "Quay cùng lõi" để thấy gió thổi các cơn bão trôi với tốc độ khác nhau theo vĩ độ.',
  facts: [
    ['Bán kính xích đạo / cực (1 bar)', '25 559 / 24 973 km (4,0 Trái Đất)'], ['Khối lượng', '8,681×10²⁵ kg (14,5 Trái Đất)'],
    ['Chu kỳ tự quay', '17 giờ 14 phút 52 giây, nghịch hành (Lamy và cs. 2025)'], ['Độ nghiêng trục', '97,77°'],
    ['Quỹ đạo', '19,19 AU · 84,0 năm · e = 0,047'], ['Nhiệt độ hiệu dụng', '59 K; đỉnh tầng đối lưu 53 K'],
    ['Gió', '−40 m/s ở xích đạo; +258 m/s ở 55° (Sromovsky và cs. 2015)'],
    ['Khí quyển', 'H₂ 83 %, He 15 %, CH₄ ~2,3 % (tới ~4 % dưới mây)'],
    ['Vành', '13 vành; vành ε cách tâm 51 149 km, rộng 20 – 96 km'],
    ['Vệ tinh lớn', 'Miranda, Ariel, Umbriel, Titania, Oberon'],
    ['Dữ liệu ảnh', 'HST OPAL 2014/2025 (NASA/ESA, A. Simon, M. Wong; CC BY 4.0), Keck 2012–2014, JWST 2023, Voyager 2'],
  ],
  unitKm: RE,
  camera: { distance: 4.4, polar: 80, azimuth: 32, min: 1.04, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.3,
  params: [
    { key: 'view', label: 'Ảnh', type: 'seg', options: [['true', 'Màu thật'], ['enh', 'Tăng tương phản'], ['ir', 'Hồng ngoại']], default: 'true',
      note: 'Màu thật: hệ số phản xạ đo từ ảnh Hubble OPAL 2025. Tăng tương phản: kéo giãn độ lệch màu như các ảnh Voyager/Keck đã xử lý. Hồng ngoại: dải hấp thụ mê-tan (như Keck H và JWST F140M) — chỉ mây cao và sương mù sáng lên.' },
    { key: 'year', label: 'Năm', type: 'range', min: 1980, max: 2070, step: 0.1, default: Math.round(NOW_YEAR * 10) / 10, fmt: v => vn(v, 1),
      note: 'Vị trí thật trên quỹ đạo: 1986 Voyager 2 (cực Nam hướng Mặt Trời), 12/2007 điểm phân, 2030 hạ chí Bắc, 2050 điểm phân.' },
    { key: 'tlog', label: 'Tua nhanh', type: 'range', min: 1, max: 4.5, step: 0.05, default: 2.8, fmt: v => '×' + fmtInt(10 ** v),
      note: 'Số giây trên Sao Thiên Vương cho mỗi giây mô phỏng (nhân thêm với thanh tốc độ thời gian chung). Gió và chuyển động là thật theo cùng tỉ lệ.' },
    { key: 'frame', label: 'Khung', type: 'seg', options: [['inertial', 'Quán tính'], ['sys3', 'Quay cùng lõi']], default: 'inertial',
      note: 'Quay cùng lõi: bỏ chuyển động tự quay 17 h 15 m của phần lõi (System III) — chỉ còn gió: vùng cực quay nhanh hơn 4°/giờ, xích đạo chậm hơn.' },
    { key: 'contrast', label: 'Độ tăng tương phản', type: 'range', min: 1, max: 20, step: 0.5, default: 7, fmt: v => '×' + vn(v, 1) },
    { key: 'sunAz', label: 'Hướng Mặt Trời', type: 'range', min: 0, max: 360, step: 1, default: 40, fmt: v => v.toFixed(0) + '°', note: 'Xoay ra sau hành tinh để thấy vành bụi sáng lên khi nhìn ngược sáng.' },
    { key: 'ringGain', label: 'Độ sáng vành (phóng đại)', type: 'range', min: 1, max: 60, step: 1, default: 12, fmt: v => '×' + v.toFixed(0), note: 'Hạt vành có hệ số phản xạ ~0,02 — tối như than; vành ε rộng ≤ 96 km nên chỉ là một sợi chỉ.' },
    { key: 'rings', label: 'Vành', type: 'toggle', default: true },
    { key: 'moons', label: 'Năm vệ tinh lớn', type: 'toggle', default: true },
    { key: 'axis', label: 'Trục quay', type: 'toggle', default: false },
    { key: 'spin', label: 'Chạy thời gian', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Voyager 2 · 1986', key: 'v', title: 'Tháng 1/1986: cực Nam hướng Mặt Trời', run: ctx => epoch(ctx, 1986.07, 'true', 'Voyager 2 bay qua (24/1/1986): Mặt Trời gần như ở thiên đỉnh cực Nam. Ở màu thật hành tinh gần như không có đường nét — chỉ một vùng cực sáng hơn chút. Chuyển sang <b>Tăng tương phản</b> để thấy chỏm sương mù cực Nam.') },
    { label: 'Điểm phân 2007', key: 'e', title: 'Mặt Trời trên xích đạo', run: ctx => epoch(ctx, 2007.95, null, 'Điểm phân 12/2007: Mặt Trời chiếu ngang xích đạo, vành nhìn từ Mặt Trời gần như một vạch. Chỏm cực Nam đã mờ, cực Bắc vừa ló ra sau 42 năm đêm.') },
    { label: 'Keck 2014', key: 'i', title: 'Mùa bão kỷ lục, cận hồng ngoại', run: ctx => epoch(ctx, 2014.6, 'ir', 'Tháng 8/2014, kính Keck (quang học thích nghi) ghi nhận mùa bão kỷ lục ở bán cầu Bắc; cơn bão sáng nhất từng thấy ở 2,2 µm nằm gần 15°B. Đặt <b>Khung</b> = Quay cùng lõi để thấy chúng trôi.') },
    { label: 'JWST 2023', key: 'j', title: 'Chỏm cực Bắc sáng, vành', run: ctx => epoch(ctx, 2023.1, 'ir', 'JWST NIRCam (6/2/2023): chỏm cực Bắc sáng theo mùa, một đám mây sáng ở rìa chỏm, vành ε và vành bụi ζ mờ phía trong.') },
    { label: 'Hạ chí 2030', key: 'o', title: 'Cực Bắc hướng Mặt Trời', run: ctx => epoch(ctx, 2030.0, null, 'Hạ chí Bắc (2030): Mặt Trời ở 82° B, gần thiên đỉnh cực Bắc; cả bán cầu Nam chìm trong đêm 42 năm. Vành nhìn từ Mặt Trời như một tấm bia.') },
    { label: 'Nhìn từ Trái Đất', key: 'g', title: 'Đặt camera theo hướng Trái Đất', run: ctx => { earthView(ctx); ctx.toast('Hướng nhìn từ Trái Đất vào năm đang chọn (góc pha luôn < 3°).', 5000); } },
  ],
  create(ctx) {
    const R = ctx.renderer;
    const U = {
      ...ctx.shared,
      uSunDir: { value: new V3(0, 0, 1) }, uW2B: { value: new THREE.Matrix3() },
      uFlat: { value: FLAT }, uSpin: { value: 0 }, uSunI: { value: 0.75 },
      uPh: { value: new V3(0, 0, 1) }, uSeed: { value: new THREE.Vector2() },
      uPolRot: { value: 0 }, uPopT: { value: 0 }, uWaveRot: { value: 0 },
      uCap: { value: new THREE.Vector2() }, uPop: { value: new THREE.Vector2() },
      uView: { value: 0 }, uContrast: { value: 7 }, uIRGain: { value: 3.4 },
      uRingOn: { value: 1 }, uRingGain: { value: 12 }, uDustGain: { value: 25 }, uAxis: { value: 0 }, uMoonOn: { value: 1 },
      uStA: { value: STORMS.map(() => new THREE.Vector4()) }, uStB: { value: STORMS.map(() => new THREE.Vector4()) },
      uRing: { value: RINGS.map(() => new THREE.Vector4()) }, uRingW: { value: RINGS.map(r => new THREE.Vector2(r[3] / RE, r[4] / RE)) },
      uMoon: { value: MOONS.map(() => new THREE.Vector4()) }, uMoonA: { value: new THREE.Vector4() }, uMoonA4: { value: 0 },
    };
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    const st = {
      U, W2B: U.uW2B.value, sun: U.uSunDir.value, pole: new V3(), earth: new V3(1, 0, 0), geo: null, B: 0,
      lastSim: null, T: 0, spin: 0, lastKey: '',
    };
    // planet clock: T (Uranus seconds) integrates the time-lapse factor; idempotent within a frame
    const clock = (c) => {
      const p = c.params, sim = c.time.sim;
      if (st.lastSim === null) { st.lastSim = sim; st.T = sim * 10 ** p.tlog; }
      const dsim = Math.max(0, sim - st.lastSim); st.lastSim = sim;
      const run = p.spin !== false;
      const dT = run ? dsim * 10 ** p.tlog : 0;
      st.T += dT;
      if (p.frame !== 'sys3') st.spin = wrap(st.spin - OMEGA * dT);
      const key = p.year + '|' + p.sunAz;
      if (key !== st.lastKey) { orient(st, p.year, p.sunAz); st.lastKey = key; }
    };
    st.clock = clock;
    const inst = {
      material, st,
      update(dt, c) {
        const p = c.params;
        clock(c);
        const T = st.T;
        U.uSpin.value = st.spin;
        const t1 = T % PHASE_P, t2 = (T + PHASE_P / 2) % PHASE_P;
        U.uPh.value.set(t1, t2, 1 - Math.abs(2 * t1 / PHASE_P - 1));
        U.uSeed.value.set(Math.floor(T / PHASE_P) % 29, (Math.floor((T + PHASE_P / 2) / PHASE_P) % 29) + 0.5);
        U.uPolRot.value = wrap(driftRad(75) * T);
        U.uPopT.value = (T / (10 * 3600)) % 200;                     // cells renew on ~10 h (approx.)
        // equatorial wave: pattern speed 0.4°/h prograde relative to the interior, against the −0.3°/h flow
        U.uWaveRot.value = wrap(10 * (-0.4 * D2R / 3600) * T);
        const yr = p.year;
        const ca = capAmp(yr), pa = popAmp(yr);
        U.uCap.value.set(ca[0], ca[1]); U.uPop.value.set(pa[0], pa[1]);
        U.uView.value = p.view === 'ir' ? 2 : p.view === 'enh' ? 1 : 0;
        U.uContrast.value = p.contrast;
        U.uRingOn.value = p.rings === false ? 0 : 1;
        U.uRingGain.value = p.ringGain;
        U.uAxis.value = p.axis ? 1 : 0;
        U.uMoonOn.value = p.moons === false ? 0 : 1;
        // storms
        for (let i = 0; i < NS; i++) {
          const s = STORMS[i], dr = driftRad(s.lat);
          const lon = wrap(s.lon * D2R + dr * T);
          const ageH = ((T / 3600 + s.ph * s.P) % s.P + s.P) % s.P, f = ageH / s.P;
          const env = s.long ? 0.8 + 0.2 * Math.sin(TAU * f) : smooth(0, 0.12, f) * (1 - smooth(0.55, 1, f));
          const tailAge = s.long ? s.tail * 3600 : Math.min(ageH, s.tail) * 3600;
          const lat = s.lat * D2R;
          U.uStA.value[i].set(lat, lon, s.ax / RE, s.ay / RE);
          // vortices drift ~0.2–0.3°/h off the zonal wind at their latitude (e.g. the 2006 dark spot), approx.
          const off = (i % 2 ? 1 : -1) * (0.45 + 0.15 * (i % 3)) * D2R / 3600;
          U.uStB.value[i].set(s.vis * env, s.ir * env, dr - off, tailAge);
        }
        // rings: apses precess prograde (= toward −east longitude for retrograde-spinning Uranus)
        for (let i = 0; i < RINGS.length; i++) {
          const r = RINGS[i];
          U.uRing.value[i].set(r[1] / RE, r[2], wrap(r[6] * D2R - ringPrec(r[1]) * T), r[5] / RE);
        }
        // moons: circular, equatorial, prograde orbits (approx.: Miranda's 4.3° inclination ignored)
        for (let i = 0; i < 5; i++) {
          const m = MOONS[i], a = m[1] / RE;
          const ang = wrap(i * 1.9 + 0.6 - TAU * T / (m[2] * DAY));
          U.uMoon.value[i].set(a * Math.cos(ang), 0, -a * Math.sin(ang), m[3] / RE);
          if (i < 4) U.uMoonA.value.setComponent(i, ang); else U.uMoonA4.value = ang;
        }
      },
      hud(c) {
        clock(c);
        const p = c.params;
        const cam = c.camera.position.clone().normalize();
        const d = cam.clone().applyMatrix3(st.W2B);
        const k = 1 / (1 - FLAT), sc = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
        const ll = bodyLatLon(d.multiplyScalar(sc), k, st.spin);
        const ca = capAmp(p.year);
        const tl = 10 ** p.tlog * (c.bhGlobals?.timeScale ?? 6);
        const phase = Math.acos(THREE.MathUtils.clamp(cam.dot(st.sun), -1, 1)) * R2D;
        return [
          ['Năm', `${vn(p.year, 1)} — ${seasonText(st.B, p.year)}`],
          ['Vĩ độ điểm dưới Mặt Trời', `${latStr(st.B)} (Trái Đất: ${latStr(st.geo.BE)})`],
          ['Cách Mặt Trời', `${vn(st.geo.rAU, 2)} AU · ánh sáng đi ${vn(st.geo.rAU * 499 / 3600, 2)} giờ`],
          ['Vĩ độ dưới camera', latStr(ll.lat)],
          ['Gió ở đó', windStr(windMS(ll.lat))],
          ['Mây ở đó quay một vòng', `${vn(cloudPeriod(ll.lat), 2)} giờ (lõi: 17,25 giờ)`],
          ['Chỏm cực Bắc / Nam', `${Math.round(ca[0] * 100)} % / ${Math.round(ca[1] * 100)} %`],
          ['Góc pha', vn(phase, 0) + '°'],
          ['Tua nhanh', `×${fmtInt(tl)} (1 giây = ${tl >= 3600 ? vn(tl / 3600, 1) + ' giờ' : vn(tl / 60, 0) + ' phút'})`],
        ];
      },
      onClick(info, c) {
        clock(c);
        const h = hitBody(info.origin, info.dir, st.W2B);
        if (h.t === null) {
          // ring plane?
          if (Math.abs(h.rd.y) > 1e-6) {
            const t = -h.ro.y / h.rd.y;
            if (t > 0) {
              const pr = h.ro.clone().addScaledVector(h.rd, t), r = Math.hypot(pr.x, pr.z) * RE;
              let best = null;
              for (const rg of RINGS) if (!best || Math.abs(rg[1] - r) < Math.abs(best[1] - r)) best = rg;
              if (best && Math.abs(best[1] - r) < 1500) {
                const nDay = Math.sqrt(GM / best[1] ** 3) * DAY * R2D, pr = ringPrec(best[1]) * DAY * R2D;
                c.toast(`Vành <b>${best[0]}</b>: bán kính ${fmtInt(best[1])} km, độ lệch tâm ${vn(best[2] * 1000, 3)}×10⁻³, rộng ${vn(best[3], 1)}–${vn(best[4], 1)} km. Hạt vành quay ${vn(nDay, 0)}°/ngày; trục elip tiến động ${vn(pr, 3)}°/ngày do Sao Thiên Vương dẹt (J₂).`, 8000);
              }
            }
          }
          return;
        }
        const p = h.ro.clone().addScaledVector(h.rd, h.t);
        const ll = bodyLatLon(p, h.k, st.spin);
        const B = st.B;
        const day = Math.abs(ll.lat) > 90 - Math.abs(B) ? (Math.sign(ll.lat) === Math.sign(B) ? 'Mặt Trời không lặn ở đây vào mùa này' : 'đêm kéo dài suốt mùa này') : 'ngày và đêm luân phiên theo vòng quay';
        c.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>: gió ${windStr(windMS(ll.lat))}, mây quay một vòng trong ${vn(cloudPeriod(ll.lat), 2)} giờ; ${day}.`, 7000);
      },
    };
    DEF.__inst = inst;
    return inst;
  },
};

export default [DEF];
