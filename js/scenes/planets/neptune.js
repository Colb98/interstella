// Sao Hải Vương — Neptune from real imagery, with its weather moving at the measured speeds.
//
// One scene unit = Neptune's equatorial radius (24 764 km). Fullscreen ray cast of the oblate (f = 0.0171)
// planet, tilted 28.32°, the Sun in the world xz-plane so the season is geometric (sub-solar latitude
// −23° in 1989, −25° in 2018).
//
// Two epochs:
//   * 1989 · Voyager 2 — no usable global map of the 1989 clouds exists, so the cloud deck is procedural but
//     calibrated on the Voyager 2 images (PIA01492, PIA00052, PIA01142, PIA00064, PIA00058): the Great Dark
//     Spot with its bright SW companion clouds and NE streak, the Scooter, Dark Spot 2 with its bright core
//     and concentric bands, cirrus bands at 27°N, 20–40°S and 70°S.
//   * 2018 · Hubble OPAL — the real global map (WFC3/UVIS F657N/F547M/F467M + FQ727N methane band, Simon,
//     Wong et al., CC BY 4.0) with the Northern Dark Spot NDS-2018 and its companion clouds, re-projected
//     here as a relative-reflectance texture (assets/neptune/, see CREDITS.txt) plus fine procedural detail.
//
// Motion (all in "planet time" T, s; T = 0 at Voyager 2 closest approach / OPAL rotation 2):
//   * body rotation 16 h 6.6 min (Voyager radio period);
//   * zonal winds: Sromovsky et al. (1993) polynomial, −400 m/s at the equator, +235 m/s at 70°;
//   * flow-map advection of the cloud field with two phases reset alternately (no rigid scroll, no wind-up);
//     inside the vortices the flow follows Kida-vortex elliptical streamlines (uniform rotation, rate from
//     the local shear: Ω = s·λ²/(λ²−1)), blended into the zonal flow outside;
//   * features drift at their measured periods: GDS 18.30 h, Scooter 16.8 h, DS2 16.0 h; the GDS oscillates in
//     shape every 8 days (aspect ratio 2.0 ± 0.35, axis wobble ±8°, area conserved) and drifts equatorward
//     at 1.3°/month (Sromovsky et al. 1993);
//   * Adams-ring arcs co-orbiting at 820.1185°/day; Triton on its retrograde, 156.9°-inclined 5.877-day orbit.
// Photometry: true colour from Irwin et al. (2024) (Voyager re-calibrated with HST/STIS + VLT/MUSE), absolute
// normal reflectance from OPAL (I/F 0.54 at 547 nm), per-channel Minnaert exponents from the OPAL maps
// (k = 0.50 / 0.80 / 0.88 for red / green / blue: red limb-brightens because of methane + haze).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';
import { t, dec, int } from '../../i18n.js';

const V3 = THREE.Vector3;
const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAU = Math.PI * 2;
const RN = 24764e3;                 // m
const FLAT = 0.0171, OBLIQ = 28.32;
const PROT = 57996;                 // s (16 h 6.6 min)
const DAY = 86400, HOUR = 3600;
const CA_MS = Date.UTC(1989, 7, 25, 3, 56);     // Voyager 2 closest approach
const OPAL_MS = Date.UTC(2018, 10, 5, 9, 3);     // OPAL 2018, middle of rotation 2
const SOUND = 580;                  // m/s, H2/He/CH4 (μ ≈ 2.6 g/mol, γ ≈ 1.45) at 72 K, 1 bar
const ASSET = f => new URL(`../../../assets/neptune/${f}`, import.meta.url).href;

/* ---------------------------------------------------------------------------------------------- */
/* helpers                                                                                          */
/* ---------------------------------------------------------------------------------------------- */
const vn = (x, d = 1) => dec(x, d);
const fmtInt = n => int(n);
const latStr = d => `${vn(Math.abs(d))}° ${d >= 0 ? t('B', 'N') : t('N', 'S')}`;
const wrap = a => ((a % TAU) + TAU) % TAU;
const wrapPi = a => wrap(a + Math.PI) - Math.PI;
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// Sromovsky et al. (1993) zonal wind (m/s, + = east) vs planetographic latitude (deg), tapered to 0 at the poles
const windU = d => { const d2 = d * d; return (-398 + 0.188 * d2 - 1.2e-5 * d2 * d2) * (1 - sstep(70, 90, Math.abs(d))); };
// meridional shear |du/dy| (1/s)
const shearAt = d => Math.abs((windU(d + 0.5) - windU(d - 0.5)) * R2D / RN);
// Kida vortex in uniform shear s: vorticity ω = s λ(λ+1)/(λ−1); fluid parcels circle the elliptical
// streamlines at Ω = ω λ/(1+λ)² = s λ²/(λ²−1)
const kidaOmega = (s, lam) => s * lam * lam / (lam * lam - 1);
// rotation rate (rad/s, + = east) relative to the interior of a feature with synodic period P hours
const relRate = Ph => TAU * (1 / (Ph * HOUR) - 1 / PROT);
const latlonJS = (latD, lonE) => { const la = latD * D2R; return new V3(Math.cos(la) * Math.cos(lonE), Math.sin(la), -Math.cos(la) * Math.sin(lonE)); };
const pad2 = n => String(n).padStart(2, '0');
const dateStr = ms => { const d = new Date(ms); return t(`${pad2(d.getUTCDate())}/${pad2(d.getUTCMonth() + 1)}/${d.getUTCFullYear()} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())} UTC`, `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())} UTC`); };

/* ---------------------------------------------------------------------------------------------- */
/* features (measured)                                                                              */
/* ---------------------------------------------------------------------------------------------- */
const GDS = {
  period: 18.296, lat0: -20.0, dLat: 1.3 / 30.44,          // deg/day equatorward (Sromovsky+ 1993)
  area: 0.2625 * 0.1333,                                       // semi-axes 6 500 × 3 300 km (13 000 × 6 600 km)
  lam: 2.0, dLam: 0.35, tilt: 8 * D2R, osc: 8 * DAY,
  dark: 0.17,                                                  // fractional darkening in blue (green ×0.75, red ×0.45)
};
const SCOOTER = { period: 16.8, lat: -42, lon0: -0.38 };
const DS2 = { period: 16.0, lat: -55, lon0: 0.45, ax: [0.088, 0.05], dark: 0.2 };
const NDS = { lat: 24.8, lonTex: 307.8 * D2R, ax: [0.22, 0.12], dark: 0.08 };   // NDS-2018 position in the OPAL map
const ARC_N = 820.1185 * D2R / DAY;                                          // Adams ring arcs, rad/s
const TRITON = { a: 354759 / 24764, r: 1353.4 / 24764, P: 5.876854 * DAY, inc: 156.885 * D2R };

/* ---------------------------------------------------------------------------------------------- */
/* shader                                                                                           */
/* ---------------------------------------------------------------------------------------------- */
const FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir;      // world, toward the Sun
uniform mat3 uW2B;         // world → body frame (y = spin axis, not spinning)
uniform float uFlat, uSpin, uSunI;
uniform vec3 uPh;          // flow-map phases: x = age of layer 1 (s), y = age of layer 2, z = weight of layer 1
uniform vec2 uSeed;
uniform vec4 uV1, uV1b;    // main vortex (GDS or NDS-2018): (lonE, lat, a, b) · (tilt, Ω (+ = CCW from outside), drift rad/s, darkness)
uniform vec4 uV2, uV2b;    // Dark Spot 2
uniform vec4 uSc;          // Scooter: lonE, lat
uniform float uEra, uMode, uOpalLon, uOpalRate, uTexOK;
uniform sampler2D uOpal;
uniform float uRingOn, uRingGain, uRingRot, uArcLib;
uniform vec4 uTri;         // Triton centre (body frame, R_N) and radius
uniform float uTriOn;
#define RNM 24764000.0
#define R2D 57.2957795
#define D2R 0.0174532925
// Deck reflectance at normal incidence/emission, true colour. Physically (OPAL I/F 0.54 at 547 nm, Irwin et al.
// 2024 chromaticity 0.50 : 0.84 : 1) it is (0.32, 0.54, 0.64); the host's per-channel ACES curve desaturates, so the
// value below is the one that, at the scene's exposure (1.4), reproduces the re-calibrated Voyager disc centre
// (sRGB 174, 220, 238) on screen.
const vec3 A_TRUE = vec3(0.21, 0.51, 0.96);
const vec3 CLOUD = vec3(1.2, 1.28, 1.38);       // methane-ice clouds, I/F ~0.85 (white, above most of the CH4)
// Voyager 2 'enhanced' rendition (1989 releases): deep blue, contrast stretched
const vec3 A_VOY = vec3(0.026, 0.044, 0.40);   // matches PIA01492's disc centre (sRGB 58, 86, 222)
// Minnaert exponents: OPAL 2018 fits are 0.50 / 0.80 / 0.88 (F657N, F547M, F467M); softened toward the limb
// (and μ clamped) so the disc profile matches the Irwin et al. 2024 true-colour image (no pink rim)
const vec3 MINN_K = vec3(0.60, 0.80, 0.86);
float gMu, gMu0, gFoot;
vec3 gSunQ;

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
vec3 latlon(float la, float lo){ return vec3(cos(la) * cos(lo), sin(la), -cos(la) * sin(lo)); }
void tframe(vec3 c, out vec3 e, out vec3 n){ e = normalize(vec3(c.z, 0.0, -c.x) + vec3(0.0, 0.0, 1e-6)); n = cross(c, e); }
float bump(float x, float c, float w){ float t = (x - c) / w; return exp(-t * t); }
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * c, 1.5); }
vec3 seedOff(float s){ return vec3(s * 13.7, s * 5.3, s * 9.1); }
float nWind(float d){ float d2 = d * d; return (-398.0 + 0.188 * d2 - 1.2e-5 * d2 * d2) * (1.0 - smoothstep(70.0, 90.0, abs(d))); }
float omegaAt(float la){ return nWind(la * R2D) / (RNM * max(cos(la), 0.06)); }

// tilted-ellipse coordinates of q around a vortex (1 = edge)
vec2 ell(vec3 q, vec4 V, float th){
  vec3 c = latlon(V.y, V.x), e, n; tframe(c, e, n); vec3 d = q - c;
  vec2 l = vec2(dot(d, e), dot(d, n));
  float cs = cos(th), sn = sin(th);
  return vec2(cs * l.x + sn * l.y, -sn * l.x + cs * l.y) / V.zw * step(0.0, dot(q, c)) + vec2(99.0) * step(dot(q, c), 0.0);
}
// where the parcel now at q was t seconds ago, for a Kida vortex: uniform rotation on elliptical streamlines
// inside, decaying as 1/r² outside, carried along with the vortex drift. w = blend weight against the zonal flow.
vec4 vortexSrc(vec3 q, vec4 V, vec4 Vb, float t){
  vec3 c = latlon(V.y, V.x);
  if (dot(q, c) < 0.8) return vec4(q, 0.0);
  vec3 e, n; tframe(c, e, n); vec3 d = q - c;
  vec2 l = vec2(dot(d, e), dot(d, n));
  float cs = cos(Vb.x), sn = sin(Vb.x);
  vec2 lt = vec2(cs * l.x + sn * l.y, -sn * l.x + cs * l.y) / V.zw;
  float r = length(lt);
  float m = 1.0 - smoothstep(1.3, 2.3, r);
  if (m <= 0.0) return vec4(q, 0.0);
  float a = -Vb.y * t / max(r * r, 1.0);
  float ca = cos(a), sa = sin(a);
  lt = vec2(ca * lt.x - sa * lt.y, sa * lt.x + ca * lt.y) * V.zw;
  l = vec2(cs * lt.x - sn * lt.y, sn * lt.x + cs * lt.y);
  vec3 c0 = latlon(V.y, V.x - Vb.z * t), e0, n0; tframe(c0, e0, n0);
  return vec4(normalize(c0 + e0 * l.x + n0 * l.y), m);
}
vec3 advect(vec3 q, float la, float t){
  vec3 p = rotY(omegaAt(la) * t) * q;           // zonal wind: the parcel came from lon − ω t
  vec4 v = vortexSrc(q, uV1, uV1b, t);
  p = mix(p, v.xyz, v.w);
  if (uEra < 0.5){ v = vortexSrc(q, uV2, uV2b, t); p = mix(p, v.xyz, v.w); }
  return normalize(p);
}
// thin zonal cirrus streaks (methane ice ~50–100 km above the deck)
float nStreak(vec3 p, float seed){ return fbm(vec3(p.x * 2.2, p.y * 40.0 + 1.5 * snoise(p * 2.5 + seedOff(seed)), p.z * 2.2) + seedOff(seed) * 0.5, 3); }
vec3 nField(vec3 p, float seed, int oct){
  vec3 Q = vec3(p.x, p.y * 3.0, p.z) + seedOff(seed);
  float a = fbm(Q * 2.0, oct);
  float c = fbm(Q * 7.0 + vec3(a * 1.5), max(oct - 1, 1));
  return vec3(a, nStreak(p, seed), c);
}
float streakBand(float d){
  if (uEra < 0.5) return 0.85 * bump(d, 27.0, 3.5) + 0.55 * bump(d, -28.0, 5.0) + 0.6 * bump(d, -40.0, 3.0) + 0.8 * bump(d, -70.0, 3.5) + 0.3 * bump(d, 45.0, 5.0);
  return 0.5 * bump(d, 28.0, 4.0) + 0.45 * bump(d, -37.0, 4.0) + 0.5 * bump(d, -70.0, 4.0);
}
// GDS companion clouds: SW rim (~215°), a tail trailing west along the southern edge, a thin NE streak
float gdsCloud(vec2 g, vec3 f){
  float rg = length(g) + 0.07 * f.z;
  float ang = atan(g.y, g.x);
  float rim = smoothstep(1.5, 1.08, rg) * smoothstep(0.86, 1.04, rg) * exp(-pow((ang + 2.53) / 0.8, 2.0));
  float tail = bump(g.y + 0.12 * f.x, -1.02, 0.17) * smoothstep(0.5, -0.4, g.x) * smoothstep(-3.4, -1.4, g.x);
  float ne = bump(g.y + 0.08 * f.x, 1.03, 0.08) * smoothstep(-0.3, 0.5, g.x) * smoothstep(3.0, 1.1, g.x);
  float cn = smoothstep(-0.2, 0.3, f.y + 0.2 * f.z);
  return max(0.95 * max(rim, tail) * cn, 0.7 * ne * smoothstep(-0.05, 0.4, f.y));
}
// rel = deck reflectance relative to A_TRUE, cloud = high-cloud cover, shd = cirrus shadow on the deck
void surface(vec3 q, float la, out vec3 rel, out float cloud, out float shd){
  float latD = la * R2D;
  float w1 = uPh.z, w2 = 1.0 - w1, wn = inversesqrt(w1 * w1 + w2 * w2);
  int oct = gFoot < 0.003 ? 4 : (gFoot < 0.01 ? 3 : 2);
  vec3 pa = advect(q, la, uPh.x), pb = advect(q, la, uPh.y);
  vec3 f = (nField(pa, uSeed.x, oct) * w1 + nField(pb, uSeed.y, oct) * w2) * wn;
  float lw = latD + 3.0 * f.x;
  float L = 1.0 + 0.035 * f.z + 0.02 * f.x;
  float sb = streakBand(latD);
  cloud = 0.0;
  vec2 g = vec2(99.0);
  if (uEra < 0.5){
    // zonal albedo pattern seen by Voyager: darker band 45–65°S, bright south polar cap, faint dark band ~27°N
    L *= 1.0 - 0.05 * bump(lw, -55.0, 8.0) + 0.07 * smoothstep(-66.0, -82.0, lw) - 0.03 * bump(lw, 27.0, 7.0) + 0.02 * bump(lw, -5.0, 12.0);
    float dark = 0.0;
    // Great Dark Spot: darker collar, bright cloudlets carried round inside by the circulation
    g = ell(q, uV1, uV1b.x);
    float rg = length(g) + 0.07 * f.z;
    float inG = smoothstep(1.04, 0.82, rg);
    dark = uV1b.w * (inG * (0.8 + 0.25 * f.z) + 0.45 * bump(rg, 0.9, 0.11));
    cloud = max(gdsCloud(g, f), 0.6 * inG * smoothstep(0.42, 0.75, f.y + 0.3 * f.z) * smoothstep(-0.2, 0.5, g.y));   // cloudlets in the northern half (PIA00052)
    // Scooter: fast bright cloud complex (shape changes every rotation)
    vec2 sc = ell(q, vec4(uSc.xy, 0.05, 0.027), 0.0);
    float rs = length(sc) + 0.4 * f.z - 0.25 * f.x;
    cloud = max(cloud, 0.95 * smoothstep(1.0, 0.55, rs) * step(length(sc), 4.0));
    // Dark Spot 2 with bright core and concentric dark/light bands (PIA00064)
    vec2 d2 = ell(q, uV2, uV2b.x);
    float r2 = length(d2) + 0.08 * f.z;
    dark = max(dark, uV2b.w * smoothstep(1.0, 0.65, r2));
    L *= 1.0 - 0.05 * cos(6.2832 * (r2 - 0.1 * f.x)) * smoothstep(3.6, 1.4, r2) * smoothstep(0.8, 1.2, r2);
    cloud = max(cloud, 0.9 * smoothstep(0.42, 0.15, r2 + 0.12 * f.y));
    rel = vec3(L) * (1.0 - dark * vec3(0.45, 0.75, 1.0));
  } else {
    // Hubble OPAL 2018 map, co-moving with NDS-2018; within each flow phase the other latitudes shear past it
    float om = omegaAt(la) - uOpalRate;
    float lonE = atan(-q.z, q.x);
    float v = 0.5 + latD / 180.0;
    float lod = log2(max(gFoot, 1e-5) * 1024.0 / 6.2832);
    vec4 ta = textureLod(uOpal, vec2((lonE - uOpalLon - om * uPh.x) / 6.2832, v), lod);
    vec4 tb = textureLod(uOpal, vec2((lonE - uOpalLon - om * uPh.y) / 6.2832, v), lod);
    vec4 tx = mix(vec4(0.5, 0.5, 0.5, 0.0), ta * w1 + tb * w2, uTexOK);
    // the map is blurred to ~1 000 km by Hubble: add the small-scale texture the telescope cannot resolve
    float north = smoothstep(36.0, 48.0, latD);                         // unobserved in 2018 (tilted away)
    rel = tx.rgb * 2.0 * L * (1.0 - 0.03 * bump(lw, 27.0, 7.0) * north);
    // high clouds from the 727-nm methane band; Hubble blurs them to ~1 000 km, so break them into streaks and
    // cloudlets with the advected field (and fade them out near the limb-degraded northern edge of the data)
    float ca = tx.a * (1.0 - smoothstep(34.0, 44.0, latD));
    // (the band saturates over the whole companion-cloud complex: treat it as cloud likelihood, carve cloudlets)
    float nz = clamp(0.45 + 1.2 * f.y + 0.6 * f.z, 0.0, 1.0);
    cloud = 0.85 * smoothstep(0.25, 0.9, ca * nz * 1.25);
    // sharpen NDS-2018 itself (Hubble: ~11 000 × 6 000 km), the map only holds its blurred image
    g = ell(q, uV1, uV1b.x);
    float rn = length(g) + 0.08 * f.z;
    rel *= 1.0 - uV1b.w * smoothstep(1.05, 0.75, rn) * vec3(0.45, 0.75, 1.0);
    sb *= 1.0 - 0.6 * smoothstep(1.6, 0.9, length(g));
  }
  // high methane-ice cirrus
  float stk = sb * smoothstep(0.3, 0.72, f.y);
  cloud = max(cloud, 0.8 * stk);
  // shadows of the high clouds on the deck, long near the terminator (cloud tops ~75 km above the deck)
  shd = 0.0;
  if (gMu0 > 0.0 && gMu0 < 0.5 && (sb > 0.03 || length(g) < 3.5)){
    vec3 tS = normalize(gSunQ - q * dot(gSunQ, q) + 1e-6);
    float off = min(0.003 * sqrt(1.0 - gMu0 * gMu0) / max(gMu0, 0.04), 0.07);
    vec3 q2 = normalize(q + tS * off);
    float s2 = (nStreak(advect(q2, la, uPh.x), uSeed.x) * w1 + nStreak(advect(q2, la, uPh.y), uSeed.y) * w2) * wn;
    shd = sb * smoothstep(0.3, 0.72, s2);
    if (uEra < 0.5) shd = max(shd, gdsCloud(ell(q2, uV1, uV1b.x), f));
    shd *= (1.0 - cloud) * smoothstep(0.0, 0.05, gMu0);
  }
}
// shadow of the (oblate) planet on point p (body frame) toward the Sun s
float bodyShadow(vec3 p, vec3 s, vec3 E){
  vec3 o = p * E, d = s * E;
  float tc = -dot(o, d) / dot(d, d);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.996, 1.004, length(o + d * tc));
}
float tritonShadow(vec3 p, vec3 s){
  if (uTriOn < 0.5) return 1.0;
  vec3 o = uTri.xyz - p; float tc = dot(o, s);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.9 * uTri.w, 1.1 * uTri.w, length(o - s * tc));
}
// rings, radius in R_N
float rh1(float i){ return hash13(vec3(i, 7.31, 1.97)); }
float rbox(float r, float r0, float w, float fw){ float we = max(w, fw); return (w / we) * (1.0 - smoothstep(0.35 * we, 0.65 * we, abs(r - r0))); }
// Galle (41 900 km), Le Verrier (53 200), Lassell plateau, Arago (57 200), Adams (62 932 km, 35 km wide) with the
// arcs Fraternité, Égalité 1/2, Liberté, Courage in the frame co-moving at 820.1185°/day (leading = larger a)
void ringProfile(float r, float azE, float fw, out float tau, out vec3 alb, out float dust){
  float a = mod(azE - uRingRot, TAU) * R2D;
  float arcs = smoothstep(0.0, 1.0, a) * (1.0 - smoothstep(9.0, 10.0, a))                   // Fraternité
             + smoothstep(13.0, 13.5, a) * (1.0 - smoothstep(14.0, 14.5, a))                // Égalité 1
             + smoothstep(15.0, 15.5, a) * (1.0 - smoothstep(18.0, 18.5, a))                // Égalité 2
             + uArcLib * smoothstep(24.5, 25.0, a) * (1.0 - smoothstep(28.5, 29.0, a))      // Liberté
             + uArcLib * smoothstep(33.0, 33.3, a) * (1.0 - smoothstep(33.8, 34.1, a));     // Courage
  float clump = 0.7 + 0.6 * rh1(floor(a * 6.0));
  tau = 1e-4 * rbox(r, 1.692, 0.08, fw) + 0.002 * rbox(r, 2.148, 0.0046, fw)
      + 1.5e-4 * smoothstep(2.15, 2.16, r) * (1.0 - smoothstep(2.30, 2.31, r))
      + 5e-4 * rbox(r, 2.31, 0.004, fw) + (0.003 + 0.08 * arcs * clump) * rbox(r, 2.5413, 0.0014, fw);
  alb = vec3(0.075, 0.062, 0.055);   // dark, slightly red ring material
  dust = 0.55;
}
vec3 rimCol(){ return uMode < 0.5 ? vec3(0.05, 0.085, 0.12) : (uMode < 1.5 ? vec3(0.01, 0.03, 0.16) : vec3(0.05, 0.055, 0.065)); }
vec3 sunDisc(vec3 rdW){
  float c = dot(rdW, uSunDir);
  // Sun at 30 AU: 1/900 of the flux at Earth, 64" across — essentially a dazzling star
  return bb(5772.0) * (smoothstep(0.999999, 0.9999995, c) * 40.0 + pow(max(c, 0.0), 20000.0) * 2.0 + pow(max(c, 0.0), 600.0) * 0.02);
}
vec3 tritonShade(vec3 p, vec3 rd, vec3 s, vec3 E){
  vec3 n = normalize(p - uTri.xyz);
  float m0 = dot(n, s);
  float nz = fbm(n * 7.0, 3);
  // nitrogen-frost south polar cap (pinkish) and the 'cantaloupe' terrain (greyer); Bond albedo ~0.76
  vec3 alb = mix(vec3(0.62, 0.60, 0.58), vec3(0.86, 0.76, 0.72), smoothstep(-0.1, -0.45, n.y + 0.15 * nz)) * (0.92 + 0.12 * nz);
  return alb * max(m0, 0.0) * bodyShadow(p, s, E) * uSunI;
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
  gSunQ = Rs * s;
  float tT = 1e9;
  if (uTriOn > 0.5){ vec2 h = sphereHit(ro, rd, uTri.xyz, uTri.w); if (h.x < h.y && h.x > 0.0) tT = h.x; }
  float tN = min(tP, tT);
  vec3 col;
  if (tP < 1e8 && tP <= tT){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = Rs * (p * E);
    float la = asin(clamp(n.y, -1.0, 1.0));
    gMu0 = dot(n, s); gMu = max(dot(n, -rd), 0.0);
    gFoot = tP * pixAng / max(gMu, 0.15);
    vec3 rel; float cloud, shd;
    surface(q, la, rel, cloud, shd);
    float sh = tritonShadow(p, s);
    float m0 = max(gMu0, 0.0), mu = max(gMu, 0.2);
    if (uMode < 1.5){
      vec3 lit = pow(vec3(m0), MINN_K) * pow(vec3(mu), MINN_K - 1.0);
      // grazing sunlight crosses the stratospheric haze + H2 above the clouds: soft, slightly reddened terminator
      vec3 tIn = exp(-vec3(0.025, 0.035, 0.05) / max(m0, 0.01));
      lit *= tIn;
      lit.b *= mix(0.85, 1.0, smoothstep(0.1, 0.5, gMu));            // slightly greener extreme limb (Irwin et al. image)
      vec3 deck = A_TRUE * rel * lit * (1.0 - 0.6 * shd);
      vec3 cl = CLOUD * m0 * pow(mu, -0.12) * sqrt(tIn);       // high clouds sit above most of the methane
      col = mix(deck, cl, cloud);
      col += A_TRUE * 0.012 * smoothstep(-0.12, 0.03, gMu0) * (1.0 - smoothstep(0.0, 0.15, gMu0));   // twilight in the haze
      if (uMode > 0.5){
        // Voyager 1989 release processing: contrast stretch + blue colour balance
        // (orange/green/violet filters stretched hard; brightness carried by the green–blue light)
        float xl = min(dot(col / A_TRUE, vec3(0.1, 0.4, 0.5)), 2.6);
        col = A_VOY * pow(xl, 2.4) * 1.3 + vec3(0.30, 0.42, 0.40) * smoothstep(1.25, 2.4, xl);
      }
    } else {
      // JWST NIRCam (1–2 µm): methane absorbs the deck, high CH4-ice clouds, the equatorial line and the
      // south-polar cloud collar stand out; the haze brightens toward the limb
      // (displayed with the stretch of the STScI release: the disc is a pale lavender-grey, clouds saturate)
      float lD = la * R2D;
      float ir = 0.10 * (0.85 + 0.3 * rel.g) + 0.12 * bump(lD, 0.0, 1.6) + 0.08 * smoothstep(-62.0, -72.0, lD) * (1.0 - smoothstep(-80.0, -86.0, lD)) + 0.06 * smoothstep(68.0, 84.0, lD);
      col = vec3(0.86, 0.86, 1.0) * (ir * pow(m0, 0.8) * pow(mu, -0.3) + 1.3 * cloud * m0) * (1.0 - 0.5 * shd);
    }
    col *= uSunI * sh;
    col += rimCol() * uSunI * pow(1.0 - gMu, 5.0) * smoothstep(-0.3, 0.2, gMu0) * (0.35 + 0.65 * hg(dot(rd, s), 0.7)) * sh;
  } else if (tT < 1e8){
    col = tritonShade(ro + rd * tT, rd, s, E);
  } else {
    col = sky(rdW, 1.0) * 0.3 + sunDisc(rdW);
    float tc = -B / A;
    if (tc > 0.0){
      vec3 pc = roE + rdE * tc;
      float hgt = max(length(pc) - 1.0, 0.0);
      col += rimCol() * smoothstep(-0.3, 0.2, dot(normalize(pc), s)) * (0.35 + 0.65 * hg(dot(rd, s), 0.7)) * uSunI * exp(-hgt / 0.012);
    }
    if (uTriOn > 0.5){
      // Triton as a point when it is smaller than a pixel
      vec3 oc = uTri.xyz - ro; float tc2 = dot(oc, rd);
      if (tc2 > 0.0 && tc2 < tP){
        float th = length(oc - rd * tc2) / tc2, rho = uTri.w / tc2, sg = pixAng * 0.9;
        float ph = 0.5 + 0.5 * dot(normalize(-oc), s);
        col += vec3(0.75, 0.68, 0.63) * ph * uSunI * min(rho * rho / (sg * sg), 1.0) * exp(-th * th / (sg * sg)) * step(rho, 1.5 * pixAng);
      }
    }
  }
  if (uRingOn > 0.5 && abs(rd.y) > 1e-6){
    float tR = -ro.y / rd.y;
    if (tR > 0.0 && tR < tN){
      vec3 pr = ro + rd * tR;
      float r = length(pr.xz);
      if (r > 1.6 && r < 2.6){
        float mu = abs(rd.y), mu0 = max(abs(s.y), 1e-4);
        vec2 vh = normalize(rd.xz + 1e-7);
        float cr = dot(pr.xz / r, vh);
        float fw = min(tR * pixAng * sqrt(cr * cr / (mu * mu) + 1.0 - cr * cr), 0.3);
        float tau, dust; vec3 alb;
        ringProfile(r, atan(-pr.z, pr.x), fw, tau, alb, dust);
        if (tau > 0.0){
          float shd = bodyShadow(pr, s, E);
          float c = dot(s, rd);
          float P = mix(hg(c, -0.3), hg(c, 0.75), dust);
          float IF;
          if (s.y * rd.y < 0.0) IF = 0.25 * P * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu)));
          else { float dm = mu - mu0; IF = abs(dm) > 1e-3 ? 0.25 * P * mu0 / dm * (exp(-tau / mu) - exp(-tau / mu0)) : 0.25 * P * mu0 * tau / (mu * mu) * exp(-tau / mu); }
          if (uMode > 1.5) alb = vec3(0.07);
          col = col * exp(-tau / mu) + alb * max(IF, 0.0) * uSunI * uRingGain * shd * (uMode > 1.5 ? 25.0 : 1.0);   // in the CH4 bands the disc is ~30× darker: rings stand out
        }
      }
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ---------------------------------------------------------------------------------------------- */
/* JS: geometry, clocks                                                                             */
/* ---------------------------------------------------------------------------------------------- */
// Body frame for obliquity ob (deg) with the Sun (world azimuth sunAz) at sub-solar latitude B (deg).
function orient(st, BDeg, sunAzDeg) {
  const az = sunAzDeg * D2R, ob = OBLIQ * D2R;
  st.sun.set(Math.sin(az), 0, Math.cos(az));
  const c = THREE.MathUtils.clamp(Math.sin(BDeg * D2R) / Math.sin(ob), -1, 1);
  const paz = az + Math.acos(c);
  const P = st.pole.set(Math.sin(ob) * Math.sin(paz), Math.cos(ob), Math.sin(ob) * Math.cos(paz));
  const a = new V3(0, 1, 0);
  const xb = new V3().crossVectors(a, P).normalize(), zb = new V3().crossVectors(xb, P);
  st.W2B.set(xb.x, xb.y, xb.z, P.x, P.y, P.z, zb.x, zb.y, zb.z);
  st.B2W.copy(st.W2B).transpose();
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
// body point (non-spinning frame) → planetographic latitude and body-fixed east longitude
function bodyLatLon(p, spin) {
  const k = 1 / (1 - FLAT);
  const n = new V3(p.x, p.y * k * k, p.z).normalize();
  const c = Math.cos(spin), s = Math.sin(spin);
  const qx = c * p.x - s * p.z, qz = s * p.x + c * p.z;     // Ry(−spin)
  return { lat: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) * R2D, lonE: Math.atan2(-qz, qx), n };
}

/* ---- Voyager 2 flyby: a Keplerian hyperbola (GM_N = 6.8365e15 m³/s², periapsis 29 700 km = 4 950 km above
   the 1-bar level, v∞ ≈ 16 km/s), arriving from the sunward side and passing over the north pole ---- */
const GM = 6.8365e15, RP = 29700e3, VINF = 16.0e3;
const HA = GM / (VINF * VINF), HE = 1 + RP * VINF * VINF / GM, HN = Math.sqrt(GM / (HA * HA * HA));
const NU_INF = Math.acos(-1 / HE);
function hyper(tSec) {
  const M = HN * tSec;
  let Hh = Math.asinh(M / HE);
  for (let i = 0; i < 30; i++) { const f = HE * Math.sinh(Hh) - Hh - M; Hh -= f / (HE * Math.cosh(Hh) - 1); if (Math.abs(f) < 1e-12) break; }
  const x = HA * (HE - Math.cosh(Hh)), y = HA * Math.sqrt(HE * HE - 1) * Math.sinh(Hh);
  const r = HA * (HE * Math.cosh(Hh) - 1);
  return { x: x / RN, y: y / RN, r, v: Math.sqrt(GM * (2 / r + 1 / HA)) };
}
function flyFrame(st) {
  // incoming asymptote ~28° from the Sun direction (south side), in the plane of the Sun and the spin axis, so that
  // periapsis (118° further round) lies over the north-polar terminator, as for Voyager 2
  const u = st.sun.clone(), w = st.pole.clone().addScaledVector(u, -st.pole.dot(u)).normalize();
  const el = (90 - NU_INF * R2D) * D2R;
  const din = u.clone().multiplyScalar(Math.cos(el)).addScaledVector(w, Math.sin(el));
  const wp = u.clone().multiplyScalar(-Math.sin(el)).addScaledVector(w, Math.cos(el));
  const P = din.clone().multiplyScalar(Math.cos(NU_INF)).addScaledVector(wp, Math.sin(NU_INF)).normalize();
  const Q = P.clone().multiplyScalar(Math.cos(NU_INF)).sub(din).multiplyScalar(1 / Math.sin(NU_INF)).normalize();
  return { P, Q };
}

/* ---------------------------------------------------------------------------------------------- */
/* scene                                                                                            */
/* ---------------------------------------------------------------------------------------------- */
let hostSpeed = 6;   // host time-speed slider, for the time-lapse read-out
const lapseNote = 'Giây trên Sao Hải Vương trôi qua trong mỗi giây thật (đã nhân với thanh “Tốc độ thời gian” của trình xem). Ở ×4 000 hành tinh quay một vòng trong ~15 s; ở ×50 000 thấy Vết Tối Lớn đổi dạng (chu kỳ 8 ngày) và Scooter vượt qua nó.';
const lapseNoteEn = 'Seconds on Neptune per real second (already multiplied by the viewer\'s “Time speed” slider). At ×4,000 the planet turns once in ~15 s; at ×50,000 you can watch the Great Dark Spot change shape (8-day cycle) and the Scooter overtake it.';

const NEPTUNE = {
  id: 'neptune', name: 'Sao Hải Vương', category: 'planets', order: 43,
  subtitle: 'Hành tinh băng khổng lồ · gió mạnh nhất hệ Mặt Trời',
  swatch: ['#a9d3ea', '#1d3f7a'],
  blurb: 'Ở 30 AU, Sao Hải Vương chỉ nhận 1/900 ánh nắng của Trái Đất, vậy mà có <b>gió mạnh nhất hệ Mặt Trời</b>: dòng tia xích đạo thổi ngược chiều quay ~400 m/s, một số đám mây đo được tới ~600 m/s — sát tốc độ âm thanh ở đó. Khí mê-tan hấp thụ ánh sáng đỏ nên nó có màu xanh lơ nhạt (không xanh thẫm như ảnh Voyager 1989 đã được tăng cường).',
  intro: 'Phím <b>G</b>: bám theo Vết Tối Lớn và tua nhanh ×50 000 để thấy nó xoay, đổi dạng và bị Scooter vượt. Phím <b>V</b>: bay qua như Voyager 2. Đổi <b>Màu</b> để so sánh màu thật (2024) với ảnh Voyager tăng cường và JWST.',
  facts: [
    ['Bán kính xích đạo / cực', '24 764 km / 24 341 km (3,9 Trái Đất)'], ['Khối lượng', '1,024×10²⁶ kg (17,1 Trái Đất)'],
    ['Chu kỳ tự quay', '16 giờ 6,6 phút (vô tuyến, Voyager 2)'], ['Độ nghiêng trục', '28,32°'],
    ['Khoảng cách tới Mặt Trời', '30,07 AU · 1 năm = 164,8 năm Trái Đất'], ['Nhiệt độ ở mức 1 bar', '72 K (−201 °C)'],
    ['Gió xích đạo', '≈ −400 m/s (1 440 km/h, về tây); mây lẻ tới ~600 m/s (≈ 2 100 km/h)'],
    ['Tốc độ âm thanh (1 bar)', '≈ 580 m/s'],
    ['Vết Tối Lớn (1989)', '13 000 × 6 600 km · chu kỳ 18,3 h · đổi dạng mỗi 8 ngày · tan rã trước 1994'],
    ['Nhiệt lượng tỏa ra', '2,6 lần năng lượng nhận từ Mặt Trời'],
    ['Vành Adams & các cung', '62 932 km · Fraternité, Égalité, Liberté, Courage'],
    ['Triton', 'R = 1 353 km · quỹ đạo nghịch hành 5,88 ngày, nghiêng 156,9°'],
    ['Ảnh & dữ liệu', 'Voyager 2 (NASA/JPL, 1989); Hubble OPAL 2018 (Simon, Wong et al., CC BY 4.0, doi:10.17909/T9G593); màu thật: Irwin et al. 2024 (MNRAS); gió: Sromovsky et al. 1993'],
  ],
  en: {
    name: 'Neptune',
    subtitle: 'Ice giant · the strongest winds in the Solar System',
    blurb: 'At 30 AU Neptune receives only 1/900 of the sunlight that reaches Earth, yet it has <b>the strongest winds in the Solar System</b>: the equatorial jet blows against the rotation at ~400 m/s, and some clouds have been clocked at ~600 m/s — close to the local speed of sound. Methane absorbs red light, giving the planet a pale azure color (not the deep blue of the contrast-enhanced 1989 Voyager images).',
    intro: 'Key <b>G</b>: follow the Great Dark Spot and speed time up ×50,000 to watch it spin, change shape and get overtaken by the Scooter. Key <b>V</b>: fly past like Voyager 2. Change <b>Color</b> to compare true color (2024) with the enhanced Voyager images and JWST.',
    facts: [
      ['Equatorial / polar radius', '24,764 km / 24,341 km (3.9 Earths)'], ['Mass', '1.024×10²⁶ kg (17.1 Earths)'],
      ['Rotation period', '16 h 6.6 min (radio, Voyager 2)'], ['Axial tilt', '28.32°'],
      ['Distance from the Sun', '30.07 AU · 1 year = 164.8 Earth years'], ['Temperature at 1 bar', '72 K (−201 °C)'],
      ['Equatorial wind', '≈ −400 m/s (1,440 km/h, westward); individual clouds up to ~600 m/s (≈ 2,100 km/h)'],
      ['Speed of sound (1 bar)', '≈ 580 m/s'],
      ['Great Dark Spot (1989)', '13,000 × 6,600 km · 18.3 h period · changes shape every 8 days · gone by 1994'],
      ['Heat emitted', '2.6 times the energy received from the Sun'],
      ['Adams ring & arcs', '62,932 km · Fraternité, Égalité, Liberté, Courage'],
      ['Triton', 'R = 1,353 km · retrograde 5.88-day orbit, inclination 156.9°'],
      ['Images & data', 'Voyager 2 (NASA/JPL, 1989); Hubble OPAL 2018 (Simon, Wong et al., CC BY 4.0, doi:10.17909/T9G593); true color: Irwin et al. 2024 (MNRAS); winds: Sromovsky et al. 1993'],
    ],
  },
  unitKm: RN / 1000,
  camera: { distance: 3.4, polar: 84, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.4, bloom: 0.3,
  params: [
    { key: 'era', label: 'Thời điểm', type: 'seg', options: [['v89', 'Voyager 2 · 1989'], ['h18', 'Hubble · 2018']], default: 'v89', note: '1989: Vết Tối Lớn, Scooter, Vết Tối 2 (dựng lại theo ảnh Voyager 2). 2018: bản đồ toàn cầu thật của chương trình OPAL (Hubble) với Vết Tối Bắc NDS-2018 và mây đồng hành.',
      en: { label: 'Epoch', note: '1989: the Great Dark Spot, the Scooter and Dark Spot 2 (reconstructed from Voyager 2 images). 2018: the real global map from Hubble\'s OPAL program, with the northern dark spot NDS-2018 and its companion clouds.' } },
    { key: 'color', label: 'Màu', type: 'seg', options: [['real', 'Màu thật (hiệu chỉnh 2024)'], ['voy', 'Voyager tăng cường'], ['jwst', 'JWST hồng ngoại gần']], default: 'real', note: 'Irwin et al. (2024) hiệu chỉnh lại ảnh Voyager bằng phổ Hubble/STIS và VLT/MUSE: Sao Hải Vương xanh lơ nhạt, gần giống Sao Thiên Vương. Ảnh 1989 đã bị tăng tương phản và đẩy màu xanh để làm nổi các đám mây. JWST (2022, 1–2 µm): mê-tan hấp thụ gần hết, chỉ mây băng cao, vạch sáng xích đạo và vành còn sáng.',
      en: { label: 'Color', options: ['True color (2024 recalibration)', 'Enhanced Voyager', 'JWST near-infrared'], note: 'Irwin et al. (2024) recalibrated the Voyager images using Hubble/STIS and VLT/MUSE spectra: Neptune is a pale azure, much like Uranus. The 1989 images had their contrast stretched and their blue boosted to bring out the clouds. JWST (2022, 1–2 µm): methane absorbs almost everything, so only high ice clouds, the bright equatorial band and the rings stay bright.' } },
    { key: 'lapse', label: 'Tua nhanh thời gian', type: 'range', min: 1.5, max: 4.5, step: 0.05, default: 2.8, fmt: v => '×' + fmtInt(10 ** v * hostSpeed), note: lapseNote,
      en: { label: 'Time-lapse', note: lapseNoteEn } },
    { key: 'frame', label: 'Khung nhìn', type: 'seg', options: [['iner', 'Quán tính'], ['gds', 'Bám theo vết tối']], default: 'iner', note: 'Bám theo vết tối: camera quay cùng xoáy (Vết Tối Lớn hoặc NDS-2018), nên thấy các dải mây trượt qua nó với tốc độ tương đối thật.',
      en: { label: 'View frame', options: ['Inertial', 'Follow the dark spot'], note: 'Follow the dark spot: the camera turns with the vortex (the Great Dark Spot or NDS-2018), so you see the cloud bands slide past it at their true relative speeds.' } },
    { key: 'sunAz', label: 'Hướng Mặt Trời', type: 'range', min: 0, max: 360, step: 1, default: 50, fmt: v => v.toFixed(0) + '°', note: 'Đưa đường phân chia ngày–đêm tới gần vùng mây ti để thấy chúng đổ bóng xuống tầng mây bên dưới (cao hơn ~50–100 km).',
      en: { label: 'Sun direction', note: 'Bring the terminator close to the cirrus clouds to see them cast shadows on the cloud deck below (they float ~50–100 km higher).' } },
    { key: 'rings', label: 'Vành & cung Adams', type: 'toggle', default: true, en: { label: 'Rings & Adams arcs' } },
    { key: 'ringGain', label: 'Độ sáng vành (phóng đại)', type: 'range', min: 1, max: 200, step: 1, default: 60, fmt: v => '×' + v.toFixed(0), note: 'Vành Sao Hải Vương là bụi tối, độ dày quang học 10⁻⁴–0,1: gần như vô hình nếu không phóng đại; sáng nhất khi nhìn ngược sáng.',
      en: { label: 'Ring brightness (exaggerated)', note: 'Neptune\'s rings are dark dust with an optical depth of 10⁻⁴–0.1: almost invisible unless exaggerated, and brightest when seen backlit.' } },
    { key: 'triton', label: 'Triton', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Vết Tối Lớn', key: 'g', title: 'Bay tới Vết Tối Lớn, bám theo nó và tua nhanh', run: ctx => NEPTUNE.__inst?.gdsShot(ctx),
      en: { label: 'Great Dark Spot', title: 'Fly to the Great Dark Spot, follow it and speed up time' } },
    { label: 'Voyager 2 bay qua', key: 'v', title: 'Tái hiện chuyến bay sát ngày 25/8/1989 (bấm lần nữa để dừng)', run: ctx => NEPTUNE.__inst?.flyby(ctx),
      en: { label: 'Voyager 2 flyby', title: 'Recreate the close flyby of 25 August 1989 (press again to stop)' } },
    { label: 'Vành ngược sáng', key: 'n', title: 'Ra sau Sao Hải Vương nhìn về phía Mặt Trời: vành bụi và các cung Adams', run: ctx => NEPTUNE.__inst?.backlit(ctx),
      en: { label: 'Backlit rings', title: 'Go behind Neptune and look toward the Sun: the dusty rings and the Adams arcs' } },
  ],
  create(ctx) {
    const R = ctx.renderer;
    const blank = new THREE.DataTexture(new Uint8Array([128, 128, 128, 0]), 1, 1, THREE.RGBAFormat); blank.needsUpdate = true;
    const U = {
      ...ctx.shared,
      uSunDir: { value: new V3(0, 0, 1) }, uW2B: { value: new THREE.Matrix3() },
      uFlat: { value: FLAT }, uSpin: { value: 0 }, uSunI: { value: 1.0 },
      uPh: { value: new V3(0, 0, 1) }, uSeed: { value: new THREE.Vector2() },
      uV1: { value: new THREE.Vector4() }, uV1b: { value: new THREE.Vector4() },
      uV2: { value: new THREE.Vector4() }, uV2b: { value: new THREE.Vector4() },
      uSc: { value: new THREE.Vector4() },
      uEra: { value: 0 }, uMode: { value: 0 }, uOpalLon: { value: 0 }, uOpalRate: { value: 0 }, uTexOK: { value: 0 },
      uOpal: { value: blank },
      uRingOn: { value: 1 }, uRingGain: { value: 60 }, uRingRot: { value: 0 }, uArcLib: { value: 1 },
      uTri: { value: new THREE.Vector4(0, 0, 0, TRITON.r) }, uTriOn: { value: 1 },
    };
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    const ready = new Promise((resolve, reject) => {
      const load = (url, again) => new THREE.TextureLoader().load(url, tex => {
        tex.colorSpace = THREE.NoColorSpace;               // relative reflectance, linear
        tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = true;
        tex.needsUpdate = true;
        U.uOpal.value = tex; R.initTexture(tex); U.uTexOK.value = 1;
        resolve();
      }, undefined, () => again ? setTimeout(() => load(url + '?r=' + Date.now(), false), 500) : reject(new Error(t('không tải được bản đồ OPAL 2018', 'could not load the 2018 OPAL map'))));
      load(ASSET('neptune_opal2018.webp'), true);
    });

    const st = {
      W2B: U.uW2B.value, B2W: new THREE.Matrix3(), sun: U.uSunDir.value, pole: new V3(),
      T: null, lastSim: null, ph: 0, T0: [0, 0], seed: [0, 0.5], P: 1,
      spin: 0, spinOff: 0, follow: false, followC: 0, lastFrame: 'iner', lastEra: null,
      main: { lon: 0, lat: 0 }, triTh0: 2.3, arcOff: 0, fly: null, tween: null, pending: null, entered: false,
    };
    const omega = TAU / PROT;
    const query = new URLSearchParams(location.search);
    const lapseRate = p => 10 ** p.lapse;              // planet seconds per simulation second

    // ---- feature state at planet time T ------------------------------------------------------
    function features(p) {
      const T = st.T, era = p.era === 'h18' ? 1 : 0;
      const f = { era };
      if (!era) {
        const day = ((T / DAY + 60) % 120 + 120) % 120 - 60;   // the scene replays Voyager's 120-day window
        const ph = TAU * T / GDS.osc;
        const lam = GDS.lam + GDS.dLam * Math.sin(ph);
        const lat = GDS.lat0 + GDS.dLat * day;
        f.day = day; f.lam = lam;
        f.gds = { lon: wrap(relRate(GDS.period) * T), lat, a: Math.sqrt(GDS.area * lam), b: Math.sqrt(GDS.area / lam), tilt: GDS.tilt * Math.cos(ph), rate: relRate(GDS.period) };
        f.gds.Om = kidaOmega(shearAt(lat), GDS.lam);                         // + = counter-clockwise (southern anticyclone)
        f.sco = { lon: wrap(SCOOTER.lon0 + relRate(SCOOTER.period) * T), lat: SCOOTER.lat, rate: relRate(SCOOTER.period) };
        f.ds2 = { lon: wrap(DS2.lon0 + relRate(DS2.period) * T), lat: DS2.lat + 1.2 * Math.sin(TAU * T / (36 * HOUR)), rate: relRate(DS2.period) };
        f.ds2.Om = kidaOmega(shearAt(DS2.lat), 1.8);
        f.main = f.gds;
      } else {
        const rate = windU(NDS.lat) / (RN * Math.cos(NDS.lat * D2R));
        f.nds = { lon: wrap(NDS.lonTex + rate * T), lat: NDS.lat, a: NDS.ax[0], b: NDS.ax[1], tilt: 0, rate };
        f.nds.Om = -kidaOmega(shearAt(NDS.lat), 1.6);                       // northern anticyclone: clockwise
        f.main = f.nds;
        f.day = T / DAY;
      }
      return f;
    }
    // ---- clock: planet time, flow-map phases, spin -----------------------------------------
    function clock(c) {
      const p = c.params, sim = c.time.sim, rate = lapseRate(p);
      const Pof = r => Math.min(Math.max(r * 18, 2000), 6e4);   // phase period (planet s): ~3 s of wall time
      if (st.T === null) {
        st.T = sim * rate; st.lastSim = sim;
        const P = Pof(rate), k = st.T / P;
        st.ph = k - Math.floor(k);
        st.T0 = [st.T - st.ph * P, st.T - ((st.ph + 0.5) % 1) * P];
        st.seed = [Math.floor(k) % 29, (Math.floor(k + 0.5) % 29) + 0.5];
      } else {
        const ds = Math.max(0, sim - st.lastSim); st.lastSim = sim;
        const dT = ds * rate;
        st.T += dT;
        const before = st.ph;
        st.ph += dT / Pof(rate);
        if (st.ph >= 1) { st.ph -= Math.floor(st.ph); st.T0[0] = st.T - st.ph * Pof(rate); st.seed[0] = (st.seed[0] + 7) % 29; }
        if ((before + 0.5) % 1 > (st.ph + 0.5) % 1 || dT / Pof(rate) >= 1) { st.T0[1] = st.T - ((st.ph + 0.5) % 1) * Pof(rate); st.seed[1] = (Math.floor(st.seed[1]) + 11) % 29 + 0.5; }
      }
      orient(st, p.era === 'h18' ? -25 : -23, p.sunAz);
      st.f = features(p);
      const m = st.f.main;
      // frame: inertial (the planet spins) or co-moving with the main vortex
      if (st.lastEra !== null && st.f.era !== st.lastEra && p.frame === 'gds' && st.lastFrame === 'gds') st.followC = st.lastMainInertial;
      if (p.frame !== st.lastFrame) {
        if (p.frame === 'gds') st.followC = m.lon + st.spin;
        else st.spinOff = st.spin - omega * st.T;
      }
      st.lastFrame = p.frame; st.lastEra = st.f.era;
      st.spin = p.frame === 'gds' ? st.followC - m.lon : omega * st.T + st.spinOff;
      st.lastMainInertial = m.lon + st.spin;
    }
    st.clock = clock;
    // rotate the planet so that body point (lat, lonE) faces world direction dirW
    st.faceFeature = (latD, lonE, dirW) => {
      const qf = latlonJS(latD, lonE), cb = dirW.clone().normalize().applyMatrix3(st.W2B);
      const want = Math.atan2(cb.x, cb.z) - Math.atan2(qf.x, qf.z);
      if (st.lastFrame === 'gds') st.followC += want - st.spin;
      else st.spinOff += want - st.spin;
      st.spin = want;
      st.lastMainInertial = st.f.main.lon + st.spin;
    };
    st.subPoint = dirW => {
      const d = dirW.clone().normalize().applyMatrix3(st.W2B), k = 1 / (1 - FLAT);
      const s = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
      return bodyLatLon(d.multiplyScalar(s), st.spin);
    };
    const tritonPos = T => {
      const th = st.triTh0 + TAU * T / TRITON.P, i = TRITON.inc;
      // e1 = node, e2 = (0, sin i, −cos i): prograde for i = 0, retrograde for i = 157°
      return new V3(Math.cos(th), Math.sin(th) * Math.sin(i), -Math.sin(th) * Math.cos(i)).multiplyScalar(TRITON.a);
    };

    // ---- camera helpers -----------------------------------------------------------------------
    const tweenTo = (c, pos, dur, then) => {
      const from = c.camera.position.clone();
      st.tween = { t0: performance.now(), dur: dur * 1000, from, to: pos.clone(), then };
    };
    function runTween(c) {
      const tw = st.tween; if (!tw) return;
      const k = Math.min(1, (performance.now() - tw.t0) / tw.dur), e = k * k * (3 - 2 * k);
      const ra = tw.from.length(), rb = tw.to.length();
      const dir = tw.from.clone().normalize().lerp(tw.to.clone().normalize(), e).normalize();
      c.camera.position.copy(dir.multiplyScalar(Math.exp(Math.log(ra) + (Math.log(rb) - Math.log(ra)) * e)));
      c.controls.target.set(0, 0, 0); c.camera.lookAt(0, 0, 0);
      if (k >= 1) { st.tween = null; tw.then?.(); }
    }
    function startFly(c, hStart) {
      const p = c.params;
      p.frame = 'iner'; p.era = 'v89'; p.lapse = 2.725;            // 23 h of flyby in ~26 s at the default speed
      clock(c);
      st.fly = { Tca: st.T - hStart * HOUR, end: 7, ...flyFrame(st) };
      // put Triton where Voyager met it: CA + 5 h 08 min, ~40 000 km away
      const tTri = st.fly.Tca + 5.13 * HOUR;
      const sc = flyPos(5.13).applyMatrix3(st.W2B);
      let best = 1e9, bestTh = 0;
      for (let k = 0; k < 720; k++) {
        st.triTh0 = k / 720 * TAU;
        const d = tritonPos(tTri).distanceTo(sc);
        if (d < best) { best = d; bestTh = st.triTh0; }
      }
      st.triTh0 = bestTh;
      c.controls.minDistance = 1.1;
    }
    function stopFly(c) {
      if (!st.fly) return false;
      st.fly = null; c.controls.minDistance = NEPTUNE.camera.min; c.controls.target.set(0, 0, 0); c.camera.lookAt(0, 0, 0);
      return true;
    }
    const flyPos = h => { const o = hyper(h * HOUR); return st.fly.P.clone().multiplyScalar(o.x).addScaledVector(st.fly.Q, o.y); };
    function runFly(c) {
      const h = (st.T - st.fly.Tca) / HOUR;
      if (h > st.fly.end) { st.fly = null; c.controls.minDistance = NEPTUNE.camera.min; c.controls.target.set(0, 0, 0); c.toast?.(t('Voyager 2 rời Sao Hải Vương, hướng về phía nam mặt phẳng hoàng đạo — đến nay nó đã cách Mặt Trời hơn 140 AU.', 'Voyager 2 leaves Neptune, heading south of the ecliptic plane — it is now more than 140 AU from the Sun.'), 6000); return; }
      const pos = flyPos(h), vel = flyPos(h + 0.01).sub(pos).normalize(), r = pos.length();
      // near closest approach look down and back at the sunlit limb (as the CA images did); otherwise at the planet
      const w = 0.9 * sstep(3.5, 1.25, r);
      const look = pos.clone().normalize().negate().addScaledVector(vel, -w).normalize();
      c.camera.position.copy(pos);
      c.controls.target.copy(pos).addScaledVector(look, r * (1 - 0.6 * w / 0.9));
      c.camera.lookAt(c.controls.target);
    }

    const inst = {
      material, ready, st,
      enter(c) { st.pending = 'enter'; hostSpeed = c.bhGlobals?.timeScale ?? 6; },
      exit(c) { stopFly(c); st.tween = null; },
      update(dt, c) {
        const p = c.params;
        hostSpeed = c.bhGlobals?.timeScale ?? hostSpeed;
        clock(c);
        if (st.pending === 'enter') {
          st.pending = null;
          if (!st.entered) {
            st.entered = true;
            const cam = NEPTUNE.camera, dir = new V3().setFromSphericalCoords(1, cam.polar * D2R, (cam.azimuth - 12) * D2R);
            st.faceFeature(st.f.main.lat, st.f.main.lon, dir);
            if (query.get('nepShot') === 'gds') inst.gdsShot(c, true);
            if (query.get('nepShot') === 'back') inst.backlit(c, true);
            if (query.get('nepFly')) { startFly(c, +query.get('nepFly')); }
          }
        }
        if (st.fly) runFly(c);
        runTween(c);
        const f = st.f;
        U.uSpin.value = wrap(st.spin);
        U.uPh.value.set(st.T - st.T0[0], st.T - st.T0[1], 1 - Math.abs(2 * st.ph - 1));
        U.uSeed.value.set(st.seed[0], st.seed[1]);
        U.uEra.value = f.era;
        U.uMode.value = p.color === 'voy' ? 1 : p.color === 'jwst' ? 2 : 0;
        const m = f.main;
        U.uV1.value.set(m.lon, m.lat * D2R, m.a, m.b);
        U.uV1b.value.set(m.tilt, m.Om, m.rate, f.era ? NDS.dark : GDS.dark);
        if (!f.era) {
          U.uV2.value.set(f.ds2.lon, f.ds2.lat * D2R, DS2.ax[0], DS2.ax[1]);
          U.uV2b.value.set(0, f.ds2.Om, f.ds2.rate, DS2.dark);
          U.uSc.value.set(f.sco.lon, f.sco.lat * D2R, 0, 0);
        } else {
          U.uOpalLon.value = wrap(m.lon - NDS.lonTex);
          U.uOpalRate.value = m.rate;
        }
        U.uRingOn.value = p.rings === false ? 0 : 1;
        U.uRingGain.value = p.ringGain;
        // arcs: Liberté and Courage have faded since 1989 (de Pater et al. 2005, 2018)
        U.uArcLib.value = f.era ? 0.25 : 1;
        U.uRingRot.value = wrap(st.arcOff + ARC_N * st.T);
        U.uTriOn.value = p.triton === false ? 0 : 1;
        const tp = tritonPos(st.T);
        U.uTri.value.set(tp.x, tp.y, tp.z, TRITON.r);
      },
      hud(c) {
        clock(c);
        const p = c.params, f = st.f, rows = [];
        const rateWall = lapseRate(p) * (c.bhGlobals?.timeScale ?? hostSpeed);
        const ms = (f.era ? OPAL_MS : CA_MS) + (f.era ? st.T : f.day * DAY) * 1000;
        rows.push([t('Thời điểm', 'Date'), dateStr(ms) + (f.era ? ' (Hubble OPAL)' : ` · ${f.day >= 0 ? '+' : '−'}${vn(Math.abs(f.day), 1)} ${t('ngày so với Voyager 2', 'days from the Voyager 2 flyby')}`)]);
        rows.push([t('Tua nhanh', 'Time-lapse'), `×${fmtInt(rateWall)} · ${t('1 vòng quay', '1 rotation')} = ${rateWall > 0 ? vn(PROT / Math.max(rateWall, 1e-6), 1) + ' s' : '∞'}`]);
        const sc = st.subPoint(c.camera.position);
        const u = windU(sc.lat);
        rows.push([t('Vĩ độ dưới camera', 'Latitude below camera'), latStr(sc.lat)]);
        rows.push([t('Gió tại đó', 'Wind there'), `${u >= 0 ? '+' : '−'}${fmtInt(Math.abs(u))} m/s = ${fmtInt(Math.abs(u) * 3.6)} km/h ${u >= 0 ? t('về đông', 'eastward') : t('về tây', 'westward')} · Mach ${vn(Math.abs(u) / SOUND, 2)}`]);
        const m = f.main;
        if (!f.era) {
          rows.push([t('Vết Tối Lớn', 'Great Dark Spot'), t(`${latStr(m.lat)} · tỉ lệ trục ${vn(f.lam, 2)} · hoàn lưu 1 vòng/${vn(TAU / m.Om / DAY, 1)} ngày`, `${latStr(m.lat)} · axis ratio ${vn(f.lam, 2)} · one circulation every ${vn(TAU / m.Om / DAY, 1)} days`)]);
          const dl = wrapPi(f.sco.lon - m.lon), rel = f.sco.rate - m.rate;
          const tCatch = (dl < 0 ? -dl : TAU - dl) / rel;
          rows.push([t('Scooter so với Vết Tối Lớn', 'Scooter vs Great Dark Spot'), t(`${dl < 0 ? 'phía tây' : 'phía đông'} ${vn(Math.abs(dl) * R2D, 0)}° kinh độ · nhanh hơn ${vn(rel * DAY * R2D, 0)}°/ngày · ${dl < 0 ? 'vượt sau' : 'vượt lần tới sau'} ${vn(tCatch / HOUR, 0)} giờ`,
            `${vn(Math.abs(dl) * R2D, 0)}° of longitude ${dl < 0 ? 'west' : 'east'} · ${vn(rel * DAY * R2D, 0)}°/day faster · ${dl < 0 ? 'overtakes it in' : 'laps it again in'} ${vn(tCatch / HOUR, 0)} h`)]);
        } else {
          rows.push([t('Vết Tối Bắc NDS-2018', 'Northern dark spot NDS-2018'), t(`${latStr(m.lat)} · trôi ${fmtInt(Math.abs(windU(m.lat)) * 3.6)} km/h về tây`, `${latStr(m.lat)} · drifting westward at ${fmtInt(Math.abs(windU(m.lat)) * 3.6)} km/h`)]);
        }
        const cam = c.camera.position.clone().normalize();
        rows.push([t('Góc pha (Mặt Trời–hành tinh–bạn)', 'Phase angle (Sun–planet–you)'), vn(Math.acos(THREE.MathUtils.clamp(cam.dot(st.sun), -1, 1)) * R2D, 0) + '°']);
        if (p.triton !== false) {
          const tw = tritonPos(st.T).applyMatrix3(st.B2W), cp = c.camera.position;
          const ang = Math.acos(THREE.MathUtils.clamp(tw.clone().sub(cp).normalize().dot(c.controls.target.clone().sub(cp).normalize()), -1, 1)) * R2D;
          rows.push(['Triton', t(`cách camera ${fmtInt(tw.distanceTo(cp) * RN / 1000)} km · ${ang < 30 ? 'trong khung hình' : 'lệch ' + vn(ang, 0) + '° khỏi tâm nhìn'}`, `${fmtInt(tw.distanceTo(cp) * RN / 1000)} km from the camera · ${ang < 30 ? 'in frame' : vn(ang, 0) + '° off the line of sight'}`)]);
        }
        if (st.fly) {
          const h = (st.T - st.fly.Tca) / HOUR, o = hyper(h * HOUR);
          const hh = Math.floor(Math.abs(h)), mm = Math.floor((Math.abs(h) - hh) * 60);
          rows.push(['Voyager 2', t(`${h < 0 ? 'T−' : 'T+'}${hh}:${pad2(mm)} so với điểm gần nhất · cách tâm ${fmtInt(o.r / 1000)} km · ${vn(o.v / 1000, 1)} km/s`, `${h < 0 ? 'T−' : 'T+'}${hh}:${pad2(mm)} from closest approach · ${fmtInt(o.r / 1000)} km from the center · ${vn(o.v / 1000, 1)} km/s`)]);
        }
        return rows;
      },
      onClick(info, c) {
        clock(c);
        const h = hitBody(info.origin, info.dir, st.W2B);
        if (h.t === null) return;
        const pp = h.ro.clone().addScaledVector(h.rd, h.t);
        const ll = bodyLatLon(pp, st.spin), u = windU(ll.lat);
        const lapH = Math.abs(TAU / (u / (RN * Math.cos(ll.lat * D2R))) / HOUR);
        c.toast(t(`Vĩ độ <b>${latStr(ll.lat)}</b>: gió ${u >= 0 ? 'về đông' : 'về tây'} <b>${fmtInt(Math.abs(u) * 3.6)} km/h</b> (${fmtInt(Math.abs(u))} m/s, Mach ${vn(Math.abs(u) / SOUND, 2)}). So với lõi hành tinh, mây ở đây đi hết một vòng vĩ tuyến trong ${vn(lapH, 0)} giờ${Math.abs(ll.lat) < 15 ? ' — dòng tia xích đạo thổi <b>ngược</b> chiều quay' : ''}.`,
          `Latitude <b>${latStr(ll.lat)}</b>: wind blowing ${u >= 0 ? 'eastward' : 'westward'} at <b>${fmtInt(Math.abs(u) * 3.6)} km/h</b> (${fmtInt(Math.abs(u))} m/s, Mach ${vn(Math.abs(u) / SOUND, 2)}). Relative to the planet's interior, the clouds here take ${vn(lapH, 0)} h to go once around this latitude circle${Math.abs(ll.lat) < 15 ? ' — the equatorial jet blows <b>against</b> the rotation' : ''}.`), 7000);
        c.audio.ping(300 + Math.abs(u) * 0.8, 0.05, info.ndcX);
      },
      gdsShot(c, instant) {
        const p = c.params;
        stopFly(c);
        p.frame = 'gds'; p.lapse = 3.9;
        clock(c);
        const m = st.f.main;
        // put the vortex under a camera ~1.75 R_N away, lit from ~35° to the side
        const camDir = c.camera.position.clone().normalize();
        const az = Math.atan2(camDir.x, camDir.z) * R2D;
        p.sunAz = ((az + (st.f.era ? 12 : 35)) % 360 + 360) % 360;
        clock(c);
        const look = new V3().setFromSphericalCoords(1, (90 - m.lat * 0.7) * D2R, az * D2R);
        st.faceFeature(m.lat, m.lon, look);
        const qf = latlonJS(m.lat + (st.f.era ? -4 : -11), m.lon);
        const c0 = Math.cos(st.spin), s0 = Math.sin(st.spin);
        const pb = new V3(c0 * qf.x + s0 * qf.z, qf.y, -s0 * qf.x + c0 * qf.z);   // Ry(spin): body-fixed → body frame
        const target = pb.applyMatrix3(st.B2W).normalize().multiplyScalar(2.05);
        if (instant) { c.camera.position.copy(target); c.controls.target.set(0, 0, 0); c.camera.lookAt(0, 0, 0); }
        else tweenTo(c, target, 3);
        c.syncUI?.();
        c.toast?.(st.f.era ? t('Vết Tối Bắc NDS-2018 (Hubble): xoáy nghịch rộng ~11 000 km ở 25°B, có mây mê-tan sáng ở rìa. Ảnh Hubble chỉ phân giải ~1 000 km, nên các chi tiết nhỏ hơn là mô phỏng.', 'Northern dark spot NDS-2018 (Hubble): an anticyclone ~11,000 km across at 25° N, with bright methane clouds at its edge. Hubble resolves only ~1,000 km, so finer details are simulated.')
          : t('Camera bám theo Vết Tối Lớn, tua nhanh ~×50 000. Xoáy nghịch quay ngược chiều kim đồng hồ (khoảng 3 ngày một vòng theo mô hình Kida), phình–dẹt theo chu kỳ 8 ngày và trôi dần về xích đạo; mây “đồng hành” trắng lơ lửng ở rìa tây-nam. Scooter (42°N) và Vết Tối 2 (55°N) chạy nhanh hơn nên lần lượt vượt qua.', 'The camera follows the Great Dark Spot with time sped up ~×50,000. The anticyclone spins counterclockwise (about one turn every 3 days in the Kida model), swells and flattens on an 8-day cycle and drifts slowly toward the equator; white “companion” clouds hover at its southwest edge. The Scooter (42° S) and Dark Spot 2 (55° S) move faster and overtake it in turn.'), 10000);
      },
      flyby(c) {
        if (stopFly(c)) { c.toast?.(t('Đã dừng chuyến bay.', 'Flyby stopped.')); return; }
        startFly(c, -16);
        c.syncUI?.();
        c.toast?.(t('Voyager 2, 24–25/8/1989: tiếp cận từ phía có nắng với ~16 km/s, lướt qua cực Bắc ở độ cao 4 950 km trên đỉnh mây (26,8 km/s), rồi 5 giờ sau bay ngang Triton. Toàn bộ 23 giờ được tua trong ~25 giây — hành tinh kịp quay gần 1,5 vòng.', 'Voyager 2, 24–25 August 1989: approaching from the sunlit side at ~16 km/s, it skimmed over the north pole 4,950 km above the cloud tops (26.8 km/s), then passed Triton 5 hours later. The whole 23 hours are compressed into ~25 seconds — enough for the planet to turn almost 1.5 times.'), 10000);
      },
      backlit(c, instant) {
        stopFly(c);
        clock(c);
        const sgn = Math.sign(st.sun.dot(st.pole)) || 1;
        const perp = st.pole.clone().addScaledVector(st.sun, -st.pole.dot(st.sun)).normalize();
        const dir = st.sun.clone().negate().addScaledVector(perp, -sgn * 0.16).normalize();
        const pos = dir.multiplyScalar(7);
        if (instant) { c.camera.position.copy(pos); c.controls.target.set(0, 0, 0); c.camera.lookAt(0, 0, 0); } else tweenTo(c, pos, 3);
        c.params.ringGain = Math.max(c.params.ringGain, 150);
        c.toast?.(t('Nhìn ngược sáng (như Voyager 2 sau khi bay qua): bụi mịn tán xạ ánh sáng về phía trước nên vành Galle, Le Verrier, Lassell, Arago và Adams hiện ra; các cung Liberté, Égalité, Fraternité là những khúc vành dày hơn, bị vệ tinh Galatea giữ lại bằng cộng hưởng 42:43 và trôi cùng nhau mỗi 10,5 giờ một vòng.', 'Backlit view (as Voyager 2 saw it after the flyby): fine dust scatters light forward, revealing the Galle, Le Verrier, Lassell, Arago and Adams rings; the Liberté, Égalité and Fraternité arcs are denser clumps of the ring, confined by the moon Galatea through a 42:43 resonance and circling together once every 10.5 hours.'), 9000);
      },
    };
    NEPTUNE.__inst = inst;
    return inst;
  },
};

export default [NEPTUNE];
