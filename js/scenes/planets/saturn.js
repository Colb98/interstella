// Sao Thổ — Saturn from Cassini data, with moving clouds, rings, spokes and moons.
//
// One scene unit = Saturn's equatorial radius (60 268 km). The planet is an oblate ellipsoid (polar radius
// 54 364 km) tilted by its real obliquity; the Sun lies in the world xz-plane, so the season (the Sun's
// elevation B′ above the ring plane) is pure geometry: ring shadow on the clouds, planet shadow on the rings.
//
// DATA (assets/saturn/, see CREDITS.txt)
//   * saturn_map_4k.jpg — Cassini ISS RGB global map of 11 Aug 2011 (NASA PDS Atmospheres node, co_iss_global-maps),
//     resampled to planetographic latitude. Gaps filled: the ring-shadowed band 2–17°S is the mirrored 2–17°N band
//     (rolled 180°, zonal mean matched at both edges); the poles beyond 79°N / 82°S are zonal extensions.
//   * saturn_npole.jpg — north polar region from PIA17652 (Cassini, 10 Dec 2012, polar projection). The
//     750 nm channel gives the cloud brightness; the colour is the map's 76°N colour, tinted blue-grey where the
//     small-haze (blue-excess) signal marks the inside of the hexagon, as in Cassini true-colour images of 2012–13.
//   * saturn_rings.f16 — 16384 × (τ, albedo RGB) half floats, 73 000–137 500 km (3.9 km per texel):
//     τ = normal optical depth from the Cassini RSS X-band occultation of 3 May 2005 (Rev 7E, 1 km product),
//     binned in transmission; colour from the PIA11142 natural-colour radial mosaic (2008), inverted through the
//     same single-scattering model the shader uses (sun 4°, view 10°) and smoothed to 25 km.
//   * saturn_zonal.f16 — zonal-mean colour of the map (used for variance-preserving flow blending).
//
// MOTION
//   * Body frame rotates with the interior period 10 h 33 m 38 s (ring seismology, Mankovich+ 2019).
//     Winds are the Cassini/Voyager System III profile (≈ +450 m/s at the equator) re-referenced to that frame.
//   * Clouds: the map is advected by the zonal winds. Latitude is cut into 1.5° strips (two staggered sets,
//     triangular weights); each strip drifts rigidly at its centre speed for all time (so features really travel
//     at their jet's speed, and neighbouring strips shear past each other), while the shear inside a strip is
//     applied over a one-day phase that is reset (two phases half a period apart). Four map samples are blended
//     around the zonal mean with variance-preserving weights, so nothing scrolls rigidly, smears or fades.
//   * North pole: the hexagon pattern rotates with System III (10 h 39 m 22 s); inside it the cap image is
//     advected along hexagonal streamlines (hexagon jet ~120 m/s, polar cyclone ~150 m/s) with the same scheme.
//   * Great White Spot (2010–11): head at 37.7°N drifting −28 m/s (System III); the tail is the material it
//     released, carried by the local winds — its extent at each latitude is (u(φ) − u_head) × age, so it wraps
//     around the planet after ~55 days; the head dies after ~200 days. Plus the large anticyclone that formed.
//   * Rings: Keplerian Ω(r) = √(GM/r³). Small clumps/wakes drift and shear at the local Kepler rate (two-phase
//     re-seeded noise, drawn at the scale the pixel resolves — a visual stand-in for real ~100 m wakes), spokes are born near the dawn ansa as radial streaks over the B ring and then shear
//     because inner particles orbit faster, the F ring precesses, its clumps co-move and Prometheus carves
//     channels every 3.25°; three "propeller" moonlets ride in the outer A ring; density waves are in the data.
//   * Self-gravity wakes: azimuth-dependent effective optical depth of the A ring (wakes trail ~21°).
//
// PHOTOMETRY: Minnaert limb darkening on the clouds; rings with single scattering + multiple-scattering boost on
// the lit face, diffuse transmission on the unlit face, a backscattering (with opposition surge) phase function
// for the big icy particles and a strongly forward-scattering one for micron dust (D, F, G, E rings, spokes,
// Enceladus' plumes): backlit from inside Saturn's shadow the dusty rings glow while the B ring goes dark.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';
import { t, dec, int } from '../../i18n.js';

const V3 = THREE.Vector3;
const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAU = Math.PI * 2;
// 1 unit of ctx.time.sim = 60 s on Saturn (360× real time at the host's default 6× speed): one turn ≈ 106 s.
const SECS = 60;
const RKM = 60268, RPOL = 54364, FLAT = 1 - RPOL / RKM;
const GM = 3.7931207e16;                     // m³/s²
const P_INT = 38018;                         // 10 h 33 m 38 s
const P_S3 = 38362.4;                        // 10 h 39 m 22.4 s
const DOM3 = TAU / P_S3 - TAU / P_INT;       // System III relative to the interior (rad/s, < 0)
const PH_P = 86400;                          // weather phase period (s)
const RPH_P = 3 * 3600;                      // ring clump phase period (s)
const SPK_DT = 900, SPK_N = 12;              // a new spoke every 15 min, each lives ≤ SPK_N × SPK_DT (1–3 h)
const ASSET = f => new URL(`../../../assets/saturn/${f}`, import.meta.url).href;

// Zonal winds, System III, planetographic latitude (Voyager/Cassini cloud tracking: Sánchez-Lavega+ 2000,
// García-Melendo+ 2011; polar vortices from Dyudina+ 2009, Antuñano+ 2015), smoothed.
const SAT_WIND = [[-90, 0], [-89.4, 60], [-87.5, 150], [-84, 30], [-79, -15], [-75, 20], [-72, 60], [-68, -10], [-64, 40],
  [-60, 120], [-55, 50], [-50, -10], [-46, 60], [-42, 110], [-38, 10], [-35, -20], [-32, 40], [-28, 120], [-25, 200],
  [-20, 320], [-15, 410], [-10, 460], [-5, 470], [0, 455], [5, 470], [10, 460], [15, 410], [20, 320], [25, 210],
  [30, 110], [34, 40], [37.7, -25], [41, -10], [44, 60], [47, 120], [50, 60], [53, -10], [56, 20], [60, 100], [63, 50],
  [66, -15], [70, 10], [73, 60], [76, 100], [78, 120], [80, 50], [83, -10], [86, 60], [88, 150], [89.4, 60], [90, 0]];
// North-polar region inside the cap image, colatitude (°) → m/s relative to System III
const CAP_WIND = [[0, 0], [0.5, 90], [1.0, 140], [1.6, 150], [2.4, 120], [3.5, 70], [5, 30], [7, 10], [9, 15], [10.8, 60],
  [12.0, 110], [12.6, 120], [13.4, 80], [14.5, 20], [16, 0]];

const ff = v => (+v).toFixed(5);
function tab1(name, pts) {
  const n = pts.length;
  return `float ${name}(float x){
  const float X[${n}] = float[${n}](${pts.map(p => ff(p[0])).join(',')});
  const float Y[${n}] = float[${n}](${pts.map(p => ff(p[1])).join(',')});
  if (x <= X[0]) return Y[0];
  for (int i = 1; i < ${n}; i++){ if (x < X[i]) return mix(Y[i-1], Y[i], (x - X[i-1]) / (X[i] - X[i-1])); }
  return Y[${n - 1}];
}`;
}
function lerpTab(pts, x) {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) if (x < pts[i][0]) return pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * (x - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
  return pts[pts.length - 1][1];
}
const vn = (x, d = 1) => dec(x, d);
const latStr = d => `${vn(Math.abs(d))}° ${d >= 0 ? t('B', 'N') : t('N', 'S')}`;
const wrap = a => ((a % TAU) + TAU) % TAU;
const fmtInt = n => int(n);
const E2 = 1 - (RPOL / RKM) ** 2;
// distance from the spin axis (m) at planetographic latitude (deg)
const rhoAxis = latD => { const p = latD * D2R; return RKM * 1e3 * Math.cos(p) / Math.sqrt(1 - E2 * Math.sin(p) ** 2); };
const windS3 = latD => lerpTab(SAT_WIND, latD);
// angular drift rate of the clouds relative to the interior frame (rad/s)
const omegaInt = latD => windS3(latD) / Math.max(rhoAxis(latD), 2e5) + DOM3;
const kepler = rU => Math.sqrt(GM / (rU * RKM * 1e3) ** 3);   // rad/s

// moons: [name, a (units), radius (units), period (s), phase0 (rad), albedo rgb, kind]
const MOONS = [
  ['Mimas', 185539 / RKM, 198.2 / RKM, 81425, 0.4, [0.62, 0.62, 0.60], 1],
  ['Enceladus', 237948 / RKM, 252.1 / RKM, 118387, 2.2, [0.95, 0.96, 0.98], 2],
  ['Tethys', 294619 / RKM, 531.1 / RKM, 163106, 4.0, [0.80, 0.80, 0.78], 3],
  ['Dione', 377396 / RKM, 561.4 / RKM, 236469, 5.4, [0.66, 0.65, 0.63], 3],
  ['Rhea', 527108 / RKM, 763.8 / RKM, 390374, 1.1, [0.62, 0.61, 0.58], 3],
  ['Titan', 1221870 / RKM, 2574.7 / RKM, 1377648, 3.3, [0.42, 0.28, 0.13], 4],
  ['Prometheus', 139380 / RKM, 43.1 / RKM, 52962, 0.0, [0.55, 0.55, 0.53], 0],
  ['Pandora', 141720 / RKM, 40.7 / RKM, 54329, 2.9, [0.55, 0.55, 0.53], 0],
];
const N_MOON = MOONS.length;
// giant propellers in the outer A ring: [name, radius km, phase0]
const PROPS = [['Earhart', 133800, 0.7], ['Blériot', 134912, 2.5], ['Santos-Dumont', 135520, 4.6]];
// Great White Spot 2010–11
const GWS_LAT = 37.7, GWS_U = -27.9;
const GWS_OM = GWS_U / rhoAxis(GWS_LAT) + DOM3;
const AV_LAT = 41.5, AV_U = -15;
const AV_OM = AV_U / rhoAxis(AV_LAT) + DOM3;

/* ------------------------------------------------------------------------------------------------ */
/* shader                                                                                             */
/* ------------------------------------------------------------------------------------------------ */
const ringPer = r => (TAU / kepler(r));
const FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir, uEarth;
uniform mat3 uW2B;
uniform float uSpin, uSunI, uTexOK, uAxis, uRingOn, uDust, uMoonsOn, uAlbK;
uniform sampler2D uMap, uZonal, uCap, uRingTex;
uniform vec3 uPhW;     // weather phases: τ0, τ1 (s since reset), weight of phase 0
uniform vec2 uPhT;     // weather-clock start times of the two current phases (s)
uniform vec2 uSeedW;
uniform float uHex;    // angle of the System III pattern relative to the interior frame (rad)
uniform vec3 uPhR;     // ring clump phases: τ0, τ1 (s), weight of phase 0
uniform vec2 uSeedR;
uniform vec4 uSpk;     // spokes: amount, dawn azimuth, slot clock (tp / DT), unused
uniform vec4 uGws;     // Great White Spot: age (days, < 0 = none), head lon, anticyclone lon, unused
uniform vec4 uMoon[${N_MOON}];
uniform vec4 uFR;      // F ring: comoving angle, periapse lon, Prometheus lon, Pandora lon
uniform vec3 uProp;
#define R2D 57.2957795
#define D2R 0.0174532925
#define RKM ${RKM}.0
#define FLAT ${FLAT.toFixed(6)}
#define GMU ${(GM / (RKM * 1e3) ** 3).toExponential(6)}
#define RT0 ${(73000 / RKM).toFixed(6)}
#define RT1 ${(137500 / RKM).toFixed(6)}
#define RTEX 16384.0
#define STRIP 0.0261799
#define CSTRIP 0.8
#define CAP_PX ${(34.9 / 1024).toFixed(6)}
#define CAP_C vec2(${(503 / 1024).toFixed(6)}, ${(1 - 509 / 1024).toFixed(6)})
#define HEX_V 0.951
#define GWS_OM ${GWS_OM.toExponential(6)}
#define SPK_DT ${SPK_DT}.0
#define CH_D ${(TAU * (1 - ringPer(139380 / RKM) / ringPer(140221 / RKM))).toFixed(6)}
#define P_PROM ${ringPer(139380 / RKM).toFixed(1)}
#define OM_F ${kepler(140221 / RKM).toExponential(6)}
${tab1('capWind', CAP_WIND)}
float gMu, gMu0;

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
float omK(float r){ return sqrt(GMU / (r * r * r)); }
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * c, 1e-5), 1.5); }
float bodyShadow(vec3 p, vec3 s, vec3 E){
  vec3 o = p * E, d = s * E;
  float tc = -dot(o, d) / dot(d, d);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.996, 1.004, length(o + d * tc));
}
float h11(float i){ return hash13(vec3(i, 7.31, 1.97)); }
float vn1(float x){ float i = floor(x), f = fract(x); return mix(h11(i), h11(i + 1.0), f * f * (3.0 - 2.0 * f)); }
// narrow feature of width w (km) at x0, anti-aliased to footprint fw (km), preserving equivalent width
float kbox(float x, float x0, float w, float fw){ float we = max(w, fw); return (w / we) * (1.0 - smoothstep(0.3 * we, 0.7 * we, abs(x - x0))); }

/* ---------------- clouds ---------------- */
vec4 zonal(float lat){ return textureLod(uZonal, vec2(0.5, 0.5 - lat / PI), 0.0); }   // rgb albedo, a = ω (1e-5 rad/s)

// Cassini map advected by the zonal jets (strips × phases, see the file header)
vec3 mapFlow(float lon, float lat, vec2 gx, vec2 gy){
  float v = 0.5 + lat / PI;
  vec4 Z = zonal(lat);
  float om = Z.a * 1e-5;
  float u = lat / STRIP;
  float wA = 1.0 - 2.0 * abs(fract(u) - 0.5);
  float oA = zonal((floor(u) + 0.5) * STRIP).a * 1e-5, oB = zonal(floor(u + 0.5) * STRIP).a * 1e-5;
  vec3 acc = vec3(0.0); float ww = 0.0;
  for (int p = 0; p < 2; p++){
    float tau = p == 0 ? uPhW.x : uPhW.y, ts = p == 0 ? uPhT.x : uPhT.y;
    float wp = p == 0 ? uPhW.z : 1.0 - uPhW.z;
    for (int g = 0; g < 2; g++){
      float w = wp * (g == 0 ? wA : 1.0 - wA);
      if (w < 0.003) continue;
      float d = (g == 0 ? oA : oB) * ts + om * tau;
      vec3 s = textureGrad(uMap, vec2(fract((lon - d) / TAU), v), gx, gy).rgb;
      acc += w * (s - Z.rgb); ww += w * w;
    }
  }
  return Z.rgb + acc * inversesqrt(max(ww, 1e-4));
}
// north-polar cap: image angle a, streamline colatitude s (deg) → texture uv
float hexF(float a){ return 0.866025 / cos(mod(a - HEX_V, PI / 3.0) - PI / 6.0); }
float hexG(float a, float s){ return mix(1.0, hexF(a) / 0.9085, exp(-pow((s - 12.6) / 4.5, 2.0))); }
vec2 capUV(float a, float s){ return CAP_C + s * hexG(a, s) * CAP_PX * vec2(cos(a), sin(a)); }
float capOm(float s){ return capWind(s) / (max(s, 0.25) * 1150.0e3); }
vec3 capFlow(float lon, float colat, float lod){
  float a = lon - uHex;
  float s = colat / hexG(a, colat);
  float om = capOm(s);
  float u = s / CSTRIP;
  float wA = 1.0 - 2.0 * abs(fract(u) - 0.5);
  float oA = capOm((floor(u) + 0.5) * CSTRIP), oB = capOm(max(floor(u + 0.5), 0.3) * CSTRIP);
  vec3 acc = vec3(0.0); float ww = 0.0; vec3 mean = vec3(0.0);
  for (int p = 0; p < 2; p++){
    float tau = p == 0 ? uPhW.x : uPhW.y, ts = p == 0 ? uPhT.x : uPhT.y;
    float wp = p == 0 ? uPhW.z : 1.0 - uPhW.z;
    for (int g = 0; g < 2; g++){
      float w = wp * (g == 0 ? wA : 1.0 - wA);
      if (w < 0.003) continue;
      float d = mod((g == 0 ? oA : oB) * ts + om * tau, TAU);
      vec3 c = textureLod(uCap, capUV(a - d, s), lod).rgb;
      acc += w * c; ww += w * w; mean += w * textureLod(uCap, capUV(a - d, s), 6.0).rgb;
    }
  }
  return mean + (acc - mean) * inversesqrt(max(ww, 1e-4));
}
// south-polar cyclone (no data beyond 83°S): dark eye, bright eyewall, spiral bands
vec3 southPole(vec3 alb, float lon, float latD){
  float rho = 90.0 + latD;
  if (rho > 9.0) return alb;
  float sw = 0.0;
  for (int p = 0; p < 2; p++){
    float tau = p == 0 ? uPhW.x : uPhW.y, wp = p == 0 ? uPhW.z : 1.0 - uPhW.z;
    float ang = -lon - 150.0 / (max(rho, 0.4) * 1150.0e3) * tau * min(rho, 2.0) / 2.0 - 1.6 * log(max(rho, 0.2));
    vec2 q = vec2(cos(ang), sin(ang)) * rho;
    sw += wp * fbm(vec3(q * 0.9, (p == 0 ? uSeedW.x : uSeedW.y)), 3);
  }
  float eye = smoothstep(1.0, 1.7, rho), wall = exp(-pow((rho - 2.0) / 0.5, 2.0));
  vec3 c = alb * (0.62 + 0.38 * eye) * (1.0 + 0.10 * wall + 0.07 * sw * smoothstep(9.0, 3.0, rho));
  return mix(alb, c, smoothstep(9.0, 6.0, rho));
}
// Great White Spot 2010–11: head, released material carried by the jets, the anticyclone
vec3 gwsLayer(vec3 alb, float lon, float lat, float om){
  float age = uGws.x;
  float latD = lat * R2D;
  if (age < 0.0 || latD < 22.0 || latD > 52.0) return alb;
  float ageS = age * 86400.0;
  float rel = om - GWS_OM;                              // drift relative to the head (rad/s)
  float dl = rel >= 0.0 ? mod(lon - uGws.y, TAU) : mod(uGws.y - lon, TAU);
  float arel = max(abs(rel), 3e-8);
  float I = 0.0;
  for (int k = 0; k < 4; k++){
    float tr = (dl + float(k) * TAU) / arel;            // seconds since this material left the head
    if (tr > ageS) break;
    float trd = tr / 86400.0;
    float src = 1.0 - smoothstep(195.0, 210.0, age - trd);           // head still active at release
    float sig = min(1.3 + 3.2 * sqrt(trd / 30.0), 6.0);
    float latc = GWS_LAT_D - 2.5 * min(trd / 30.0, 1.0);
    I += exp(-pow((latD - latc) / sig, 2.0)) * src * exp(-trd / 140.0) * (0.55 + 0.45 * exp(-trd / 8.0));
  }
  I *= smoothstep(0.0, 1.5, age) * (1.0 - smoothstep(230.0, 300.0, age));
  // head: compact convective cloud growing to ~10 000 km
  float cphi = cos(lat);
  float dlon = mod(lon - uGws.y + PI, TAU) - PI;
  float hx = dlon * cphi * RKM, hy = (latD - GWS_LAT_D) * 1050.0;
  float hs = mix(2200.0, 5500.0, smoothstep(0.0, 25.0, age));
  float head = exp(-(hx * hx / (hs * hs * 2.2) + hy * hy / (hs * hs * 0.7))) * smoothstep(0.0, 0.6, age) * (1.0 - smoothstep(195.0, 210.0, age));
  // turbulent texture carried by the local wind (two re-seeded phases)
  float n = 0.0;
  for (int p = 0; p < 2; p++){
    float tau = p == 0 ? uPhW.x : uPhW.y, wp = p == 0 ? uPhW.z : 1.0 - uPhW.z;
    float a = lon - om * tau;
    // zonally stretched eddies (~3 000 × 1 000 km), like the streaky tail in Cassini images
    vec3 q = vec3(cos(a) * cphi * 22.0, sin(lat) * 66.0, sin(a) * cphi * 22.0) + (p == 0 ? uSeedW.x : uSeedW.y) * 3.1;
    n += wp * fbm(q, 4);
  }
  float cover = clamp(I * (0.55 + 2.4 * n) + head * (1.25 + 1.2 * n), 0.0, 1.0);
  vec3 ice = vec3(0.98, 0.95, 0.88);
  alb = mix(alb, ice, cover * 0.85);
  alb *= 1.0 - 0.18 * clamp(I * 1.5, 0.0, 1.0) * smoothstep(0.1, 0.5, -n);     // dark lanes between cloud streaks
  // the large anticyclone that formed in the storm's wake (~12 000 km, reddish core, bright collar)
  float av = smoothstep(10.0, 20.0, age);
  if (av > 0.0){
    float dla = mod(lon - uGws.z + PI, TAU) - PI;
    float ax = dla * cos(AV_LAT_D * D2R) * RKM / 6000.0, ay = (latD - AV_LAT_D) * 1050.0 / 3800.0;
    float rr = length(vec2(ax, ay));
    float core = smoothstep(1.0, 0.55, rr), collar = exp(-pow((rr - 1.0) / 0.25, 2.0));
    alb = mix(alb, alb * vec3(0.86, 0.76, 0.62), core * av * 0.8);
    alb = mix(alb, ice, collar * av * 0.45);
  }
  return alb;
}
vec3 surface(vec3 q, float lat, vec2 gx, vec2 gy, float fwKm){
  float lon = atan(-q.z, q.x);
  if (lon < 0.0) lon += TAU;
  float latD = lat * R2D;
  vec4 Z = zonal(lat);
  vec3 alb = uTexOK > 0.5 ? (latD < 77.0 ? mapFlow(lon, lat, gx, gy) : Z.rgb) : Z.rgb;
  if (latD > 75.8 && uTexOK > 0.5){
    float lod = log2(max(fwKm / 1150.0 * 34.9, 1.0));
    alb = mix(alb, capFlow(lon, 90.0 - latD, lod), smoothstep(75.8, 76.8, latD));
  }
  if (latD < -78.0) alb = southPole(alb, lon, latD);
  // sub-texel weather for close-ups (the map has ~105 km pixels): small eddies carried by the jets
  if (fwKm < 160.0){
    float om = Z.a * 1e-5, n = 0.0;
    for (int p = 0; p < 2; p++){
      float tau = p == 0 ? uPhW.x : uPhW.y, wp = p == 0 ? uPhW.z : 1.0 - uPhW.z;
      float a = lon - om * tau;
      vec3 qq = vec3(cos(a) * cos(lat), sin(lat) * 1.6, sin(a) * cos(lat)) * 90.0 + (p == 0 ? uSeedW.x : uSeedW.y) * 5.7;
      n += wp * fbm(qq, 3);
    }
    alb *= 1.0 + 0.045 * n * smoothstep(160.0, 50.0, fwKm);
  }
  alb = gwsLayer(alb, lon, lat, Z.a * 1e-5);
  return alb;
}

/* ---------------- rings ---------------- */
float dustFrac(float r){
  float rk = r * RKM;
  if (rk < 74490.0) return 1.0;
  if (rk < 92000.0) return 0.12;
  if (rk < 117580.0) return 0.015;
  if (rk < 122050.0) return 0.22;
  return mix(0.02, 0.07, smoothstep(128000.0, 136700.0, rk));
}
// spokes: dusty radial streaks over the B ring, born near the dawn ansa, sheared by Keplerian motion
float spokes(float r, float th){
  if (uSpk.x <= 0.0 || r < 1.62 || r > 1.96) return 0.0;
  float acc = 0.0;
  float slot = floor(uSpk.z);
  for (int k = 0; k < ${SPK_N}; k++){
    float n = slot - float(k);
    vec3 h = hash33(vec3(n, 3.7, 11.3)), h2 = hash33(vec3(n, 9.1, 4.4));
    float age = (uSpk.z - n - 0.8 * h.x) * SPK_DT;          // s
    if (age < 0.0) continue;
    float life = SPK_DT * (4.0 + 8.0 * h2.z);
    if (age > life) continue;
    float th0 = uSpk.y + (h.y - 0.5) * 1.3;
    float r0 = 1.64 + 0.14 * h.z, r1 = min(r0 + 0.12 + 0.14 * h2.x, 1.95);
    float rw = smoothstep(r0 - 0.01, r0 + 0.015, r) * (1.0 - smoothstep(r1 - 0.02, r1 + 0.01, r));
    if (rw <= 0.0) continue;
    // birth: co-rotating with the magnetic field (≈ synchronous at 1.85 R); afterwards each radius orbits at Ω_K(r)
    float ths = th0 + omK(r) * age;
    float d = abs(mod(th - ths + PI, TAU) - PI) * r;       // arc distance (units)
    float w = 0.010 + 0.022 * h2.y + age * 1.5e-6;
    float env = smoothstep(0.0, 900.0, age) * (1.0 - smoothstep(life * 0.5, life, age));
    acc += rw * env * exp(-d * d / (w * w)) * (0.6 + 0.8 * h2.x);
  }
  return acc * uSpk.x;
}
// main rings from the data texture, plus azimuthal structure. tau: normal optical depth; alb: particle albedo×phase
void ringSample(float r, float th, float fw, vec3 rdB, out float tau, out vec3 alb, out float dust){
  tau = 0.0; alb = vec3(0.5); dust = 0.0;
  float rk = r * RKM, fwk = max(fw * RKM, 0.05);
  if (r < RT0 + 0.02){
    // D ring: faint dust sheet and ringlets D68, D72, D73
    tau = (2e-4 * smoothstep(1.11, 1.15, r) + 1.5e-3 * kbox(rk, 67630.0, 60.0, fwk) + 6e-4 * kbox(rk, 71700.0, 200.0, fwk)
         + 3e-4 * kbox(rk, 73200.0, 300.0, fwk)) * uDust;
    alb = vec3(0.70, 0.62, 0.55); dust = 1.0;
  }
  if (r > RT0 && r < RT1 && uTexOK > 0.5){
    float lod = log2(max(fw / ((RT1 - RT0) / RTEX), 1.0));
    vec4 t = textureLod(uRingTex, vec2((r - RT0) / (RT1 - RT0), 0.5), lod);
    float tt = t.r;
    alb = max(t.gba, vec3(0.01));
    dust = max(dust, dustFrac(r));
    tau = max(tau, tt);
    // azimuthal fine structure: clumps / self-gravity wakes drifting and shearing at the local Kepler rate
    float amp = 1.0 - smoothstep(25.0, 120.0, fwk);
    if (amp > 0.0 && tau > 0.02){
      float om = omK(r), n = 0.0;
      // unresolved clumps / wake trains: drawn at the scale the pixel resolves (≥ 6 km radially, 12× longer in
      // azimuth; two octaves cross-faded so zooming never pops) — a visual stand-in for structure below ~100 m
      float lk = log2(max(fwk * 4.0, 6.0) / 6.0), fo = fract(lk);
      float L1 = 6.0 * exp2(floor(lk));
      for (int p = 0; p < 2; p++){
        float tauP = p == 0 ? uPhR.x : uPhR.y, wp = p == 0 ? uPhR.z : 1.0 - uPhR.z;
        float a = th - om * tauP + (p == 0 ? uSeedR.x : uSeedR.y);
        vec2 cs = vec2(cos(a), sin(a)) * rk;
        vec3 c1 = vec3(cs / (12.0 * L1), rk / L1), c2 = vec3(cs / (24.0 * L1), rk / (2.0 * L1)) + 7.7;
        n += wp * mix(snoise(c1), snoise(c2), fo);
      }
      n *= inversesqrt(uPhR.z * uPhR.z + (1.0 - uPhR.z) * (1.0 - uPhR.z));
      float aZone = rk > 122050.0 ? 0.35 : rk > 92000.0 ? 0.14 : 0.12;
      tau *= max(1.0 + aZone * amp * n, 0.05);
      alb *= 1.0 + 0.8 * aZone * amp * n;      // opaque B-ring clumps show up in brightness, not transparency
    }
    // A-ring self-gravity wakes: more transparent when looking along the wakes (trailing ~21° from azimuthal)
    if (rk > 122050.0 || (rk > 92000.0 && rk < 100000.0)){
      vec2 er = vec2(sin(th), cos(th)), et = vec2(cos(th), -sin(th));
      vec2 v2 = rdB.xz; float lv = length(v2);
      if (lv > 1e-4){
        v2 /= lv;
        float psi = atan(dot(v2, et), dot(v2, er));
        float A = rk > 122050.0 ? 0.28 : 0.12;
        tau *= 1.0 - A * cos(2.0 * (psi - 1.204)) * smoothstep(0.02, 0.5, lv);
      }
    }
    // giant propellers (partial gaps around km-sized moonlets): bright S-shaped lobes, only in close-ups
    if (rk > 133600.0 && rk < 135700.0 && fwk < 25.0){
      for (int i = 0; i < 3; i++){
        float rp = i == 0 ? ${PROPS[0][1]}.0 : i == 1 ? ${PROPS[1][1]}.0 : ${PROPS[2][1]}.0;
        float dr = rk - rp;
        if (abs(dr) > 40.0) continue;
        float ds = (mod(th - uProp[i] + PI, TAU) - PI) * rk;
        float sc = i == 1 ? 1.6 : 1.0;
        float lob = exp(-pow((dr - 7.0 * sc) / (4.0 * sc), 2.0) - pow((ds + 260.0 * sc) / (300.0 * sc), 2.0))
                  + exp(-pow((dr + 7.0 * sc) / (4.0 * sc), 2.0) - pow((ds - 260.0 * sc) / (300.0 * sc), 2.0));
        float gap = exp(-pow((dr - 3.0 * sc) / (2.5 * sc), 2.0) - pow((ds + 120.0 * sc) / (160.0 * sc), 2.0))
                  + exp(-pow((dr + 3.0 * sc) / (2.5 * sc), 2.0) - pow((ds - 120.0 * sc) / (160.0 * sc), 2.0));
        float vis = 1.0 - smoothstep(4.0, 25.0, fwk);
        tau *= 1.0 + vis * (0.8 * lob - 0.7 * gap);
        alb *= 1.0 + vis * 0.35 * lob;
      }
    }
  }
  // F ring: eccentric core, braided strands, co-moving clumps, Prometheus channels
  float aF = 140221.0 / RKM;
  float rc = aF * (1.0 - 0.00235 * cos(th - uFR.y));
  float dr = (r - rc) * RKM;
  if (abs(dr) < 1200.0 + fwk){
    float thc = th - uFR.x;
    float bl = 1.0 - smoothstep(0.3, 0.8, fw * 70.0 / aF);
    float cl = 0.45 + 1.1 * mix(0.5, vn1(thc * 70.0 + 11.0), bl) * vn1(thc * 9.0 + 3.0);
    float s1 = 110.0 * sin(thc * 3.0 + 1.3) + 45.0 * sin(thc * 17.0);
    float s2 = -140.0 * sin(thc * 2.0 + 0.4) + 40.0 * sin(thc * 13.0 + 2.0);
    // channels: one per Prometheus orbit, spaced CH_D behind it, tilted by Keplerian shear as they age
    float dth = mod(th - uFR.z, TAU) - TAU;
    float k = floor(-dth / CH_D + 0.5);
    float age = k * P_PROM;
    float shear = 1.5 * OM_F * (dr / (aF * RKM)) * age;
    float k2 = floor((-dth - shear) / CH_D + 0.5);
    float dch = (dth + k2 * CH_D + shear) * aF * RKM;
    float chan = (1.0 - 0.85 * exp(-pow(dch / 260.0, 2.0)) * exp(-k2 / 14.0) * step(0.5, k2)) * (1.0 + 0.6 * exp(-pow((dch - 450.0) / 200.0, 2.0)) * exp(-k2 / 14.0));
    float fT = (0.5 * kbox(dr, 0.0, 12.0, fwk) + 0.06 * (kbox(dr, s1, 45.0, fwk) + kbox(dr, s2, 45.0, fwk))) * cl * chan
             + 0.012 * exp(-pow(dr / 380.0, 2.0)) * (0.5 + cl) * chan * uDust;
    if (fT > tau){ dust = 0.85; alb = vec3(0.80, 0.74, 0.70); }
    tau = max(tau, fT);
  }
  // G ring and E ring (diffuse dust; E ring peaks at Enceladus' orbit and is blue)
  if (r > 2.6){
    float g = 1.2e-6 * exp(-pow((r - 2.787) / 0.03, 2.0)) + 3e-7 * exp(-pow((r - 2.82) / 0.09, 2.0));
    float e = 1.5e-5 * (r < 3.95 ? exp(-pow((r - 3.95) / 0.45, 2.0)) : exp(-pow((r - 3.95) / 1.7, 2.0))) * smoothstep(2.9, 3.2, r);
    tau = (g + e) * uDust;
    alb = mix(vec3(0.85, 0.80, 0.75), vec3(0.55, 0.75, 1.0), e / max(g + e, 1e-12));
    dust = 1.0;
  }
  tau = max(tau, 0.0);
}
float ringTauShadow(float r, float fw){
  if (r < RT0 || r > RT1 || uTexOK < 0.5) return 0.0;
  return textureLod(uRingTex, vec2((r - RT0) / (RT1 - RT0), 0.5), log2(max(fw / ((RT1 - RT0) / RTEX), 1.0))).r;
}

/* ---------------- moons ---------------- */
vec2 moonHit(vec3 ro, vec3 rd, out int id){
  id = -1; float tb = 1e9, tex = 0.0;
  if (uMoonsOn < 0.5) return vec2(1e9, 0.0);
  for (int i = 0; i < ${N_MOON}; i++){
    vec2 h = sphereHit(ro, rd, uMoon[i].xyz, uMoon[i].w);
    if (h.x < h.y && h.x > 0.0 && h.x < tb){ tb = h.x; id = i; }
  }
  return vec2(tb, 0.0);
}
vec3 moonAlb(int i){
  ${MOONS.map((m, i) => `if (i == ${i}) return vec3(${m[5].map(v => v.toFixed(3)).join(',')});`).join('\n  ')}
  return vec3(0.5);
}
vec3 moonShade(int i, vec3 p, vec3 rd, vec3 s, vec3 E){
  vec3 c = uMoon[i].xyz; float R = uMoon[i].w;
  vec3 n = normalize(p - c);
  vec3 toS = -normalize(c), vel = normalize(vec3(c.z, 0.0, -c.x));   // sub-Saturn and leading directions (tidally locked)
  vec3 alb = moonAlb(i);
  vec3 lp = vec3(dot(n, toS), n.y, dot(n, vel));
  float nz = fbm(lp * 6.0 + float(i) * 7.0, 3);
  if (i == 0){ // Mimas: Herschel crater on the leading hemisphere
    float d = acos(clamp(dot(n, vel), -1.0, 1.0));
    alb *= 0.9 + 0.1 * smoothstep(0.30, 0.36, d) + 0.25 * exp(-pow((d - 0.36) / 0.03, 2.0)) - 0.12 * exp(-pow(d / 0.08, 2.0));
  }
  if (i == 1){ // Enceladus: fresh ice; tiger stripes near the south pole
    float tg = (1.0 - smoothstep(-0.97, -0.90, n.y)) * smoothstep(0.75, 0.95, abs(sin((lp.x + lp.z) * 14.0)));
    alb *= 1.0 - 0.3 * tg;
  }
  if (i == 3 || i == 4){ // Dione, Rhea: darker trailing hemisphere with bright wispy fractures
    float tr = smoothstep(0.2, -0.6, dot(n, vel));
    alb *= mix(1.0, 0.7 + 0.35 * smoothstep(0.25, 0.6, nz), tr);
  }
  if (i != 5) alb *= 0.9 + 0.12 * nz;
  float mu0 = dot(n, s), mu = max(dot(n, -rd), 0.0);
  float sh = bodyShadow(p, s, E);
  vec3 col;
  if (i == 5){ // Titan: thick orange haze, soft terminator, bluish detached haze at the limb
    float lit = smoothstep(-0.15, 0.35, mu0) * (0.6 + 0.4 * max(mu0, 0.0));
    col = alb * lit + vec3(0.06, 0.08, 0.12) * pow(1.0 - mu, 3.0) * smoothstep(-0.3, 0.2, mu0);
  } else {     // icy surfaces: Lommel–Seeliger / Lambert mix
    col = alb * max(mu0, 0.0) * (0.55 / max(mu0 + mu, 0.05) + 0.45);
  }
  return col * uSunI * sh;
}
// Enceladus' south-polar plumes: forward-scattering ice grains, seen when backlit
vec3 plumes(vec3 ro, vec3 rd, vec3 s, float tMax){
  if (uMoonsOn < 0.5) return vec3(0.0);
  vec3 c = uMoon[1].xyz; float R = uMoon[1].w;
  vec2 hb = sphereHit(ro, rd, c - vec3(0.0, 4.0 * R, 0.0), 5.0 * R);
  if (!(hb.x < hb.y && hb.y > 0.0)) return vec3(0.0);
  vec3 base = c - vec3(0.0, R, 0.0);
  // closest approach between the ray and the (downward) plume axis
  vec3 ax = vec3(0.0, -1.0, 0.0);
  vec3 w0 = ro - base; float b = dot(rd, ax), d = dot(rd, w0), e = dot(ax, w0);
  float den = 1.0 - b * b; if (den < 1e-5) return vec3(0.0);
  float sc = (b * e - d) / den, tc = (e - b * d) / den;
  if (sc < 0.0 || sc > tMax || tc < 0.0) return vec3(0.0);
  float dist = length(w0 + rd * sc - ax * tc);
  float wdt = 0.35 * R + 0.45 * tc;
  // column optical depth across the plume ~1e-3 near the vents (×uDust)
  float col = 2.5e-3 * exp(-tc / (1.6 * R)) * exp(-dist * dist / (wdt * wdt)) * (0.6 + 0.4 * snoise(vec3(dot(w0 + rd * sc - ax * tc, vec3(1.0, 0.0, 0.3)) / (0.25 * wdt), tc / R, 1.0)));
  float ph = 0.5 * hg(dot(rd, s), 0.95) + 0.5 * hg(dot(rd, s), 0.6);
  return vec3(0.85, 0.92, 1.0) * col * ph * uSunI * uDust * bodyShadow(base, s, vec3(1.0, 1.0 / (1.0 - FLAT), 1.0));
}

// moons smaller than a pixel: draw their integrated light as a point (so they show as dots, like in real images)
vec3 moonPoints(vec3 ro, vec3 rd, vec3 s, float pixAng){
  vec3 acc = vec3(0.0);
  if (uMoonsOn < 0.5) return acc;
  float sg = 0.8 * pixAng;
  for (int i = 0; i < ${N_MOON}; i++){
    vec3 d = uMoon[i].xyz - ro; float L = length(d);
    float ar = uMoon[i].w / L;
    float c = dot(rd, d / L);
    if (c < 0.99) continue;
    float ang = sqrt(max(2.0 * (1.0 - c), 0.0));
    float ph = 0.5 + 0.5 * dot(-d / L, s);                     // fraction of the disc lit, as seen
    float flux = ar * ar * PI * ph * (1.0 - smoothstep(0.6, 1.6, ar / pixAng));
    acc += moonAlb(i) * uSunI * 0.7 * flux / (PI * sg * sg) * exp(-ang * ang / (sg * sg)) * bodyShadow(uMoon[i].xyz, s, vec3(1.0, 1.0 / (1.0 - FLAT), 1.0));
  }
  return acc;
}
/* ---------------- main ---------------- */
vec3 rimLight(vec3 pn, vec3 rd, vec3 s){
  float il = smoothstep(-0.3, 0.2, dot(pn, s));
  float c = dot(rd, s);
  return vec3(0.13, 0.115, 0.09) * il * (0.35 + 0.45 * hg(c, 0.7) + 0.2 * hg(c, 0.93));
}
vec3 sunDisc(vec3 rdW){
  float c = dot(rdW, uSunDir);
  // the Sun from 9.6 AU: 3.3′ across
  return bb(5772.0) * (smoothstep(0.9999988, 0.9999994, c) * 60.0 + pow(max(c, 0.0), 30000.0) * 3.0 + pow(max(c, 0.0), 600.0) * 0.03);
}
void main(){
  vec3 rdW = camRay(), roW = uCamPos;
  vec3 ro = uW2B * roW, rd = uW2B * rdW, s = uW2B * uSunDir;
  vec3 E = vec3(1.0, 1.0 / (1.0 - FLAT), 1.0);
  vec3 roE = ro * E, rdE = rd * E;
  float A = dot(rdE, rdE), B = dot(roE, rdE), C = dot(roE, roE) - 1.0, H = B * B - A * C;
  float tP = 1e9;
  if (H > 0.0){ float t0 = (-B - sqrt(H)) / A; if (t0 > 0.0) tP = t0; }
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  mat3 Rs = rotY(-uSpin);
  // seam-free texture derivatives: body-fixed direction of the hit (or of the closest approach for misses)
  float tS = tP < 1e8 ? tP : max(-B / A, 0.0);
  vec3 pS = normalize(Rs * ((ro + rd * tS) * E));
  vec3 dpx = dFdx(pS), dpy = dFdy(pS);
  float qq = max(pS.x * pS.x + pS.z * pS.z, 1e-6);
  vec2 gx = vec2((pS.z * dpx.x - pS.x * dpx.z) / qq / TAU, dpx.y / (PI * sqrt(qq)));
  vec2 gy = vec2((pS.z * dpy.x - pS.x * dpy.z) / qq / TAU, dpy.y / (PI * sqrt(qq)));
  int mid; vec2 mh = moonHit(ro, rd, mid);
  float tM = mh.x, tN = min(tP, tM);
  vec3 col;
  if (tP < 1e8 && tP <= tM){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = Rs * (p * E);
    float lat = asin(clamp(n.y, -1.0, 1.0));
    gMu0 = dot(n, s); gMu = max(dot(n, -rd), 0.0);
    float fwKm = tP * pixAng * RKM / max(gMu, 0.2);
    vec3 alb = surface(q, lat, gx, gy, fwKm);
    // the ISS RED/GRN/BL1 map is paler than Cassini's natural-colour mosaics (e.g. PIA17474): saturation ×1.3
    alb = max(mix(vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722))), alb, 1.3), 0.0) * uAlbK;
    // Minnaert limb darkening: I ∝ μ0^k μ^(k−1)
    float lit = pow(max(gMu0, 0.0), 0.9) * pow(max(gMu, 0.06), -0.1);
    lit += 0.01 * smoothstep(-0.1, 0.03, gMu0) * (1.0 - smoothstep(0.0, 0.15, gMu0));
    float sh = 1.0;
    if (uRingOn > 0.5 && p.y * s.y < 0.0){
      vec3 pr = p - s * (p.y / s.y);
      sh = exp(-ringTauShadow(length(pr.xz), tP * pixAng / max(abs(s.y), 0.03)) / max(abs(s.y), 0.004));
    }
    col = alb * lit * uSunI * sh;
    col += rimLight(n, rd, s) * uSunI * pow(1.0 - gMu, 6.0) * sh;
    // ringshine: the lit face of the rings lights the night side (the shaded hemisphere gets transmitted light)
    col += alb * uSunI * uRingOn * 0.12 * sqrt(abs(s.y)) * mix(0.35, 1.0, step(0.0, p.y * s.y)) * sqrt(abs(p.y)) * (1.0 - smoothstep(-0.05, 0.12, gMu0));
  } else if (tM < 1e8){
    col = moonShade(mid, ro + rd * tM, rd, s, E);
  } else {
    col = sky(rdW, 1.0) * 0.6 + sunDisc(rdW);
    float ce = dot(rdW, uEarth);   // Earth: the pale blue dot within 6° of the Sun
    col += vec3(0.45, 0.65, 1.0) * (smoothstep(0.9999985, 0.9999996, ce) * 1.6 + pow(max(ce, 0.0), 200000.0) * 0.4);
    col += moonPoints(ro, rd, s, pixAng);
    float tc = -B / A;
    if (tc > 0.0){
      vec3 pc = roE + rdE * tc;
      float hgt = max(length(pc) - 1.0, 0.0);
      col += rimLight(normalize(pc), rd, s) * uSunI * exp(-hgt / 0.02);
    }
  }
  if (abs(rd.y) > 1e-6){
    float tR = -ro.y / rd.y;
    if (tR > 0.0 && tR < tN){
      vec3 pr = ro + rd * tR;
      float r = length(pr.xz);
      if (r > 1.11 && r < 8.5 && (uRingOn > 0.5 || r > 2.5)){
        float mu = abs(rd.y), mu0 = max(abs(s.y), 1e-4);
        vec2 vh = normalize(rd.xz + 1e-7);
        float th = atan(pr.x, pr.z);
        float cr = dot(pr.xz / r, vh);
        float fw = min(tR * pixAng * sqrt(cr * cr / (mu * mu) + 1.0 - cr * cr), 0.3);
        float tau, dust; vec3 alb;
        ringSample(r, th, fw, rd, tau, alb, dust);
        // spokes: an extra layer of micron dust (dark in backscatter, bright in forward scatter)
        float tauS = (r > 1.6 && r < 1.97) ? 0.035 * spokes(r, th) : 0.0;
        if (r > 2.6) mu = max(mu, 0.02);     // E/G rings are thick: no infinite slant path at grazing views
        if (tau + tauS > 0.0){
          float shd = bodyShadow(pr, s, E);
          float c = dot(s, rd);                              // 1 = looking toward the Sun
          float alpha = acos(clamp(-c, -1.0, 1.0));          // phase angle
          float Pbig = hg(c, -0.3) * (1.0 + 0.55 * exp(-alpha / 0.02) + 0.25 * exp(-alpha / 0.12));   // + opposition surge
          float Pdust = 0.5 * hg(c, 0.7) + 0.5 * hg(c, 0.95);
          float P = mix(Pbig, Pdust, dust);
          bool litFace = s.y * rd.y < 0.0;
          float IF;
          // lit face: single scattering + a multiple-scattering gain that keeps growing slowly in thick rings
          if (litFace) IF = 0.25 * P * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu))) * (1.0 + 1.9 * (1.0 - exp(-tau / 2.2)) * (1.0 - dust));
          else {
            float dm = mu - mu0;
            IF = abs(dm) > 1e-3 ? 0.25 * P * mu0 / dm * (exp(-tau / mu) - exp(-tau / mu0)) : 0.25 * P * mu0 * tau / (mu * mu) * exp(-tau / mu);
          }
          // Saturnshine on the rings: light from the lit part of the planet (≈ phase of Saturn seen from there)
          float ph = 0.5 + 0.5 * dot(pr / r, s);
          float ssh = 0.34 * ph * ph / (r * r) * 0.25 * (1.0 - exp(-tau / mu)) * (1.0 - dust);
          vec3 ringCol = alb * (max(IF, 0.0) * shd + ssh) * uSunI;
          float ext = exp(-tau / mu);
          if (tauS > 0.0){
            float extS = exp(-tauS / mu);
            float IFs = 0.25 * Pdust * tauS / mu * 0.7;
            // (the dust cloud floats above the ring: the illumination slant path is capped)
            ringCol = ringCol * (litFace ? exp(-tauS * (1.0 / mu + 1.0 / max(mu0, 0.12))) : extS) + vec3(0.75, 0.72, 0.70) * IFs * uSunI * shd;
            ext *= extS;
          }
          col = col * ext + ringCol;
        }
      }
    }
  }
  col += plumes(ro, rd, s, tN);
  if (uAxis > 0.5){
    float b = rd.y, dd = dot(rd, ro), den = 1.0 - b * b;
    if (den > 1e-6){
      float sc = (b * ro.y - dd) / den, ta = ro.y + b * sc;
      if (sc > 0.0 && sc < tN){
        float dist = length(ro + rd * sc - vec3(0.0, ta, 0.0));
        float line = exp(-pow(dist / (1.4 * sc * pixAng), 2.0)) * step(0.92, abs(ta)) * (1.0 - smoothstep(1.35, 1.6, abs(ta)));
        col += (ta > 0.0 ? vec3(1.0, 0.42, 0.22) : vec3(0.3, 0.6, 1.0)) * line * 0.7;
      }
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`.replace(/GWS_LAT_D/g, GWS_LAT.toFixed(2)).replace(/AV_LAT_D/g, AV_LAT.toFixed(2));

/* ------------------------------------------------------------------------------------------------ */
/* data loading                                                                                       */
/* ------------------------------------------------------------------------------------------------ */
function loadImageTex(url, aniso, maxSize) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    const img = tex.image;
    if (img.width > maxSize) {
      const sc = maxSize / img.width, cv = document.createElement('canvas');
      cv.width = Math.round(img.width * sc); cv.height = Math.round(img.height * sc);
      const g = cv.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(img, 0, 0, cv.width, cv.height);
      tex.image = cv;
    }
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = aniso;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(t(`không tải được ${url}`, `could not load ${url}`)))));
}
const fetchBin = url => fetch(url).then(r => { if (!r.ok) throw new Error(t(`không tải được ${url}`, `could not load ${url}`)); return r.arrayBuffer(); });
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
const H2F = THREE.DataUtils.fromHalfFloat, F2H = THREE.DataUtils.toHalfFloat;
// ring profile: RGBA16F 16384×1 with a CPU mip chain (τ averaged in transmission, colour linearly)
function ringTexture(buf, maxW) {
  const src = new Uint16Array(buf), N = src.length / 4;
  let cur = new Float32Array(N * 4);
  for (let i = 0; i < N * 4; i++) cur[i] = H2F(src[i]);
  let W = N;
  while (W > maxW) { cur = halve(cur, W); W >>= 1; }
  const mips = [];
  for (;;) {
    const d = new Uint16Array(W * 4);
    for (let i = 0; i < W * 4; i++) d[i] = F2H(cur[i]);
    mips.push({ data: d, width: W, height: 1 });
    if (W === 1) break;
    cur = halve(cur, W); W >>= 1;
  }
  const tex = new THREE.DataTexture(mips[0].data, mips[0].width, 1, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.mipmaps = mips; tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping; tex.needsUpdate = true;
  return tex;
}
function halve(a, W) {
  const o = new Float32Array(W * 2);
  for (let i = 0; i < W / 2; i++) {
    const p = i * 8;
    o[i * 4] = -Math.log(Math.max(0.5 * (Math.exp(-a[p]) + Math.exp(-a[p + 4])), 1e-6));
    for (let c = 1; c < 4; c++) o[i * 4 + c] = 0.5 * (a[p + c] + a[p + 4 + c]);
  }
  return o;
}
// zonal profile: rgb from the map's zonal mean, a = drift rate ω(φ) relative to the interior (1e-5 rad/s)
function zonalTexture(buf) {
  const src = new Uint16Array(buf), H = src.length / 3;
  const d = new Uint16Array(H * 4);
  for (let i = 0; i < H; i++) {
    const latD = 90 - (i + 0.5) * 180 / H;
    for (let c = 0; c < 3; c++) d[i * 4 + c] = src[i * 3 + c];
    d[i * 4 + 3] = F2H(omegaInt(latD) * 1e5);
  }
  const tex = new THREE.DataTexture(d, 1, H, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = false;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping; tex.needsUpdate = true;
  return tex;
}
function solid(rgba) { const t = new THREE.DataTexture(new Uint8Array(rgba), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType); t.needsUpdate = true; return t; }
function zonalFallback() {
  const H = 256, d = new Uint16Array(H * 4);
  for (let i = 0; i < H; i++) {
    const latD = 90 - (i + 0.5) * 180 / H;
    d.set([F2H(0.80), F2H(0.62), F2H(0.38), F2H(omegaInt(latD) * 1e5)], i * 4);
  }
  const t = new THREE.DataTexture(d, 1, H, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = t.magFilter = THREE.LinearFilter; t.needsUpdate = true; return t;
}

/* ------------------------------------------------------------------------------------------------ */
/* geometry helpers                                                                                   */
/* ------------------------------------------------------------------------------------------------ */
function orient(st, obDeg, BDeg, sunAzDeg) {
  const az = sunAzDeg * D2R, ob = obDeg * D2R;
  st.sun.set(Math.sin(az), 0, Math.cos(az));
  const c = THREE.MathUtils.clamp(Math.sin(BDeg * D2R) / Math.max(Math.sin(ob), 1e-6), -1, 1);
  const paz = az + Math.acos(c);
  const P = st.pole.set(Math.sin(ob) * Math.sin(paz), Math.cos(ob), Math.sin(ob) * Math.cos(paz));
  const a = Math.abs(P.y) < 0.99 ? new V3(0, 1, 0) : new V3(1, 0, 0);
  const xb = new V3().crossVectors(a, P).normalize(), zb = new V3().crossVectors(xb, P);
  st.W2B.set(xb.x, xb.y, xb.z, P.x, P.y, P.z, zb.x, zb.y, zb.z);
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
function bodyLatLon(p, k, spin) {
  const n = new V3(p.x, p.y * k * k, p.z).normalize();
  const c = Math.cos(spin), s = Math.sin(spin);
  const qx = c * p.x - s * p.z, qz = s * p.x + c * p.z;
  return { lat: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) * R2D, lonE: wrap(Math.atan2(-qz, qx)), n };
}
// body-fixed east longitude under a world direction
function subLon(st, dirW) {
  const d = dirW.clone().normalize().applyMatrix3(st.W2B);
  const k = 1 / (1 - FLAT), sc = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
  return bodyLatLon(d.multiplyScalar(sc), k, st.spin);
}
const zoneName = rk => rk < 74490 ? t('vành D', 'D ring') : rk < 92000 ? t('vành C', 'C ring') : rk < 117580 ? t('vành B', 'B ring') : rk < 122050 ? t('khe Cassini', 'Cassini Division')
  : Math.abs(rk - 133589) < 165 ? t('khe Encke (vệ tinh Pan)', 'Encke Gap (moon Pan)') : Math.abs(rk - 136505) < 25 ? t('khe Keeler (vệ tinh Daphnis)', 'Keeler Gap (moon Daphnis)') : rk < 136775 ? t('vành A', 'A ring')
  : Math.abs(rk - 140221) < 500 ? t('vành F', 'F ring') : rk < 160000 ? t('khoảng trống giữa vành A và G', 'gap between the A and G rings') : rk < 176000 ? t('vành G', 'G ring') : t('vành E', 'E ring');

/* ------------------------------------------------------------------------------------------------ */
/* scene                                                                                              */
/* ------------------------------------------------------------------------------------------------ */
const instOf = def => def.__inst;
const SATURN = {
  id: 'saturn',
  name: 'Sao Thổ',
  category: 'planets',
  order: 41,
  subtitle: 'Khí khổng lồ có vành · dữ liệu Cassini',
  swatch: ['#f0dcae', '#7a5f38'],
  blurb: 'Sao Thổ nhẹ hơn nước và dẹt nhất hệ Mặt Trời. Bề mặt mây ở đây là <b>bản đồ Cassini thật</b> (8/2011), cuốn theo các dòng tia đo được — dòng tia xích đạo ~450 m/s, nhanh gấp 4 lần của Sao Mộc. Vành là hàng tỉ hạt băng quay theo định luật Kepler (vòng trong nhanh hơn vòng ngoài); độ dày quang học của vành lấy từ phép che sóng vô tuyến của Cassini.',
  intro: 'Phím <b>W</b>: Bão Trắng Lớn 2010–11 (tua nhanh). <b>X</b>: lục giác cực Bắc. <b>R</b>: bay sát vành xem nan hoa và các cục vật chất bị kéo xiên. <b>B</b>: nhìn ngược sáng từ trong bóng Sao Thổ như Cassini năm 2013.',
  facts: [
    ['Bán kính xích đạo / cực', '60 268 km / 54 364 km · độ dẹt 0,098'],
    ['Khối lượng · khối lượng riêng', '5,683×10²⁶ kg (95 Trái Đất) · 0,69 g/cm³'],
    ['Chu kỳ tự quay', '10 h 33 m 38 s (lõi, địa chấn vành — Mankovich 2019); System III 10 h 39 m 22 s'],
    ['Độ nghiêng trục', '26,73° · một năm Sao Thổ = 29,46 năm'],
    ['Gió xích đạo', '~450 m/s (System III) ≈ 1 600 km/h'],
    ['Lục giác cực Bắc', 'dòng tia ~120 m/s ở 77,5° B, mỗi cạnh ~14 500 km'],
    ['Vành chính (C → A)', '74 500 – 136 780 km, dày ~10 m'],
    ['Vành B', 'độ dày quang học > 5 (gần như đục)'],
    ['Bão Trắng Lớn 2010–11', 'bùng ở 37,7° B ngày 5/12/2010, quấn quanh hành tinh sau ~55 ngày'],
    ['Dữ liệu', 'Bản đồ Cassini ISS RGB 11/8/2011 (NASA PDS), cực Bắc PIA17652, vành: Cassini RSS Rev 7 + PIA11142 (NASA/JPL/SSI)'],
  ],
  en: {
    name: 'Saturn',
    subtitle: 'Ringed gas giant · Cassini data',
    blurb: "Saturn is less dense than water and the most oblate planet in the Solar System. The cloud tops here are a <b>real Cassini map</b> (August 2011), carried along by the measured jet streams — the equatorial jet runs at ~450 m/s, four times faster than Jupiter's. The rings are billions of ice particles orbiting by Kepler's laws (inner ringlets faster than outer ones); their optical depth comes from Cassini radio occultations.",
    intro: "Key <b>W</b>: the 2010–11 Great White Spot (fast-forward). <b>X</b>: the north polar hexagon. <b>R</b>: skim the rings to see spokes and sheared clumps. <b>B</b>: look back toward the Sun from inside Saturn's shadow, as Cassini did in 2013.",
    facts: [
      ['Equatorial / polar radius', '60,268 km / 54,364 km · oblateness 0.098'],
      ['Mass · density', '5.683×10²⁶ kg (95 Earths) · 0.69 g/cm³'],
      ['Rotation period', '10 h 33 m 38 s (interior, from ring seismology — Mankovich 2019); System III 10 h 39 m 22 s'],
      ['Axial tilt', '26.73° · one Saturn year = 29.46 years'],
      ['Equatorial wind', '~450 m/s (System III) ≈ 1,600 km/h'],
      ['North polar hexagon', 'jet stream ~120 m/s at 77.5° N, each side ~14,500 km'],
      ['Main rings (C → A)', '74,500 – 136,780 km, ~10 m thick'],
      ['B ring', 'optical depth > 5 (nearly opaque)'],
      ['Great White Spot 2010–11', 'erupted at 37.7° N on 5 Dec 2010, wrapped around the planet after ~55 days'],
      ['Data', 'Cassini ISS RGB map 11 Aug 2011 (NASA PDS), north pole PIA17652, rings: Cassini RSS Rev 7 + PIA11142 (NASA/JPL/SSI)'],
    ],
  },
  unitKm: RKM,
  camera: { distance: 5.5, polar: 66, azimuth: 20, min: 1.04, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.35,
  params: [
    { key: 'season', label: 'Mặt Trời trên mặt phẳng vành (B′)', type: 'range', min: -26.7, max: 26.7, step: 0.1, default: 16, fmt: v => v.toFixed(1) + '°', note: '0° = điểm phân (2009, 2025): vành tối, bóng vành co thành một vạch, nan hoa xuất hiện. ±26,7° = điểm chí (Bắc: 2017).',
      en: { label: 'Sun above the ring plane (B′)', note: '0° = equinox (2009, 2025): the rings go dark, their shadow shrinks to a line and spokes appear. ±26.7° = solstice (northern: 2017).' } },
    { key: 'sunAz', label: 'Hướng Mặt Trời', type: 'range', min: 0, max: 360, step: 1, default: 45, fmt: v => v.toFixed(0) + '°', note: 'Phương vị của Mặt Trời. Xoay camera ra phía sau hành tinh để nhìn ngược sáng.',
      en: { label: 'Sun direction', note: "The Sun's azimuth. Orbit the camera behind the planet for a backlit view." } },
    { key: 'lapse', label: 'Tua nhanh thời tiết', type: 'range', min: 0, max: 200, step: 0.5, default: 4, fmt: v => '×' + v.toFixed(1), note: 'Nhân thêm tốc độ của gió (so với đồng hồ tự quay). ×1 = gió và tự quay cùng nhịp thời gian thật đã tua.',
      en: { label: 'Weather fast-forward', note: 'Extra speed-up of the winds (relative to the rotation clock). ×1 = winds and rotation share the same accelerated real time.' } },
    { key: 'gws', label: 'Bão Trắng Lớn — ngày thứ', type: 'range', min: -1, max: 300, step: 0.5, default: -1, fmt: v => v < 0 ? t('tắt', 'off') : vn(v, 0), note: 'Số ngày kể từ khi bão bùng phát (5/12/2010). Nhấn W để xem tua nhanh.',
      en: { label: 'Great White Spot — day', note: 'Days since the storm erupted (5 Dec 2010). Press W to watch it fast-forwarded.' } },
    { key: 'track', label: 'Bám theo đầu bão', type: 'toggle', default: false, note: 'Quay hành tinh theo đầu cơn bão (như phim time-lapse của Cassini).',
      en: { label: 'Track the storm head', note: "Turn the planet to follow the head of the storm (like Cassini's time-lapse movies)." } },
    { key: 'spokes', label: 'Nan hoa trên vành B', type: 'toggle', default: true, note: 'Bụi mịn bị nhiễm điện bay lên khỏi vành; thường gặp gần điểm phân.',
      en: { label: 'Spokes on the B ring', note: 'Fine, electrically charged dust levitating above the rings; most common near equinox.' } },
    { key: 'dust', label: 'Phóng đại vành bụi (D, G, E)', type: 'range', min: 1, max: 200, step: 1, default: 1, fmt: v => '×' + v.toFixed(0), note: 'Vành E có độ dày quang học ~10⁻⁵: chỉ thấy khi nhìn ngược sáng với phơi sáng rất dài.',
      en: { label: 'Boost dusty rings (D, G, E)', note: 'The E ring has an optical depth of ~10⁻⁵: it only shows up backlit, in very long exposures.' } },
    { key: 'moons', label: 'Vệ tinh', type: 'toggle', default: true, en: { label: 'Moons' } },
    { key: 'spin', label: 'Tự quay', type: 'toggle', default: true, en: { label: 'Rotation' } },
    { key: 'axis', label: 'Trục quay', type: 'toggle', default: false, en: { label: 'Spin axis' } },
  ],
  actions: [
    { label: 'Bão Trắng Lớn', key: 'w', title: 'Bão Trắng Lớn 2010–11, tua nhanh (1 ngày ≈ 2,5 giây)', run: ctx => instOf(SATURN)?.gwsShot(ctx),
      en: { label: 'Great White Spot', title: 'The 2010–11 Great White Spot, fast-forwarded (1 day ≈ 2.5 seconds)' } },
    { label: 'Lục giác', key: 'x', title: 'Bay lên trên cực Bắc', run: ctx => instOf(SATURN)?.hexShot(ctx),
      en: { label: 'Hexagon', title: 'Fly up over the north pole' } },
    { label: 'Cận cảnh vành', key: 'r', title: 'Bay sát vành B–A (bấm lần nữa để quay lại)', run: ctx => instOf(SATURN)?.ringShot(ctx),
      en: { label: 'Ring close-up', title: 'Skim the B–A rings (press again to return)' } },
    { label: 'Ngược sáng', key: 'b', title: 'Vào bóng Sao Thổ, nhìn về phía Mặt Trời (Cassini 19/7/2013)', run: ctx => instOf(SATURN)?.backlitShot(ctx),
      en: { label: 'Backlit', title: "Into Saturn's shadow, looking toward the Sun (Cassini, 19 July 2013)" } },
    { label: 'Điểm phân', key: 'e', title: 'Mặt Trời nằm trong mặt phẳng vành', run: ctx => { ctx.params.season = 0.6; ctx.toast(t('Gần điểm phân (8/2009, 5/2025): Mặt Trời chiếu ngang mép vành, vành tối hẳn, bóng vành co thành một vạch. Lúc này nan hoa hiện rõ nhất trên vành B.',
      'Near equinox (Aug 2009, May 2025): sunlight grazes the rings edge-on, they go dark and their shadow shrinks to a line. This is when spokes show best on the B ring.'), 7000); },
      en: { label: 'Equinox', title: 'The Sun in the ring plane' } },
    { label: 'Enceladus', key: 'n', title: 'Bám theo Enceladus và các mạch phun ở cực Nam (bấm lần nữa để quay lại)', run: ctx => instOf(SATURN)?.enceladusShot(ctx),
      en: { label: 'Enceladus', title: 'Follow Enceladus and its south-polar plumes (press again to return)' } },
  ],
  create(ctx) {
    const R = ctx.renderer, caps = R.capabilities;
    const aniso = Math.min(8, caps.getMaxAnisotropy());
    const U = {
      ...ctx.shared,
      uSunDir: { value: new V3(0, 0, 1) }, uEarth: { value: new V3(0, 0, 1) }, uW2B: { value: new THREE.Matrix3() },
      uSpin: { value: 0 }, uSunI: { value: 0.9 }, uTexOK: { value: 0 }, uAxis: { value: 0 }, uRingOn: { value: 1 }, uDust: { value: 1 },
      uMoonsOn: { value: 1 }, uAlbK: { value: 0.70 },
      uMap: { value: solid([220, 190, 140, 255]) }, uZonal: { value: zonalFallback() }, uCap: { value: solid([200, 180, 140, 255]) },
      uRingTex: { value: solid([0, 0, 0, 0]) },
      uPhW: { value: new V3(0, 0, 1) }, uPhT: { value: new THREE.Vector2() }, uSeedW: { value: new THREE.Vector2() }, uHex: { value: 0 },
      uPhR: { value: new V3(0, 0, 1) }, uSeedR: { value: new THREE.Vector2() },
      uSpk: { value: new THREE.Vector4() }, uGws: { value: new THREE.Vector4(-1, 0, 0, 0) },
      uMoon: { value: Array.from({ length: N_MOON }, () => new THREE.Vector4()) },
      uFR: { value: new THREE.Vector4() }, uProp: { value: new V3() },
    };
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    const maxTex = caps.maxTextureSize;
    const ready = Promise.all([
      retry(loadImageTex, ASSET('saturn_map_4k.jpg'), aniso, maxTex),
      retry(loadImageTex, ASSET('saturn_npole.jpg'), 4, maxTex),
      retry(fetchBin, ASSET('saturn_rings.f16')),
      retry(fetchBin, ASSET('saturn_zonal.f16')),
    ]).then(([map, cap, ring, zon]) => {
      cap.wrapS = THREE.ClampToEdgeWrapping;
      U.uMap.value = map; U.uCap.value = cap;
      U.uRingTex.value = ringTexture(ring, Math.min(16384, maxTex));
      U.uZonal.value = zonalTexture(zon);
      for (const t of [map, cap, U.uRingTex.value, U.uZonal.value]) R.initTexture(t);
      U.uTexOK.value = 1;
    });

    const st = {
      W2B: U.uW2B.value, sun: U.uSunDir.value, pole: new V3(), spinOff: 0, frozen: 0, lastSim: null,
      tp: 0, W: null, spin: 0, gwsLon0: 0, avLon0: 0, gwsPlaying: false, follow: null, ringBack: null,
    };
    const omega = TAU / P_INT;
    const clock = (ctx) => {
      const p = ctx.params, sim = ctx.time.sim;
      if (st.lastSim === null) st.lastSim = sim;
      const dsim = Math.max(0, sim - st.lastSim); st.lastSim = sim;
      if (p.spin === false) st.frozen += dsim;
      st.tp = (sim - st.frozen) * SECS;
      const dW = p.spin !== false ? dsim * SECS * (p.lapse ?? 4) : 0;
      if (st.W === null) st.W = st.tp * (p.lapse ?? 4); else st.W += dW;
      // Great White Spot clock: runs on the weather clock while playing
      if (st.gwsPlaying && p.gws >= 0) {
        p.gws = Math.min(300, p.gws + dW / 86400);
        if (p.gws >= 300) st.gwsPlaying = false;
      }
      orient(st, 26.73, p.season, p.sunAz);
      st.spin = omega * st.tp + st.spinOff;
      // keep the storm head under the camera ('track')
      if (p.track && p.gws >= 0) {
        const headLon = wrap(st.gwsLon0 + GWS_OM * p.gws * 86400);
        const sc = subLon(st, ctx.camera.position);
        st.spinOff += -(headLon - sc.lonE);
        st.spinOff = wrap(st.spinOff);
        st.spin = omega * st.tp + st.spinOff;
      }
      return dW;
    };
    st.clock = clock;
    const moonPos = (i) => {
      const m = MOONS[i], a = m[4] + TAU * st.tp / m[3];
      return new V3(Math.sin(a) * m[1], 0, Math.cos(a) * m[1]);
    };
    st.moonPos = moonPos;
    const B2W = () => st.W2B.clone().transpose();

    const inst = {
      material, ready, st,
      enter(ctx) { st.pendingEnter = true; },
      exit(ctx) { if (st.follow) inst.enceladusShot(ctx, true); },
      update(dt, ctx) {
        const p = ctx.params;
        clock(ctx);
        if (st.pendingEnter) {
          st.pendingEnter = false;
          // dev hook for tools/snap.sh: ?shot=gws|hex|ring|backlit|enceladus runs that action on arrival
          const shot = new URLSearchParams(location.search).get('shot');
          if (shot && new URLSearchParams(location.search).get('snap')) {
            ({ gws: inst.gwsShot, hex: inst.hexShot, ring: inst.ringShot, backlit: inst.backlitShot, enceladus: inst.enceladusShot, ringtop: inst.ringTop })[shot]?.(ctx, false, true);
          }
        }
        U.uSpin.value = st.spin % TAU;
        // weather phases
        const W = st.W, t1 = W % PH_P, t2 = (W + PH_P / 2) % PH_P;
        U.uPhW.value.set(t1, t2, 1 - Math.abs(2 * t1 / PH_P - 1));
        U.uPhT.value.set(W - t1, W - t2);
        U.uSeedW.value.set(Math.floor(W / PH_P) % 37, (Math.floor((W + PH_P / 2) / PH_P) % 37) + 0.5);
        U.uHex.value = wrap(DOM3 * W);
        // rings
        const tp = st.tp, r1 = tp % RPH_P, r2 = (tp + RPH_P / 2) % RPH_P;
        U.uPhR.value.set(r1, r2, 1 - Math.abs(2 * r1 / RPH_P - 1));
        U.uSeedR.value.set((Math.floor(tp / RPH_P) % 41) * 1.7, (Math.floor((tp + RPH_P / 2) / RPH_P) % 41) * 1.7 + 0.9);
        const sB = st.sun.clone().applyMatrix3(st.W2B);
        // dawn ansa: where ring particles come out of Saturn's shadow (prograde = increasing atan(x, z))
        const thAnti = Math.atan2(-sB.x, -sB.z);
        const spokeAmt = p.spokes === false ? 0 : Math.exp(-Math.pow(p.season / 13, 2));
        U.uSpk.value.set(spokeAmt, thAnti + Math.PI / 2 + 0.25, tp / SPK_DT, 0);
        const aF = 140221 / RKM;
        U.uFR.value.set(wrap(kepler(aF) * tp), wrap(1.3 + 2.70 * D2R / 86400 * tp), wrap(MOONS[6][4] + TAU * tp / MOONS[6][3]), wrap(MOONS[7][4] + TAU * tp / MOONS[7][3]));
        U.uProp.value.set(...PROPS.map(pr => wrap(pr[2] + kepler(pr[1] / RKM) * tp)));
        for (let i = 0; i < N_MOON; i++) { const m = moonPos(i); U.uMoon.value[i].set(m.x, m.y, m.z, MOONS[i][2]); }
        U.uMoonsOn.value = p.moons === false ? 0 : 1;
        // storm
        if (p.gws >= 0) {
          const ageS = p.gws * 86400;
          U.uGws.value.set(p.gws, wrap(st.gwsLon0 + GWS_OM * ageS), wrap(st.avLon0 + AV_OM * Math.max(ageS - 12 * 86400, 0)), 0);
        } else U.uGws.value.x = -1;
        U.uAxis.value = p.axis ? 1 : 0;
        U.uDust.value = p.dust ?? 1;
        // Earth seen from Saturn: within 6.0° of the Sun (here at 5°, along the ecliptic ≈ world xz-plane)
        const ea = Math.atan2(st.sun.x, st.sun.z) + 5.0 * D2R;
        U.uEarth.value.set(Math.sin(ea), 0.012, Math.cos(ea)).normalize();
        // follow Enceladus
        if (st.follow) {
          const m = moonPos(1).applyMatrix3(B2W());
          const d = m.clone().sub(st.follow.last); st.follow.last.copy(m);
          ctx.camera.position.add(d); ctx.controls.target.add(d);
        }
      },
      hud(ctx) {
        const p = ctx.params; clock(ctx);
        const cam = ctx.camera.position.clone();
        const sc = subLon(st, cam);
        const u3 = windS3(sc.lat), ui = u3 + DOM3 * rhoAxis(sc.lat);
        const rows = [[t('Vĩ độ dưới camera', 'Latitude below the camera'), latStr(sc.lat)],
          [t('Gió tại đó', 'Wind there'), `${u3 >= 0 ? '+' : '−'}${Math.abs(u3).toFixed(0)} m/s (System III) · ${ui >= 0 ? '+' : '−'}${Math.abs(ui).toFixed(0)} ${t('m/s so với lõi', 'm/s relative to the interior')}`]];
        const cb = cam.clone().normalize().applyMatrix3(st.W2B);
        rows.push([t('Độ mở vành nhìn từ camera', 'Ring opening seen from the camera'), vn(Math.asin(cb.y) * R2D) + '°'], [t('Mặt Trời trên mặt phẳng vành', 'Sun above the ring plane'), vn(p.season) + '°']);
        const real = SECS * (ctx.bhGlobals?.timeScale ?? 6);
        rows.push([t('Tua nhanh', 'Fast-forward'), t(`tự quay ×${fmtInt(real)} · thời tiết ×${fmtInt(real * (p.lapse ?? 4))}`, `rotation ×${fmtInt(real)} · weather ×${fmtInt(real * (p.lapse ?? 4))}`)]);
        if (p.gws >= 0) {
          const d0 = new Date(Date.UTC(2010, 11, 5) + p.gws * 86400e3);
          rows.push([t('Bão Trắng Lớn', 'Great White Spot'), t(`ngày ${vn(p.gws, 0)} (${d0.getUTCDate()}/${d0.getUTCMonth() + 1}/${d0.getUTCFullYear()})${p.gws < 55 ? ' — đuôi đang lan' : p.gws < 200 ? ' — đã quấn quanh hành tinh' : ' — đầu bão đã tan'}`,
            `day ${vn(p.gws, 0)} (${d0.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })})${p.gws < 55 ? ' — tail spreading' : p.gws < 200 ? ' — wrapped around the planet' : ' — storm head has dissipated'}`)]);
        }
        return rows;
      },
      onClick(info, ctx) {
        const h = hitBody(info.origin, info.dir, st.W2B);
        if (Math.abs(h.rd.y) > 1e-6) {
          const tc = -h.ro.y / h.rd.y;
          if (tc > 0 && (h.t === null || tc < h.t)) {
            const pr = h.ro.clone().addScaledVector(h.rd, tc), r = Math.hypot(pr.x, pr.z), rk = r * RKM;
            if (rk > 66900 && rk < 141500) {
              const T = TAU / kepler(r) / 3600, v = Math.sqrt(GM / (rk * 1e3)) / 1000;
              const dTdr = 1.5 * T / rk * 100 * 60;   // minutes per 100 km
              ctx.toast(t(`<b>${zoneName(rk)}</b>, cách tâm ${fmtInt(rk)} km. Hạt băng ở đây bay ${vn(v)} km/s, hết một vòng trong <b>${vn(T)} giờ</b> — ${T < P_INT / 3600 ? 'nhanh hơn' : 'chậm hơn'} một vòng tự quay (10,56 giờ). Dịch ra ngoài 100 km, chu kỳ dài thêm ${vn(dTdr, 1)} phút: vì vậy mọi cục vật chất đều bị kéo xiên.`,
                `<b>${zoneName(rk)}</b>, ${fmtInt(rk)} km from the centre. Ice particles here move at ${vn(v)} km/s and complete an orbit in <b>${vn(T)} h</b> — ${T < P_INT / 3600 ? 'faster' : 'slower'} than one rotation of the planet (10.56 h). Move 100 km outward and the period grows by ${vn(dTdr, 1)} min: that is why every clump gets sheared.`), 8000);
              ctx.audio.ping(300 + 600 / r, 0.05, info.ndcX);
              return;
            }
          }
        }
        if (h.t === null) return;
        const p = h.ro.clone().addScaledVector(h.rd, h.t);
        const ll = bodyLatLon(p, h.k, st.spin);
        const u = windS3(ll.lat);
        ctx.toast(t(`Vĩ độ <b>${latStr(ll.lat)}</b>: gió ${u >= 0 ? '+' : '−'}${Math.abs(u).toFixed(0)} m/s ${u >= 0 ? '(về đông)' : '(về tây)'}${Math.abs(ll.lat) < 20 ? ' — dòng tia xích đạo nhanh gấp 4 lần của Sao Mộc' : ll.lat > 74 ? ' — dòng tia lục giác' : Math.abs(ll.lat - 37.7) < 4 ? ' — vĩ độ của Bão Trắng Lớn 2010–11' : ''}.`,
          `Latitude <b>${latStr(ll.lat)}</b>: wind ${u >= 0 ? '+' : '−'}${Math.abs(u).toFixed(0)} m/s ${u >= 0 ? '(eastward)' : '(westward)'}${Math.abs(ll.lat) < 20 ? " — the equatorial jet, four times faster than Jupiter's" : ll.lat > 74 ? ' — the hexagon jet' : Math.abs(ll.lat - 37.7) < 4 ? ' — the latitude of the 2010–11 Great White Spot' : ''}.`));
      },
      gwsShot(ctx, _unused, quiet) {
        const p = ctx.params; clock(ctx);
        p.season = 8; p.lapse = 80; p.track = true;
        if (!(quiet && p.gws >= 0)) p.gws = 0;
        st.gwsPlaying = !quiet;
        // camera: above 25°N on the sunlit side, 40° phase
        const side = new V3().crossVectors(st.pole, st.sun).normalize();
        const dir = st.sun.clone().multiplyScalar(0.85).addScaledVector(side, 0.45).addScaledVector(st.pole, 0.55).normalize();
        ctx.camera.position.copy(dir.multiplyScalar(3.0)); ctx.controls.target.set(0, 0, 0); ctx.camera.lookAt(0, 0, 0); ctx.controls.update();
        orient(st, 26.73, p.season, p.sunAz);
        const sc = subLon(st, ctx.camera.position);
        st.gwsLon0 = sc.lonE; st.avLon0 = wrap(sc.lonE - 0.35);
        if (!quiet) ctx.toast(t('Bão Trắng Lớn (5/12/2010): một cột đối lưu khổng lồ trồi lên ở 37,7° B, phun băng amoniac trắng lên đỉnh mây. Đầu bão trôi về tây ~28 m/s; vật chất nó thả ra bị dòng tia cuốn về đông, sau ~55 ngày đuôi quấn kín hành tinh. Thời tiết đang tua: 1 ngày ≈ 2,5 giây; camera bám theo đầu bão.',
          'Great White Spot (5 Dec 2010): a giant convective plume punched up at 37.7° N, spraying white ammonia ice onto the cloud tops. The storm head drifted west at ~28 m/s; the material it shed was swept east by the jet, and after ~55 days the tail had wrapped all the way around the planet. Weather is fast-forwarded: 1 day ≈ 2.5 seconds; the camera tracks the storm head.'), 11000);
      },
      hexShot(ctx, _u, quiet) {
        const p = ctx.params; if (p.season < 8) p.season = 20; p.track = false;
        clock(ctx);
        const d = 2.6;
        const side = new V3().crossVectors(st.pole, st.sun).normalize();
        const dir = st.pole.clone().multiplyScalar(0.96).addScaledVector(st.sun, 0.25).addScaledVector(side, 0.05).normalize();
        ctx.camera.position.copy(dir.multiplyScalar(d)); ctx.controls.target.set(0, 0, 0); ctx.camera.lookAt(0, 0, 0); ctx.controls.update();
        if (!quiet) ctx.toast(t('Lục giác cực Bắc: dòng tia ~120 m/s uốn khúc thành sóng đứng 6 cạnh, rộng gấp hơn 2 lần Trái Đất. Hình lục giác quay đúng chu kỳ System III (10 h 39 m 22 s) — chậm hơn lõi ~6 phút mỗi vòng. Ở tâm là xoáy thuận với mắt bão rộng ~2 000 km, gió ~150 m/s. Màu xanh xám bên trong: sương mù hạt nhỏ, bị dòng tia ngăn lại như "lỗ thủng ôzôn".',
          'North polar hexagon: a ~120 m/s jet stream meanders into a six-sided standing wave more than twice as wide as Earth. The hexagon rotates with exactly the System III period (10 h 39 m 22 s) — about 6 minutes per turn slower than the interior. At its centre sits a cyclone with an eye ~2,000 km wide and ~150 m/s winds. The blue-grey colour inside is fine-particle haze held back by the jet, like an "ozone hole".'), 10000);
      },
      ringShot(ctx, _u, quiet) {
        const p = ctx.params;
        if (st.ringBack && !quiet) {
          ctx.camera.position.copy(st.ringBack.pos); ctx.controls.target.copy(st.ringBack.tgt); ctx.camera.lookAt(ctx.controls.target); ctx.controls.update();
          ctx.controls.minDistance = SATURN.camera.min; st.ringBack = null; return;
        }
        st.ringBack = { pos: ctx.camera.position.clone(), tgt: ctx.controls.target.clone() };
        if (Math.abs(p.season) < 3 || Math.abs(p.season) > 12) p.season = 8 * (Math.sign(p.season) || 1);
        clock(ctx);
        const sB = st.sun.clone().applyMatrix3(st.W2B);
        // a point on the outer B ring on the morning side, seen from 6 000 km above the lit face
        const thAnti = Math.atan2(-sB.x, -sB.z), th = thAnti + Math.PI / 2 + 0.35;
        const rT = 1.80, Bw = B2W();
        const tgt = new V3(Math.sin(th) * rT, 0, Math.cos(th) * rT);
        const up = Math.sign(sB.y) || 1;
        const out = new V3(Math.sin(th), 0, Math.cos(th)), along = new V3(Math.cos(th), 0, -Math.sin(th));
        const cam = tgt.clone().addScaledVector(out, 0.42).addScaledVector(along, -0.10).add(new V3(0, 0.075 * up, 0));
        if (ctx.bhGlobals) ctx.bhGlobals.exposure = Math.max(ctx.bhGlobals.exposure, 1.4);
        ctx.controls.minDistance = 0.01;
        ctx.controls.target.copy(tgt.applyMatrix3(Bw)); ctx.camera.position.copy(cam.applyMatrix3(Bw));
        ctx.camera.lookAt(ctx.controls.target); ctx.controls.update();
        if (!quiet) ctx.toast(t('Lơ lửng 4 500 km phía trên vành A (khe Encke ở dưới), nhìn vào vành B ngay sau rìa bình minh — nơi nan hoa hình thành. Các cục vật chất và "sóng thức" (wake) bị kéo xiên vì hạt ở vòng trong quay nhanh hơn; nan hoa (bụi tích điện) hiện lên rồi bị kéo nghiêng trong vài giờ. Nhấp vào vành để xem chu kỳ quỹ đạo tại đó.',
          'Hovering 4,500 km above the A ring (the Encke Gap below), looking at the B ring just past the dawn ansa — where spokes form. Clumps and self-gravity wakes are sheared because inner particles orbit faster; spokes (charged dust) appear and are tilted over within hours. Click the rings to see the orbital period there.'), 9000);
      },
      // (verification view) straight down onto the A ring from 3 000 km: clumps drifting and shearing at Ω_K(r)
      ringTop(ctx) {
        clock(ctx);
        const Bw = B2W(), th = 0.7, rT = 2.12, up = Math.sign(st.sun.clone().applyMatrix3(st.W2B).y) || 1;
        const tgt = new V3(Math.sin(th) * rT, 0, Math.cos(th) * rT);
        const cam = tgt.clone().add(new V3(0, 0.05 * up, 0)).addScaledVector(new V3(Math.sin(th), 0, Math.cos(th)), 0.004);
        ctx.controls.minDistance = 0.01;
        ctx.controls.target.copy(tgt.applyMatrix3(Bw)); ctx.camera.position.copy(cam.applyMatrix3(Bw));
        ctx.camera.lookAt(ctx.controls.target); ctx.controls.update();
      },
      backlitShot(ctx, _u, quiet) {
        const p = ctx.params; clock(ctx);
        p.season = 17; p.dust = 80; p.track = false;
        orient(st, 26.73, p.season, p.sunAz);
        const sgn = Math.sign(st.sun.dot(st.pole)) || 1;
        const perp = st.pole.clone().addScaledVector(st.sun, -st.pole.dot(st.sun)).normalize();
        // inside the shadow cylinder (0.6 R off-axis): the planet hides the Sun, the rings open at B′ as seen from the Sun
        const dir = st.sun.clone().negate().addScaledVector(perp, -sgn * 0.04).normalize();
        ctx.camera.position.copy(dir.multiplyScalar(15)); ctx.controls.target.set(0, 0, 0); ctx.camera.lookAt(0, 0, 0); ctx.controls.update();
        if (ctx.bhGlobals) ctx.bhGlobals.exposure = Math.max(ctx.bhGlobals.exposure, 1.5);
        if (!quiet) ctx.toast(t('Trong bóng Sao Thổ, nhìn về phía Mặt Trời (Cassini, 19/7/2013 — "ngày Trái Đất mỉm cười"): bụi mịn tán xạ về phía trước nên vành C, khe Cassini, vành F, G và E sáng rực (vành bụi đang phóng đại ×80, như ảnh ghép đã kéo sáng của NASA), còn vành B dày đặc lại tối. Chấm xanh nhạt cạnh Sao Thổ là Trái Đất.',
          'Inside Saturn\'s shadow, looking toward the Sun (Cassini, 19 July 2013 — "The Day the Earth Smiled"): fine dust forward-scatters sunlight, so the C ring, the Cassini Division and the F, G and E rings blaze (dusty rings boosted ×80, like NASA\'s stretched mosaic), while the dense B ring goes dark. The pale blue dot beside Saturn is Earth.'), 11000);
      },
      enceladusShot(ctx, leave, quiet) {
        if (st.follow) {
          ctx.camera.position.copy(st.follow.pos); ctx.controls.target.copy(st.follow.tgt); ctx.camera.lookAt(ctx.controls.target); ctx.controls.update();
          ctx.controls.minDistance = SATURN.camera.min; ctx.params.dust = st.follow.dust; st.follow = null; return;
        }
        if (leave === true) return;
        clock(ctx);
        const p = ctx.params;
        st.follow = { pos: ctx.camera.position.clone(), tgt: ctx.controls.target.clone(), dust: p.dust, last: new V3() };
        p.moons = true; p.dust = 4;
        const m = moonPos(1).applyMatrix3(B2W());
        st.follow.last.copy(m);
        const Rm = MOONS[1][2];
        // backlit: the camera on the far side of Enceladus from the Sun, a little south
        const dir = st.sun.clone().negate().addScaledVector(st.pole, -0.35).normalize();
        ctx.controls.minDistance = Rm * 1.5;
        ctx.controls.target.copy(m); ctx.camera.position.copy(m.clone().addScaledVector(dir, Rm * 14)); ctx.camera.lookAt(m); ctx.controls.update();
        if (!quiet) ctx.toast(t('Enceladus (đường kính 504 km): hơn 100 mạch phun hơi nước và hạt băng thoát ra từ các "vết hổ" ở cực Nam, nuôi vành E. Đại dương lỏng nằm dưới lớp băng. Mạch phun sáng nhất khi nhìn ngược sáng.',
          'Enceladus (504 km across): more than 100 jets of water vapour and ice grains escape from the south-polar "tiger stripes", feeding the E ring. A liquid ocean lies beneath the ice shell. The plumes are brightest when backlit.'), 9000);
      },
    };
    SATURN.__inst = inst;
    return inst;
  },
};

export default [SATURN];
