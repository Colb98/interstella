// Hành tinh khí khổng lồ & sao lùn nâu — Jupiter, Saturn, Neptune, Uranus and the brown dwarf Luhman 16B.
//
// One scene unit = the body's equatorial radius. The four planets share one ray-cast renderer:
//   * oblate ellipsoid (scale y by 1/(1−f)), body frame tilted by the real obliquity, Sun in the world
//     xz-plane (= the planet's orbital plane), so axial tilt and seasons are geometric, not painted on;
//   * cloud bands advected by a latitude-dependent zonal wind profile (measured jets, m/s). The noise
//     domain is sheared by ω(φ)·t, using the two-phase trick (two copies offset by half a period,
//     each reset when its weight is zero) so the shear never winds up;
//   * Minnaert limb darkening, a forward-scattering haze rim, analytic shadows (rings ↔ planet,
//     Galilean moons on Jupiter), and a ring-plane intersection with real radial structure and
//     single-scattering ring photometry (reflected / diffusely transmitted light, dust phase function).
// The brown dwarf has its own emission shader (grey-atmosphere limb law, blackbody × alkali absorption).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

const V3 = THREE.Vector3;
const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAU = Math.PI * 2;
// 1 unit of ctx.time.sim = 60 s on the planet (≈ 360× real time at the host's default 6× slider):
// Jupiter turns once in ~100 s of wall time.
const SECS = 60;

/* ------------------------------------------------------------------------------------------------ */
/* small helpers                                                                                      */
/* ------------------------------------------------------------------------------------------------ */
const ff = v => (+v).toFixed(4);
// piecewise-linear table → GLSL function (x in degrees of latitude)
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
function tab3(name, pts) {
  const n = pts.length;
  return `vec3 ${name}(float x){
  const float X[${n}] = float[${n}](${pts.map(p => ff(p[0])).join(',')});
  const vec3 Y[${n}] = vec3[${n}](${pts.map(p => `vec3(${p[1].map(ff).join(',')})`).join(',')});
  if (x <= X[0]) return Y[0];
  for (int i = 1; i < ${n}; i++){ if (x < X[i]){ float u = (x - X[i-1]) / (X[i] - X[i-1]); return mix(Y[i-1], Y[i], u * u * (3.0 - 2.0 * u)); } }
  return Y[${n - 1}];
}`;
}
function lerpTab(pts, x) {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) if (x < pts[i][0]) return pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * (x - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
  return pts[pts.length - 1][1];
}
const latlonJS = (latD, lonE) => { const la = latD * D2R; return new V3(Math.cos(la) * Math.cos(lonE), Math.sin(la), -Math.cos(la) * Math.sin(lonE)); };
const vn = (x, d = 1) => x.toFixed(d).replace('.', ',');
const latStr = d => `${vn(Math.abs(d))}° ${d >= 0 ? 'B' : 'N'}`;
const windStr = u => `${u >= 0 ? '+' : '−'}${Math.abs(u).toFixed(0)} m/s ${u >= 0 ? '(về đông)' : '(về tây)'}`;
const wrap = a => ((a % TAU) + TAU) % TAU;
const fmtInt = n => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');

/* ------------------------------------------------------------------------------------------------ */
/* zonal wind profiles (planetographic latitude °, m/s, relative to the interior rotation)            */
/* ------------------------------------------------------------------------------------------------ */
// Jupiter: after Voyager/Cassini cloud tracking (Limaye 1986, Porco+ 2003), smoothed. Belts sit where
// the shear is cyclonic, zones where it is anticyclonic — the 23.7°N jet is the fastest (~165 m/s).
const JUP_WIND = [[-90, 0], [-72, 0], [-66, 25], [-61, -10], [-57, 30], [-53, -12], [-49, 28], [-45, -15], [-39.5, 42], [-36, -22],
  [-32, 48], [-29, -18], [-26.5, 55], [-22, 0], [-19.5, -58], [-16, -15], [-12, 15], [-7, 135], [-3, 95], [0, 85], [3, 95], [7, 128],
  [10, 60], [14, 10], [17, -32], [20, 25], [23.7, 165], [26.5, -20], [29.5, 25], [31.5, 10], [35.5, 32], [39, -12], [43, 30],
  [46.5, -10], [49, 25], [53, -10], [56, 28], [60, -5], [65, 18], [70, 0], [90, 0]];
// Saturn: broad 400+ m/s equatorial jet (Sánchez-Lavega 2000, Cassini), the hexagon jet at 78°N.
const SAT_WIND = [[-90, 0], [-88, 150], [-82, -10], [-74, 60], [-67, -10], [-62, 120], [-55, 60], [-50, -10], [-42, 110], [-36, 0],
  [-28, 110], [-20, 260], [-10, 395], [0, 430], [10, 395], [20, 260], [28, 110], [35, 0], [42, 110], [47, -10], [55, 60], [62, 130],
  [67, -10], [72, 60], [75, 0], [78, 110], [82, -10], [88, 150], [90, 0]];
// Neptune: polynomial fit to Voyager tracking (Sromovsky+ 1993): −400 m/s retrograde equator, +250 m/s at 70°.
const nepWind = d => -398 + 0.188 * d * d - 1.2e-5 * d * d * d * d;
// Uranus: retrograde equator, prograde jets ~250 m/s near ±60° (Voyager, Keck; Sromovsky 2015).
const URA_WIND = [[-90, 0], [-72, 170], [-60, 240], [-45, 180], [-25, 40], [-15, -30], [0, -70], [15, -30], [25, 40], [45, 180],
  [60, 250], [72, 190], [90, 0]];

// Jupiter cloud albedo (linear RGB) by latitude: zones cream, belts brown, NTBs orange, polar regions blue-grey
const JUP_COL = [
  [-90, [0.36, 0.37, 0.41]], [-62, [0.47, 0.46, 0.47]], [-54, [0.56, 0.53, 0.49]], [-48, [0.50, 0.48, 0.45]], [-44, [0.60, 0.57, 0.51]],
  [-41, [0.70, 0.66, 0.58]], [-38, [0.55, 0.50, 0.44]], [-34, [0.80, 0.74, 0.62]], [-31, [0.54, 0.45, 0.36]], [-28, [0.64, 0.56, 0.46]],
  [-25, [0.85, 0.78, 0.64]], [-21, [0.84, 0.77, 0.62]], [-19, [0.46, 0.29, 0.18]], [-15, [0.58, 0.42, 0.29]], [-11, [0.43, 0.27, 0.17]],
  [-7.5, [0.80, 0.69, 0.52]], [-3, [0.87, 0.77, 0.60]], [0, [0.80, 0.66, 0.48]], [3, [0.87, 0.77, 0.60]], [6.5, [0.70, 0.57, 0.43]],
  [8.5, [0.38, 0.23, 0.14]], [15, [0.36, 0.22, 0.13]], [17.5, [0.58, 0.44, 0.32]], [19.5, [0.85, 0.79, 0.65]], [23, [0.84, 0.77, 0.62]],
  [24.5, [0.66, 0.42, 0.24]], [28, [0.58, 0.44, 0.32]], [31, [0.80, 0.74, 0.62]], [35, [0.74, 0.69, 0.60]], [37, [0.52, 0.46, 0.40]],
  [40, [0.70, 0.66, 0.60]], [43, [0.55, 0.50, 0.46]], [47, [0.62, 0.59, 0.55]], [55, [0.50, 0.50, 0.50]], [65, [0.44, 0.45, 0.48]],
  [90, [0.36, 0.37, 0.41]]];
// how turbulent the cloud deck is at each latitude (belts and polar regions: high; zones: smooth)
const JUP_TURB = [[-90, 0.9], [-50, 0.8], [-42, 0.55], [-34, 0.35], [-31, 0.6], [-25, 0.25], [-21, 0.35], [-18, 0.9], [-12, 0.9], [-8, 0.6],
  [-4, 0.35], [0, 0.45], [4, 0.35], [7, 0.7], [12, 0.85], [17, 0.8], [20, 0.3], [23, 0.3], [26, 0.6], [30, 0.5], [34, 0.35], [38, 0.6],
  [45, 0.7], [90, 0.9]];
// Saturn: pale butterscotch bands under a thick ammonia haze; the hexagon band near 77°N, blue-grey interior, dark polar eye
const SAT_COL = [
  [-90, [0.40, 0.35, 0.28]], [-87, [0.55, 0.48, 0.37]], [-80, [0.63, 0.54, 0.40]], [-70, [0.71, 0.59, 0.41]], [-60, [0.78, 0.66, 0.46]],
  [-50, [0.69, 0.57, 0.38]], [-42, [0.80, 0.68, 0.48]], [-35, [0.66, 0.52, 0.33]], [-27, [0.84, 0.72, 0.50]], [-20, [0.68, 0.54, 0.34]],
  [-12, [0.88, 0.78, 0.56]], [0, [0.92, 0.84, 0.64]], [12, [0.88, 0.77, 0.54]], [20, [0.66, 0.52, 0.32]], [27, [0.84, 0.71, 0.49]],
  [35, [0.66, 0.53, 0.34]], [42, [0.82, 0.70, 0.49]], [50, [0.72, 0.62, 0.44]], [58, [0.70, 0.64, 0.50]], [65, [0.62, 0.61, 0.54]],
  [72, [0.58, 0.60, 0.56]], [75.5, [0.68, 0.66, 0.58]], [77, [0.46, 0.48, 0.46]], [78.5, [0.62, 0.62, 0.57]], [82, [0.46, 0.52, 0.54]],
  [87, [0.42, 0.48, 0.52]], [88.7, [0.60, 0.58, 0.52]], [90, [0.22, 0.24, 0.26]]];

/* ------------------------------------------------------------------------------------------------ */
/* shared GLSL for the four planets                                                                   */
/* ------------------------------------------------------------------------------------------------ */
const HEAD = /* glsl */`
uniform vec3 uSunDir;      // world, toward the Sun
uniform mat3 uW2B;         // world → body frame (y = spin axis, not spinning)
uniform float uFlat, uSpin, uRm, uSunI;
uniform vec3 uPh;          // two-phase wind clock: x = t1, y = t2 (s), z = weight of phase 1
uniform vec2 uSeed;        // noise seeds of the two phases
uniform vec4 uDrift;       // longitudes (rad) of drifting features
uniform float uAxis, uRingOn, uRingGain, uRingRot;
uniform vec4 uMoonA;       // Jupiter: orbital angles of the Galilean moons
#define R2D 57.2957795
#define D2R 0.0174532925
float gMu, gMu0;           // cos(view), cos(sun) at the current surface point
vec3 gSunQ;                // Sun direction in body-fixed (spinning) coordinates

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
// body-fixed unit vector at planetographic latitude (deg) and east longitude (rad)
vec3 latlon(float latD, float lonE){ float la = latD * D2R; return vec3(cos(la) * cos(lonE), sin(la), -cos(la) * sin(lonE)); }
void tframe(vec3 c, out vec3 e, out vec3 nN){ e = normalize(vec3(c.z, 0.0, -c.x) + vec3(0.0, 0.0, 1e-6)); nN = cross(c, e); }
// elliptical radius around a feature centre c (semi-axes ax, radians); l = local (east, north) coords
float vrad(vec3 q, vec3 c, vec2 ax, out vec2 l){
  vec3 e, nN; tframe(c, e, nN); vec3 d = q - c;
  l = vec2(dot(d, e) / ax.x, dot(d, nN) / ax.y);
  return dot(q, c) > 0.0 ? length(l) : 99.0;
}
// vortex advection: rotate the noise domain about c (elliptical streamlines) by ang (+ = counter-clockwise
// seen from outside) plus a static spiral twist. Returns the source point and the vortex mask.
vec4 vortex(vec3 q, vec3 c, vec2 ax, float ang, float tw, float rIn, float rOut){
  vec3 e, nN; tframe(c, e, nN); vec3 d = q - c;
  vec2 l = vec2(dot(d, e) / ax.x, dot(d, nN) / ax.y);
  float r = length(l);
  float m = smoothstep(rOut, rIn, r) * step(0.0, dot(q, c));
  float a = -(ang + tw * max(1.0 - r / rOut, 0.0)) * m;
  float cs = cos(a), sn = sin(a);
  vec2 l2 = vec2(cs * l.x - sn * l.y, sn * l.x + cs * l.y);
  return vec4(normalize(q + e * ((l2.x - l.x) * ax.x) + nN * ((l2.y - l.y) * ax.y)), m);
}
vec3 seedOff(float s){ return vec3(s * 13.7, s * 5.3, s * 9.1); }
float bump(float x, float c, float w){ float t = (x - c) / w; return exp(-t * t); }
// Henyey–Greenstein phase function normalised to mean 1 over the sphere (c = cos scattering angle)
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * c, 1.5); }
// shadow of the (oblate) planet on point p (body frame) toward the Sun s
float bodyShadow(vec3 p, vec3 s, vec3 E){
  vec3 o = p * E, d = s * E;
  float tc = -dot(o, d) / dot(d, d);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.996, 1.004, length(o + d * tc));
}
// radial value noise for ring structure, band-limited by the pixel footprint fw
float rh1(float i){ return hash13(vec3(i, 7.31, 1.97)); }
float rn1(float x){ float i = floor(x), f = fract(x); return mix(rh1(i), rh1(i + 1.0), f * f * (3.0 - 2.0 * f)); }
float rband(float r, float freq, float fw){ return (rn1(r * freq) - 0.5) * (1.0 - smoothstep(0.25, 0.8, fw * freq)); }
// narrow ring / gap of width w at r0, anti-aliased: keeps the integrated optical depth when sub-pixel
float rbox(float r, float r0, float w, float fw){ float we = max(w, fw); return (w / we) * (1.0 - smoothstep(0.35 * we, 0.65 * we, abs(r - r0))); }
`;

const MAIN = /* glsl */`
vec3 rimLight(vec3 pn, vec3 rd, vec3 s){
  float il = smoothstep(-0.3, 0.2, dot(pn, s));
  return RIM_COL * il * (0.35 + 0.65 * hg(dot(rd, s), 0.7));
}
vec3 sunDisc(vec3 rdW){
  float c = dot(rdW, uSunDir);
  return bb(5772.0) * (smoothstep(0.99997, 0.999985, c) * 60.0 + pow(max(c, 0.0), 6000.0) * 3.0 + pow(max(c, 0.0), 300.0) * 0.04);
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
  float tN = tP;
#ifdef MOONS
  int mid = -1; float tM = moonsHit(ro, rd, mid);
  tN = min(tN, tM);
#endif
  vec3 col;
  if (tP < 1e8 && tP <= tN){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = Rs * (p * E);
    float lat = asin(clamp(n.y, -1.0, 1.0));
    gMu0 = dot(n, s); gMu = max(dot(n, -rd), 0.0);
    vec3 alb = surface(q, lat);
    // Minnaert limb darkening: I ∝ μ0^k μ^(k−1)
    float lit = pow(max(gMu0, 0.0), MINN_K) * pow(max(gMu, 0.06), MINN_K - 1.0);
    lit += 0.01 * smoothstep(-0.1, 0.03, gMu0) * (1.0 - smoothstep(0.0, 0.15, gMu0));  // twilight through the upper haze
    float sh = 1.0;
#ifdef RING_SHADOW
    if (uRingOn > 0.5 && p.y * s.y < 0.0){
      vec3 pr = p - s * (p.y / s.y);
      sh = exp(-ringTau(length(pr.xz), tP * pixAng / max(abs(s.y), 0.03)) / max(abs(s.y), 0.004));
    }
#endif
#ifdef MOONS
    sh *= moonShadows(p, s);
#endif
    col = alb * lit * uSunI * sh;
    col += rimLight(n, rd, s) * uSunI * pow(1.0 - gMu, 6.0) * sh;
#ifdef RINGSHINE
    // sunlight scattered by the lit face of the rings onto the night side
    // (the shaded hemisphere gets light diffusely transmitted through the rings, weaker)
    col += alb * uSunI * uRingOn * 0.12 * sqrt(abs(s.y)) * mix(0.35, 1.0, step(0.0, p.y * s.y)) * sqrt(abs(p.y)) * (1.0 - smoothstep(-0.05, 0.12, gMu0));
#endif
  }
#ifdef MOONS
  else if (tM < 1e8){ col = moonShade(mid, ro + rd * tM, rd, s, E); }
#endif
  else {
    col = sky(rdW, 1.0) * 0.6 + sunDisc(rdW);
#ifdef MOONS
    col += moonGlow(ro, rd, pixAng);
#endif
    float tc = -B / A;
    if (tc > 0.0){
      vec3 pc = roE + rdE * tc;
      float hgt = max(length(pc) - 1.0, 0.0);
      col += rimLight(normalize(pc), rd, s) * uSunI * exp(-hgt / RIM_H);
    }
  }
#ifdef RINGS
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
        ringProfile(r, atan(pr.x, pr.z), fw, tau, alb, dust);
        if (tau > 0.0){
          float shd = bodyShadow(pr, s, E);
          float c = dot(s, rd);                    // cos(scattering angle): 1 = looking toward the Sun
          float P = mix(hg(c, -0.3), hg(c, 0.7), dust);   // big icy particles backscatter, dust forward-scatters
          float IF;
          if (s.y * rd.y < 0.0) IF = 0.25 * P * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu))) * (1.0 + 1.6 * (1.0 - exp(-tau)));   // lit face (+ multiple scattering in dense rings, approx.)
          else {                                                                                              // unlit face: diffuse transmission
            float dm = mu - mu0;
            IF = abs(dm) > 1e-3 ? 0.25 * P * mu0 / dm * (exp(-tau / mu) - exp(-tau / mu0)) : 0.25 * P * mu0 * tau / (mu * mu) * exp(-tau / mu);
          }
          col = col * exp(-tau / mu) + alb * max(IF, 0.0) * uSunI * uRingGain * shd;
        }
      }
    }
  }
#endif
  if (uAxis > 0.5){
    // closest approach between the camera ray and the spin axis
    float b = rd.y, dd = dot(rd, ro), den = 1.0 - b * b;
    if (den > 1e-6){
      float sc = (b * ro.y - dd) / den, ta = ro.y + b * sc;
      if (sc > 0.0 && sc < tN){
        float dist = length(ro + rd * sc - vec3(0.0, ta, 0.0));
        float line = exp(-pow(dist / (1.4 * sc * pixAng), 2.0)) * step(0.85, abs(ta)) * (1.0 - smoothstep(AXIS_LEN - 0.25, AXIS_LEN, abs(ta)));
        col += (ta > 0.0 ? vec3(1.0, 0.42, 0.22) : vec3(0.3, 0.6, 1.0)) * line * 0.7;
      }
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ------------------------------------------------------------------------------------------------ */
/* Jupiter                                                                                            */
/* ------------------------------------------------------------------------------------------------ */
const JUPITER_GLSL = /* glsl */`
#define MINN_K 0.95
#define RIM_COL vec3(0.11, 0.1, 0.09)
#define RIM_H 0.015
#define AXIS_LEN 1.7
#define MOONS
#define RINGS
#define RING_IN 1.29
#define RING_OUT 3.2
${tab1('jWind', JUP_WIND)}
${tab3('jCol', JUP_COL)}
${tab1('jTurb', JUP_TURB)}
const vec3 GRS_C = vec3(0.927184, -0.374607, 0.0);   // 22°S, System III longitude 0
const vec2 GRS_AX = vec2(0.114, 0.077);               // 16 350 × 11 000 km (≈ 1.3 Earths wide)

vec3 jAdvect(vec3 q, float lat, float t){
  float om = jWind(lat * R2D) / (uRm * max(cos(lat), 0.08));
  vec3 p = rotY(-om * t) * q;
  // Great Red Spot: anticyclone, counter-clockwise in the south; collar turns in ~4.5 days
  vec4 v = vortex(q, GRS_C, GRS_AX, 8e-6 * t, 0.6, 0.55, 1.5);
  p = normalize(mix(p, v.xyz, v.w));
  if (lat < -0.45 && lat > -0.85){
    v = vortex(q, latlon(-33.0, uDrift.z), vec2(0.055, 0.04), 2.5e-5 * t, 1.6, 0.5, 1.6);   // Oval BA
    p = normalize(mix(p, v.xyz, v.w));
    for (int i = 0; i < 6; i++){   // 'string of pearls' anticyclones near 40°S
      v = vortex(q, latlon(-40.5, uDrift.y + float(i) * 0.62), vec2(0.03, 0.022), 3e-5 * t, 1.2, 0.5, 1.6);
      p = normalize(mix(p, v.xyz, v.w));
    }
  }
  if (abs(lat) > 1.25){
    // Juno: circumpolar cyclones — 8 around one at the north pole, 5 around one at the south pole
    float sg = sign(lat), nn = sg > 0.0 ? 8.0 : 5.0;
    for (int i = 0; i < 9; i++){
      if (float(i) > nn) break;
      float clat = i == 0 ? 89.4 : (sg > 0.0 ? 83.0 : 84.0);
      vec3 c = latlon(sg * clat, float(i) * TAU / nn + 0.4 + 0.1 * sin(float(i) * 3.1));
      v = vortex(q, c, vec2(sg > 0.0 ? 0.032 : 0.042), sg * 1.0e-5 * t, sg * 3.5, 0.3, 1.5);
      p = normalize(mix(p, v.xyz, v.w));
    }
  }
  return p;
}
vec3 jField(vec3 p, float seed, float lat){
  float st = mix(3.4, 1.0, smoothstep(0.65, 0.95, abs(lat)));    // features stretched east–west by the shear
  vec3 Q = vec3(p.x, p.y * st, p.z) + seedOff(seed);
  float a = fbm(Q * 2.4, 4);
  float b = fbm(Q * 6.5 + vec3(a * 2.2, a * 0.6, -a * 2.2), 4);
  float c = fbm(Q * 18.0 + vec3(b * 1.6), 4);
  return vec3(a, b, c);
}
vec3 surface(vec3 q, float lat){
  float w1 = uPh.z, w2 = 1.0 - w1;
  vec3 f = (jField(jAdvect(q, lat, uPh.x), uSeed.x, lat) * w1 + jField(jAdvect(q, lat, uPh.y), uSeed.y, lat) * w2) * inversesqrt(w1 * w1 + w2 * w2);
  float latD = lat * R2D;
  float lonE = atan(-q.z, q.x);
  vec2 lg; float rg = vrad(q, GRS_C, GRS_AX, lg);
  float wake = smoothstep(-0.6, -2.0, lg.x) * (1.0 - smoothstep(-6.0, -11.0, lg.x)) * bump(latD, -15.5, 5.0);   // turbulent SEB wake west of the GRS
  float lw = latD + 2.3 * f.x + 1.0 * f.y;
  vec3 alb = jCol(lw) * vec3(1.0, 0.95, 0.86);
  float tb = jTurb(lw) + 0.7 * wake;
  float det = 0.6 * f.y + 0.4 * f.z;
  alb *= 1.0 + tb * det * 1.2;
  alb = mix(alb, vec3(0.92, 0.88, 0.80), clamp(tb * smoothstep(0.22, 0.55, det), 0.0, 1.0) * 0.5);   // convective white plumes
  alb = mix(alb, vec3(0.32, 0.33, 0.38), clamp(tb * smoothstep(-0.22, -0.5, det), 0.0, 1.0) * 0.45);  // dark blue-grey gaps
  // NEB hot spots + festoons: ~11 dark 5-µm windows on the NEB's southern edge with blue-grey plumes trailing
  // south-west into the Equatorial Zone; they drift as one wave pattern (≈ 103 m/s)
  {
    float ph = (lonE - uDrift.x) * 11.0 / TAU;
    float s = clamp((6.8 - latD) / 7.5, 0.0, 1.0);
    float cx = fract(ph + 0.45 * s + 0.3 * s * s + 0.05 * f.y) - 0.5;
    float cw = cx + 0.04 * f.z * s;
    float plume = exp(-cw * cw / (0.0015 + 0.004 * s)) * (1.0 - smoothstep(0.5, 0.95, s)) * smoothstep(7.5, 6.0, latD) * (0.8 + 0.6 * f.z);
    float hot = exp(-cx * cx / 0.0025) * bump(latD + 0.6 * f.y, 6.2, 1.0);
    alb = mix(alb, vec3(0.26, 0.29, 0.37), clamp(1.0 * hot + 1.0 * plume, 0.0, 0.9));
    float wp = exp(-pow(cx - 0.24, 2.0) / 0.01) * bump(latD, 3.5, 2.0);
    alb = mix(alb, vec3(0.95, 0.93, 0.88), 0.5 * wp);
  }
  // Great Red Spot: pale orange core, brick-red annulus, bright 'Red Spot Hollow' around it
  {
    float r = rg + 0.06 * f.y;
    alb = mix(alb, vec3(0.90, 0.86, 0.78), 0.6 * smoothstep(1.6, 1.15, r) * smoothstep(0.9, 1.1, r));
    vec3 grs = mix(vec3(0.82, 0.52, 0.36), vec3(0.72, 0.31, 0.17), smoothstep(0.15, 0.65, r + 0.25 * f.z));
    grs = mix(grs, vec3(0.56, 0.24, 0.14), 0.65 * smoothstep(0.68, 0.95, r));
    grs *= 1.0 + 0.7 * det;
    alb = mix(alb, grs, smoothstep(1.04, 0.92, r));
  }
  if (lat < -0.45 && lat > -0.85){
    vec2 l;
    float r = vrad(q, latlon(-33.0, uDrift.z), vec2(0.055, 0.04), l) + 0.05 * f.y;   // Oval BA ('Red Spot Jr.')
    alb = mix(alb, mix(vec3(0.90, 0.86, 0.80), vec3(0.76, 0.48, 0.36), smoothstep(0.3, 0.85, r)) * (0.92 + 0.3 * det), smoothstep(1.0, 0.82, r));
    for (int i = 0; i < 6; i++){
      r = vrad(q, latlon(-40.5, uDrift.y + float(i) * 0.62), vec2(0.026 + 0.006 * sin(float(i) * 2.3), 0.019), l) + 0.12 * f.z;
      alb *= 1.0 - 0.2 * smoothstep(1.4, 1.05, r) * smoothstep(0.8, 1.05, r);
      alb = mix(alb, vec3(0.93, 0.92, 0.88) * (0.9 + 0.25 * det), smoothstep(1.0, 0.7, r));
    }
  }
  if (lat > 0.2 && lat < 0.35){
    vec2 l;
    for (int i = 0; i < 4; i++){   // dark-red 'barges' (cyclones) on the NEB's northern edge
      float r = vrad(q, latlon(15.5, uDrift.w + float(i) * 1.55), vec2(0.045, 0.016), l) + 0.08 * f.z;
      alb = mix(alb, vec3(0.30, 0.17, 0.11), 0.85 * smoothstep(1.0, 0.7, r));
    }
  }
  if (abs(lat) > 1.3){
    // polar cyclones: alternate pale cores and dark spiral lanes
    float sg = sign(lat), nn = sg > 0.0 ? 8.0 : 5.0;
    for (int i = 0; i < 9; i++){
      if (float(i) > nn) break;
      float clat = i == 0 ? 89.4 : (sg > 0.0 ? 83.0 : 84.0);
      vec2 l; float r = vrad(q, latlon(sg * clat, float(i) * TAU / nn + 0.4 + 0.1 * sin(float(i) * 3.1)), vec2(sg > 0.0 ? 0.032 : 0.042), l);
      float m = smoothstep(1.25, 0.5, r);
      float arms = sin(atan(l.y, l.x) * 2.0 - sg * 5.0 * log(max(r, 0.05)) + 4.0 * det);   // logarithmic spiral bands
      vec3 cc = mod(float(i), 2.0) < 0.5 ? vec3(0.80, 0.76, 0.66) : vec3(0.62, 0.60, 0.60);
      alb = mix(alb, cc * (0.8 + 0.35 * arms), m * 0.6);
      alb *= 1.0 - 0.45 * smoothstep(0.25, 0.0, r);
    }
  }
  return alb;
}

const vec4 MOON_ORB = vec4(5.905, 9.397, 14.99, 26.37);         // semi-major axes, R_J
const vec4 MOON_RAD = vec4(0.02548, 0.02183, 0.03685, 0.03371); // radii, R_J
vec3 moonPos(int i){ float a = uMoonA[i]; return MOON_ORB[i] * vec3(sin(a), 0.0, cos(a)); }
float moonsHit(vec3 ro, vec3 rd, out int id){
  float tb = 1e9; id = -1;
  for (int i = 0; i < 4; i++){
    vec2 h = sphereHit(ro, rd, moonPos(i), MOON_RAD[i]);
    if (h.x < h.y && h.x > 0.0 && h.x < tb){ tb = h.x; id = i; }
  }
  return tb;
}
// sub-pixel moons still show as points of light
vec3 moonGlow(vec3 ro, vec3 rd, float pixAng){
  vec3 g = vec3(0.0);
  for (int i = 0; i < 4; i++){
    vec3 v = moonPos(i) - ro; float t = dot(v, rd);
    if (t <= 0.0) continue;
    float d = length(v - rd * t), px = t * pixAng;
    float ph = 0.5 + 0.5 * dot(normalize(ro - moonPos(i)), uW2B * uSunDir);   // lit fraction seen from the camera
    g += vec3(0.75, 0.7, 0.6) * uSunI * 3.0 * ph * (1.0 - smoothstep(0.4, 1.2, MOON_RAD[i] / px)) * exp(-d * d / (1.5 * px * px));
  }
  return g;
}
// moon shadows on the cloud tops: umbra + penumbra from the Sun's 0.051° radius at 5.2 AU
float moonShadows(vec3 p, vec3 s){
  float sh = 1.0;
  for (int i = 0; i < 4; i++){
    vec3 v = moonPos(i) - p; float tca = dot(v, s);
    if (tca > 0.0){
      float d = length(v - s * tca), pen = tca * 0.00089 + 0.0012;
      sh *= mix(0.03, 1.0, smoothstep(MOON_RAD[i] - pen, MOON_RAD[i] + pen, d));
    }
  }
  return sh;
}
vec3 moonShade(int i, vec3 p, vec3 rd, vec3 s, vec3 E){
  vec3 n = normalize(p - moonPos(i));
  vec3 m = rotY(-uMoonA[i]) * n;      // tidally locked
  float n1 = fbm(m * 3.0 + float(i) * 7.0, 4), n2 = fbm(m * 9.0 + float(i) * 3.0, 3);
  vec3 alb;
  if (i == 0){        // Io: sulphur yellows, red-brown and black volcanic spots, white SO2 frost
    alb = mix(vec3(0.82, 0.74, 0.34), vec3(0.93, 0.90, 0.72), smoothstep(0.0, 0.5, n1));
    alb = mix(alb, vec3(0.55, 0.24, 0.10), smoothstep(0.35, 0.6, n2) * 0.8);
    alb = mix(alb, vec3(0.06, 0.05, 0.04), smoothstep(0.55, 0.75, n2));
  } else if (i == 1){ // Europa: bright water ice crossed by brown lineae
    alb = mix(vec3(0.86, 0.82, 0.74), vec3(0.62, 0.48, 0.34), smoothstep(0.1, 0.5, n1) * 0.6);
    alb = mix(alb, vec3(0.50, 0.33, 0.22), (1.0 - smoothstep(0.0, 0.06, abs(n2))) * 0.6);
  } else if (i == 2){ // Ganymede: dark old terrain, bright grooved terrain
    alb = mix(vec3(0.30, 0.27, 0.24), vec3(0.62, 0.60, 0.56), smoothstep(-0.1, 0.2, n1));
  } else {            // Callisto: dark, saturated with bright crater rays
    alb = vec3(0.22, 0.20, 0.18) + vec3(0.35) * smoothstep(0.45, 0.7, n2);
  }
  float mu0 = max(dot(n, s), 0.0);
  return alb * mu0 * uSunI * 1.2 * bodyShadow(p, s, E) + alb * 0.002;
}
// Jupiter's faint dusty rings (halo, main ring, two gossamer rings). Real optical depth ~1e-6; drawn ×1000 —
// visible only toward the Sun because micron dust scatters forward.
void ringProfile(float r, float az, float fw, out float tau, out vec3 alb, out float dust){
  dust = 1.0; alb = vec3(0.75, 0.62, 0.50);
  tau = 0.25 * smoothstep(1.29, 1.40, r) * (1.0 - smoothstep(1.66, 1.72, r))
      + 3.0 * smoothstep(1.72, 1.75, r) * (1.0 - smoothstep(1.795, 1.806, r))
      + 0.15 * step(1.806, r) * (1.0 - smoothstep(2.45, 2.55, r))
      + 0.06 * step(1.806, r) * (1.0 - smoothstep(3.05, 3.15, r));
  tau *= 1e-3 * (1.0 + 2.0 * rband(r, 40.0, fw));
}
`;

/* ------------------------------------------------------------------------------------------------ */
/* Saturn                                                                                             */
/* ------------------------------------------------------------------------------------------------ */
const SATURN_GLSL = /* glsl */`
#define MINN_K 0.9
#define RIM_COL vec3(0.13, 0.115, 0.09)
#define RIM_H 0.02
#define AXIS_LEN 1.6
#define RINGS
#define RING_SHADOW
#define RINGSHINE
#define RING_IN 1.11
#define RING_OUT 2.36
${tab1('sWind', SAT_WIND)}
${tab3('sCol', SAT_COL)}
vec2 sField(vec3 p, float seed, float lat){
  float st = mix(4.5, 1.5, smoothstep(1.1, 1.35, abs(lat)));
  vec3 Q = vec3(p.x, p.y * st, p.z) + seedOff(seed);
  float a = fbm(Q * 2.0, 4);
  float b = fbm(Q * 5.0 + vec3(a * 2.2, a * 0.4, -a * 2.2), 4);
  return vec2(a, b);
}
vec3 sAdvect(vec3 q, float lat, float t){ return rotY(-sWind(lat * R2D) / (uRm * max(cos(lat), 0.08)) * t) * q; }
vec3 surface(vec3 q, float lat){
  float w1 = uPh.z, w2 = 1.0 - w1;
  vec2 f = (sField(sAdvect(q, lat, uPh.x), uSeed.x, lat) * w1 + sField(sAdvect(q, lat, uPh.y), uSeed.y, lat) * w2) * inversesqrt(w1 * w1 + w2 * w2);
  float latD = lat * R2D, le = latD;
  if (latD > 60.0){
    // the north-polar hexagon: a meandering jet locked into a 6-lobed standing wave (rotates with the interior)
    float ang = atan(q.z, q.x);
    float sect = mod(ang + 0.3, PI / 3.0) - PI / 6.0;
    float rho = 90.0 - latD;
    float wh = 1.0 - smoothstep(14.5, 22.0, rho);
    le = 90.0 - rho * mix(1.0, cos(sect) * 1.035, wh);
  }
  le += 1.1 * f.x + 0.5 * f.y;
  vec3 alb = sCol(le);
  alb *= 1.0 + 0.07 * f.y + 0.06 * f.x;
  // a few bright convective storms in 'storm alley' (35°S)
  alb = mix(alb, vec3(0.95, 0.92, 0.84), 0.6 * bump(latD, -35.0, 1.5) * smoothstep(0.42, 0.6, f.y));
  return mix(alb, vec3(0.86, 0.77, 0.58), 0.1);   // haze veil
}
// Radial structure (Cassini UVIS/VIMS occultations, simplified). r in Saturn radii (60 268 km).
void ringProfile(float r, float az, float fw, out float tau, out vec3 alb, out float dust){
  float n = 0.5 * rband(r, 25.0, fw) + 0.45 * rband(r, 70.0, fw) + 0.35 * rband(r, 180.0, fw) + 0.3 * rband(r, 600.0, fw) + 0.25 * rband(r, 2000.0, fw);
  tau = 0.0; dust = 0.0; alb = vec3(0.6);
  if (r < 1.239){                 // D ring: faint dust
    tau = 0.002 * smoothstep(1.11, 1.2, r) * (1.0 + n); dust = 0.9; alb = vec3(0.6, 0.57, 0.54);
  } else if (r < 1.527){          // C ring: dark, greyer, τ ~ 0.1 with dense ringlet 'plateaus'
    float x = (r - 1.239) / 0.288;
    tau = 0.06 + 0.05 * x + 0.07 * n;
    tau += 0.3 * (rbox(r, 1.3725, 0.006, fw) + rbox(r, 1.4045, 0.005, fw) + rbox(r, 1.4285, 0.008, fw)
                + rbox(r, 1.4735, 0.007, fw) + rbox(r, 1.4905, 0.005, fw) + rbox(r, 1.5035, 0.004, fw));
    tau *= 1.0 - rbox(r, 1.2905, 0.0016, fw) - rbox(r, 1.4525, 0.0045, fw);   // Colombo & Maxwell gaps
    tau += 0.3 * smoothstep(1.50, 1.527, r);
    alb = vec3(0.50, 0.46, 0.42) * (1.0 + 0.4 * n); dust = 0.2;
  } else if (r < 1.951){          // B ring: densest, nearly opaque core (τ up to ~5)
    float x = (r - 1.527) / 0.424;
    tau = 1.0 + 0.8 * smoothstep(0.0, 0.25, x) + 2.8 * smoothstep(0.3, 0.45, x) * (1.0 - smoothstep(0.82, 0.93, x));
    tau *= max(1.0 + 1.0 * n, 0.15) * (1.0 - smoothstep(1.951 - fw, 1.951, r));   // sharp outer edge (anti-aliased)
    alb = vec3(0.82, 0.71, 0.56) * (1.0 + 0.45 * n); dust = 0.02;
  } else if (r < 2.025){          // Cassini division (Mimas 2:1 resonance clears the B ring's edge)
    float x = (r - 1.951) / 0.074;
    tau = (0.08 + 0.12 * smoothstep(0.7, 1.0, x) + 0.05 * n) * smoothstep(0.12, 0.16, x);   // Huygens gap
    tau += 0.6 * rbox(r, 1.9549, 0.0004, fw);                                               // Huygens ringlet
    alb = vec3(0.56, 0.54, 0.51); dust = 0.25;
  } else if (r < 2.268){          // A ring, with the Encke (Pan) and Keeler (Daphnis) gaps
    float x = (r - 2.025) / 0.243;
    tau = mix(0.7, 0.45, x) * max(1.0 + 0.5 * n, 0.2);
    tau *= 1.0 - rbox(r, 2.2138, 0.0054, fw);
    tau += 0.03 * rbox(r, 2.2138, 0.0008, fw);
    tau *= 1.0 - rbox(r, 2.2647, 0.0007, fw);
    tau *= 1.0 - smoothstep(2.268 - fw, 2.268, r);
    alb = vec3(0.74, 0.68, 0.60) * (1.0 + 0.3 * n); dust = mix(0.03, 0.15, smoothstep(2.15, 2.268, r));
  } else {                        // F ring: narrow, clumpy, braided by Prometheus and Pandora
    float a = az - uRingRot;
    float clump = 0.35 + 1.3 * rn1(a * 40.0 + 50.0) * rn1(a * 7.0 + 3.0);
    float rr = r - 0.0012 * sin(a * 3.0 + 1.0);
    tau = 0.5 * rbox(rr, 2.3256, 0.0012, fw) * clump + 0.03 * rbox(rr, 2.3256, 0.009, fw);
    alb = vec3(0.72, 0.70, 0.68); dust = 0.95;
  }
  tau = max(tau, 0.0);
}
float ringTau(float r, float fw){ float t, d; vec3 a; ringProfile(r, 0.0, fw, t, a, d); return t; }
`;

/* ------------------------------------------------------------------------------------------------ */
/* ice giants                                                                                         */
/* ------------------------------------------------------------------------------------------------ */
const ICE_GLSL = /* glsl */`
// Reflectance of an ice-giant atmosphere (simplified radiative transfer): a cloud deck under a column of
// H2/He + CH4. Methane absorbs red (bands at 619, 727, 890 nm), Rayleigh scattering adds blue and a
// haze layer whitens the light. Longer slant paths near the limb absorb more red and add more haze.
vec3 iceColour(float deck, float Ncol, float haze, float cloud){
  float L = 0.5 * (1.0 / max(gMu, 0.1) + 1.0 / max(gMu0, 0.1));
  vec3 Tm = exp(-vec3(1.9, 0.42, 0.06) * Ncol * L);
  vec3 ray = vec3(0.025, 0.065, 0.19) * (1.0 - exp(-0.8 * L));
  vec3 hz = vec3(0.07, 0.075, 0.08) * haze * min(L, 4.0);
  return mix(deck * Tm + ray + hz, vec3(0.93, 0.95, 1.0), cloud);
}
`;
const NEPTUNE_GLSL = /* glsl */`
#define MINN_K 0.92
#define RIM_COL vec3(0.05, 0.08, 0.16)
#define RIM_H 0.015
#define AXIS_LEN 1.7
#define RINGS
#define RING_IN 1.6
#define RING_OUT 2.6
${ICE_GLSL}
float nWind(float d){ float d2 = d * d; return -398.0 + 0.188 * d2 - 1.2e-5 * d2 * d2; }
const vec2 GDS_AX = vec2(0.262, 0.133);      // Great Dark Spot, 13 000 × 6 600 km (Voyager 2, 1989)
vec3 nAdvect(vec3 q, float lat, float t){
  vec3 p = rotY(-nWind(lat * R2D) / (uRm * max(cos(lat), 0.08)) * t) * q;
  vec4 v = vortex(q, latlon(-20.0, uDrift.x), GDS_AX, 1.2e-5 * t, 1.8, 0.5, 1.4);   // anticyclone: CCW in the south
  return normalize(mix(p, v.xyz, v.w));
}
float nStreak(vec3 p, float seed){ return fbm(vec3(p.x * 4.0, p.y * 70.0, p.z * 4.0) + seedOff(seed) * 0.5, 3); }
vec3 nField(vec3 p, float seed){
  vec3 Q = vec3(p.x, p.y * 3.0, p.z) + seedOff(seed);
  float a = fbm(Q * 2.0, 4);
  float c = fbm(Q * 7.0 + vec3(a * 1.5), 3);
  return vec3(a, nStreak(p + vec3(a * 0.03), seed), c);
}
float streakBand(float d){ return smoothstep(22.0, 26.0, abs(d)) * (1.0 - smoothstep(33.0, 40.0, abs(d))) + 0.5 * bump(d, -12.0, 4.0) + 0.8 * bump(d, -69.0, 3.5); }
vec3 surface(vec3 q, float lat){
  float w1 = uPh.z, w2 = 1.0 - w1, wn = inversesqrt(w1 * w1 + w2 * w2);
  vec3 pa = nAdvect(q, lat, uPh.x), pb = nAdvect(q, lat, uPh.y);
  vec3 f = (nField(pa, uSeed.x) * w1 + nField(pb, uSeed.y) * w2) * wn;
  float latD = lat * R2D, lw = latD + 4.0 * f.x;
  float deck = 0.8 * (1.0 + 0.07 * sin(lw * 0.21) - 0.16 * bump(lw, -57.0, 8.0) + 0.1 * bump(lw, -78.0, 6.0) + 0.12 * f.z);
  float cloud = 0.0;
  vec2 l;
  // Great Dark Spot: a hole in the methane-haze layer, with bright orographic 'companion' clouds on its southern rim
  float r = vrad(q, latlon(-20.0, uDrift.x), GDS_AX, l) + 0.08 * f.z;
  deck *= mix(1.0, 0.4, smoothstep(1.05, 0.8, r));
  deck *= 1.0 - 0.2 * smoothstep(0.75, 0.95, r) * smoothstep(1.2, 0.95, r);
  cloud = max(cloud, 0.9 * smoothstep(1.4, 1.08, r) * smoothstep(0.92, 1.1, r) * smoothstep(-0.2, -0.75, l.y) * smoothstep(-0.1, 0.3, f.y + 0.1 * l.x));
  // 'Scooter' (42°S) and Dark Spot 2 (55°S) with its bright core
  r = vrad(q, latlon(-42.0, uDrift.y), vec2(0.05, 0.03), l) + 0.25 * f.z;
  cloud = max(cloud, 0.95 * smoothstep(1.0, 0.6, r));
  r = vrad(q, latlon(-55.0, uDrift.z), vec2(0.075, 0.045), l) + 0.1 * f.z;
  deck *= mix(1.0, 0.5, smoothstep(1.0, 0.75, r));
  cloud = max(cloud, 0.85 * smoothstep(0.35, 0.15, r));
  // high methane-ice cirrus streaks (~50–100 km above the deck); near the terminator they cast shadows
  float sb = streakBand(latD);
  float stk = sb * smoothstep(0.38, 0.66, f.y);
  cloud = max(cloud, 0.9 * stk);
  if (sb > 0.01 && gMu0 > 0.0 && gMu0 < 0.6){
    vec3 tS = normalize(gSunQ - q * dot(gSunQ, q) + 1e-6);
    vec3 q2 = normalize(q + tS * (0.004 / max(gMu0, 0.06)));
    float s2 = (nStreak(nAdvect(q2, lat, uPh.x), uSeed.x) * w1 + nStreak(nAdvect(q2, lat, uPh.y), uSeed.y) * w2) * wn;
    deck *= 1.0 - 0.6 * sb * smoothstep(0.38, 0.66, s2) * (1.0 - stk);
  }
  return iceColour(deck, 2.3, 0.3, cloud);
}
// Neptune's dusty rings (radius in R_N = 24 764 km). Optical depths 1e-4–0.1; brightness boosted by uRingGain.
void ringProfile(float r, float az, float fw, out float tau, out vec3 alb, out float dust){
  float a = mod(az - uRingRot, TAU) * R2D;   // longitude in the frame co-moving with the Adams arcs
  float arcs = smoothstep(0.0, 1.0, a) * (1.0 - smoothstep(9.0, 10.0, a))        // Fraternité
             + smoothstep(13.0, 13.5, a) * (1.0 - smoothstep(14.0, 14.5, a))     // Égalité 1
             + smoothstep(15.0, 15.5, a) * (1.0 - smoothstep(18.0, 18.5, a))     // Égalité 2
             + smoothstep(24.5, 25.0, a) * (1.0 - smoothstep(28.5, 29.0, a))     // Liberté
             + smoothstep(33.0, 33.3, a) * (1.0 - smoothstep(33.8, 34.1, a));    // Courage
  tau = 1e-4 * rbox(r, 1.692, 0.08, fw) + 0.002 * rbox(r, 2.148, 0.0046, fw) + 1.5e-4 * smoothstep(2.15, 2.16, r) * (1.0 - smoothstep(2.30, 2.31, r))
      + 5e-4 * rbox(r, 2.31, 0.004, fw) + (0.003 + 0.08 * arcs) * rbox(r, 2.541, 0.0007, fw);
  tau *= 10.0;
  alb = vec3(0.07, 0.06, 0.055); dust = 0.6;
}
`;
const URANUS_GLSL = /* glsl */`
#define MINN_K 0.95
#define RIM_COL vec3(0.06, 0.1, 0.13)
#define RIM_H 0.015
#define AXIS_LEN 2.0
#define RINGS
#define RING_IN 1.6
#define RING_OUT 3.95
${ICE_GLSL}
${tab1('uWind', URA_WIND)}
vec3 uField(vec3 p, float seed){
  vec3 Q = vec3(p.x, p.y * 5.0, p.z) + seedOff(seed);
  float a = fbm(Q * 1.8, 4);
  float b = fbm(Q * 6.0 + vec3(a * 1.4), 3);
  return vec3(a, b, 0.0);
}
vec3 surface(vec3 q, float lat){
  float w1 = uPh.z, w2 = 1.0 - w1;
  vec3 pa = rotY(-uWind(lat * R2D) / (uRm * max(cos(lat), 0.08)) * uPh.x) * q;
  vec3 pb = rotY(-uWind(lat * R2D) / (uRm * max(cos(lat), 0.08)) * uPh.y) * q;
  vec3 f = (uField(pa, uSeed.x) * w1 + uField(pb, uSeed.y) * w2) * inversesqrt(w1 * w1 + w2 * w2);
  float latD = lat * R2D, lw = latD + 2.5 * f.x;
  // almost featureless: faint bands, the bright north-polar 'hood' of the 2020s and a southern collar
  float deck = 0.84 * (1.0 + 0.035 * sin(lw * 0.22) + 0.14 * smoothstep(48.0, 62.0, lw) + 0.05 * bump(lw, 60.0, 4.0)
                       + 0.05 * bump(lw, -45.0, 4.0) + 0.03 * f.y);
  float cloud = 0.35 * bump(latD, 32.0, 4.0) * smoothstep(0.45, 0.7, f.y)   // occasional bright convective storms
              + 0.3 * smoothstep(52.0, 66.0, lw);                              // north-polar hood (haze/ice, 2020s)
  return iceColour(deck, 0.95, 0.9, cloud);
}
// Uranus' rings (radius in R_U = 25 559 km): nine narrow dark rings + ε, and the dusty ν and μ rings.
// Ring particles are as dark as charcoal (albedo ~0.03); uRingGain brightens them for display.
void ringProfile(float r, float az, float fw, out float tau, out vec3 alb, out float dust){
  float ca = cos(az - uRingRot);
  float rE = 2.0012 * (1.0 - 0.0079 * ca), wE = (58.0 - 38.0 * ca) / 25559.0;   // ε ring: eccentric, 20–96 km wide
  float narrow = 0.3 * rbox(r, 1.6369, 0.00006, fw) + 0.5 * rbox(r, 1.6524, 0.00008, fw) + 0.3 * rbox(r, 1.6656, 0.00008, fw)   // 6, 5, 4
               + 0.4 * rbox(r, 1.7496, 0.0004, fw) + 0.3 * rbox(r, 1.7865, 0.0003, fw)                                          // α, β
               + 0.4 * rbox(r, 1.8458, 0.00006, fw) + 1.5 * rbox(r, 1.8634, 0.00012, fw) + 0.5 * rbox(r, 1.8898, 0.0002, fw)    // η, γ, δ
               + 0.1 * rbox(r, 1.9572, 0.00008, fw) + 1.5 * rbox(r, rE, wE, fw);                                                // λ, ε
  float dusty = 2e-4 * rbox(r, 2.633, 0.15, fw) + 2e-4 * rbox(r, 3.82, 0.66, fw);   // ν and μ (×100)
  tau = narrow + dusty;
  dust = dusty / max(tau, 1e-9);
  alb = mix(vec3(0.05, 0.05, 0.05), r > 3.0 ? vec3(0.25, 0.45, 0.9) : vec3(0.35, 0.3, 0.27), dust);
}
`;

/* ------------------------------------------------------------------------------------------------ */
/* brown dwarf                                                                                        */
/* ------------------------------------------------------------------------------------------------ */
const BD_FRAG = COMMON_GLSL + HEAD + /* glsl */`
uniform float uTeff, uMode, uClouds, uGlow;
uniform vec3 uEye, uEyeCool; // chromaticity of the alkali-absorbed spectrum at Teff and at 0.85 Teff (luminance 1)
uniform vec3 uCompDir, uCompCol;
float bdWind(float lat){ return 650.0 * cos(lat) * cos(lat) * cos(5.0 * lat); }
vec3 bdField(vec3 p, float seed){
  vec3 Q = vec3(p.x, p.y * 2.4, p.z) + seedOff(seed);
  float a = fbm(Q * 1.7, 4);
  float b = fbm(Q * 4.5 + vec3(a * 1.8, 0.0, -a * 1.8), 4);
  float c = fbm(Q * 13.0 + vec3(b * 1.2), 3);
  return vec3(a, b, c);
}
void main(){
  vec3 rdW = camRay(), roW = uCamPos;
  vec3 ro = uW2B * roW, rd = uW2B * rdW;
  vec3 E = vec3(1.0, 1.0 / (1.0 - uFlat), 1.0);
  vec3 roE = ro * E, rdE = rd * E;
  float A = dot(rdE, rdE), B = dot(roE, rdE), C = dot(roE, roE) - 1.0, H = B * B - A * C;
  vec3 col;
  float tP = H > 0.0 ? (-B - sqrt(H)) / A : -1.0;
  if (tP > 0.0){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = rotY(-uSpin) * (p * E);
    float lat = asin(clamp(n.y, -1.0, 1.0)), lonE = atan(-q.z, q.x);
    float mu = max(dot(n, -rd), 0.0);
    float w1 = uPh.z, w2 = 1.0 - w1;
    float om = bdWind(lat * R2D * D2R) / (uRm * max(cos(lat), 0.08));
    vec3 f = (bdField(rotY(-om * uPh.x) * q, uSeed.x) * w1 + bdField(rotY(-om * uPh.y) * q, uSeed.y) * w2) * inversesqrt(w1 * w1 + w2 * w2);
    float latD = lat * R2D;
    // cloud thickness: zonal bands modulated by planetary-scale waves (Apai+ 2017) plus patchy clouds (Doppler map, Crossfield+ 2014)
    float bands = sin(latD * 0.13 + 1.1 * f.x + 0.7 * sin(lonE * 2.0 + latD * 0.05));
    float cl = 0.5 + 0.42 * bands + 0.5 * f.y + 0.25 * f.z;
    cl = mix(0.5, cl, uClouds);
    // thin clouds reveal deeper, hotter layers
    float Tloc = uTeff * (1.0 + 0.065 * (0.5 - cl));
    // grey (Eddington) atmosphere: T(τ)^4 = ¾ Teff^4 (τ + ⅔); at slant μ we see down to τ ≈ μ
    float Tb = Tloc * pow(0.75 * (mu + 2.0 / 3.0), 0.25);
    if (uMode < 0.5){                 // what an eye would see: Wien tail at ~0.75 µm × alkali-absorbed spectrum
      float I = exp(19184.0 * (1.0 / uTeff - 1.0 / Tb));
      col = mix(uEye, uEyeCool, clamp((uTeff - Tb) / (0.15 * uTeff), 0.0, 1.0)) * I * uGlow;
    } else if (uMode < 1.5){          // J band (1.25 µm): the window where cloud holes stand out (false colour)
      float I = exp(11510.0 * (1.0 / uTeff - 1.0 / Tb)) * 0.55;
      col = mix(vec3(0.25, 0.05, 0.02), vec3(1.0, 0.55, 0.18), smoothstep(0.1, 0.8, I)) * I + vec3(1.0, 0.9, 0.7) * smoothstep(0.8, 1.6, I) * 0.6;
    } else {                          // a pure blackbody of the same temperature (no molecular / alkali absorption)
      float I = exp(19184.0 * (1.0 / uTeff - 1.0 / Tb));
      col = bb(Tb) * I * uGlow;
    }
  } else {
    col = sky(rdW, 1.0) * 0.6;
    float c = dot(rdW, uCompDir);   // the companion Luhman 16A, 3.5 AU away: a brilliant point
    col += uCompCol * (smoothstep(0.999994, 0.999999, c) * 14.0 + pow(max(c, 0.0), 40000.0) * 1.5 + pow(max(c, 0.0), 3000.0) * 0.03);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ------------------------------------------------------------------------------------------------ */
/* JS side: orientation, clocks, picking                                                              */
/* ------------------------------------------------------------------------------------------------ */
// Body frame for obliquity ob (deg) with the Sun (world azimuth sunAz) at sub-solar latitude B (deg).
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
function hitBody(o, d, W2B, flat) {
  const ro = o.clone().applyMatrix3(W2B), rd = d.clone().applyMatrix3(W2B);
  const k = 1 / (1 - flat);
  const A = rd.x * rd.x + rd.y * rd.y * k * k + rd.z * rd.z, B = ro.x * rd.x + ro.y * rd.y * k * k + ro.z * rd.z;
  const C = ro.x * ro.x + ro.y * ro.y * k * k + ro.z * ro.z - 1, H = B * B - A * C;
  let t = null;
  if (H >= 0) { const t0 = (-B - Math.sqrt(H)) / A; if (t0 > 0) t = t0; }
  return { t, ro, rd, k };
}
// body point → planetographic latitude / body-fixed east longitude
function bodyLatLon(p, k, spin) {
  const n = new V3(p.x, p.y * k * k, p.z).normalize();
  const c = Math.cos(spin), s = Math.sin(spin);
  const qx = c * p.x - s * p.z, qz = s * p.x + c * p.z;     // Ry(−spin)
  return { lat: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) * R2D, lonE: Math.atan2(-qz, qx), n };
}

function makeGiant(cfg) {
  const def = {
    id: cfg.id, name: cfg.name, category: cfg.category || 'planets', order: cfg.order, subtitle: cfg.subtitle, swatch: cfg.swatch,
    blurb: cfg.blurb, intro: cfg.intro, facts: cfg.facts, unitKm: cfg.R / 1000, camera: cfg.camera,
    exposure: cfg.exposure ?? 1, bloom: cfg.bloom ?? 0.35, params: cfg.params, actions: cfg.actions,
    create(ctx) {
      const U = {
        ...ctx.shared,
        uSunDir: { value: new V3(0, 0, 1) }, uW2B: { value: new THREE.Matrix3() },
        uFlat: { value: cfg.flat }, uSpin: { value: 0 }, uRm: { value: cfg.R }, uSunI: { value: cfg.sunI ?? 1.1 },
        uPh: { value: new V3(0, 0, 1) }, uSeed: { value: new THREE.Vector2() }, uDrift: { value: new THREE.Vector4() },
        uAxis: { value: 0 }, uRingOn: { value: 1 }, uRingGain: { value: 1 }, uRingRot: { value: 0 }, uMoonA: { value: new THREE.Vector4() },
      };
      const frag = COMMON_GLSL + HEAD + cfg.glsl + MAIN;
      const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: frag, depthTest: false, depthWrite: false });
      const st = {
        U, cfg, W2B: U.uW2B.value, sun: U.uSunDir.value, pole: new V3(), spinOff: 0, frozen: 0, lastSim: null, T: null, tp: 0, spin: 0,
        moonOff: [0, 0, 0, 0], nextMoon: 0, drift: [0, 0, 0, 0],
      };
      const omega = TAU / cfg.prot * (cfg.retro ? -1 : 1);
      // advance the planet clock; idempotent within a frame
      const clock = (ctx) => {
        const p = ctx.params, sim = ctx.time.sim;
        if (st.lastSim === null) st.lastSim = sim;
        const dsim = Math.max(0, sim - st.lastSim); st.lastSim = sim;
        if (p.spin === false) st.frozen += dsim;
        st.tp = (sim - st.frozen) * SECS;
        const boost = p.windx ?? 1;
        if (st.T === null) st.T = st.tp * boost; else if (p.spin !== false) st.T += dsim * SECS * boost;
        orient(st, cfg.obliq, cfg.subsolar(p), p.sunAz);
        st.spin = omega * st.tp + st.spinOff;
        for (let i = 0; i < (cfg.drifts || []).length; i++) {
          const [lat, u, lon0] = cfg.drifts[i];
          st.drift[i] = wrap(lon0 + u / (cfg.R * Math.cos(lat * D2R)) * st.T);
        }
      };
      st.clock = clock;
      st.faceFeature = (latD, lonE, dirW) => {
        const qf = latlonJS(latD, lonE), cb = dirW.clone().applyMatrix3(st.W2B);
        st.spinOff += Math.atan2(cb.x, cb.z) - Math.atan2(qf.x, qf.z) - st.spin;
        st.spin = omega * st.tp + st.spinOff;
      };
      st.subCam = (ctx) => st.subPoint(ctx.camera.position);
      st.subPoint = (dirW) => {
        const d = dirW.clone().normalize().applyMatrix3(st.W2B);
        const k = 1 / (1 - cfg.flat);
        // geocentric direction → point on the ellipsoid → planetographic latitude
        const s = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
        return bodyLatLon(d.multiplyScalar(s), k, st.spin);
      };
      const inst = {
        material, st,
        // (deferred to the first frame: ctx.time is only current once the host's loop has run)
        enter(ctx) { st.pendingEnter = true; },
        update(dt, ctx) {
          const p = ctx.params;
          clock(ctx);
          if (st.pendingEnter) {
            st.pendingEnter = false;
            if (cfg.face) {
              const c = cfg.camera, dir = new V3().setFromSphericalCoords(1, c.polar * D2R, c.azimuth * D2R);
              const [lat, lon] = cfg.face(st);
              st.faceFeature(lat, lon, dir);
            }
            cfg.onEnter?.(st, ctx, !st.entered); st.entered = true;
          }
          U.uSpin.value = st.spin % TAU;
          const P = cfg.phaseP, T = st.T, t1 = T % P, t2 = (T + P / 2) % P;
          U.uPh.value.set(t1, t2, 1 - Math.abs(2 * t1 / P - 1));
          U.uSeed.value.set(Math.floor(T / P) % 29, (Math.floor((T + P / 2) / P) % 29) + 0.5);
          U.uDrift.value.set(st.drift[0], st.drift[1], st.drift[2], st.drift[3]);
          U.uAxis.value = p.axis ? 1 : 0;
          U.uRingOn.value = p.rings === false ? 0 : 1;
          U.uRingGain.value = cfg.ringGain ? cfg.ringGain(p) : 1;
          U.uRingRot.value = cfg.ringPeriod ? wrap(TAU * st.tp / cfg.ringPeriod) : 0;
          if (cfg.moons) for (let i = 0; i < 4; i++) U.uMoonA.value.setComponent(i, wrap(cfg.moons[i][0] + TAU * st.tp / cfg.moons[i][1] + st.moonOff[i]));
          cfg.update?.(st, ctx);
        },
        hud(ctx) {
          clock(ctx);
          const sc = st.subCam(ctx);
          const cam = ctx.camera.position.clone().normalize();
          const phase = Math.acos(THREE.MathUtils.clamp(cam.dot(st.sun), -1, 1)) * R2D;
          const rows = [['Vĩ độ dưới camera', latStr(sc.lat)], ['Gió tại vĩ độ đó', windStr(cfg.wind(sc.lat))]];
          rows.push(['Góc pha (Mặt Trời–hành tinh–bạn)', vn(phase, 0) + '°']);
          // local solar time under the camera, in 'planet hours' (24 per rotation)
          const ss = st.subPoint(st.sun);
          const hr = ((12 + Math.sign(omega) * (sc.lonE - ss.lonE) * 24 / TAU) % 24 + 24) % 24;
          const hh = Math.floor(hr), mm = Math.floor((hr - hh) * 60);
          rows.push(['Giờ địa phương dưới camera', `${hh}:${String(mm).padStart(2, '0')} (1 ngày = ${vn(cfg.prot / 3600)} giờ)`]);
          if (cfg.hud) rows.push(...cfg.hud(st, ctx, sc));
          return rows;
        },
        onClick(info, ctx) { cfg.click?.(st, info, ctx); },
      };
      return inst;
    },
  };
  return def;
}

// Ring-plane hit (body frame) in front of the planet, or null
function hitRing(h, st) {
  if (Math.abs(h.rd.y) < 1e-6) return null;
  const t = -h.ro.y / h.rd.y;
  if (t <= 0 || (h.t !== null && h.t < t)) return null;
  const p = h.ro.clone().addScaledVector(h.rd, t);
  return { t, r: Math.hypot(p.x, p.z) };
}
const planetClickText = (st, h, cfg) => {
  const p = h.ro.clone().addScaledVector(h.rd, h.t);
  const ll = bodyLatLon(p, h.k, st.spin);
  const u = cfg.wind(ll.lat);
  return { ll, u, p };
};

/* ------------------------------------------------------------------------------------------------ */
/* scene definitions                                                                                  */
/* ------------------------------------------------------------------------------------------------ */
const commonParams = (sunAz, wind, windNote) => [
  { key: 'sunAz', label: 'Hướng Mặt Trời', type: 'range', min: 0, max: 360, step: 1, default: sunAz, fmt: v => v.toFixed(0) + '°', note: 'Phương vị của Mặt Trời trong mặt phẳng quỹ đạo. Xoay camera ra phía sau hành tinh để nhìn ngược sáng.' },
  { key: 'windx', label: 'Tua nhanh gió', type: 'range', min: 0, max: 40, step: 0.5, default: wind, fmt: v => '×' + v.toFixed(1), note: windNote || 'Khuếch đại tốc độ các dòng tia (so với thời gian mô phỏng) để thấy mây bị trượt theo vĩ độ.' },
  { key: 'spin', label: 'Tự quay', type: 'toggle', default: true },
];

const JUPITER = makeGiant({
  id: 'jupiter', name: 'Sao Mộc', order: 40, R: 71492e3, flat: 0.06487, obliq: 3.13, prot: 35730, phaseP: 1.2e5,
  subtitle: 'Hành tinh khí khổng lồ · H₂ + He',
  swatch: ['#e9d3b0', '#6b3f24'],
  blurb: 'Hành tinh lớn nhất hệ Mặt Trời quay một vòng chưa tới 10 giờ. Sự quay nhanh ấy kéo khí quyển thành các <b>dòng tia</b> đông–tây xen kẽ: vùng sáng (zone) có gió xoáy nghịch, dải tối (belt) có gió xoáy thuận. Vết Đỏ Lớn là một cơn bão xoáy nghịch đã tồn tại ít nhất 190 năm.',
  intro: 'Nhấp vào đĩa Sao Mộc để đặt một vệ tinh Galileo sao cho <b>bóng</b> của nó rơi đúng chỗ đó. Phím <b>G</b>: quay Vết Đỏ Lớn về phía bạn.',
  facts: [
    ['Bán kính xích đạo', '71 492 km (11,2 Trái Đất)'], ['Bán kính cực', '66 854 km · độ dẹt 0,065'],
    ['Khối lượng', '1,898×10²⁷ kg (318 Trái Đất)'], ['Chu kỳ tự quay (System III)', '9 giờ 55 phút 30 giây'],
    ['Khoảng cách tới Mặt Trời', '5,20 AU'], ['Nhiệt độ ở mức 1 bar', '165 K'],
    ['Vết Đỏ Lớn', '~16 000 km — vừa 1,3 Trái Đất'], ['Số Trái Đất chứa vừa', '≈ 1 321'],
  ],
  camera: { distance: 3.2, polar: 82, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.3, sunI: 0.75,
  params: [
    ...commonParams(55, 10),
    { key: 'rings', label: 'Vành bụi mờ (phóng đại ×1000)', type: 'toggle', default: true, note: 'Vành của Sao Mộc là bụi mịn, độ dày quang học ~10⁻⁶: chỉ thấy khi nhìn ngược sáng.' },
    { key: 'axis', label: 'Trục quay', type: 'toggle', default: false },
  ],
  actions: [
    { label: 'Vết Đỏ Lớn', key: 'g', title: 'Quay Vết Đỏ Lớn về phía camera', run: ctx => faceAction(ctx, 'Vết Đỏ Lớn: xoáy nghịch rộng ~16 000 km, gió ở viền ~150 m/s, quay ngược chiều kim đồng hồ mỗi ~4,5 ngày.') },
    { label: 'Cực Bắc (Juno)', key: 'j', title: 'Bay lên trên cực Bắc', run: ctx => poleAction(ctx, 'Cụm xoáy thuận quanh cực Bắc: 8 cơn bão, mỗi cơn rộng ~4 000 km, xếp thành hình bát giác quanh một cơn ở chính cực (Juno, 2017). Mặt Trời chỉ cao ~3° trên cực nên ảnh JunoCam phải kéo sáng mạnh.', 2.5) },
  ],
  wind: lat => lerpTab(JUP_WIND, lat),
  subsolar: () => 2.0,
  // festoon wave (7°N, +103 m/s), pearls (40.5°S), Oval BA (33°S), NEB barges (15.5°N)
  drifts: [[7, 103, 0], [-40.5, -3, 1.2], [-33, -4, 2.6], [15.5, -2, 0.7]],
  face: () => [-22, 0],
  // Io, Europa, Ganymede, Callisto: [initial angle, period s]
  moons: [[0.6, 152854], [2.4, 306822], [4.0, 618153], [5.3, 1441931]],
  glsl: JUPITER_GLSL,
  hud(st, ctx, sc) {
    const lonW = wrap(-sc.lonE) * R2D;
    return [['Kinh độ System III', vn(lonW, 0) + '° T']];
  },
  onEnter(st, ctx, first) {
    // first visit: put Io's shadow on the visible, sunlit disc
    if (!first) return;
    const cam = new V3().setFromSphericalCoords(1, st.cfg.camera.polar * D2R, st.cfg.camera.azimuth * D2R);
    const d = cam.add(st.sun).normalize().applyMatrix3(st.W2B);
    const k = 1 / (1 - st.cfg.flat), sc = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
    placeShadow(st, 0, d.multiplyScalar(sc));
  },
  click(st, info, ctx) {
    const h = hitBody(info.origin, info.dir, st.W2B, st.cfg.flat);
    if (h.t === null) return;
    const { ll, u, p } = planetClickText(st, h, st.cfg);
    const s = st.sun.clone().applyMatrix3(st.W2B);
    const n = ll.n;
    if (n.dot(s) <= 0.02) { ctx.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>, gió ${windStr(u)}. Điểm này đang là ban đêm — không có bóng vệ tinh nào ở đó.`); return; }
    const MO = [['Io', 3643], ['Europa', 3122], ['Ganymede', 5268], ['Callisto', 4821]];
    const i = st.nextMoon; st.nextMoon = (i + 1) % 4;
    const trackLat = placeShadow(st, i, p);
    ctx.toast(`Bóng của <b>${MO[i][0]}</b> (đường kính ${fmtInt(MO[i][1])} km) rơi lên đỉnh mây. Bóng chỉ có thể nằm trên một vệt quanh vĩ độ <b>${latStr(trackLat)}</b> — quỹ đạo các vệ tinh nằm gần mặt phẳng xích đạo. Dưới bóng là nhật thực toàn phần.`, 7000);
    ctx.audio.ping(520 - i * 70, 0.05, info.ndcX);
  },
});

const SATURN = makeGiant({
  id: 'saturn', name: 'Sao Thổ', order: 41, R: 60268e3, flat: 0.09796, obliq: 26.73, prot: 38362, phaseP: 4e4,
  subtitle: 'Khí khổng lồ có vành · khối lượng riêng 0,69 g/cm³',
  swatch: ['#f0dcae', '#7a5f38'],
  blurb: 'Sao Thổ nhẹ hơn nước và dẹt nhất hệ Mặt Trời. Vành của nó là hàng tỉ hạt băng quay theo định luật Kepler — rộng 280 000 km nhưng chỉ dày khoảng chục mét. Ở cực Bắc, một dòng tia uốn khúc thành <b>hình lục giác</b> rộng hơn Trái Đất.',
  intro: 'Kéo thanh <b>B′</b> về 0 để xem vành ở điểm phân, hoặc nhấn <b>B</b> để bay vào bóng Sao Thổ và nhìn vành ngược sáng như Cassini. Nhấp vào vành để biết hạt băng ở đó quay nhanh đến đâu.',
  facts: [
    ['Bán kính xích đạo', '60 268 km (9,45 Trái Đất)'], ['Bán kính cực', '54 364 km · độ dẹt 0,098'],
    ['Khối lượng', '5,683×10²⁶ kg (95 Trái Đất)'], ['Chu kỳ tự quay', '10 giờ 39 phút (System III)'],
    ['Độ nghiêng trục', '26,73°'], ['Khoảng cách tới Mặt Trời', '9,58 AU'],
    ['Vành chính (C → A)', '74 700 – 136 780 km'], ['Độ dày vành', '~10 m – 1 km'],
    ['Lục giác cực Bắc', 'mỗi cạnh ~14 500 km, ở 78° B'],
  ],
  camera: { distance: 5.5, polar: 66, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.35, sunI: 0.9,
  params: [
    { key: 'season', label: 'Mặt Trời trên mặt phẳng vành (B′)', type: 'range', min: -26.7, max: 26.7, step: 0.1, default: 18, fmt: v => v.toFixed(1) + '°', note: '0° = điểm phân (vành gần như tối và mỏng như sợi chỉ, bóng vành chỉ là một vạch), ±26,7° = điểm chí.' },
    ...commonParams(45, 4),
    { key: 'axis', label: 'Trục quay', type: 'toggle', default: false },
  ],
  actions: [
    { label: 'Ngược sáng', key: 'b', title: 'Bay vào bóng Sao Thổ, nhìn về phía Mặt Trời', run: ctx => backlitAction(ctx) },
    { label: 'Điểm phân', key: 'e', title: 'Mặt Trời nằm trong mặt phẳng vành', run: ctx => { ctx.params.season = 0; ctx.toast('Điểm phân (lần gần nhất: 2025): Mặt Trời chiếu ngang mép vành, vành gần như tối hẳn, bóng vành trên Sao Thổ co thành một vạch mảnh.'); } },
    { label: 'Lục giác', key: 'x', title: 'Bay lên trên cực Bắc', run: ctx => poleAction(ctx, 'Lục giác cực Bắc: một dòng tia ~100 m/s chảy uốn khúc thành sóng đứng 6 thùy, quay cùng chu kỳ với lõi hành tinh (10 h 39 m). Ở tâm là một xoáy thuận với mắt bão rộng ~2 000 km.', 1.0) },
  ],
  wind: lat => lerpTab(SAT_WIND, lat),
  subsolar: p => p.season,
  ringPeriod: 14.87 * 3600,  // F ring
  glsl: SATURN_GLSL,
  hud(st, ctx) {
    const cb = ctx.camera.position.clone().normalize().applyMatrix3(st.W2B);
    return [['Độ mở vành nhìn từ camera', vn(Math.asin(cb.y) * R2D) + '°'], ['Mặt Trời trên mặt phẳng vành', vn(ctx.params.season) + '°']];
  },
  click(st, info, ctx) {
    const h = hitBody(info.origin, info.dir, st.W2B, st.cfg.flat);
    const rh = hitRing(h, st);
    if (rh && rh.r > 1.11 && rh.r < 2.36) {
      const r = rh.r;
      const name = r < 1.239 ? 'vành D' : r < 1.527 ? 'vành C' : r < 1.951 ? 'vành B' : r < 2.025 ? 'khe Cassini' : (Math.abs(r - 2.2138) < 0.004 ? 'khe Encke' : r < 2.268 ? 'vành A' : r > 2.31 && r < 2.34 ? 'vành F' : 'khoảng giữa vành A và F');
      const rm = r * 60268e3, GM = 3.7931e16;
      const T = TAU * Math.sqrt(rm ** 3 / GM) / 3600, v = Math.sqrt(GM / rm) / 1000;
      ctx.toast(`<b>${name}</b>, cách tâm ${fmtInt(rm / 1000)} km. Hạt băng ở đây bay ${vn(v)} km/s, hết một vòng trong <b>${vn(T)} giờ</b> — ${T < 10.66 ? 'nhanh hơn' : 'chậm hơn'} một ngày Sao Thổ (10,7 giờ). Vòng trong quay nhanh hơn vòng ngoài: định luật Kepler.`, 7000);
      ctx.audio.ping(300 + 600 / r, 0.05, info.ndcX);
      return;
    }
    if (h.t === null) return;
    const { ll, u } = planetClickText(st, h, st.cfg);
    ctx.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>: gió ${windStr(u)}${Math.abs(ll.lat) < 20 ? ' — dòng tia xích đạo của Sao Thổ nhanh gấp 4 lần của Sao Mộc' : ll.lat > 74 ? ' — gần dòng tia lục giác' : ''}.`);
  },
});

const NEPTUNE = makeGiant({
  id: 'neptune', name: 'Sao Hải Vương', order: 43, R: 24764e3, flat: 0.0171, obliq: 28.32, prot: 57996, phaseP: 1.2e4,
  subtitle: 'Hành tinh băng khổng lồ · H₂, He, CH₄',
  swatch: ['#6f9cf0', '#14245e'],
  blurb: 'Khí mê-tan hấp thụ ánh sáng đỏ nên Sao Hải Vương ánh xanh lam. Dù nhận ánh nắng yếu hơn Trái Đất 900 lần, nó có <b>gió mạnh nhất hệ Mặt Trời</b> — tới 2 100 km/h, gần bằng tốc độ âm thanh ở đó. Voyager 2 (1989) chụp được Vết Tối Lớn cùng các dải mây ti trắng đổ bóng xuống tầng mây bên dưới.',
  intro: 'Phím <b>G</b>: quay Vết Tối Lớn về phía bạn. Đưa Mặt Trời ra sát rìa (Hướng Mặt Trời) để thấy các dải mây ti đổ bóng.',
  facts: [
    ['Bán kính xích đạo', '24 764 km (3,9 Trái Đất)'], ['Khối lượng', '1,024×10²⁶ kg (17,1 Trái Đất)'],
    ['Chu kỳ tự quay', '16 giờ 6,6 phút'], ['Độ nghiêng trục', '28,32°'], ['Khoảng cách tới Mặt Trời', '30,07 AU'],
    ['Nhiệt độ ở mức 1 bar', '72 K'], ['Gió mạnh nhất', '~580 m/s (xích đạo, về tây)'],
    ['Vết Tối Lớn (1989)', '13 000 × 6 600 km'], ['Nhiệt lượng tỏa ra', '2,6 lần năng lượng nhận từ Mặt Trời'],
  ],
  camera: { distance: 3.2, polar: 84, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.05, bloom: 0.35, sunI: 1.15,
  params: [
    ...commonParams(50, 1, 'Gió trên Sao Hải Vương đã rất nhanh: ×1 là tốc độ thật (đã tua 360 lần theo thời gian mô phỏng).'),
    { key: 'rings', label: 'Vành & cung Adams', type: 'toggle', default: true },
    { key: 'ringGain', label: 'Độ sáng vành (phóng đại)', type: 'range', min: 1, max: 60, step: 1, default: 20, fmt: v => '×' + v.toFixed(0), note: 'Vành Sao Hải Vương là bụi tối, gần như vô hình — thấy rõ nhất khi nhìn ngược sáng.' },
    { key: 'axis', label: 'Trục quay', type: 'toggle', default: false },
  ],
  actions: [
    { label: 'Vết Tối Lớn', key: 'g', title: 'Quay Vết Tối Lớn về phía camera', run: ctx => faceAction(ctx, 'Vết Tối Lớn: xoáy nghịch cỡ Trái Đất, trôi về tây ~300 m/s. Các đám mây trắng “bạn đồng hành” là băng mê-tan ngưng tụ khi khí bị đẩy lên trên sườn xoáy. Nó biến mất sau vài năm (Hubble, 1994).') },
  ],
  wind: nepWind,
  subsolar: () => -22,
  ringGain: p => p.ringGain,
  ringPeriod: 37925,   // Adams ring arcs (Keplerian at 62 930 km)
  drifts: [[-20, nepWind(-20) + 25, 0], [-42, nepWind(-42), 2.0], [-55, nepWind(-55), 4.1]],
  face: st => [-20, st.drift[0]],
  glsl: NEPTUNE_GLSL,
  click(st, info, ctx) {
    const h = hitBody(info.origin, info.dir, st.W2B, st.cfg.flat);
    if (h.t === null) return;
    const { ll, u } = planetClickText(st, h, st.cfg);
    ctx.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>: gió ${windStr(u)} = ${fmtInt(Math.abs(u) * 3.6)} km/h${Math.abs(u) > 300 ? ' — gần tốc độ âm thanh trong khí H₂/He lạnh 72 K (~600 m/s)' : ''}.`);
  },
});

const URANUS = makeGiant({
  id: 'uranus', name: 'Sao Thiên Vương', order: 42, R: 25559e3, flat: 0.0229, obliq: 97.77, prot: 62064, retro: true, phaseP: 1.5e4,
  subtitle: 'Hành tinh băng khổng lồ · lăn nghiêng trên quỹ đạo',
  swatch: ['#bdeef0', '#2f6e7a'],
  blurb: 'Trục quay nghiêng <b>97,8°</b>: Sao Thiên Vương lăn trên quỹ đạo như một quả bóng. Mỗi cực có 42 năm ngày liên tục rồi 42 năm đêm. Lớp sương mù dày phủ trên tầng mê-tan khiến nó gần như không có đường nét — một quả cầu màu lục lam nhạt với các vành hẹp và tối như than.',
  intro: 'Kéo <b>Vĩ độ điểm dưới Mặt Trời</b> để đi qua các mùa: ở điểm chí, cực hướng thẳng về Mặt Trời và các vành trông như một tấm bia; ở điểm phân, vành dựng đứng.',
  facts: [
    ['Bán kính xích đạo', '25 559 km (4,0 Trái Đất)'], ['Khối lượng', '8,681×10²⁵ kg (14,5 Trái Đất)'],
    ['Chu kỳ tự quay', '17 giờ 14 phút (nghịch hành)'], ['Độ nghiêng trục', '97,77°'], ['Chu kỳ quỹ đạo', '84 năm'],
    ['Khoảng cách tới Mặt Trời', '19,2 AU'], ['Nhiệt độ thấp nhất', '49 K — lạnh nhất hệ Mặt Trời'],
    ['Vành ε', 'cách tâm 51 149 km, rộng 20 – 96 km'],
  ],
  camera: { distance: 4.2, polar: 78, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.35, sunI: 1.1,
  params: [
    { key: 'season', label: 'Vĩ độ điểm dưới Mặt Trời', type: 'range', min: -82, max: 82, step: 0.5, default: 58, fmt: v => v.toFixed(1) + '°', note: '0° = điểm phân (2007, 2049); +82° = hạ chí Bắc (2030). Bây giờ (~2025): khoảng +60°.' },
    ...commonParams(40, 1),
    { key: 'ringGain', label: 'Độ sáng vành (phóng đại)', type: 'range', min: 1, max: 80, step: 1, default: 30, fmt: v => '×' + v.toFixed(0), note: 'Hạt vành có hệ số phản xạ ~0,03 — tối như than.' },
    { key: 'axis', label: 'Trục quay', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Điểm phân', key: 'e', title: 'Mặt Trời trên xích đạo', run: ctx => { ctx.params.season = 0; ctx.toast('Điểm phân (2007): Mặt Trời chiếu ngang xích đạo, vành nhìn từ Mặt Trời là một vạch thẳng đứng. Ngày–đêm luân phiên mỗi 17 giờ ở khắp hành tinh.'); } },
    { label: 'Hạ chí', key: 'o', title: 'Cực Bắc hướng về Mặt Trời', run: ctx => { ctx.params.season = 82; ctx.toast('Hạ chí Bắc (2030): Mặt Trời gần như ở thiên đỉnh của cực Bắc. Cả bán cầu Nam chìm trong đêm dài 42 năm.'); } },
  ],
  wind: lat => lerpTab(URA_WIND, lat),
  subsolar: p => p.season,
  ringGain: p => p.ringGain,
  ringPeriod: 8.38 * 3600,   // ε ring (precessing ellipse, drawn co-rotating)
  glsl: URANUS_GLSL,
  hud(st, ctx, sc) {
    const s = ctx.params.season;
    return [['Ở cực Bắc bây giờ', s > 1 ? 'ngày liên tục' : s < -1 ? 'đêm liên tục' : 'Mặt Trời ở chân trời']];
  },
  click(st, info, ctx) {
    const h = hitBody(info.origin, info.dir, st.W2B, st.cfg.flat);
    if (h.t === null) return;
    const { ll, u } = planetClickText(st, h, st.cfg);
    const B = ctx.params.season;
    const day = Math.abs(ll.lat) > 90 - Math.abs(B) ? (Math.sign(ll.lat) === Math.sign(B) ? 'Mặt Trời không bao giờ lặn ở đây vào mùa này' : 'đêm kéo dài suốt mùa này') : 'ngày và đêm luân phiên mỗi vòng quay 17 giờ';
    ctx.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>: gió ${windStr(u)}; ${day}.`);
  },
});

// Jupiter: choose the orbital phase of Galilean moon i so its shadow lands as close as possible to body point p.
// Returns the latitude of the shadow centre (the moons orbit in the equatorial plane, so shadows follow fixed tracks).
function placeShadow(st, i, p) {
  const A = [5.905, 9.397, 14.99, 26.37][i];
  const s = st.sun.clone().applyMatrix3(st.W2B);
  let best = 1e9, bestA = 0;
  for (let k = 0; k < 2880; k++) {
    const a = k / 2880 * TAU, v = new V3(Math.sin(a), 0, Math.cos(a)).multiplyScalar(A).sub(p), tca = v.dot(s);
    if (tca <= 0) continue;
    const d = v.addScaledVector(s, -tca).length();
    if (d < best) { best = d; bestA = a; }
  }
  st.moonOff[i] = bestA - (st.cfg.moons[i][0] + TAU * st.tp / st.cfg.moons[i][1]);
  const m = new V3(Math.sin(bestA), 0, Math.cos(bestA)).multiplyScalar(A);
  const hs = hitBody(m.applyMatrix3(st.W2B.clone().transpose()), st.sun.clone().negate(), st.W2B, st.cfg.flat);
  return hs.t !== null ? bodyLatLon(hs.ro.clone().addScaledVector(hs.rd, hs.t), hs.k, st.spin).lat : 0;
}

/* ---------- actions shared by the planets ---------- */
function instOf(ctx) { return ctx.scene && ctx.scene.__inst; }
function faceAction(ctx, msg) {
  const inst = instOf(ctx); if (!inst) return;
  const st = inst.st, f = st.cfg.face(st);
  st.clock(ctx);
  st.faceFeature(f[0], f[1], ctx.camera.position.clone().sub(ctx.controls.target).normalize());
  ctx.toast(msg, 7000);
}
function poleAction(ctx, msg, exposure) {
  const inst = instOf(ctx); if (!inst) return;
  const st = inst.st; st.clock(ctx);
  const d = Math.max(2.0, Math.min(3.2, ctx.camera.position.length()));
  const side = new V3().crossVectors(st.pole, st.sun).normalize();
  // above the pole, leaning slightly toward the Sun so the lit half fills the view
  const dir = st.pole.clone().multiplyScalar(0.94).addScaledVector(st.sun, 0.3).addScaledVector(side, 0.05).normalize();
  ctx.camera.position.copy(dir.multiplyScalar(d)); ctx.camera.lookAt(0, 0, 0); ctx.controls.update();
  if (exposure && ctx.bhGlobals) ctx.bhGlobals.exposure = Math.max(ctx.bhGlobals.exposure, exposure);
  ctx.toast(msg, 8000);
}
function backlitAction(ctx) {
  const inst = instOf(ctx); if (!inst) return;
  const st = inst.st; st.clock(ctx);
  const sgn = Math.sign(st.sun.dot(st.pole)) || 1;
  const perp = st.pole.clone().addScaledVector(st.sun, -st.pole.dot(st.sun)).normalize();
  const dir = st.sun.clone().negate().addScaledVector(perp, -sgn * 0.1).normalize();
  ctx.camera.position.copy(dir.multiplyScalar(7.5)); ctx.camera.lookAt(0, 0, 0); ctx.controls.update();
  ctx.toast('Trong bóng Sao Thổ, nhìn về phía Mặt Trời (như Cassini, 2013): vành C, khe Cassini và vành F bụi bặm sáng rực vì bụi tán xạ về phía trước, còn vành B dày đặc lại tối vì ánh sáng không xuyên qua được.', 9000);
}

/* ------------------------------------------------------------------------------------------------ */
/* brown dwarf                                                                                        */
/* ------------------------------------------------------------------------------------------------ */
// Perceived colour of a brown-dwarf spectrum: Planck × transmission through the pressure-broadened wings
// of the Na I D (589 nm) and K I (766/770 nm) resonance lines, integrated against CIE 1931 (same fits as
// the host's blackbody table). Removing yellow and near-red leaves deep red + a little blue: magenta.
function dwarfColour(T, alkali) {
  const g = (x, mu, s1, s2) => { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
  const h = 6.62607015e-34, c = 2.99792458e8, k = 1.380649e-23;
  const s = Math.min(1.6, Math.max(0.15, (2300 - T) / 1100));   // alkali wings strengthen as clouds sink in cooler dwarfs
  let X = 0, Y = 0, Z = 0;
  for (let lam = 380; lam <= 780; lam += 2.5) {
    const l = lam * 1e-9;
    const B = 1 / (Math.pow(l, 5) * (Math.exp(h * c / (l * k * T)) - 1));
    const tr = alkali ? Math.exp(-s * (10 / (1 + ((lam - 589) / 80) ** 2) + 10 / (1 + ((lam - 768) / 35) ** 2))) : 1;
    X += B * tr * (1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2));
    Y += B * tr * (0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1));
    Z += B * tr * (1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8));
  }
  X /= Y; Z /= Y;
  let r = 3.2406 * X - 1.5372 - 0.4986 * Z, gg = -0.9689 * X + 1.8758 + 0.0415 * Z, b = 0.0557 * X - 0.2040 + 1.0570 * Z;
  const m = Math.min(r, gg, b); if (m < 0) { r -= m; gg -= m; b -= m; }
  const lum = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
  return [r / lum, gg / lum, b / lum];
}
const specType = T => T > 2300 ? 'M muộn' : T > 1300 ? 'L' + Math.min(9, Math.max(0, Math.round((2300 - T) / 110))) : T > 550 ? 'T' + Math.min(9, Math.max(0, Math.round((1300 - T) / 85))) : 'Y';

const BROWN_DWARF = {
  id: 'brown-dwarf', name: 'Sao lùn nâu', category: 'stars', order: 60,
  subtitle: 'Luhman 16B · kiểu phổ T0,5 · 6,5 năm ánh sáng',
  swatch: ['#d0507a', '#2a0614'],
  blurb: 'Quá nặng để là hành tinh, quá nhẹ để là ngôi sao: sao lùn nâu (13–80 lần khối lượng Sao Mộc) chỉ đốt được <b>deuterium</b> trong thời gian ngắn rồi nguội dần mãi mãi. Luhman 16B nóng ~1 200 K, tự phát sáng đều cả hai mặt; Na và K hấp thụ ánh vàng–cam nên mắt người sẽ thấy nó màu <b>đỏ tía (magenta)</b>.',
  intro: 'Đổi <b>Bước sóng</b> sang hồng ngoại J để thấy các khe hở giữa mây — đúng thứ mà phép chụp Doppler (Crossfield 2014) và JWST dùng để vẽ bản đồ thời tiết của nó.',
  facts: [
    ['Khối lượng', '~28 M_Sao Mộc (Garcia 2017)'], ['Bán kính', '~0,85 – 1 R_Sao Mộc'], ['Nhiệt độ hiệu dụng', '~1 210 K'],
    ['Chu kỳ tự quay', '5,28 giờ'], ['Khoảng cách', '2,0 pc (6,5 năm ánh sáng) — sao lùn nâu gần nhất'],
    ['Bạn đồng hành', 'Luhman 16A (L7,5, ~34 M_J), cách 3,5 AU'],
    ['Ngưỡng đốt deuterium', '≈ 13 M_J'], ['Ngưỡng đốt hydro (thành sao)', '≈ 75 – 80 M_J'],
  ],
  unitKm: 70000,
  camera: { distance: 3.2, polar: 80, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.5,
  params: [
    { key: 'teff', label: 'Nhiệt độ hiệu dụng', type: 'range', min: 500, max: 2400, step: 10, default: 1210, fmt: v => v.toFixed(0) + ' K', note: 'Kéo để đi qua dãy L → T → Y: sao lùn nâu nguội dần theo tuổi.' },
    { key: 'view', label: 'Bước sóng', type: 'seg', options: [['eye', 'Mắt người'], ['ir', 'Hồng ngoại J'], ['bb', 'Vật đen thuần']], default: 'eye', note: '“Vật đen thuần” bỏ đi sự hấp thụ của Na, K, H₂O, CH₄ — so sánh để thấy vì sao nó không “nâu”.' },
    { key: 'clouds', label: 'Độ loang lổ của mây', type: 'range', min: 0, max: 1, step: 0.01, default: 0.7 },
    { key: 'windx', label: 'Tua nhanh gió', type: 'range', min: 0, max: 20, step: 0.5, default: 1, fmt: v => '×' + v.toFixed(1) },
    { key: 'spin', label: 'Tự quay', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Hồng ngoại', key: 'i', title: 'Chuyển giữa mắt người và hồng ngoại J', run: ctx => { ctx.params.view = ctx.params.view === 'ir' ? 'eye' : 'ir'; ctx.toast(ctx.params.view === 'ir' ? 'Hồng ngoại J (1,25 µm): ánh sáng thoát ra từ sâu hơn qua các khe mây — chỗ sáng là nơi mây mỏng, nhìn thấy lớp khí nóng hơn.' : 'Ánh sáng khả kiến (mô phỏng mắt người).'); } },
  ],
  create(ctx) {
    const R = 7.0e7, prot = 5.28 * 3600, phaseP = 4e4, flat = 0.012;
    const U = {
      ...ctx.shared,
      uW2B: { value: new THREE.Matrix3() }, uFlat: { value: flat }, uSpin: { value: 0 }, uRm: { value: R },
      uPh: { value: new V3(0, 0, 1) }, uSeed: { value: new THREE.Vector2() },
      uTeff: { value: 1210 }, uMode: { value: 0 }, uClouds: { value: 0.7 }, uGlow: { value: 0.13 },
      uEye: { value: new V3(1, 0.3, 0.4) }, uEyeCool: { value: new V3(1, 0.3, 0.4) }, uCompDir: { value: new V3(-0.55, 0.32, -0.77).normalize() }, uCompCol: { value: new V3(1, 0.5, 0.3) },
      // unused HEAD uniforms
      uSunDir: { value: new V3(0, 0, 1) }, uSunI: { value: 0 }, uDrift: { value: new THREE.Vector4() }, uAxis: { value: 0 },
      uRingOn: { value: 0 }, uRingGain: { value: 0 }, uRingRot: { value: 0 }, uMoonA: { value: new THREE.Vector4() },
    };
    // slight obliquity so the poles show (unknown for Luhman 16B; its rotation axis is thought to be near edge-on)
    const ob = 12 * D2R, P = new V3(Math.sin(ob), Math.cos(ob), 0);
    const xb = new V3().crossVectors(new V3(1, 0, 0), P).normalize(), zb = new V3().crossVectors(xb, P);
    U.uW2B.value.set(xb.x, xb.y, xb.z, P.x, P.y, P.z, zb.x, zb.y, zb.z);
    // place Luhman 16A in the sky, up and to the left of the default view
    { const c = this.camera, v = new V3().setFromSphericalCoords(1, c.polar * D2R, c.azimuth * D2R).negate();
      const rt = new V3().crossVectors(v, new V3(0, 1, 0)).normalize(), up = new V3().crossVectors(rt, v);
      U.uCompDir.value.copy(v).addScaledVector(rt, -0.42).addScaledVector(up, 0.22).normalize(); }
    const comp = dwarfColour(1350, true);
    U.uCompCol.value.set(...comp).multiplyScalar(1.4);
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: BD_FRAG, depthTest: false, depthWrite: false });
    const st = { lastSim: null, frozen: 0, T: null, lastTeff: -1 };
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params, sim = ctx.time.sim;
        if (st.lastSim === null) st.lastSim = sim;
        const dsim = Math.max(0, sim - st.lastSim); st.lastSim = sim;
        if (!p.spin) st.frozen += dsim;
        const tp = (sim - st.frozen) * SECS;
        if (st.T === null) st.T = tp * p.windx; else if (p.spin) st.T += dsim * SECS * p.windx;
        U.uSpin.value = (TAU * tp / prot) % TAU;
        const T = st.T, t1 = T % phaseP;
        U.uPh.value.set(t1, (T + phaseP / 2) % phaseP, 1 - Math.abs(2 * t1 / phaseP - 1));
        U.uSeed.value.set(Math.floor(T / phaseP) % 29, (Math.floor((T + phaseP / 2) / phaseP) % 29) + 0.5);
        U.uTeff.value = p.teff; U.uClouds.value = p.clouds;
        U.uMode.value = p.view === 'ir' ? 1 : p.view === 'bb' ? 2 : 0;
        // displayed brightness grows gently with Teff (true visible surface brightness spans ~10⁵ over this range)
        U.uGlow.value = 0.13 * Math.pow(p.teff / 1210, 1.5);
        if (p.teff !== st.lastTeff) { st.lastTeff = p.teff; U.uEye.value.set(...dwarfColour(p.teff, true)); U.uEyeCool.value.set(...dwarfColour(p.teff * 0.85, true)); }
        st.tp = tp;
      },
      hud(ctx) {
        const T = ctx.params.teff;
        return [
          ['Kiểu phổ (theo T)', specType(T)],
          ['Đỉnh phát xạ (Wien)', vn(2898 / T, 2) + ' µm — hồng ngoại'],
          ['Tốc độ quay ở xích đạo', vn(TAU * R / prot / 1000) + ' km/s'],
          ['Độ sáng so với Mặt Trời', '~' + ctx.fmt.sci(Math.pow(T / 5772, 4) * 0.01) + ' L☉'],
        ];
      },
      onClick(info, ctx) {
        const W2B = U.uW2B.value;
        const h = hitBody(info.origin, info.dir, W2B, flat);
        if (h.t === null) return;
        const p = h.ro.clone().addScaledVector(h.rd, h.t);
        const ll = bodyLatLon(p, h.k, 0);
        const mu = Math.max(0, -ll.n.dot(h.rd));
        const Tb = ctx.params.teff * Math.pow(0.75 * (mu + 2 / 3), 0.25);
        ctx.toast(`Nhiệt độ sáng ở đây ≈ <b>${fmtInt(Tb)} K</b> (μ = ${vn(mu, 2)}). Ở rìa đĩa ta nhìn xiên vào các lớp khí cao hơn, lạnh hơn — nên rìa tối và đỏ hơn tâm (khí quyển xám Eddington).`, 6000);
      },
    };
  },
};

// let actions find their instance through ctx.scene
for (const def of [JUPITER, SATURN, NEPTUNE, URANUS]) {
  const create = def.create;
  def.create = function (ctx) { const inst = create.call(this, ctx); def.__inst = inst; return inst; };
}

export default [JUPITER, SATURN, URANUS, NEPTUNE, BROWN_DWARF];
