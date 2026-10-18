// Sao Mộc — Jupiter from real Hubble data, set in motion by its measured winds.
//
// Surface: Hubble OPAL global maps of 11 Dec 2025 (WFC3/UVIS, 10 px/deg, Minnaert-corrected I/F), turned
// into true colour from the calibrated FITS (631 / 502 / 467 nm, checked against Karkoschka's full-disc
// albedo), plus the 889 nm methane band and the 275 nm ultraviolet map for the other views. The Great Red
// Spot and Oval BA were cut out of the global map (in-painted with the same latitudes from elsewhere) and kept
// as separate patches so they can turn and drift as vortices of their own (see assets/jupiter/CREDITS.txt).
//
// Motion = flow-map advection through an analytic velocity field, evaluated per pixel (stateless, so the
// time slider and snapshots are exact):
//   * zonal jets: the zonal-wind profile measured from these very maps (OPAL rotation 1 vs rotation 2, one
//     Jupiter day apart, cross-correlated latitude by latitude), blended with Voyager/Cassini where the
//     correlation is poor;
//   * Great Red Spot & Oval BA: anticyclones (counter-clockwise in the south) with a near-calm core, a
//     ~120 m/s collar, slow westward drift (+ the GRS's 90-day longitude oscillation) and a weak inflow at the
//     rim that winds small features into the collar;
//   * Kelvin–Helmholtz-like eddies: rows of short-lived vortices drifting with the jets, whose spin is the
//     local shear vorticity (×3) — they exist only where the shear is strong and roll the band edges up;
// Each pixel is traced backwards (RK2) through that field for the age of a "copy" of the map. Two copies are
// staggered by half a period and re-born (from the observed map) when their weight is zero, so shear never
// winds features up without bound; the hand-over is a noise-shaped dissolve rather than a double exposure.
// The vortex patches are re-born pre-rotated by the collar's rotation, so the GRS keeps turning across resets.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';

const V3 = THREE.Vector3;
const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAU = Math.PI * 2;
const A_EQ = 71492e3, B_POL = 66854e3, FLAT = 1 - B_POL / A_EQ, E2 = 1 - (B_POL / A_EQ) ** 2;
const PROT = 35729.7;                 // System III, s
const OMEGA = TAU / PROT;
const RESET_P = 54000;                // flow-map copy period (s of Jupiter time) — 15 h
const EPOCH = Date.UTC(2025, 11, 11, 16, 0, 0);   // mid-time of the OPAL maps (t = 0)
const ASSET = f => new URL(`../../../assets/jupiter/${f}`, import.meta.url).href;

/* ---------------------------------------------------------------------------------------------- */
/* data                                                                                             */
/* ---------------------------------------------------------------------------------------------- */
// Fallback zonal wind (Voyager/Cassini, Limaye 1986, Porco+ 2003), replaced by the measured profile on load.
const CANON = [[-90, 0], [-72, 0], [-66, 25], [-61, -10], [-57, 30], [-53, -12], [-49, 28], [-45, -15], [-39.5, 42], [-36, -22],
  [-32, 48], [-29, -18], [-26.5, 55], [-22, 0], [-19.5, -58], [-16, -15], [-12, 15], [-7, 135], [-3, 95], [0, 85], [3, 95], [7, 128],
  [10, 60], [14, 10], [17, -32], [20, 25], [23.7, 165], [26.5, -20], [29.5, 25], [31.5, 10], [35.5, 32], [39, -12], [43, 30],
  [46.5, -10], [49, 25], [53, -10], [56, 28], [60, -5], [65, 18], [70, 0], [90, 0]];
const lerpTab = (pts, x) => {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) if (x < pts[i][0]) return pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * (x - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
  return pts[pts.length - 1][1];
};
// geometry of the oblate planet at planetographic latitude (deg): parallel radius and meridional radius (m)
function radii(latD) {
  const s = Math.sin(latD * D2R), c = Math.cos(latD * D2R), den = Math.sqrt(1 - E2 * s * s);
  return { par: A_EQ * c / den, mer: A_EQ * (1 - E2) / (den * den * den), N: A_EQ / den };
}
// vortices measured on the 2025 map: centre (°E, °planetographic), semi-axes (° lon, ° lat)
const GRS = { lon: 38.1, lat: -22.8, a: 6.0, b: 4.22, vmax: 120, rm: 0.9, inflow: 6,
  // drift: −4.3 m/s (westward in System III ≈ 1.75°/month in System II) + the 90-day oscillation (±1.2°)
  drift: -4.3, oscA: 1.2, oscP: 89.8 * 86400 };
const BA = { lon: 147.3, lat: -33.6, a: 3.3, b: 1.8, vmax: 110, rm: 0.9, inflow: 3, drift: -4.0, oscA: 0, oscP: 1 };
function vortexGeom(v) {
  const r = radii(v.lat);
  const A = v.a * D2R * r.par, B = v.b * D2R * r.mer;
  const L = Math.sqrt((A * A + B * B) / 2);           // streamline 'radius' of the ρ = 1 ellipse
  const f = x => { const x4 = (x / v.rm) ** 4; return x4 * Math.exp(1 - x4); };
  const om = rho => v.vmax * f(rho) / (rho * L);      // rad/s in normalised (elliptic) coordinates
  return { A, B, L, par: r.par, om, periodEdge: TAU / om(1.0), periodCollar: TAU / om(v.rm), vEdge: v.vmax * f(1) };
}
const GRS_G = vortexGeom(GRS), BA_G = vortexGeom(BA);
// longitude (rad, east) of a drifting vortex at Jupiter time T (s)
const vLon = (v, g, T) => v.lon * D2R + v.drift / g.par * T + v.oscA * D2R * Math.sin(TAU * T / v.oscP);
const vDrift = (v, g, T) => v.drift / g.par + v.oscA * D2R * TAU / v.oscP * Math.cos(TAU * T / v.oscP);

/* ---------------------------------------------------------------------------------------------- */
/* GLSL                                                                                             */
/* ---------------------------------------------------------------------------------------------- */
const FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir;
uniform mat3 uW2B;
uniform float uFlat, uSpin, uSunI, uP, uMode, uFlow, uDetail, uTexOK, uAur, uRingOn, uAxis, uJunoL;
uniform sampler2D uVis, uIR, uVtx, uBand, uMean, uZone;
uniform vec4 uClk;            // x: t/P (cycles, mod 4096) · y: t (s, mod 4096 P) · z: t (days, mod 1000) for the polar fields
uniform vec4 uG, uGA, uGV;    // GRS  centre (lonE, lat rad, drift rad/s, Ωc) · axes (a, b rad, L m, -) · (vmax m/s, ρm, inflow m/s, -)
uniform vec4 uB, uBA, uBV;    // Oval BA
uniform vec4 uMoonA;
uniform float uDbg;
#define R2D 57.2957795
#define D2R 0.0174532925
#define VLO 0.3
#define VSC 0.7
const vec3 IRS = vec3(0.22, 0.42, 0.8);
#define MRAD 6.6e7            // meridional radius of curvature (m), mid-latitudes (eddies only)
#define EH 0.0349             // eddy rows: 2° of latitude
#define EW 8.0e6              // eddy spacing along a row (m)
#define ELIFE 108000.0        // eddy life (s) — 30 h
#define ECONC 4.0             // eddy vorticity / shear vorticity

float gMu, gMu0;
mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
vec3 latlon(float latD, float lonE){ float la = latD * D2R; return vec3(cos(la) * cos(lonE), sin(la), -cos(la) * sin(lonE)); }
float wrapPi(float a){ return a - TAU * floor(a / TAU + 0.5); }
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * c, 1.5); }
mat2 rot2(float a){ float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }

// band texture: x = ω (1e-6 rad/s), y = ζ (1e-5 1/s), z = turbulence, w = parallel radius (1e7 m)
vec4 bandAt(float phi){ return textureLod(uBand, vec2(phi * (1.0 / PI) + 0.5, 0.5), 0.0); }

// ---- velocity field (rad/s in east longitude, latitude) ----------------------------------------
// Kelvin–Helmholtz-like eddies: one row of short-lived Gaussian vortices per 2° of latitude, drifting with the
// wind at the row centre; each spins with the local shear vorticity (concentrated ECONC×) through a 30 h life.
vec2 eddyVel(vec2 p, float t){
  float row = floor(p.y / EH);
  float phc = (row + 0.5) * EH;
  if (abs(phc) > 1.2) return vec2(0.0);
  vec4 bc = bandAt(phc);
  float zeta = bc.y * 1e-5;
  float zs = smoothstep(0.5e-5, 2.0e-5, abs(zeta));
  if (zs <= 0.0) return vec2(0.0);
  float rp = bc.w * 1e7;
  float n = max(floor(TAU * rp / EW), 3.0);
  float x = (p.x - bc.x * 1e-6 * t) * (n / TAU);
  float ci = floor(x), cim = mod(ci, n);
  float age = t / ELIFE + hash13(vec3(cim, row, 3.7));
  float gen = floor(age); age -= gen;
  vec3 hh = hash33(vec3(cim + 0.13, row + 0.71, gen));
  if (hh.z < 0.3) return vec2(0.0);
  float fx = x - ci, fy = p.y / EH - row;
  vec2 d = vec2((fx - 0.3 - 0.4 * hh.x) * (TAU / n) * rp, (fy - 0.5 - 0.25 * (hh.y - 0.5)) * EH * MRAD);
  float re = 0.26 * EH * MRAD;
  float env = smoothstep(0.0, 0.2, fy) * smoothstep(1.0, 0.8, fy) * smoothstep(0.0, 0.12, fx) * smoothstep(1.0, 0.88, fx);
  float W = 0.5 * ECONC * zeta * sin(PI * age) * zs * (0.4 + 0.9 * hh.z) * env * exp(-dot(d, d) / (re * re));
  vec2 v = W * vec2(-d.y, d.x);
  return vec2(v.x / rp, v.y / MRAD);
}
// elliptical vortex: near-calm core, collar of peak speed vmax at ρm, decaying outside; ax = (a, b, L)
vec2 vortexVel(vec2 p, vec2 c, vec4 ax, vec4 vp, out float rho){
  vec2 l = vec2(wrapPi(p.x - c.x) / ax.x, (p.y - c.y) / ax.y);
  rho = length(l);
  if (rho > 2.6) return vec2(0.0);
  float x = rho / vp.y, x4 = x * x * x * x;
  float om = vp.x * x4 * exp(1.0 - x4) / (max(rho, 1e-3) * ax.z);
  float inflow = vp.z / ax.z * exp(-pow((rho - 1.25) / 0.3, 2.0));
  vec2 dl = om * vec2(-l.y, l.x) - inflow * l / max(rho, 1e-3);   // counter-clockwise = anticyclone in the south
  return dl * ax.xy;
}
vec2 vel(vec2 p, float s){
  vec4 b = bandAt(p.y);
  vec2 v = vec2(b.x * 1e-6, 0.0) + (mod(uDbg, 2.0) > 0.5 ? vec2(0.0) : eddyVel(p, uClk.y + s));
  float r;
  vec2 vg = vortexVel(p, vec2(uG.x + uG.z * s, uG.y), uGA, uGV, r) + vec2(uG.z, 0.0);
  v = mix(v, vg, 1.0 - smoothstep(1.1, 1.9, r));
  if (abs(p.y - uB.y) < 0.11){
    vg = vortexVel(p, vec2(uB.x + uB.z * s, uB.y), uBA, uBV, r) + vec2(uB.z, 0.0);
    v = mix(v, vg, 1.0 - smoothstep(1.1, 1.9, r));
  }
  return v;
}
// backward trajectory: where was the cloud now at p, tau seconds ago? (RK2 midpoint)
vec2 trace(vec2 p, float tau){
  int n = int(clamp(ceil(tau / 10800.0), 1.0, 6.0));
  float h = tau / float(n);
  for (int i = 0; i < 6; i++){
    if (i >= n) break;
    float s = -h * float(i);
    vec2 k1 = vel(p, s);
    vec2 k2 = vel(vec2(p.x - 0.5 * h * k1.x, clamp(p.y - 0.5 * h * k1.y, -1.55, 1.55)), s - 0.5 * h);
    p -= h * k2;
    p.y = clamp(p.y, -1.55, 1.55);
  }
  return p;
}

// ---- the observed map at birth -------------------------------------------------------------------
vec3 decodeMap(vec3 t){ return uMode > 2.5 ? t * IRS : VLO + VSC * t; }
vec3 mapAt(vec2 pb, vec2 gx, vec2 gy){
  vec2 uv = vec2(pb.x * (1.0 / TAU), pb.y * (1.0 / PI) + 0.5);
  return decodeMap(uMode > 2.5 ? textureGrad(uIR, uv, gx, gy).rgb : textureGrad(uVis, uv, gx, gy).rgb);
}
// Zone frames: before its birth a copy has been carried by its zone's mean rotation (zones are split at the
// shear maxima, so a constant drift per zone never winds up); Tk = birth time / P. Across a zone boundary the
// two frames slide past each other: blend them over ±0.45° around a jittered boundary.
vec3 zoneMap(vec2 pb, float Tk, vec2 gx, vec2 gy, out float lonZ){
  vec4 z = textureLod(uZone, vec2(pb.y * (1.0 / PI) + 0.5, 0.5), 0.0);   // ω·P self, ω·P neighbour, boundary lat
  lonZ = pb.x - z.x * Tk;
  vec3 c = mapAt(vec2(lonZ, pb.y), gx, gy);
  float dz = pb.y - z.z, sg = dz >= 0.0 ? 1.0 : -1.0;
  if (abs(dz) < 0.012 && mod(floor(uDbg / 2.0), 2.0) < 0.5){
    float j = 0.0035 * sin(pb.x * 11.0 + 2.0 * sin(pb.x * 5.0 + z.z * 40.0)) + 0.002 * sin(pb.x * 37.0 + z.z * 90.0);
    float ws = smoothstep(-0.0078, 0.0078, sg * (dz + j));
    if (ws < 0.999) c = mix(mapAt(vec2(pb.x - z.y * Tk, pb.y), gx, gy), c, ws);
  }
  return c;
}
// vortex patch (atlas: GRS ±24°×±16° at (0,0), BA ±12°×±8° at (480,0); IR copies 320 px lower)
vec3 patchAt(vec2 l, vec4 ax, vec2 org, vec2 hsz, float lod){
  vec2 dd = l * ax.xy * R2D;
  float yo = uMode > 2.5 ? 320.0 : 0.0;
  vec2 px = vec2(org.x + (dd.x + hsz.x) * 10.0, org.y + yo + (hsz.y - dd.y) * 10.0);
  return decodeMap(textureLod(uVtx, vec2(px.x / 720.0, 1.0 - px.y / 640.0), lod).rgb);
}
vec3 vortexBirth(vec3 c, vec2 pb, vec2 cen, vec4 ax, float th, vec2 org, vec2 hsz, float lod){
  vec2 l = vec2(wrapPi(pb.x - cen.x) / ax.x, (pb.y - cen.y) / ax.y);
  float r = length(l);
  if (r > 2.2) return c;
  l = rot2(-th * (1.0 - smoothstep(0.95, 1.35, r))) * l;    // collar pre-rotation (bounded twist in the rim)
  return mix(c, patchAt(l, ax, org, hsz, lod), 1.0 - smoothstep(1.6, 2.15, r));
}
// small-scale detail below the map's 120 km pixels: Jupiter's own cloud texture, 4× smaller (the map's
// high-pass sampled at 4× frequency from a mirrored mid-latitude strip, offset per copy), carried by the flow.
// Mip-mapping band-limits it automatically, so it only appears when the view resolves it.
float detail1(vec2 pb, vec2 k, vec2 o, vec2 gx, vec2 gy){
  float y = fract(pb.y * (k.y / PI) + o.y);
  vec2 uv = vec2(pb.x * (k.x / TAU) + o.x, 0.28 + 0.44 * abs(2.0 * y - 1.0));
  vec2 g = max(abs(gx), abs(gy)) * k;
  float lod = max(log2(max(max(g.x * 3600.0, g.y * 1800.0), 1e-4)), 0.0);
  vec3 a = textureLod(uVis, uv, lod).rgb, b = textureLod(uVis, uv, lod + 2.0).rgb;
  return dot(a - b, vec3(0.4, 0.4, 0.2)) * VSC;
}
// (the strip is offset so that the miniature never lines up with the band it decorates)
float detail(vec2 pb, vec2 gx, vec2 gy){
  return detail1(pb, vec2(4.0), vec2(0.37, 0.21), gx, gy);
}
// one flow-map copy: trace back to birth, sample the map (+ vortex patches + detail)
vec3 copyColour(vec2 p, float f, float cyc, float k, float h, vec2 gx, vec2 gy, float lod, float turb){
  float tau = f * uP * uFlow;
  vec2 pb = tau > 0.0 ? trace(p, tau) : p;
  float lonZ;
  vec3 c = zoneMap(pb, (cyc - 0.5 * k - h) * uFlow, gx, gy, lonZ);
  float th = mod(uG.w * uP * (cyc - 0.5 * k), TAU);
  c = vortexBirth(c, pb, vec2(uG.x - uG.z * tau, uG.y), uGA, th, vec2(0.0), vec2(24.0, 16.0), lod);
  if (abs(pb.y - uB.y) < 0.12){
    th = mod(uB.w * uP * (cyc - 0.5 * k), TAU);
    c = vortexBirth(c, pb, vec2(uB.x - uB.z * tau, uB.y), uBA, th, vec2(480.0, 0.0), vec2(12.0, 8.0), lod);
  }
  float dA = uDetail * (0.35 + 0.65 * turb) * (1.0 - smoothstep(0.5, 2.5, lod));
  if (dA > 0.01 && mod(floor(uDbg / 4.0), 2.0) < 0.5) c *= 1.0 + dA * 1.7 * detail(vec2(lonZ, pb.y), gx, gy);
  return c;
}

// ---- polar regions (Juno): cyclone clusters + folded filamentary turbulence ------------------------
// A modulation m of the observed polar I/F (≈ ±15 % — the OPAL stretch makes it read like JunoCam).
// North: 8 cyclones (~4 500 km) packed in an octagon around a central one; south: 5 around 1 (pentagon).
// Inside each cyclone the cloud filaments are noise sampled in log-spiral coordinates (winding tighter toward
// the core) that turn cyclonically (with the planet's spin as seen from above that pole) in ~1 day; the
// polygon itself drifts slowly. In between: chaotic folded filaments and small vortices.
float polarMod(float lat, float lon, out float cov){
  float sg = sign(lat);
  float colat = (PI * 0.5 - abs(lat)) * R2D;              // degrees from the pole (1° ≈ 1 330 km here)
  vec2 X = colat * vec2(cos(lon), sin(lon));              // azimuthal-equidistant (deg); +angle = east = cyclonic
  float td = uClk.z;                                      // days
  float n1 = fbm(vec3(X * 0.32, td * 0.04 + sg * 7.0), 3);
  vec2 W = X * 0.75 + 2.6 * vec2(n1, snoise(vec3(X * 0.3 + 9.1, td * 0.04)));
  float fil = 1.0 - abs(snoise(vec3(W, td * 0.08 - sg * 3.0)));
  float fil2 = 1.0 - abs(snoise(vec3(W * 2.1 + 4.0, td * 0.11 + sg)));
  float m = 0.13 * n1 + 0.09 * (fil * fil * fil - 0.3) + 0.025 * (fil2 * fil2 - 0.45) - 0.02;
  // small vortices in the turbulent belt: bright anticyclones, dark cyclones
  vec2 w = worley(vec3(X * 0.8, sg * 11.0 + floor(td * 0.02))).xy;
  float kind = hash13(vec3(floor(X * 0.8), sg));
  m += (kind > 0.55 ? 0.12 : kind < 0.3 ? -0.1 : 0.0) * smoothstep(0.32, 0.08, w.x);
  cov = 0.0;
  float nn = sg > 0.0 ? 8.0 : 5.0;
  float ring = sg > 0.0 ? 6.4 : 6.8;                       // centres at ~83.6°N / ~83.2°S
  float drift = td * 0.012;                               // polygon drift ≈ 0.7°/day
  for (int i = 0; i < 9; i++){
    if (float(i) > nn) break;
    float fi = float(i);
    float ph = fi * TAU / nn + 0.4 + drift;
    vec2 C = i == 0 ? vec2(0.35, -0.25) + 0.25 * vec2(sin(td * 0.05), cos(td * 0.04))
                    : ring * vec2(cos(ph), sin(ph)) + 0.3 * vec2(sin(td * 0.07 + fi), cos(td * 0.05 + 2.0 * fi));
    float rc = (i == 0 ? (sg > 0.0 ? 2.3 : 2.9) : (sg > 0.0 ? 2.5 : 3.3)) * (0.92 + 0.16 * hash13(vec3(fi, sg, 1.0)));
    vec2 d = X - C; float r = length(d) / rc;
    if (r > 1.25) continue;
    // turbulent radius: the spiral bands are broken and folded, not a clean roll
    float lr = log(r + 0.03) + 0.35 * snoise(vec3(d * 0.9 / rc, fi + td * 0.1));
    float ap = atan(d.y, d.x) - td * TAU / (0.9 + 0.3 * hash13(vec3(fi, sg, 2.0))) + 2.6 * lr;
    vec3 S = vec3(cos(ap) * 2.2, sin(ap) * 2.2, lr * 1.6 + fi * 3.7 + sg);
    float f1 = snoise(S);
    float fl = 1.0 - abs(snoise(S * vec3(2.4, 2.4, 1.6) + 5.0));
    float arms = sin(ap * 2.0 + 2.6 * f1);
    float core = hash13(vec3(fi, sg, 3.0)) > 0.5 ? 0.1 : -0.08;     // some cores bright, some dark (JunoCam)
    float inner = 0.11 * arms * smoothstep(0.05, 0.35, r) + 0.06 * f1 + 0.035 * (fl * fl * fl - 0.3)
                + core * smoothstep(0.3, 0.0, r) + 0.03 * (1.0 - r * r) - 0.05 * smoothstep(0.75, 1.05, r);
    float msk = smoothstep(1.2, 0.75, r + 0.2 * f1);      // ragged rim: the cyclone mixes into its surroundings
    m = mix(m, inner, msk);
    cov = max(cov, msk);
  }
  return m;
}
// ---- aurora (FUV main ovals + Io footprint), drawn in the UV view ----------------------------------
float aurora(float latD, float lonE){
  vec3 q = latlon(latD, lonE);
  vec3 cN = latlon(74.0, 3.14), cS = latlon(-80.0, 5.6);
  float dN = acos(clamp(dot(q, cN), -1.0, 1.0)) * R2D, dS = acos(clamp(dot(q, cS), -1.0, 1.0)) * R2D;
  // main ovals (footprints of the middle magnetosphere, ~20–30 R_J): the northern one is kidney-shaped.
  // Shape and brightness vary with the azimuth around each oval's own centre (no pole singularity).
  vec3 eN, nN, eS, nS;
  eN = normalize(vec3(cN.z, 0.0, -cN.x)); nN = cross(cN, eN);
  eS = normalize(vec3(cS.z, 0.0, -cS.x)); nS = cross(cS, eS);
  float aN = atan(dot(q, nN), dot(q, eN)), aS = atan(dot(q, nS), dot(q, eS));
  float rN = 13.0 + 3.0 * cos(aN - 1.0) + 1.0 * cos(2.0 * aN), rS = 10.5 + 1.3 * cos(aS + 0.7);
  float patN = 0.6 + 0.4 * snoise(vec3(cos(aN) * 2.0, sin(aN) * 2.0, uClk.z * 3.0));
  float patS = 0.6 + 0.4 * snoise(vec3(cos(aS) * 2.0, sin(aS) * 2.0, 5.0 + uClk.z * 3.0));
  float e = exp(-pow((dN - rN) / 1.0, 2.0)) * patN + exp(-pow((dS - rS) / 0.9, 2.0)) * patS
          + 0.1 * smoothstep(rN - 1.0, rN - 7.0, dN) * (0.3 + 0.7 * smoothstep(-0.2, 0.7, snoise(q * 6.0 + uClk.z * 2.0)))
          + 0.08 * smoothstep(rS - 1.0, rS - 6.0, dS);
  // Io's footprint (on the flux tube, ~66°N / −74°S at Io's System III longitude)
  vec3 io = rotY(-uSpin) * vec3(sin(uMoonA.x), 0.0, cos(uMoonA.x));
  float lio = atan(-io.z, io.x);
  e += 1.6 * exp(-dot(q - latlon(64.0, lio - 0.3), q - latlon(64.0, lio - 0.3)) / 2e-4)
     + 1.2 * exp(-dot(q - latlon(-73.0, lio - 0.3), q - latlon(-73.0, lio - 0.3)) / 2e-4);
  return e;
}

// ---- the cloud tops -----------------------------------------------------------------------------
vec3 surface(vec3 q, float lat, vec2 gx, vec2 gy, float lod){
  float lon = atan(-q.z, q.x);
  vec2 p = vec2(lon, lat);
  vec4 b = bandAt(lat);
  float turb = b.z;
  // staggered copy phases: cycles (h is fixed in System III, smooth and periodic in longitude)
  float h = 0.25 * sin(lon + 1.3 * sin(3.0 * lat)) + 0.15 * sin(2.0 * lon - 4.0 * lat) + 0.1 * sin(9.0 * lat);
  float a0 = uClk.x + h;
  vec3 col;
  if (uFlow < 0.5){
    col = copyColour(p, 0.0, 0.0, 0.0, 0.0, gx, gy, lod, turb);
  } else {
    float f0 = fract(a0), f1 = fract(a0 + 0.5);
    float w0 = smoothstep(0.0, 0.3, f0) * (1.0 - smoothstep(0.5, 0.8, f0));
    // noise-shaped dissolve instead of a cross-fade: each point shows one copy almost all the time
    float nd = 0.5 + 0.32 * snoise(q * 6.0 + 4.0);
    w0 = smoothstep(nd - 0.12, nd + 0.12, w0);
    col = vec3(0.0);
    if (w0 > 0.003) col += w0 * copyColour(p, f0, floor(a0), 0.0, h, gx, gy, lod, turb);
    if (w0 < 0.997) col += (1.0 - w0) * copyColour(p, f1, floor(a0 + 0.5), 1.0, h, gx, gy, lod, turb);
  }
  float al = abs(lat) * R2D;
  if (al > 66.0){
    float wp = smoothstep(66.0, 75.0, al);
    vec3 m = textureLod(uMean, vec2(lat / PI + 0.5, uMode > 2.5 ? 0.75 : 0.25), 0.0).rgb;
    float cov, pm = polarMod(lat, lon, cov);
    // visible light: dark lanes teal/blue-grey, bright cloud tops cream (the polar palette of JunoCam;
    // the factors are chosen for the OPAL stretch, whose green black point sits highest)
    vec3 k = uMode < 2.5 ? (pm < 0.0 ? vec3(1.4, 0.65, 0.45) : vec3(1.0, 1.0, 0.85)) : vec3(1.0);
    vec3 pc = mix(col, m, 0.7) * (1.0 + pm * k);
    if (uMode < 2.5) pc *= mix(vec3(0.975, 1.0, 1.0), vec3(1.0), cov);
    col = mix(col, pc, wp);
  }
  return col;
}

/* ---------------------------------------------------------------------------------------------- */
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
vec3 moonGlow(vec3 ro, vec3 rd, float pixAng, vec3 s){
  vec3 g = vec3(0.0);
  for (int i = 0; i < 4; i++){
    vec3 v = moonPos(i) - ro; float t = dot(v, rd);
    if (t <= 0.0) continue;
    float d = length(v - rd * t), px = t * pixAng;
    float ph = 0.5 + 0.5 * dot(normalize(ro - moonPos(i)), s);
    g += vec3(0.75, 0.7, 0.6) * uSunI * 3.0 * ph * (1.0 - smoothstep(0.4, 1.2, MOON_RAD[i] / px)) * exp(-d * d / (1.5 * px * px));
  }
  return g;
}
// umbra + penumbra (the Sun is 0.051° in radius at 5.2 AU)
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
float bodyShadow(vec3 p, vec3 s, vec3 E){
  vec3 o = p * E, d = s * E;
  float tc = -dot(o, d) / dot(d, d);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.996, 1.004, length(o + d * tc));
}
vec3 moonShade(int i, vec3 p, vec3 s, vec3 E){
  vec3 n = normalize(p - moonPos(i));
  vec3 m = rotY(-uMoonA[i]) * n;
  float n1 = fbm(m * 3.0 + float(i) * 7.0, 4), n2 = fbm(m * 9.0 + float(i) * 3.0, 3);
  vec3 alb;
  if (i == 0){
    alb = mix(vec3(0.82, 0.74, 0.34), vec3(0.93, 0.90, 0.72), smoothstep(0.0, 0.5, n1));
    alb = mix(alb, vec3(0.55, 0.24, 0.10), smoothstep(0.35, 0.6, n2) * 0.8);
    alb = mix(alb, vec3(0.06, 0.05, 0.04), smoothstep(0.55, 0.75, n2));
  } else if (i == 1){
    alb = mix(vec3(0.86, 0.82, 0.74), vec3(0.62, 0.48, 0.34), smoothstep(0.1, 0.5, n1) * 0.6);
    alb = mix(alb, vec3(0.50, 0.33, 0.22), (1.0 - smoothstep(0.0, 0.06, abs(n2))) * 0.6);
  } else if (i == 2){
    alb = mix(vec3(0.30, 0.27, 0.24), vec3(0.62, 0.60, 0.56), smoothstep(-0.1, 0.2, n1));
  } else {
    alb = vec3(0.22, 0.20, 0.18) + vec3(0.35) * smoothstep(0.45, 0.7, n2);
  }
  float mu0 = max(dot(n, s), 0.0);
  return alb * mu0 * uSunI * 1.2 * bodyShadow(p, s, E) + alb * 0.002;
}
// faint dusty rings (halo, main, gossamer); real τ ~ 1e-6, drawn ×1000 — visible toward the Sun (forward scattering)
float ringTau(float r){
  return (0.25 * smoothstep(1.29, 1.40, r) * (1.0 - smoothstep(1.66, 1.72, r))
        + 3.0 * smoothstep(1.72, 1.75, r) * (1.0 - smoothstep(1.795, 1.806, r))
        + 0.15 * step(1.806, r) * (1.0 - smoothstep(2.45, 2.55, r))
        + 0.06 * step(1.806, r) * (1.0 - smoothstep(3.05, 3.15, r))) * 1e-3;
}

vec3 rimLight(vec3 pn, vec3 rd, vec3 s){
  float il = smoothstep(-0.3, 0.2, dot(pn, s));
  vec3 rc = uMode > 3.5 ? vec3(0.06, 0.05, 0.12) : vec3(0.1);
  return rc * il * (0.35 + 0.65 * hg(dot(rd, s), 0.7));
}
vec3 sunDisc(vec3 rdW){
  float c = dot(rdW, uSunDir);
  return bb(5772.0) * (smoothstep(0.99997, 0.999985, c) * 60.0 + pow(max(c, 0.0), 6000.0) * 3.0 + pow(max(c, 0.0), 300.0) * 0.04);
}
// displayed colour for the chosen view; c = I/F (vis: RGB at 631/≈540/≈460 nm; IR views: 889, 275, 395 nm)
vec3 viewColour(vec3 c, float lat){
  if (uMode < 0.5) return c;
  // Hubble OPAL colour composite: the linear stretch of I/F the OPAL team applied (fitted to their 2025a map, r ≥ 0.97)
  if (uMode < 1.5) return max(vec3(1.359, 1.300, 1.109) * c - vec3(0.705, 0.688, 0.541), 0.0) * 1.7;
  if (uMode < 2.5){   // enhanced contrast (JunoCam-style): stretch around the band mean, boost saturation
    vec3 m = textureLod(uMean, vec2(lat / PI + 0.5, 0.25), 0.0).rgb;
    vec3 e = m + (c - m) * 3.0;
    float L = dot(e, vec3(0.3, 0.5, 0.2));
    e = mix(vec3(L), e, 2.2) * vec3(0.96, 1.0, 1.08);
    return max(e, 0.0) * 0.9;
  }
  if (uMode < 3.5) return vec3(0.92, 0.95, 1.0) * max(c.r - 0.025, 0.0) * 7.0;   // methane 889 nm: high clouds & haze bright
  return vec3(0.72, 0.62, 1.0) * max(c.g - 0.09, 0.0) * 2.2;                      // UV 275 nm: high haze absorbs (dark poles)
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
  // texture derivatives from a seam-free, silhouette-continuous field (hit point, or closest approach)
  float tS = tP < 1e8 ? tP : max(-B / A, 0.0);
  vec3 qS = Rs * normalize((ro + rd * tS) * E);
  vec3 dqx = dFdx(qS), dqy = dFdy(qS);
  float rxz = max(qS.x * qS.x + qS.z * qS.z, 1e-6);
  vec2 gx = vec2((qS.z * dqx.x - qS.x * dqx.z) / rxz / TAU, dqx.y / (PI * sqrt(rxz)));
  vec2 gy = vec2((qS.z * dqy.x - qS.x * dqy.z) / rxz / TAU, dqy.y / (PI * sqrt(rxz)));
  // at the silhouette the field jumps between hit and closest approach: cap the footprint at 32 texels
  float lx = length(gx * vec2(3600.0, 1800.0)), ly = length(gy * vec2(3600.0, 1800.0));
  gx *= min(1.0, 32.0 / max(lx, 1e-6)); gy *= min(1.0, 32.0 / max(ly, 1e-6));
  float lod = log2(max(min(max(lx, ly), 32.0), 1e-4));

  int mid = -1; float tM = moonsHit(ro, rd, mid);
  float tN = min(tP, tM);
  vec3 col;
  if (tP < 1e8 && tP <= tN){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = Rs * (p * E);
    float lat = asin(clamp(n.y, -1.0, 1.0));
    gMu0 = dot(n, s); gMu = max(dot(n, -rd), 0.0);
    vec3 If = uTexOK > 0.5 ? surface(q, lat, gx, gy, lod) : vec3(0.6, 0.55, 0.5);
    vec3 alb = viewColour(If, lat);
    // Minnaert limb darkening per channel, k from the OPAL map headers (631: 1.0, 502: 0.95, 467: 0.95; 889: 1.0; 275: 0.52)
    vec3 K = uMode < 2.5 ? vec3(1.0, 0.97, 0.95) : uMode < 3.5 ? vec3(1.0) : vec3(0.6);
    float m0 = max(gMu0, 0.0), m1 = max(gMu, 0.06);
    vec3 lit = pow(vec3(m0), K) * pow(vec3(m1), K - 1.0);
    lit += 0.01 * smoothstep(-0.1, 0.03, gMu0) * (1.0 - smoothstep(0.0, 0.15, gMu0));
    // JunoCam-style processing: brightness flattened toward the terminator (the poles only ever see a low Sun)
    if (uJunoL > 0.5) lit = max(lit, vec3(0.5 * smoothstep(-0.2, 0.18, gMu0)));
    float sh = moonShadows(p, s);
    col = alb * lit * uSunI * sh;
    col += rimLight(n, rd, s) * uSunI * pow(1.0 - gMu, 6.0) * sh;
    if (uMode > 3.5 && uAur > 0.5){
      float lonE = atan(-q.z, q.x);
      col += vec3(0.55, 0.35, 1.0) * aurora(lat * R2D, lonE) * 0.35;
    }
  } else if (tM < 1e8){
    col = moonShade(mid, ro + rd * tM, s, E);
  } else {
    col = sky(rdW, 1.0) * 0.6 + sunDisc(rdW);
    col += moonGlow(ro, rd, pixAng, s);
    float tc = -B / A;
    if (tc > 0.0){
      vec3 pc = roE + rdE * tc;
      float hgt = max(length(pc) - 1.0, 0.0);
      col += rimLight(normalize(pc), rd, s) * uSunI * exp(-hgt / 0.015);
    }
  }
  // rings (plane y = 0 of the body frame)
  if (uRingOn > 0.5 && abs(rd.y) > 1e-6){
    float tR = -ro.y / rd.y;
    if (tR > 0.0 && tR < tN){
      vec3 pr = ro + rd * tR;
      float r = length(pr.xz);
      if (r > 1.29 && r < 3.2){
        float tau = ringTau(r);
        float mu = abs(rd.y), mu0 = max(abs(s.y), 1e-4);
        float c = dot(s, rd);
        float IF = 0.25 * hg(c, 0.7) * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu)));
        col = col * exp(-tau / mu) + vec3(0.75, 0.62, 0.50) * IF * uSunI * bodyShadow(pr, s, E);
      }
    }
  }
  if (uAxis > 0.5){
    float b = rd.y, dd = dot(rd, ro), den = 1.0 - b * b;
    if (den > 1e-6){
      float sc = (b * ro.y - dd) / den, ta = ro.y + b * sc;
      if (sc > 0.0 && sc < tN){
        float dist = length(ro + rd * sc - vec3(0.0, ta, 0.0));
        float line = exp(-pow(dist / (1.4 * sc * pixAng), 2.0)) * step(0.85, abs(ta)) * (1.0 - smoothstep(1.45, 1.7, abs(ta)));
        col += (ta > 0.0 ? vec3(1.0, 0.42, 0.22) : vec3(0.3, 0.6, 1.0)) * line * 0.7;
      }
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ---------------------------------------------------------------------------------------------- */
/* JS helpers                                                                                       */
/* ---------------------------------------------------------------------------------------------- */
const latlonJS = (latD, lonE) => { const la = latD * D2R; return new V3(Math.cos(la) * Math.cos(lonE), Math.sin(la), -Math.cos(la) * Math.sin(lonE)); };
const vn = (x, d = 1) => x.toFixed(d).replace('.', ',');
const latStr = d => `${vn(Math.abs(d))}° ${d >= 0 ? 'B' : 'N'}`;
const windStr = u => `${u >= 0 ? '+' : '−'}${Math.abs(u).toFixed(0)} m/s ${u >= 0 ? '(về đông)' : '(về tây)'}`;
const wrap = a => ((a % TAU) + TAU) % TAU;
const fmtInt = n => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
const toHalf = THREE.DataUtils.toHalfFloat;

function loadImageTex(url, { srgb = false, aniso = 1 }) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = aniso; tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(`không tải được ${url}`))));
}
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
const loadJSON = url => fetch(url).then(r => { if (!r.ok) throw new Error(`không tải được ${url}`); return r.json(); });

// per-latitude band texture (0.25° steps): ω, relative vorticity, turbulence, parallel radius
function bandTexture(windAt, turbAt) {
  const N = 721, data = new Uint16Array(N * 4);
  for (let i = 0; i < N; i++) {
    const lat = -90 + i * 0.25, r = radii(lat);
    const u = windAt(lat);
    const om = r.par > 1e6 ? u / r.par : 0;
    const dudphi = (windAt(Math.min(90, lat + 0.25)) - windAt(Math.max(-90, lat - 0.25))) / (0.5 * D2R);
    const zeta = Math.abs(lat) < 88 ? -dudphi / r.mer + u * Math.tan(lat * D2R) / r.N : 0;
    data.set([toHalf(om * 1e6), toHalf(zeta * 1e5), toHalf(turbAt(lat)), toHalf(r.par / 1e7)], i * 4);
  }
  const t = new THREE.DataTexture(data, N, 1, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = t.magFilter = THREE.LinearFilter; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.needsUpdate = true;
  return t;
}
// Zones between shear maxima (|du/dy| > 1.2e-5 s⁻¹, ≥ 2° apart, |lat| < 72°), each with its mean angular
// speed. Texture (0.125° steps, nearest): ω·P of this zone, ω·P of the zone across the nearest boundary, that boundary (rad).
function zoneTexture(windAt) {
  const N = 1441, d = 0.125, lat = i => -90 + i * d;
  const shear = new Float64Array(N);
  for (let i = 1; i < N - 1; i++) shear[i] = Math.abs(windAt(lat(i + 1)) - windAt(lat(i - 1))) / (2 * d * D2R * radii(lat(i)).mer);
  let bnd = [];
  for (let i = 2; i < N - 2; i++) {
    if (Math.abs(lat(i)) > 72 || shear[i] < 1.2e-5 || shear[i] < shear[i - 1] || shear[i] < shear[i + 1]) continue;
    const L = lat(i), last = bnd[bnd.length - 1];
    if (last && L - last.L < 2) { if (shear[i] > last.s) bnd[bnd.length - 1] = { L, s: shear[i] }; continue; }
    bnd.push({ L, s: shear[i] });
  }
  bnd = bnd.map(b => b.L);
  const edges = [-90, ...bnd, 90], zw = [];
  for (let z = 0; z < edges.length - 1; z++) {
    let sw = 0, so = 0;
    for (let L = edges[z]; L < edges[z + 1]; L += 0.05) { const r = radii(L), w = Math.cos(L * D2R); if (r.par > 1e6) { so += w * windAt(L) / r.par; sw += w; } }
    zw.push(sw > 0 ? so / sw : 0);
  }
  const data = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const L = lat(i);
    let z = 0; while (z < bnd.length && L >= bnd[z]) z++;
    // nearest boundary
    let nb = -1, nd = 1e9;
    for (let k = 0; k < bnd.length; k++) if (Math.abs(bnd[k] - L) < nd) { nd = Math.abs(bnd[k] - L); nb = k; }
    const other = nb < 0 ? z : (L >= bnd[nb] ? nb : nb + 1);
    data.set([zw[z] * RESET_P, zw[other] * RESET_P, nb < 0 ? 9 : bnd[nb] * D2R, 0], i * 4);
  }
  const t = new THREE.DataTexture(data, N, 1, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = t.magFilter = THREE.NearestFilter; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.needsUpdate = true;
  t.userData = { bnd, zw };
  return t;
}
function meanTexture(prof) {
  const N = prof ? prof.meanVis.length : 2, data = new Uint16Array(N * 2 * 4);
  for (let i = 0; i < N; i++) {
    const v = prof ? prof.meanVis[i] : [0.75, 0.68, 0.62], r = prof ? prof.meanIR[i] : [0.06, 0.3, 0.45];
    data.set([toHalf(v[0]), toHalf(v[1]), toHalf(v[2]), toHalf(1)], i * 4);
    data.set([toHalf(r[0]), toHalf(r[1]), toHalf(r[2]), toHalf(1)], (N + i) * 4);
  }
  const t = new THREE.DataTexture(data, N, 2, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = t.magFilter = THREE.LinearFilter; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.needsUpdate = true;
  return t;
}
function solidTex(rgba) { const t = new THREE.DataTexture(new Uint8Array(rgba), 1, 1, THREE.RGBAFormat); t.needsUpdate = true; return t; }

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
  return { lat: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) * R2D, lonE: Math.atan2(-qz, qx), n };
}

const MOONS = [[0.6, 152854], [2.4, 306822], [4.0, 618153], [5.3, 1441931]];   // [initial angle, period s]
const MOON_NAMES = [['Io', 3643], ['Europa', 3122], ['Ganymede', 5268], ['Callisto', 4821]];
const OBLIQ = 3.13, SUBSOLAR = 2.0;

/* ---------------------------------------------------------------------------------------------- */
/* scene                                                                                            */
/* ---------------------------------------------------------------------------------------------- */
const def = {
  id: 'jupiter',
  name: 'Sao Mộc',
  category: 'planets',
  order: 40,
  subtitle: 'Hành tinh khí khổng lồ · H₂ + He · ảnh Hubble 2025',
  swatch: ['#e9d3b0', '#6b3f24'],
  blurb: 'Bề mặt ở đây là <b>ảnh thật</b>: bản đồ toàn cầu của kính Hubble (chương trình OPAL, 12/2025) — rồi được cho <b>chuyển động theo gió đo được</b>. Sự quay nhanh (chưa tới 10 giờ) kéo khí quyển thành các <b>dòng tia</b> đông–tây ngược chiều nhau tới 140 m/s; ở ranh giới giữa chúng mây bị xé và cuộn thành xoáy. <b>Vết Đỏ Lớn</b> là một xoáy nghịch rộng hơn Trái Đất, quay ngược chiều kim đồng hồ và trôi chậm về phía tây.',
  intro: 'Nhấn <b>G</b> để bay tới Vết Đỏ Lớn và bám theo nó: các dải mây trượt qua nhau, xoáy cuộn lên ở ranh giới, cơn bão quay tròn. Tắt <b>Gió & xoáy</b> để so với ảnh tĩnh.',
  facts: [
    ['Bán kính xích đạo', '71 492 km (11,2 Trái Đất)'], ['Bán kính cực', '66 854 km · độ dẹt 0,065'],
    ['Khối lượng', '1,898×10²⁷ kg (318 Trái Đất)'], ['Chu kỳ tự quay (System III)', '9 giờ 55 phút 30 giây'],
    ['Khoảng cách tới Mặt Trời', '5,20 AU'], ['Nhiệt độ ở mức 1 bar', '165 K'],
    ['Dòng tia mạnh nhất', '~140 m/s ở 24° B (đo trên chính hai ảnh này)'],
    ['Vết Đỏ Lớn (12/2025)', '≈ 14 000 × 9 500 km, gió viền ~110–120 m/s'],
    ['Dữ liệu', 'NASA/ESA Hubble OPAL — A. Simon, M. H. Wong (CC BY 4.0); cực theo Juno/JunoCam (NASA/JPL-Caltech/SwRI/MSSS)'],
  ],
  unitKm: A_EQ / 1000,
  camera: { distance: 3.2, polar: 82, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.3,
  params: [
    { key: 'band', label: 'Ảnh / bước sóng', type: 'seg', default: 'hst',
      options: [['hst', 'Hubble (OPAL)'], ['vis', 'Màu thật'], ['enh', 'Tăng tương phản'], ['ch4', 'Mê-tan 889 nm'], ['uv', 'Tử ngoại']],
      note: 'Hubble: cách đội OPAL kéo giãn độ sáng cho ảnh màu công bố. Màu thật: hệ số phản xạ đã hiệu chuẩn ở 631/502/467 nm — Sao Mộc thật nhạt hơn ảnh báo chí nhiều. Mê-tan: chỉ mây cao và sương mù ở cực sáng lên. Tử ngoại (275 nm): sương mù cực hấp thụ, kèm cực quang tia cực tím.' },
    { key: 'lapse', label: 'Tua nhanh thời tiết', type: 'range', min: 1, max: 60, step: 1, default: 10, fmt: v => '×' + fmtInt(v * 60),
      note: 'Số giây trên Sao Mộc ứng với một giây mô phỏng. Mặc định (×600, đồng hồ chung 6×) ≈ 1 giờ Sao Mộc mỗi giây thật.' },
    { key: 'track', label: 'Khung nhìn', type: 'seg', default: 'spin',
      options: [['spin', 'Tự quay'], ['s3', 'Đứng yên (System III)'], ['grs', 'Bám Vết Đỏ']],
      note: 'Tự quay: hành tinh quay thật (9 h 55 m). Đứng yên: nhìn trong hệ quay cùng lõi hành tinh, chỉ thấy gió. Bám Vết Đỏ: khung trôi theo cơn bão, như các phim Juno/Cassini.' },
    { key: 'flow', label: 'Gió & xoáy', type: 'toggle', default: true, note: 'Tắt để thấy ảnh Hubble tĩnh.' },
    { key: 'detail', label: 'Rối nhỏ (dưới độ phân giải ảnh)', type: 'range', min: 0, max: 1.5, step: 0.05, default: 0.8,
      note: 'Ảnh Hubble có ~120 km mỗi điểm ảnh; khi phóng to, các xoáy nhỏ hơn được sinh thêm theo thống kê rối của từng dải, và trôi cùng dòng chảy.' },
    { key: 'sunAz', label: 'Hướng Mặt Trời', type: 'range', min: 0, max: 360, step: 1, default: 55, fmt: v => v.toFixed(0) + '°' },
    { key: 'junoLight', label: 'Ánh sáng kiểu JunoCam', type: 'toggle', default: false,
      note: 'Mặt Trời chỉ cao vài độ trên vùng cực (trục Sao Mộc nghiêng 3°). Ảnh JunoCam được xử lý làm sáng vùng gần đường phân chia ngày–đêm; bật để xem các cơn bão cực như vậy.' },
    { key: 'aurora', label: 'Cực quang (ở chế độ tử ngoại)', type: 'toggle', default: true },
    { key: 'rings', label: 'Vành bụi mờ (phóng đại ×1000)', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Vết Đỏ Lớn', key: 'g', title: 'Bay tới Vết Đỏ Lớn và bám theo nó', run: ctx => def.__inst?.goGRS(ctx) },
    { label: 'Oval BA', key: 'b', title: 'Quay Oval BA (“Vết Đỏ Nhỏ”) về phía camera', run: ctx => def.__inst?.goBA(ctx) },
    { label: 'Cực Bắc (Juno)', key: 'j', title: 'Bay lên trên cực Bắc', run: ctx => def.__inst?.goPole(ctx, 1) },
    { label: 'Cực Nam (Juno)', key: 'n', title: 'Bay xuống dưới cực Nam', run: ctx => def.__inst?.goPole(ctx, -1) },
  ],
  create(ctx) {
    const R = ctx.renderer, aniso = Math.min(8, R.capabilities.getMaxAnisotropy());
    const grey = solidTex([170, 150, 130, 255]);
    let windAt = lat => lerpTab(CANON, lat), turbAt = () => 0.7;
    const U = {
      ...ctx.shared,
      uSunDir: { value: new V3(0, 0, 1) }, uW2B: { value: new THREE.Matrix3() },
      uFlat: { value: FLAT }, uSpin: { value: 0 }, uSunI: { value: 0.5 }, uP: { value: RESET_P },
      uMode: { value: 0 }, uFlow: { value: 1 }, uDetail: { value: 0.8 }, uTexOK: { value: 0 }, uAur: { value: 1 }, uRingOn: { value: 1 }, uAxis: { value: 0 }, uJunoL: { value: 0 },
      uVis: { value: grey }, uIR: { value: grey }, uVtx: { value: grey },
      uBand: { value: bandTexture(windAt, turbAt) }, uMean: { value: meanTexture(null) }, uZone: { value: zoneTexture(windAt) },
      uClk: { value: new THREE.Vector4() },
      uG: { value: new THREE.Vector4() }, uGA: { value: new THREE.Vector4(GRS.a * D2R, GRS.b * D2R, GRS_G.L, 0) },
      uGV: { value: new THREE.Vector4(GRS.vmax, GRS.rm, GRS.inflow, 0) },
      uB: { value: new THREE.Vector4() }, uBA: { value: new THREE.Vector4(BA.a * D2R, BA.b * D2R, BA_G.L, 0) },
      uBV: { value: new THREE.Vector4(BA.vmax, BA.rm, BA.inflow, 0) },
      uMoonA: { value: new THREE.Vector4() },
      uDbg: { value: +(typeof location !== 'undefined' && new URLSearchParams(location.search).get('jupDbg')) || 0 },
    };
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    // collar angular speed used for the patches' pre-rotation (ρ ≈ 0.85)
    U.uG.value.w = GRS_G.om(0.85); U.uB.value.w = BA_G.om(0.85);

    const ready = Promise.all([
      retry(loadImageTex, ASSET('jupiter_vis.jpg'), { aniso }),
      retry(loadImageTex, ASSET('jupiter_ir.jpg'), { aniso: 4 }),
      retry(loadImageTex, ASSET('jupiter_vortices.jpg'), { aniso: 4 }),
      retry(loadJSON, ASSET('jupiter_profiles.json')),
    ]).then(([vis, ir, vtx, prof]) => {
      vtx.wrapS = THREE.ClampToEdgeWrapping;
      U.uVis.value = vis; U.uIR.value = ir; U.uVtx.value = vtx;
      const tab = (arr, lat) => { const x = (lat - prof.lat0) / prof.dlat, i = Math.max(0, Math.min(arr.length - 2, Math.floor(x))), f = Math.max(0, Math.min(1, x - i)); return arr[i] * (1 - f) + arr[i + 1] * f; };
      windAt = lat => tab(prof.wind, lat);
      turbAt = lat => tab(prof.turb, lat);
      U.uBand.value = bandTexture(windAt, turbAt);
      U.uZone.value = zoneTexture(windAt);
      U.uMean.value = meanTexture(prof);
      for (const t of [vis, ir, vtx]) R.initTexture(t);
      U.uTexOK.value = 1;
    });

    const st = {
      W2B: U.uW2B.value, sun: U.uSunDir.value, pole: new V3(), lastSim: null, T: null, spin: 0,
      mode: null, spinOff: 0, moonOff: [0, 0, 0, 0], nextMoon: 0, fly: null, entered: false,
    };
    // the spin angle for a viewing mode at time T (without offset)
    const spinBase = (mode, T) => mode === 'spin' ? OMEGA * T : mode === 'grs' ? -vLon(GRS, GRS_G, T) : 0;
    const clock = (ctx) => {
      const p = ctx.params, sim = ctx.time.sim;
      if (st.T === null) { st.T = sim * p.lapse * 60; st.lastSim = sim; }
      const dsim = sim - st.lastSim; st.lastSim = sim;
      if (dsim > 0 && dsim < 10) st.T += dsim * p.lapse * 60;
      else if (dsim !== 0) st.T = sim * p.lapse * 60;     // time slider jumps / snapshot: restart from sim
      orient(st, OBLIQ, SUBSOLAR, p.sunAz);
      if (st.mode !== p.track) {   // keep the planet where it is when switching frames
        if (st.mode !== null) st.spinOff = st.spin - spinBase(p.track, st.T);
        st.mode = p.track;
      }
      st.spin = spinBase(st.mode, st.T) + st.spinOff;
    };
    st.clock = clock;
    // world direction of a body-fixed point
    const worldOf = (latD, lonE) => latlonJS(latD, lonE).applyAxisAngle(new V3(0, 1, 0), st.spin).applyMatrix3(st.W2B.clone().transpose());
    const faceFeature = (latD, lonE, dirW) => {
      const qf = latlonJS(latD, lonE), cb = dirW.clone().applyMatrix3(st.W2B);
      const ds = Math.atan2(cb.x, cb.z) - Math.atan2(qf.x, qf.z) - st.spin;
      st.spinOff += ds; st.spin += ds;
    };
    st.subPoint = (dirW) => {
      const d = dirW.clone().normalize().applyMatrix3(st.W2B);
      const k = 1 / (1 - FLAT), s = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
      return bodyLatLon(d.multiplyScalar(s), k, st.spin);
    };
    const flyTo = (ctx, pos, ms = 1600) => {
      st.fly = { from: ctx.camera.position.clone(), to: pos.clone(), t0: performance.now(), ms };
    };
    function placeShadow(i, p) {
      const A = [5.905, 9.397, 14.99, 26.37][i];
      const s = st.sun.clone().applyMatrix3(st.W2B);
      let best = 1e9, bestA = 0;
      for (let k = 0; k < 2880; k++) {
        const a = k / 2880 * TAU, v = new V3(Math.sin(a), 0, Math.cos(a)).multiplyScalar(A).sub(p), tca = v.dot(s);
        if (tca <= 0) continue;
        const d = v.addScaledVector(s, -tca).length();
        if (d < best) { best = d; bestA = a; }
      }
      st.moonOff[i] = bestA - (MOONS[i][0] + TAU * st.T / MOONS[i][1]);
      const m = new V3(Math.sin(bestA), 0, Math.cos(bestA)).multiplyScalar(A);
      const hs = hitBody(m.applyMatrix3(st.W2B.clone().transpose()), st.sun.clone().negate(), st.W2B);
      return hs.t !== null ? bodyLatLon(hs.ro.clone().addScaledVector(hs.rd, hs.t), hs.k, st.spin).lat : 0;
    }

    const inst = {
      material, ready, st,
      enter() { st.pendingEnter = true; },
      update(dt, ctx) {
        const p = ctx.params;
        clock(ctx);
        if (st.pendingEnter) {
          st.pendingEnter = false;
          if (!st.entered) {
            // first visit: GRS on the visible disc, a little east of the central meridian, and Io's shadow on the disc
            const c = def.camera, dir = new V3().setFromSphericalCoords(1, c.polar * D2R, c.azimuth * D2R);
            faceFeature(GRS.lat, vLon(GRS, GRS_G, st.T) - 0.45, dir);
            const d = dir.clone().add(st.sun).normalize().applyMatrix3(st.W2B);
            const k = 1 / (1 - FLAT), sc = 1 / Math.sqrt(d.x * d.x + d.y * d.y * k * k + d.z * d.z);
            placeShadow(0, d.multiplyScalar(sc));
            st.entered = true;
            // snapshot hook: ?jupShot=grs|ba|np|sp[:distance] puts the camera there instantly
            const shot = typeof location !== 'undefined' && new URLSearchParams(location.search).get('jupShot');
            if (shot) {
              const [what, dd] = shot.split(':');
              ({ grs: inst.goGRS, ba: inst.goBA, np: c => inst.goPole(c, 1), sp: c => inst.goPole(c, -1) })[what]?.(ctx);
              if (st.fly) {
                const to = st.fly.to.clone(); if (dd) to.setLength(+dd);
                ctx.camera.position.copy(to); ctx.camera.lookAt(ctx.controls.target); ctx.controls.update(); st.fly = null;
              }
            }
          }
        }
        if (st.fly) {
          const f = Math.min(1, (performance.now() - st.fly.t0) / st.fly.ms), e = f * f * (3 - 2 * f);
          const a = st.fly.from, b = st.fly.to;
          const ra = a.length(), rb = b.length();
          const dir = a.clone().normalize().lerp(b.clone().normalize(), e).normalize();
          ctx.camera.position.copy(dir.multiplyScalar(ra + (rb - ra) * e));
          ctx.camera.lookAt(ctx.controls.target); ctx.controls.update();
          if (f >= 1) st.fly = null;
        }
        const T = st.T;
        U.uSpin.value = wrap(st.spin);
        const cyc = T / RESET_P;
        U.uClk.value.set(cyc % 4096, T % (RESET_P * 4096), (T / 86400) % 1000, 0);
        U.uG.value.x = wrap(vLon(GRS, GRS_G, T)); U.uG.value.y = GRS.lat * D2R; U.uG.value.z = vDrift(GRS, GRS_G, T);
        U.uB.value.x = wrap(vLon(BA, BA_G, T)); U.uB.value.y = BA.lat * D2R; U.uB.value.z = vDrift(BA, BA_G, T);
        U.uMode.value = { vis: 0, hst: 1, enh: 2, ch4: 3, uv: 4 }[p.band] ?? 1;
        U.uFlow.value = p.flow ? 1 : 0;
        U.uDetail.value = p.detail;
        U.uAur.value = p.aurora ? 1 : 0;
        U.uJunoL.value = p.junoLight ? 1 : 0;
        U.uRingOn.value = p.rings ? 1 : 0;
        for (let i = 0; i < 4; i++) U.uMoonA.value.setComponent(i, wrap(MOONS[i][0] + TAU * T / MOONS[i][1] + st.moonOff[i]));
      },
      hud(ctx) {
        clock(ctx);
        const sc = st.subPoint(ctx.camera.position);
        const cam = ctx.camera.position.clone().normalize();
        const phase = Math.acos(THREE.MathUtils.clamp(cam.dot(st.sun), -1, 1)) * R2D;
        const hPerS = (ctx.bhGlobals?.timeScale ?? 6) * ctx.params.lapse * 60 / 3600;
        const days = st.T / 86400;
        const date = new Date(EPOCH + st.T * 1000);
        const gW = (360 - wrap(vLon(GRS, GRS_G, st.T)) * R2D) % 360;
        const rows = [
          ['Tua nhanh', `1 giây ≈ ${hPerS >= 10 ? vn(hPerS, 0) : vn(hPerS)} giờ trên Sao Mộc`],
          ['Đã trôi qua', `${vn(days, days < 10 ? 2 : 1)} ngày · ${date.getUTCDate()}/${date.getUTCMonth() + 1}/${date.getUTCFullYear()}`],
          ['Vĩ độ dưới camera', latStr(sc.lat)], ['Gió tại vĩ độ đó', windStr(windAt(sc.lat))],
          ['Vết Đỏ Lớn', `kinh độ ${vn(gW, 1)}° T (System III) · viền quay 1 vòng ≈ ${vn(GRS_G.periodEdge / 86400)} ngày`],
          ['Góc pha', vn(phase, 0) + '°'],
        ];
        return rows;
      },
      goGRS(ctx) {
        clock(ctx);
        ctx.params.track = 'grs'; clock(ctx);
        const dirCam = ctx.camera.position.clone().normalize();
        faceFeature(GRS.lat, vLon(GRS, GRS_G, st.T), dirCam);
        const w = worldOf(GRS.lat + 1.0, vLon(GRS, GRS_G, st.T));
        flyTo(ctx, w.multiplyScalar(1.55));
        ctx.syncUI();
        ctx.toast(`Vết Đỏ Lớn: xoáy nghịch rộng ≈ 14 000 km, quay <b>ngược chiều kim đồng hồ</b>; gió ở viền ~${GRS.vmax} m/s nên viền quay một vòng mất ≈ ${vn(GRS_G.periodEdge / 86400)} ngày, còn lõi gần như lặng. Nó trôi chậm về phía tây (~4 m/s) và dao động ±1,2° kinh độ với chu kỳ 90 ngày. Phía tây bắc là vùng rối nơi dòng tia 19,5° N (về tây, ~50 m/s) bị cơn bão chặn lại.`, 10000);
      },
      goBA(ctx) {
        clock(ctx);
        const dirCam = ctx.camera.position.clone().normalize();
        faceFeature(BA.lat, vLon(BA, BA_G, st.T), dirCam);
        flyTo(ctx, worldOf(BA.lat + 1, vLon(BA, BA_G, st.T)).multiplyScalar(1.45));
        ctx.toast('Oval BA (“Vết Đỏ Nhỏ”): sinh ra năm 2000 khi ba xoáy trắng hợp nhất, ngả đỏ năm 2006. Xoáy nghịch rộng ~7 000 km ở 33° N, quay ngược chiều kim đồng hồ.', 8000);
      },
      goPole(ctx, sg) {
        clock(ctx);
        // like Juno at perijove: ~1.3 R_J above the pole, leaning a little toward the Sun. The Sun is only ~2–5°
        // above the polar horizon (obliquity 3°), so — like every JunoCam picture — the exposure is pushed up.
        const side = new V3().crossVectors(st.pole, st.sun).normalize();
        const dir = st.pole.clone().multiplyScalar(sg).addScaledVector(st.sun, 0.07).addScaledVector(side, 0.03).normalize();
        flyTo(ctx, dir.multiplyScalar(1.62));
        ctx.params.junoLight = true; ctx.syncUI?.();
        if (ctx.bhGlobals) ctx.bhGlobals.exposure = Math.max(ctx.bhGlobals.exposure, 1.6);
        ctx.toast(sg > 0
          ? 'Cực Bắc (Juno, 2017): 8 xoáy thuận, mỗi cơn rộng ~4 000 km, xếp thành hình bát giác quanh một cơn ở chính cực — không trôi, không hợp nhất suốt nhiều năm. Giữa chúng là vùng rối “sợi gấp nếp”. Mặt Trời chỉ cao vài độ nên rất tối.'
          : 'Cực Nam (Juno): 5 xoáy thuận quanh một cơn ở cực (năm 2019 có thêm cơn thứ 6 chen vào rồi tan). Các xoáy thuận quay cùng chiều tự quay của hành tinh.', 9000);
      },
      onClick(info, ctx) {
        clock(ctx);
        const h = hitBody(info.origin, info.dir, st.W2B);
        if (h.t === null) return;
        const p = h.ro.clone().addScaledVector(h.rd, h.t);
        const ll = bodyLatLon(p, h.k, st.spin);
        const u = windAt(ll.lat);
        const s = st.sun.clone().applyMatrix3(st.W2B);
        if (ll.n.dot(s) <= 0.02) { ctx.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>, gió ${windStr(u)}. Điểm này đang là ban đêm.`); return; }
        const i = st.nextMoon; st.nextMoon = (i + 1) % 4;
        const trackLat = placeShadow(i, p);
        ctx.toast(`Vĩ độ <b>${latStr(ll.lat)}</b>, gió ${windStr(u)}. Bóng của <b>${MOON_NAMES[i][0]}</b> (đường kính ${fmtInt(MOON_NAMES[i][1])} km) vừa được đặt gần đó — bóng vệ tinh chỉ rơi trên một vệt quanh vĩ độ ${latStr(trackLat)}.`, 7000);
        ctx.audio.ping(520 - i * 70, 0.05, info.ndcX);
      },
    };
    def.__inst = inst;
    return inst;
  },
};

export default [def];
