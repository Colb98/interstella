// Sao Hỏa — Mars from real data, with its weather and seasons in motion.
//
// Surface: USGS Viking colorized global mosaic (colour rebuilt as true-colour Lambert albedo) and
// MGS MOLA MEGDR topography (16 px/° global + a 64 px/° Tharsis–Valles Marineris patch), ray-marched
// as an exaggerated height field with soft relief shadows. One scene unit = the mean radius 3 389.5 km.
//
// Motion (two clocks):
//   · rotation clock S (sols): the planet turns once per sol (24 h 39 min 35 s solar day); everything
//     tied to local solar time follows it — afternoon orographic clouds over the Tharsis volcanoes,
//     morning fog in the canyons, Phobos rising in the west and Deimos crawling across the sky, cloud
//     and dust drift with the zonal winds.
//   · climate clock C (sols since perihelion) = S × "pace": the orbit (Kepler, e = 0.0934) gives Ls,
//     the Sun's declination and distance; Ls drives the seasonal CO₂ caps, the aphelion cloud belt,
//     the polar hoods, the background dust and the dust-storm generator.
// The Sun direction is set from "local time under the camera" (like the other planet scenes), so the
// lighting you ask for holds while you orbit.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';
import { t, dec, locale } from '../../i18n.js';

const D2R = Math.PI / 180;
const RKM = 3389.5;
// Right-handed geography (as on the real planet): body = rotY(−spin)·world, east longitude = atan2(x, z)
const llE = (lat, lon) => [Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R)];
const v3 = a => `vec3(${a.map(x => x.toFixed(6)).join(', ')})`;

// high-resolution MOLA / Viking patch: lon 138°W … 42°W, lat 20°S … 28°N
const PATCH = { lon0: -138, lat0: -20, dlon: 96, dlat: 48, elevW: 3072, globalW: 4096 };
const PRATIO = (PATCH.elevW / PATCH.dlon) / (PATCH.globalW / 360);

// Tharsis / Elysium volcanoes: [lat, lon, radius km, cloud strength]
const VOLCANOES = [
  [18.65, -133.8, 300, 1.0],   // Olympus Mons
  [11.92, -104.08, 190, 1.0],  // Ascraeus Mons
  [1.48, -112.96, 190, 0.9],   // Pavonis Mons
  [-8.26, -120.09, 215, 1.0],  // Arsia Mons
  [40.47, -109.6, 450, 0.6],   // Alba Mons
  [24.8, 146.9, 120, 0.6],     // Elysium Mons
];

/* ================================================================================================
   GLSL
   ================================================================================================ */
const MARS_FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir;
uniform float uRot, uExag, uHMax, uHMin, uTexOK, uTexel, uAtmo, uESun, uSunAU, uHd;
uniform sampler2D uColTex, uElevTex, uPatchCol, uPatchElev;
uniform float uS, uTauBg, uCloud, uACB, uHoodN, uHoodS, uOro, uAMEC, uFog, uJetN, uJetS, uIceSupp;
uniform float uCapN, uCapS, uCapOff, uCryptic, uFrost;
uniform vec4 uStC[4], uStP[4];
uniform vec3 uPhoP, uDeiP;
uniform mat3 uPhoM, uDeiM;
uniform float uMoons;

const float RKM = ${RKM.toFixed(1)};
const float RTOP0 = 1.0 + 120.0 / RKM;
const float HCL = 16.0 / RKM;                     // water-ice cloud deck, km above the smoothed ground
const vec3 DUST_ALB = vec3(0.45, 0.25, 0.11);     // multiple-scattering albedo of a thick dust veil
const vec2 P0 = vec2(${((PATCH.lon0 + 180) / 360).toFixed(7)}, ${((PATCH.lat0 + 90) / 180).toFixed(7)});
const vec2 PS = vec2(${(PATCH.dlon / 360).toFixed(7)}, ${(PATCH.dlat / 180).toFixed(7)});
const float PRATIO = ${PRATIO.toFixed(5)};
const vec3 CAPS_RES = ${v3(llE(-87.0, -45.0))};    // centre of the south residual cap (offset from the pole)
const vec3 VOL[6] = vec3[6](${VOLCANOES.map(v => v3(llE(v[0], v[1]))).join(', ')});
const float VOLR[6] = float[6](${VOLCANOES.map(v => (v[2] / RKM).toFixed(5)).join(', ')});
const float VOLA[6] = float[6](${VOLCANOES.map(v => v[3].toFixed(2)).join(', ')});
const vec3 PHO_AX = vec3(${[13.0, 11.4, 9.1].map(x => (x / RKM).toExponential(5)).join(', ')});   // Phobos semi-axes
const vec3 DEI_AX = vec3(${[7.8, 6.0, 5.1].map(x => (x / RKM).toExponential(5)).join(', ')});     // Deimos

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
// rotate a body vector about the spin axis so that its east longitude grows by a
vec3 spinE(vec3 p, float a){ float c = cos(a), s = sin(a); return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); }

// ---- atmosphere: Chapman column + 14-sample single scattering (molecules + dust) ----------------
float chapman(float X, float h, float mu){
  float c = sqrt(1.5708 * (X + h));
  if (mu >= 0.0) return c / (c * mu + 1.0) * exp(-h);
  float x0 = sqrt(max(1.0 - mu * mu, 0.0)) * (X + h);
  float c0 = sqrt(1.5708 * x0);
  return 2.0 * c0 * exp(min(X - x0, 60.0)) - c / (1.0 - c * mu) * exp(-h);
}
float phaseR(float mu){ return 0.0596831 * (1.0 + mu * mu); }
vec3 phaseCS(float mu, vec3 g){
  vec3 g2 = g * g;
  return 0.1193662 * (1.0 - g2) * (1.0 + mu * mu) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, vec3(1.5)));
}
float phaseHG(float mu, float g){ float g2 = g * g; return 0.0795775 * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5); }
struct Atm { float R; float Hr; float Hm; vec3 bR; vec3 bMs; vec3 bMe; vec3 g; };
Atm atmM(float tau){
  Atm A;
  A.R = 1.0; A.Hr = 11.1 / RKM; A.Hm = uHd;
  // CO₂ Rayleigh: Earth's coefficients (10⁻⁶ m⁻¹) × density ratio 0.016 × CO₂/air cross-section ≈ 2.4
  A.bR = vec3(5.8, 13.5, 33.1) * 0.039 * 3.3895 * uAtmo;
  // dust: vertical optical depth tau; single-scattering albedo falls toward the blue (ferric oxide)
  // ~1.5 µm grains: larger size parameter in the blue → stronger forward scattering → blue sunsets
  A.g = vec3(0.58, 0.68, 0.84);
  vec3 w0 = vec3(0.97, 0.88, 0.70);
  A.bMs = vec3(tau / uHd) * w0 * uAtmo;
  // δ-scaled extinction (f = g²): light scattered into the narrow forward peak keeps travelling with the
  // beam, so images of the ground are dimmed by (1 − ω g²)·τ rather than the full τ
  A.bMe = vec3(tau / uHd) * (1.0 - w0 * A.g * A.g) * uAtmo;
  return A;
}
vec3 sunTrans(Atm A, float h, float mu){
  float oR = A.Hr * chapman(A.R / A.Hr, h / A.Hr, mu);
  float oM = A.Hm * chapman(A.R / A.Hm, h / A.Hm, mu);
  return exp(-(A.bR * oR + A.bMe * oM));
}
vec3 atmosphere(vec3 ro, vec3 rd, float t0, float t1, vec3 L, Atm A, out vec3 T){
  const int N = 14;
  float dt = (t1 - t0) / float(N);
  vec3 sR = vec3(0.0), sM = vec3(0.0);
  float oR = 0.0, oM = 0.0;
  float XR = A.R / A.Hr, XM = A.R / A.Hm;
  for (int i = 0; i < N; i++){
    vec3 p = ro + rd * (t0 + (float(i) + 0.5) * dt);
    // (clamped at the datum: with exaggerated relief, deep basins would otherwise sit in absurdly dense air)
    float r = max(length(p), A.R), h = r - A.R;
    float dR = exp(-h / A.Hr) * dt, dM = exp(-h / A.Hm) * dt;
    oR += 0.5 * dR; oM += 0.5 * dM;
    float mu = dot(p, L) / r;
    float lR = A.Hr * chapman(XR, h / A.Hr, mu), lM = A.Hm * chapman(XM, h / A.Hm, mu);
    vec3 att = exp(-(A.bR * (oR + lR) + A.bMe * (oM + lM)));
    sR += dR * att; sM += dM * att;
    oR += 0.5 * dR; oM += 0.5 * dM;
  }
  T = exp(-(A.bR * oR + A.bMe * oM));
  float nu = dot(rd, L);
  return sR * A.bR * phaseR(nu) + sM * A.bMs * phaseCS(nu, A.g);
}
vec3 sunDisk(vec3 rd, vec3 L, float dAU){
  float c = dot(rd, L), a = acos(clamp(c, -1.0, 1.0)), r = 0.00465 / dAU;
  return bb(5772.0) * (smoothstep(r * 1.15, r * 0.85, a) * 400.0 + pow(max(c, 0.0), 3000.0 * dAU * dAU) * 4.0);
}

// ---- data access (equirectangular, u = east longitude from −180°, v = latitude from the south) ----
vec2 uvOf(vec3 p){ return vec2(atan(p.x, p.z) * (0.5 / PI) + 0.5, asin(clamp(p.y, -1.0, 1.0)) / PI + 0.5); }
vec2 uvD(vec3 p, vec3 dp){
  float q = max(p.x * p.x + p.z * p.z, 1e-6);
  return vec2((p.z * dp.x - p.x * dp.z) / q * (0.5 / PI), dp.y / (PI * sqrt(q)));
}
float patchW(vec2 uv, out vec2 q){
  q = (uv - P0) / PS;
  vec2 m = min(q, 1.0 - q);
  return smoothstep(0.0, 0.05, min(m.x, m.y));
}
// MOLA elevation (km above the areoid) at a global-texture LOD
float elevAt(vec2 uv, float lod){
  vec2 q; float w = patchW(uv, q);
  float g = textureLod(uElevTex, uv, max(lod, 0.0)).r;
  if (w > 0.0) g = mix(g, textureLod(uPatchElev, q, max(lod + log2(PRATIO), 0.0)).r, w);
  return g;
}
float texelAt(vec2 uv){ vec2 q; return patchW(uv, q) > 0.5 ? uTexel / PRATIO : uTexel; }
vec3 colorAt(vec2 uv, vec2 gx, vec2 gy){
  vec3 c = textureGrad(uColTex, uv, gx, gy).rgb;
  vec2 q; float w = patchW(uv, q);
  if (w > 0.0) c = mix(c, textureGrad(uPatchCol, q, gx / PS, gy / PS).rgb, w);
  return c;
}
// height above the deepest point (Hellas floor), km
float hgtKm(vec3 pb, float lod){ return elevAt(uvOf(pb), lod) - uHMin; }

// ---- relief ray march (uniform prefiltered steps + secant refinement) -----------------------------
float marchRelief(vec3 ro, vec3 rd, float tA, float tB, bool onBase, mat3 R, float k, float rb, float pixAng){
  vec3 pm = normalize(ro + rd * (0.5 * (tA + tB)));
  float sinT = max(length(cross(rd, pm)), 0.02);
  float lat = (tB - tA) * sinT;
  float tex = texelAt(uvOf(R * pm));
  float n = clamp(ceil(lat / max(1.6 * tex, tA * pixAng)), 2.0, 64.0);
  float dt = (tB - tA) / n;
  float lodS = log2(max(dt * sinT, 1e-7) / uTexel) - 0.5;
  float tP = tA, dP = 1.0;
  for (int i = 0; i <= 64; i++){
    if (float(i) > n) break;
    float t = tA + dt * float(i);
    vec3 q = ro + rd * t; float r = length(q);
    float lod = max(lodS, log2(t * pixAng / uTexel));
    float d = r - rb - k * hgtKm(R * (q / r), lod);
    if (d <= 0.0 || (onBase && float(i) >= n)){
      if (i == 0) return t;
      for (int j = 0; j < 2; j++){
        float tc = tP + (t - tP) * dP / max(dP - d, 1e-9);
        vec3 qc = ro + rd * tc; float rc = length(qc);
        float dc = rc - rb - k * hgtKm(R * (qc / rc), lod);
        if (dc <= 0.0){ t = tc; d = dc; } else { tP = tc; dP = dc; }
      }
      return tP + (t - tP) * dP / max(dP - d, 1e-9);
    }
    tP = t; dP = d;
  }
  return -1.0;
}
float reliefShadow(vec3 pw, vec3 L, mat3 R, float k, float rb, float ds, float shellTop){
  float t = ds * 1.5, res = 1.0;
  for (int i = 0; i < 22; i++){
    vec3 q = pw + L * t; float r = length(q);
    if (r > shellTop) break;
    float h = rb + k * hgtKm(R * (q / r), log2(max(t * 0.25, ds) / uTexel));
    res = min(res, (r - h) / (t * 0.012 + ds * 0.25));
    if (res < -0.5) break;
    t *= 1.3;
  }
  return smoothstep(-0.5, 1.0, res);
}

// ---- winds and evolving, advected noise -----------------------------------------------------------
// zonal wind (m/s, + eastward): tropical easterlies, westerly jets strongest in the winter hemisphere
float windU(float lat){
  float d = lat / 0.0174533;
  return -7.0 * exp(-pow(d / 22.0, 2.0)) + uJetN * exp(-pow((d - 55.0) / 14.0, 2.0)) + uJetS * exp(-pow((d + 55.0) / 14.0, 2.0));
}
// Flow-map noise: two layers advected by the zonal wind over a period, half a period apart, each
// regenerated (new random offset) while its weight is zero. Shear between latitudes + regeneration
// make cloud and dust patterns drift and evolve rather than slide rigidly.
float flowFbm(vec3 p, float period, float f, float stretch, int oct, float seed){
  float lat = asin(clamp(p.y, -1.0, 1.0));
  float w = windU(lat) * (88775.0 / 3.3895e6) / max(cos(lat), 0.25);      // rad of longitude per sol
  w += 0.35 * sin(3.0 * atan(p.x, p.z) + 2.0 * lat * 6.0) * 0.06;          // a little wave meandering
  float ph = uS / period + seed;
  float a0 = fract(ph), a1 = fract(ph + 0.5);
  vec3 o0 = hash33(vec3(floor(ph), seed, 1.7)) * 37.0, o1 = hash33(vec3(floor(ph + 0.5), seed, 4.3)) * 37.0;
  vec3 q0 = spinE(p, -w * a0 * period), q1 = spinE(p, -w * a1 * period);
  vec3 sc = vec3(1.0, stretch, 1.0) * f;
  float n0 = fbm(q0 * sc + o0, oct), n1 = fbm(q1 * sc + o1, oct);
  float b = abs(2.0 * a0 - 1.0);
  return mix(n0, n1, b) / sqrt((1.0 - b) * (1.0 - b) + b * b);
}

float localHour(vec3 pb, vec3 Lb){
  float h = 12.0 + (atan(pb.x, pb.z) - atan(Lb.x, Lb.z)) * (12.0 / PI);
  return mod(h, 24.0);
}

// ---- water-ice clouds: aphelion belt, polar hoods, afternoon orographic clouds -------------------
float iceTau(vec3 p, float hour, int oct){
  if (uCloud <= 0.0) return 0.0;
  float lat = asin(clamp(p.y, -1.0, 1.0));
  float base = flowFbm(p, 0.7, 5.0, 3.0, oct, 3.1);       // streaky (stretched N–S → elongated E–W)
  float tau = 0.0;
  float aft = smoothstep(10.0, 13.0, hour) * smoothstep(19.5, 16.5, hour);
  float morn = smoothstep(4.8, 6.2, hour) * smoothstep(10.5, 8.5, hour);
  // aphelion cloud belt (Ls ~40–150, 10°S–30°N), thicker in the afternoon
  float band = smoothstep(-0.25, -0.05, lat) * smoothstep(0.62, 0.32, lat);
  tau += uACB * band * (0.55 + 0.45 * aft) * (0.04 + 0.14 * smoothstep(-0.2, 0.95, base));
  // polar hoods (autumn / winter): banded, wavy
  float waves = 0.85 + 0.15 * sin(lat * 40.0 + base * 5.0);
  tau += uHoodN * smoothstep(0.75, 1.05, lat) * waves * (0.3 + 0.7 * smoothstep(-0.5, 0.6, base)) * 0.45;
  tau += uHoodS * smoothstep(-0.75, -1.05, lat) * waves * (0.3 + 0.7 * smoothstep(-0.5, 0.6, base)) * 0.3;
  // orographic: plumes downwind (west / north-west) of the big volcanoes; lee waves downstream
  for (int i = 0; i < 6; i++){
    vec3 c = VOL[i]; float r = VOLR[i];
    vec3 d = p - c;
    if (dot(d, d) > 100.0 * r * r) continue;
    vec3 ec = normalize(vec3(c.z, 0.0, -c.x)), nc = cross(c, ec);
    float x = dot(d, ec) / r, y = dot(d, nc) / r;
    float al = -x * 0.94 + y * 0.34, cr = x * 0.34 + y * 0.94;
    float amec = i == 3 ? uAMEC * morn : 0.0;          // Arsia Mons elongated cloud (southern spring dawn)
    float Lp = mix(2.2, 9.0, amec);
    float wd = 0.5 + 0.22 * max(al, 0.0) * (1.0 - 0.7 * amec);
    float plume = smoothstep(-1.15, -0.3, al) * exp(-max(al, 0.0) / Lp) * exp(-cr * cr / (wd * wd));
    float lee = mix(0.55 + 0.45 * cos(al * 6.0 - uS * 9.0 + 2.0 * base), 1.0, 0.75 * amec);   // the AMEC is a long streak, not wave rows
    float amp = uOro * aft * VOLA[i] + amec * 1.6;
    float cell = 1.0 - worley(p / r * 1.4 + vec3(0.0, 0.0, uS * 0.8)).x;   // convective cells on the flanks
    tau += amp * plume * lee * smoothstep(-0.35, 0.55, base + 0.25 * cell) * (0.35 + 0.65 * cell) * 0.55;
  }
  return tau * uCloud * uIceSupp;
}

// ---- dust storms: up to four cells (centre, radius, peak τ) with lobed, billowing edges ---------
float stormTau(vec3 p, out float bill){
  bill = 0.0;
  float near = 0.0;
  for (int i = 0; i < 4; i++){
    if (uStP[i].x <= 0.0) continue;
    float d = acos(clamp(dot(p, uStC[i].xyz), -1.0, 1.0));
    near = max(near, step(d, uStC[i].w * 1.8 + 0.08));
  }
  if (near == 0.0) return 0.0;
  float en = flowFbm(p, 0.45, 2.6, 1.0, 3, 7.7);
  bill = flowFbm(p, 0.22, 6.0, 1.3, 4, 9.1);
  float tau = 0.0;
  for (int i = 0; i < 4; i++){
    if (uStP[i].x <= 0.0) continue;
    float d = acos(clamp(dot(p, uStC[i].xyz), -1.0, 1.0));
    float R0 = uStC[i].w;
    float glob = smoothstep(1.5, 3.0, R0);                     // a planet-encircling storm has no edge
    // lobed, ragged margin: large-scale lobes plus billow-scale fingers displace the edge
    float soft = 0.4 * R0 + 0.03;
    float r = R0 * (1.0 + 0.3 * en * (1.0 - glob)) + 0.35 * soft * bill;
    // optical depth falls off smoothly over the margin (no threshold): thin haze outside, the
    // densest dust well inside, where the lifting fronts are
    float x = clamp((r + 0.6 * soft - d) / (1.6 * soft), 0.0, 1.0);
    float w = x * x * (3.0 - 2.0 * x);
    w *= w;
    float core = smoothstep(0.35, 0.85, x);
    float vary = mix(0.45 + 0.55 * (0.5 + 0.5 * bill), 0.75 + 0.25 * bill, glob);
    tau += uStP[i].x * w * vary * (1.0 + 0.35 * core * max(bill, 0.0) * (1.0 - glob));
  }
  return tau;
}

// ---- Phobos & Deimos ------------------------------------------------------------------------------
float moonHit(vec3 ro, vec3 rd, vec3 P, mat3 M, vec3 ax, out vec3 nW){
  vec3 o = M * (ro - P), d = M * rd;
  vec3 os = o / ax, ds = d / ax;
  float a = dot(ds, ds), b = dot(os, ds), c = dot(os, os) - 1.0;
  float h = b * b - a * c;
  nW = vec3(0.0);
  if (h < 0.0) return -1.0;
  float t = (-b - sqrt(h)) / a;
  if (t < 0.0) return -1.0;
  vec3 q = (o + d * t) / ax;
  vec3 nl = normalize(q / ax);
  // cratered, lumpy: perturb the normal (Stickney-scale and smaller craters)
  vec3 w = worley(q * 2.2).xxy;
  nl = normalize(nl + 0.35 * vec3(snoise(q * 2.0), snoise(q * 2.0 + 5.1), snoise(q * 2.0 + 9.7)) + 0.3 * (w.x - 0.4) * nl);
  nW = transpose(M) * nl;
  return t;
}
float moonShadow(vec3 pw, vec3 L, vec3 P, float rM){
  vec3 d = P - pw; float dist = length(d);
  float cs = dot(d, L) / dist;
  if (cs < 0.999) return 0.0;
  float sep = acos(min(cs, 1.0)), aM = rM / dist, aS = 0.00465 / uSunAU;
  return min(aM * aM / (aS * aS), 1.0) * smoothstep(aS + aM, abs(aS - aM), sep);
}

// ---- surface ------------------------------------------------------------------------------------
vec3 marsSurface(vec3 pw, vec3 pb, vec3 rd, vec3 L, vec3 Lb, Atm A, mat3 R, float k, float rb, float fp, vec2 gx, vec2 gy,
                 float shellTop, float sh0, float hour, float tau){
  vec3 n0 = normalize(pw);
  vec2 uv = uvOf(pb);
  float tex = texelAt(uv);
  float ds = max(fp, tex);
  float lod = log2(ds / uTexel);
  float cphi = max(length(pb.xz), 0.02);
  float du = ds / (TAU * cphi), dv = ds / PI;
  float e = elevAt(uv, lod);
  float hE = elevAt(uv + vec2(du, 0.0), lod), hW = elevAt(uv - vec2(du, 0.0), lod);
  float hN = elevAt(uv + vec2(0.0, dv), lod), hS = elevAt(uv - vec2(0.0, dv), lod);
  vec3 east = vec3(pb.z, 0.0, -pb.x) / cphi, north = cross(pb, east);
  vec2 g = vec2(hE - hW, hN - hS) * k / (2.0 * ds);
  vec3 nb = pb - g.x * east - g.y * north;
  // sub-texel relief below the data resolution (amplitude ∝ texel size), only in close-ups
  float fd = smoothstep(0.7, 0.2, fp / tex);
  float rn = 0.0;
  if (fd > 0.0){
    float F = 2.2 / tex, eps = max(fp * 0.7, tex * 0.03);
    rn = fbm(pb * F, 4);
    float rx = fbm((pb + east * eps) * F, 4), ry = fbm((pb + north * eps) * F, 4);
    float amp = (0.004 * uExag + length(g) * 0.15) * tex / 2.2 * fd;
    nb -= (east * (rx - rn) + north * (ry - rn)) / eps * amp;
  }
  vec3 n = transpose(R) * normalize(nb);
  float hGeo = k * e, hTrue = e / RKM;
  float lat = asin(clamp(pb.y, -1.0, 1.0));

  vec3 alb = colorAt(uv, gx, gy) * 0.6;
  // calibrate the Viking-derived albedo (its dark terrains are veiled by the 1970s dust) toward HST /
  // TES contrast: dark basaltic sand red albedo ≈ 0.08 and nearly neutral grey-brown, bright dust 0.37
  if (alb.b < 0.6 * alb.r){
    float r2 = max(0.08 + (alb.r - 0.135) * 1.234, 0.05);
    alb = r2 * mix(vec3(1.0, 0.8, 0.58), alb / max(alb.r, 1e-3), smoothstep(0.1, 0.3, r2));
  }
  if (fd > 0.0) alb *= 1.0 + 0.12 * rn * fd;

  // ---- seasonal CO₂ frost: edge latitude from Ls; pole-facing slopes act as higher latitudes
  float slopeLat = atan(g.y / max(uExag, 1.0));
  float fn = fbm(pb * 18.0, 3) * 0.03 + fbm(pb * 4.0 + 3.0, 2) * 0.035;
  float latN = lat - slopeLat;
  float frN = smoothstep(uCapN - 0.02, uCapN + 0.02, latN + fn);
  vec3 sc = normalize(mix(vec3(0.0, -1.0, 0.0), CAPS_RES, uCapOff));
  float latS = 1.5707963 - acos(clamp(dot(pb, sc), -1.0, 1.0)) + slopeLat;
  float frS = smoothstep(uCapS - 0.02, uCapS + 0.02, latS + fn);
  // the "cryptic" region: translucent slab ice, dark in southern spring (≈ 150°E–300°E)
  float cry = uCryptic * smoothstep(0.6, 0.95, cos(atan(pb.x, pb.z) + 2.27)) * smoothstep(-1.5, -1.2, lat) * frS;
  vec3 frostAlb = mix(vec3(0.62, 0.61, 0.60), vec3(0.26, 0.21, 0.18), cry) * (0.92 + 0.1 * fn / 0.03);
  float frost = max(frN, frS) * uFrost;
  alb = mix(alb, frostAlb, frost);

  // ---- illumination: Minnaert photometry (k = 0.85), relief shadows, dusty skylight
  float mu0s = dot(n0, L);
  float muE = mu0s + sqrt(2.0 * max(hGeo, 0.0)) - sqrt(2.0 * max(hTrue, 0.0));
  vec3 Ts = sunTrans(A, max(hTrue, -0.0025), muE);
  float sh = sh0;
  if (hGeo - k * uHMin > 0.0 && muE > -0.02 && dot(Ts, vec3(1.0)) > 0.003){
    float tanZ = sqrt(max(1.0 - mu0s * mu0s, 0.0)) / max(mu0s, 0.02);
    if ((hGeo - k * uHMin) * tanZ > 0.4 * fp) sh *= reliefShadow(pw, L, R, k, rb, ds, shellTop);
  }
  float mu0 = max(dot(n, L), 0.0), muv = max(dot(n, -rd), 0.08);
  float minn = pow(mu0, 0.85) * pow(muv, -0.15);
  float eC = elevAt(uv, lod + 3.0);
  float ao = clamp(1.0 - (eC - e) * k / (ds * 12.0), 0.55, 1.0) * (0.75 + 0.25 * dot(n, n0));
  vec3 skyAmb = vec3(0.11, 0.07, 0.042) * smoothstep(-0.15, 0.35, mu0s) * (0.3 + 0.7 * (1.0 - exp(-tau))) * ao * uAtmo;
  vec3 col = alb / PI * uESun * (Ts * minn * sh + skyAmb);

  // ---- morning fog: water-ice haze pooled in deep canyons (Valles Marineris, Noctis Labyrinthus,
  // also Hellas) after sunrise, burning off by mid-morning
  float fogT = smoothstep(5.7, 6.6, hour) * smoothstep(10.0, 7.6, hour) * uFog;
  if (fogT > 0.0){
    float dep = elevAt(uv, lod + 4.0) - e;                          // depth below the surroundings, km
    // where it is observed: the Valles Marineris / Noctis Labyrinthus trough system, and the deep
    // floors of Hellas and Argyre
    float lon = atan(pb.x, pb.z) / 0.0174533, latd = lat / 0.0174533;
    float vm = smoothstep(-108.0, -102.0, lon) * smoothstep(-30.0, -38.0, lon) * smoothstep(-21.0, -16.0, latd) * smoothstep(2.0, -2.0, latd);
    float where = max(vm * smoothstep(0.8, 2.8, dep), smoothstep(-4.8, -6.3, e));
    float fogN = flowFbm(pb, 0.3, 70.0, 1.0, 3, 5.5);
    float f = fogT * where * smoothstep(-0.45, 0.35, fogN) * (1.0 - frost) * 0.75;
    float mf = max(mu0s, 0.0);
    vec3 fogCol = vec3(0.92, 0.94, 1.0) * 0.8 / PI * uESun * (Ts * (0.2 + 0.8 * mf) * (0.65 + 0.35 * sh) * (0.7 + 0.6 * phaseHG(dot(rd, L), 0.5) * 4.0 * PI * 0.25) + skyAmb * 1.5);
    col = mix(col, fogCol, clamp(f, 0.0, 0.8));
  }
  return col;
}

// intersection with the water-ice cloud deck draped over heavily smoothed relief (summits poke through)
float cloudHit(vec3 ro, vec3 rd, mat3 R, float k, out float lift){
  lift = 0.0;
  float hcl = HCL * (1.0 + 0.5 * max(uExag * uTexOK - 1.0, 0.0));
  vec2 h = sphereHit(ro, rd, vec3(0.0), 1.0 + hcl + k * 3.0);
  if (h.x > h.y || h.y < 0.0) return -1.0;
  float tt = h.x > 0.0 ? h.x : max(-dot(ro, rd), 0.0);
  for (int i = 0; i < 2; i++){
    vec3 q = normalize(ro + rd * tt);
    lift = k * elevAt(uvOf(R * q), 6.0);
    vec2 g = sphereHit(ro, rd, vec3(0.0), 1.0 + hcl + lift);
    if (g.x > g.y) return -1.0;
    tt = g.x > 0.0 ? g.x : g.y;
  }
  return tt > 0.0 ? tt : -1.0;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  mat3 R = rotY(uRot);
  vec3 Lb = R * L;
  float k = uExag / RKM * uTexOK;
  float rb = 1.0 + k * uHMin;
  float shellTop = 1.0 + k * uHMax + 1e-5;
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  vec2 hs = sphereHit(ro, rd, vec3(0.0), 1.0);
  bool onUnit = hs.x < hs.y && hs.x > 0.0;
  float tS = onUnit ? hs.x : max(-dot(ro, rd), 0.0);
  vec3 pSw = normalize(ro + rd * tS), pS = R * pSw;
  vec3 dpx = dFdx(pS), dpy = dFdy(pS);

  // dust: background climatology + storm cells at the reference point; storm-top relief from the
  // screen-space gradient of the storm optical depth
  float bill;
  float tauS = stormTau(pS, bill);
  float dTx = dFdx(tauS), dTy = dFdy(tauS);
  float tau = uTauBg + tauS;
  Atm A = atmM(tau);

  vec3 col = sky(rd, 1.0) * 0.6;
  // ---- relief ----
  float tG = -1.0;
  vec2 hO = sphereHit(ro, rd, vec3(0.0), shellTop);
  vec2 hb = sphereHit(ro, rd, vec3(0.0), rb);
  bool onBase = hb.x < hb.y && hb.x > 0.0;
  if (hO.x < hO.y && hO.y > 0.0){
    float tA = max(hO.x, 0.0), tB = onBase ? hb.x : hO.y;
    tG = marchRelief(ro, rd, tA, tB, onBase, R, k, rb, pixAng);
  }
  bool ground = tG > 0.0;
  if (!ground) col += sunDisk(rd, L, uSunAU) * 0.35;

  // ---- moons ----
  vec3 nPh, nDe;
  float tM = -1.0; vec3 nM = vec3(0.0); float albM = 0.07;
  if (uMoons > 0.5){
    float t1 = moonHit(ro, rd, uPhoP, uPhoM, PHO_AX, nPh);
    float t2 = moonHit(ro, rd, uDeiP, uDeiM, DEI_AX, nDe);
    if (t1 > 0.0){ tM = t1; nM = nPh; }
    if (t2 > 0.0 && (tM < 0.0 || t2 < tM)){ tM = t2; nM = nDe; albM = 0.068; }
  }

  // ---- water-ice cloud deck ----
  float tC = -1.0, lift = 0.0;
  if (uCloud > 0.001) tC = cloudHit(ro, rd, R, k, lift);
  if (ground && tC > tG) tC = -1.0;
  vec3 pcw = ro + rd * max(tC, 0.0), pcb = R * normalize(pcw);
  float tauC = 0.0;
  if (tC > 0.0) tauC = iceTau(pcb, localHour(pcb, Lb), 4);

  // ---- surface ----
  vec3 bg = col;
  float hourG = 12.0;
  if (ground){
    vec3 pw = ro + rd * tG, pb = R * normalize(pw);
    hourG = localHour(pb, Lb);
    float sh = 1.0;
    // ice-cloud shadows (the deck where the sunward ray crosses it)
    if (uCloud > 0.001 && dot(pw, L) > 0.0){
      float rDeck = 1.0 + HCL * (1.0 + 0.5 * max(uExag - 1.0, 0.0)) + k * elevAt(uvOf(pb), 6.0);
      vec2 hc = sphereHit(pw, L, vec3(0.0), rDeck);
      if (hc.y > 0.0 && dot(pw, pw) < rDeck * rDeck){
        vec3 pq = R * normalize(pw + L * hc.y);
        sh *= exp(-0.8 * iceTau(pq, localHour(pq, Lb), 2));
      }
    }
    // Phobos / Deimos shadows (penumbral: their discs are smaller than the Sun's)
    if (uMoons > 0.5){
      sh *= 1.0 - moonShadow(pw, L, uPhoP, 11.1 / RKM);
      sh *= 1.0 - moonShadow(pw, L, uDeiP, 6.2 / RKM);
    }
    bg = marsSurface(pw, pb, rd, L, Lb, A, R, k, rb, tG * pixAng, uvD(pb, dpx), uvD(pb, dpy), shellTop, sh, hourG, tau);
  }
  // ---- clouds over it ----
  float ca = 0.0;
  if (tauC > 0.002){
    vec3 nc = normalize(pcw);
    float mu0 = dot(nc, L), muV = max(dot(nc, -rd), 0.12);
    ca = 1.0 - exp(-tauC / muV);
    // fade where the (exaggerated) terrain rises close to the deck: mountains poke through, not cut
    float eT = elevAt(uvOf(pcb), 3.0);
    float gap = (length(pcw) - 1.0 - k * eT) * RKM / max(uExag * uTexOK, 1.0);
    ca *= smoothstep(0.0, 4.0, gap);
    vec3 Ts = sunTrans(A, HCL, mu0);
    float lit = max(mu0, 0.0) * 0.8 + 0.2 * smoothstep(-0.12, 0.25, mu0) * max(mu0 + 0.12, 0.0);
    float ph = 0.55 + 2.2 * phaseHG(dot(rd, L), 0.65) * 4.0 * PI * 0.25;
    vec3 cc = vec3(0.86, 0.9, 0.97) * 0.85 / PI * uESun * (Ts * lit * ph + vec3(0.06, 0.045, 0.035) * smoothstep(-0.1, 0.3, mu0));
    bg = mix(bg, cc, ca);
  }
  // ---- moon in front of everything it hits ----
  float tPlanet = ground ? tG : 1e9;
  if (ca > 0.5) tPlanet = min(tPlanet, tC);
  bool moonFront = tM > 0.0 && tM < tPlanet;
  if (moonFront){
    vec3 pm = ro + rd * tM;
    float mu = max(dot(nM, L), 0.0);
    // Mars's shadow (eclipse) on the moon
    vec2 e = sphereHit(pm, L, vec3(0.0), 1.0);
    float ecl = (e.x < e.y && e.y > 0.0) ? 0.0 : 1.0;
    // Mars-shine: the lit planet seen from the moon
    float ms = 0.02 * max(dot(nM, -normalize(pm)), 0.0) * max(dot(normalize(pm), L), 0.0);
    bg = albM / PI * uESun * vec3(1.0, 0.92, 0.85) * (pow(mu, 0.9) * ecl + ms);
  }

  // ---- atmosphere ----
  float RTOP = max(RTOP0, shellTop + 0.004);
  vec2 ha = sphereHit(ro, rd, vec3(0.0), RTOP);
  if (ha.x < ha.y && ha.y > 0.0){
    float t0 = max(ha.x, 0.0);
    float tEnd = ground ? tG : ha.y;
    if (ca > 0.002) tEnd = mix(tEnd, tC, ca);
    if (moonFront) tEnd = min(tEnd, tM);
    float t1 = min(tEnd, ha.y);
    if (t1 > t0){
      vec3 T;
      vec3 S = atmosphere(ro, rd, t0, t1, L, A, T) * uESun;
      // multiple scattering in the dust: a Lambertian veil lit at the reference point, weighted by the
      // light the dust removed (single scattering alone would leave a thick storm black)
      float tc = ground ? tG : max(-dot(ro, rd), 0.0);
      vec3 pr = normalize(ro + rd * tc);
      float muR = dot(pr, L);
      float lit = max(muR, 0.0) * 0.8 + 0.2 * smoothstep(-0.15, 0.25, muR) * max(muR + 0.15, 0.0);
      // storm-top relief: tilt the veil normal by the gradient of τ (tangent-plane solve from screen
      // derivatives); towering dust fronts catch the low Sun and cast their own shade
      if (tauS > 0.05){
        float cphi = max(length(pS.xz), 0.02);
        vec3 eb = vec3(pS.z, 0.0, -pS.x) / cphi, nbv = cross(pS, eb);
        float a11 = dot(dpx, eb), a12 = dot(dpx, nbv), a21 = dot(dpy, eb), a22 = dot(dpy, nbv);
        float det = a11 * a22 - a12 * a21;
        if (abs(det) > 1e-12){
          vec2 gr = vec2(a22 * dTx - a12 * dTy, -a21 * dTx + a11 * dTy) / det;   // dτ/d(east, north) per radian
          vec3 nt = transpose(R) * normalize(pS - (eb * gr.x + nbv * gr.y) * (0.008 / (1.0 + tauS)));
          float l2 = max(dot(nt, L), 0.0) * 0.8 + 0.2 * smoothstep(-0.15, 0.25, dot(nt, L)) * max(dot(nt, L) + 0.15, 0.0);
          lit = mix(lit, l2, smoothstep(0.05, 1.5, tauS));
        }
        lit *= 0.9 + 0.2 * bill / (1.0 + 0.15 * tauS);
      }
      float ext = 1.0 - dot(T, vec3(0.3333));
      vec3 msv = DUST_ALB / PI * uESun * lit * ext * pow(1.0 - exp(-0.4 * tau * uAtmo), 1.5);
      bg = S + T * bg + msv;
    }
  }
  col = bg;
  gl_FragColor = vec4(col, 1.0);
}`;

/* ================================================================================================
   Orbit, seasons, climatology (CPU)
   ================================================================================================ */
const YEAR_SOLS = 668.6;            // Mars year in sols
const ECC = 0.0934, A_AU = 1.52368, LS_PERI = 251.0, OBLIQ = 25.19;
const SOL_SIM = 2 * Math.PI / 0.004; // simulation seconds per sol (one turn ≈ 4.4 min at the default 6× time speed)
const wrap360 = a => ((a % 360) + 360) % 360;
const wrap180 = a => ((a + 180) % 360 + 360) % 360 - 180;
// climate time C (sols since perihelion) → Ls, Sun distance
function orbitAt(C) {
  const M = 2 * Math.PI * (((C / YEAR_SOLS) % 1) + 1) % (2 * Math.PI);
  let E = M;
  for (let i = 0; i < 8; i++) E -= (E - ECC * Math.sin(E) - M) / (1 - ECC * Math.cos(E));
  const nu = 2 * Math.atan2(Math.sqrt(1 + ECC) * Math.sin(E / 2), Math.sqrt(1 - ECC) * Math.cos(E / 2));
  return { Ls: wrap360(nu / D2R + LS_PERI), r: A_AU * (1 - ECC * Math.cos(E)) };
}
function climateOfLs(Ls) {
  const nu = wrap360(Ls - LS_PERI) * D2R;
  const E = 2 * Math.atan2(Math.sqrt(1 - ECC) * Math.sin(nu / 2), Math.sqrt(1 + ECC) * Math.cos(nu / 2));
  const M = E - ECC * Math.sin(E);
  return ((M / (2 * Math.PI)) % 1 + 1) % 1 * YEAR_SOLS;
}
const declOf = Ls => Math.asin(Math.sin(OBLIQ * D2R) * Math.sin(Ls * D2R)) / D2R;
// piecewise-linear periodic table look-up over Ls
function lsTable(tab, Ls) {
  const x = wrap360(Ls);
  for (let i = 0; i < tab.length - 1; i++) {
    const [a, va] = tab[i], [b, vb] = tab[i + 1];
    if (x >= a && x <= b) return va + (vb - va) * (x - a) / (b - a);
  }
  return tab[0][1];
}
// Seasonal cap edge latitude (|deg|), after MGS TES / MOC and MRO MCS recession / advance curves
// (James & Cantor 2001; Piqueux et al. 2015). The north seasonal cap forms under the polar hood in
// autumn and retreats to the residual water-ice cap (~80°N) by Ls ≈ 90; the south cap reaches ~50°S
// in late winter and shrinks to the small, offset CO₂ residual cap by Ls ≈ 300.
const CAP_N = [[0, 59], [20, 63], [40, 68], [60, 74], [80, 79], [100, 84], [160, 84], [175, 75], [200, 64], [240, 58], [270, 56], [300, 55], [330, 56], [360, 59]];
const CAP_S = [[0, 85], [20, 76], [40, 66], [70, 58], [100, 53], [130, 51], [160, 52], [180, 55], [200, 60], [220, 66], [240, 72], [260, 78], [280, 83], [300, 86.3], [330, 86.3], [360, 85]];
// background (non-storm) visible dust optical depth: clear aphelion season, dusty perihelion season
// with the usual regional-storm peaks near Ls 230 and 320 (after Montabone et al. 2015, ×~2.6 for visible)
const tauClim = Ls => {
  const d1 = wrap180(Ls - 235), d2 = wrap180(Ls - 320);
  return 0.32 + 0.3 * Math.exp(-((d1 / 38) ** 2)) + 0.25 * Math.exp(-((d2 / 16) ** 2));
};
const win = (Ls, a, b, ramp) => {  // 0→1 between a and b (deg, with wrap), smooth ramps
  const s = x => Math.min(1, Math.max(0, x)), sm = x => { x = s(x); return x * x * (3 - 2 * x); };
  const x = wrap360(Ls - a), w = wrap360(b - a);
  return x > w ? 0 : sm(x / ramp) * sm((w - x) / ramp);
};
// Viking Lander-like seasonal pressure cycle at the datum (~25 % of the CO₂ freezes onto the caps)
const pDatum = Ls => 6.1 * (1 + 0.11 * Math.cos((Ls - 250) * D2R) + 0.05 * Math.cos(2 * (Ls - 55) * D2R));

/* ---- dust-storm generator ------------------------------------------------------------------------
   Deterministic in climate time: every 2.5-sol slot may spawn a storm at one of the observed genesis
   regions (weights by season), as a local (most) or regional storm; storms drift with the winds,
   grow, then decay. The "2018" action adds a scripted planet-encircling event. */
const hashf = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const GENESIS = [
  // lat, lon, season weight, drift (deg/sol east, deg/sol north)
  { lat: 52, lon: -30, w: Ls => win(Ls, 170, 400, 25), u: 28, v: -4.5 },   // Acidalia → Chryse "flushing" storms
  { lat: 48, lon: 115, w: Ls => win(Ls, 170, 400, 25), u: 30, v: -2 },     // Utopia frontal storms
  { lat: 52, lon: -160, w: Ls => win(Ls, 180, 390, 25), u: 30, v: -2 },    // Arcadia
  { lat: -38, lon: 68, w: Ls => win(Ls, 160, 330, 25) + 0.4 * win(Ls, 0, 70, 15), u: 6, v: 1.0 },  // Hellas rim
  { lat: -50, lon: -30, w: Ls => win(Ls, 190, 300, 20), u: 14, v: 1.5 },   // Noachis / Argyre
  { lat: -60, lon: 0, w: Ls => win(Ls, 180, 260, 20), u: 16, v: 0, cap: true },   // receding south cap edge
  { lat: -22, lon: -95, w: Ls => win(Ls, 200, 300, 20), u: -8, v: 0 },     // Solis / Syria
  { lat: 8, lon: 5, w: Ls => win(Ls, 180, 290, 20), u: -8, v: -0.5 },      // Arabia / Meridiani
];
const SLOT = 2.5;
function stormsAt(C, dustMul, capSLat) {
  const out = [];
  if (dustMul <= 0) return out;
  const j1 = Math.floor(C / SLOT);
  for (let j = j1 - 14; j <= j1; j++) {
    const t0 = (j + hashf(j * 1.13 + 0.5)) * SLOT, age = C - t0;
    if (age < 0) continue;
    const { Ls } = orbitAt(t0);
    // choose a region with probability ∝ weight
    const ws = GENESIS.map(g => g.w(Ls));
    const tot = ws.reduce((a, b) => a + b, 0) + 0.15;
    if (hashf(j * 7.31 + 2.2) > Math.min(0.9, 0.12 + 0.17 * tot) * Math.min(dustMul, 2.5) / 1.0) continue;
    let pick = hashf(j * 3.77 + 9.1) * tot, gi = -1;
    for (let i = 0; i < ws.length; i++) { pick -= ws[i]; if (pick < 0) { gi = i; break; } }
    if (gi < 0) continue;
    const g = GENESIS[gi];
    const regional = hashf(j * 5.9 + 1.3) < 0.12 + 0.18 * win(Ls, 190, 330, 20);
    const life = regional ? 9 + 12 * hashf(j * 2.1) : 2 + 3 * hashf(j * 2.7);
    if (age > life * 1.35) continue;
    const Rmax = (regional ? 14 + 16 * hashf(j * 4.4) : 3 + 5 * hashf(j * 4.9)) * D2R;
    const tmax = (regional ? 1.5 + 1.3 * hashf(j * 8.8) : 1.0 + 1.0 * hashf(j * 6.6)) * Math.min(1.5, 0.6 + 0.4 * dustMul);
    const grow = Math.min(1, age / (0.35 * life));
    const R = Rmax * (0.25 + 0.75 * grow * grow * (3 - 2 * grow));
    const tauNow = tmax * Math.min(1, age / (0.12 * life)) * (age < 0.45 * life ? 1 : Math.exp(-(age - 0.45 * life) / (0.25 * life)));
    const lat0 = g.cap ? -capSLat - 3 : g.lat + (hashf(j * 1.9) - 0.5) * 10;
    const lon0 = g.cap ? -180 + 360 * hashf(j * 9.3) : g.lon + (hashf(j * 2.3) - 0.5) * 30;
    const lat = Math.max(-80, Math.min(80, lat0 + g.v * age)), lon = lon0 + g.u * age;
    out.push({ lat, lon, R, tau: tauNow, kind: regional ? 'regional' : 'local' });
  }
  return out;
}
// the 2018 planet-encircling storm: began as a local storm over Arabia Terra (Ls ≈ 185), encircled the
// planet within ~3 weeks, τ > 8 at Opportunity's site, and took ~2–3 months to settle
function storm2018(a) {
  if (a < 0 || a > 170) return null;
  const decay = a < 40 ? 1 : Math.exp(-(a - 40) / 22);
  const main = { lat: 12 - 14 * Math.min(1, a / 14), lon: -2 - 18 * Math.min(1, a / 20), R: (5 + 190 * smooth01((a - 1.5) / 22)) * D2R, tau: (1.6 + 6.5 * smooth01((a - 2) / 16)) * decay, kind: 'global' };
  const hel = a > 6 ? { lat: -38, lon: 70 + 3 * (a - 6), R: (6 + 40 * smooth01((a - 6) / 10)) * D2R, tau: 3.5 * smooth01((a - 6) / 6) * decay, kind: 'regional' } : null;
  const bg = 1.6 * smooth01((a - 8) / 14) * decay;
  return { cells: hel ? [main, hel] : [main], bg, stage: a < 6 ? 'local' : a < 15 ? 'regional' : decay > 0.6 ? 'global' : 'decay' };
}
function smooth01(x) { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); }

/* ---- surface temperature (HUD) — a simple energy-balance estimate ---------------------------------
   Peak ≈ radiative equilibrium with the direct+diffuse insolation at ~13:30 (×0.95 for conduction into
   the regolith), night-time floor from the diurnal mean, damped by dust; CO₂ frost pins at 148 K. */
function surfaceTemp(latDeg, hour, decl, rAU, tau, frost) {
  const S = 586.2 * (A_AU / rAU) ** 2, A = 0.25, eps = 0.95, sig = 5.67e-8;
  const ph = latDeg * D2R, de = decl * D2R;
  const tr = Math.exp(-0.25 * tau);
  const czNoon = Math.max(0, Math.sin(ph) * Math.sin(de) + Math.cos(ph) * Math.cos(de));
  const h0 = Math.acos(Math.max(-1, Math.min(1, -Math.tan(ph) * Math.tan(de))));
  const Qd = S / Math.PI * (h0 * Math.sin(ph) * Math.sin(de) + Math.cos(ph) * Math.cos(de) * Math.sin(h0));
  const Tpk = 0.95 * Math.pow((1 - A) * S * czNoon * tr / (eps * sig), 0.25);
  const Tmean = Math.pow(Math.max((1 - A) * Qd * tr, 0) / (eps * sig) + 4000 * (1 + tau), 0.25) * 0.97;
  const damp = Math.exp(-0.18 * tau);
  // day: rises after sunrise, peaks ~13:30; night: exponential-ish cooling to before dawn
  const x = ((hour - 13.5) / 24 + 1) % 1;            // 0 at 13:30
  const day = Math.cos(2 * Math.PI * x);
  let T = Tmean + Math.max(0, Tpk - Tmean) * damp * (day > 0 ? day : 0.75 * day);
  if (frost) T = Math.min(T, 148);
  return Math.max(T, 145);
}

/* ================================================================================================
   Asset loading
   ================================================================================================ */
const MARS_URL = f => new URL(`../../../assets/mars/${f}`, import.meta.url).href;
const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
function toHalf(v) {
  _f32[0] = v; const x = _u32[0], s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7bff;
  return s | ((e << 10) + (((x & 0x7fffff) + 0x1000) >>> 13));
}
function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const W = img.naturalWidth, H = img.naturalHeight;
      const cv = Object.assign(document.createElement('canvas'), { width: W, height: H });
      const g = cv.getContext('2d', { willReadFrequently: true });
      g.drawImage(img, 0, 0);
      resolve({ W, H, px: g.getImageData(0, 0, W, H).data });
    };
    img.onerror = () => reject(new Error(t(`không tải được ${url}`, `could not load ${url}`)));
    img.src = url;
  });
}
// MOLA WebP: v = R·256 + G (G folded: 255 − G when R is odd), elevation = v·step − 9000 m.
// → R16F texture (km) with a CPU box-filtered mip chain, rows flipped so v = 0 is south; plus a
// coarse float copy for CPU look-ups.
async function loadElevation(url, step, keep = 1, repeat = true) {
  const { W, H, px } = await decodeImage(url);
  let lv = new Float32Array(W * H), max = -1e9, min = 1e9;
  for (let y = 0; y < H; y++) {
    let si = (H - 1 - y) * W * 4, di = y * W;
    for (let x = 0; x < W; x++, si += 4, di++) {
      const hi = px[si], lo = hi & 1 ? 255 - px[si + 1] : px[si + 1];
      const e = (((hi << 8) | lo) * step - 9000) / 1000;
      lv[di] = e; if (e > max) max = e; if (e < min) min = e;
    }
  }
  const mips = [];
  let w = W, h = H, cpu = null;
  for (;;) {
    const half = new Uint16Array(w * h);
    for (let i = 0; i < half.length; i++) half[i] = toHalf(lv[i]);
    mips.push({ data: half, width: w, height: h });
    if (w === W / keep && !cpu) cpu = { data: lv, W: w, H: h };
    if (w === 1 && h === 1) break;
    const w2 = Math.max(1, w >> 1), h2 = Math.max(1, h >> 1), nx = new Float32Array(w2 * h2);
    const sx = w > 1 ? 2 : 1, sy = h > 1 ? 2 : 1;
    for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
      const a = (y * sy) * w + x * sx, b = a + (sx - 1), d = a + (sy - 1) * w, f = d + (sx - 1);
      nx[y * w2 + x] = 0.25 * (lv[a] + lv[b] + lv[d] + lv[f]);
    }
    lv = nx; w = w2; h = h2;
  }
  const tex = new THREE.DataTexture(mips[0].data, W, H, THREE.RedFormat, THREE.HalfFloatType);
  tex.mipmaps = mips;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return { tex, max, min, cpu };
}
function loadImageTex(url, { srgb = false, aniso = 1, maxSize = 16384, repeat = true }) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    const img = tex.image;
    if (img.width > maxSize) {
      const s = maxSize / img.width, cv = document.createElement('canvas');
      cv.width = Math.round(img.width * s); cv.height = Math.round(img.height * s);
      const g = cv.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(img, 0, 0, cv.width, cv.height);
      tex.image = cv;
    }
    tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = aniso;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(t(`không tải được ${url}`, `could not load ${url}`)))));
}
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
function solidTex(data, format, type) {
  const t = new THREE.DataTexture(data, 1, 1, format, type); t.needsUpdate = true; return t;
}
function sampleElev(cpu, lat, lon, lon0 = -180, lat0 = -90, spanLon = 360, spanLat = 180) {
  const { data, W, H } = cpu;
  const x = (lon - lon0) / spanLon * W - 0.5, y = (lat - lat0) / spanLat * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.max(0, Math.min(H - 2, Math.floor(y))), fx = x - x0, fy = Math.max(0, Math.min(1, y - y0));
  const at = (i, j) => data[j * W + ((i % W) + W) % W];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

/* ================================================================================================
   Scene
   ================================================================================================ */
const vn = (x, d) => dec(x, d);
const hhmm = h => { h = ((h % 24) + 24) % 24; return `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`; };
const latStr = v => `${vn(Math.abs(v), 1)}°${v >= 0 ? t('B', 'N') : t('N', 'S')}`;
const lonStr = v => `${vn(Math.abs(v), 1)}°${v >= 0 ? t('Đ', 'E') : t('T', 'W')}`;
function sunFromHourE(cam, hour, declDeg, out) {
  const az = Math.atan2(cam.x, cam.z) - (hour - 12) * Math.PI / 12, d = declDeg * D2R;
  return out.set(Math.cos(d) * Math.sin(az), Math.sin(d), Math.cos(d) * Math.cos(az));
}
const bodyLatLonE = (p, spin) => ({ lat: Math.asin(p.y / p.length()) / D2R, lon: wrap180((Math.atan2(p.x, p.z) - spin) / D2R) });
const localHourE = (p, L) => (((12 + (Math.atan2(p.x, p.z) - Math.atan2(L.x, L.z)) / D2R / 15) % 24) + 24) % 24;
function hitSphere(o, d, R = 1) {
  const b = o.dot(d), c = o.lengthSq() - R * R, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
function seasonName(Ls) {
  const n = t(['xuân', 'hạ', 'thu', 'đông'], ['spring', 'summer', 'autumn', 'winter']);
  const s = t(['thu', 'đông', 'xuân', 'hạ'], ['autumn', 'winter', 'spring', 'summer']);
  const i = Math.floor(wrap360(Ls) / 90);
  return t(`Bắc bán cầu: ${n[i]} · Nam bán cầu: ${s[i]}`, `northern ${n[i]} · southern ${s[i]}`);
}
function sunElevStr(n, L) {
  const e = Math.asin(Math.max(-1, Math.min(1, n.dot(L)))) / D2R;
  const tag = e > 6 ? t('ban ngày', 'daytime') : e > -0.8 ? t('bình minh/hoàng hôn', 'sunrise/sunset') : e > -18 ? t('chạng vạng', 'twilight') : t('ban đêm', 'night');
  return t(`Mặt Trời cao <b>${e.toFixed(1)}°</b> (${tag})`, `Sun elevation <b>${e.toFixed(1)}°</b> (${tag})`);
}
const PHOBOS = { a: 9376 / RKM, P: 0.31891 * 86400 / 88775.244, r: 11.1 };   // orbital period in sols
const DEIMOS = { a: 23463.2 / RKM, P: 1.26244 * 86400 / 88775.244, r: 6.2 };

export default [{
  id: 'mars',
  name: 'Sao Hỏa',
  category: 'planets',
  order: 26,
  subtitle: 'Hành tinh đất đá · hành tinh đỏ',
  swatch: ['#f0a070', '#5a1e0c'],
  blurb: 'Bề mặt ở đây là <b>dữ liệu thật</b>: ảnh ghép Viking và địa hình laser MOLA — Olympus Mons, Valles Marineris, Hellas, Syrtis Major. Bụi oxit sắt nhuộm cam cả mặt đất lẫn bầu trời; khí quyển CO₂ chỉ ~6 mbar, nhưng mỗi mùa ~25 % lượng CO₂ ấy <b>đóng băng thành chỏm cực</b> rồi thăng hoa trở lại. Mây băng nước bám quanh núi lửa mỗi buổi chiều, bão bụi có thể lan khắp hành tinh, và hoàng hôn ở đây <b>màu xanh lam</b>.',
  intro: 'Nhấn <b>O</b> bay tới Olympus Mons, <b>V</b> tới Valles Marineris lúc bình minh, <b>B</b> để xem cơn bão bụi toàn cầu 2018, <b>X</b> cho hoàng hôn xanh. Kéo <b>Ls</b> để đổi mùa: chỏm băng co lại và nở ra.',
  facts: [
    ['Bán kính trung bình', '3 389,5 km'], ['Khối lượng', '6,417×10²³ kg'], ['Trọng lực bề mặt', '3,72 m/s²'],
    ['Ngày sao / sol (ngày Mặt Trời)', '24 h 37 min 22,7 s / 24 h 39 min 35 s'], ['Độ nghiêng trục quay', '25,19°'],
    ['Năm Sao Hỏa', '686,97 ngày Trái Đất = 668,6 sol'], ['Quỹ đạo', 'a = 1,524 AU, e = 0,0934 (1,38–1,67 AU)'],
    ['Áp suất bề mặt', '~6 mbar (4–9 mbar theo mùa; 0,7 mbar ở đỉnh Olympus, 12 mbar ở đáy Hellas)'],
    ['Khí quyển', 'CO₂ 95 %, N₂ 2,6 %, Ar 1,9 % · độ cao thang ~11 km'],
    ['Nhiệt độ bề mặt', 'trung bình −63 °C (−140 … +20 °C)'],
    ['Olympus Mons', 'cao 21,9 km so với nền xung quanh, rộng ~600 km'], ['Valles Marineris', 'dài ~4 000 km, sâu tới 7 km'],
    ['Hellas Planitia', 'đáy sâu −8,2 km so với mốc địa hình'],
    ['Phobos', '27×22×18 km · cách tâm 9 376 km · chu kỳ 7 h 39 min (mọc ở phía tây)'],
    ['Deimos', '15×12×11 km · cách tâm 23 463 km · chu kỳ 30 h 18 min'],
    ['Dữ liệu', 'Ảnh ghép màu Viking (NASA/JPL, USGS Astrogeology), địa hình MGS MOLA MEGDR (NASA GSFC, PDS)'],
  ],
  en: {
    name: 'Mars',
    subtitle: 'Terrestrial planet · the Red Planet',
    blurb: 'The surface here is <b>real data</b>: the Viking colour mosaic and MOLA laser altimetry — Olympus Mons, Valles Marineris, Hellas, Syrtis Major. Iron-oxide dust tints both the ground and the sky orange; the CO₂ atmosphere is only ~6 mbar, yet every season ~25% of it <b>freezes out onto the polar caps</b> and then sublimates back. Water-ice clouds cling to the volcanoes every afternoon, dust storms can engulf the whole planet, and sunsets here are <b>blue</b>.',
    intro: 'Press <b>O</b> to fly to Olympus Mons, <b>V</b> to Valles Marineris at dawn, <b>B</b> for the 2018 global dust storm, <b>X</b> for a blue sunset. Drag <b>Ls</b> to change the season: the polar caps shrink and grow.',
    facts: [
      ['Mean radius', '3,389.5 km'], ['Mass', '6.417×10²³ kg'], ['Surface gravity', '3.72 m/s²'],
      ['Sidereal day / sol (solar day)', '24 h 37 min 22.7 s / 24 h 39 min 35 s'], ['Axial tilt', '25.19°'],
      ['Martian year', '686.97 Earth days = 668.6 sols'], ['Orbit', 'a = 1.524 AU, e = 0.0934 (1.38–1.67 AU)'],
      ['Surface pressure', '~6 mbar (4–9 mbar with the seasons; 0.7 mbar atop Olympus, 12 mbar at the floor of Hellas)'],
      ['Atmosphere', 'CO₂ 95%, N₂ 2.6%, Ar 1.9% · scale height ~11 km'],
      ['Surface temperature', 'mean −63 °C (−140 … +20 °C)'],
      ['Olympus Mons', '21.9 km above the surrounding plains, ~600 km across'], ['Valles Marineris', '~4,000 km long, up to 7 km deep'],
      ['Hellas Planitia', 'floor −8.2 km below the datum'],
      ['Phobos', '27×22×18 km · 9,376 km from the centre · period 7 h 39 min (rises in the west)'],
      ['Deimos', '15×12×11 km · 23,463 km from the centre · period 30 h 18 min'],
      ['Data', 'Viking colour mosaic (NASA/JPL, USGS Astrogeology), MGS MOLA MEGDR topography (NASA GSFC, PDS)'],
    ],
  },
  unitKm: RKM,
  camera: { distance: 3.2, polar: 66, azimuth: 0, min: 1.006, max: 30, fov: 45 },
  exposure: 0.7,
  bloom: 0.3,
  params: [
    { key: 'hour', label: 'Giờ địa phương (dưới camera)', type: 'range', min: 0, max: 24, step: 0.1, default: 14, fmt: v => hhmm(v),
      note: '12:00 = Mặt Trời sau lưng bạn (đĩa tròn đầy); ~19:00 = đường phân chia ngày–đêm đi qua giữa đĩa.',
      en: { label: 'Local time (below the camera)', note: '12:00 = the Sun behind you (full disc); ~19:00 = the terminator crosses the middle of the disc.' } },
    { key: 'ls', label: 'Mùa — kinh độ Mặt Trời Ls', type: 'range', min: 0, max: 359.9, step: 0.1, default: 60, fmt: v => 'Ls ' + v.toFixed(0) + '°',
      note: 'Ls 0° = xuân phân Bắc bán cầu, 90° = hạ chí Bắc, 180° = thu phân, 270° = đông chí Bắc. Cận nhật ở Ls 251° (mùa bão bụi), viễn nhật ở Ls 71° (mùa mây băng).',
      en: { label: 'Season — solar longitude Ls', note: 'Ls 0° = northern spring equinox, 90° = northern summer solstice, 180° = autumn equinox, 270° = northern winter solstice. Perihelion at Ls 251° (dust-storm season), aphelion at Ls 71° (ice-cloud season).' } },
    { key: 'pace', label: 'Nhịp mùa & thời tiết', type: 'seg', options: [[0, 'Dừng'], [1, 'Thực'], [30, '×30'], [300, '×300']], default: 30,
      note: '"Thực": mùa trôi đúng nhịp một vòng quay = một sol (668,6 sol mỗi năm). ×30, ×300 tua nhanh mùa và các cơn bão bụi, còn mây và gió vẫn chuyển động theo đúng giờ địa phương.',
      en: { label: 'Season & weather pace', options: ['Stop', 'Real', '×30', '×300'], note: '"Real": the seasons advance at the true rate, one rotation = one sol (668.6 sols per year). ×30 and ×300 fast-forward the seasons and dust storms, while clouds and winds still follow local time.' } },
    { key: 'exag', label: 'Phóng đại địa hình', type: 'range', min: 1, max: 20, step: 0.5, default: 4, fmt: v => '×' + v.toFixed(1),
      note: '×1 = tỉ lệ thật: Olympus Mons cao 0,6 % bán kính. Địa hình là dữ liệu laser MOLA (độ phân giải ~5 km, ~1,9 km quanh Tharsis–Valles Marineris).',
      en: { label: 'Terrain exaggeration', note: '×1 = true scale: Olympus Mons rises just 0.6% of the radius. The terrain is MOLA laser altimetry (~5 km resolution, ~1.9 km around Tharsis–Valles Marineris).' } },
    { key: 'dust', label: 'Hoạt động bụi', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Nhân độ sâu quang học bụi nền và tần suất bão. Bão cục bộ (vài trăm km) xuất hiện quanh năm; bão khu vực chủ yếu từ Ls 180° đến 330° khi Sao Hỏa ở gần Mặt Trời.',
      en: { label: 'Dust activity', note: 'Scales the background dust optical depth and the storm frequency. Local storms (a few hundred km) occur year-round; regional storms mostly from Ls 180° to 330°, when Mars is closest to the Sun.' } },
    { key: 'clouds', label: 'Mây băng nước', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Vành đai mây viễn nhật (Ls 40°–150°), mây chụp cực mùa thu–đông, mây địa hình quanh núi lửa Tharsis buổi chiều, sương sớm trong hẻm vực.',
      en: { label: 'Water-ice clouds', note: 'The aphelion cloud belt (Ls 40°–150°), autumn–winter polar hoods, afternoon orographic clouds around the Tharsis volcanoes, morning fog in the canyons.' } },
    { key: 'atmo', label: 'Khí quyển', type: 'range', min: 0, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), en: { label: 'Atmosphere' } },
    { key: 'moons', label: 'Phobos & Deimos', type: 'toggle', default: true, en: { label: 'Phobos & Deimos' } },
  ],
  actions: [
    { label: 'Olympus Mons', key: 'o', title: 'Bay tới Olympus Mons lúc nắng sớm (bấm lần nữa để quay lại)', run: ctx => ctx.scene._marsShot?.(ctx, 'olympus'),
      en: { label: 'Olympus Mons', title: 'Fly to Olympus Mons in the early-morning sun (press again to return)' } },
    { label: 'Valles Marineris', key: 'v', title: 'Bay tới Valles Marineris lúc bình minh: sương mù trong hẻm vực (bấm lần nữa để quay lại)', run: ctx => ctx.scene._marsShot?.(ctx, 'valles'),
      en: { label: 'Valles Marineris', title: 'Fly to Valles Marineris at dawn: fog in the canyons (press again to return)' } },
    { label: 'Bão bụi 2018', key: 'b', title: 'Cơn bão bụi toàn cầu năm 2018 (Ls 185°)', run: ctx => ctx.scene._marsStorm?.(ctx),
      en: { label: '2018 dust storm', title: 'The 2018 global dust storm (Ls 185°)' } },
    { label: 'Hoàng hôn xanh', key: 'x', title: 'Mặt Trời sau hành tinh: tán xạ thuận', en: { label: 'Blue sunset', title: 'The Sun behind the planet: forward scattering' }, run: ctx => {
      // put the Sun just behind the planet as seen from the camera: move the camera to the latitude
      // opposite the Sun's declination, then set the local time (just inside the limb)
      const c = ctx.camera.position, sph = new THREE.Spherical().setFromVector3(c);
      const decl = declOf(ctx.params.ls); sph.phi = Math.PI / 2 + decl * D2R; c.setFromSpherical(sph); ctx.camera.lookAt(ctx.controls.target);
      ctx.params.hour = Math.asin(1 / c.length()) / D2R * 0.8 / 15;
      ctx.params.dust = Math.max(ctx.params.dust, 1.2);
      ctx.toast(t('Ngược sáng: hạt bụi ~1,5 µm tán xạ thuận ánh sáng xanh mạnh hơn đỏ (g ≈ 0,84 so với 0,58), nên quầng sáng quanh Mặt Trời và viền khí quyển ngả xanh lam — đúng như hoàng hôn mà Curiosity và Spirit chụp được, trong khi bầu trời ban ngày lại màu bơ caramel.',
        'Backlit: ~1.5 µm dust grains forward-scatter blue light more strongly than red (g ≈ 0.84 versus 0.58), so the glow around the Sun and the atmospheric limb turn blue — just like the sunsets Curiosity and Spirit photographed, while the daytime sky is butterscotch.')); } },
  ],
  create(ctx) {
    const Rr = ctx.renderer, caps = Rr.capabilities;
    const maxTex = caps.maxTextureSize, aniso = Math.min(8, caps.getMaxAnisotropy());
    const flat = () => solidTex(new Uint16Array([toHalf(0)]), THREE.RedFormat, THREE.HalfFloatType);
    const rust = () => solidTex(new Uint8Array([200, 130, 90, 255]), THREE.RGBAFormat, THREE.UnsignedByteType);
    const V = () => new THREE.Vector3();
    const U = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) }, uRot: { value: 0 }, uExag: { value: 4 }, uHMax: { value: 21.2 }, uHMin: { value: -8.2 },
      uTexOK: { value: 0 }, uTexel: { value: 2 * Math.PI / 4096 }, uAtmo: { value: 1 }, uESun: { value: 4 }, uSunAU: { value: 1.52 }, uHd: { value: 11.1 / RKM },
      uColTex: { value: rust() }, uElevTex: { value: flat() }, uPatchCol: { value: rust() }, uPatchElev: { value: flat() },
      uS: { value: 0 }, uTauBg: { value: 0.4 }, uCloud: { value: 1 }, uACB: { value: 0 }, uHoodN: { value: 0 }, uHoodS: { value: 0 }, uOro: { value: 0 }, uAMEC: { value: 0 },
      uFog: { value: 0.5 }, uJetN: { value: 10 }, uJetS: { value: 10 }, uIceSupp: { value: 1 },
      uCapN: { value: 1.3 }, uCapS: { value: 1.3 }, uCapOff: { value: 0 }, uCryptic: { value: 0 }, uFrost: { value: 1 },
      uStC: { value: [0, 1, 2, 3].map(() => new THREE.Vector4(0, 1, 0, 0)) }, uStP: { value: [0, 1, 2, 3].map(() => new THREE.Vector4()) },
      uPhoP: { value: V() }, uDeiP: { value: V() }, uPhoM: { value: new THREE.Matrix3() }, uDeiM: { value: new THREE.Matrix3() }, uMoons: { value: 1 },
    };
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: MARS_FRAG, depthTest: false, depthWrite: false });

    let elevCPU = null, patchCPU = null;
    const ready = Promise.all([
      retry(loadImageTex, MARS_URL('mars_color_8k.jpg'), { srgb: true, aniso, maxSize: maxTex }),
      retry(loadElevation, MARS_URL('mars_elev_4k.webp'), 12, 2),
      retry(loadImageTex, MARS_URL('mars_tharsis_color.jpg'), { srgb: true, aniso, repeat: false }),
      retry(loadElevation, MARS_URL('mars_tharsis_elev.webp'), 8, 1, false),
    ]).then(([col, el, pcol, pel]) => {
      U.uColTex.value = col; U.uElevTex.value = el.tex; U.uPatchCol.value = pcol; U.uPatchElev.value = pel.tex;
      U.uTexel.value = 2 * Math.PI / el.tex.image.width;
      U.uHMax.value = Math.max(el.max, pel.max); U.uHMin.value = Math.min(el.min, pel.min);
      elevCPU = el.cpu; patchCPU = pel.cpu;
      for (const t of [col, el.tex, pcol, pel.tex]) Rr.initTexture(t);
      U.uTexOK.value = 1;
    });
    // elevation (m) at a body-frame lat/lon
    const elevAt = (lat, lon) => {
      if (!elevCPU) return null;
      const pl = wrap360(lon - PATCH.lon0);
      if (patchCPU && pl < PATCH.dlon && lat > PATCH.lat0 && lat < PATCH.lat0 + PATCH.dlat)
        return 1000 * sampleElev(patchCPU, lat, PATCH.lon0 + pl, PATCH.lon0, PATCH.lat0, PATCH.dlon, PATCH.dlat);
      return 1000 * sampleElev(elevCPU, lat, lon);
    };

    /* ---- clocks ---- */
    const Y = new THREE.Vector3(0, 1, 0);
    const solsOf = t => t / SOL_SIM;                 // rotation clock
    const rotOf = t => 2 * Math.PI * solsOf(t);
    const clim = { off: null, pace: null, lastLs: null, C: 0, storm0: null };
    const qs = new URLSearchParams(location.search);
    const stormQ = qs.get('marsStorm');               // snapshot hook: start the 2018 storm, that many sols ago
    const climateNow = (t, pace) => clim.off + solsOf(t) * pace;
    let cur = { Ls: 60, r: 1.52, decl: 0, tauBg: 0.4, storms: [], stage: null, capN: 80, capS: 85, s18: null };

    /* ---- close-ups: a camera that rides with the rotating planet ---- */
    const shot = { on: null, tw: null, pending: qs.get('marsShot') };
    const def = ctx.scene;
    const bodyDir = (lat, lon) => new THREE.Vector3(...llE(lat, lon));
    const toWorld = (v, rot) => v.clone().applyAxisAngle(Y, rot);
    const toBody = (v, rot) => v.clone().applyAxisAngle(Y, -rot);
    const SHOTS = {
      // from ~420 km south-east of the caldera, looking north-west across the shield and its scarp
      olympus: { cam: [9.0, -125.5, 420], tgt: [19.5, -134.5], hour: 7.6, exag: 2.5,
        msg: () => t('Olympus Mons lúc nắng sớm, nhìn từ độ cao ~420 km về phía tây bắc. Ngọn núi lửa hình khiên cao <b>21,9 km</b> (gấp 2,5 lần Everest), rộng ~600 km, viền bởi vách đá dựng đứng cao tới 8 km; miệng núi lửa kép trên đỉnh rộng 80 km. Địa hình là dữ liệu laser MOLA, phóng đại <b>×2,5</b> theo chiều đứng (kéo về ×1 để thấy tỉ lệ thật — sườn núi chỉ dốc ~5°). Buổi chiều, mây băng nước hình thành quanh sườn núi.',
          'Olympus Mons in the early-morning sun, seen from ~420 km up looking north-west. The shield volcano rises <b>21.9 km</b> (2.5 times Everest), is ~600 km across and is ringed by a scarp up to 8 km high; the nested summit caldera is 80 km wide. The terrain is MOLA laser altimetry, exaggerated <b>×2.5</b> vertically (drag back to ×1 for true scale — the flanks slope only ~5°). In the afternoon, water-ice clouds form around its flanks.') },
      // over Ius/Melas, looking east down the trough system at dawn
      valles: { cam: [-18.5, -78.0, 280], tgt: [-8.0, -73.0], hour: 7.2, exag: 3,
        msg: () => t('Valles Marineris lúc bình minh: hệ thống hẻm vực dài ~4 000 km, rộng tới 200 km, sâu tới 7 km (Grand Canyon chỉ sâu 1,8 km). Hơi nước đông lại qua đêm thành <b>sương mù băng</b> đọng dưới đáy hẻm — như các tàu Viking từng chụp ở Noctis Labyrinthus — rồi tan khi Mặt Trời lên cao. Địa hình MOLA phóng đại <b>×3</b>.',
          'Valles Marineris at dawn: a canyon system ~4,000 km long, up to 200 km wide and up to 7 km deep (the Grand Canyon is only 1.8 km deep). Water vapour freezes overnight into <b>ice fog</b> that pools on the canyon floors — as the Viking orbiters photographed in Noctis Labyrinthus — and burns off as the Sun climbs. MOLA terrain exaggerated <b>×3</b>.') },
    };
    function frameOf(name) {
      const s = SHOTS[name];
      const tgt = bodyDir(s.tgt[0], s.tgt[1]);
      const e0 = (elevAt(s.tgt[0], s.tgt[1]) || 0) / 1000;
      tgt.multiplyScalar(1 + e0 * s.exag / RKM);
      const pos = bodyDir(s.cam[0], s.cam[1]).multiplyScalar(1 + s.cam[2] / RKM);
      return { pos, tgt, up: pos.clone().normalize() };
    }
    function globalFrame(c) {
      const cam = def.camera, rot = rotOf(c.time.sim);
      const pos = new THREE.Vector3().setFromSphericalCoords(cam.distance, cam.polar * D2R, cam.azimuth * D2R);
      return { pos: toBody(pos, rot), tgt: V(), up: toBody(Y, rot) };
    }
    function applyFrame(c, f, rot) {
      c.camera.position.copy(toWorld(f.pos, rot));
      c.controls.target.copy(toWorld(f.tgt, rot));
      c.camera.up.copy(toWorld(f.up, rot)).normalize();
      c.camera.lookAt(c.controls.target);
    }
    function startTween(c, b, dur, onDone) {
      const rot = rotOf(c.time.sim);
      shot.tw = { t0: performance.now(), dur: dur * 1000, onDone,
        a: { pos: toBody(c.camera.position, rot), tgt: toBody(c.controls.target, rot), up: toBody(c.camera.up, rot) }, b };
    }
    function goShot(c, name, instant) {
      const s = SHOTS[name], p = c.params, keep = instant ? (qs.get('params') || '') : '';
      const set = (k, v) => { if (!keep.includes(k + ':')) p[k] = v; };
      set('exag', s.exag); set('hour', s.hour);
      shot.on = name;
      c.controls.minDistance = 0.003;
      if (instant) { applyFrame(c, frameOf(name), rotOf(c.time.sim)); shot.tw = null; }
      else startTween(c, frameOf(name), 4.5);
      c.toast?.(s.msg(), 11000);
    }
    function leaveShot(c) {
      c.params.exag = 4; c.params.hour = 14;
      startTween(c, globalFrame(c), 3.5, () => { shot.on = null; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); c.controls.target.set(0, 0, 0); });
    }
    def._marsShot = (c, name) => { if (shot.tw) return; if (shot.on === name) leaveShot(c); else goShot(c, name, false); };
    def._marsStorm = c => {
      if (clim.storm0 !== null && clim.C - clim.storm0 < 170) {
        clim.storm0 = null; c.toast(t('Bụi lắng xuống; độ sâu quang học trở về mức nền của mùa.', 'The dust settles; the optical depth returns to the seasonal background.'));
        return;
      }
      c.params.pace = 300;
      setLs(c, 185);
      clim.storm0 = clim.C;
      c.toast(t('Bão bụi toàn cầu 2018 (tua ×300): bắt đầu là một cơn bão cục bộ ở Arabia Terra (Ls ≈ 185°), chỉ ~3 tuần sau đã bao trùm cả hành tinh; τ > 8 khiến tàu Opportunity thiếu nắng và ngừng hoạt động vĩnh viễn. Bụi cuộn lên tới 60 km, chỉ đỉnh các núi lửa Tharsis nhô lên khỏi màn bụi. Phải 2–3 tháng bụi mới lắng.',
        'The 2018 global dust storm (×300 fast-forward): it began as a local storm in Arabia Terra (Ls ≈ 185°) and within ~3 weeks had engulfed the whole planet; τ > 8 starved the Opportunity rover of sunlight and ended its mission for good. Dust was lofted up to 60 km, leaving only the summits of the Tharsis volcanoes poking above it. It took 2–3 months to settle.'), 12000);
    };
    function setLs(c, Ls) {
      c.params.ls = Ls;
      clim.C = climateOfLs(Ls);
      clim.pace = c.params.pace;
      clim.off = clim.C - solsOf(c.time.sim) * (c.params.pace || 0);
      clim.lastLs = Ls;
    }

    const mPho = new THREE.Matrix4(), mDei = new THREE.Matrix4(), tmpM = new THREE.Matrix3();
    let lastRot = null;
    function moonState(P, S, phase0, out, m) {
      const th = phase0 + 2 * Math.PI * S / P.P;
      out.set(Math.sin(th), 0, Math.cos(th)).multiplyScalar(P.a);
      // tidally locked: local x toward Mars, y along track, z = orbit normal
      const xA = out.clone().normalize().negate(), zA = Y.clone(), yA = zA.clone().cross(xA);
      m.makeBasis(xA, yA, zA);   // local → world
      return m;
    }

    return {
      material,
      ready,
      exit(c) {
        if (shot.on) { shot.on = null; shot.tw = null; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); }
      },
      onParam(key, v, c) { if (key === 'ls') setLs(c, v); },
      update(dt, c) {
        const p = c.params, L = U.uSunDir.value, t = c.time.sim;
        const rot = rotOf(t), S = solsOf(t);
        // ---- climate clock ----
        if (clim.off === null) { clim.pace = p.pace; setLs(c, p.ls); }
        if (clim.pace !== p.pace) { clim.off = clim.C - S * p.pace; clim.pace = p.pace; }   // keep C continuous
        if (clim.lastLs !== null && Math.abs(p.ls - clim.lastLs) > 0.05) setLs(c, p.ls);     // slider moved
        clim.C = climateNow(t, p.pace);
        if (stormQ !== null && clim.storm0 === null && !clim._q) { clim._q = true; clim.storm0 = clim.C - +stormQ; }
        const orb = orbitAt(clim.C);
        if (p.pace > 0) { p.ls = Math.round(orb.Ls * 10) / 10; clim.lastLs = p.ls; }
        const Ls = p.pace > 0 ? orb.Ls : p.ls;
        const decl = declOf(Ls);
        cur.Ls = Ls; cur.r = orb.r; cur.decl = decl;
        U.uSunAU.value = orb.r;
        U.uESun.value = 4 * (A_AU / orb.r) ** 2;

        // ---- close-up camera riding with the planet ----
        if (shot.pending && U.uTexOK.value > 0 && SHOTS[shot.pending]) { const n = shot.pending; shot.pending = null; goShot(c, n, true); }
        if (shot.tw) {
          const tw = shot.tw, k0 = Math.min(1, (performance.now() - tw.t0) / tw.dur), e = k0 * k0 * (3 - 2 * k0);
          const ra = tw.a.pos.length(), rb = tw.b.pos.length();
          const dir = tw.a.pos.clone().normalize().lerp(tw.b.pos.clone().normalize(), e);
          if (dir.lengthSq() < 1e-6) dir.copy(tw.b.pos).normalize();
          const r = Math.exp(Math.log(ra) * (1 - e) + Math.log(rb) * e) * (1 + 0.35 * Math.sin(Math.PI * e) * Math.min(1, Math.abs(ra - rb)));
          applyFrame(c, { pos: dir.normalize().multiplyScalar(r), tgt: tw.a.tgt.clone().lerp(tw.b.tgt, e), up: tw.a.up.clone().lerp(tw.b.up, e) }, rot);
          if (k0 >= 1) { shot.tw = null; tw.onDone?.(); }
        } else if (shot.on && lastRot !== null && rot !== lastRot) {
          const d = rot - lastRot;
          c.camera.position.applyAxisAngle(Y, d); c.controls.target.applyAxisAngle(Y, d); c.camera.up.applyAxisAngle(Y, d);
          c.camera.lookAt(c.controls.target);
        }
        lastRot = rot;
        // keep the camera above the (exaggerated) relief under it
        {
          const g = bodyLatLonE(c.camera.position, rot), e = elevAt(g.lat, g.lon);
          const top = 1 + ((e === null ? U.uHMax.value * 1000 : Math.max(e, 0)) / 1000 * U.uExag.value + 6) / RKM;
          if (c.camera.position.length() < top) c.camera.position.setLength(top);
        }
        sunFromHourE(c.camera.position, p.hour, decl, L);
        U.uRot.value = -rot;
        U.uS.value = S;
        U.uAtmo.value = p.atmo;
        U.uExag.value += (p.exag - U.uExag.value) * Math.min(1, dt * 4);
        if (Math.abs(p.exag - U.uExag.value) < 0.01) U.uExag.value = p.exag;

        // ---- seasons: caps, clouds, winds ----
        const capN = lsTable(CAP_N, Ls), capS = lsTable(CAP_S, Ls);
        cur.capN = capN; cur.capS = capS;
        U.uCapN.value = capN * D2R; U.uCapS.value = capS * D2R;
        U.uCapOff.value = smooth01((capS - 76) / 9);
        U.uCryptic.value = win(Ls, 185, 265, 20);
        U.uACB.value = win(Ls, 25, 170, 35);
        U.uHoodN.value = win(Ls, 150, 395, 30);
        U.uHoodS.value = 0.8 * win(Ls, 10, 175, 30);
        U.uOro.value = 0.35 + 0.65 * win(Ls, 0, 200, 40);
        U.uAMEC.value = win(Ls, 215, 320, 25);
        U.uFog.value = (0.45 + 0.55 * win(Ls, 20, 180, 40)) * Math.min(1, p.clouds);
        const nWinter = Math.max(0, -Math.sin(Ls * D2R)), sWinter = Math.max(0, Math.sin(Ls * D2R));
        U.uJetN.value = 8 + 32 * nWinter; U.uJetS.value = 8 + 32 * sWinter;
        U.uCloud.value = p.clouds;
        U.uMoons.value = p.moons ? 1 : 0;

        // ---- dust ----
        let storms = stormsAt(clim.C, p.dust, capS);
        let bgExtra = 0, stage = null;
        cur.s18 = null;
        if (clim.storm0 !== null) {
          const a = clim.C - clim.storm0;
          const s18 = storm2018(a);
          if (s18) { storms = [...s18.cells, ...storms]; bgExtra = s18.bg; stage = s18.stage; cur.s18 = a; }
          else if (a > 170) clim.storm0 = null;
        }
        storms.sort((a, b) => (b.kind === 'global') - (a.kind === 'global') || b.tau * b.R - a.tau * a.R);
        const tauBg = (tauClim(Ls) * p.dust + bgExtra);
        cur.tauBg = tauBg; cur.storms = storms.slice(0, 4); cur.stage = stage;
        U.uTauBg.value = Math.max(0.02, tauBg);
        for (let i = 0; i < 4; i++) {
          const s = storms[i];
          if (s) {
            const d = llE(s.lat, s.lon);
            U.uStC.value[i].set(d[0], d[1], d[2], s.R);
            U.uStP.value[i].set(s.tau, 0, 0, 0);
          } else U.uStP.value[i].set(0, 0, 0, 0);
        }
        const tauMax = tauBg + (storms[0]?.tau || 0);
        U.uHd.value = (11.1 + 10 * smooth01((tauMax - 1.5) / 5)) / RKM;   // storms loft dust higher
        U.uIceSupp.value = Math.exp(-0.6 * Math.max(0, tauBg - 0.9));      // a dusty, warm atmosphere sheds its ice clouds

        // ---- Phobos and Deimos (world frame; prograde, equatorial) ----
        moonState(PHOBOS, S, 0.9, U.uPhoP.value, mPho);
        moonState(DEIMOS, S, 2.6, U.uDeiP.value, mDei);
        // shader wants world → local
        U.uPhoM.value.setFromMatrix4(mPho).transpose();
        U.uDeiM.value.setFromMatrix4(mDei).transpose();
      },
      hud(c) {
        const cp = c.camera.position, rot = rotOf(c.time.sim), g = bodyLatLonE(cp, rot);
        const e = elevAt(g.lat, g.lon);
        const em = e === null ? 0 : e;
        const P = pDatum(cur.Ls) * Math.exp(-em / 11100);
        const frost = g.lat > cur.capN || g.lat < -cur.capS;
        const T = surfaceTemp(g.lat, c.params.hour, cur.decl, cur.r, cur.tauBg, frost);
        const big = cur.storms[0];
        const stormTxt = cur.stage ? t({ local: 'cục bộ, đang lớn', regional: 'khu vực, đang lan rộng', global: 'toàn cầu', decay: 'toàn cầu, đang lắng' },
          { local: 'local, growing', regional: 'regional, spreading', global: 'global', decay: 'global, settling' })[cur.stage] + ` (sol ${cur.s18.toFixed(0)})`
          : big ? t(`${cur.storms.length} cơn, lớn nhất ${big.kind === 'regional' ? 'khu vực' : 'cục bộ'} ~${Math.round(big.R * RKM * 2 / 100) * 100} km`,
            `${cur.storms.length}, largest ${big.kind === 'regional' ? 'regional' : 'local'} ~${(Math.round(big.R * RKM * 2 / 100) * 100).toLocaleString('en-US')} km`) : t('không', 'none');
        const d = cp.length(), hKm = (d - 1) * RKM;
        return [
          [t('Độ cao', 'Altitude'), c.fmt.dist(hKm * 1000)],
          [t('Mùa', 'Season'), `Ls ${vn(cur.Ls, 1)}° · ${seasonName(cur.Ls)}`],
          [t('Khoảng cách tới Mặt Trời', 'Distance from the Sun'), t(`${vn(cur.r, 3)} AU (nắng ×${vn((A_AU / cur.r) ** 2, 2)} so với trung bình)`, `${vn(cur.r, 3)} AU (sunlight ×${vn((A_AU / cur.r) ** 2, 2)} the average)`)],
          [t('Điểm dưới camera', 'Point below the camera'), `${latStr(g.lat)} ${lonStr(g.lon)}` + (e === null ? '' : t(` · cao ${Math.round(e).toLocaleString(locale())} m`, ` · elevation ${Math.round(e).toLocaleString(locale())} m`))],
          [t('Giờ Mặt Trời tại đó', 'Local solar time there'), hhmm(c.params.hour)],
          [t('Áp suất bề mặt (ước tính)', 'Surface pressure (est.)'), `${vn(P, 1)} mbar`],
          [t('Nhiệt độ mặt đất (ước tính)', 'Ground temperature (est.)'), `${Math.round(T - 273.15)} °C (${Math.round(T)} K)`],
          [t('Bụi: độ sâu quang học nền τ', 'Dust: background optical depth τ'), vn(cur.tauBg, 2)],
          [t('Bão bụi', 'Dust storms'), stormTxt],
          [t('Chỏm băng mùa (mép)', 'Seasonal polar caps (edge)'), t(`Bắc ${cur.capN.toFixed(0)}°B · Nam ${cur.capS.toFixed(0)}°N`, `north ${cur.capN.toFixed(0)}°N · south ${cur.capS.toFixed(0)}°S`)],
          [t('Phóng đại địa hình', 'Terrain exaggeration'), '×' + vn(U.uExag.value, 1)],
        ];
      },
      onClick(info, c) {
        const th = hitSphere(info.origin, info.dir);
        if (th === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, th).normalize(), L = U.uSunDir.value;
        const g = bodyLatLonE(p, rotOf(c.time.sim));
        const e = elevAt(g.lat, g.lon);
        const hl = localHourE(p, L);
        const P = pDatum(cur.Ls) * Math.exp(-(e || 0) / 11100);
        const T = surfaceTemp(g.lat, hl, cur.decl, cur.r, cur.tauBg, g.lat > cur.capN || g.lat < -cur.capS);
        const hs = e === null ? '' : t(` · độ cao <b>${Math.round(e).toLocaleString(locale())} m</b>`, ` · elevation <b>${Math.round(e).toLocaleString(locale())} m</b>`);
        c.toast(t(`Vĩ độ <b>${latStr(g.lat)}</b>, kinh độ <b>${lonStr(g.lon)}</b>${hs} · giờ địa phương <b>${hhmm(hl)}</b> · ${sunElevStr(p, L)} · áp suất ~<b>${dec(P, 1)} mbar</b> · mặt đất ~<b>${Math.round(T - 273.15)} °C</b>.`,
          `Latitude <b>${latStr(g.lat)}</b>, longitude <b>${lonStr(g.lon)}</b>${hs} · local time <b>${hhmm(hl)}</b> · ${sunElevStr(p, L)} · pressure ~<b>${dec(P, 1)} mbar</b> · ground ~<b>${Math.round(T - 273.15)} °C</b>.`));
      },
    };
  },
}];
