// Trái Đất — the Earth, from real data, moving like the real planet.
//
// One scene unit = one Earth radius (6 371 km). Surface: NASA Blue Marble NG (June + January, blended
// by date), NOAA ETOPO 2022 relief/bathymetry (ray-marched, exaggerated), NASA Black Marble 2016
// night lights. Atmosphere: Rayleigh + ozone + Mie single scattering with the Chapman function.
//
// Motion ("thời gian Trái Đất"): an Earth clock T (hours) runs at `pace` Earth-seconds per simulation
// second. The planet turns at the sidereal rate (one turn per 23 h 56 min 4 s of T). The Sun's
// declination comes from the date (real solar ephemeris, axial tilt 23.44°); the Moon is placed from a
// low-precision lunar ephemeris for the same date and time.
//
// Weather (all a pure function of T, so any frame can be reproduced exactly):
//  * background cloud = NASA Blue Marble cloud composite advected by an analytic wind field
//    (trade easterlies converging on a seasonal ITCZ, mid-latitude westerlies/jets with travelling
//    Rossby-wave meanders, polar easterlies). Two-phase flow map (30 h period, resets staggered in
//    space) so nothing smears; storm-track latitudes take a fresh, zonally re-drawn copy of the real
//    composite at every reset (new weather systems), dissolved in by erosion — thin cloud forms /
//    evaporates first, thick cores last.
//  * travelling storms = real satellite images (NASA GIBS VIIRS: hurricanes Irma 2017 and Florence
//    2018, a North-Pacific cyclone 2022, a Southern-Ocean cyclone 2026), re-projected storm-centred,
//    mirrored for the other hemisphere (so they turn the right way: counter-clockwise north, clockwise
//    south), moved along deterministic tracks (extratropical lows: genesis east of continents,
//    eastward ~12 m/s and poleward; tropical cyclones: westward then recurving north-east, decaying
//    over land and cold water), rotating and winding up as they mature.
//  * tropical deep convection = cumulonimbus anvils that bloom and decay with the local solar time
//    (afternoon over land, night/dawn over the ocean), where the real composite says the tropics are
//    convective; stratocumulus decks thin out in the afternoon; fair-weather cumulus over warm land
//    by day.
//  * optional live clouds: yesterday's global VIIRS true-colour mosaic from NASA GIBS (WMS, CORS),
//    turned into a cloud map on the GPU against the Blue Marble surface; falls back to the bundled map.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';

const D2R = Math.PI / 180;
const SIDEREAL_H = 23.9344696;                 // hours per rotation relative to the stars
const R_KM = 6371;
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// deterministic hash → [0, 1)
function hash(a, b) {
  let h = Math.imul((a | 0) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((b | 0) + 0x632be5ab, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d); h = Math.imul(h ^ (h >>> 12), 0x297a2d39); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/* ------------------------------------------------------------------------------------------------
   Shared GLSL: body rotation, the Chapman function, phase functions, the single-scattering march
   ------------------------------------------------------------------------------------------------ */
const ATMO_GLSL = /* glsl */`
mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
// Chapman grazing-incidence function (Schüler 2012 form, with the √(πx/2) horizontal limit)
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
struct Atm { float R; float Hr; float Hm; vec3 bR; vec3 bA; vec3 bMs; vec3 bMe; vec3 g; };
vec3 sunTrans(Atm A, float h, float mu){
  float oR = A.Hr * chapman(A.R / A.Hr, h / A.Hr, mu);
  float oM = A.Hm * chapman(A.R / A.Hm, h / A.Hm, mu);
  return exp(-((A.bR + A.bA) * oR + A.bMe * oM));
}
vec3 atmosphere(vec3 ro, vec3 rd, float t0, float t1, vec3 L, Atm A, out vec3 T){
  const int N = 14;
  float dt = (t1 - t0) / float(N);
  vec3 sR = vec3(0.0), sM = vec3(0.0);
  float oR = 0.0, oM = 0.0;
  float XR = A.R / A.Hr, XM = A.R / A.Hm;
  vec3 eR = A.bR + A.bA;
  for (int i = 0; i < N; i++){
    vec3 p = ro + rd * (t0 + (float(i) + 0.5) * dt);
    float r = length(p), h = r - A.R;
    float dR = exp(-h / A.Hr) * dt, dM = exp(-h / A.Hm) * dt;
    oR += 0.5 * dR; oM += 0.5 * dM;
    float mu = dot(p, L) / r;
    float lR = A.Hr * chapman(XR, h / A.Hr, mu), lM = A.Hm * chapman(XM, h / A.Hm, mu);
    vec3 att = exp(-(eR * (oR + lR) + A.bMe * (oM + lM)));
    sR += dR * att; sM += dM * att;
    oR += 0.5 * dR; oM += 0.5 * dM;
  }
  T = exp(-(eR * oR + A.bMe * oM));
  float nu = dot(rd, L);
  return sR * A.bR * phaseR(nu) + sM * A.bMs * phaseCS(nu, A.g);
}
vec3 sunDisk(vec3 rd, vec3 L, float dAU){
  float c = dot(rd, L), a = acos(clamp(c, -1.0, 1.0)), r = 0.00465 / dAU;
  return bb(5772.0) * (smoothstep(r * 1.15, r * 0.85, a) * 400.0 + pow(max(c, 0.0), 3000.0 * dAU * dAU) * 4.0);
}
`;

/* ================================================================================================
   Shader
   ================================================================================================ */
const PATCH = { lon0: 66, lat0: 22, w: 2048, h: 1024 };   // 1′ Himalaya / Karakoram / Tibet patch
const NST = 11;                                            // storm slots: 4 tropical + 7 extratropical
const EARTH_FRAG = COMMON_GLSL + ATMO_GLSL + /* glsl */`
#define NST ${NST}
uniform vec3 uSunDir, uAurN, uAurS;
uniform float uRot, uClouds, uCities, uAtmo, uKp, uAurora;
uniform float uExag, uHMax, uTexOK, uTexel, uCloudTexel;
uniform sampler2D uColTex, uJanTex, uElevTex, uNightTex, uCloudTex, uPatchCol, uPatchElev, uStormTex, uClimTex, uMoonTex;
uniform float uWxT, uSolarT, uDecl, uWx, uWinter, uIceN, uIceS, uConv, uCovBoost, uCirrus;
uniform vec4 uStA[NST], uStB[NST], uStC[NST];
uniform vec3 uMoonPos, uMoonF, uMoonN, uMoonE;
uniform float uMoon;
const float RKM = 6371.0;
const float RTOP = 1.0 + 70.0 / RKM;
const float HCL = 8.0 / RKM;             // cloud deck: ~8 km above the (smoothed) ground
const float ESUN = 4.0;                  // solar irradiance in scene radiance units
const float RMOON = 1737.4 / RKM;
const float MS = 3.6 / RKM;              // 1 m/s in radians of arc per hour
const vec2 P0 = vec2(${((PATCH.lon0 + 180) / 360).toFixed(7)}, ${((PATCH.lat0 + 90) / 180).toFixed(7)});
const vec2 PS = vec2(${(PATCH.w / 60 / 360).toFixed(7)}, ${(PATCH.h / 60 / 180).toFixed(7)});
const float PRATIO = ${(PATCH.w / (PATCH.w / 60 / 360) / 4096).toFixed(4)};

float sq(float x){ return x * x; }

Atm atmE(){
  Atm A;
  A.R = 1.0; A.Hr = 8.0 / RKM; A.Hm = 1.2 / RKM;
  A.bR = vec3(5.8, 13.5, 33.1) * 6.371 * uAtmo;               // β in 10⁻⁶ m⁻¹ × 6.371·10⁶ m per unit
  A.bA = vec3(0.65, 1.881, 0.085) * 6.371 * 2.0 * uAtmo;      // ozone (Chappuis band) on the molecular profile
  A.bMs = vec3(60.0 * 6.371) * uAtmo; A.bMe = A.bMs / 0.9; A.g = vec3(0.76);   // aerosol, AOD ≈ 0.08
  return A;
}

// ---- data access -----------------------------------------------------------------------------
vec2 uvOf(vec3 p){ return vec2(atan(p.x, p.z) * (0.5 / PI) + 0.5, asin(clamp(p.y, -1.0, 1.0)) / PI + 0.5); }
vec2 llOfUV(vec2 uv){ return vec2((uv.x - 0.5) * TAU, (uv.y - 0.5) * PI); }
vec3 dirOfLL(vec2 ll){ float c = cos(ll.y); return vec3(c * sin(ll.x), sin(ll.y), c * cos(ll.x)); }
vec2 uvD(vec3 p, vec3 dp){
  float q = max(p.x * p.x + p.z * p.z, 1e-6);
  return vec2((p.z * dp.x - p.x * dp.z) / q * (0.5 / PI), dp.y / (PI * sqrt(q)));
}
float patchW(vec2 uv, out vec2 q){
  q = (uv - P0) / PS;
  vec2 m = min(q, 1.0 - q);
  return smoothstep(0.0, 0.06, min(m.x, m.y));
}
vec2 elevAt(vec2 uv, float lod){
  vec2 q; float w = patchW(uv, q);
  if (w > 0.0){
    vec2 hp = textureLod(uPatchElev, q, max(lod + log2(PRATIO), 0.0)).rg;
    if (w >= 1.0) return hp;
    return mix(textureLod(uElevTex, uv, max(lod, 0.0)).rg, hp, w);
  }
  return textureLod(uElevTex, uv, max(lod, 0.0)).rg;
}
float texelAt(vec2 uv){ vec2 q; return patchW(uv, q) > 0.5 ? uTexel / PRATIO : uTexel; }
vec3 colorAt(vec2 uv, vec2 gx, vec2 gy){
  vec3 c = textureGrad(uColTex, uv, gx, gy).rgb;
  vec2 q; float w = patchW(uv, q);
  if (w > 0.0) c = mix(c, textureGrad(uPatchCol, q, gx / PS, gy / PS).rgb, w);
  if (uWinter > 0.002){
    // January map (4k) for the season, keeping the June map's finer detail on top
    vec3 cj = textureGrad(uJanTex, uv, gx, gy).rgb;
    vec3 cb = textureGrad(uColTex, uv, gx * 2.0, gy * 2.0).rgb;
    c = mix(c, max(cj + (c - cb) * 0.8, 0.0), uWinter);
  }
  return c;
}
float hgtKm(vec3 pb, float lod){ return max(elevAt(uvOf(pb), lod).r, 0.0); }

// ---- relief ray march ------------------------------------------------------------------------
float marchRelief(vec3 ro, vec3 rd, float tA, float tB, bool onSphere, mat3 R, float k, float pixAng){
  vec3 pm = normalize(ro + rd * (0.5 * (tA + tB)));
  float sinT = max(length(cross(rd, pm)), 0.02);
  float lat = (tB - tA) * sinT;
  float tex = texelAt(uvOf(R * pm));
  float n = clamp(ceil(lat / max(1.6 * tex, tA * pixAng)), 2.0, 72.0);
  float dt = (tB - tA) / n;
  float lodS = log2(max(dt * sinT, 1e-7) / uTexel) - 0.5;
  float tP = tA, dP = 1.0;
  for (int i = 0; i <= 72; i++){
    if (float(i) > n) break;
    float t = tA + dt * float(i);
    vec3 q = ro + rd * t; float r = length(q);
    float lod = max(lodS, log2(t * pixAng / uTexel));
    float d = r - 1.0 - k * hgtKm(R * (q / r), lod);
    if (d <= 0.0 || (onSphere && float(i) >= n)){
      if (i == 0) return t;
      for (int j = 0; j < 2; j++){
        float tc = tP + (t - tP) * dP / max(dP - d, 1e-9);
        vec3 qc = ro + rd * tc; float rc = length(qc);
        float dc = rc - 1.0 - k * hgtKm(R * (qc / rc), lod);
        if (dc <= 0.0){ t = tc; d = dc; } else { tP = tc; dP = dc; }
      }
      return tP + (t - tP) * dP / max(dP - d, 1e-9);
    }
    tP = t; dP = d;
  }
  return -1.0;
}
float reliefShadow(vec3 pw, vec3 L, mat3 R, float k, float ds, float shellTop){
  float t = ds * 1.5, res = 1.0;
  for (int i = 0; i < 22; i++){
    vec3 q = pw + L * t; float r = length(q);
    if (r > shellTop) break;
    float h = 1.0 + k * hgtKm(R * (q / r), log2(max(t * 0.25, ds) / uTexel));
    res = min(res, (r - h) / (t * 0.012 + ds * 0.25));
    if (res < -0.5) break;
    t *= 1.3;
  }
  return smoothstep(-0.5, 1.0, res);
}
float ridged(vec3 x){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++){ float n = 1.0 - abs(snoise(x)); s += a * n * n; x = x * 2.07 + vec3(3.1, -1.7, 5.3); a *= 0.5; }
  return s;
}

/* ================= WEATHER ===================================================================== */
// Large-scale wind (u east, v north) in m/s at (lon, lat) [radians], Earth time T [hours].
// Zonal means: trades ~7 m/s, westerlies ~14 (N) / 17 (S) m/s at the mid-troposphere steering level,
// polar easterlies; the belts follow the Sun by 0.3 × declination. The trades converge on the ITCZ.
// Rossby waves from a streamfunction ψ = R·B·e(φ)·cos(kλ − ωT + θ): non-divergent meanders that
// travel east at a few m/s (u = −∂ψ/R∂φ, v = ∂ψ/(R cosφ ∂λ)).
vec2 rossby(vec2 ll, float T, float k, float lat0, float w, float vAmp, float c, float th){
  float e = exp(-sq((ll.y - lat0) / w));
  float ph = k * ll.x - k * c * MS / cos(lat0) * T + th;
  float B = vAmp * cos(lat0) / k;
  return vec2(B * e * 2.0 * (ll.y - lat0) / (w * w) * cos(ph), -B * e * k / max(cos(ll.y), 0.2) * sin(ph));
}
vec2 wind(vec2 ll, float T){
  float latd = degrees(ll.y), ls = latd - 0.3 * uDecl, a = abs(ls);
  float u = -7.0 * exp(-sq((a - 13.0) / 9.0))
          + 14.0 * exp(-sq((ls - 44.0) / 11.0)) + 17.0 * exp(-sq((ls + 49.0) / 10.0))
          - 4.0 * exp(-sq((a - 72.0) / 7.0));
  float di = latd - (6.0 + 0.35 * uDecl);
  float v = -2.2 * sign(di) * exp(-sq((abs(di) - 9.0) / 8.0));
  vec2 w = vec2(u, v);
  float sh = radians(0.3 * uDecl);
  w += rossby(ll, T, 5.0, 0.80 + sh, 0.22, 9.0, 7.0, 0.0);
  w += rossby(ll, T, 3.0, 0.88 + sh, 0.25, 6.0, 1.5, 2.1);
  w += rossby(ll, T, 4.0, -0.86 + sh, 0.22, 9.0, 8.0, 4.0);
  w += rossby(ll, T, 6.0, -0.78 + sh, 0.20, 7.0, 10.0, 1.3);
  return w;
}
vec2 llRate(vec2 ll, float T){ vec2 w = wind(ll, T); return vec2(w.x / max(cos(ll.y), 0.2), w.y) * MS; }
// where the air now at ll was tau hours earlier (midpoint rule; tau < 0 traces forward)
vec2 trace(vec2 ll, float T, float tau){
  vec2 m = ll - llRate(ll, T) * (0.5 * tau);
  m.y = clamp(m.y, -1.55, 1.55);
  vec2 r = ll - llRate(m, T - 0.5 * tau) * tau;
  r.y = clamp(r.y, -1.55, 1.55);
  return r;
}
float hash1(float n){ return fract(sin(n * 12.9898 + 4.1414) * 43758.5453); }
// Cloud value v (the composite's brightness scale) <-> visible optical depth tau. Everything that makes
// cloud appear, fade, merge or dissolve acts on tau (thickness), and the opacity is 1 - exp(-tau): thin
// cloud stays partly transparent, edges fall off softly like real optical depth.
float tauOf(float v){ return 14.0 * pow(max(v - 0.07, 0.0), 1.6); }
float vOf(float t){ return 0.07 + pow(max(t, 0.0) / 14.0, 0.625); }
vec2 worley2(vec2 p){
  vec2 i = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec2 o = vec2(x, y), r = o + hash33(vec3(i + o, 5.0)).xy - f;
    float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; } else if (d < d2) d2 = d;
  }
  return sqrt(vec2(d1, d2));
}

// two-phase flow map with spatially staggered resets
const float PFLOW = 30.0;
struct Flow { vec2 uv, a0, a1, b0, b1; float w0, w1, m; };
Flow flowAt(vec2 uv, vec3 pb){
  Flow F; F.uv = uv;
  vec2 ll = llOfUV(uv);
  float ph = uWxT / PFLOW + 0.3 * snoise(pb * 1.7 + 3.1);
  float f0 = fract(ph), f1 = fract(ph + 0.5);
  float c0 = floor(ph), c1 = floor(ph + 0.5);
  vec2 l0 = trace(ll, uWxT, (f0 - 0.5) * PFLOW), l1 = trace(ll, uWxT, (f1 - 0.5) * PFLOW);
  F.a0 = vec2(l0.x * (0.5 / PI) + 0.5, l0.y / PI + 0.5);
  F.a1 = vec2(l1.x * (0.5 / PI) + 0.5, l1.y / PI + 0.5);
  // storm-track latitudes: every cycle brings a new draw of real weather (the composite re-drawn
  // at a random longitude), not the same clouds back again
  float al = abs(degrees(ll.y));
  F.m = smoothstep(27.0, 38.0, al) * (1.0 - smoothstep(62.0, 72.0, al));
  float hs = ll.y > 0.0 ? 0.0 : 57.0;
  F.b0 = F.a0 + vec2(0.12 + 0.76 * hash1(c0 + hs), 0.0);
  F.b1 = F.a1 + vec2(0.12 + 0.76 * hash1(c1 + 101.0 + hs), 0.0);
  // optical-depth cross-fade (w0 + w1 = 1 keeps the mean thickness, so global cover holds steady)
  float w = 1.0 - abs(2.0 * f0 - 1.0);
  F.w0 = smoothstep(0.05, 0.95, w + 0.15 * snoise(pb * 4.3 + 7.7));
  F.w1 = 1.0 - F.w0;
  return F;
}
float bgTau(vec2 a, vec2 b, float m, vec2 gx, vec2 gy){
  float t = m < 1.0 ? tauOf(textureGrad(uCloudTex, a, gx, gy).r) : 0.0;
  if (m > 0.0) t = mix(t, tauOf(textureGrad(uCloudTex, b, gx, gy).r), m);
  return t;
}

// travelling storms: real satellite images re-projected around the storm centre (returns tau)
float storms(vec3 pb, float fpRad, out float sup){
  float ts = 0.0; sup = 0.0;
  for (int j = 0; j < NST; j++){
    vec4 A = uStA[j];
    if (A.w <= 0.002) continue;
    vec4 C = uStC[j];
    float cd = dot(pb, A.xyz);
    if (cd < C.z) continue;
    vec4 B = uStB[j];
    vec3 e = normalize(vec3(A.z, 0.0, -A.x) + vec3(1e-5, 0.0, 0.0));
    vec3 n = cross(A.xyz, e);
    vec2 q = vec2(dot(pb, e), dot(pb, n));
    q *= acos(clamp(cd, -1.0, 1.0)) / max(length(q), 1e-7) / B.w;       // azimuthal-equidistant, in half-sizes
    q.x /= abs(C.w);                                                     // systems stretched along the flow
    float r = length(q);
    bool tc = C.w < 0.0;
    float win = 1.0 - smoothstep(tc ? 0.62 : 0.35, 0.97, r);
    q.y *= B.y;                                                          // into the image's own hemisphere
    vec2 qn = q;
    float ang = (B.z + C.y / (1.0 + sq(r / 0.22))) * C.x;                 // cyclonic turn + wind-up of the core
    float ca = cos(ang), sa = sin(ang);
    q = vec2(ca * qn.x + sa * qn.y, -sa * qn.x + ca * qn.y);
    vec2 qs = q + (B.x == 1.0 ? vec2(0.0, -0.033) : vec2(0.0));          // Florence's eye sits 8 px low
    vec2 st = clamp(qs * 0.5 + 0.5, 0.004, 0.996);
    float row = floor(B.x * 0.25);
    vec2 uv = vec2((B.x - 4.0 * row + st.x) * 0.25, (1.0 - row + st.y) * 0.5);
    float v = textureLod(uStormTex, uv, max(log2(fpRad * 256.0 / B.w), 0.0)).r;
    float t = tauOf(v * 0.93) * A.w * win;                               // thickness fades with life and window
    if (tc){
      // tropical-cyclone anatomy on top of the real image (in its cyclonic frame):
      float th = atan(q.y, q.x), lr = log(max(r, 1e-3));
      // spiral rain bands: two log-spiral arms curling inward counter-clockwise (NH frame)
      float arm = 0.5 + 0.5 * cos(2.0 * (th + 2.4 * lr) + 0.6 * snoise(vec3(q * 4.0, B.x)));
      float zone = smoothstep(0.16, 0.3, r) * (1.0 - smoothstep(0.55, 0.9, r));
      float bands = smoothstep(0.45, 0.9, arm) * zone;
      t *= 1.0 - 0.75 * zone * (1.0 - smoothstep(0.25, 0.7, arm));           // clear lanes between the bands
      t = max(t, 3.0 * bands * (0.65 + 0.35 * snoise(vec3(q * 22.0, B.x * 3.0 + 1.0))) * A.w);
      // outflow cirrus: fibres spiralling out anticyclonically at the top (pattern turns the other way)
      float ao = -0.35 * ang, co = cos(ao), so = sin(ao);
      vec2 qo = vec2(co * qn.x + so * qn.y, -so * qn.x + co * qn.y);
      float fib = 0.5 + 0.5 * cos(13.0 * (atan(qo.y, qo.x) + 1.7 * lr) + 2.5 * snoise(vec3(qo * 7.0, 4.0)));
      t += 0.32 * fib * fib * smoothstep(0.12, 0.32, r) * (1.0 - smoothstep(0.45, 0.92, r)) * A.w;
      // the eye: clear, ringed by a sharp, tall eyewall
      float re = 0.034;
      float eye = smoothstep(re * 0.7, re * 1.2, r);
      t = t * eye + (1.0 - eye) * 0.25 * A.w + 7.0 * A.w * exp(-sq((r - re * 1.75) / (re * 0.6)));   // low cloud in the eye
    }
    ts += t; sup = max(sup, win * A.w);
  }
  return ts;
}

// tropical deep convection: anvils born at a preferred local solar time, growing ~3 h, decaying (tau)
vec4 climAt(float lonDeg, float latDeg){ return textureLod(uClimTex, vec2(lonDeg / 360.0 + 0.5, (latDeg + 90.0) / 180.0), 0.0); }
float convection(vec2 ll){
  const float CELL = 3.0;
  float latd = degrees(ll.y);
  if (abs(latd) > 40.0) return 0.0;
  float drift = degrees(5.5 * MS * uWxT) / 0.966;              // the cells drift west with the trades
  float lonP = mod(degrees(ll.x) + drift + 180.0, 360.0);
  vec2 g = vec2(lonP, latd + 90.0) / CELL, gi = floor(g);
  float shift = 0.35 * (uDecl - 23.0);                        // the composite is June: follow the season
  float acc = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec2 cc = gi + vec2(x, y);
    vec2 cid = vec2(mod(cc.x, 120.0), cc.y);
    vec3 h = hash33(vec3(cid, 7.0));
    vec2 ctr = (cc + 0.2 + 0.6 * h.xy) * CELL;
    float latc = ctr.y - 90.0;
    float lonc = ctr.x - drift - 180.0;
    float p = climAt(lonc, latc - shift).r;
    if (p < 0.02) continue;
    float land = climAt(lonc, latc).g;
    // birth hour: afternoon over land (surface heating), night–dawn over the sea
    float tb = mix(22.0 + 8.0 * h.z, 12.5 + 4.0 * h.z, step(0.5, land));
    float s = uSolarT + lonc / 15.0 - tb;
    float day = floor(s / 24.0), age = s - 24.0 * day;
    vec3 h2 = hash33(vec3(cid, day + 3.0));
    float life = 6.0 + 7.0 * h2.x;
    if (age > life || h2.y > p * uConv * mix(0.45, 1.0, land)) continue;
    float R = 0.45 + 1.4 * h2.z * h2.z;                        // anvil radius, degrees (50 – 200 km)
    vec2 dd = vec2((lonP - ctr.x) * cos(radians(latc)), latd + 90.0 - ctr.y);
    dd *= vec2(0.75 + 0.5 * h.x, 1.25 - 0.5 * h.x);            // anvils are seldom round
    float d = length(dd);
    float grow = smoothstep(0.0, 3.0, age), decay = smoothstep(life * 0.45, life, age);
    float r = R * (0.3 + 0.7 * grow) * (1.0 + 0.35 * decay);
    if (d > r * 1.9) continue;
    float edge = d / r + 0.35 * snoise(vec3(dd * 1.2, h.z * 40.0 + day));
    // texture: a rotated, rescaled piece of a real tropical storm's cloud shield (VIIRS, atlas cell 4)
    float an = h2.x * 40.0;
    vec2 qf = mat2(cos(an), sin(an), -sin(an), cos(an)) * dd / r * 0.13 + 0.5 + (h.xy - 0.5) * 0.08;
    float fab = textureLod(uStormTex, vec2(qf.x * 0.25, qf.y * 0.5), 1.0).r;
    float t = tauOf(min(fab * 1.12, 0.9)) * (1.0 - smoothstep(0.45, 1.05, edge)) * (1.0 - 0.75 * decay);
    // anvil blow-off: a fibrous, semi-transparent cirrus skirt, spreading as the cell ages
    float fib = 0.5 + 0.5 * snoise(vec3(dd.x * 1.5, dd.y * 6.0, h.y * 20.0));
    t += (0.35 + 0.5 * decay) * fib * (1.0 - smoothstep(0.8, 1.8, edge)) * grow * (1.0 - smoothstep(0.85, 1.0, age / life));
    acc = max(acc, t);
  }
  return acc;
}

// cirrus: thin, fibrous veils drawn out along the jets and over tropical convection (tau <~ 0.6)
float cirrus(vec2 uvs){
  vec2 ll = llOfUV(uvs);
  vec3 p = vec3(cos(ll.x) * 6.0, sin(ll.x) * 6.0, ll.y * 60.0);      // stretched along the zonal flow
  float n = 0.6 * snoise(p) + 0.3 * snoise(p * vec3(2.1, 2.1, 2.7) + 4.1) + 0.15 * snoise(p * vec3(4.7, 4.7, 6.3) + 9.3);
  return smoothstep(0.05, 0.7, n);
}

// the cloud value (same scale as the NASA composite) at uv, reusing the flow computed nearby
float cloudAt(vec2 uv, Flow F, vec2 gx, vec2 gy, float fpRad){
  if (uWx < 0.5) return textureGrad(uCloudTex, uv, gx, gy).r;
  vec2 o = uv - F.uv;
  float tb = F.w0 * bgTau(F.a0 + o, F.b0 + o, F.m, gx, gy) + F.w1 * bgTau(F.a1 + o, F.b1 + o, F.m, gx, gy);
  vec2 ll = llOfUV(uv);
  vec3 pb = dirOfLL(ll);
  float latd = degrees(ll.y);
  // diurnal cycle at this place (local solar time)
  float hl = mod(uSolarT + degrees(ll.x) / 15.0, 24.0);
  vec4 cl = climAt(degrees(ll.x), latd);
  float morning = smoothstep(4.0, 7.0, hl) * (1.0 - smoothstep(10.0, 13.5, hl));
  float afternoon = smoothstep(11.0, 14.0, hl) * (1.0 - smoothstep(17.5, 20.5, hl));
  float nightSc = 1.0 - smoothstep(6.0, 10.0, hl) * (1.0 - smoothstep(19.0, 23.0, hl));
  // tropical land clears by morning; stratocumulus is thickest at dawn and thins in the afternoon
  tb *= uCovBoost * (1.0 - 0.5 * cl.g * (1.0 - smoothstep(22.0, 38.0, abs(latd))) * morning);
  // stratocumulus decks: closed-cell sheets (~25 km cells, dark rims) off the western coasts
  if (cl.b > 0.01){
    vec2 cp = vec2(degrees(ll.x) * cos(ll.y) + 0.12 * uWxT, latd) * 4.5;
    vec2 wv = worley2(cp);
    float f = smoothstep(0.15, 0.6, fpRad * RKM / 25.0);
    float cellv = mix(smoothstep(0.02, 0.32, wv.y - wv.x), 0.7, f);
    // the real deck (its shape comes from the composite) gets its cellular texture and diurnal cycle
    tb = mix(tb, tb * (0.45 + 0.8 * cellv) * (1.2 + 0.35 * nightSc - 0.45 * afternoon), cl.b);
  }
  float sup; float ts = storms(pb, fpRad, sup);
  float t = tb * (1.0 - 0.85 * sup) + ts;
  t = max(t, convection(ll));
  // cirrus veils, carried by the same flow
  float ls = latd - 0.3 * uDecl;
  float ciM = 0.8 * exp(-sq((abs(ls) - 42.0) / 12.0)) + 0.6 * cl.r + 0.5 * sup;
  if (ciM > 0.03) t += uCirrus * ciM * (F.w0 * cirrus(F.a0 + o) + F.w1 * cirrus(F.a1 + o));
  // fair-weather cumulus over warm land by day (about 1-10 km cells, a haze when unresolved)
  float cu = cl.g * smoothstep(9.5, 12.5, hl) * (1.0 - smoothstep(16.0, 19.0, hl)) * (1.0 - smoothstep(25.0, 50.0, abs(latd - 0.8 * uDecl)));
  if (cu > 0.01){
    float f = smoothstep(0.25, 1.4, fpRad * 900.0);
    float n = f < 0.99 ? smoothstep(0.15, 0.7, snoise(pb * 900.0 + uWxT * 0.02)) : 0.0;
    t += cu * 0.9 * mix(n, 0.08, f);
  }
  return vOf(t);
}
// coverage 0.5 = the composite as observed; it scales the optical depth (lower keeps only thick cloud)
float cloudAlpha(float v){
  float t = max(tauOf(v) - max(0.5 - uClouds, 0.0) * 12.0, 0.0) * exp2((uClouds - 0.5) * 3.0);
  return (1.0 - exp(-t)) * smoothstep(0.0, 0.04, uClouds);
}
float cloudHit(vec3 ro, vec3 rd, mat3 R, float k, out float lift){
  lift = 0.0;
  float hcl = HCL * (1.0 + 0.5 * max(uExag * uTexOK - 1.0, 0.0));
  vec2 h = sphereHit(ro, rd, vec3(0.0), 1.0 + hcl);
  float tt = h.x;
  if (h.x > h.y){
    h = sphereHit(ro, rd, vec3(0.0), 1.0 + hcl + k * uHMax * 0.7);
    if (h.x > h.y || h.y < 0.0) return -1.0;
    tt = max(-dot(ro, rd), 0.0);
  }
  for (int i = 0; i < 2; i++){
    vec3 q = normalize(ro + rd * tt);
    lift = k * hgtKm(R * q, 3.5);
    vec2 g = sphereHit(ro, rd, vec3(0.0), 1.0 + hcl + lift);
    if (g.x > g.y) return -1.0;
    tt = g.x;
  }
  return tt > 0.0 ? tt : -1.0;
}

// ---- aurora: oxygen 557.7 nm (green, ~100–150 km) and 630 nm (red, 200–300 km), N₂⁺ 427.8 nm fringe
vec3 aurora(vec3 ro, vec3 rd, float tEnd){
  const float R1 = 1.0 + 85.0 / RKM, R2 = 1.0 + 330.0 / RKM;
  vec2 h = sphereHit(ro, rd, vec3(0.0), R2);
  if (h.x > h.y || h.y < 0.0) return vec3(0.0);
  float t0 = max(h.x, 0.0), t1 = min(h.y, tEnd);
  vec2 hi = sphereHit(ro, rd, vec3(0.0), R1);
  if (hi.x < hi.y && hi.x > t0) t1 = min(t1, hi.x);
  if (t1 <= t0) return vec3(0.0);
  float th0 = radians(19.0 + 2.2 * uKp), w = radians(2.2 + 0.45 * uKp);
  const int N = 28;
  float dt = (t1 - t0) / float(N);
  vec3 acc = vec3(0.0);
  for (int i = 0; i < N; i++){
    vec3 p = ro + rd * (t0 + (float(i) + 0.5) * dt);
    float r = length(p); vec3 d = p / r;
    vec3 c = dot(d, uAurN) > dot(d, uAurS) ? uAurN : uAurS;
    float colat = acos(clamp(dot(d, c), -1.0, 1.0));
    if (abs(colat - th0) > w * 3.5) continue;
    float fold = 0.04 * snoise(d * 9.0 + vec3(0.0, uWall * 0.05, 0.0)) + 0.018 * snoise(d * 31.0 + uWall * 0.08);
    float x = (colat - th0 + fold) / w;
    float ring = (exp(-x * x) * 0.2 + exp(-pow((x + 0.55) * 6.0, 2.0)) + 0.5 * exp(-pow((x - 0.35) * 7.0, 2.0))) * (0.4 + 0.6 * smoothstep(-0.2, 0.6, snoise(d * 5.0 + 20.0 + uWall * 0.02)));
    float rays = 0.2 + 0.8 * pow(0.5 + 0.5 * snoise(d * 160.0 + vec3(uWall * 0.3, 0.0, 0.0)), 3.0);
    float hk = (r - 1.0) * RKM;
    ring *= 0.1 + 0.9 * smoothstep(0.15, -0.25, dot(d, uSunDir));
    float green = smoothstep(90.0, 104.0, hk) * exp(-max(hk - 104.0, 0.0) / 28.0);
    float red = exp(-pow((hk - 230.0) / 70.0, 2.0)) * 0.22;
    float blue = exp(-pow((hk - 96.0) / 5.0, 2.0)) * 0.25 * smoothstep(5.0, 8.0, uKp);
    acc += ring * rays * (vec3(0.12, 1.0, 0.32) * green + vec3(1.0, 0.08, 0.12) * red * (0.5 + 0.12 * uKp) + vec3(0.5, 0.25, 1.0) * blue) * dt;
  }
  return acc * 16.0 * uAurora;
}

vec3 cityLight(float v){
  float l = pow(v, 2.4);
  return mix(bb(2100.0), bb(3900.0), smoothstep(0.35, 0.95, v)) * l * 1.2;
}

// ---- surface -------------------------------------------------------------------------------
vec3 earthSurface(vec3 pw, vec3 pb, vec3 rd, vec3 L, Atm A, mat3 R, float k, float fp, vec2 gx, vec2 gy, float shellTop, float cloudDim){
  vec3 n0 = normalize(pw);
  vec2 uv = uvOf(pb);
  float tex = texelAt(uv);
  float ds = max(fp, tex);
  float lod = log2(ds / uTexel);
  float cphi = max(length(pb.xz), 0.02);
  float du = ds / (TAU * cphi), dv = ds / PI;
  vec2 c0 = elevAt(uv, lod);
  float e = c0.r, water = smoothstep(0.25, 0.75, c0.g);
  float hE = max(elevAt(uv + vec2(du, 0.0), lod).r, 0.0), hW = max(elevAt(uv - vec2(du, 0.0), lod).r, 0.0);
  float hN = max(elevAt(uv + vec2(0.0, dv), lod).r, 0.0), hS = max(elevAt(uv - vec2(0.0, dv), lod).r, 0.0);
  vec3 east = vec3(pb.z, 0.0, -pb.x) / cphi, north = cross(pb, east);
  vec2 g = vec2(hE - hW, hN - hS) * k / (2.0 * ds);
  float slope = length(g);
  vec3 nb = pb - g.x * east - g.y * north;
  float fd = smoothstep(0.7, 0.2, fp / tex) * (1.0 - water);
  float rn = 0.5;
  if (fd > 0.0){
    float F = 1.6 / tex;
    float eps = max(fp * 0.7, tex * 0.03);
    rn = ridged(pb * F);
    float rx = ridged((pb + east * eps) * F), ry = ridged((pb + north * eps) * F);
    float amp = (slope * 0.3 + 0.003 * uExag) * tex / 1.6 * fd;
    nb -= (east * (rx - rn) + north * (ry - rn)) / eps * amp;
  }
  vec3 n = transpose(R) * normalize(nb);
  float hGeo = k * max(e, 0.0), hTrue = max(e, 0.0) / RKM;

  vec3 cl = colorAt(uv, gx, gy);
  float mu0s = dot(n0, L);
  float muE = mu0s + sqrt(2.0 * hGeo) - sqrt(2.0 * hTrue);
  vec3 Ts = sunTrans(A, hTrue, muE);
  float sh = cloudDim;
  if (hGeo > 0.0 && muE > -0.02 && dot(Ts, vec3(1.0)) > 0.003){
    float tanZ = sqrt(max(1.0 - mu0s * mu0s, 0.0)) / max(mu0s, 0.02);
    if (hGeo * tanZ > 0.4 * fp) sh *= reliefShadow(pw, L, R, k, ds, shellTop);
  }
  float eC = max(elevAt(uv, lod + 3.0).r, 0.0);
  float ao = clamp(1.0 - (eC - max(e, 0.0)) * k / (ds * 10.0), 0.55, 1.0) * (0.75 + 0.25 * dot(n, n0));
  vec3 skyAmb = vec3(0.05, 0.085, 0.16) * smoothstep(-0.15, 0.4, mu0s) * uAtmo * ao;
  float wc = max((1.0 - smoothstep(0.006, 0.02, max(cl.r, cl.g))) * step(cl.r, cl.b) * smoothstep(0.0, 0.15, c0.g),
                 (1.0 - smoothstep(0.02, 0.06, dot(cl, vec3(0.2126, 0.7152, 0.0722)))) * smoothstep(0.1, 0.4, c0.g)) * (1.0 - smoothstep(0.25, 0.9, e));
  water = clamp(max(water, wc), 0.0, 1.0);
  vec3 land = vec3(0.0), sea = vec3(0.0);
  if (water < 0.999){
    vec3 alb = cl * 0.8;
    float mx = max(cl.r, max(cl.g, cl.b)), mn = min(cl.r, min(cl.g, cl.b));
    float snow = smoothstep(0.32, 0.6, mn) * smoothstep(0.75, 0.9, mn / max(mx, 1e-4));
    float fs = smoothstep(1.6, 0.6, fp / tex) * step(0.01, snow);
    if (fs > 0.0){
      if (fd <= 0.0) rn = ridged(pb * (1.6 / tex));
      float steep = smoothstep(0.25, 0.7, slope / max(uExag, 1.0));
      float s2 = smoothstep(0.25, 0.75, snow * 1.2 - steep * 0.55 + (rn - 0.42) * 0.9);
      s2 = mix(snow, s2, fs);
      vec3 rock = mix(vec3(0.065, 0.06, 0.055), alb, 0.2);
      alb = mix(rock, vec3(0.74, 0.76, 0.8), s2) * mix(1.0, 0.92 + 0.16 * rn, fd);
    } else if (fd > 0.0) alb *= 0.92 + 0.16 * rn * fd;
    float mu0 = max(dot(n, L), 0.0);
    land = alb / PI * ESUN * (Ts * mu0 * sh + skyAmb);
  }
  if (water > 0.001){
    float depth = max(-e, 0.0);
    vec3 alb = mix(vec3(0.0035, 0.011, 0.032), vec3(0.012, 0.045, 0.058), exp(-depth / 0.06));
    alb = max(alb, cl * 0.8 * (1.0 - wc));
    // sea ice: edges follow the date (Arctic minimum in September, Antarctic maximum in September)
    float nz = fbm(pb * 9.0, 3);
    float yy = pb.y + 0.03 * nz;
    float ice = max(smoothstep(uIceN - 0.01, uIceN + 0.01, yy), 1.0 - smoothstep(-uIceS - 0.01, -uIceS + 0.01, yy));
    alb = mix(alb, vec3(0.55, 0.6, 0.66) * (0.85 + 0.2 * nz), ice);
    float shw = cloudDim;
    sea = alb / PI * ESUN * (Ts * max(mu0s, 0.0) * shw + skyAmb);
    // sun glint: GGX ocean whose roughness follows the local wind (Cox–Munk: σ² = 0.003 + 0.00512·U)
    vec2 ww = wind(vec2(atan(pb.x, pb.z), asin(clamp(pb.y, -1.0, 1.0))), uWxT);
    float U = length(ww) * 0.8 + 2.0;
    vec3 V = -rd, H = normalize(L + V);
    float nh = max(dot(n0, H), 0.0), nv = max(dot(n0, V), 1e-3), nl = max(mu0s, 0.0);
    float a2 = clamp(0.0144 * (0.45 + 0.09 * U), 0.008, 0.035);
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float F = 0.02 + 0.98 * pow(1.0 - max(dot(V, H), 0.0), 5.0);
    float kk = 0.06;
    float G = nl / (nl * (1.0 - kk) + kk) * nv / (nv * (1.0 - kk) + kk);
    sea += (1.0 - ice) * D * F * G / (4.0 * nv) * ESUN * Ts * shw;
    float Fv = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
    sea += (1.0 - ice) * Fv * vec3(0.06, 0.11, 0.22) * ESUN * 0.25 * smoothstep(-0.1, 0.3, mu0s) * uAtmo;
  }
  vec3 col = mix(land, sea, water);
  float night = smoothstep(0.03, -0.12, mu0s);
  if (uCities > 0.5 && night > 0.0){
    float v = textureGrad(uNightTex, uv, gx, gy).r;
    col += cityLight(v) * night * (0.35 + 0.65 * cloudDim);
  }
  return col;
}

vec3 cloudShade(vec3 pcw, vec3 pcb, vec2 uvc, Flow F, float v, float a, vec3 L, Atm A, mat3 R, float lod, vec2 gx, vec2 gy, float fpRad){
  vec3 nc = normalize(pcw);
  float mu0 = dot(nc, L);
  // cloud-top relief: thicker (brighter) cloud stands taller (≈ 2 → 12 km); bump from the gradient
  float cphi = max(length(pcb.xz), 0.02);
  float ds = uCloudTexel * exp2(max(lod, 0.0)) * 1.5;
  float vE = cloudAt(uvc + vec2(ds / (TAU * cphi), 0.0), F, gx, gy, fpRad);
  float vN = cloudAt(uvc + vec2(0.0, ds / PI), F, gx, gy, fpRad);
  vec3 east = vec3(pcb.z, 0.0, -pcb.x) / cphi, north = cross(pcb, east);
  float hk = 10.0 / RKM;
  vec3 nb = normalize(pcb - ((vE - v) * east + (vN - v) * north) * hk / ds);
  vec3 n = transpose(R) * nb;
  // self-shadowing: cloud mass a little toward the sun
  vec3 Lb = R * L;
  vec3 lt = normalize(Lb - pcb * dot(Lb, pcb));
  vec2 duvL = vec2(dot(lt, east) / (TAU * cphi), dot(lt, north) / PI) * ds * 2.0;
  float ahead = cloudAlpha(cloudAt(uvc + duvL, F, gx * 2.0, gy * 2.0, fpRad * 2.0));
  float self = 1.0 - 0.45 * clamp(ahead - 0.7 * a, 0.0, 1.0);
  vec3 Ts = sunTrans(A, HCL, mu0);
  float alb = mix(0.5, 0.86, smoothstep(0.25, 0.85, v));
  float lit = max(dot(n, L) * 0.75 + 0.25 * mu0 + 0.06, 0.0) / 1.06 * self;
  vec3 skyAmb = vec3(0.05, 0.08, 0.14) * smoothstep(-0.1, 0.3, mu0) * uAtmo;
  return alb / PI * ESUN * (Ts * lit + skyAmb);
}

// the Moon (LRO WAC mosaic, near side toward the Earth), Lommel–Seeliger reflectance, albedo ≈ 0.12
vec3 moonColor(vec3 ro, vec3 rd, vec3 L, float pixAng, out float tM){
  tM = -1.0;
  vec2 h = sphereHit(ro, rd, uMoonPos, RMOON);
  if (h.x > h.y || h.x < 0.0) return vec3(0.0);
  tM = h.x;
  vec3 n = normalize(ro + rd * h.x - uMoonPos);
  vec2 uv = vec2(atan(dot(n, uMoonE), dot(n, uMoonF)) / TAU + 0.5, asin(clamp(dot(n, uMoonN), -1.0, 1.0)) / PI + 0.5);
  float lod = log2(max(h.x * pixAng / RMOON * 1024.0 / TAU, 1e-3));
  vec3 alb = textureLod(uMoonTex, uv, max(lod, 0.0)).rgb * 0.24;
  float mu0 = max(dot(n, L), 0.0), mu = max(dot(n, -rd), 1e-3);
  return alb * ESUN / PI * 2.0 * mu0 / (mu0 + mu);
}

uniform float uDbg;
void main(){
  if (uDbg > 0.5){
    // debug: the cloud field as an equirectangular map (?earthDbg=1)
    vec2 uv = gl_FragCoord.xy / uRes;
    vec3 pb = dirOfLL(llOfUV(uv));
    Flow Fd = flowAt(uv, pb);
    float fp = TAU / uRes.x;
    float v = cloudAt(uv, Fd, vec2(1.0 / uRes.x, 0.0), vec2(0.0, 1.0 / uRes.y), fp);
    float al = cloudAlpha(v);
    gl_FragColor = vec4(uDbg > 1.5 ? vec3(step(0.3, al) * 0.5) : vec3(al) * 0.8 + vec3(0.0, 0.0, 0.05), 1.0);
    return;
  }
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  Atm A = atmE();
  mat3 R = rotY(uRot);
  float k = uExag / RKM * uTexOK;
  float shellTop = 1.0 + k * uHMax + 1e-5;
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  vec2 hs = sphereHit(ro, rd, vec3(0.0), 1.0);
  bool onSphere = hs.x < hs.y && hs.x > 0.0;
  float tS = onSphere ? hs.x : max(-dot(ro, rd), 0.0);
  vec3 pS = R * normalize(ro + rd * tS);
  vec3 dpx = dFdx(pS), dpy = dFdy(pS);

  vec3 col = sky(rd, 1.0) * 0.6;
  if (uMoon > 0.5){
    float tM; vec3 mc = moonColor(ro, rd, L, pixAng, tM);
    if (tM > 0.0) col = mc;
  }
  float tG = -1.0;
  vec2 hO = sphereHit(ro, rd, vec3(0.0), shellTop);
  if (hO.x < hO.y && hO.y > 0.0){
    float tA = max(hO.x, 0.0), tB = onSphere ? hs.x : hO.y;
    tG = marchRelief(ro, rd, tA, tB, onSphere, R, k, pixAng);
  }
  bool ground = tG > 0.0;
  if (!ground) col += sunDisk(rd, L, 1.0) * 0.35;
  // ---- cloud deck ----
  float tC = -1.0, lift = 0.0;
  if (uClouds > 0.001) tC = cloudHit(ro, rd, R, k, lift);
  if (ground && tC > tG) tC = -1.0;
  vec3 pcw = ro + rd * max(tC, 0.0), pcb = R * normalize(pcw);
  vec2 uvc = uvOf(pcb);
  vec2 gxc = uvD(pcb, dpx), gyc = uvD(pcb, dpy);
  float cv = 0.0, ca = 0.0;
  Flow F;
  if (tC > 0.0){
    float fpc = tC * pixAng;
    F = flowAt(uvc, pcb);
    cv = cloudAt(uvc, F, gxc, gyc, fpc);
    // finer, slowly boiling structure at close range (the composite is ~10 km per pixel)
    float fc = smoothstep(1.0, 0.25, fpc / uCloudTexel);
    if (fc > 0.0) cv += fc * fbm(pcb / uCloudTexel * 0.3 + vec3(0.0, uWxT * 0.05, 0.0), 3) * 0.15 * smoothstep(0.05, 0.4, cv) * (1.2 - cv);
    ca = cloudAlpha(cv);
  }
  // ---- surface ----
  vec3 bg = col;
  if (ground){
    vec3 pw = ro + rd * tG, pb = R * normalize(pw);
    float cd = 1.0;
    if (uClouds > 0.001){
      float rDeck = 1.0 + HCL * (1.0 + 0.5 * max(uExag - 1.0, 0.0)) + k * hgtKm(pb, 3.5);
      vec2 hc = sphereHit(pw, L, vec3(0.0), rDeck);
      if (hc.y > 0.0 && dot(pw, pw) < rDeck * rDeck){
        vec3 pq = R * normalize(pw + L * hc.y);
        vec2 uq = uvOf(pq);
        Flow Fq = flowAt(uq, pq);
        cd = 1.0 - 0.8 * cloudAlpha(cloudAt(uq, Fq, uvD(pq, dpx) * 2.0, uvD(pq, dpy) * 2.0, tG * pixAng * 2.0));
      }
    }
    bg = earthSurface(pw, pb, rd, L, A, R, k, tG * pixAng, uvD(pb, dpx), uvD(pb, dpy), shellTop, cd);
  }
  if (ca > 0.002){
    float lodc = log2(max(tC * pixAng, 1e-7) / uCloudTexel);
    vec3 cc = cloudShade(pcw, pcb, uvc, F, cv, ca, L, A, R, lodc, gxc, gyc, tC * pixAng);
    float nightC = smoothstep(0.0, -0.15, dot(normalize(pcw), L));
    if (uCities > 0.5 && nightC > 0.0) cc += cityLight(textureLod(uNightTex, uvc, max(lodc + 1.5, 2.0)).r) * 0.35 * nightC;
    bg = mix(bg, cc, ca);
  }
  // ---- atmosphere ----
  vec2 ha = sphereHit(ro, rd, vec3(0.0), RTOP);
  if (ha.x < ha.y && ha.y > 0.0){
    float t0 = max(ha.x, 0.0);
    float tEnd = ground ? tG : ha.y;
    if (ca > 0.002) tEnd = mix(tEnd, tC, ca);
    float t1 = min(tEnd, ha.y);
    if (t1 > t0){
      vec3 T;
      vec3 S = atmosphere(ro, rd, t0, t1, L, A, T) * ESUN;
      bg = S + T * bg;
    }
  }
  col = bg;
  if (uAurora > 0.0) col += aurora(ro, rd, ground ? tG : 1e9);
  gl_FragColor = vec4(col, 1.0);
}`;

// live-cloud extraction: VIIRS corrected-reflectance true colour → cloud brightness on the composite's
// scale, against the Blue Marble surface; no-data (polar night) and sea ice fall back to the composite
const LIVE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uLive, uColTex, uJanTex, uElevTex, uCloudTex;
uniform float uWinter, uIceN, uIceS;
varying vec2 vUv;
void main(){
  vec2 uv = vUv;
  vec3 O = texture2D(uLive, uv).rgb;
  vec3 S = pow(max(mix(textureLod(uColTex, uv, 1.0).rgb, textureLod(uJanTex, uv, 0.0).rgb, uWinter), 0.0), vec3(1.0 / 2.2));
  vec2 el = textureLod(uElevTex, uv, 0.0).rg;
  float wat = smoothstep(0.3, 0.7, el.g) * step(el.r, -0.005);
  vec3 Y3 = vec3(0.2126, 0.7152, 0.0722);
  float Yo = dot(O, Y3), Ys = dot(S, Y3);
  float white = min(O.r, min(O.g, O.b)) / max(max(O.r, max(O.g, O.b)), 1e-3);
  float vo = clamp((Yo - 0.17) / 0.8, 0.0, 1.0);
  float ref = max(Ys * 1.1 + 0.05, 0.17);
  float vl = clamp((Yo - ref) / max(0.9 - ref, 0.08), 0.0, 1.0) * clamp((white - 0.55) / 0.25, 0.0, 1.0);
  float v = 0.84 * pow(mix(vl, vo, wat), 1.5);
  float lat = (uv.y - 0.5) * 3.14159265;
  float sy = sin(lat);
  float keep = step(0.004, max(O.r, max(O.g, O.b)))
             * (1.0 - smoothstep(uIceN - 0.06, uIceN - 0.02, sy)) * smoothstep(-uIceS - 0.02, -uIceS + 0.02 + 0.04, sy);
  float fb = texture2D(uCloudTex, uv).r;
  v = mix(fb, v, keep);
  gl_FragColor = vec4(v, v, v, 1.0);
}`;

/* ================================================================================================
   JS helpers
   ================================================================================================ */
const llE = (lat, lon) => [Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R)];
const wrap180 = a => ((a + 180) % 360 + 360) % 360 - 180;
const wrap24 = h => ((h % 24) + 24) % 24;
function hitSphere(o, d, R = 1) {
  const b = o.dot(d), c = o.lengthSq() - R * R, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
const hhmm = h => { h = wrap24(h); return `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`; };
const latStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'B' : 'N'}`;
const lonStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'Đ' : 'T'}`;
function orbitRows(ctx, Rkm, refR = 1) {
  const d = ctx.camera.position.length(), hKm = (d - refR) * Rkm;
  const horizon = Math.sqrt(Math.max(d * d - refR * refR, 0)) * Rkm;
  return [['Độ cao', ctx.fmt.dist(hKm * 1000)], ['Khoảng cách tới đường chân trời', ctx.fmt.dist(horizon * 1000)]];
}
function sunElevStr(n, L) {
  const e = Math.asin(Math.max(-1, Math.min(1, n.dot(L)))) / D2R;
  const tag = e > 6 ? 'ban ngày' : e > -0.8 ? 'bình minh/hoàng hôn' : e > -18 ? 'chạng vạng' : 'ban đêm';
  return `Mặt Trời cao <b>${e.toFixed(1)}°</b> (${tag})`;
}
// Right-handed geography (east = counter-clockwise seen from the north, like the real planet and the
// NASA maps): body = rotY(−spin)·world, east longitude = atan2(x, z). Afternoon puts the Sun west.
function sunFromHourE(cam, hour, declDeg, out) {
  const az = Math.atan2(cam.x, cam.z) - (hour - 12) * Math.PI / 12, d = declDeg * D2R;
  return out.set(Math.cos(d) * Math.sin(az), Math.sin(d), Math.cos(d) * Math.cos(az));
}
const bodyLatLonE = (p, spin) => ({ lat: Math.asin(p.y / p.length()) / D2R, lon: wrap180((Math.atan2(p.x, p.z) - spin) / D2R) });
const localHourE = (p, L) => wrap24(12 + (Math.atan2(p.x, p.z) - Math.atan2(L.x, L.z)) / D2R / 15);

// ---- calendar & ephemerides (2026; low-precision formulas from the Astronomical Almanac / Meeus) ----
const YEAR = 2026, J2000_YEAR0 = 9496.5;      // days from J2000.0 to 2026-01-01 00:00 UT
const dateOf = doy => new Date(Date.UTC(YEAR, 0, 1) + (Math.round(doy) - 1) * 864e5);
const dateStr = doy => { const d = dateOf(doy); return `${d.getUTCDate()}/${d.getUTCMonth() + 1}`; };
function solar(d) {
  const g = (357.529 + 0.98560028 * d) * D2R, q = 280.459 + 0.98564736 * d;
  const lam = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * D2R, eps = (23.439 - 3.6e-7 * d) * D2R;
  return { ra: Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam)), dec: Math.asin(Math.sin(eps) * Math.sin(lam)) };
}
function lunar(d) {
  const L = (218.316 + 13.176396 * d) * D2R, M = (134.963 + 13.064993 * d) * D2R, F = (93.272 + 13.229350 * d) * D2R;
  const D = (297.850 + 12.190749 * d) * D2R;
  const lam = L + D2R * (6.289 * Math.sin(M) + 1.274 * Math.sin(2 * D - M) + 0.658 * Math.sin(2 * D));
  const bet = 5.128 * D2R * Math.sin(F), dist = 385001 - 20905 * Math.cos(M), eps = 23.439 * D2R;
  const x = Math.cos(bet) * Math.cos(lam);
  const y = Math.cos(eps) * Math.cos(bet) * Math.sin(lam) - Math.sin(eps) * Math.sin(bet);
  const z = Math.sin(eps) * Math.cos(bet) * Math.sin(lam) + Math.cos(eps) * Math.sin(bet);
  return { ra: Math.atan2(y, x), dec: Math.asin(z), dist };
}
const declOf = doy => solar(J2000_YEAR0 + doy - 1 + 0.5).dec / D2R;

/* ---- assets (assets/earth/, see CREDITS.txt) ---- */
const EARTH_URL = f => new URL(`../../../assets/earth/${f}`, import.meta.url).href;
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
    img.onerror = () => reject(new Error(`không tải được ${url}`));
    img.src = url;
  });
}
// Elevation WebP: R,G = (metres + 11000) as a 16-bit big-endian integer, B = water fraction.
// → RG16F texture (km, water) with a CPU-built box-filtered mip chain; rows flipped so v = 0 is south.
async function loadElevation(url, keep = 1, repeat = true) {
  const { W, H, px } = await decodeImage(url);
  let lv = new Float32Array(W * H * 2), max = -1e9;
  for (let y = 0; y < H; y++) {
    let si = (H - 1 - y) * W * 4, di = y * W * 2;
    for (let x = 0; x < W; x++, si += 4, di += 2) {
      const e = (((px[si] << 8) | px[si + 1]) - 11000) / 1000;
      lv[di] = e; lv[di + 1] = px[si + 2] / 255;
      if (e > max) max = e;
    }
  }
  const mips = [];
  let w = W, h = H, cpu = null;
  for (;;) {
    const half = new Uint16Array(w * h * 2);
    for (let i = 0; i < half.length; i++) half[i] = toHalf(lv[i]);
    mips.push({ data: half, width: w, height: h });
    if (w === W / keep && !cpu) cpu = { data: lv, W: w, H: h };
    if (w === 1 && h === 1) break;
    const w2 = Math.max(1, w >> 1), h2 = Math.max(1, h >> 1), nx = new Float32Array(w2 * h2 * 2);
    const sx = w > 1 ? 2 : 1, sy = h > 1 ? 2 : 1;
    for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) for (let c = 0; c < 2; c++) {
      const a = ((y * sy) * w + x * sx) * 2 + c, b = a + (sx - 1) * 2, d = a + (sy - 1) * w * 2, f = d + (sx - 1) * 2;
      nx[(y * w2 + x) * 2 + c] = 0.25 * (lv[a] + lv[b] + lv[d] + lv[f]);
    }
    lv = nx; w = w2; h = h2;
  }
  const tex = new THREE.DataTexture(mips[0].data, W, H, THREE.RGFormat, THREE.HalfFloatType);
  tex.mipmaps = mips;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return { tex, max, cpu };
}
function loadImageTex(url, { srgb = false, red = false, aniso = 1, maxSize = 16384, repeat = true }) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    const img = tex.image;
    if (img.width > maxSize) {
      const s = maxSize / img.width, cv = document.createElement('canvas');
      cv.width = Math.round(img.width * s); cv.height = Math.round(img.height * s);
      const g = cv.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(img, 0, 0, cv.width, cv.height);
      tex.image = cv;
    }
    tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    if (red) tex.format = THREE.RedFormat;
    tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = aniso;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(`không tải được ${url}`))));
}
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
function solidTex(data, format, type) {
  const t = new THREE.DataTexture(data, 1, 1, format, type); t.needsUpdate = true; return t;
}
function sampleElev(cpu, lat, lon, lon0 = -180, lat0 = -90, spanLon = 360, spanLat = 180, ch = 0) {
  const { data, W, H } = cpu;
  const x = (lon - lon0) / spanLon * W - 0.5, y = (lat - lat0) / spanLat * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.max(0, Math.min(H - 2, Math.floor(y))), fx = x - x0, fy = Math.max(0, Math.min(1, y - y0));
  const at = (i, j) => data[(j * W + ((i % W) + W) % W) * 2 + ch];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

/* ---- weather climatology (CPU bake, 360×180): R = deep-convection likelihood (where the real June
   composite is cloudy in the tropics), G = land fraction, B = stratocumulus decks (subtropical oceans
   off western coasts, where cold upwelling water lies under the subsiding branch of the Hadley cell) */
const SC_BOXES = [[15, 35, -145, -115], [-35, -5, -100, -72], [-30, -5, -15, 14], [15, 32, -35, -12], [-35, -18, 90, 115]];
function bakeClimate(cloudImg, elevCPU) {
  const W = 360, H = 180;
  const cv = Object.assign(document.createElement('canvas'), { width: W, height: H });
  const g = cv.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingQuality = 'high'; g.drawImage(cloudImg, 0, 0, W, H);
  const px = g.getImageData(0, 0, W, H).data;
  const raw = new Float32Array(W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) raw[j * W + i] = px[((H - 1 - j) * W + i) * 4] / 255;   // row 0 = south
  const sm = new Float32Array(W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    let s = 0, n = 0;
    for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) {
      const jj = j + dj; if (jj < 0 || jj >= H) continue;
      s += raw[jj * W + ((i + di + W) % W)]; n++;
    }
    sm[j * W + i] = s / n;
  }
  const out = new Uint8Array(W * H * 4);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const lat = -90 + j + 0.5, lon = -180 + i + 0.5;
    const water = elevCPU ? sampleElev(elevCPU, lat, lon, -180, -90, 360, 180, 1) : 1;
    const land = 1 - water;
    let sc = 0;
    for (const [a, b, c, d] of SC_BOXES) sc = Math.max(sc, sstep(a - 4, a + 2, lat) * (1 - sstep(b - 2, b + 4, lat)) * sstep(c - 5, c + 3, lon) * (1 - sstep(d - 3, d + 5, lon)));
    sc *= water;
    const trop = 1 - sstep(16, 28, Math.abs(lat - 6));
    const conv = sstep(0.12, 0.42, sm[j * W + i]) * trop * (1 - sc) * (0.75 + 0.25 * land);
    const k = (j * W + i) * 4;
    out[k] = Math.round(conv * 255); out[k + 1] = Math.round(land * 255); out[k + 2] = Math.round(sc * 255); out[k + 3] = 255;
  }
  const tex = new THREE.DataTexture(out, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/* ---- storms ---------------------------------------------------------------------------------
   The atlas earth_storms.jpg holds four real storms (2×2 cells, storm-centred azimuthal-equidistant,
   north up). half = half-size of the cell in degrees of arc; hemi = hemisphere the image comes from. */
const STAMPS = [
  { cell: 0, hemi: 1, half: 8, name: 'bão Irma (5/9/2017, cấp 5)' },
  { cell: 1, hemi: 1, half: 8, name: 'bão Florence (11/9/2018, cấp 4)' },
  { cell: 2, hemi: 1, half: 13, name: 'xoáy thuận Bắc Thái Bình Dương (1/2/2022)' },
  { cell: 3, hemi: -1, half: 13, name: 'xoáy thuận Nam Ấn Độ Dương (16/10/2026)' },
  { cell: 4, hemi: 1, half: 8, name: 'một bão nhiệt đới đang mạnh lên ở Tây Thái Bình Dương (16/10/2026)' },
  { cell: 5, hemi: -1, half: 11, name: 'xoáy thuận Đông Nam Thái Bình Dương ngoài khơi Chile (16/10/2026)' },
  { cell: 6, hemi: 1, half: 11, name: 'xoáy thuận Bắc Đại Tây Dương (10/3/2023)' },
];
// tropical-cyclone basins per slot: genesis box [lat0, lat1], [lon0, lon1], hemisphere
const TC_BASINS = [
  [{ h: 1, lat: [8, 17], lon: [128, 160], name: 'Tây Bắc Thái Bình Dương' }],
  [{ h: 1, lat: [10, 16], lon: [-55, -25], name: 'Bắc Đại Tây Dương' }, { h: 1, lat: [10, 15], lon: [-112, -95], name: 'Đông Bắc Thái Bình Dương' }],
  [{ h: -1, lat: [-15, -8], lon: [58, 95], name: 'Nam Ấn Độ Dương' }, { h: -1, lat: [-18, -11], lon: [150, 175], name: 'Nam Thái Bình Dương' }, { h: -1, lat: [-16, -11], lon: [108, 125], name: 'Tây Bắc Úc' }],
  [{ h: 1, lat: [11, 17], lon: [86, 91], name: 'vịnh Bengal' }, { h: 1, lat: [9, 15], lon: [135, 165], name: 'Tây Bắc Thái Bình Dương' }],
];
const ETC_GEN_N = [142, 160, -72, -58, -175, -100, -42];   // NH lows: off Japan/Kuroshio, Gulf Stream, Aleutians, lee of the Rockies…
function smoothLife(a, a0, a1, b0, b1) { return sstep(a0, a1, a) * (1 - sstep(b0, b1, a)); }

/* ================================================================================================
   Scene definition
   ================================================================================================ */
const SATS = {
  g19: { lon: -75.2, name: 'GOES-19 (GOES-East, 75,2°T)' }, g18: { lon: -137.0, name: 'GOES-18 (GOES-West, 137°T)' },
  h9: { lon: 140.7, name: 'Himawari-9 (140,7°Đ)' }, m0: { lon: 0, name: 'Meteosat (0°)' }, iodc: { lon: 45.5, name: 'Meteosat IODC (45,5°Đ)' },
};
const R_GEO = 42164 / R_KM;

export default [{
  id: 'earth',
  name: 'Trái Đất',
  category: 'planets',
  order: 20,
  subtitle: 'Hành tinh đất đá · ngôi nhà của chúng ta',
  swatch: ['#9fd0ff', '#0b2a5a'],
  blurb: 'Hành tinh duy nhất được biết có sự sống, 71 % bề mặt là đại dương. Bề mặt là <b>dữ liệu thật</b> (Blue Marble, ETOPO 2022, Black Marble) và Trái Đất <b>chuyển động thật</b>: quay một vòng mỗi 23 h 56 min so với các vì sao, mùa theo trục nghiêng 23,44°, mây trôi theo hoàn lưu khí quyển — gió mậu dịch, gió tây ôn đới uốn khúc, xoáy thuận cuộn ngược chiều kim đồng hồ ở Bắc bán cầu và thuận chiều ở Nam bán cầu (lực Coriolis), bão nhiệt đới có mắt bão, mây đối lưu bùng lên mỗi chiều trên lục địa nhiệt đới.',
  intro: '<b>O</b>: nhìn từ vệ tinh địa tĩnh như ảnh GOES/Himawari tua nhanh · <b>B</b>: bay theo một cơn bão · <b>L</b>: tìm Mặt Trăng · <b>Y</b>: dãy Himalaya lúc bình minh.',
  facts: [
    ['Bán kính trung bình', '6 371 km'], ['Khối lượng', '5,97×10²⁴ kg'], ['Trọng lực bề mặt', '9,81 m/s²'],
    ['Độ nghiêng trục quay', '23,44°'], ['Ngày Mặt Trời / ngày sao', '24 h / 23 h 56 min 4 s'],
    ['Tốc độ quay ở xích đạo', '1 674 km/h'],
    ['Suất phản chiếu Bond', '0,30'], ['Khí quyển', 'N₂ 78 %, O₂ 21 %, Ar 0,9 % · 1 013 hPa'],
    ['Độ cao thang (Rayleigh / Mie)', '8 km / 1,2 km'],
    ['Đỉnh cao nhất / điểm sâu nhất', 'Everest 8 849 m / vực Challenger −10 935 m'],
    ['Dữ liệu', 'NASA Blue Marble NG (6 & 1/2004), NOAA ETOPO 2022, NASA Black Marble 2016, mây NASA Blue Marble, bão từ NASA GIBS/VIIRS, Mặt Trăng LRO'],
  ],
  unitKm: 6371,
  camera: { distance: 3.2, polar: 72, azimuth: 20, min: 1.03, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.3,
  params: [
    { key: 'hour', label: 'Giờ địa phương (dưới camera)', type: 'range', min: 0, max: 24, step: 0.1, default: 13.5, fmt: v => hhmm(v),
      note: '12:00 = Mặt Trời sau lưng bạn (Trái Đất tròn đầy); 18:00 = đường phân chia đi qua giữa đĩa; 00:00 = mặt đêm. Khi nhìn từ vệ tinh địa tĩnh, thanh này chạy như đồng hồ.' },
    { key: 'date', label: 'Ngày trong năm (mùa)', type: 'range', min: 1, max: 365, step: 1, default: 213, fmt: v => `${dateStr(v)} · δ ${declOf(v).toFixed(1)}°`,
      note: 'Trục quay nghiêng 23,44° nên điểm Mặt Trời đứng bóng dời từ 23,4°N (21/12) lên 23,4°B (21/6). Theo ngày: vĩ độ Mặt Trời, mép băng biển, tuyết mùa đông (ảnh Blue Marble tháng 1 và 6), mùa bão, dải hội tụ nhiệt đới và vị trí Mặt Trăng.' },
    { key: 'pace', label: 'Tua nhanh thời gian Trái Đất', type: 'seg', options: [[0, 'Dừng'], [1, 'Thật'], [120, '×120'], [360, '×360'], [1200, '×1200']], default: 120,
      note: 'Số giây Trái Đất trôi qua trong một giây mô phỏng (nhân thêm thanh Tốc độ thời gian chung, mặc định 6×). ×1200 ở 6×: một ngày trong 12 giây, như phim tua nhanh ảnh vệ tinh.' },
    { key: 'sat', label: 'Nhìn từ vệ tinh địa tĩnh', type: 'seg', options: [['off', 'Tự do'], ['g19', 'GOES-E'], ['m0', 'Meteosat'], ['iodc', 'IODC'], ['h9', 'Himawari'], ['g18', 'GOES-W']], default: 'off',
      note: 'Camera đứng yên trên xích đạo ở độ cao 35 786 km, quay cùng Trái Đất — đúng góc nhìn của các vệ tinh khí tượng. Ngày và đêm lướt qua đĩa, mây chuyển động. Cuộn chuột để phóng to.' },
    { key: 'exag', label: 'Phóng đại địa hình', type: 'range', min: 1, max: 40, step: 0.5, default: 8, fmt: v => '×' + v.toFixed(1), note: '×1 = tỉ lệ thật: Everest chỉ cao 0,14 % bán kính, gần như vô hình từ quỹ đạo.' },
    { key: 'clouds', label: 'Độ che phủ mây', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5, note: '0,5 = đúng như ảnh mây thật.' },
    { key: 'wx', label: 'Thời tiết chuyển động', type: 'toggle', default: true, note: 'Tắt: ảnh mây tổng hợp đứng yên (như ảnh chụp một thời điểm).' },
    { key: 'live', label: 'Mây thật (NASA GIBS, hôm qua)', type: 'toggle', default: false, note: 'Tải ảnh toàn cầu VIIRS (NOAA-20) của ngày hôm qua từ NASA GIBS và tách lớp mây. Cần mạng; nếu không tải được sẽ dùng ảnh mây đi kèm.' },
    { key: 'atmo', label: 'Mật độ khí quyển', type: 'range', min: 0, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), note: 'Nhân hệ số tán xạ Rayleigh, Mie và hấp thụ ozone.' },
    { key: 'cities', label: 'Đèn thành phố', type: 'toggle', default: true },
    { key: 'moon', label: 'Mặt Trăng', type: 'toggle', default: true, note: 'Đúng vị trí, khoảng cách (~60 bán kính Trái Đất) và pha theo ngày giờ.' },
    { key: 'kp', label: 'Bão địa từ (chỉ số Kp)', type: 'range', min: 0, max: 9, step: 0.1, default: 4, fmt: v => 'Kp ' + v.toFixed(1), note: 'Kp càng lớn, vành cực quang càng mở rộng về phía xích đạo.' },
    { key: 'aurora', label: 'Cực quang', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Himalaya', key: 'y', title: 'Bay tới dãy Himalaya lúc bình minh (bấm lần nữa để quay lại)', run: ctx => ctx.scene._earth?.himalaya(ctx) },
    { label: 'Vệ tinh', key: 'o', title: 'Lần lượt các vệ tinh khí tượng địa tĩnh', run: ctx => ctx.scene._earth?.nextSat(ctx) },
    { label: 'Theo bão', key: 'b', title: 'Bay tới và bám theo một cơn bão', run: ctx => ctx.scene._earth?.storm(ctx) },
    { label: 'Mặt Trăng', key: 'l', title: 'Xoay giờ và góc nhìn để thấy Mặt Trăng sau Trái Đất', run: ctx => ctx.scene._earth?.moon(ctx) },
    { label: 'Ban đêm', key: 'n', title: 'Nhìn mặt đêm: đèn thành phố và cực quang', run: ctx => { ctx.params.hour = 0.5; ctx.scene._earth?.reSun(); ctx.toast('Nửa đêm dưới camera: ánh đèn đô thị (Black Marble 2016) bám theo bờ biển, sông ngòi và đường sá; vành cực quang xanh lục (oxy 557,7 nm) bao quanh cực từ.'); } },
    { label: 'Hoàng hôn', key: 'j', title: 'Đặt đường phân chia ngày–đêm vào giữa đĩa', run: ctx => { ctx.params.hour = 17.8; ctx.scene._earth?.reSun(); ctx.toast('Ánh nắng chiều đi xuyên qua ~40 lần bề dày khí quyển: ánh xanh bị tán xạ hết, chỉ còn đỏ cam.'); } },
    { label: 'Bão địa từ', key: 'g', title: 'Bão địa từ cực mạnh (Kp 8,7 như tháng 5/2024)', run: ctx => { ctx.params.kp = 8.7; ctx.params.aurora = true; ctx.params.hour = 0.5; ctx.scene._earth?.reSun(); ctx.toast('Bão địa từ G5 (5/2024, Kp ≈ 9): vành cực quang mở rộng xuống vĩ độ ~45°, ánh đỏ oxy 630 nm ở độ cao lớn hiện rõ.'); } },
  ],
  create(ctx) {
    const R = ctx.renderer, caps = R.capabilities;
    const maxTex = caps.maxTextureSize, aniso = Math.min(8, caps.getMaxAnisotropy());
    const ocean = () => solidTex(new Uint16Array([toHalf(-4), toHalf(1)]), THREE.RGFormat, THREE.HalfFloatType);
    const black = () => solidTex(new Uint8Array([0]), THREE.RedFormat, THREE.UnsignedByteType);
    const navy = () => solidTex(new Uint8Array([2, 5, 20, 255]), THREE.RGBAFormat, THREE.UnsignedByteType);
    const grey = () => solidTex(new Uint8Array([120, 120, 120, 255]), THREE.RGBAFormat, THREE.UnsignedByteType);
    const V4 = () => Array.from({ length: NST }, () => new THREE.Vector4());
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) }, uAurN: { value: new THREE.Vector3(0, 1, 0) }, uAurS: { value: new THREE.Vector3(0, -1, 0) },
      uRot: { value: 0 }, uClouds: { value: 0.5 }, uCities: { value: 1 }, uAtmo: { value: 1 }, uKp: { value: 4 }, uAurora: { value: 1 },
      uExag: { value: 8 }, uHMax: { value: 8.2 }, uTexOK: { value: 0 },
      uTexel: { value: 2 * Math.PI / 4096 }, uCloudTexel: { value: 2 * Math.PI / 4096 },
      uColTex: { value: navy() }, uJanTex: { value: navy() }, uElevTex: { value: ocean() }, uNightTex: { value: black() }, uCloudTex: { value: black() },
      uPatchCol: { value: navy() }, uPatchElev: { value: ocean() }, uStormTex: { value: black() }, uClimTex: { value: black() }, uMoonTex: { value: grey() },
      uWxT: { value: 0 }, uSolarT: { value: 0 }, uDecl: { value: 18 }, uWx: { value: 1 }, uWinter: { value: 0 }, uIceN: { value: 0.96 }, uIceS: { value: 0.88 }, uConv: { value: 1 },
      uStA: { value: V4() }, uStB: { value: V4() }, uStC: { value: V4() },
      uMoonPos: { value: new THREE.Vector3(0, 0, -60) }, uMoonF: { value: new THREE.Vector3(0, 0, 1) }, uMoonN: { value: new THREE.Vector3(0, 1, 0) }, uMoonE: { value: new THREE.Vector3(1, 0, 0) },
      uCovBoost: { value: +(new URLSearchParams(location.search).get('earthCov') || 1.15) }, uCirrus: { value: +(new URLSearchParams(location.search).get('earthCi') || 0.6) },
      uMoon: { value: 1 }, uDbg: { value: +(new URLSearchParams(location.search).get('earthDbg') || 0) },
    };
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: EARTH_FRAG, depthTest: false, depthWrite: false });
    const U = uniforms;
    let elevCPU = null, patchCPU = null, cloudBundled = null;
    const ready = Promise.all([
      retry(loadImageTex, EARTH_URL('earth_color_8k.jpg'), { srgb: true, aniso, maxSize: maxTex }),
      retry(loadElevation, EARTH_URL('earth_elev_4k.webp'), 2),
      retry(loadImageTex, EARTH_URL('earth_night_8k.jpg'), { red: true, aniso: 4, maxSize: maxTex }),
      retry(loadImageTex, EARTH_URL('earth_clouds_4k.jpg'), { red: true, aniso: 4, maxSize: maxTex }),
      retry(loadImageTex, EARTH_URL('earth_himalaya_color.jpg'), { srgb: true, aniso, repeat: false }),
      retry(loadElevation, EARTH_URL('earth_himalaya_elev.webp'), 1, false),
      retry(loadImageTex, EARTH_URL('earth_color_jan_4k.jpg'), { srgb: true, aniso, maxSize: maxTex }),
      retry(loadImageTex, EARTH_URL('earth_storms.jpg'), { red: true, repeat: false }),
      retry(loadImageTex, EARTH_URL('moon_lroc_1k.jpg'), { srgb: true }),
    ]).then(([col, el, night, cl, pcol, pel, jan, storm, moon]) => {
      U.uColTex.value = col; U.uElevTex.value = el.tex; U.uNightTex.value = night; U.uCloudTex.value = cl;
      U.uPatchCol.value = pcol; U.uPatchElev.value = pel.tex; U.uJanTex.value = jan; U.uStormTex.value = storm; U.uMoonTex.value = moon;
      U.uTexel.value = 2 * Math.PI / el.tex.image.width;
      U.uCloudTexel.value = 2 * Math.PI / cl.image.width;
      U.uHMax.value = Math.max(el.max, pel.max);
      elevCPU = el.cpu; patchCPU = pel.cpu; cloudBundled = cl;
      U.uClimTex.value = bakeClimate(cl.image, elevCPU);
      for (const t of [col, el.tex, night, cl, pcol, pel.tex, jan, storm, moon, U.uClimTex.value]) R.initTexture(t);
      U.uTexOK.value = 1;
    });
    const elevAt = (lat, lon) => {
      if (!elevCPU) return null;
      const pl = (lon - PATCH.lon0 + 360) % 360;
      if (patchCPU && pl < PATCH.w / 60 && lat > PATCH.lat0 && lat < PATCH.lat0 + PATCH.h / 60)
        return 1000 * sampleElev(patchCPU, lat, PATCH.lon0 + pl, PATCH.lon0, PATCH.lat0, PATCH.w / 60, PATCH.h / 60);
      return 1000 * sampleElev(elevCPU, lat, lon);
    };
    const isLand = (lat, lon) => elevCPU ? sampleElev(elevCPU, lat, wrap180(lon), -180, -90, 360, 180, 1) < 0.5 : false;

    /* ---- the Earth clock ---------------------------------------------------------------------- */
    const clk = { T: null, lastSim: 0 };
    const SPIN0 = -30 * D2R;   // at T = 0 the camera's default azimuth looks down on ~50°E
    const spinOf = T => SPIN0 + 2 * Math.PI * T / SIDEREAL_H;
    const Y = new THREE.Vector3(0, 1, 0);
    const bodyDir = (lat, lon) => new THREE.Vector3(...llE(lat, lon));
    const toWorld = (v, rot) => v.clone().applyAxisAngle(Y, rot);
    const toBody = (v, rot) => v.clone().applyAxisAngle(Y, -rot);
    let lastRot = null;
    const st = { utc: 12, solarT: 12, sub: { lat: 0, lon: 0 }, storms: [], moon: null, decl: 18 };

    /* ---- storms (CPU, deterministic in T) ------------------------------------------------------ */
    function season(doy) {
      const nh = 0.15 + 0.85 * Math.exp(-(((doy - 245) / 62) ** 2));
      const dS = ((doy - 55 + 182.5) % 365 + 365) % 365 - 182.5;
      const sh = 0.12 + 0.88 * Math.exp(-((dS / 55) ** 2));
      const winterN = 0.5 + 0.5 * Math.cos(2 * Math.PI * (doy - 15) / 365);
      return { nh, sh, winterN };
    }
    function tcSlot(k, T, se) {
      const D = 190 + 60 * hash(k, 901), off = D * hash(k, 902);
      const u = (T + off) / D, n = Math.floor(u), age = (u - n) * D;
      const h = i => hash(k * 31 + i, n * 7 + 3);
      const B = TC_BASINS[k], basin = B[Math.floor(h(1) * B.length)];
      let act = basin.h > 0 ? se.nh : se.sh;
      if (k === 0) act = 0.35 + 0.65 * se.nh;            // the West Pacific has typhoons in every month
      if (h(2) > act) return null;
      const life = D * (0.72 + 0.22 * h(3));
      if (age > life) return null;
      let lat = basin.lat[0] + (basin.lat[1] - basin.lat[0]) * h(4), lon = basin.lon[0] + (basin.lon[1] - basin.lon[0]) * h(5);
      const latRec = h(6) < 0.72 ? 21 + 9 * h(7) : 99;   // most recurve into the westerlies
      let landH = 0;
      for (let t = 0; t < age; t += 3) {
        const s = Math.min(3, age - t), al = Math.abs(lat);
        const turn = sstep(latRec - 4, latRec + 7, al);
        const ue = -5.5 + 18 * turn + 0.8 * Math.sin(t * 0.07 + h(8) * 6);   // m/s east
        const vp = 1.6 + 2.8 * turn;                                          // m/s poleward (β drift, then steering)
        lon += ue * 3.6 * s / (111.2 * Math.cos(lat * D2R));
        lat += Math.sign(basin.h) * vp * 3.6 * s / 111.2;
        if (isLand(lat, lon)) landH += s;
      }
      const a = age / life;
      let I = smoothLife(a, 0, 0.28, 0.72, 1) * Math.exp(-landH / 14) * (1 - sstep(30, 42, Math.abs(lat)));
      I *= 0.8 + 0.2 * h(9);
      if (I < 0.02) return null;
      const stamp = STAMPS[[0, 1, 4][Math.floor(h(10) * 3)]];
      return {
        kind: 'tc', lat, lon: wrap180(lon), I, stamp, mirror: basin.h * stamp.hemi, half: stamp.half * (0.85 + 0.4 * h(11)),
        rot: h(12) * 6.283 + age * (2 * Math.PI / 15), wind: 0, basin: basin.name, age, stretch: -1,
      };
    }
    function etcSlot(k, T, se, decl) {
      const nh = k < 3, hemi = nh ? 1 : -1;
      const D = 104 + 40 * hash(k, 801), off = D * hash(k, 802);
      const u = (T + off) / D, n = Math.floor(u), a = u - n;
      const h = i => hash(k * 37 + i, n * 5 + 11);
      let lon0, lat0;
      if (nh) { lon0 = ETC_GEN_N[Math.floor(h(1) * ETC_GEN_N.length)] + (h(2) - 0.5) * 16; lat0 = 36 + 9 * h(3) + 0.3 * decl; }
      else { lon0 = 360 * h(1) - 180; lat0 = -(40 + 7 * h(3)) + 0.15 * decl; }
      const lat1 = lat0 + hemi * (9 + 8 * h(4));
      const speed = 10 + 6 * h(5);
      const latm = 0.5 * (lat0 + lat1);
      const dlon = speed * 3.6 * D / (111.2 * Math.cos(latm * D2R));
      const lat = lat0 + (lat1 - lat0) * a + 1.5 * Math.sin(a * 6 + h(6) * 6);
      const lon = lon0 + dlon * a;
      const strength = nh ? 0.5 + 0.5 * se.winterN : 0.8 + 0.2 * (1 - se.winterN);
      const I = smoothLife(a, 0, 0.22, 0.62, 1) * strength;
      if (I < 0.02) return null;
      const stamp = STAMPS[[2, 3, 5, 6][Math.floor(h(7) * 4)]];
      const S = sstep(0.05, 0.95, a);
      return {
        kind: 'etc', lat, lon: wrap180(lon), I, stamp, mirror: hemi * stamp.hemi, half: stamp.half * (0.72 + 0.32 * h(8)),
        rot: (h(9) - 0.5) * 1.0 + 0.6 * S, wind: -1.1 + 2.3 * S, age: a * D, stretch: 1.0 + 0.45 * h(10),
      };
    }
    function updateStorms(T, doy, decl, on) {
      const se = season(doy), list = [];
      if (on) {
        for (let k = 0; k < 4; k++) { const s = tcSlot(k, T, se); if (s) list.push(s); }
        for (let k = 0; k < 7; k++) { const s = etcSlot(k, T, se, decl); if (s) list.push(s); }
      }
      // two systems never sit on top of each other: the weaker one fades (it is absorbed)
      for (const a of list) for (const b of list) {
        if (a === b || a.I < b.I || (a.I === b.I && a.lat < b.lat)) continue;
        const cosd = new THREE.Vector3(...llE(a.lat, a.lon)).dot(new THREE.Vector3(...llE(b.lat, b.lon)));
        const dist = Math.acos(Math.min(1, cosd)) / D2R, need = 0.75 * (a.half + b.half);
        b.I *= sstep(0.45 * need, need, dist);
      }
      for (let j = 0; j < NST; j++) {
        const s = list[j];
        if (!s) { U.uStA.value[j].set(0, 0, 1, 0); continue; }
        const c = llE(s.lat, s.lon), hr = s.half * D2R;
        U.uStA.value[j].set(c[0], c[1], c[2], s.I);
        U.uStB.value[j].set(s.stamp.cell, s.mirror, s.rot, hr);
        U.uStC.value[j].set(s.stamp.hemi, s.wind, Math.cos(hr * 0.97 * Math.abs(s.stretch)), s.stretch);
      }
      st.storms = list;
    }

    /* ---- live clouds from NASA GIBS ------------------------------------------------------------- */
    const live = { state: 'off', tex: null, rt: null, date: '' };
    function fetchLive(c) {
      if (live.state !== 'off') return;
      live.state = 'loading';
      live.date = new Date(Date.now() - 864e5).toISOString().slice(0, 10);
      const url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0'
        + '&LAYERS=VIIRS_NOAA20_CorrectedReflectance_TrueColor&CRS=EPSG:4326&BBOX=-90,-180,90,180&WIDTH=4096&HEIGHT=2048'
        + `&FORMAT=image/jpeg&TIME=${live.date}`;
      c.toast?.(`Đang tải ảnh mây toàn cầu ngày ${live.date} từ NASA GIBS…`, 4000);
      const timer = setTimeout(() => { if (live.state === 'loading') { live.state = 'failed'; live.err = 'timeout'; c.toast?.('Không tải được ảnh NASA GIBS (hết thời gian chờ) — dùng ảnh mây đi kèm.'); } }, 45000);
      new THREE.TextureLoader().load(url, tex => {
        clearTimeout(timer);
        if (live.state !== 'loading') return;
        tex.colorSpace = THREE.NoColorSpace; tex.generateMipmaps = false; tex.minFilter = THREE.LinearFilter;
        live.tex = tex; live.state = 'extract';
      }, undefined, e => { clearTimeout(timer); live.state = 'failed'; live.err = 'error ' + (e && (e.message || e.type) || ''); c.toast?.('Không tải được ảnh NASA GIBS (ngoại tuyến?) — dùng ảnh mây đi kèm.'); });
    }
    function extractLive(c) {
      const rt = new THREE.WebGLRenderTarget(4096, 2048, { format: THREE.RedFormat, type: THREE.UnsignedByteType, depthBuffer: false,
        generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, wrapS: THREE.RepeatWrapping, wrapT: THREE.ClampToEdgeWrapping });
      rt.texture.anisotropy = 4;
      const mat = new THREE.ShaderMaterial({
        vertexShader: FULLSCREEN_VERT, fragmentShader: LIVE_FRAG, depthTest: false, depthWrite: false,
        uniforms: { uLive: { value: live.tex }, uColTex: { value: U.uColTex.value }, uJanTex: { value: U.uJanTex.value }, uElevTex: { value: U.uElevTex.value },
          uCloudTex: { value: cloudBundled }, uWinter: { value: U.uWinter.value }, uIceN: { value: U.uIceN.value }, uIceS: { value: U.uIceS.value } },
      });
      const scene = new THREE.Scene(), cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
      const prev = R.getRenderTarget();
      R.setRenderTarget(rt); R.render(scene, cam); R.setRenderTarget(prev);
      mat.dispose(); live.tex.dispose(); live.tex = null;
      live.rt = rt; live.state = 'ready';
      // the season follows the picture
      const ld = new Date(live.date + 'T12:00:00Z');
      c.params.date = Math.min(365, Math.round((ld - Date.UTC(ld.getUTCFullYear(), 0, 1)) / 864e5) + 1);
      c.toast?.(`Mây thật ngày <b>${live.date}</b>: ảnh ghép toàn cầu VIIRS (vệ tinh NOAA-20, mỗi nơi chụp lúc ~13:30 giờ địa phương), đã tách lớp mây khỏi bề mặt. Vùng đêm vùng cực và băng biển dùng ảnh mây đi kèm.`, 9000);
    }

    /* ---- camera modes: free / Himalaya close-up / geostationary / storm-following (co-rotating) */
    const mode = { kind: 'free', tw: null, sat: 'off', sunFixed: false, lastHour: null, follow: null, pending: null };
    const def = ctx.scene;
    const qs = new URLSearchParams(location.search);
    mode.pending = qs.get('earthShot');
    function frameOf(cam, ctl) { return { pos: cam.position.clone(), tgt: ctl.target.clone(), up: cam.up.clone() }; }
    function startTween(c, toBodyFrame, dur, onDone) {
      const rot = spinOf(clk.T ?? 0), from = frameOf(c.camera, c.controls);
      mode.tw = { t0: performance.now(), dur: dur * 1000, onDone, a: { pos: toBody(from.pos, rot), tgt: toBody(from.tgt, rot), up: toBody(from.up, rot) }, b: toBodyFrame };
    }
    function applyFrame(c, f, rot) {
      c.camera.position.copy(toWorld(f.pos, rot));
      c.controls.target.copy(toWorld(f.tgt, rot));
      c.camera.up.copy(toWorld(f.up, rot)).normalize();
      c.camera.lookAt(c.controls.target);
    }
    const SHOT = (qs.get('shotGeo') || '23.5,87.5,330,28.6,86.0').split(',').map(Number);
    function himalayaFrame() {
      const tgt = bodyDir(SHOT[3], SHOT[4]);
      const pos = bodyDir(SHOT[0], SHOT[1]).multiplyScalar(1 + SHOT[2] / R_KM);
      return { pos, tgt, up: pos.clone().normalize().add(tgt).normalize() };
    }
    function globalFrame() {
      const cam = def.camera, rot = spinOf(clk.T ?? 0);
      const pos = new THREE.Vector3().setFromSphericalCoords(cam.distance, cam.polar * D2R, cam.azimuth * D2R);
      return { pos: toBody(pos, rot), tgt: new THREE.Vector3(), up: toBody(Y, rot) };
    }
    const unFov = c => { if (mode.fovSet) { c.setFov?.(def.camera.fov); mode.fovSet = false; } };
    function toFree(c) {
      unFov(c);
      mode.kind = 'free'; mode.sunFixed = false; mode.follow = null;
      c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); c.controls.target.set(0, 0, 0);
      if (c.params.sat !== 'off') c.params.sat = 'off';
      mode.sat = 'off';
    }
    function goHimalaya(c, instant) {
      const p = c.params, keep = instant ? (qs.get('params') || '') : '';
      const set = (k, v) => { if (!keep.includes(k + ':')) p[k] = v; };
      set('exag', 3); set('hour', 7.4); set('date', 172); set('clouds', Math.min(p.clouds, 0.2));
      unFov(c);
      p.sat = 'off'; mode.sat = 'off'; mode.sunFixed = false; mode.follow = null;
      mode.kind = 'shot';
      c.controls.minDistance = 0.004;
      if (instant) { applyFrame(c, himalayaFrame(), spinOf(clk.T ?? 0)); mode.tw = null; }
      else startTween(c, himalayaFrame(), 4.5);
      c.toast?.(`Buổi sáng trên dãy Himalaya, nhìn từ độ cao ~330 km (như từ Trạm Vũ trụ Quốc tế) qua đồng bằng sông Hằng về phía cao nguyên Tây Tạng. <b>Everest cao 8 849 m — chỉ bằng 0,14 % bán kính Trái Đất</b>, nên địa hình được phóng đại <b>×${p.exag}</b> theo chiều đứng. Camera quay cùng Trái Đất; mây gió mùa trôi bên dưới các đỉnh tuyết.`, 11000);
    }
    function leaveShot(c) {
      c.params.exag = 8; c.params.hour = 13.5;
      startTween(c, globalFrame(), 3.5, () => toFree(c));
    }
    function goSat(c, key) {
      const s = SATS[key]; if (!s) return;
      mode.kind = 'corot'; mode.sat = key; mode.sunFixed = true; mode.lastHour = null; mode.follow = null; mode.tw = null;
      c.controls.minDistance = def.camera.min;
      const rot = spinOf(clk.T ?? 0);
      // (?earthGeoR=… brings the camera closer, for checking the clouds in snapshots)
      applyFrame(c, { pos: bodyDir(0, s.lon).multiplyScalar(+qs.get('earthGeoR') || R_GEO), tgt: new THREE.Vector3(), up: Y.clone() }, rot);
      c.setFov?.(+qs.get('earthGeoFov') || 19.5); mode.fovSet = true;                 // Earth spans 17.4° from GEO: frame it like the real imagers
      if (c.params.pace < 360) c.params.pace = 1200;
      c.toast?.(`Nhìn từ <b>${s.name}</b>, 35 786 km trên xích đạo: vệ tinh quay đúng một vòng mỗi ngày sao nên đứng yên so với mặt đất. Thời gian tua ×${c.params.pace}: theo dõi xoáy thuận trôi về phía đông, bão nhiệt đới đi về phía tây rồi quặt lên, mây đối lưu bùng lên mỗi chiều trên lục địa.`, 9000);
    }
    function goStorm(c, kind) {
      const list = st.storms.filter(s => !kind || s.kind === kind).sort((a, b) => (b.kind === 'tc') - (a.kind === 'tc') || b.I - a.I);
      if (!list.length) { c.toast?.('Hiện không có cơn bão nào (bật “Thời tiết chuyển động”).'); return; }
      const cur = mode.follow ? list.findIndex(s => s.stamp === mode.follow.stamp && Math.abs(s.lat - mode.follow.lat) < 3) : -1;
      const s = list[(cur + 1) % list.length];
      unFov(c);
      mode.kind = 'corot'; mode.sat = 'off'; c.params.sat = 'off'; mode.sunFixed = true; mode.lastHour = null;
      mode.follow = { stamp: s.stamp, lat: s.lat, lon: s.lon, kind: s.kind, dir: null };
      c.controls.minDistance = def.camera.min;
      const rot = spinOf(clk.T ?? 0), dist = s.kind === 'tc' ? 1.9 : 2.6;
      // look from a little equatorward so the sunlit hemisphere frames the storm
      const dirB = bodyDir(s.lat - Math.sign(s.lat) * 6, s.lon);
      startTween(c, { pos: dirB.multiplyScalar(dist), tgt: bodyDir(s.lat, s.lon).multiplyScalar(0.3), up: Y.clone() }, 3.5);
      if (c.params.pace < 360) c.params.pace = 1200;
      const sense = s.lat > 0 ? 'ngược chiều kim đồng hồ (Bắc bán cầu)' : 'thuận chiều kim đồng hồ (Nam bán cầu)';
      c.toast?.(s.kind === 'tc'
        ? `<b>Bão nhiệt đới</b> ở ${s.basin} (${latStr(s.lat)} ${lonStr(s.lon)}), xoáy ${sense}. Hình ảnh là cơn ${s.stamp.name} chụp bởi vệ tinh VIIRS. Bão đi về phía tây theo gió mậu dịch, trôi dần về phía cực (hiệu ứng β), rồi gặp gió tây ôn đới thì quặt về đông bắc; lên bờ hoặc gặp biển lạnh thì suy yếu.`
        : `<b>Xoáy thuận ngoài nhiệt đới</b> (${latStr(s.lat)} ${lonStr(s.lon)}), xoáy ${sense}: “dấu phẩy” mây gồm đầu mây dày quanh tâm thấp và frông lạnh kéo dài về phía xích đạo. Ảnh thật: ${s.stamp.name}. Hệ thống trôi về phía đông theo gió tây và cuộn chặt dần khi già đi (cố tù).`, 11000);
    }
    function goMoon(c) {
      const m = st.moon; if (!m) return;
      if (mode.kind !== 'free') toFree(c);
      const p = c.params, cam = c.camera.position;
      // Put the Moon 27° to the side of the Earth as seen from the camera, behind it
      const azCam = Math.atan2(cam.x, cam.z);
      const azMoon = azCam + Math.PI - 27 * D2R;
      const azSun = azMoon - m.dRA;
      p.hour = wrap24(12 - (azSun - azCam) * 12 / Math.PI);
      const d = Math.max(cam.length(), 3.2), pol = 90 + m.dec / D2R * 0.95;
      cam.setFromSphericalCoords(d, pol * D2R, azCam);
      c.camera.lookAt(c.controls.target);
      mode.lastHour = null;
      c.toast?.(`Mặt Trăng: cách ${Math.round(m.dist).toLocaleString('vi-VN')} km (${(m.dist / R_KM).toFixed(1)} lần bán kính Trái Đất), pha ${m.phaseName}, ${Math.round(m.illum * 100)} % được chiếu sáng. Đường kính chỉ bằng 27 % Trái Đất và suất phản chiếu 0,12 — tối như nhựa đường so với mây.`, 9000);
    }
    def._earth = {
      himalaya: c => { if (mode.tw) return; mode.kind === 'shot' ? leaveShot(c) : goHimalaya(c, false); },
      nextSat: c => { const keys = Object.keys(SATS), i = keys.indexOf(c.params.sat); c.params.sat = i + 1 < keys.length ? keys[i + 1] : 'off'; },
      storm: c => goStorm(c),
      moon: c => goMoon(c),
      reSun: () => { mode.lastHour = null; },
    };

    const tmpV = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
    const magB = new THREE.Vector3(...llE(80.7, -72.7)), mW = new THREE.Vector3(), anti = new THREE.Vector3();

    return {
      material,
      ready,
      exit(c) {
        if (mode.kind !== 'free') { mode.tw = null; toFree(c); }
      },
      update(dt, c) {
        const p = c.params, L = U.uSunDir.value;
        // ---- clock ----
        const sim = c.time.sim;
        if (clk.T === null) clk.T = sim * p.pace / 3600;
        else { const d = sim - clk.lastSim; if (Math.abs(d) < 600) clk.T += d * p.pace / 3600; }
        clk.lastSim = sim;
        const T = clk.T, rot = spinOf(T);

        // ---- camera modes ----
        if (mode.pending && U.uTexOK.value > 0) {
          const k = mode.pending; mode.pending = null;
          if (k === 'himalaya') goHimalaya(c, true);
          else if (k === 'storm' || k === 'etc') { goStorm(c, k === 'etc' ? 'etc' : null); if (mode.tw) { applyFrame(c, mode.tw.b, rot); mode.tw = null; } }
          else if (k === 'moon') goMoon(c);
        }
        if (p.sat !== mode.sat) { if (p.sat === 'off') toFree(c); else goSat(c, p.sat); }
        if (mode.tw) {
          const tw = mode.tw, k0 = Math.min(1, (performance.now() - tw.t0) / tw.dur), e = k0 * k0 * (3 - 2 * k0);
          const ra = tw.a.pos.length(), rb = tw.b.pos.length();
          const dir = tw.a.pos.clone().normalize().lerp(tw.b.pos.clone().normalize(), e);
          if (dir.lengthSq() < 1e-6) dir.copy(tw.b.pos).normalize();
          const r = Math.exp(Math.log(ra) * (1 - e) + Math.log(rb) * e) * (1 + 0.35 * Math.sin(Math.PI * e) * Math.min(1, Math.abs(ra - rb)));
          applyFrame(c, { pos: dir.normalize().multiplyScalar(r), tgt: tw.a.tgt.clone().lerp(tw.b.tgt, e), up: tw.a.up.clone().lerp(tw.b.up, e) }, rot);
          if (k0 >= 1) { mode.tw = null; tw.onDone?.(); }
        } else if (mode.kind !== 'free' && lastRot !== null && rot !== lastRot) {
          // co-rotate with the ground
          const d = rot - lastRot;
          c.camera.position.applyAxisAngle(Y, d); c.controls.target.applyAxisAngle(Y, d); c.camera.up.applyAxisAngle(Y, d);
          c.camera.lookAt(c.controls.target);
        }
        lastRot = rot;
        const top = 1 + (U.uHMax.value * p.exag + 12) / R_KM;
        if (c.camera.position.length() < top) c.camera.position.setLength(top);

        // ---- the Sun: date → declination; hour → direction relative to the camera, or fixed in
        //      space while the camera rides with the Earth (then the hour slider runs like a clock)
        const decl = declOf(p.date);
        st.decl = decl;
        // (?earthSweep: the hour slider means the local time at T = 0, so snapshots of a co-rotating
        //  view at different t show the terminator sweeping — a verification hook)
        if (mode.sunFixed && mode.lastHour === null && qs.has('earthSweep') && !mode.swept) { mode.swept = true; p.hour = wrap24(p.hour + T * 24 / SIDEREAL_H); }
        if (mode.sunFixed && mode.lastHour !== null && Math.abs(p.hour - mode.lastHour) < 1e-6) {
          const ad = decl * D2R, h = Math.hypot(L.x, L.z) || 1;
          L.set(L.x / h * Math.cos(ad), Math.sin(ad), L.z / h * Math.cos(ad));
          p.hour = localHourE(c.camera.position, L);
        } else sunFromHourE(c.camera.position, p.hour, decl, L);
        mode.lastHour = mode.sunFixed ? p.hour : null;

        // ---- solar clock: UTC from the subsolar longitude, made continuous with T ----
        const Lb = toBody(L, rot), subLon = Math.atan2(Lb.x, Lb.z) / D2R;
        st.sub = { lat: decl, lon: subLon };
        st.utc = wrap24(12 - subLon / 15);
        let dd = st.utc - wrap24(T); dd = ((dd + 12) % 24 + 24) % 24 - 12;
        st.solarT = T + dd;

        // ---- storms & weather ----
        const wxOn = !!p.wx;
        updateStorms(T, p.date, decl, wxOn && !(p.live && live.state === 'ready'));
        U.uWxT.value = T; U.uSolarT.value = st.solarT; U.uDecl.value = decl; U.uWx.value = wxOn ? 1 : 0;
        U.uConv.value = p.live && live.state === 'ready' ? 0.5 : 1;
        // follow a storm: keep its centre under the camera
        if (mode.follow && !mode.tw) {
          const f = mode.follow, s = st.storms.find(q => q.stamp === f.stamp && Math.abs(q.lat - f.lat) < 4 && Math.abs(wrap180(q.lon - f.lon)) < 6);
          if (s) {
            const now = toWorld(bodyDir(s.lat, s.lon), rot), prev = toWorld(bodyDir(f.lat, f.lon), rot);
            tmpQ.setFromUnitVectors(prev, now);
            c.camera.position.applyQuaternion(tmpQ); c.controls.target.applyQuaternion(tmpQ); c.camera.up.applyQuaternion(tmpQ);
            c.camera.lookAt(c.controls.target);
            f.lat = s.lat; f.lon = s.lon;
          }
        }
        // live clouds
        if (p.live) {
          if (live.state === 'off') fetchLive(c);
          else if (live.state === 'extract' && U.uTexOK.value > 0) extractLive(c);
        }
        if (U.uTexOK.value > 0) U.uCloudTex.value = p.live && live.state === 'ready' ? live.rt.texture : cloudBundled;

        // ---- season on the surface ----
        U.uWinter.value = Math.max(0, 0.5 + 0.5 * Math.cos(2 * Math.PI * (p.date - 15) / 365));
        const edgeN = 71 + 9 * Math.cos(2 * Math.PI * (p.date - 258) / 365);
        const edgeS = 66 - 7 * Math.cos(2 * Math.PI * (p.date - 245) / 365);
        U.uIceN.value = Math.sin(edgeN * D2R); U.uIceS.value = Math.sin(edgeS * D2R);

        // ---- the Moon ----
        const dJ = J2000_YEAR0 + p.date - 1 + st.utc / 24, so = solar(dJ), lu = lunar(dJ);
        const dRA = lu.ra - so.ra, azS = Math.atan2(L.x, L.z), azM = azS + dRA;
        const mDir = new THREE.Vector3(Math.cos(lu.dec) * Math.sin(azM), Math.sin(lu.dec), Math.cos(lu.dec) * Math.cos(azM));
        U.uMoonPos.value.copy(mDir).multiplyScalar(lu.dist / R_KM);
        U.uMoonF.value.copy(mDir).negate();
        U.uMoonN.value.copy(Y).addScaledVector(U.uMoonF.value, -U.uMoonF.value.y).normalize();
        U.uMoonE.value.crossVectors(U.uMoonN.value, U.uMoonF.value);
        const cosE = Math.sin(so.dec) * Math.sin(lu.dec) + Math.cos(so.dec) * Math.cos(lu.dec) * Math.cos(dRA);
        const illum = (1 - cosE) / 2, waxing = ((dRA % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) < Math.PI;
        const phaseName = illum < 0.03 ? 'trăng mới' : illum > 0.97 ? 'trăng tròn' : illum < 0.45 ? (waxing ? 'trăng lưỡi liềm đầu tháng' : 'trăng lưỡi liềm cuối tháng')
          : illum < 0.55 ? (waxing ? 'thượng huyền' : 'hạ huyền') : (waxing ? 'trăng khuyết đầu tháng' : 'trăng khuyết cuối tháng');
        st.moon = { dir: mDir, dist: lu.dist, dec: lu.dec, dRA, illum, phaseName };
        U.uMoon.value = p.moon ? 1 : 0;

        // ---- the rest ----
        U.uRot.value = -rot;
        U.uClouds.value = p.clouds; U.uCities.value = p.cities ? 1 : 0; U.uAtmo.value = p.atmo;
        U.uKp.value = p.kp; U.uAurora.value = p.aurora ? 1 : 0;
        U.uExag.value += (p.exag - U.uExag.value) * Math.min(1, dt * 4);
        if (Math.abs(p.exag - U.uExag.value) < 0.01) U.uExag.value = p.exag;
        mW.copy(magB).applyAxisAngle(Y, rot);
        anti.copy(L).addScaledVector(mW, -L.dot(mW)).normalize();
        U.uAurN.value.copy(mW).addScaledVector(anti, -0.07).normalize();
        U.uAurS.value.copy(mW).negate().addScaledVector(anti, -0.07).normalize();
      },
      hud(c) {
        const cp = c.camera.position, T = clk.T ?? 0, ll2 = bodyLatLonE(cp, spinOf(T));
        const e = elevAt(ll2.lat, ll2.lon);
        const nTC = st.storms.filter(s => s.kind === 'tc').length, nE = st.storms.length - nTC;
        const rows = [
          ...orbitRows(c, R_KM),
          ['Điểm dưới camera', `${latStr(ll2.lat)} ${lonStr(ll2.lon)}` + (e === null ? '' : e >= 0 ? ` · cao ${Math.round(e).toLocaleString('vi-VN')} m` : ` · biển sâu ${Math.round(-e).toLocaleString('vi-VN')} m`)],
          ['Giờ Mặt Trời tại đó', hhmm(c.params.hour)],
          ['Giờ quốc tế (≈ UTC) · ngày', `${hhmm(st.utc)} · ${dateStr(c.params.date)}/${YEAR}`],
          ['Mặt Trời đứng bóng tại', `${latStr(st.sub.lat)} ${lonStr(wrap180(st.sub.lon))}`],
          ['Thời gian Trái Đất đã trôi', c.fmt.time(T * 3600)],
          ['Bão nhiệt đới / xoáy thuận ôn đới', c.params.wx ? `${nTC} / ${nE}` : 'tắt'],
        ];
        if (st.moon) rows.push(['Mặt Trăng', `${st.moon.phaseName}, ${Math.round(st.moon.illum * 100)} % sáng · ${Math.round(st.moon.dist).toLocaleString('vi-VN')} km`]);
        if (qs.has('earthInfo')) rows.push(['storms', st.storms.map(s => `${s.kind}:${s.stamp.cell} ${s.lat.toFixed(1)},${s.lon.toFixed(1)} I${s.I.toFixed(2)} h${s.half.toFixed(1)} a${s.age.toFixed(0)}`).join(' | ')]);
        if (c.params.live) rows.push(['Mây thật NASA GIBS', live.state === 'ready' ? `ngày ${live.date}` : live.state === 'failed' ? `không tải được (${live.err}) — dùng ảnh đi kèm` : 'đang tải…']);
        return rows;
      },
      onClick(info, c) {
        const t = hitSphere(info.origin, info.dir);
        if (t === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, t).normalize(), L = U.uSunDir.value;
        const g = bodyLatLonE(p, spinOf(clk.T ?? 0));
        const e = elevAt(g.lat, g.lon);
        const hs = e === null ? '' : e >= 0 ? ` · độ cao <b>${Math.round(e).toLocaleString('vi-VN')} m</b>` : ` · độ sâu đáy biển <b>${Math.round(-e).toLocaleString('vi-VN')} m</b>`;
        // a storm under the cursor?
        const pb = new THREE.Vector3(...llE(g.lat, g.lon));
        const s = st.storms.find(q => pb.dot(new THREE.Vector3(...llE(q.lat, q.lon))) > Math.cos(q.half * 0.5 * D2R) && q.I > 0.25);
        const sm = s ? (s.kind === 'tc' ? ` · <b>bão nhiệt đới</b> (${s.basin}), ảnh thật của ${s.stamp.name}` : ` · <b>xoáy thuận ôn đới</b>, ảnh thật của ${s.stamp.name}`) : '';
        c.toast(`Vĩ độ <b>${latStr(g.lat)}</b>, kinh độ <b>${lonStr(g.lon)}</b>${hs} · giờ địa phương <b>${hhmm(localHourE(p, L))}</b> · ${sunElevStr(p, L)}${sm}.`);
      },
    };
  },
}];
