// Sao Kim (Venus) — built from real spacecraft data (assets/venus/, see CREDITS.txt):
//   * Magellan SAR global mosaic (C3-MIDR, 2 km) + a 450 m left-look patch of Maat Mons / Ozza Mons
//   * Magellan altimetry (GTDR v02, 4.6 km) for the relief
//   * Akatsuki UVI 365 nm cloud-top map: 26 low-phase images (14–19 Aug 2017) composited in the
//     frame that follows the super-rotation (zonal winds measured by cloud tracking on those images)
//   * Akatsuki IR2 2.26 µm night-side map: 27 images (15–30 Aug 2016) composited the same way, with
//     the slower lower-cloud winds (also measured from the images)
// One scene unit = one mean planetary radius (6 051,8 km).
//
// Frames. Body (surface) coordinates use east longitude λ = atan2(x, z) (IAU: positive east, north
// pole on the north side of the ecliptic). Venus spins retrograde, so the surface drifts toward −λ:
// body = rotY(−spin)·world with spin(t) = −Ω⊙·t. The cloud tops super-rotate in the same (westward)
// sense ~60× faster: a cloud parcel's body longitude decreases at ω(φ) = u(φ)/(R cos φ).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';
import { t, locale } from '../../i18n.js';

const D2R = Math.PI / 180;
const RKM = 6051.8;
const TAU = 2 * Math.PI;
// ---- dynamics (hours) ---------------------------------------------------------------------------
const W_TOP = 102.9 * 3.6 / (RKM + 70);        // rad/h: 102.9 m/s at 70 km (measured on the UVI images) → 4,33 d
const W_LOW = 76.0 * 3.6 / (RKM + 50);         // rad/h: 76 m/s at ~50 km (measured on the IR2 images) → 5,8 d
const W_VORTEX = TAU / (2.5 * 24);             // polar vortex: one turn in ~2,5 days (Venus Express VIRTIS)
const W_SURF = TAU / (243.0226 * 24);          // sidereal rotation, retrograde
const PH = 60;                                 // flow-map phase length (hours of Venus time)
// cloud-top angular velocity relative to the equator (fit to the Akatsuki 2017 profile, with the faster
// polar vortex core from Venus Express) — must match ratioTop() in the shader
function ratioTop(phi) {
  const s2 = Math.sin(phi) ** 2, x = Math.min(Math.max((Math.abs(phi) / D2R - 62) / 20, 0), 1);
  return 1 - 0.25 * s2 - 0.3 * s2 ** 4 + 1.1 * x * x * (3 - 2 * x);
}
function ratioLow(phi) {
  const x = Math.min(Math.max((Math.abs(phi) / D2R - 48) / 22, 0), 1);
  return 1 - 0.7 * x * x * (3 - 2 * x);
}
// Maat Mons radar patch (Magellan left-look 225 m mosaic, averaged to 450 m), east-longitude bounds
const MAAT = { lon0: -171.49937, lon1: -159.50050, lat0: -3.49947, lat1: 5.49962 };
// UV / IR map encodings (see CREDITS.txt): value = lo + (hi − lo)·byte/255
const UV_LO = 0.78, UV_HI = 1.56, IR_LO = -0.75, IR_HI = 0.85;

const llB = (lat, lon) => new THREE.Vector3(Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R));

/* ================================================================================================
   Shader
   ================================================================================================ */
const FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir;
uniform vec4 uFlow;      // x: solid-body drift of the cloud tops, y: of the lower clouds, z: flow phase (0..1), w: vortex angle
uniform float uView;     // 0 visible, 1 UV 365 nm, 2 near-IR 2.26 µm, 3 thermal IR 10 µm, 4 radar
uniform float uRot, uTH, uAtmo, uContrast, uWave, uExag, uTint, uTexOK, uHMin, uHMax, uTexel, uRadTexel;
uniform sampler2D uUV, uIR, uRadar, uMaat, uTopo;
const float RKM = 6051.8;
const float RC = 1.0 + 70.0 / RKM;          // cloud tops (unit optical depth at 365 nm ≈ 70 km)
const float RTOP = RC + 35.0 / RKM;         // top of the upper haze for the scattering integral
const float ESUN = 3.0;                     // solar irradiance in scene radiance units
const float PH = 60.0;                      // flow-map phase length, hours
const float W_TOP = ${W_TOP.toFixed(6)};    // rad/h, equatorial cloud-top super-rotation
const float W_LOW = ${W_LOW.toFixed(6)};    // rad/h, lower clouds (night-side 2.3 µm silhouettes)
const float V_TOP = ${(7 * 3.6 / (RKM + 70)).toFixed(6)};   // rad/h: 7 m/s poleward Hadley flow (peak, ∝ sin 2φ)
const float V_LOW = ${(-2 * 3.6 / (RKM + 50)).toFixed(6)};  // rad/h: weak equatorward return flow lower down
const vec2 MP0 = vec2(${((MAAT.lon0 + 180) / 360).toFixed(7)}, ${((MAAT.lat0 + 90) / 180).toFixed(7)});
const vec2 MPS = vec2(${((MAAT.lon1 - MAAT.lon0) / 360).toFixed(7)}, ${((MAAT.lat1 - MAAT.lat0) / 180).toFixed(7)});
const float UV_LO = ${UV_LO.toFixed(3)}, UV_HI = ${UV_HI.toFixed(3)}, IR_LO = ${IR_LO.toFixed(3)}, IR_HI = ${IR_HI.toFixed(3)};

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
float wrapPi(float a){ return a - TAU * floor((a + PI) / TAU); }

// ---- single scattering above the cloud deck (Chapman grazing-incidence optical depths) ---------
float chapman(float X, float h, float mu){
  float c = sqrt(1.5708 * (X + h));
  if (mu >= 0.0) return c / (c * mu + 1.0) * exp(-h);
  float x0 = sqrt(max(1.0 - mu * mu, 0.0)) * (X + h);
  float c0 = sqrt(1.5708 * x0);
  return 2.0 * c0 * exp(min(X - x0, 60.0)) - c / (1.0 - c * mu) * exp(-h);
}
float phaseR(float mu){ return 0.0596831 * (1.0 + mu * mu); }
float phaseHG(float mu, float g){ float g2 = g * g; return 0.0795775 * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5); }
struct Atm { float R; float Hr; float Hm; vec3 bR; vec3 bMs; vec3 bMe; float g; };
vec3 sunTrans(Atm A, float h, float mu){
  return exp(-(A.bR * A.Hr * chapman(A.R / A.Hr, h / A.Hr, mu) + A.bMe * A.Hm * chapman(A.R / A.Hm, h / A.Hm, mu)));
}
vec3 atmosphere(vec3 ro, vec3 rd, float t0, float t1, vec3 L, Atm A, out vec3 T){
  const int N = 12;
  float dt = (t1 - t0) / float(N);
  vec3 sR = vec3(0.0), sM = vec3(0.0);
  float oR = 0.0, oM = 0.0, XR = A.R / A.Hr, XM = A.R / A.Hm;
  for (int i = 0; i < N; i++){
    vec3 p = ro + rd * (t0 + (float(i) + 0.5) * dt);
    float r = length(p), h = max(r - A.R, 0.0);
    float dR = exp(-h / A.Hr) * dt, dM = exp(-h / A.Hm) * dt;
    oR += 0.5 * dR; oM += 0.5 * dM;
    float mu = dot(p, L) / r;
    vec3 att = exp(-(A.bR * (oR + A.Hr * chapman(XR, h / A.Hr, mu)) + A.bMe * (oM + A.Hm * chapman(XM, h / A.Hm, mu))));
    sR += dR * att; sM += dM * att;
    oR += 0.5 * dR; oM += 0.5 * dM;
  }
  T = exp(-(A.bR * oR + A.bMe * oM));
  float nu = dot(rd, L);
  // haze: two-term HG (strong forward peak of the ~1 µm H₂SO₄ droplets + a weak backscatter lobe)
  return sR * A.bR * phaseR(nu) + sM * A.bMs * (0.9 * phaseHG(nu, 0.72) + 0.1 * phaseHG(nu, -0.3));
}
vec3 sunDisk(vec3 rd, vec3 L){
  float c = dot(rd, L), a = acos(clamp(c, -1.0, 1.0)), r = 0.00465 / 0.723;
  return bb(5772.0) * (smoothstep(r * 1.15, r * 0.85, a) * 400.0 + pow(max(c, 0.0), 3000.0 * 0.523) * 4.0);
}
Atm atmV(bool uv){
  Atm A;
  A.R = RC; A.Hr = 4.9 / RKM; A.Hm = 4.5 / RKM;
  // CO₂ at the cloud tops: ~35 mbar, 230 K → n ≈ 1,1·10²⁴ m⁻³ (0,043× Earth sea level) × 2,4 (CO₂ vs air
  // cross-section) ≈ 0,10× Earth's sea-level Rayleigh coefficients, per scene unit of 6 051,8 km
  vec3 br = vec3(5.8, 13.5, 33.1) * 0.104 * 6.0518;
  if (uv) br = vec3(69.6 * 0.104 * 6.0518);                 // λ⁻⁴ at 365 nm
  A.bR = br * uAtmo;
  // upper haze (70–90 km, sub-µm H₂SO₄): vertical τ ≈ 0,1 in the visible, ≈ 0,2 at 365 nm (VMC/SPICAV)
  float tau = uv ? 0.2 : 0.1;
  A.bMe = vec3(tau / A.Hm) * uAtmo;
  A.bMs = A.bMe * (uv ? vec3(0.93) : vec3(1.0, 0.995, 0.97));
  A.g = 0.72;
  return A;
}

// ---- data access ---------------------------------------------------------------------------
vec2 uvOf(vec3 p){ return vec2(atan(p.x, p.z) * (0.5 / PI) + 0.5, asin(clamp(p.y, -1.0, 1.0)) / PI + 0.5); }
vec2 uvD(vec3 p, vec3 dp){
  float q = max(p.x * p.x + p.z * p.z, 1e-6);
  return vec2((p.z * dp.x - p.x * dp.z) / q * (0.5 / PI), dp.y / (PI * sqrt(q)));
}
float hgt(vec3 pb, float lod){ return textureLod(uTopo, uvOf(pb), max(lod, 0.0)).r; }   // km above 6 051,8 km

// ---- winds -----------------------------------------------------------------------------------
float ratioTop(float phi){
  float s2 = sin(phi); s2 *= s2;
  float x = clamp((abs(phi) * 57.29578 - 62.0) / 20.0, 0.0, 1.0);
  float s8 = s2 * s2; s8 *= s8;
  return 1.0 - 0.25 * s2 - 0.3 * s8 + 1.1 * x * x * (3.0 - 2.0 * x);
}
float ratioLow(float phi){ float x = clamp((abs(phi) * 57.29578 - 48.0) / 22.0, 0.0, 1.0); return 1.0 - 0.7 * x * x * (3.0 - 2.0 * x); }
// Texture coordinates of the cloud that is at body (λ, φ) now. The bulk of the drift is the exact
// solid-body super-rotation (solid = W·t); the latitudinal shear (ratio − 1)·W and the meridional
// Hadley flow act over a bounded phase time τ ∈ [−PH/2, PH/2] — two such phases are cross-faded
// (flow-map trick), so features shear, stretch and drift poleward without the map winding up forever.
vec2 flowUV(float lam, float phi, float tau, float solid, float W, float ratio, float vm){
  float lt = lam + solid + (ratio - 1.0) * W * tau;
  float pt = phi - vm * sin(2.0 * phi) * tau;
  return vec2(lt * (0.5 / PI) + 0.5, pt / PI + 0.5);
}
// small-scale cloud texture below the map resolution (~19 km): convective cells at low latitudes,
// streaks stretched along the flow at mid-latitudes
float cloudDetail(vec2 uvt, float fade){
  if (fade <= 0.0) return 0.0;
  float lt = (uvt.x - 0.5) * TAU, pt = (uvt.y - 0.5) * PI;
  vec3 q = vec3(cos(pt) * sin(lt), sin(pt), cos(pt) * cos(lt)) + uTH * 0.003;
  float st = smoothstep(0.3, 0.75, abs(pt));
  // streaks ~5:1 along the flow, slightly tilted poleward (spiral) — cells ~200–300 km near the equator
  float tilt = 0.25 * sign(pt) * (q.x + q.z);
  float streak = fbm(vec3(q.x * 6.0, (q.y + tilt) * 30.0, q.z * 6.0), 3);
  float cells = fbm(q * 26.0, 3);
  return mix(cells, streak, st) * fade;
}

// ---- polar vortices: x = warm / thin-cloud core (0..1), y = cold collar (0..1) -------------------
// Dipole ("S"-shaped) core rotating westward every ~2,5 days, its centre wandering ~3° around the
// pole and its shape switching between dipole, elongated and monopole forms within a day or two
// (Venus Express VIRTIS: Piccioni et al. 2007, Luz et al. 2011, Garate-Lopez et al. 2013).
vec2 vortex(vec3 pb){
  float hemi = pb.y >= 0.0 ? 1.0 : -1.0;
  float colat = acos(clamp(abs(pb.y), 0.0, 1.0));
  float collar = exp(-pow((colat - 0.42) / 0.09, 2.0));        // ~66° latitude, ~10 K colder
  if (colat > 0.45) return vec2(0.0, collar);
  vec2 q = colat * vec2(pb.z, pb.x) / max(length(pb.xz), 1e-5);
  float hs = hemi > 0.0 ? 0.0 : 1.0;
  float t = uTH;
  float pc = (t + hs * 61.0) * TAU / 120.0;
  vec2 d = q - 0.05 * vec2(cos(pc), sin(pc));
  float r = length(d);
  float th = atan(d.y, d.x) + uFlow.w + hs * 2.1;
  float m1 = 0.5 + 0.5 * sin((t + hs * 23.0) * TAU / 41.0);
  float m2 = 0.5 + 0.5 * sin((t + hs * 11.0) * TAU / 67.0 + 1.3);
  // the two warm 'eyes' are ~1 500–2 500 km apart (Piccioni et al. 2007)
  float a = mix(0.06, 0.12, m1);                                // half-separation of the two warm lobes (rad)
  float sp = th - 4.5 * r;                                      // S-shaped arms
  float lobes = exp(-pow((r - a) / (0.05 + 0.03 * m2), 2.0)) * pow(0.5 + 0.5 * cos(2.0 * sp), 2.0 + 2.0 * m2);
  float core = exp(-pow(r / 0.07, 2.0)) * (1.0 - m1) * 0.85;
  float arms = exp(-pow((r - 0.22) / 0.07, 2.0)) * pow(0.5 + 0.5 * cos(2.0 * sp - 0.9), 5.0) * 0.5;
  float fil = 0.8 + 0.2 * snoise(vec3(d * 25.0, t * 0.03 + hs * 7.0));
  float warm = clamp((max(lobes, core) + arms) * fil, 0.0, 1.0) * smoothstep(0.45, 0.3, colat);
  return vec2(warm, collar);
}

// ---- Aphrodite Terra stationary gravity wave (Akatsuki LIR, Dec 2015: Fukuhara et al. 2017) ------
// A bow ~10 000 km from north to south, fixed above the mountains (~90°E) while the 100 m/s wind blows
// through it; ±~1,5 K at the cloud tops, seen only when the highland is in the afternoon, strongest
// toward the evening terminator, absent at night. Returns the brightness-temperature anomaly (K).
float bowWave(float lam, float phi, float LT){
  float x = wrapPi(lam - 1.5708) * 57.29578, y = phi * 57.29578;
  float s = x - 0.011 * y * y;                                  // arms bend east with latitude
  float env = exp(-pow(y / 42.0, 4.0)) * exp(-pow(s / 13.0, 2.0));
  float day = smoothstep(12.5, 15.0, LT) * (1.0 - smoothstep(18.2, 19.0, LT));
  return 1.6 * env * sin(s * TAU / 15.0 + 0.9) * day;
}

// ---- Magellan radar ---------------------------------------------------------------------------
// Muhleman's empirical backscatter law σ₀(θ) for Venus
float muhleman(float c){ float s = sqrt(max(1.0 - c * c, 0.0)); return 0.0118 * c / pow(s + 0.111 * c, 3.0); }
float radarDN(vec2 uv, vec2 gx, vec2 gy){
  float v = textureGrad(uRadar, uv, gx, gy).r;
  vec2 q = (uv - MP0) / MPS;
  vec2 m = min(q, 1.0 - q);
  float w = smoothstep(0.0, 0.05, min(m.x, m.y));
  if (w > 0.0) v = mix(v, textureGrad(uMaat, q, gx / MPS, gy / MPS).r, w);
  return v;
}
// first crossing of the ray with the exaggerated relief r = 1 + k·e inside the shell
float marchRelief(vec3 ro, vec3 rd, float tA, float tB, bool hitBottom, mat3 R, float k, float pixAng){
  vec3 pm = normalize(ro + rd * (0.5 * (tA + tB)));
  float sinT = max(length(cross(rd, pm)), 0.02);
  float lat = (tB - tA) * sinT;
  float n = clamp(ceil(lat / max(1.6 * uTexel, tA * pixAng)), 2.0, 64.0);
  float dt = (tB - tA) / n;
  float lodS = log2(max(dt * sinT, 1e-7) / uTexel) - 0.5;
  float tP = tA, dP = 1.0;
  for (int i = 0; i <= 64; i++){
    if (float(i) > n) break;
    float t = tA + dt * float(i);
    vec3 q = ro + rd * t; float r = length(q);
    float lod = max(lodS, log2(t * pixAng / uTexel));
    float d = r - 1.0 - k * hgt(R * (q / r), lod);
    if (d <= 0.0 || (hitBottom && float(i) >= n)){
      if (i == 0) return t;
      for (int j = 0; j < 2; j++){
        float tc = tP + (t - tP) * dP / max(dP - d, 1e-9);
        vec3 qc = ro + rd * tc; float rc = length(qc);
        float dc = rc - 1.0 - k * hgt(R * (qc / rc), lod);
        if (dc <= 0.0){ t = tc; d = dc; } else { tP = tc; dP = dc; }
      }
      return tP + (t - tP) * dP / max(dP - d, 1e-9);
    }
    tP = t; dP = d;
  }
  return -1.0;
}
vec3 hypso(float e){
  // colour ramp of the USGS colourised Magellan maps: lowland plains blue → green → tan → red-brown → white peaks
  vec3 c = mix(vec3(0.08, 0.16, 0.48), vec3(0.12, 0.42, 0.72), smoothstep(-2.0, -0.6, e));
  c = mix(c, vec3(0.22, 0.62, 0.36), smoothstep(-0.6, 0.6, e));
  c = mix(c, vec3(0.82, 0.74, 0.32), smoothstep(0.6, 2.2, e));
  c = mix(c, vec3(0.70, 0.30, 0.14), smoothstep(2.2, 4.5, e));
  return mix(c, vec3(0.95, 0.92, 0.88), smoothstep(5.0, 9.0, e));
}
vec3 radarView(vec3 ro, vec3 rd, mat3 R, float pixAng, vec3 dpx, vec3 dpy, out bool hit){
  hit = false;
  float k = uExag / RKM * uTexOK;
  float rTop = 1.0 + k * uHMax + 1e-5, rBot = 1.0 + k * uHMin - 1e-5;
  vec2 hO = sphereHit(ro, rd, vec3(0.0), rTop);
  if (hO.x > hO.y || hO.y < 0.0) return vec3(0.0);
  vec2 hb = sphereHit(ro, rd, vec3(0.0), rBot);
  bool hitBottom = hb.x < hb.y && hb.x > 0.0;
  float tA = max(hO.x, 0.0), tB = hitBottom ? hb.x : hO.y;
  float tG = marchRelief(ro, rd, tA, tB, hitBottom, R, k, pixAng);
  if (tG < 0.0) return vec3(0.0);
  hit = true;
  vec3 pw = ro + rd * tG, n0 = normalize(pw), pb = R * n0;
  vec2 uv = uvOf(pb);
  float fp = tG * pixAng;
  float ds = max(fp, uTexel), lod = log2(ds / uTexel);
  float cphi = max(length(pb.xz), 0.02);
  vec3 east = vec3(pb.z, 0.0, -pb.x) / cphi, north = cross(pb, east);
  float e = hgt(pb, lod);
  float hE = hgt(normalize(pb + east * ds), lod), hW = hgt(normalize(pb - east * ds), lod);
  float hN = hgt(normalize(pb + north * ds), lod), hS = hgt(normalize(pb - north * ds), lod);
  vec2 g = vec2(hE - hW, hN - hS) * k / (2.0 * ds);
  vec3 n = transpose(R) * normalize(pb - g.x * east - g.y * north);
  // Magellan (cycles 1/3, left-looking from a north–south orbit) illuminated the surface from the west
  // at 15–45° incidence; the mosaic already carries that slope signal, so the relief here is lit the
  // same way and only the *change* of backscatter with the exaggerated slope is applied
  vec3 eW = transpose(R) * east;
  vec3 Lr = normalize(n0 * 0.82 - eW * 0.57);                  // ~35° incidence from the west
  float s = muhleman(clamp(dot(n, Lr), 0.02, 1.0)) / muhleman(dot(n0, Lr));
  float dn = radarDN(uv, uvD(pb, dpx), uvD(pb, dpy));
  // C3-MIDR DN is ~logarithmic in σ₀ (0,2 dB per DN step): back to a display-linear brightness
  float b = pow(dn, 2.2) * 1.5 * clamp(pow(s, 0.55), 0.05, 4.0);
  vec3 c = uTint < 0.5 ? vec3(b) : uTint < 1.5 ? vec3(1.0, 0.56, 0.24) * b * 1.15 : hypso(e) * (0.25 + 1.1 * b);
  return c;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  mat3 R = rotY(uRot);
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  // seam-free derivatives from the cloud-top sphere (or the closest approach for rays that miss)
  vec2 hs = sphereHit(ro, rd, vec3(0.0), RC);
  bool deck = hs.x < hs.y && hs.x > 0.0;
  float tS = deck ? hs.x : max(-dot(ro, rd), 0.0);
  vec3 pS = R * normalize(ro + rd * tS);
  vec3 dpx = dFdx(pS), dpy = dFdy(pS);
  vec3 col = vec3(0.0);

  // ======================= radar: the clouds are transparent at 12,6 cm ======================
  if (uView > 3.5){
    bool hit;
    vec3 c = radarView(ro, rd, R, pixAng, dpx, dpy, hit);
    gl_FragColor = vec4(hit ? c : sky(rd, 1.0) * 0.6, 1.0);
    return;
  }

  // ======================= cloud deck =========================================================
  bool uvMode = uView > 0.5 && uView < 1.5;
  vec3 bg = col;
  if (!deck) bg = sky(rd, 1.0) * 0.6 + sunDisk(rd, L) * 0.35;      // (the sky is skipped behind the opaque deck)
  if (deck){
    vec3 pw = ro + rd * hs.x, n0 = normalize(pw), pb = R * n0;
    float lam = atan(pb.x, pb.z), phi = asin(clamp(pb.y, -1.0, 1.0));
    vec2 gx = uvD(pb, dpx), gy = uvD(pb, dpy);
    float fA = uFlow.z, fB = fract(uFlow.z + 0.5), wB = abs(2.0 * fA - 1.0);
    float tauA = (fA - 0.5) * PH, tauB = (fB - 0.5) * PH;
    float fp = hs.x * pixAng * RKM;                              // pixel footprint, km
    float fade = smoothstep(25.0, 4.0, fp);                      // sub-texel detail only when resolved
    float mu0 = dot(n0, L), mu = max(dot(n0, -rd), 0.03);
    // local solar time at this point (afternoon is west of the subsolar meridian on retrograde Venus)
    float LT = mod(12.0 + (atan(L.x, L.z) - atan(n0.x, n0.z)) * 3.819719, 24.0);
    vec2 vx = vortex(pb);
    if (uView < 1.5){
      // ---- reflected sunlight: visible / UV 365 nm ----
      float rt = ratioTop(phi);
      vec2 a = flowUV(lam, phi, tauA, uFlow.x, W_TOP, rt, V_TOP), b = flowUV(lam, phi, tauB, uFlow.x, W_TOP, rt, V_TOP);
      // the composite map is ~20 km/texel and blurred by the 5-day blend: finer structure fades in up close
      float amp = 0.012 + 0.07 * fade;
      float va = textureGrad(uUV, a, gx, gy).r + amp * cloudDetail(a, 1.0);
      float vb = textureGrad(uUV, b, gx, gy).r + amp * cloudDetail(b, 1.0);
      float rel = (UV_LO + (UV_HI - UV_LO) * mix(va, vb, wB)) / 1.04;      // relative 365 nm albedo, mean ≈ 1
      rel *= 1.0 - 0.08 * vx.x;                                 // the vortex eye is slightly darker in the UV
      // true contrast ±30 %; displayed with the ×3 stretch typical of published UVI images (× the contrast slider)
      float A365 = clamp(0.47 * (1.0 + (rel - 1.0) * 3.0 * uContrast), 0.03, 1.0);
      vec3 alb;
      // (a monochrome UV camera: exposed lower, as the mission images are, so the stretch sits on the
      // linear part of the display curve)
      if (uvMode) alb = vec3(0.86, 0.9, 1.0) * A365 * 0.8;
      // visible: the unknown UV absorber leaves only a faint (few %) yellowish trace in the blue
      else alb = vec3(0.95, 0.87, 0.68) * (1.0 - vec3(0.01, 0.025, 0.08) * (1.0 - rel) * uContrast);
      // Minnaert photometry of the deep, conservatively scattering deck (k ≈ 0,85 fits the UVI images)
      float m0 = max(mu0, 0.0);
      float f = pow(m0, 0.85) * pow(mu, -0.15);
      Atm A = atmV(uvMode);
      vec3 Ts = sunTrans(A, 0.0, mu0);
      // diffuse skylight from the haze keeps the twilight band glowing past the geometric terminator
      float twi = 0.035 * smoothstep(-0.12, 0.08, mu0) * (1.0 - smoothstep(0.08, 0.4, mu0)) * uAtmo;
      bg = alb / PI * ESUN * (f * Ts + twi);
    } else if (uView < 2.5){
      // ---- near-IR 2.26 µm: the hot deep atmosphere (~35 km, 500 K) shining through the clouds ----
      float rl = ratioLow(phi);
      vec2 a = flowUV(lam, phi, tauA, uFlow.y, W_LOW, rl, V_LOW), b = flowUV(lam, phi, tauB, uFlow.y, W_LOW, rl, V_LOW);
      float va = textureGrad(uIR, a, gx, gy).r + 0.05 * cloudDetail(a * vec2(1.0, 1.0) + 0.37, fade);
      float vb = textureGrad(uIR, b, gx, gy).r + 0.05 * cloudDetail(b + 0.37, fade);
      float lnI = (IR_LO + (IR_HI - IR_LO) * mix(va, vb, wB)) * 2.6 * uContrast;   // ×2,6 display stretch
      // thick cold collar clouds are opaque, the vortex core is thinner
      lnI += 0.9 * vx.x - 1.1 * vx.y;
      // limb darkening: longer slant path through the cloud (fit to the IR2 images: d ln I / d(1/μ) ≈ −0,35)
      lnI -= 0.35 * (1.0 / mu - 1.0);
      float E = 0.2 * exp(lnI);
      // false colour like the published IR2 / VIRTIS night-side images
      vec3 heat = mix(vec3(0.45, 0.04, 0.01), vec3(1.0, 0.42, 0.06), smoothstep(0.04, 0.4, E));
      heat = mix(heat, vec3(1.0, 0.85, 0.55), smoothstep(0.35, 1.0, E));
      vec3 night = heat * E;
      // the sunlit crescent is ~10³× brighter at 2,3 µm and saturates the exposure
      float m0 = max(mu0 + 0.02, 0.0);
      vec3 day = vec3(1.0, 0.9, 0.75) * 2.2 * smoothstep(0.0, 0.08, m0);
      bg = night + day;
    } else {
      // ---- thermal IR 10 µm (Akatsuki LIR): cloud-top brightness temperature, day and night alike ----
      float rt = ratioTop(phi);
      vec2 a = flowUV(lam, phi, tauA, uFlow.x, W_TOP, rt, V_TOP), b = flowUV(lam, phi, tauB, uFlow.x, W_TOP, rt, V_TOP);
      float n = mix(cloudDetail(a + 0.61, 1.0), cloudDetail(b + 0.61, 1.0), wB);
      float ap = abs(phi);
      float T = 231.0 - 3.0 * smoothstep(0.4, 0.9, ap);
      T -= 8.0 * vx.y;                                           // cold collar (~222 K)
      T += 15.0 * vx.x + 3.0 * smoothstep(1.25, 1.5, ap);        // warm vortex (~240 K)
      // thermal tides locked to the Sun: semidiurnal ±1,5 K plus diurnal (Kouyama et al. 2019)
      T += (1.5 * cos((LT - 14.0) * PI / 6.0) + 0.8 * cos((LT - 15.0) * PI / 12.0)) * exp(-pow(phi / 0.75, 2.0));
      T += 0.4 * n;
      if (uWave > 0.5) T += bowWave(lam, phi, LT);
      // emission angle: the cloud top is seen higher (colder) toward the limb
      T -= 5.0 * (1.0 - mu);
      float x = clamp(0.5 + (T - 229.0) / 36.0 * uContrast, 0.0, 1.0);
      vec3 c = mix(vec3(0.12, 0.03, 0.22), vec3(0.42, 0.08, 0.5), smoothstep(0.0, 0.3, x));
      c = mix(c, vec3(0.9, 0.3, 0.15), smoothstep(0.25, 0.6, x));
      c = mix(c, vec3(1.0, 0.85, 0.35), smoothstep(0.55, 0.9, x));
      bg = c * c * 0.9;
    }
  }
  // ---- the atmosphere above the deck (reflected-light views only) ----
  if (uView < 1.5){
    Atm A = atmV(uvMode);
    vec2 ha = sphereHit(ro, rd, vec3(0.0), RTOP);
    if (ha.x < ha.y && ha.y > 0.0){
      float t0 = max(ha.x, 0.0), t1 = deck ? hs.x : ha.y;
      vec3 T;
      vec3 S = atmosphere(ro, rd, t0, t1, L, A, T) * ESUN;
      if (uvMode) S *= vec3(0.86, 0.9, 1.0) * 0.8;
      bg = S + T * bg;
    }
  } else if (uView < 2.5 && !deck){
    // 2,26 µm: the sunlit haze limb beyond the dayside is also overexposed
    float bI = length(cross(ro, rd));
    if (dot(ro, rd) < 0.0 && bI < RTOP){
      vec3 pt = normalize(ro - rd * dot(ro, rd));
      bg += vec3(1.0, 0.85, 0.7) * exp(-(bI - RC) / (4.0 / RKM)) * 0.6 * smoothstep(-0.1, 0.1, dot(pt, L));
    }
  } else if (!deck){
    // thermal limb: emission from the warm haze layer above the cloud tops
    float bI = length(cross(ro, rd));
    if (dot(ro, rd) < 0.0 && bI < RTOP) bg += vec3(0.25, 0.05, 0.25) * exp(-(bI - RC) / (5.0 / RKM)) * 0.25;
  }
  gl_FragColor = vec4(bg, 1.0);
}`;

/* ================================================================================================
   Asset loading (helpers adapted from the Earth scene)
   ================================================================================================ */
const URL_ = f => new URL(`../../../assets/venus/${f}`, import.meta.url).href;
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
    img.onerror = () => reject(new Error(t(`không tải được ${url}`, `failed to load ${url}`)));
    img.src = url;
  });
}
function loadImageTex(url, { aniso = 1, maxSize = 16384, repeat = true }) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    const img = tex.image;
    if (img.width > maxSize) {
      const s = maxSize / img.width, cv = document.createElement('canvas');
      cv.width = Math.round(img.width * s); cv.height = Math.round(img.height * s);
      const g = cv.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(img, 0, 0, cv.width, cv.height);
      tex.image = cv;
    }
    tex.colorSpace = THREE.NoColorSpace;
    tex.format = THREE.RedFormat;
    tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = aniso;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(t(`không tải được ${url}`, `failed to load ${url}`)))));
}
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
function solidTex(data, format, type) { const t = new THREE.DataTexture(data, 1, 1, format, type); t.needsUpdate = true; return t; }

// Magellan topography: a 1024×512 base (RG = metres + 4000, 16-bit, lossless) plus a 4096×2048
// companded residual (byte b: r = 2500 m · x|x|, x = (b − 128)/127) relative to the base's bilinear
// up-sampling. Rebuilt here into a half-float R texture in km with a box-filtered mip chain.
async function loadTopo(baseUrl, detUrl) {
  const [B, D] = await Promise.all([decodeImage(baseUrl), decodeImage(detUrl)]);
  const BW = B.W, BH = B.H, W = D.W, H = D.H, F = W / BW;
  const base = new Float32Array(BW * BH);
  for (let i = 0; i < BW * BH; i++) base[i] = B.px[i * 4] * 256 + B.px[i * 4 + 1] - 4000;
  let lv = new Float32Array(W * H);
  let mn = 1e9, mx = -1e9;
  const xa = new Int32Array(W), xb = new Int32Array(W), fxs = new Float32Array(W);
  for (let x = 0; x < W; x++) { const xs = (x + 0.5) / F - 0.5, x0 = Math.floor(xs); fxs[x] = xs - x0; xa[x] = ((x0 % BW) + BW) % BW; xb[x] = (x0 + 1) % BW; }
  for (let y = 0; y < H; y++) {
    const ys = (y + 0.5) / F - 0.5, y0 = Math.floor(ys), fy = ys - y0;
    const ra = Math.min(Math.max(y0, 0), BH - 1) * BW, rb = Math.min(Math.max(y0 + 1, 0), BH - 1) * BW;
    const di = (H - 1 - y) * W;                    // image row 0 is north; texture row 0 is south
    for (let x = 0; x < W; x++) {
      const fx = fxs[x];
      const up = (base[ra + xa[x]] * (1 - fx) + base[ra + xb[x]] * fx) * (1 - fy) + (base[rb + xa[x]] * (1 - fx) + base[rb + xb[x]] * fx) * fy;
      const q = (D.px[(y * W + x) * 4] - 128) / 127;
      const e = (up + 2500 * q * Math.abs(q)) / 1000;
      lv[di + x] = e;
      if (e < mn) mn = e; if (e > mx) mx = e;
    }
  }
  const cpu = { data: lv, W, H };
  const mips = [];
  let w = W, h = H;
  for (;;) {
    const half = new Uint16Array(w * h);
    for (let i = 0; i < half.length; i++) half[i] = toHalf(lv[i]);
    mips.push({ data: half, width: w, height: h });
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
  tex.mipmaps = mips; tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return { tex, min: mn, max: mx, cpu };
}
function sampleTopo(cpu, lat, lon) {
  const { data, W, H } = cpu;
  const x = (lon + 180) / 360 * W - 0.5, y = (lat + 90) / 180 * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.max(0, Math.min(H - 2, Math.floor(y))), fx = x - x0, fy = Math.max(0, Math.min(1, y - y0));
  const at = (i, j) => data[j * W + ((i % W) + W) % W];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

/* ================================================================================================
   Helpers
   ================================================================================================ */
const wrap180 = a => ((a + 180) % 360 + 360) % 360 - 180;
const hhmm = h => `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`;
const latStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? t('B', 'N') : t('N', 'S')}`;
const lonStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? t('Đ', 'E') : t('T', 'W')}`;
const num = (x, d = 0) => x.toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d });
// afternoon is west of the subsolar meridian on retrograde Venus: the Sun sits east (+azimuth) of the camera meridian
function sunFromHour(cam, hour, out) {
  const az = Math.atan2(cam.x, cam.z) + (hour - 12) * Math.PI / 12;
  return out.set(Math.sin(az), 0, Math.cos(az));
}
const localHour = (p, L) => (((12 + (Math.atan2(L.x, L.z) - Math.atan2(p.x, p.z)) / D2R / 15) % 24) + 24) % 24;
function hitSphere(o, d, R = 1) {
  const b = o.dot(d), c = o.lengthSq() - R * R, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
// named features (planetocentric, east longitude), for click / HUD read-outs
const FEATURES = [
  ['Maat Mons', 0.5, 194.6, 'núi lửa khiên cao nhất, ~8 km', 'highest shield volcano, ~8 km'], ['Ozza Mons', 4.5, 201.0, 'núi lửa khiên', 'shield volcano'], ['Sapas Mons', 8.5, 188.3, 'núi lửa khiên', 'shield volcano'],
  ['Atla Regio', 9.2, 199.9, 'vùng cao núi lửa', 'volcanic highland'], ['Maxwell Montes', 65.2, 3.3, 'dãy núi cao nhất, ~11 km', 'highest mountain range, ~11 km'], ['Lakshmi Planum', 68.6, 339.3, 'cao nguyên ~4 km', 'plateau ~4 km high'],
  ['Ishtar Terra', 70.4, 27.5, '"lục địa" phía bắc', 'northern "continent"'], ['Cleopatra', 65.8, 7.1, 'hố va chạm 105 km trên sườn Maxwell', '105 km impact crater on the flank of Maxwell'],
  ['Aphrodite Terra', -5.8, 104.8, '"lục địa" xích đạo', 'equatorial "continent"'], ['Ovda Regio', -2.8, 85.6, 'cao nguyên tessera', 'tessera highland'], ['Thetis Regio', -11.4, 129.9, 'cao nguyên tessera', 'tessera highland'],
  ['Artemis Corona', -41.0, 135.0, 'corona lớn nhất, ~2 600 km', 'largest corona, ~2,600 km'], ['Diana Chasma', -14.6, 155.3, 'hẻm vực sâu ~3 km', 'canyon ~3 km deep'],
  ['Beta Regio', 25.3, 282.8, 'vùng cao núi lửa – tách giãn', 'volcanic rift highland'], ['Theia Mons', 22.7, 281.0, 'núi lửa khiên', 'shield volcano'], ['Rhea Mons', 32.4, 282.1, 'núi lửa', 'volcano'],
  ['Phoebe Regio', -6.0, 282.8, 'vùng cao', 'highland'], ['Themis Regio', -37.4, 284.2, 'vùng cao núi lửa', 'volcanic highland'], ['Idunn Mons', -46.6, 214.5, 'núi lửa có thể còn hoạt động', 'possibly still-active volcano'],
  ['Gula Mons', 21.9, 359.2, 'núi lửa khiên', 'shield volcano'], ['Sif Mons', 22.0, 352.4, 'núi lửa khiên', 'shield volcano'], ['Alpha Regio', -25.5, 1.0, 'cao nguyên tessera', 'tessera highland'],
  ['Lada Terra', -59.3, 20.0, 'vùng cao phía nam', 'southern highland'], ['Mead', 12.5, 57.2, 'hố va chạm lớn nhất, ~270 km', 'largest impact crater, ~270 km'],
];
function nearestFeature(lat, lon) {
  let best = null, bd = 1e9;
  const p = llB(lat, lon);
  for (const f of FEATURES) { const d = Math.acos(Math.min(1, p.dot(llB(f[1], f[2])))) / D2R; if (d < bd) { bd = d; best = f; } }
  return bd < 7 ? { f: best, d: bd } : null;
}
const VIEWS = ['vis', 'uv', 'ir', 'lir', 'radar'];
const VIEW_TOAST = {
  vis: 'Ánh sáng khả kiến: một quả cầu màu kem gần như không có chi tiết — mây axit sulfuric phản xạ ~76 % ánh nắng.',
  uv: 'Tử ngoại 365 nm (ảnh tổng hợp thật từ camera UVI của tàu Akatsuki, 8/2017): chất hấp thụ UV chưa rõ bản chất vẽ nên chữ Y nằm ngang và các vệt xoắn, bị gió ~100 m/s cuốn quanh hành tinh trong 4,3 ngày.',
  ir: 'Cận hồng ngoại 2,26 µm, mặt đêm (ảnh tổng hợp thật từ camera IR2 của Akatsuki, 8/2016): khí quyển sâu nóng ~500 K phát sáng xuyên qua mây; mây dày hiện thành bóng tối, trôi chậm hơn (~76 m/s ở độ cao ~50 km). Lưỡi liềm ban ngày bị cháy sáng.',
  lir: 'Hồng ngoại nhiệt 10 µm (kiểu camera LIR của Akatsuki): nhiệt độ đỉnh mây ~230 K, vòng "cổ áo" lạnh ở vĩ độ ~65° và xoáy cực ấm hình chữ S.',
  radar: 'Radar 12,6 cm xuyên qua mây (ảnh khảm thật của tàu Magellan 1990–94, độ phân giải 2 km; quanh Maat Mons 450 m): sáng = gồ ghề hoặc sườn quay về phía radar.',
};
const VIEW_TOAST_EN = {
  vis: 'Visible light: an almost featureless cream-coloured ball — the sulfuric acid clouds reflect ~76 % of the sunlight.',
  uv: 'Ultraviolet 365 nm (real composite from the Akatsuki UVI camera, Aug 2017): an unknown UV absorber paints a sideways Y and spiral streaks, swept around the planet in 4.3 days by ~100 m/s winds.',
  ir: 'Near-infrared 2.26 µm, night side (real composite from the Akatsuki IR2 camera, Aug 2016): the hot ~500 K deep atmosphere glows through the clouds; thick clouds show up as dark silhouettes drifting more slowly (~76 m/s at ~50 km altitude). The dayside crescent is saturated.',
  lir: 'Thermal infrared 10 µm (in the style of the Akatsuki LIR camera): cloud tops at ~230 K, a cold "collar" at ~65° latitude and a warm S-shaped polar vortex.',
  radar: '12.6 cm radar sees through the clouds (real Magellan mosaic, 1990–94, 2 km resolution; 450 m around Maat Mons): bright = rough terrain or slopes facing the radar.',
};

/* ================================================================================================
   Scene definition
   ================================================================================================ */
export default [
{
  id: 'venus',
  name: 'Sao Kim',
  category: 'planets',
  order: 15,
  subtitle: 'Hành tinh đất đá · nhà kính mất kiểm soát',
  swatch: ['#fff1c8', '#a07a3a'],
  blurb: 'Lớp mây axit sulfuric phản xạ ~76 % ánh nắng và che kín bề mặt; bên dưới là <b>464 °C</b> và <b>92 bar</b>. Bản thân hành tinh quay ngược chiều, chậm đến mức một vòng mất 243 ngày — nhưng đỉnh mây lại <b>siêu tự quay</b> ~100 m/s, vòng quanh hành tinh chỉ trong ~4 ngày, nhanh ở xích đạo và chậm dần về phía cực nên các vệt tử ngoại bị kéo thành chữ Y. Mọi bản đồ ở đây là <b>dữ liệu thật</b> của Magellan và Akatsuki.',
  intro: 'Nhấn <b>V</b> để đổi bước sóng (khả kiến · tử ngoại · hồng ngoại đêm · nhiệt · radar). <b>O</b>: bay tới núi lửa Maat Mons; <b>P</b>: xoáy cực; <b>G</b>: sóng trọng lực khổng lồ trên Aphrodite.',
  facts: [
    ['Bán kính trung bình', '6 051,8 km'], ['Khối lượng', '4,87×10²⁴ kg'], ['Trọng lực bề mặt', '8,87 m/s²'],
    ['Nhiệt độ bề mặt', '~464 °C (737 K), ngày cũng như đêm'], ['Áp suất bề mặt', '92 bar (CO₂ 96,5 %, N₂ 3,5 %)'],
    ['Chu kỳ tự quay', '243,02 ngày (ngược chiều, trục nghiêng 177,4°)'], ['Ngày Mặt Trời', '116,75 ngày'], ['Chu kỳ quỹ đạo', '224,7 ngày'],
    ['Gió đỉnh mây (70 km)', '103 m/s ở xích đạo → một vòng trong 4,3 ngày (đo trên ảnh UVI 8/2017)'],
    ['Gió tầng mây dưới (~50 km)', '~76 m/s → 5,8 ngày (đo trên ảnh IR2 mặt đêm 8/2016)'],
    ['Đỉnh mây', '~70 km, ~−45 °C, ~0,03 bar'], ['Suất phản chiếu Bond / hình học', '0,76 / 0,69'],
    ['Đỉnh cao nhất', 'Maxwell Montes ~11 km'],
    ['Dữ liệu', 'NASA/JPL Magellan (USGS Astrogeology), JAXA/ISAS Akatsuki UVI & IR2 (DARTS, CC BY 4.0)'],
  ],
  en: {
    name: 'Venus',
    subtitle: 'Rocky planet · runaway greenhouse',
    blurb: 'A deck of sulfuric acid clouds reflects ~76 % of the sunlight and hides the surface completely; beneath it lie <b>464 °C</b> and <b>92 bar</b>. The planet itself spins backwards (retrograde) so slowly that one turn takes 243 days — yet the cloud tops <b>super-rotate</b> at ~100 m/s, circling the planet in only ~4 days, fastest at the equator and slower towards the poles, which stretches the ultraviolet markings into a Y. Every map here is <b>real data</b> from Magellan and Akatsuki.',
    intro: 'Press <b>V</b> to change wavelength (visible · ultraviolet · night-side infrared · thermal · radar). <b>O</b>: fly to the Maat Mons volcano; <b>P</b>: the polar vortex; <b>G</b>: the giant gravity wave over Aphrodite.',
    facts: [
      ['Mean radius', '6,051.8 km'], ['Mass', '4.87×10²⁴ kg'], ['Surface gravity', '8.87 m/s²'],
      ['Surface temperature', '~464 °C (737 K), day and night alike'], ['Surface pressure', '92 bar (CO₂ 96.5 %, N₂ 3.5 %)'],
      ['Rotation period', '243.02 days (retrograde, axial tilt 177.4°)'], ['Solar day', '116.75 days'], ['Orbital period', '224.7 days'],
      ['Cloud-top wind (70 km)', '103 m/s at the equator → one circuit in 4.3 days (measured on UVI images, Aug 2017)'],
      ['Lower-cloud wind (~50 km)', '~76 m/s → 5.8 days (measured on night-side IR2 images, Aug 2016)'],
      ['Cloud tops', '~70 km, ~−45 °C, ~0.03 bar'], ['Bond / geometric albedo', '0.76 / 0.69'],
      ['Highest peak', 'Maxwell Montes ~11 km'],
      ['Data', 'NASA/JPL Magellan (USGS Astrogeology), JAXA/ISAS Akatsuki UVI & IR2 (DARTS, CC BY 4.0)'],
    ],
  },
  unitKm: RKM,
  camera: { distance: 3.2, polar: 80, azimuth: 30, min: 1.003, max: 30, fov: 45 },
  exposure: 0.85,
  bloom: 0.3,
  params: [
    { key: 'view', label: 'Bước sóng quan sát', type: 'seg', options: [['vis', 'Khả kiến'], ['uv', 'UV 365 nm'], ['ir', 'IR 2,3 µm'], ['lir', 'Nhiệt 10 µm'], ['radar', 'Radar']], default: 'vis', note: 'UV: Akatsuki UVI · IR 2,3 µm: Akatsuki IR2 (mặt đêm) · Nhiệt: kiểu Akatsuki LIR (mô hình) · Radar: Magellan SAR.',
      en: { label: 'Observing wavelength', options: ['Visible', 'UV 365 nm', 'IR 2.3 µm', 'Thermal 10 µm', 'Radar'], note: 'UV: Akatsuki UVI · IR 2.3 µm: Akatsuki IR2 (night side) · Thermal: Akatsuki LIR style (model) · Radar: Magellan SAR.' } },
    { key: 'hour', label: 'Giờ địa phương (dưới camera)', type: 'range', min: 0, max: 24, step: 0.1, default: 14.5, fmt: v => hhmm(v), note: '12:00 = Mặt Trời sau lưng bạn. Sao Kim quay ngược chiều nên Mặt Trời mọc ở phía tây, lặn ở phía đông; buổi chiều nằm ở phía tây điểm cận nhật. Xem IR 2,3 µm vào ~00:00 để thấy mặt đêm.',
      en: { label: 'Local time (below camera)', note: '12:00 = the Sun behind you. Venus spins backwards, so the Sun rises in the west and sets in the east; afternoon lies west of the subsolar point. Use IR 2.3 µm at ~00:00 to see the night side.' } },
    { key: 'speed', label: 'Nhịp thời gian Sao Kim', type: 'range', min: 0, max: 3, step: 0.01, default: 0.4, fmt: v => t(`${(v * 6).toFixed(1)} giờ / giây`, `${(v * 6).toFixed(1)} h / s`), note: 'Số giờ trên Sao Kim trôi qua mỗi giây (với tốc độ thời gian chung 6×). Đỉnh mây quay một vòng sau 104 giờ; bề mặt sau 5 832 giờ.',
      en: { label: 'Venus time rate', note: 'Hours on Venus elapsed per second (with the global time speed at 6×). The cloud tops go round once in 104 hours; the surface in 5,832 hours.' } },
    { key: 'contrast', label: 'Độ tương phản ảnh', type: 'range', min: 0.3, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), note: '×1 = mức kéo giãn của ảnh công bố (UV ×3, IR ×2,6 so với tương phản thật). Kéo về 0,33 để thấy tương phản thật của UV: chữ Y chỉ tối hơn ~30 %.',
      en: { label: 'Image contrast', note: '×1 = the stretch of the published images (UV ×3, IR ×2.6 relative to true contrast). Pull it down to 0.33 to see the true UV contrast: the Y is only ~30 % darker.' } },
    { key: 'atmo', label: 'Lớp sương mù trên mây', type: 'range', min: 0, max: 4, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), note: 'Tán xạ Rayleigh của CO₂ và lớp sương mù H₂SO₄ ở 70–90 km: viền sáng quanh đĩa và ánh hoàng hôn vượt qua đường phân chia ngày–đêm.',
      en: { label: 'Haze above the clouds', note: 'Rayleigh scattering by CO₂ and the H₂SO₄ haze at 70–90 km: the bright rim around the disc and twilight glow reaching past the terminator.' } },
    { key: 'exag', label: 'Phóng đại địa hình (radar)', type: 'range', min: 1, max: 40, step: 0.5, default: 10, fmt: v => '×' + v.toFixed(1), note: 'Maat Mons cao 8 km — chỉ 0,13 % bán kính. Ảnh phối cảnh nổi tiếng của JPL về Maat Mons phóng đại ×10.',
      en: { label: 'Terrain exaggeration (radar)', note: 'Maat Mons is 8 km high — only 0.13 % of the radius. JPL\'s famous perspective view of Maat Mons is exaggerated ×10.' } },
    { key: 'tint', label: 'Màu ảnh radar', type: 'seg', options: [['grey', 'Xám SAR'], ['venera', 'Cam Venera'], ['topo', 'Màu độ cao']], default: 'venera', note: 'Xám = tán xạ ngược radar gốc. Cam: màu thật của đá dưới ánh sáng lọc qua khí quyển (ảnh Venera 13). Màu độ cao: bản đồ cao độ Magellan.',
      en: { label: 'Radar image colour', options: ['SAR grey', 'Venera orange', 'Elevation colour'], note: 'Grey = raw radar backscatter. Orange: the true colour of the rock under light filtered by the atmosphere (Venera 13 images). Elevation colour: Magellan altimetry map.' } },
    { key: 'wave', label: 'Sóng trọng lực đứng (Aphrodite)', type: 'toggle', default: true, note: 'Sóng hình cánh cung dài 10 000 km đứng yên trên dãy núi Aphrodite Terra (Akatsuki LIR, 12/2015). Chỉ thấy trong ảnh nhiệt, vào buổi chiều.',
      en: { label: 'Stationary gravity wave (Aphrodite)', note: 'A 10,000 km bow-shaped wave standing still over the Aphrodite Terra mountains (Akatsuki LIR, Dec 2015). Visible only in the thermal view, in the afternoon.' } },
  ],
  actions: [
    { label: 'Đổi bước sóng', key: 'v', title: 'Khả kiến → UV → IR đêm → nhiệt → radar', run: ctx => {
      const nx = VIEWS[(VIEWS.indexOf(ctx.params.view) + 1) % VIEWS.length];
      ctx.params.view = nx;
      if (nx === 'ir') ctx.params.hour = 0.5;
      else if (nx === 'vis' || nx === 'uv') ctx.params.hour = 14.5;
      ctx.toast(t(VIEW_TOAST[nx], VIEW_TOAST_EN[nx]), 7000);
    }, en: { label: 'Change wavelength', title: 'Visible → UV → night IR → thermal → radar' } },
    { label: 'Maat Mons', key: 'o', title: 'Phối cảnh núi lửa Maat Mons (radar Magellan, phóng đại ×10)', run: ctx => ctx.scene._venusShot?.(ctx, 'maat'),
      en: { label: 'Maat Mons', title: 'Perspective view of the Maat Mons volcano (Magellan radar, exaggerated ×10)' } },
    { label: 'Xoáy cực', key: 'p', title: 'Nhìn xuống xoáy cực nam hình chữ S', run: ctx => ctx.scene._venusShot?.(ctx, 'pole'),
      en: { label: 'Polar vortex', title: 'Look down on the S-shaped south polar vortex' } },
    { label: 'Sóng núi', key: 'g', title: 'Sóng trọng lực đứng 10 000 km trên Aphrodite Terra', run: ctx => ctx.scene._venusShot?.(ctx, 'wave'),
      en: { label: 'Mountain wave', title: 'The 10,000 km stationary gravity wave over Aphrodite Terra' } },
  ],
  create(ctx) {
    const R = ctx.renderer, caps = R.capabilities;
    const maxTex = caps.maxTextureSize, aniso = Math.min(8, caps.getMaxAnisotropy());
    const grey = v => solidTex(new Uint8Array([v]), THREE.RedFormat, THREE.UnsignedByteType);
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) }, uFlow: { value: new THREE.Vector4() },
      uView: { value: 0 }, uRot: { value: 0 }, uTH: { value: 0 }, uAtmo: { value: 1 }, uContrast: { value: 1 },
      uWave: { value: 1 }, uExag: { value: 10 }, uTint: { value: 1 }, uTexOK: { value: 0 },
      uHMin: { value: -3 }, uHMax: { value: 11 }, uTexel: { value: TAU / 4096 }, uRadTexel: { value: TAU / 8192 },
      uUV: { value: grey(128) }, uIR: { value: grey(128) }, uRadar: { value: grey(120) }, uMaat: { value: grey(120) },
      uTopo: { value: solidTex(new Uint16Array([0]), THREE.RedFormat, THREE.HalfFloatType) },
    };
    const U = uniforms;
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    let topoCPU = null;
    const ready = Promise.all([
      retry(loadImageTex, URL_('venus_uv365.webp'), { aniso: 4 }),
      retry(loadImageTex, URL_('venus_ir226.webp'), { aniso: 4 }),
      retry(loadImageTex, URL_('venus_radar_8k.webp'), { aniso, maxSize: maxTex }),
      retry(loadImageTex, URL_('venus_maat_radar.webp'), { aniso, repeat: false }),
      loadTopo(URL_('venus_topo_base.webp'), URL_('venus_topo_detail.webp')).catch(() => loadTopo(URL_('venus_topo_base.webp?r=1'), URL_('venus_topo_detail.webp?r=1'))),
    ]).then(([uv, ir, rad, maat, topo]) => {
      U.uUV.value = uv; U.uIR.value = ir; U.uRadar.value = rad; U.uMaat.value = maat; U.uTopo.value = topo.tex;
      U.uTexel.value = TAU / topo.tex.image.width; U.uRadTexel.value = TAU / rad.image.width;
      U.uHMin.value = topo.min; U.uHMax.value = topo.max;
      topoCPU = topo.cpu;
      for (const t of [uv, ir, rad, maat, topo.tex]) R.initTexture(t);
      U.uTexOK.value = 1;
    });

    // ---- Venus time and rotation -------------------------------------------------------------
    let tH = null, lastSim = null;
    const Y = new THREE.Vector3(0, 1, 0);
    const spinOf = h => -W_SURF * h;                      // retrograde: world = rotY(spin)·body
    const bodyLatLon = (p, h) => ({ lat: Math.asin(p.y / p.length()) / D2R, lon: wrap180((Math.atan2(p.x, p.z) - spinOf(h)) / D2R) });

    /* ---- guided shots: a camera that rides with the (slowly) rotating surface ----------------- */
    const def = ctx.scene;
    const shot = { on: false, tw: null, pending: new URLSearchParams(location.search).get('venusShot') };
    const toWorld = (v, rot) => v.clone().applyAxisAngle(Y, rot);
    const toBody = (v, rot) => v.clone().applyAxisAngle(Y, -rot);
    function applyFrame(c, f, rot) {
      c.camera.position.copy(toWorld(f.pos, rot));
      c.controls.target.copy(toWorld(f.tgt, rot));
      c.camera.up.copy(toWorld(f.up, rot)).normalize();
      c.camera.lookAt(c.controls.target);
    }
    function startTween(c, bodyFrame, dur, onDone) {
      const rot = spinOf(tH ?? 0);
      shot.tw = { t0: performance.now(), dur: dur * 1000, onDone,
        a: { pos: toBody(c.camera.position, rot), tgt: toBody(c.controls.target, rot), up: toBody(c.camera.up, rot) }, b: bodyFrame };
    }
    const kmU = km => km / RKM;
    function frames(kind) {
      if (kind === 'maat') {
        // like JPL's PIA00254: from ~630 km north of Maat Mons, a few km up, looking south (relief ×10)
        const tgt = llB(0.5, 194.6).multiplyScalar(1 + kmU(30));
        const pos = llB(6.2, 194.6).multiplyScalar(1 + kmU(75));
        return { pos, tgt, up: pos.clone().normalize() };
      }
      if (kind === 'pole') {
        const pos = llB(-78, 40).multiplyScalar(2.3);
        return { pos, tgt: new THREE.Vector3(0, -0.55, 0), up: llB(10, 40) };
      }
      // the Aphrodite bow wave: looking down on 90°E from 2,6 radii
      const pos = llB(-3, 92).multiplyScalar(2.7);
      return { pos, tgt: new THREE.Vector3(), up: Y.clone() };
    }
    function go(c, kind, instant) {
      const p = c.params, keep = instant ? (new URLSearchParams(location.search).get('params') || '') : '';
      const set = (k, v) => { if (!keep.includes(k + ':')) p[k] = v; };
      if (kind === 'maat') { set('view', 'radar'); set('exag', 10); }
      if (kind === 'pole') { set('view', 'lir'); }
      if (kind === 'wave') { set('view', 'lir'); set('wave', true); set('hour', 16.5); set('contrast', 1.6); }
      shot.on = kind === 'maat';
      c.controls.minDistance = shot.on ? 0.002 : def.camera.min;
      const f = frames(kind);
      if (instant) { applyFrame(c, f, spinOf(tH ?? 0)); shot.tw = null; }
      else startTween(c, f, 4);
      c.toast?.(t({
        maat: 'Maat Mons (0,5°B, 194,6°Đ): núi lửa khiên cao nhất Sao Kim, ~8 km trên bán kính trung bình, chân rộng ~400 km. Ảnh radar Magellan độ phân giải 450 m (khảm nhìn trái 225 m, lấy trung bình): các dòng dung nham trẻ, gồ ghề phản xạ radar mạnh nên sáng, toả ra từ miệng núi. Địa hình phóng đại ×10 như ảnh phối cảnh PIA00254 của JPL. Bên phải là Ozza Mons.',
        pole: 'Xoáy cực nam: lõi ấm (~240 K) hình chữ S quay một vòng trong ~2,5 ngày, tâm lắc lư vài độ quanh cực và đổi dạng (lưỡng cực ↔ đơn cực) trong một–hai ngày — Venus Express phát hiện năm 2006. Bao quanh là "cổ áo" lạnh ~220 K ở vĩ độ ~65°.',
        wave: 'Tháng 12/2015 camera LIR của Akatsuki thấy một sóng hình cánh cung dài 10 000 km, chênh nhiệt ~1–2 K, <b>đứng yên</b> phía trên dãy núi Aphrodite Terra suốt nhiều ngày dù gió ở đó thổi 100 m/s: sóng trọng lực sinh ra khi khí quyển tầng thấp trượt qua núi, lan thẳng lên 65 km. Nó chỉ xuất hiện khi vùng núi đang buổi chiều (lớp khí sát mặt đất ổn định nhất), mạnh nhất gần đường hoàng hôn.',
      }[kind], {
        maat: 'Maat Mons (0.5°N, 194.6°E): the highest shield volcano on Venus, ~8 km above the mean radius, ~400 km across at the base. Magellan radar at 450 m resolution (left-looking 225 m mosaic, averaged): young, rough lava flows reflect the radar strongly, so they shine bright as they spread from the summit. Terrain exaggerated ×10 as in JPL\'s perspective view PIA00254. Ozza Mons is on the right.',
        pole: 'South polar vortex: an S-shaped warm core (~240 K) turning once every ~2.5 days, its centre wandering a few degrees around the pole and changing shape (dipole ↔ monopole) within a day or two — discovered by Venus Express in 2006. It is surrounded by a cold ~220 K "collar" at ~65° latitude.',
        wave: 'In December 2015 Akatsuki\'s LIR camera saw a 10,000 km bow-shaped wave, ~1–2 K warmer than its surroundings, <b>standing still</b> above the Aphrodite Terra mountains for days even though the wind there blows at 100 m/s: a gravity wave launched as the lower atmosphere flows over the mountains, propagating straight up to 65 km. It appears only when the highlands are in the afternoon (when the near-surface layer is most stable), strongest near the evening terminator.',
      }[kind]), 12000);
    }
    function leave(c) {
      c.params.exag = 10;
      const cam = def.camera, rot = spinOf(tH ?? 0);
      const pos = new THREE.Vector3().setFromSphericalCoords(cam.distance, cam.polar * D2R, cam.azimuth * D2R);
      startTween(c, { pos: toBody(pos, rot), tgt: new THREE.Vector3(), up: toBody(Y, rot) }, 3.5, () => {
        shot.on = false; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); c.controls.target.set(0, 0, 0);
      });
    }
    def._venusShot = (c, kind) => {
      if (shot.tw) return;
      if (kind === 'maat' && shot.on) return leave(c);
      if (shot.on && kind !== 'maat') { shot.on = false; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); }
      go(c, kind, false);
    };
    let lastRot = null;

    return {
      material,
      ready,
      exit(c) {
        if (shot.on || shot.tw) { shot.on = false; shot.tw = null; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); c.controls.target.set(0, 0, 0); }
      },
      update(dt, c) {
        const p = c.params, sim = c.time.sim;
        if (tH === null) tH = sim * p.speed;
        else tH += (sim - lastSim) * p.speed;
        lastSim = sim;
        const rot = spinOf(tH);
        if (shot.pending && U.uTexOK.value > 0) { const k = shot.pending; shot.pending = null; go(c, k, true); }
        if (shot.tw) {
          const tw = shot.tw, k0 = Math.min(1, (performance.now() - tw.t0) / tw.dur), e = k0 * k0 * (3 - 2 * k0);
          const ra = tw.a.pos.length(), rb = tw.b.pos.length();
          const dir = tw.a.pos.clone().normalize().lerp(tw.b.pos.clone().normalize(), e);
          if (dir.lengthSq() < 1e-6) dir.copy(tw.b.pos).normalize();
          const r = Math.exp(Math.log(ra) * (1 - e) + Math.log(rb) * e) * (1 + 0.3 * Math.sin(Math.PI * e) * Math.min(1, Math.abs(ra - rb)));
          applyFrame(c, { pos: dir.normalize().multiplyScalar(r), tgt: tw.a.tgt.clone().lerp(tw.b.tgt, e), up: tw.a.up.clone().lerp(tw.b.up, e) }, rot);
          if (k0 >= 1) { shot.tw = null; tw.onDone?.(); }
        } else if (shot.on && lastRot !== null && rot !== lastRot) {
          const d = rot - lastRot;
          c.camera.position.applyAxisAngle(Y, d); c.controls.target.applyAxisAngle(Y, d); c.camera.up.applyAxisAngle(Y, d);
          c.camera.lookAt(c.controls.target);
        }
        lastRot = rot;
        // keep the camera above the clouds (or the exaggerated relief in the radar view)
        // (radar: the exaggerated terrain under the camera, from the CPU copy of the topography)
        let top = 1 + 72 / RKM;
        if (p.view === 'radar') {
          const g = bodyLatLon(c.camera.position, tH), eL = topoCPU ? sampleTopo(topoCPU, g.lat, g.lon) : U.uHMax.value;
          top = 1 + ((eL + 1.5) * U.uExag.value + 2) / RKM;
        }
        if (c.camera.position.length() < top) c.camera.position.setLength(top);

        sunFromHour(c.camera.position, p.hour, U.uSunDir.value);
        U.uView.value = VIEWS.indexOf(p.view);
        U.uRot.value = -rot % TAU;
        U.uTH.value = tH % 98400;                         // (41·120·20 h: keeps the vortex cycles continuous)
        U.uFlow.value.set((W_TOP * tH) % TAU, (W_LOW * tH) % TAU, ((tH / PH) % 1 + 1) % 1, (W_VORTEX * tH) % TAU);
        U.uAtmo.value = p.atmo; U.uContrast.value = p.contrast; U.uWave.value = p.wave ? 1 : 0;
        U.uTint.value = ['grey', 'venera', 'topo'].indexOf(p.tint);
        U.uExag.value += (p.exag - U.uExag.value) * Math.min(1, dt * 4);
        if (Math.abs(p.exag - U.uExag.value) < 0.01) U.uExag.value = p.exag;
      },
      hud(c) {
        const cp = c.camera.position, h = tH ?? 0, g = bodyLatLon(cp, h), phi = g.lat * D2R;
        const uTop = 102.9 * ratioTop(phi) * Math.cos(phi), uLow = 76 * ratioLow(phi) * Math.cos(phi);
        const perTop = TAU / (W_TOP * ratioTop(phi)) / 24;
        const alt = (cp.length() - 1) * RKM;
        const rows = [
          [c.params.view === 'radar' ? t('Độ cao trên bề mặt', 'Altitude above surface') : t('Độ cao trên đỉnh mây', 'Altitude above cloud tops'), c.fmt.dist((c.params.view === 'radar' ? alt : alt - 70) * 1000)],
          [t('Điểm dưới camera', 'Point below camera'), `${latStr(g.lat)} ${lonStr(g.lon)}`],
          [t('Giờ Mặt Trời tại đó', 'Local solar time there'), hhmm(c.params.hour)],
          [t('Gió đỉnh mây ở vĩ độ này', 'Cloud-top wind at this latitude'), t(`${num(uTop)} m/s về phía tây · vòng quanh trong ${num(perTop, 1)} ngày`, `${num(uTop)} m/s westward · one circuit in ${num(perTop, 1)} days`)],
          [t('Gió tầng mây dưới (~50 km)', 'Lower-cloud wind (~50 km)'), `${num(uLow)} m/s`],
          [t('Thời gian trên Sao Kim', 'Time on Venus'), t(`${num(h / 24, 2)} ngày · mây đỉnh đã quay ${num(W_TOP * h / TAU, 2)} vòng, bề mặt ${num(W_SURF * h / TAU, 3)} vòng`, `${num(h / 24, 2)} days · cloud tops have turned ${num(W_TOP * h / TAU, 2)} times, the surface ${num(W_SURF * h / TAU, 3)} times`)],
        ];
        if (topoCPU) rows.push([t('Độ cao địa hình dưới camera', 'Terrain height below camera'), t(`${num(sampleTopo(topoCPU, g.lat, g.lon) * 1000)} m (so với bán kính 6 051,8 km)`, `${num(sampleTopo(topoCPU, g.lat, g.lon) * 1000)} m (relative to the 6,051.8 km radius)`)]);
        return rows;
      },
      onClick(info, c) {
        const radar = c.params.view === 'radar';
        const th = hitSphere(info.origin, info.dir, radar ? 1 : 1 + 70 / RKM);
        if (th === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, th).normalize(), L = U.uSunDir.value;
        const g = bodyLatLon(p, tH ?? 0), phi = g.lat * D2R;
        const e = topoCPU ? sampleTopo(topoCPU, g.lat, g.lon) : null;
        const nf = nearestFeature(g.lat, ((g.lon % 360) + 360) % 360);
        const near = nf ? t(` · gần <b>${nf.f[0]}</b> (${nf.f[3]})`, ` · near <b>${nf.f[0]}</b> (${nf.f[4]})`) : '';
        const where = t(`Vĩ độ <b>${latStr(g.lat)}</b>, kinh độ <b>${lonStr(g.lon)}</b>${near}`, `Latitude <b>${latStr(g.lat)}</b>, longitude <b>${lonStr(g.lon)}</b>${near}`);
        if (radar) {
          c.toast(t(`${where} · độ cao <b>${e === null ? '?' : num(e * 1000)} m</b>. Trên mặt đất: 464 °C, 92 bar — áp suất như ở độ sâu 900 m dưới biển; bầu trời màu cam đục và nắng chỉ còn vài phần trăm.`,
            `${where} · elevation <b>${e === null ? '?' : num(e * 1000)} m</b>. On the ground: 464 °C, 92 bar — the pressure 900 m deep in the ocean; the sky is a murky orange and only a few percent of the sunlight gets through.`));
        } else {
          const u = 102.9 * ratioTop(phi) * Math.cos(phi);
          c.toast(t(`${where} · giờ địa phương <b>${hhmm(localHour(p, L))}</b>. Ở đỉnh mây (70 km): ~−45 °C, ~0,03 bar, gió <b>${num(u)} m/s</b> về phía tây. Ở độ cao ~55 km nhiệt độ và áp suất gần giống mặt đất Trái Đất.`,
            `${where} · local time <b>${hhmm(localHour(p, L))}</b>. At the cloud tops (70 km): ~−45 °C, ~0.03 bar, wind <b>${num(u)} m/s</b> westward. At ~55 km the temperature and pressure are close to those on Earth's surface.`));
        }
      },
    };
  },
},
];
