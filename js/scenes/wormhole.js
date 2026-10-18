// Hố sâu (Wormhole) — the Interstellar wormhole, ray traced in the "Dneg" metric of
// James, von Tunzelmann, Franklin & Thorne (2015), "Visualizing Interstellar's Wormhole",
// Am. J. Phys. 83, 486:
//
//   ds² = −dt² + dℓ² + r(ℓ)² (dθ² + sin²θ dφ²)
//   |ℓ| ≤ a : r = ρ
//   |ℓ| > a : x = 2(|ℓ| − a)/(πM),  r = ρ + M[x·atan x − ½ ln(1 + x²)],  W = 1.42953 M
//
// One scene unit = the throat radius ρ. ℓ > 0 is our universe (A, with Saturn), ℓ < 0 the other
// universe (B, with the black hole Gargantua). Null geodesics are planar; in the plane we integrate
//   ℓ'' = b² r'(ℓ)/r³,  φ' = b/r²   (unit-speed affine parameter, b = impact parameter)
// inside the "wormhole region" |ℓ| < ℓ_W. Outside it each universe is treated as flat Cartesian space
// with radius R = r(ℓ) (r' → 1 there), so rays go straight — except near Gargantua, where the host's
// Schwarzschild tracer (js/glsl/blackhole.js) takes over.
//
// Universe frames. Inside the integrator both sides use the "naive" picture: a point is (ℓ, n̂) and
// n̂ is the same unit vector on both sides of the throat. Universe B's Cartesian frame is the naive
// one reflected by M_B = diag(1, 1, −1). Why: the local frame (ê_ℓ, ê_θ, ê_φ) is continuous through
// the throat, but ê_ℓ is +n̂ (outward) in A and −n̂ (inward) in B, so the naive Cartesian picture has a
// parity flip at the throat. M_B is a second reflection that cancels it: flying in along −z at n̂ = +z
// in A, the camera comes out of the throat in B still moving along −z with x and y unchanged.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';
import { BH_GLSL } from '../glsl/blackhole.js';
import { t } from '../i18n.js';

const V3 = THREE.Vector3;
const W_TO_M = 1 / 1.42953;            // lensing width W = 1.42953 M (James et al. 2015, eq. 7)
const L_PAD = 24;                      // wormhole region: |ℓ| < ℓ_W = a + 24 ρ
const CB = new V3(20, 4, -125);        // Gargantua's centre in universe B's frame
const RSB = 1.5;                       // Gargantua's Schwarzschild radius (scene units)
const BHR = 60 * RSB;                  // radius of the region traced with the Schwarzschild metric
const SAT_C = new V3(-52, -8, -62);    // Saturn's centre in universe A
const SAT_R = 9;                       // Saturn's equatorial radius (scene units)
const START = new V3(9, 2.5, 26);      // default viewpoint in A
const ARRIVE = new V3(-12, 5, 35);     // arrival offset from Gargantua (≈ 26 r_s), mouth side
const MOUTH_VIEW_B = new V3(-5, 1.3, 10);   // B side opposite Gargantua: mouth in front of the black hole

// Blackbody lookups with an explicit LOD: implicit-derivative sampling inside the branchy
// multi-segment loop forces the D3D compiler to flatten branches (every pixel would pay for traceBH).
const LOD_FIX = `
#undef texture2D
#define texture2D(s, c) textureLod(s, c, 0.0)
`;
const FRAG = LOD_FIX + COMMON_GLSL + BH_GLSL + /* glsl */`
uniform float uA, uM, uLW, uRW;        // half-length a, lensing mass M, region limits ℓ_W and R_W = r(ℓ_W)
uniform int uCamMode;                  // 0: A Cartesian · 1: B Cartesian · 2: inside wormhole region · 3: Gargantua region of B
uniform float uCamL, uCamS;            // camera ℓ and side sign (mode 2)
uniform vec3 uCamN;                    // camera angular position n̂ in the naive frame (mode 2)
uniform vec3 uBHC; uniform float uBHrs, uBHR;
uniform vec3 uSatC, uSatX, uSatY, uSatZ;   // Saturn centre and body axes (Y = spin pole)
uniform float uSatR, uSatOn;
uniform vec3 uSunDir;

#define RHO 1.0
#define WH_K 0.085
const vec3 MB = vec3(1.0, 1.0, -1.0);   // universe B Cartesian = M_B · naive
const float SAT_F = 0.09796;            // Saturn's flattening (polar radius 54 364 km / equatorial 60 268 km)
const float SAT_OMEGA = 1.6528e-4;      // rotation, 2π / 10 h 33 min (rad per simulated second)

/* ---------------- Dneg wormhole ---------------- */
float rOf(float l){
  float d = abs(l) - uA;
  if (d <= 0.0) return RHO;
  float x = 2.0 * d / (PI * uM);
  return RHO + uM * (x * atan(x) - 0.5 * log(1.0 + x * x));
}
float drOf(float l){
  float d = abs(l) - uA;
  if (d <= 0.0) return 0.0;
  return sign(l) * (2.0 / PI) * atan(2.0 * d / (PI * uM));
}
vec3 anyPerp(vec3 n){ return normalize(cross(n, abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0))); }

// Straight ray (universe sign s) arriving on the region boundary |X| = R_W → geodesic state.
// The exterior is flat with R = r(ℓ), so a Cartesian direction (d_r, d_t) corresponds to
// r'ℓ' : rφ' = d_r : d_t with ℓ'² + r²φ'² = 1 (the exact inverse of the exit map below).
void enterWH(vec3 X, vec3 D, float s, out float l, out float p, out float b, out vec3 e1, out vec3 e2){
  vec3 Xn = s < 0.0 ? X * MB : X, Dn = s < 0.0 ? D * MB : D;
  e1 = normalize(Xn);
  float dr = dot(Dn, e1); vec3 t = Dn - dr * e1; float tl = length(t);
  e2 = tl > 1e-6 ? t / tl : anyPerp(e1);
  l = s * uLW;
  float r = rOf(l), rp = max(abs(drOf(l)), 1e-3);
  float k = inversesqrt(dr * dr / (rp * rp) + tl * tl);
  p = s * dr / rp * k;
  b = r * tl * k;
}

// Integrates the planar null geodesic with leapfrog until it leaves the region; returns the
// universe sign and writes the Cartesian exit position / direction in that universe's frame.
float traceWH(float l, float p, float b, vec3 e1, vec3 e2, out vec3 X, out vec3 D, inout float path){
  float ph = 0.0, r = rOf(l);
  for (int i = 0; i < WH_STEPS; i++){
    if (abs(l) >= uLW && l * p > 0.0) break;
    float d = abs(l) - uA;
    if (d < 0.0 && abs(p) > 1e-3){
      // inside the cylindrical throat r = ρ: geodesics are straight lines in (ℓ, φ) — jump to its end
      float tg = p > 0.0 ? uA : -uA, ds = (tg - l) / p;
      ph += ds * b / (RHO * RHO); path += ds; l = tg; r = RHO;
      continue;
    }
    // step resolves the |ℓ| ≈ a transition (width ~M) and grows with distance from it
    float h = WH_K * abs(d) + 0.1 * uM + 0.002;
    h = min(h, 0.3 * r * r / max(b, 1e-4));        // ≤ 0.3 rad of φ per step
    float acc = b * b * drOf(l) / (r * r * r);
    p += 0.5 * h * acc;
    float r0 = r;
    l += h * p; r = rOf(l);
    ph += h * b * 0.5 * (1.0 / (r0 * r0) + 1.0 / (r * r));
    acc = b * b * drOf(l) / (r * r * r);
    p += 0.5 * h * acc;
    path += h;
  }
  float s = l >= 0.0 ? 1.0 : -1.0;
  // restore |ℓ'| from the null condition ℓ'² + b²/r² = 1 (removes integrator drift)
  p = sign(p) * sqrt(max(1.0 - b * b / (r * r), 0.0));
  vec3 n = cos(ph) * e1 + sin(ph) * e2, ep = -sin(ph) * e1 + cos(ph) * e2;
  vec3 Dn = normalize(abs(drOf(l)) * p * s * n + (b / r) * ep + 1e-7 * n);
  X = r * n; D = Dn;
  if (s < 0.0){ X *= MB; D *= MB; }
  return s;
}

/* ---------------- Saturn ---------------- */
vec3 toSat(vec3 v){ return vec3(dot(v, uSatX), dot(v, uSatY), dot(v, uSatZ)); }
// ray vs oblate spheroid in body units (equatorial radius 1); returns near t or -1
float ellHit(vec3 o, vec3 d){
  const float k = 1.0 / (1.0 - SAT_F);
  vec3 os = vec3(o.x, o.y * k, o.z), ds = vec3(d.x, d.y * k, d.z);
  float A = dot(ds, ds), B = dot(os, ds), C = dot(os, os) - 1.0;
  float h = B * B - A * C;
  if (h < 0.0) return -1.0;
  return (-B - sqrt(h)) / A;
}
float vnoise(float x){ float i = floor(x), f = fract(x); return mix(hash13(vec3(i, 3.1, 7.7)), hash13(vec3(i + 1.0, 3.1, 7.7)), f * f * (3.0 - 2.0 * f)); }
// a gap of width w at c, blended to its coverage when the pixel footprint fw is wider
float gapF(float x, float c, float w, float fw){ float W = max(w, fw); return 1.0 - (w / W) * (1.0 - smoothstep(0.5 * W - 0.3 * fw, 0.5 * W + 0.3 * fw, abs(x - c))); }
// Normal optical depth and particle albedo/colour of the main rings at radius x (Saturn radii).
// Radii and τ follow Voyager/Cassini occultation profiles (rounded): C 1.239–1.527, B 1.527–1.951,
// Cassini division 1.951–2.027 (Huygens gap at its inner edge), A 2.027–2.269 with the Encke gap at 2.214.
float ringProps(float x, float fw, out vec3 alb){
  alb = vec3(0.0);
  if (x < 1.239 || x > 2.270) return 0.0;
  // ringlet structure at several scales, faded out where the pixel footprint is larger
  float f1 = 37.0, f2 = 160.0, f3 = 620.0;
  float n = (vnoise(x * f1) - 0.5) * 0.9
          + (vnoise(x * f2 + 11.0) - 0.5) * 0.8 * exp(-pow(f2 * fw * 0.5, 2.0))
          + (vnoise(x * f3 + 37.0) - 0.5) * 0.7 * exp(-pow(f3 * fw * 0.5, 2.0));
  float tau;
  if (x < 1.527){                       // C ring: dusky, low τ with denser "plateaux"
    float plat = smoothstep(0.55, 0.65, vnoise(x * 60.0 + 4.0));
    tau = (0.06 + 0.32 * plat) * (1.0 + 0.6 * n);
    alb = vec3(0.60, 0.55, 0.50) * 0.28;
  } else if (x < 1.951){                // B ring: the brightest and most opaque, τ ≈ 1–5
    tau = mix(0.9, 2.8, smoothstep(1.56, 1.66, x)) * (1.0 - 0.3 * smoothstep(1.86, 1.95, x)) * (1.0 + 0.55 * n);
    alb = mix(vec3(0.80, 0.70, 0.56), vec3(0.90, 0.79, 0.64), smoothstep(1.6, 1.8, x)) * 0.62;
  } else if (x < 2.027){                // Cassini division
    tau = (x < 1.962 ? 0.01 : 0.10 + 0.08 * n);
    alb = vec3(0.58, 0.54, 0.50) * 0.25;
  } else {                              // A ring
    tau = mix(0.75, 0.45, smoothstep(2.05, 2.24, x)) * (1.0 + 0.4 * n);
    tau *= gapF(x, 2.2141, 0.0054, fw); // Encke gap (325 km)
    alb = vec3(0.80, 0.75, 0.68) * 0.52;
  }
  // particle albedo / colour also varies from ringlet to ringlet (this is what shows in the opaque B ring)
  float na = (vnoise(x * 83.0 + 2.0) - 0.5) * exp(-pow(83.0 * fw * 0.5, 2.0))
           + 0.7 * (vnoise(x * 290.0 + 9.0) - 0.5) * exp(-pow(290.0 * fw * 0.5, 2.0))
           + 0.6 * (vnoise(x * 23.0 + 5.0) - 0.5);
  alb *= 1.0 + 0.55 * na;
  // soft (sub-pixel resolved) outer edges
  tau *= smoothstep(1.239, 1.239 + max(fw, 0.002), x) * (1.0 - smoothstep(2.268 - max(fw, 0.002), 2.270, x));
  return max(tau, 0.0);
}
// single-scattering radiance of a ring layer (isotropic particles + opposition surge), premultiplied
vec4 ringShade(vec3 p, vec3 d, vec3 s, float fw, vec3 sun){
  vec3 alb; float tau = ringProps(length(p.xz), fw, alb);
  if (tau <= 0.0) return vec4(0.0);
  float mu = max(abs(d.y), 1e-3), mu0 = max(abs(s.y), 1e-3);
  float alpha = 1.0 - exp(-tau / mu);
  float shadow = ellHit(p, s) > 0.0 ? 0.0 : 1.0;          // Saturn's shadow on the rings
  float I;
  if ((s.y > 0.0) == (d.y < 0.0))                          // we see the sunlit face: diffuse reflection
    I = mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu + 1.0 / mu0)));
  else                                                     // unlit face: diffuse transmission
    I = abs(mu - mu0) < 1e-3 ? tau * mu0 / (mu * mu) * exp(-tau / mu) : mu0 / (mu - mu0) * (exp(-tau / mu) - exp(-tau / mu0));
  float ph = 1.0 + 0.8 * pow(max(dot(-d, s), 0.0), 30.0);  // opposition surge
  return vec4(sun * alb * I * shadow * ph * 1.6, alpha);
}
float sdHex(vec2 p, float r){
  const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
  p = abs(p); p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}
vec3 satAlbedo(vec3 nb){
  // body-fixed coordinates (rotating)
  float c = cos(uTime * SAT_OMEGA), sn = sin(uTime * SAT_OMEGA);
  vec3 q = vec3(c * nb.x - sn * nb.z, nb.y, sn * nb.x + c * nb.z);
  float lat = asin(clamp(q.y, -1.0, 1.0));
  float la = lat + 0.018 * fbm(q * vec3(3.0, 14.0, 3.0), 4) + 0.006 * snoise(q * vec3(9.0, 60.0, 9.0));
  float b = 0.5 + 0.5 * (0.5 * sin(la * 14.0 + 0.3) + 0.3 * sin(la * 31.0 + 1.1) + 0.2 * sin(la * 71.0 + 2.0));
  vec3 col = mix(vec3(0.74, 0.58, 0.40), vec3(0.93, 0.84, 0.65), b);
  col = mix(col, vec3(0.96, 0.89, 0.72), 0.7 * exp(-pow(la / 0.13, 2.0)));      // bright equatorial zone
  col = mix(col, vec3(0.66, 0.70, 0.74), 0.55 * smoothstep(0.9, 1.2, la));      // bluish northern polar cap (seasonal)
  col = mix(col, vec3(0.62, 0.53, 0.43), 0.45 * smoothstep(-1.0, -1.3, la));    // tan southern polar region
  // north polar hexagon: jet stream at ~78°N, inradius ≈ 12° of colatitude
  vec2 pp = acos(clamp(q.y, -1.0, 1.0)) * normalize(q.xz + 1e-6);
  float hx = sdHex(pp, 0.19);
  col = mix(col, vec3(0.45, 0.52, 0.60), 0.5 * (1.0 - smoothstep(-0.01, 0.01, hx)));
  col *= 1.0 - 0.35 * exp(-pow(hx / 0.012, 2.0)) * step(0.0, q.y);
  col *= 1.0 - 0.5 * exp(-dot(pp, pp) / 0.0012) * step(0.0, q.y);               // polar vortex
  return col * 0.55;   // geometric albedo ≈ 0.5
}
// Saturn layer along a straight ray in universe A: ring (premultiplied rgba, at tR) and planet (at tP)
void saturn(vec3 X, vec3 D, float path, float pix, out float tR, out vec4 ring, out float tP, out vec3 pc){
  tR = 1e9; tP = 1e9; ring = vec4(0.0); pc = vec3(0.0);
  vec3 o = toSat(X - uSatC) / uSatR, d = toSat(D), s = toSat(uSunDir);
  vec2 bs = sphereHit(o, d, vec3(0.0), 2.3);
  if (bs.x > bs.y || bs.y < 0.0) return;
  vec3 sun = bb(5772.0) * 1.15;
  float tp = ellHit(o, d);
  if (tp > 0.0){
    tP = tp * uSatR;
    vec3 p = o + d * tp;
    const float k = 1.0 / (1.0 - SAT_F);
    vec3 n = normalize(vec3(p.x, p.y * k * k, p.z));
    float mu0 = dot(n, s), mu = max(dot(n, -d), 0.0);
    vec3 alb = satAlbedo(n);
    float rs = 1.0;                                    // shadow of the rings on the globe
    if (abs(s.y) > 1e-4){
      float t = -p.y / s.y;
      if (t > 0.0){ vec3 q = p + s * t; vec3 a; rs = exp(-ringProps(length(q.xz), 0.003, a) / abs(s.y)); }
    }
    float lit = mu0 > 0.0 ? pow(mu0, 1.05) * pow(max(mu, 0.03), 0.05) : 0.0;   // Minnaert, k ≈ 1.05
    float haze = smoothstep(-0.12, 0.25, mu0);                                   // soft terminator (thick haze)
    pc = sun * alb * max(lit, 0.0) * haze * rs;
    // ringshine onto the side of the globe facing the lit face of the rings
    pc += sun * alb * 0.025 * max(n.y * sign(s.y), 0.0) * abs(s.y);
    pc = mix(pc, pc * vec3(0.9, 0.95, 1.08), pow(1.0 - mu, 3.0) * 0.6);         // bluish haze at the limb
  }
  if (abs(d.y) > 1e-6){
    float t = -o.y / d.y;
    if (t > 0.0){
      vec3 p = o + d * t;
      float x = length(p.xz);
      if (x > 1.2 && x < 2.3){
        float fw = (path + t * uSatR) * pix / (uSatR * max(abs(d.y), 0.08));
        ring = ringShade(p, d, s, fw, sun);
        if (ring.a > 0.0) tR = t * uSatR;
      }
    }
  }
}
vec3 sunDisk(vec3 d){
  float c = dot(d, uSunDir);
  // 9.6 AU: the real disk is 3.4' across (unresolved here); flux-preserving glare
  return bb(5772.0) * (smoothstep(0.999986, 0.999995, c) * 90.0 + pow(max(c, 0.0), 9000.0) * 3.0 + pow(max(c, 0.0), 400.0) * 0.06);
}

void main(){
  vec3 dir = camRay();
  float pix = 2.0 * uTanHalfFov / uRes.y;
  vec3 col = vec3(0.0); float trans = 1.0, path = 0.0, gCam = 1.0;
  int mode = 0;             // 0 straight · 1 wormhole · 2 black hole
  float uni = 1.0;
  vec3 X = uCamPos, D = dir;
  float wl = 0.0, wp = 0.0, wb = 0.0; vec3 e1 = vec3(1.0, 0.0, 0.0), e2 = vec3(0.0, 1.0, 0.0);
  vec3 bx = vec3(0.0), bv = vec3(0.0);
  bool gr = uLensing > 0.5;

  if (uCamMode == 2){
    // the camera's basis is its local orthonormal frame: radial ↔ s·ê_ℓ, tangential ↔ ê_θ, ê_φ
    vec3 dn = uCamS < 0.0 ? dir * MB : dir;
    float dr = dot(dn, uCamN); vec3 t = dn - dr * uCamN; float tl = length(t);
    e1 = uCamN; e2 = tl > 1e-6 ? t / tl : anyPerp(e1);
    wl = uCamL; wp = uCamS * dr; wb = rOf(uCamL) * tl;
    mode = 1; uni = uCamS;
  } else if (uCamMode == 3){
    // static observer inside Gargantua's region (like BH_MAIN): local direction → coordinate direction
    uni = -1.0;
    bx = (uCamPos - uBHC) / uBHrs;
    float lapse = gr ? sqrt(max(1.0 - 1.0 / length(bx), 1e-4)) : 1.0;
    gCam = 1.0 / lapse;
    bv = gr ? staticToCoord(bx, dir) : dir;
    mode = 2;
  } else {
    uni = uCamMode == 0 ? 1.0 : -1.0;
  }

  bool done = false;
  for (int seg = 0; seg < 6; seg++){
    if (mode == 1){
      uni = traceWH(wl, wp, wb, e1, e2, X, D, path);
      mode = 0; continue;
    }
    if (mode == 2){
      vec3 x0 = bx;
      int res = traceBH(bx, bv, gr, gCam, col, trans);
      path += length(bx - x0) * uBHrs;
      if (res != 0 || trans < 0.004){ done = true; break; }
      X = uBHC + bx * uBHrs; D = normalize(bv); uni = -1.0; mode = 0; continue;
    }
    // straight segment in universe 'uni'
    vec2 hw = sphereHit(X, D, vec3(0.0), uRW);
    float tW = (hw.x < hw.y && hw.x > 1e-3) ? hw.x : 1e9;
    if (uni > 0.0){
      if (uSatOn > 0.5){
        float tR, tP; vec4 ring; vec3 pc;
        saturn(X, D, path, pix, tR, ring, tP, pc);
        if (tR < min(tP, tW)){ col += trans * ring.rgb; trans *= 1.0 - ring.a; }
        if (tP < tW){ col += trans * pc; done = true; break; }
      }
      if (tW < 1e8){ X += D * tW; path += tW; enterWH(X, D, 1.0, wl, wp, wb, e1, e2); mode = 1; continue; }
      gUni = 1.0;
      col += trans * (sky(D, gCam) + sunDisk(D));
      done = true; break;
    } else {
      vec2 hb = sphereHit(X, D, uBHC, uBHR);
      float tB = (hb.x < hb.y && hb.x > 1e-3) ? hb.x : 1e9;
      if (tW < tB){ X += D * tW; path += tW; enterWH(X, D, -1.0, wl, wp, wb, e1, e2); mode = 1; continue; }
      if (tB < 1e8){ X += D * tB; path += tB; bx = (X - uBHC) / uBHrs; bv = D; mode = 2; continue; }
      gUni = -1.0;
      col += trans * sky(D, gCam);
      done = true; break;
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ---------------- CPU side: the same metric ---------------- */
function rOf(l, g) {
  const d = Math.abs(l) - g.a;
  if (d <= 0) return 1;
  const x = 2 * d / (Math.PI * g.M);
  return 1 + g.M * (x * Math.atan(x) - 0.5 * Math.log1p(x * x));
}
function geom(p) {
  const g = { a: p.length, W: p.width, M: p.width * W_TO_M };
  g.LW = g.a + L_PAD; g.RW = rOf(g.LW, g);
  return g;
}
// |ℓ| for a circumferential radius R ≥ ρ (r(ℓ) is monotonic for |ℓ| > a): bisection
function lOfR(R, g) {
  if (R <= 1) return g.a;
  let lo = g.a, hi = g.a + 2 * R + 20;
  for (let i = 0; i < 60; i++) { const m = 0.5 * (lo + hi); if (rOf(m, g) < R) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}
const MBv = v => v.set(v.x, v.y, -v.z);
const lookQuat = (() => {
  const m = new THREE.Matrix4(), up = new V3(0, 1, 0);
  return (eye, target) => new THREE.Quaternion().setFromRotationMatrix(m.lookAt(eye, target, up));
})();
const ease = k => k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;

// The panel only refreshes its widgets on host-driven syncs; presets change two sliders, so mirror
// the values into the existing widgets (display only — values live in ctx.params).
function refreshPanel(def, params) {
  try {
    const box = document.getElementById('sceneSection');
    if (!box) return;
    box.querySelectorAll('.ctl').forEach(c => {
      const lab = c.querySelector('.top span');
      const p = lab && def.params.find(q => q.label === lab.textContent || (q.en && q.en.label === lab.textContent));
      if (!p) return;
      if (p.type === 'range') {
        const inp = c.querySelector('input'), val = c.querySelector('.val');
        inp.value = params[p.key];
        if (val) val.textContent = p.fmt ? p.fmt(params[p.key]) : String(params[p.key]);
        inp.style.setProperty('--p', ((params[p.key] - p.min) / (p.max - p.min) * 100) + '%');
      } else if (p.type === 'seg') {
        c.querySelectorAll('.seg button').forEach((b, i) => b.classList.toggle('on', p.options[i] && p.options[i][0] === params[p.key]));
      }
    });
  } catch (e) { /* cosmetic only */ }
}
const PRESETS = { film: [0.01, 0.05], lens: [0.01, 0.6], long: [1.5, 0.3] };

const DEF = {
  id: 'wormhole',
  name: 'Hố sâu (Wormhole)',
  category: 'compact',
  order: 10,
  subtitle: 'Hố sâu xuyên được · metric “Dneg” của Interstellar',
  swatch: ['#bcd6ff', '#0b0716'],
  blurb: 'Một “quả cầu pha lê” treo cạnh Sao Thổ: nhìn vào nó là thấy bầu trời của một thiên hà khác, nơi có hố đen Gargantua. Ảnh được dựng bằng cách giải phương trình trắc địa null trong metric “Dneg” mà Kip Thorne và nhóm Double Negative dùng cho phim <i>Interstellar</i> (James, von Tunzelmann, Franklin & Thorne 2015). Để giữ cổ hố mở, thuyết tương đối rộng đòi hỏi “vật chất ngoại lai” có năng lượng âm — thứ chưa ai từng thấy ở quy mô vĩ mô.',
  intro: 'Nhấn <b>X</b> để bay xuyên qua hố sâu tới Gargantua (nhấn lại để về nhà). Thử <b>Độ rộng thấu kính W</b> lớn hơn để thấy vành Einstein phình ra.',
  facts: [
    ['Mô hình', 'Metric “Dneg”'],
    ['Bài báo', '<i>Am. J. Phys.</i> 83, 486 (2015)'],
    ['Thông số (phim)', 'a = 0,01ρ · W = 0,05ρ'],
    ['Vị trí (phim)', 'Quỹ đạo gần Sao Thổ'],
    ['Gargantua (phim)', '≈ 10⁸ M☉'],
    ['Vật chất ngoại lai', 'Bắt buộc (năng lượng âm)'],
    ['Vành Sao Thổ', '1,24–2,27 bán kính'],
  ],
  en: {
    name: 'Wormhole',
    subtitle: 'Traversable wormhole · the “Dneg” metric from Interstellar',
    blurb: 'A “crystal ball” hanging beside Saturn: look into it and you see the sky of another galaxy, home to the black hole Gargantua. The image is rendered by solving the null geodesic equations in the “Dneg” metric that Kip Thorne and the Double Negative team built for the film <i>Interstellar</i> (James, von Tunzelmann, Franklin & Thorne 2015). To hold the throat open, general relativity demands “exotic matter” with negative energy — something no one has ever seen on macroscopic scales.',
    intro: 'Press <b>X</b> to fly through the wormhole to Gargantua (press again to come home). Try a larger <b>Lensing width W</b> to watch the Einstein ring swell.',
    facts: [
      ['Model', '“Dneg” metric'],
      ['Paper', '<i>Am. J. Phys.</i> 83, 486 (2015)'],
      ['Parameters (film)', 'a = 0.01ρ · W = 0.05ρ'],
      ['Location (film)', 'Orbit near Saturn'],
      ['Gargantua (film)', '≈ 10⁸ M☉'],
      ['Exotic matter', 'Required (negative energy)'],
      ['Saturn’s rings', '1.24–2.27 radii'],
    ],
  },
  camera: { distance: 27.63, polar: 84.8, azimuth: 19.1, min: 1.3, max: 200, fov: 45 },
  exposure: 1.0,
  bloom: 0.5,
  usesBH: true,
  params: [
    { key: 'preset', label: 'Cấu hình', type: 'seg', default: 'film',
      options: [['film', 'Phim'], ['lens', 'Thấu kính rộng'], ['long', 'Hố dài'], ['custom', 'Tùy chỉnh']],
      note: 'Phim: a = 0,01ρ, W = 0,05ρ — lựa chọn của Christopher Nolan trong bài báo của Thorne.',
      en: { label: 'Configuration', options: ['Film', 'Wide lens', 'Long throat', 'Custom'], note: 'Film: a = 0.01ρ, W = 0.05ρ — Christopher Nolan’s choice in Thorne’s paper.' },
      onChange: (v, ctx) => { const pr = PRESETS[v]; if (pr) { ctx.params.length = pr[0]; ctx.params.width = pr[1]; refreshPanel(DEF, ctx.params); } } },
    { key: 'width', label: 'Độ rộng thấu kính W/ρ', type: 'range', min: 0.01, max: 1.5, step: 0.01, default: 0.05, fmt: v => v.toFixed(2),
      note: 'W = 1,43 M: vùng không gian cong quanh miệng hố. W lớn → vành Einstein rộng, ảnh vũ trụ bên kia bị bóp nhỏ.',
      en: { label: 'Lensing width W/ρ', note: 'W = 1.43 M: the region of curved space around the mouth. Larger W → a wider Einstein ring and a smaller image of the other universe.' },
      onChange: (v, ctx) => { ctx.params.preset = 'custom'; refreshPanel(DEF, ctx.params); } },
    { key: 'length', label: 'Nửa chiều dài cổ a/ρ', type: 'range', min: 0, max: 3, step: 0.01, default: 0.01, fmt: v => v.toFixed(2),
      note: 'Cổ hố là một hình trụ chu vi 2πρ dài 2a. Cổ càng dài, tia sáng càng quấn nhiều vòng → nhiều ảnh lặp của bầu trời.',
      en: { label: 'Throat half-length a/ρ', note: 'The throat is a cylinder of circumference 2πρ and length 2a. The longer the throat, the more times light rays wrap around it → more repeated images of the sky.' },
      onChange: (v, ctx) => { ctx.params.preset = 'custom'; refreshPanel(DEF, ctx.params); } },
    { key: 'saturn', label: 'Sao Thổ', type: 'toggle', default: true, en: { label: 'Saturn' } },
  ],
  actions: [
    { label: 'Xuyên qua', key: 'x', title: 'Bay xuyên qua hố sâu sang vũ trụ bên kia (và quay về)', run: ctx => ctx.scene._inst?.travel(ctx),
      en: { label: 'Go through', title: 'Fly through the wormhole to the other universe (and back)' } },
    { label: 'Nhìn về miệng hố', key: 'g', title: 'Bay tới gần miệng hố sâu trong vũ trụ hiện tại', run: ctx => ctx.scene._inst?.mouthView(ctx),
      en: { label: 'View the mouth', title: 'Fly up close to the wormhole mouth in the current universe' } },
  ],
  create(ctx) {
    const uniforms = {
      ...ctx.bhUniforms,
      uA: { value: 0.01 }, uM: { value: 0.05 * W_TO_M }, uLW: { value: L_PAD }, uRW: { value: 24 },
      uCamMode: { value: 0 }, uCamL: { value: 0 }, uCamS: { value: 1 }, uCamN: { value: new V3(0, 0, 1) },
      uBHC: { value: CB.clone() }, uBHrs: { value: RSB }, uBHR: { value: BHR },
      uSatC: { value: SAT_C.clone() }, uSatX: { value: new V3() }, uSatY: { value: new V3() }, uSatZ: { value: new V3() },
      uSatR: { value: SAT_R }, uSatOn: { value: 1 }, uSunDir: { value: new V3() },
    };
    // Saturn's spin axis and the Sun: rings open ~25° toward the default view, Sun ~18° above the ring plane
    const pole = new V3(0.30, 0.92, 0.22).normalize();
    const sx = new V3().crossVectors(pole, new V3(0, 0, 1)).normalize(), sz = new V3().crossVectors(sx, pole);
    uniforms.uSatX.value.copy(sx); uniforms.uSatY.value.copy(pole); uniforms.uSatZ.value.copy(sz);
    {
      const h = new V3(0.85, 0, 0.55); h.addScaledVector(pole, -h.dot(pole)).normalize();
      uniforms.uSunDir.value.copy(h).multiplyScalar(Math.cos(0.32)).addScaledVector(pole, Math.sin(0.32)).normalize();
    }
    const mkDefines = q => ({ MAX_STEPS: q.steps, STEP_K: q.k.toFixed(4), WH_STEPS: 260 });
    const material = new THREE.ShaderMaterial({
      uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, defines: mkDefines(ctx.quality),
      depthTest: false, depthWrite: false,
    });

    // camera state: which universe the camera's Cartesian position refers to, and scripted moves
    const st = { uni: 1, job: null, L: 0, mode: 0, debug: null, debugDone: false };
    const tmp = new V3(), tmpQ = new THREE.Quaternion();

    // position along the z axis for path coordinate s (= ℓ inside the region, extended outside)
    function axisPos(s, g, out) {
      const R = Math.abs(s) <= g.LW ? rOf(s, g) : g.RW + (Math.abs(s) - g.LW);
      return out.set(0, 0, s >= 0 ? R : -R);    // B side: M_B flips z
    }
    function finishJob(ctx, target, min, max) {
      st.job = null;
      const c = ctx.controls;
      c.target.copy(target); c.minDistance = min; c.maxDistance = max;
      ctx.camera.lookAt(c.target);
      c.enabled = true; c.update();
    }
    function flyJob(ctx, p1, look, dur, end, viaAway) {
      const cam = ctx.camera, p0 = cam.position.clone();
      // control point: midpoint, pushed away from Gargantua if the straight path would pass close to it
      const mid = p0.clone().lerp(p1, 0.5);
      if (st.uni < 0 && viaAway) {
        const seg = p1.clone().sub(p0), t = THREE.MathUtils.clamp(CB.clone().sub(p0).dot(seg) / seg.lengthSq(), 0, 1);
        const closest = p0.clone().addScaledVector(seg, t), off = closest.clone().sub(CB);
        if (off.length() < 18 * RSB) mid.add((off.lengthSq() > 1e-6 ? off.normalize() : new V3(0, 1, 0)).multiplyScalar(30 * RSB));
      }
      { // never brush past the throat itself
        const seg = p1.clone().sub(p0), t = THREE.MathUtils.clamp(-p0.dot(seg) / Math.max(seg.lengthSq(), 1e-9), 0, 1);
        const off = p0.clone().addScaledVector(seg, t);
        if (t > 0.02 && t < 0.98 && off.length() < 4) mid.add((off.lengthSq() > 1e-6 ? off.normalize() : new V3(0, 1, 0)).multiplyScalar(8));
      }
      st.job = { kind: 'fly', t: 0, dur, p0, mid, p1, q0: cam.quaternion.clone(), q1: lookQuat(p1, look), end };
      ctx.controls.enabled = false;
    }

    const inst = {
      material,
      setQuality(q) { Object.assign(material.defines, mkDefines(q)); material.needsUpdate = true; },
      enter(ctx) {
        st.uni = 1; st.job = null; st.debugDone = false;
        st.debug = null;
        try {
          const q = new URLSearchParams(location.search);
          if (q.has('snap') && q.get('wh')) st.debug = q.get('wh');   // test hook for tools/snap.sh
        } catch (e) { /* ignore */ }
      },
      exit(ctx) { if (st.job) { st.job = null; ctx.controls.enabled = true; } },
      travel(ctx) {
        if (st.job) return;
        const g = geom(ctx.params), from = st.uni;
        const p1 = new V3(0, 0, from * (g.RW + 10));
        // step 1: line up on the z axis, looking at the mouth
        flyJob(ctx, p1, new V3(0, 0, 0), THREE.MathUtils.clamp(ctx.camera.position.distanceTo(p1) / 14, 2.5, 7), { next: 'through', from }, true);
        ctx.toast(from > 0 ? t('Căn thẳng hàng với miệng hố sâu…', 'Lining up with the wormhole mouth…') : t('Quay về: hướng về miệng hố sâu…', 'Heading home: turning toward the wormhole mouth…'), 3000);
      },
      mouthView(ctx) {
        if (st.job) return;
        if (st.uni > 0) flyJob(ctx, START.clone(), new V3(), 4, { target: new V3(), min: 1.3, max: 200 }, false);
        else {
          flyJob(ctx, MOUTH_VIEW_B.clone(), new V3(), 6, { target: new V3(), min: 1.3, max: 115 }, true);
          ctx.toast(t('Nhìn ngược qua miệng hố: bên trong là Sao Thổ và Ngân Hà của chúng ta.', 'Looking back through the mouth: inside are Saturn and our Milky Way.'), 6000);
        }
      },
      update(dt, ctx) {
        const p = ctx.params, cam = ctx.camera, g = geom(p);
        let moved = false;

        if (st.debug && !st.debugDone) {
          st.debugDone = true;
          const [kind, a1, a2] = st.debug.split(':');
          if (kind === 'B') st.uni = -1;
          else if (kind === 'G' || kind === 'GB') {
            st.uni = -1;
            cam.position.copy(CB).add(ARRIVE);
            ctx.controls.target.copy(kind === 'G' ? CB : new V3());
            ctx.controls.maxDistance = 200;
            cam.lookAt(ctx.controls.target); ctx.controls.update(); moved = true;
          } else if (kind === 'S') {   // Saturn close-up
            cam.position.copy(SAT_C).add(new V3(+a1 || 22, 9, +a2 || 34));
            ctx.controls.target.copy(SAT_C); cam.lookAt(SAT_C); ctx.controls.update(); moved = true;
          } else if (kind === 'L') {
            st.job = { kind: 'hold', s: +a1 || 0, yaw: THREE.MathUtils.degToRad(+a2 || 0) };
          } else if (kind === 'T') {
            // run the fly-through state machine for a1 seconds (a2 = number of trips), in 50 ms steps
            let trips = (+a2 || 1) - 1;
            inst.travel(ctx);
            for (let i = 0, n = Math.round((+a1 || 10) / 0.05); i < n; i++) {
              if (!st.job && trips-- > 0) inst.travel(ctx);
              inst.update(0.05, ctx);
            }
            return;
          }
        }

        const job = st.job;
        if (job) {
          ctx.controls.enabled = false;
          if (job.kind === 'hold') {
            axisPos(job.s, g, cam.position); st.uni = job.s >= 0 ? 1 : -1;
            cam.quaternion.copy(lookQuat(cam.position, tmp.copy(cam.position).add(new V3(Math.sin(-job.yaw) , 0, -Math.cos(job.yaw)))));
            moved = true;
          } else if (job.kind === 'fly') {
            job.t += dt;
            const k = Math.min(1, job.t / job.dur), e = ease(k);
            const a = job.p0.clone().lerp(job.mid, e), b = job.mid.clone().lerp(job.p1, e);
            cam.position.copy(a.lerp(b, e));
            cam.quaternion.slerpQuaternions(job.q0, job.q1, e);
            moved = true;
            if (k >= 1) {
              const end = job.end;
              if (end.next === 'through') {
                st.job = { kind: 'through', from: end.from, s: end.from * (g.LW + 10), crossed: false, inThroat: false };
              } else finishJob(ctx, end.target, end.min, end.max);
            }
          } else if (job.kind === 'through') {
            // proper-distance coordinate s; slow near the throat, faster far away
            const speed = 0.35 + 0.5 * Math.abs(job.s);
            job.s -= job.from * speed * dt;
            const sEnd = -job.from * (g.LW + 10);
            const finished = job.from > 0 ? job.s <= sEnd : job.s >= sEnd;
            if (finished) job.s = sEnd;
            if (!job.inThroat && Math.abs(job.s) < g.a + 0.6) {
              job.inThroat = true;
              ctx.toast(t('Cổ hố sâu: mặt cầu có chu vi nhỏ nhất, 2πρ. Mọi hướng xung quanh bạn đều là “phía trước” lẫn “phía sau”.', 'The wormhole throat: the sphere of smallest circumference, 2πρ. Every direction around you is both “ahead” and “behind”.'), 5000);
              ctx.audio.tone(73.4, 4, 0.12, 0, 'sine');
            }
            axisPos(job.s, g, cam.position);
            st.uni = job.s >= 0 ? 1 : -1;
            cam.quaternion.copy(lookQuat(cam.position, tmp.set(0, 0, -job.from).add(cam.position)));
            moved = true;
            if (finished) {
              if (job.from > 0) {
                ctx.toast(t('Chào mừng tới vũ trụ bên kia. Phía trước là <b>Gargantua</b>.', 'Welcome to the other universe. Ahead lies <b>Gargantua</b>.'), 5000);
                st.job = { kind: 'fly', t: 0, dur: 7, p0: cam.position.clone(), mid: cam.position.clone().lerp(CB.clone().add(ARRIVE), 0.5).add(new V3(0, 3, 0)),
                  p1: CB.clone().add(ARRIVE), q0: cam.quaternion.clone(), q1: lookQuat(CB.clone().add(ARRIVE), CB), end: { target: CB.clone(), min: 1.6 * RSB, max: 115 } };
              } else {
                ctx.toast(t('Đã về nhà — Sao Thổ và Ngân Hà của chúng ta.', 'Home again — Saturn and our Milky Way.'), 4000);
                st.job = { kind: 'fly', t: 0, dur: 4.5, p0: cam.position.clone(), mid: cam.position.clone().lerp(START, 0.5).add(new V3(4, 2, 0)),
                  p1: START.clone(), q0: cam.quaternion.clone(), q1: lookQuat(START, new V3()), end: { target: new V3(), min: 1.3, max: 200 } };
              }
            }
          }
        }

        // where is the camera? (A/B Cartesian, inside the wormhole region, or near Gargantua)
        const X = cam.position, R = X.length();
        let mode, L, S = st.uni, N = uniforms.uCamN.value;
        const holdS = st.job && (st.job.kind === 'through' || st.job.kind === 'hold') ? st.job.s : null;
        if (holdS !== null && Math.abs(holdS) <= g.LW) {
          mode = 2; L = holdS; S = L >= 0 ? 1 : -1; N.set(0, 0, 1);
        } else if (R < g.RW) {
          mode = 2; L = st.uni * lOfR(R, g);
          N.copy(X).normalize(); if (st.uni < 0) MBv(N);
          if (R < 1) { X.multiplyScalar(1.0001 / Math.max(R, 1e-6)); }      // never inside the throat sphere
        } else {
          L = st.uni * (g.LW + R - g.RW);
          mode = st.uni > 0 ? 0 : (X.distanceTo(CB) < BHR ? 3 : 1);
        }
        st.mode = mode; st.L = L;

        uniforms.uA.value = g.a; uniforms.uM.value = g.M; uniforms.uLW.value = g.LW; uniforms.uRW.value = g.RW;
        uniforms.uCamMode.value = mode; uniforms.uCamL.value = L; uniforms.uCamS.value = S;
        uniforms.uSatOn.value = p.saturn ? 1 : 0;

        if (moved) {
          // the host filled the camera uniforms before this update; refresh them so ℓ and the pose agree
          cam.updateMatrixWorld();
          const e = cam.matrixWorld.elements;
          uniforms.uCamPos.value.copy(cam.position);
          uniforms.uCamRight.value.set(e[0], e[1], e[2]); uniforms.uCamUp.value.set(e[4], e[5], e[6]); uniforms.uCamFwd.value.set(-e[8], -e[9], -e[10]);
        }
      },
      hud(ctx) {
        const g = geom(ctx.params), L = st.L;
        const rows = [
          [t('Vũ trụ', 'Universe'), st.uni > 0 ? t('A · của chúng ta', 'A · ours') : 'B · Gargantua'],
          [t('Tọa độ riêng ℓ', 'Proper distance ℓ'), `${L >= 0 ? '+' : '−'}${Math.abs(L).toFixed(2)} ρ${st.mode === 2 ? '' : ' (≈)'}`],
          [t('Tới cổ hố sâu', 'To the throat'), `${Math.max(0, Math.abs(L) - g.a).toFixed(2)} ρ`],
          [t('Bán kính chu vi r(ℓ)', 'Circumferential radius r(ℓ)'), `${(Math.abs(L) <= g.LW ? rOf(L, g) : ctx.camera.position.length()).toFixed(2)} ρ`],
        ];
        if (st.uni < 0) rows.push([t('Tới Gargantua', 'To Gargantua'), `${(ctx.camera.position.distanceTo(CB) / RSB).toFixed(1)} r_s`]);
        return rows;
      },
    };
    DEF._inst = inst;
    return inst;
  },
};

export default [DEF];
