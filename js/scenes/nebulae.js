// Tinh vân — volumetric nebulae, ray-marched in the fullscreen pass.
// One scene unit = one light-year. Every scene marches a bounded volume front-to-back with
// emission + absorption, per-pixel jittered start, and early exit once the ray is opaque.
//
//   emission-nebula    Pillars of Creation (M16): UV-sculpted dust columns, ionisation fronts,
//                      Hα / [O III] / [S II] line emission, wavelength-dependent dust extinction.
//   planetary-nebula   Ring Nebula (M57): barrel-shaped shell with ionisation stratification,
//                      cometary knots, faint halo and a ~120 000 K white dwarf.
//   supernova-remnant  Crab Nebula (M1, SN 1054): Hα filament cage around a synchrotron
//                      pulsar-wind nebula with torus, wisps, jet and the flashing Crab pulsar.
//   dark-nebula        Horsehead (Barnard 33): cold dust silhouetted against IC 434.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

// ------------------------------------------------------------------------------------------
// shared GLSL helpers
// ------------------------------------------------------------------------------------------
const VOL_GLSL = /* glsl */`
#ifndef STEPS
#define STEPS 88
#endif
// interleaved gradient noise: a stable per-pixel offset that turns step banding into fine grain
float ign(){ return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))); }
float smin(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
// capsule whose radius tapers from ra (at a) to rb (at b) — approximate distance, good enough for density
float roundCone(vec3 p, vec3 a, vec3 b, float ra, float rb){
  vec3 ba = b - a, pa = p - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - mix(ra, rb, h);
}
// Point source seen through the camera: Gaussian core ~1.3 px wide (bloom makes the glare) plus a
// faint wide wing. v = star − camera, tc = distance along the ray of closest approach.
vec3 starPSF(vec3 rd, vec3 v, float tc, vec3 col, float lum){
  if (tc <= 0.0) return vec3(0.0);
  float a = length(v - rd * tc) / tc;
  float x = a / (2.0 * uTanHalfFov / uRes.y);
  return col * lum * (exp(-0.35 * x * x) + 0.012 / (1.0 + 0.015 * x * x));
}
// ray / ellipsoid (semi-axes ax, in the frame given by the rows of M): returns (tNear, tFar)
vec2 ellipsoidHit(vec3 ro, vec3 rd, mat3 M, vec3 ax){
  vec3 o = (M * ro) / ax, d = (M * rd) / ax;
  float a = dot(d, d), b = dot(o, d), c = dot(o, o) - 1.0, h = b * b - a * c;
  if (h < 0.0) return vec2(1e9, -1e9);
  h = sqrt(h); return vec2((-b - h) / a, (-b + h) / a);
}
float ridge(vec3 x){ float n = 1.0 - abs(snoise(x)); return n * n; }
`;

// Line colours in linear sRGB. Monochromatic lines lie outside the sRGB gamut; these are the
// nearest in-gamut hues (656 nm deep red, 501 nm teal-green, 486 nm sky blue, 469 nm blue).
const LINE_GLSL = /* glsl */`
const vec3 C_HA   = vec3(1.00, 0.020, 0.030);   // Hα 656.3 nm (and [N II] 658.4, [S II] 671.6)
const vec3 C_HB   = vec3(0.04, 0.36, 1.00);     // Hβ 486.1 nm
const vec3 C_OIII = vec3(0.02, 0.92, 0.55);     // [O III] 500.7 nm
const vec3 C_HEII = vec3(0.06, 0.20, 1.00);     // He II 468.6 nm
`;

// ------------------------------------------------------------------------------------------
// 1. Emission nebula — Pillars of Creation
// ------------------------------------------------------------------------------------------
const PILLARS_FRAG = COMMON_GLSL + VOL_GLSL + LINE_GLSL + /* glsl */`
uniform float uDust, uIon, uPal;      // dust column multiplier, ionising flux multiplier, palette 0/1/2
uniform vec3 uL;                      // unit vector toward the NGC 6611 cluster (the ionising O stars)
const vec3 C0 = vec3(0.0, -2.0, 0.0); // bounding ellipsoid
const vec3 BAX = vec3(7.0, 9.0, 6.0);

// Three columns rising out of the parent molecular cloud below the frame, heads pointing at the
// cluster. The tallest (left) pillar is ~4 ly from the frame edge to its head, as measured in M16.
float base(vec3 p){
  float d = roundCone(p, vec3(-2.9, -11.0, -0.9), vec3(-1.55, 2.3, 0.15), 1.75, 0.6);
  d = smin(d, length(p - vec3(-1.45, 2.3, 0.2)) - 0.74, 0.45);                // bulbous head
  d = smin(d, roundCone(p, vec3(-1.3, 2.5, 0.2), vec3(-0.75, 3.3, 0.35), 0.36, 0.2), 0.3); // finger with EGGs
  d = smin(d, roundCone(p, vec3(0.6, -11.0, -1.8), vec3(0.95, -0.45, -1.0), 1.5, 0.5), 0.6);
  d = smin(d, length(p - vec3(1.0, -0.6, -0.95)) - 0.58, 0.35);
  d = smin(d, roundCone(p, vec3(3.4, -11.0, -0.6), vec3(2.65, -2.4, 0.0), 1.3, 0.42), 0.6);
  return d;
}
float field(vec3 p){
  float d = base(p);
  if (d > 1.8) return d;
  // lumpy dust plus erosion streaks combed along the photo-evaporation flow (mostly vertical)
  float st = snoise(vec3(p.x * 2.4, p.y * 0.5, p.z * 2.4) + 7.0);
  d += 0.5 * fbm(p * 0.7 + 3.1, 3) + 0.12 * st;
  if (abs(d) < 0.6) d += 0.13 * fbm(p * 3.6, 2);       // fine erosion detail only near the skin
  return d;
}
// cheap version (first noise octave only) for the UV-facing derivative
float fieldLite(vec3 p){ return base(p) + 0.29 * snoise(p * 0.7 + 3.1); }

// The H II region (≈ 70 × 55 ly) is far larger than the pillar field: beyond the marched volume
// its glow fills the sky, brightest and most highly ionised toward the cluster, with dust lanes.
vec4 backdrop(vec3 d, out float lane){
  float n = fbm(d * 2.1 + 4.0, 4);
  float up = 0.5 + 0.5 * dot(d, uL);
  float neb = smoothstep(-0.45, 0.55, n) * (0.25 + 0.75 * up * up);
  lane = smoothstep(0.05, 0.55, fbm(d * 3.3 + 9.0, 3)) * smoothstep(0.5, -0.4, d.y + 0.3 * n);
  return vec4(0.12, 1.0, 0.55 * up * up, 1.0 / 2.86) * neb * 0.075 * uIon;
}

#define NSTAR 10
const vec3 SP[NSTAR] = vec3[NSTAR](
  vec3(-0.85, 3.2, 0.3), vec3(-1.45, 2.15, 0.3), vec3(1.05, -0.5, -0.9), vec3(2.6, -2.5, 0.05),
  vec3(3.6, 1.6, 3.0), vec3(-4.2, -0.8, 2.6), vec3(1.6, 3.9, -2.2), vec3(-3.6, 4.1, -3.2),
  vec3(4.6, -3.3, -2.0), vec3(-0.2, 1.2, 4.4));
// temperature (K), brightness, embedded? (1 = young star still buried in a pillar)
const vec3 SD[NSTAR] = vec3[NSTAR](
  vec3(4200.0, 0.5, 1.0), vec3(3300.0, 0.7, 1.0), vec3(3800.0, 0.5, 1.0), vec3(3100.0, 0.45, 1.0),
  vec3(9500.0, 1.1, 0.0), vec3(5800.0, 0.7, 0.0), vec3(22000.0, 1.0, 0.0), vec3(30000.0, 1.4, 0.0),
  vec3(5000.0, 0.5, 0.0), vec3(7500.0, 0.8, 0.0));

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  bool ir = uPal > 1.5;
  // A_λ/A_V (Cardelli R_V = 3.1) at [S II] 672, Hα 656, [O III] 501, Hβ 486 nm; near-IR ≈ A_K/A_V
  vec4 kRel = ir ? vec4(0.11) : vec4(0.79, 0.82, 1.13, 1.17);
  float kV = 7.0 * uDust;                       // visual extinction per ly inside a pillar (τ_V)
  vec4 T = vec4(1.0), I = vec4(0.0);            // transmittance and line radiance (S II, Hα, O III, Hβ)
  vec3 extra = vec3(0.0);                       // broadband: dust light, stars

  float tcs[NSTAR]; vec3 psf[NSTAR];
  for (int k = 0; k < NSTAR; k++){
    vec3 v = SP[k] - ro; tcs[k] = dot(v, rd);
    float lum = SD[k].y * clamp(100.0 / max(dot(v, v), 1e-3), 0.3, 40.0);
    if (ir) lum *= SD[k].z > 0.5 ? 3.0 : 1.4;    // young stars are bright in the infrared
    psf[k] = starPSF(rd, v, tcs[k], bb(SD[k].x), lum * 6.0);
  }
  mat3 Id = mat3(1.0);
  vec2 h = ellipsoidHit(ro - C0, rd, Id, BAX);
  float t = max(h.x, 0.0), tEnd = h.y;
  for (int k = 0; k < NSTAR; k++) if (tcs[k] < t || h.x > h.y) extra += psf[k];

  if (h.x < h.y && tEnd > 0.0){
    t += 0.12 * ign();
    for (int i = 0; i < STEPS; i++){
      if (t >= tEnd || max(max(T.x, T.y), max(T.z, T.w)) < 0.01) break;
      vec3 p = ro + rd * t;
      float d = field(p);
      float dt = clamp(abs(d) * 0.45, 0.045, 0.4);
      float se = length((p - C0) / BAX);
      float edge = smoothstep(1.0, 0.7, se);

      // --- UV illumination from the cluster -----------------------------------------------
      float nl = 0.0, sh = 1.0;
      if (d < 2.4) sh = smoothstep(-0.3, 0.45, min(base(p + uL * 1.0), base(p + uL * 2.6)));   // shadowed by a head?
      if (d < 0.9) nl = clamp((fieldLite(p + uL * 0.3) - fieldLite(p)) / 0.3, -1.0, 1.0);        // ≈ n·L (only matters near the skin)
      float uv = uIon * sh * smoothstep(-0.2, 0.9, nl);
      // ionisation front: photo-evaporated gas streams off the lit skin (Hα), with a thin
      // partially-ionised layer right at the surface where S⁺ survives ([S II])
      float skin = exp(-max(d, 0.0) / 0.085) * smoothstep(-0.2, 0.0, d);
      float front = skin * uv;
      float sFront = exp(-abs(d + 0.01) / 0.07) * uv;
      // diffuse H II ionised by the cluster: emission ∝ n_e n_p, clumpy, shadowed behind the heads
      float gas = smoothstep(-0.3, 0.7, fbm(p * 0.32 + 11.0, 2));
      gas *= edge * smoothstep(-0.1, 0.6, d) * (0.3 + 0.7 * sh) * (0.25 + 0.75 * smoothstep(3.0, -2.5, p.z));   // most ionised gas fills the cavity behind the pillars
      float ox = smoothstep(0.3, 2.5, d) * (0.4 + 0.6 * smoothstep(-6.0, 5.0, dot(p, uL)));   // O²⁺ needs the hardest photons: far from the fronts, near the stars
      vec4 em;
      em.y = uIon * gas * 0.035 + 2.6 * front;                       // Hα
      em.x = uIon * gas * 0.005 + 1.25 * sFront;                     // [S II]
      em.z = uIon * gas * 0.035 * ox + 0.12 * front;                 // [O III]
      em.w = em.y / 2.86;                                           // Hβ: case-B Balmer decrement

      // --- dust ---------------------------------------------------------------------------
      float rho = (smoothstep(0.08, -0.22, d) + 0.03 * gas) * smoothstep(1.0, 0.86, se);
      float tauV = kV * rho * dt;
      vec4 att = exp(-tauV * kRel);
      vec3 Trgb = vec3(T.y, T.z, T.w);
      float lit = 0.03 + uv * smoothstep(-0.6, 0.0, d) * max(nl, 0.0);
      if (ir){
        // warm dust + PAH emission: the pillars glow semi-transparent orange (JWST NIRCam)
        extra += Trgb * (1.0 - att.y) * vec3(1.0, 0.42, 0.18) * (0.06 + 0.9 * lit) * 2.2;
      } else {
        // dust scatters the nebula's own light and the cluster's starlight (albedo ~0.5)
        vec3 Lin = vec3(0.30, 0.13, 0.08) * (0.14 + 0.1 * uIon) + bb(40000.0) * 0.5 * lit * uIon;
        extra += Trgb * (1.0 - att.y) * Lin * vec3(0.95, 0.7, 0.5) * 0.5;
      }
      I += T * em * dt;
      T *= att;
      float t1 = t + dt;
      for (int k = 0; k < NSTAR; k++) if (tcs[k] >= t && tcs[k] < t1) extra += psf[k] * vec3(T.y, T.z, T.w);
      t = t1;
    }
    for (int k = 0; k < NSTAR; k++) if (tcs[k] >= t) extra += psf[k] * vec3(T.y, T.z, T.w);
  }

  // what lies beyond the marched volume: the rest of the H II region, dust lanes, the Milky Way
  float lane;
  vec4 bd = backdrop(rd, lane);
  float lt = 1.0 - (ir ? 0.2 : 0.75) * lane * min(uDust, 1.0);
  if (ir) bd.x = 0.0;
  I += T * bd * clamp(lt, 0.0, 1.0);
  vec3 Tb = vec3(T.y, T.z, T.w) * clamp(lt, 0.05, 1.0);

  vec3 col;
  if (uPal < 0.5)      col = I.x * C_HA + I.y * C_HA + I.z * C_OIII + I.w * C_HB;            // natural colour
  else if (uPal < 1.5) col = I.x * vec3(1.0, 0.16, 0.03) * 2.6 + I.y * vec3(0.3, 1.0, 0.16) * 0.55
                           + I.z * vec3(0.06, 0.4, 1.0) * 2.6;                                // SHO: S II→R, Hα→G, O III→B (each channel stretched)
  else                 col = I.x * vec3(1.0, 0.16, 0.08) * 1.4                                  // H₂ 2.12 µm at the fronts
                           + (I.y + I.z * 0.5) * vec3(0.22, 0.42, 0.95) * 0.55;               // Paα/Brγ recombination glow
  col += extra;
  col += sky(rd, 1.0) * 0.6 * Tb;
  gl_FragColor = vec4(col, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// 2. Planetary nebula — the Ring Nebula
// ------------------------------------------------------------------------------------------
const RING_FRAG = COMMON_GLSL + VOL_GLSL + LINE_GLSL + /* glsl */`
uniform float uW, uKnots, uSc, uBright;   // shell thickness (relative), knot density, size scale, surface-brightness scale
uniform mat3 uFrame;                       // rows: nebula frame; z = symmetry axis of the barrel

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  float A = 0.55 * uSc, C = 0.95 * uSc;     // barrel half-width / half-length, ly
  vec3 I = vec3(0.0), T = vec3(1.0);

  // the central star: a ~120 000 K white dwarf, its light ionises the whole shell
  vec3 v = -ro; float tcS = dot(v, rd);
  vec3 star = starPSF(rd, v, tcS, bb(100000.0) * vec3(0.92, 0.96, 1.0), 9.0 * clamp(7.0 / max(dot(v, v), 1e-3), 0.4, 30.0));
  bool starDone = false;

  // faint outer halo (ejected earlier on the AGB): Hα loops out to ~3× the ring. Optically thin and
  // smooth, so instead of marching it: integrate its radial profile along the ray (cheap, no noise)
  // and modulate by noise at the ray's closest approach to the star.
  float halo = 0.0;
  {
    vec3 qo = uFrame * ro / uSc, qd = uFrame * rd;
    float tc = -dot(qo, qd);
    vec3 qc = qo + qd * tc;
    float b2 = dot(qc, qc) / 0.3025;                    // (impact parameter / 0.55)²
    if (b2 < 13.7){
      float hl = sqrt(13.7 - b2);                       // half-chord inside rs = 3.7, in ring radii
      float acc = 0.0;
      for (int k = 0; k < 12; k++){
        float x = (float(k) + 0.5) / 12.0 * hl;
        float rs = sqrt(b2 + x * x);
        acc += smoothstep(1.35, 1.9, rs) * smoothstep(3.7, 2.3, rs);
      }
      float tex = 0.41 * (0.8 + 0.4 * snoise(qc * 2.0 + 5.0));     // ≈ the along-ray average of the marched loop texture
      halo = acc / 12.0 * 2.0 * hl * 0.55 * tex * 0.0045 * uBright;   // ∫ emissivity dl (both halves, world ly)
    }
  }
  vec3 haloCol = (1.8 * C_HA + C_HB / 2.86) * halo;
  I += 0.5 * haloCol;                                   // near half of the halo

  // the bright shell lives inside s < 1 + 3.4 w: march only its bounding ellipsoid
  float Sm = 1.0 + 3.4 * uW;
  vec2 h = ellipsoidHit(ro / uSc, rd, uFrame, vec3(0.62 * Sm, 0.5 * Sm, 0.95 * sqrt(Sm)) * 1.15);
  h *= uSc;
  if (h.x < h.y && h.y > 0.0){
    float t = max(h.x, 0.0) + 0.02 * uSc * ign();
    for (int i = 0; i < STEPS; i++){
      if (t >= h.y || max(T.x, max(T.y, T.z)) < 0.01) break;
      vec3 q = uFrame * (ro + rd * t) / uSc;          // co-moving coordinates: the pattern expands homologously
      float rr = length(q.xy / vec2(0.62, 0.5));                       // M57's ring is elliptical, 1.4′ × 1.0′
      float zz = abs(q.z) / 0.95;
      float s = sqrt(rr * rr + zz * zz * zz * zz);                    // super-ellipsoid: straight barrel walls, rounded ends
      float cth = abs(q.z / 0.95) / max(s, 1e-4);     // |cos| of latitude on the ellipsoid
      float u = (s - 1.0) / uW;
      float dt = mix(0.019, 0.15, smoothstep(2.2, 5.0, abs(u))) * uSc;

      // equatorial band dense (ionisation-bounded), polar caps thin (matter-bounded): this is
      // why a barrel seen end-on looks like a ring, and why the inside of M57 is blue-green
      float band = 1.0 - 0.94 * smoothstep(0.3, 0.68, cth);
      float he2 = 0.0, o3 = 0.0, ha = 0.0, n2 = 0.0, ext = 0.0;
      if (abs(u) < 3.4){
        // --- inside the shell: full detail ---
        float clump = 0.5 + 0.9 * max(fbm(q * 5.5, 2), -0.4) + 0.25 * snoise(vec3(q.xy * 22.0, q.z * 6.0));   // fine radial filaments
        float shell = exp(-u * u * 0.5) * clump;
        // ionisation stratification: high-ionisation ions sit nearest the star
        he2 = (exp(-pow((u + 1.3) / 0.75, 2.0)) * 0.55 * shell + 0.02 * smoothstep(1.0, 0.1, s) * clump) * mix(0.3, 1.0, band);
        o3  = exp(-pow((u + 0.5) / 0.85, 2.0)) * shell * mix(0.16, 1.0, band);
        ha  = exp(-pow((u - 0.1) / 0.85, 2.0)) * shell * band;
        n2  = exp(-pow((u - 0.75) / 0.75, 2.0)) * shell * band * band;

        // cometary knots: dense neutral globules at the ionisation front, tails pointing away
        // from the star (worley cells compressed radially → elongated radial spokes)
        if (uKnots > 0.01 && u > -0.5 && u < 2.6){
          // patches where knots formed (smooth selection; more of the front fragments as uKnots ↑)
          float sel = smoothstep(0.62 - uKnots * 0.9, 0.82 - uKnots * 0.9, 0.5 + 0.5 * snoise(q * 6.0 + 2.0));
          float m = smoothstep(-0.5, 0.4, u) * smoothstep(2.6, 1.2, u) * band * sel;
          if (m > 0.02){
            vec3 n = normalize(q);
            vec3 pk = q * 90.0; pk -= n * dot(pk, n) * 0.88;
            float w = worley(pk).x;
            float core = smoothstep(0.16, 0.03, w) * m;
            float rim = (smoothstep(0.3, 0.18, w) * m - core) * 0.08;
            ext = 22.0 * core;
            ha += rim; n2 += rim;
          }
        }
      } else if (s < 1.0){
        // --- central cavity: faint He II glow only ---
        he2 = 0.02 * smoothstep(1.0, 0.1, s) * (0.75 + 0.35 * snoise(q * 5.5)) * mix(0.3, 1.0, band);
      }
      float k = uBright / uSc;  // per-unit-length emissivity in world units
      vec3 em = (he2 * C_HEII * 0.9 + o3 * C_OIII * 0.85 + (ha + 1.4 * n2) * C_HA * 1.0 + ha / 2.86 * C_HB) * k;
      if (!starDone && tcS < t){ I += star * T; starDone = true; }
      I += T * em * dt;
      T *= exp(-ext * dt / uSc * vec3(0.82, 1.05, 1.2));
      t += dt;
    }
  }
  if (!starDone) I += star * T;
  I += 0.5 * haloCol * T;                               // far half of the halo, seen through the shell
  if (max(T.x, max(T.y, T.z)) > 0.004) I += sky(rd, 1.0) * 0.6 * T;
  gl_FragColor = vec4(I, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// 3. Supernova remnant — the Crab Nebula
// ------------------------------------------------------------------------------------------
const CRAB_FRAG = COMMON_GLSL + VOL_GLSL + LINE_GLSL + /* glsl */`
uniform float uSc, uPulse, uSyn, uFil, uWisp, uFade;
uniform mat3 uFrame;          // rows: remnant frame, x = long axis
uniform vec3 uSpin;           // pulsar spin axis (torus normal, jet direction)
const vec3 AX = vec3(5.6, 3.7, 3.6);   // semi-axes in ly today (≈ 11 × 7 ly)

// inner pulsar-wind nebula: equatorial torus, moving wisps and a polar jet (Chandra/HST),
// synchrotron light from the relativistic wind. Scaled to its size today; does not grow with the shell.
vec3 pwn(vec3 p){
  float hz = dot(p, uSpin);
  float rr = length(p - uSpin * hz);
  float rT = 0.42;
  float tor = exp(-pow((rr - rT) / 0.11, 2.0) - pow(hz / 0.07, 2.0));
  float wisp = 0.0;
  for (int k = 0; k < 3; k++){
    float rw = 0.12 + fract(uWisp + float(k) / 3.0) * 0.45;
    wisp += exp(-pow((rr - rw) / 0.035, 2.0) - pow(hz / 0.05, 2.0)) * (1.0 - fract(uWisp + float(k) / 3.0));
  }
  float kink = 0.035 * sin(hz * 9.0 + uWisp * 6.2832) * sin(hz * 4.1 + 1.3);   // the jet's slow kinking
  float jr = length(p - uSpin * hz + kink * cross(uSpin, vec3(0.0, 1.0, 0.0)));
  float jet = exp(-pow(jr / (0.03 + 0.06 * max(hz, 0.0)), 2.0)) * smoothstep(0.05, 0.2, hz) * smoothstep(1.3, 0.5, hz);
  float cjet = exp(-pow(jr / 0.04, 2.0)) * smoothstep(-0.05, -0.15, hz) * smoothstep(-0.55, -0.25, hz) * 0.35;
  float knot = exp(-dot(p - uSpin * 0.03, p - uSpin * 0.03) / 0.0015) * 1.5;   // the inner knot just off the pulsar
  return vec3(0.62, 0.78, 1.0) * (tor * 1.4 + wisp * 1.1 + jet * 1.5 + cjet + knot) * 2.2;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  vec3 ax = AX * uSc;
  vec3 I = vec3(0.0), T = vec3(1.0);
  float bright = uFade / max(uSc, 0.25);         // surface brightness falls as the cloud thins out

  // the Crab pulsar: a neutron star spinning 29.6 times a second, two pulses per turn
  vec3 v = -ro; float tcS = dot(v, rd);
  vec3 star = starPSF(rd, v, tcS, vec3(0.75, 0.85, 1.0), (1.2 + 22.0 * uPulse) * clamp(40.0 / max(dot(v, v), 1e-3), 0.5, 40.0));
  bool starDone = false;

  vec2 h = ellipsoidHit(ro, rd, uFrame, ax * 1.06);
  if (h.x < h.y && h.y > 0.0){
    float t = max(h.x, 0.0);
    float span = h.y - t;
    float dt0 = max(span / float(STEPS - 8), 0.04);
    t += dt0 * ign();
    for (int i = 0; i < STEPS; i++){
      if (t >= h.y || max(T.x, max(T.y, T.z)) < 0.01) break;
      vec3 p = ro + rd * t;
      vec3 e = (uFrame * p) / ax;                       // |e| = 1 on the outer boundary
      float s = length(e);
      // low-frequency warp field: shapes the ragged outline and bends the filaments (RT fingers)
      vec3 wv = vec3(0.0);
      if (s > 0.2){
        wv = vec3(snoise(e * 1.7 + 1.0), snoise(e * 1.7 + 7.0), snoise(e * 1.7 + 13.0));
        s *= 1.0 + 0.13 * wv.x;                         // ragged outline
      }
      float dt = dt0 * mix(0.6, 1.0, smoothstep(0.6, 0.3, abs(s - 0.8)));

      vec3 em = vec3(0.0);
      float fil = 0.0;
      // filaments: thin sheets of ridged noise, domain-warped like Rayleigh–Taylor fingers
      // (the pulsar-wind bubble pushing into heavier ejecta) on a thick outer shell
      float shellM = smoothstep(0.25, 0.7, s) * (1.0 - smoothstep(0.88, 1.0, s));
      if (shellM > 0.0){
        vec3 wq = e * 2.3 + 0.38 * wv;
        float rdg = ridge(wq) * 0.6 + ridge(wq * 2.4 + 4.0) * 0.4;
        fil = pow(rdg, 8.0) * shellM * uFil;
        if (fil > 2e-3){
          // composition varies: Hα/[N II]/[S II] red, [O I]+He I orange-yellow, [O III] green
          float hue = snoise(e * 1.4 + 20.0);
          vec3 fcol = mix(vec3(1.0, 0.13, 0.05), vec3(1.0, 0.45, 0.1), smoothstep(0.1, 0.7, hue));
          fcol = mix(fcol, vec3(0.55, 1.0, 0.45), smoothstep(0.45, 0.85, -hue) * 0.6);
          em = fil * fcol * 2.6;
        }
      }
      // synchrotron nebula: relativistic electrons spiralling in the pulsar's magnetic field,
      // brightest near the pulsar (fresh electrons), fibrous along the field lines
      float syn = exp(-s * s * 3.0) * smoothstep(0.98, 0.55, s) * uSyn;
      if (syn > 1e-4) em += syn * (0.5 + 0.5 * ridge(e * 4.5 + 30.0)) * vec3(0.46, 0.64, 1.0) * 0.24;
      if (!starDone && tcS < t){ I += star * T; starDone = true; }
      I += T * em * bright * dt;
      T *= exp(-fil * 1.5 * dt * vec3(0.8, 1.0, 1.2));    // a little dust condensed in the filaments
      t += dt;
    }
  }
  if (!starDone) I += star * T;

  // inner PWN sub-march: features are 0.03–0.1 ly, finer than the main step, and optically thin
  vec2 hp = sphereHit(ro, rd, vec3(0.0), 1.35 * min(uSc, 1.0));
  if (hp.x < hp.y && hp.y > 0.0){
    float t = max(hp.x, 0.0), dt = (hp.y - t) / 30.0;
    t += dt * ign();
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 30; i++){ acc += pwn((ro + rd * t) / min(uSc, 1.0)); t += dt; }
    I += acc * dt / min(uSc, 1.0) * uSyn * uFade * T;
  }
  if (max(T.x, max(T.y, T.z)) > 0.004) I += sky(rd, 1.0) * 0.6 * T;
  gl_FragColor = vec4(I, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// 4. Dark nebula — the Horsehead (Barnard 33) against IC 434
// ------------------------------------------------------------------------------------------
const HORSE_FRAG = COMMON_GLSL + VOL_GLSL + LINE_GLSL + /* glsl */`
uniform float uDust, uIon, uIR;
const vec3 LS = normalize(vec3(0.15, 1.0, -0.25));    // toward σ Orionis, ~11 ly above the head

// The horse's head is a pillar of the L1630 cloud: neck rising from the cloud edge, head tilted
// with the muzzle to the left, mane trailing right. Profile modelled in xy, ~1.2 ly thick in z.
// approximate distance to an ellipsoid with semi-axes r, rotated by ang in the xy plane
float ell(vec3 p, vec3 c, vec3 r, float ang){
  p -= c; float cs = cos(ang), sn = sin(ang);
  p.xy = vec2(cs * p.x + sn * p.y, -sn * p.x + cs * p.y);
  return (length(p / r) - 1.0) * min(r.x, min(r.y, r.z));
}
float horse(vec3 p){
  float d = roundCone(p, vec3(0.4, 1.0, 0.0), vec3(-1.35, -0.55, 0.0), 0.8, 0.5);             // forehead → blunt muzzle, pointing down-left
  d = smin(d, ell(p, vec3(-0.2, -0.2, 0.0), vec3(0.66, 0.52, 0.5), 0.35), 0.3);                // jowl at the back of the jaw
  d = smin(d, roundCone(p, vec3(0.5, 1.5, 0.0), vec3(0.72, 2.2, 0.05), 0.24, 0.06), 0.15);    // ear
  d = smin(d, roundCone(p, vec3(0.6, 0.3, 0.0), vec3(1.5, -4.6, 0.0), 0.5, 1.0), 0.25);      // neck, leaning back into the cloud
  d = smin(d, roundCone(p, vec3(0.95, 0.95, 0.0), vec3(1.9, -3.0, 0.0), 0.32, 0.6), 0.4);    // mane / back of the neck
  d = smin(d, max(p.y + 3.6 + 0.3 * sin(p.x * 0.7), abs(p.z + 1.0) - 2.6), 1.2);               // L1630 cloud: a slab whose top edge faces σ Ori
  return d;
}
float field(vec3 p){
  float d = horse(p);
  if (d > 1.5) return d;
  return d + 0.2 * fbm(p * 1.4 + 2.0, 3) + (abs(d) < 0.5 ? 0.06 * snoise(p * 5.0) : 0.0);
}

// IC 434: an ionised sheet *behind* the horse (z ≈ −1 … −4.5), lit from above by σ Ori; its vertical
// striations trace gas photo-evaporating off the cloud toward the star. Optically thin, so it is
// not marched: its emission measure is sampled on two planes and weighted by the path length.
float ic434(vec3 p){
  float stri = 0.6 + 0.4 * snoise(vec3(p.x * 2.2, p.y * 0.18, p.z * 0.8)) + 0.25 * fbm(p * 0.6, 2);
  float ion = smoothstep(-4.0, -2.8, p.y) * (0.45 + 0.55 * exp(-(p.y + 2.8) / 4.0));   // brightest just above the cloud's ionisation front
  return max(stri, 0.0) * ion * smoothstep(8.2, 3.0, length(p - vec3(0.0, -0.8, 0.0)));
}
const float ZS0 = -1.3, ZS1 = -2.5, ZS2 = -3.7;     // sample planes inside the sheet

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  vec3 I = vec3(0.0), T = vec3(1.0);
  vec3 kRel = uIR > 0.5 ? vec3(0.11) : vec3(0.82, 1.05, 1.2);
  vec3 gasCol = uIR > 0.5 ? vec3(0.3, 0.45, 0.95) * 0.12 : (C_HA + C_HB / 2.86 * 0.6) * 0.28 + C_OIII * 0.02;
  // sheet samples: ray parameter and emission (path through the ~2.9 ly sheet ≈ 1.45 ly per plane / |cos|)
  float ts0 = -1.0, ts1 = -1.0, ts2 = -1.0; vec3 g0 = vec3(0.0), g1 = vec3(0.0), g2 = vec3(0.0);
  if (abs(rd.z) > 1e-3){
    float w = 0.97 * uIon / max(abs(rd.z), 0.25);
    ts0 = (ZS0 - ro.z) / rd.z; ts1 = (ZS1 - ro.z) / rd.z; ts2 = (ZS2 - ro.z) / rd.z;
    if (ts0 > 0.0) g0 = gasCol * ic434(ro + rd * ts0) * w; else ts0 = -1.0;
    if (ts1 > 0.0) g1 = gasCol * ic434(ro + rd * ts1) * w; else ts1 = -1.0;
    if (ts2 > 0.0) g2 = gasCol * ic434(ro + rd * ts2) * w; else ts2 = -1.0;
  }
  // dust lives in the slab z ∈ [−3.8, 1.9], below y = 2.8, inside r = 9: march only that box
  vec3 bmin = vec3(-9.0, -9.8, -3.8), bmax = vec3(9.0, 2.8, 1.9);
  vec3 ib = 1.0 / (rd + vec3(equal(rd, vec3(0.0))) * 1e-6);
  vec3 ta = (bmin - ro) * ib, tb = (bmax - ro) * ib;
  vec3 t1v = min(ta, tb), t2v = max(ta, tb);
  float tn = max(max(t1v.x, t1v.y), t1v.z), tf = min(min(t2v.x, t2v.y), t2v.z);
  vec2 hs = sphereHit(ro, rd, vec3(0.0, -0.8, 0.0), 9.0);
  tn = max(tn, hs.x); tf = min(tf, hs.y);
  bool s0 = ts0 < 0.0, s1 = ts1 < 0.0, s2 = ts2 < 0.0;   // already added?
  if (tn < tf && tf > 0.0){
    float t = max(tn, 0.0) + 0.1 * ign();
    if (!s0 && ts0 < t){ I += g0; s0 = true; }
    if (!s1 && ts1 < t){ I += g1; s1 = true; }
    if (!s2 && ts2 < t){ I += g2; s2 = true; }
    for (int i = 0; i < STEPS; i++){
      if (t >= tf || max(T.x, max(T.y, T.z)) < 0.01) break;
      vec3 p = ro + rd * t;
      float d = field(p);
      // empty-space skipping: far from dust the field is a (conservative) distance bound
      float dt = d > 0.3 ? max(d * 0.6, 0.12) : clamp(abs(d) * 0.45, 0.04, 0.3);
      if (d < 0.45){
        float r = length(p - vec3(0.0, -0.8, 0.0));
        // bright rim on the cloud's star-facing skin (ionisation front)
        float rim = 0.0;
        if (d > -0.25){
          float shadowed = smoothstep(-0.2, 0.4, min(horse(p + LS * 1.2), horse(p + LS * 2.8)));
          float nl = clamp((horse(p + LS * 0.25) - horse(p)) / 0.25, -1.0, 1.0);
          rim = exp(-max(d, 0.0) / 0.07) * smoothstep(-0.2, 0.0, d) * smoothstep(0.2, 0.9, nl) * uIon * shadowed;
        }
        float rho = smoothstep(0.1, -0.3, d) * uDust * smoothstep(9.0, 7.5, r);
        vec3 att = exp(-5.0 * rho * dt * kRel);
        if (uIR > 0.5){
          I += T * rim * dt * 0.5 * vec3(1.0, 0.25, 0.1) * smoothstep(9.0, 3.5, r);
          I += T * (1.0 - att.x) * vec3(1.0, 0.5, 0.25) * (0.15 + 1.2 * rim);   // warm dust + PAH glow (JWST/Euclid)
        } else {
          I += T * rim * dt * 0.45 * (C_HA + C_HB / 2.86 * 0.6) * smoothstep(9.0, 3.5, r);
          I += T * (1.0 - att.x) * vec3(0.3, 0.1, 0.08) * 0.025 * uIon;               // dust lit by the nebula behind
        }
        T *= att;
      }
      t += dt;
      if (!s0 && ts0 < t){ I += g0 * T; s0 = true; }
      if (!s1 && ts1 < t){ I += g1 * T; s1 = true; }
      if (!s2 && ts2 < t){ I += g2 * T; s2 = true; }
    }
  }
  if (!s0) I += g0 * T;
  if (!s1) I += g1 * T;
  if (!s2) I += g2 * T;
  // σ Orionis (O9.5 V) out of frame above; a couple of field stars
  vec3 v = vec3(1.6, 11.0, -2.5) - ro;
  I += starPSF(rd, v, dot(v, rd), bb(32000.0), 14.0) * T;
  v = vec3(-3.2, -2.1, 2.2) - ro; I += starPSF(rd, v, dot(v, rd), bb(15000.0), 3.0);   // star lighting NGC 2023 (foreground)
  if (max(T.x, max(T.y, T.z)) > 0.004){
    // beyond the volume: the faint Hα glow of the wider Orion B / Barnard's Loop region
    float bn = smoothstep(-0.3, 0.7, fbm(rd * 2.4 + 6.0, 4));
    I += T * bn * uIon * (uIR > 0.5 ? vec3(0.2, 0.3, 0.6) * 0.012 : (C_HA + C_HB / 2.86) * 0.03);
    I += sky(rd, 1.0) * 0.6 * T;
  }
  gl_FragColor = vec4(I, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// JS helpers
// ------------------------------------------------------------------------------------------
const stepsFor = q => (q.steps <= 200 ? 56 : q.steps <= 320 ? 72 : q.steps <= 480 ? 88 : 104);

function makeMaterial(ctx, frag, own) {
  const uniforms = { ...ctx.shared, ...own };
  const material = new THREE.ShaderMaterial({
    uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: frag, depthTest: false, depthWrite: false,
    defines: { STEPS: stepsFor(ctx.quality) },
  });
  const setQuality = q => { material.defines.STEPS = stepsFor(q); material.needsUpdate = true; };
  return { uniforms, material, setQuality };
}

// smooth camera flight around the orbit target, interpolated in spherical coordinates so the
// path swings around the object instead of cutting through it
function makeFlyer() {
  let fly = null;
  const sph = new THREE.Spherical(), off = new THREE.Vector3();
  return {
    to(ctx, dist, polar, azim, dur = 2.4) {
      off.copy(ctx.camera.position).sub(ctx.controls.target); sph.setFromVector3(off);
      const from = { r: sph.radius, p: sph.phi, a: sph.theta };
      const to = { r: dist ?? sph.radius, p: THREE.MathUtils.degToRad(polar), a: THREE.MathUtils.degToRad(azim) };
      let da = to.a - from.a; da = Math.atan2(Math.sin(da), Math.cos(da)); to.a = from.a + da;
      fly = { from, to, t: 0, dur };
    },
    update(ctx, dt) {
      if (!fly) return;
      fly.t += dt; const k = Math.min(1, fly.t / fly.dur), e = k * k * (3 - 2 * k);
      sph.set(fly.from.r + (fly.to.r - fly.from.r) * e, fly.from.p + (fly.to.p - fly.from.p) * e, fly.from.a + (fly.to.a - fly.from.a) * e);
      ctx.camera.position.setFromSpherical(sph).add(ctx.controls.target);
      ctx.camera.lookAt(ctx.controls.target);
      if (k >= 1) fly = null;
    },
  };
}
const frameRows = (ex, ey, ez) => new THREE.Matrix3().set(ex.x, ex.y, ex.z, ey.x, ey.y, ey.z, ez.x, ez.y, ez.z);
const vi = (x, d = 1) => x.toFixed(d).replace('.', ',');
const ly = (x, d = 1) => vi(x, d) + ' năm ánh sáng';

// ------------------------------------------------------------------------------------------
// scene definitions
// ------------------------------------------------------------------------------------------
const PAL_NAMES = ['Hα đỏ · [O III] lục lam', 'R [S II] · G Hα · B [O III]', 'bụi ấm · H₂ · Paα'];

const emission = {
  id: 'emission-nebula',
  name: 'Tinh vân phát xạ',
  category: 'nebulae',
  order: 10,
  subtitle: 'Vùng H II · “Những cột trụ của sự sáng tạo” (M16, Tinh vân Đại Bàng)',
  swatch: ['#f2c27a', '#123a4a'],
  blurb: 'Những cột bụi và khí lạnh cao vài năm ánh sáng bị tia cực tím từ các sao O nóng của cụm NGC 6611 gọt đẽo. Ở mặt hướng về các sao, khí bị ion hóa và “bốc hơi” thành một <b>mặt ion hóa</b> rực sáng, phát các vạch Hα, [O III], [S II]; phần lõi bụi dày vẫn che khuất ánh sáng phía sau.',
  intro: 'Thử đổi <b>Bảng màu</b> sang <b>Hubble (SHO)</b> hoặc <b>Hồng ngoại (JWST)</b> — ở hồng ngoại, bụi gần như trong suốt và lộ ra các ngôi sao trẻ bên trong cột.',
  facts: [
    ['Khoảng cách', '≈ 5 700 năm ánh sáng'], ['Chòm sao', 'Cự Xà (Serpens)'],
    ['Chiều cao cột bên trái', '≈ 4 năm ánh sáng'], ['Cụm sao ion hóa', 'NGC 6611 (~2 triệu năm)'],
    ['Sao nóng nhất', 'HD 168076 · O3.5 V'], ['Nhiệt độ khí ion hóa', '≈ 10 000 K'],
  ],
  camera: { distance: 14.5, polar: 84, azimuth: 8, min: 0.5, max: 40, fov: 45, target: [0, -0.6, 0] },
  exposure: 1.25,
  bloom: 0.45,
  params: [
    { key: 'palette', label: 'Bảng màu', type: 'seg', options: [[0, 'Màu tự nhiên'], [1, 'Hubble (SHO)'], [2, 'Hồng ngoại (JWST)']], default: 0,
      note: 'SHO: [S II] → đỏ, Hα → lục, [O III] → lam. Hồng ngoại: bụi tắt sáng yếu hơn ~9 lần (A<sub>K</sub>/A<sub>V</sub> ≈ 0,11).' },
    { key: 'dust', label: 'Mật độ bụi', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Bụi hấp thụ ánh sáng xanh mạnh hơn đỏ — các ngôi sao phía sau bị đỏ hóa.' },
    { key: 'ion', label: 'Cường độ sao ion hóa', type: 'range', min: 0, max: 2.5, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Thông lượng tia UV ion hóa (λ < 91,2 nm) từ cụm NGC 6611.' },
  ],
  actions: [
    { label: 'Đổi bảng màu', key: 'p', run: ctx => {
      ctx.params.palette = (ctx.params.palette + 1) % 3;
      ctx.toast(['<b>Màu tự nhiên</b>: Hα đỏ + Hβ lam → hồng; [O III] lục lam.', '<b>Bảng màu Hubble</b>: mỗi vạch hẹp được gán một kênh màu — đây là màu “giả” để tách hóa học.', '<b>Hồng ngoại</b>: bụi gần trong suốt — sao trẻ trong cột lộ ra; mép cột đỏ là H₂ bị kích thích.'][ctx.params.palette]);
    } },
    { label: 'Tắt cụm sao', key: 'o', title: 'Tắt / bật nguồn UV', run: ctx => {
      ctx.params.ion = ctx.params.ion > 0.05 ? 0 : 1;
      ctx.toast(ctx.params.ion ? 'Cụm NGC 6611 sáng trở lại: các mặt ion hóa bừng lên.' : 'Không còn tia UV: khí tái hợp và tắt dần — chỉ còn bóng bụi đen trước nền sao.');
    } },
  ],
  create(ctx) {
    const { uniforms, material, setQuality } = makeMaterial(ctx, PILLARS_FRAG, {
      uDust: { value: 1 }, uIon: { value: 1 }, uPal: { value: 0 },
      uL: { value: new THREE.Vector3(0.3, 0.85, -0.45).normalize() },
    });
    const sm = { ion: 1, dust: 1 };
    return {
      material, setQuality,
      update(dt, ctx) {
        const p = ctx.params, k = 1 - Math.exp(-dt * 4);
        sm.ion += (p.ion - sm.ion) * k; sm.dust += (p.dust - sm.dust) * k;
        uniforms.uIon.value = sm.ion; uniforms.uDust.value = sm.dust; uniforms.uPal.value = p.palette;
      },
      hud(ctx) {
        const d = ctx.camera.position.distanceTo(ctx.controls.target), p = ctx.params;
        const life = p.ion > 0.02 ? '≈ ' + Math.round(100 / Math.sqrt(p.ion)) + ' nghìn năm' : 'không bị bào mòn';
        return [
          ['Camera cách tâm', ly(d)],
          ['Kênh màu', PAL_NAMES[p.palette]],
          ['A<sub>V</sub> xuyên qua cột', '≈ ' + vi(6.5 * 1.2 * 1.086 * p.dust) + ' mag' + (p.palette === 2 ? ' (A<sub>K</sub> ≈ ' + vi(0.11 * 8.5 * p.dust) + ')' : '')],
          ['Tuổi thọ còn lại của cột', life],
        ];
      },
    };
  },
};

const planetary = {
  id: 'planetary-nebula',
  name: 'Tinh vân hành tinh',
  category: 'nebulae',
  order: 20,
  subtitle: 'Tinh vân Chiếc Nhẫn (M57) · vỏ khí của một sao chết',
  swatch: ['#7fe0d0', '#b8302a'],
  blurb: 'Một ngôi sao cỡ Mặt Trời ở cuối đời thổi bay lớp vỏ ngoài; lõi trơ lại thành sao lùn trắng ~120 000 K, tia UV của nó ion hóa lớp vỏ. Vỏ có dạng <b>thùng rỗng</b> — ta nhìn gần dọc trục nên thấy một chiếc nhẫn. Các ion được <b>phân tầng</b>: He II, [O III] (lam-lục) ở trong, Hα và [N II] (đỏ) ở ngoài.',
  intro: 'Kéo để xoay sang bên (hoặc bấm <b>V</b>): chiếc “nhẫn” thực ra là một cái thùng rỗng hai đầu.',
  facts: [
    ['Khoảng cách', '≈ 2 570 năm ánh sáng (Gaia)'], ['Chòm sao', 'Thiên Cầm (Lyra)'],
    ['Kích thước vòng chính', '≈ 1,3 × 1 năm ánh sáng'], ['Quầng ngoài', '≈ 3,5 năm ánh sáng'],
    ['Sao trung tâm', 'lùn trắng ≈ 120 000 K'], ['Tốc độ giãn nở', '≈ 20 – 40 km/s'],
    ['Tuổi', '≈ 4 000 năm'], ['Phát hiện', 'Antoine Darquier, 1779'],
  ],
  camera: { distance: 3.4, polar: 90, azimuth: 0, min: 0.15, max: 14, fov: 40, target: [0, 0, 0] },
  exposure: 1.1,
  bloom: 0.5,
  params: [
    { key: 'thick', label: 'Độ dày vỏ', type: 'range', min: 0.06, max: 0.4, step: 0.005, default: 0.16, fmt: v => vi(v * 0.55 * 2, 2) + ' ly',
      note: 'Vỏ càng dày, các tầng ion hóa (lam → lục → đỏ) càng tách rõ.' },
    { key: 'knots', label: 'Nút sao chổi', type: 'range', min: 0, max: 1, step: 0.01, default: 0.55, fmt: v => Math.round(v * 100) + '%',
      note: 'Khối khí trung hòa đặc ở mặt ion hóa; đuôi hướng ra xa ngôi sao. JWST đếm được ~20 000 nút trong M57.' },
    { key: 'age', label: 'Tuổi (giãn nở)', type: 'range', min: 1500, max: 8000, step: 50, default: 4000, fmt: v => Math.round(v).toLocaleString('vi-VN') + ' năm',
      note: 'Vỏ nở đều với v ≈ 37 km/s; mật độ giảm nên tinh vân mờ dần và tan vào môi trường liên sao sau ~20 000 năm.' },
  ],
  actions: [
    { label: 'Nhìn dọc trục', key: 'r', title: 'Nhìn dọc trục đối xứng: chiếc nhẫn', run: ctx => ctx.scene._fly?.(ctx, 0) },
    { label: 'Nhìn ngang', key: 'v', title: 'Nhìn vuông góc trục: hình thùng', run: ctx => ctx.scene._fly?.(ctx, 1) },
  ],
  create(ctx) {
    const axis = new THREE.Vector3(0, 0.18, 1).normalize();          // barrel axis, ~10° from the default line of sight
    const ex = new THREE.Vector3(1, 0, 0), ey = new THREE.Vector3().crossVectors(axis, ex).normalize();
    ex.crossVectors(ey, axis).normalize();
    const { uniforms, material, setQuality } = makeMaterial(ctx, RING_FRAG, {
      uW: { value: 0.16 }, uKnots: { value: 0.55 }, uSc: { value: 1 }, uBright: { value: 1 },
      uFrame: { value: frameRows(ex, ey, axis) },
    });
    const flyer = makeFlyer();
    planetary._fly = (ctx, side) => {
      const d = ctx.camera.position.distanceTo(ctx.controls.target);
      if (side) { flyer.to(ctx, d, 90, 90); ctx.toast('Nhìn ngang trục: hai bức tường sáng của “cái thùng”, hai đầu gần như trong suốt.'); }
      else { flyer.to(ctx, d, 90 - 10.2, 0); ctx.toast('Nhìn dọc trục: thành thùng chồng lên nhau theo đường nhìn → một chiếc nhẫn.'); }
    };
    const camDir = new THREE.Vector3();
    return {
      material, setQuality,
      update(dt, ctx) {
        const p = ctx.params, sc = p.age / 4000;
        uniforms.uW.value = p.thick; uniforms.uKnots.value = p.knots; uniforms.uSc.value = sc;
        // shell mass fixed: n ∝ R⁻³, emission ∝ n² × path ∝ R⁻⁵ — softened to R⁻² for display
        uniforms.uBright.value = 1.2 / Math.pow(sc, 1.0) * (0.16 / p.thick) ** 0.5;
        flyer.update(ctx, dt);
      },
      hud(ctx) {
        const p = ctx.params, sc = p.age / 4000;
        camDir.copy(ctx.camera.position).sub(ctx.controls.target);
        const d = camDir.length();
        const ang = THREE.MathUtils.radToDeg(Math.acos(Math.min(1, Math.abs(camDir.normalize().dot(axis)))));
        const R = 0.55 * sc, v = R * 9.461e12 / (p.age * 3.156e7);
        return [
          ['Camera cách sao lùn trắng', ly(d, 2)],
          ['Góc giữa đường nhìn và trục', ang.toFixed(0) + '° ' + (ang < 30 ? '(thấy nhẫn)' : ang > 60 ? '(thấy thùng)' : '')],
          ['Bán kính vòng', ly(R, 2)],
          ['Tốc độ giãn nở trung bình', '≈ ' + Math.round(v) + ' km/s'],
        ];
      },
    };
  },
};

const crab = {
  id: 'supernova-remnant',
  name: 'Tàn dư siêu tân tinh',
  category: 'nebulae',
  order: 30,
  subtitle: 'Tinh vân Con Cua (M1) · tàn dư của SN 1054',
  swatch: ['#a8c4ff', '#d0502a'],
  blurb: 'Năm 1054, các nhà thiên văn Trung Quốc ghi lại một “khách tinh” sáng đến mức nhìn thấy giữa ban ngày suốt 23 ngày. Ngày nay mảnh vỡ của nó là một lồng sợi khí Hα đỏ-cam giãn nở ~1 500 km/s, bao quanh một đám sáng lam trắng — <b>bức xạ synchrotron</b> của electron siêu tương đối tính do <b>pulsar Con Cua</b> (quay 30 vòng/giây) thổi ra.',
  intro: 'Bấm <b>R</b> để xem tàn dư lớn lên từ năm 1054, hoặc bật âm thanh để nghe nhịp pulsar.',
  facts: [
    ['Khoảng cách', '≈ 6 500 năm ánh sáng (2,0 kpc)'], ['Chòm sao', 'Kim Ngưu (Taurus)'],
    ['Kích thước', '≈ 11 × 7 năm ánh sáng'], ['Tốc độ giãn nở', '≈ 1 500 km/s'],
    ['Bùng nổ', '4/7/1054'], ['Thấy bằng mắt thường', 'ngày: 23 ngày · đêm: 653 ngày'],
    ['Pulsar', 'PSR B0531+21 · chu kỳ 33,7 ms (29,6 Hz)'], ['Công suất hãm quay', '≈ 4,5×10³¹ W (~10⁵ L☉)'],
  ],
  camera: { distance: 17, polar: 86, azimuth: 0, min: 0.3, max: 60, fov: 45, target: [0, 0, 0] },
  exposure: 1.15,
  bloom: 0.55,
  params: [
    { key: 'year', label: 'Năm quan sát', type: 'range', min: 1100, max: 3000, step: 1, default: 2026, fmt: v => String(Math.round(v)),
      note: 'Kích thước ∝ tuổi (giãn nở gần như tự do). Pulsar chậm dần 4,2×10⁻¹³ s mỗi giây.' },
    { key: 'rate', label: 'Nhịp pulsar (hiển thị)', type: 'range', min: 0.25, max: 30, step: 0.25, default: 1, fmt: v => vi(v, 2) + ' Hz',
      note: 'Thực tế 29,6 Hz — mỗi vòng quay có xung chính và xung phụ (cách 0,4 chu kỳ).' },
    { key: 'synch', label: 'Tinh vân gió pulsar', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Bức xạ synchrotron lam trắng: electron xoắn quanh từ trường của pulsar.' },
    { key: 'click', label: 'Âm thanh pulsar', type: 'toggle', default: true, note: 'Mỗi xung chính một tiếng tách.' },
  ],
  actions: [
    { label: 'Tua lại từ 1054', key: 'r', run: ctx => { ctx.scene._replay = { t: 0 }; ctx.toast('Năm 1054 → nay: mảnh vỡ bay với ~1 500 km/s, mỗi thế kỷ rộng thêm ~1 năm ánh sáng.'); } },
    { label: 'Nhịp thật 30 Hz', key: 'j', run: ctx => {
      ctx.params.rate = ctx.params.rate >= 29 ? 1 : 29.6;
      ctx.toast(ctx.params.rate > 29 ? 'Tốc độ thật: 29,6 vòng/giây. Một sao neutron nặng 1,4 M☉, đường kính ~20 km.' : 'Đã làm chậm lại để dễ quan sát.');
    } },
  ],
  create(ctx) {
    // long axis runs NW–SE on the sky: tilt it ~35° in the image plane and a little in depth
    const ex = new THREE.Vector3(Math.cos(0.6), Math.sin(0.6), 0.25).normalize();
    const ez = new THREE.Vector3(0, 0, 1); const ey = new THREE.Vector3().crossVectors(ez, ex).normalize(); ez.crossVectors(ex, ey);
    const { uniforms, material, setQuality } = makeMaterial(ctx, CRAB_FRAG, {
      uSc: { value: 1 }, uPulse: { value: 0 }, uSyn: { value: 1 }, uFil: { value: 1 }, uWisp: { value: 0 }, uFade: { value: 1 },
      uFrame: { value: frameRows(ex, ey, ez) },
      uSpin: { value: new THREE.Vector3(-0.45, -0.5, 0.74).normalize() },   // jet toward the SE, torus inclined ~60°
    });
    let phase = 0.37, wisp = 0;
    return {
      material, setQuality,
      update(dt, ctx) {
        const p = ctx.params, rp = crab._replay;
        if (rp) {
          rp.t += dt;
          const k = Math.min(1, rp.t / 9);
          p.year = Math.round(1060 + (2026 - 1060) * (1 - Math.pow(1 - k, 2)));
          if (k >= 1) crab._replay = null;
        }
        const age = Math.max(p.year - 1054, 5);
        uniforms.uSc.value = age / 972;
        // early on the ejecta is dense and hot: brighter (sim display: ∝ 1/size, capped)
        uniforms.uFade.value = 1;
        uniforms.uSyn.value = p.synch * Math.pow(972 / Math.max(age, 100), 0.35) * (age > 2000 ? Math.pow(2000 / age, 0.6) : 1);
        // pulse profile: main pulse + interpulse 0.4 cycles later (radio/optical Crab profile)
        const prev = phase;
        phase = (phase + dt * p.rate) % 1;
        const g = (x, w) => Math.exp(-0.5 * (x / w) ** 2);
        const dph = x => { const d = phase - x; return d - Math.round(d); };
        uniforms.uPulse.value = g(dph(0), 0.025) + 0.55 * g(dph(0.4), 0.035);
        if (phase < prev && p.click && ctx.audio.started) ctx.audio.click(0.25, 0, 0.8);
        wisp = (wisp + dt * 0.12) % 1;      // wisps really move outward at ~0.5 c: a few months to cross the torus
        uniforms.uWisp.value = wisp;
      },
      hud(ctx) {
        const p = ctx.params, age = Math.max(p.year - 1054, 0);
        const d = ctx.camera.position.distanceTo(ctx.controls.target);
        const P = 33.7 + 4.21e-13 * (p.year - 2026) * 3.156e7 * 1e3;
        return [
          ['Camera cách pulsar', ly(d)],
          ['Tuổi tàn dư', Math.round(age).toLocaleString('vi-VN') + ' năm'],
          ['Đường kính lớn nhất', ly(11.2 * age / 972)],
          ['Chu kỳ pulsar năm ' + Math.round(p.year), vi(P, 1) + ' ms (' + vi(1000 / P, 1) + ' Hz)'],
        ];
      },
    };
  },
};

const dark = {
  id: 'dark-nebula',
  name: 'Tinh vân tối',
  category: 'nebulae',
  order: 40,
  subtitle: 'Tinh vân Đầu Ngựa (Barnard 33) trước nền IC 434',
  swatch: ['#d8484a', '#140a0c'],
  blurb: 'Một cột bụi và khí phân tử lạnh (~10–20 K) nhô ra từ đám mây Orion B. Bản thân nó không phát sáng: ta chỉ thấy <b>bóng</b> của nó in trên tấm khí hydro IC 434 đang phát Hα đỏ do ngôi sao σ Orionis ion hóa. Mép trên của đám mây có một viền sáng — mặt ion hóa đang bào mòn cái đầu ngựa.',
  intro: 'Thử bật <b>Hồng ngoại</b>: bụi trở nên trong suốt và tự phát sáng ấm, như ảnh của JWST và Euclid.',
  facts: [
    ['Khoảng cách', '≈ 1 375 năm ánh sáng'], ['Chòm sao', 'Lạp Hộ (Orion)'],
    ['Chiều cao “đầu ngựa”', '≈ 3,5 năm ánh sáng'], ['Sao ion hóa', 'σ Orionis · O9.5 V · ≈ 32 000 K'],
    ['Nhiệt độ bụi', '≈ 10 – 20 K'], ['Tuổi thọ còn lại', '≈ 5 triệu năm'],
  ],
  camera: { distance: 9.5, polar: 97, azimuth: 0, min: 0.5, max: 35, fov: 45, target: [0, -0.9, 0] },
  exposure: 1.3,
  bloom: 0.45,
  params: [
    { key: 'dust', label: 'Mật độ bụi', type: 'range', min: 0, max: 2, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2) },
    { key: 'ion', label: 'Độ sáng IC 434 (σ Ori)', type: 'range', min: 0, max: 2.5, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2) },
    { key: 'ir', label: 'Hồng ngoại (JWST/Euclid)', type: 'toggle', default: false, note: 'A<sub>K</sub>/A<sub>V</sub> ≈ 0,11: bụi gần trong suốt, chỉ còn bức xạ nhiệt của chính nó.' },
  ],
  actions: [
    { label: 'Hồng ngoại', key: 'i', run: ctx => { ctx.params.ir = !ctx.params.ir; } },
  ],
  create(ctx) {
    const { uniforms, material, setQuality } = makeMaterial(ctx, HORSE_FRAG, { uDust: { value: 1 }, uIon: { value: 1 }, uIR: { value: 0 } });
    return {
      material, setQuality,
      update(dt, ctx) {
        const p = ctx.params;
        uniforms.uDust.value = p.dust; uniforms.uIon.value = p.ion; uniforms.uIR.value = p.ir ? 1 : 0;
      },
      hud(ctx) {
        const d = ctx.camera.position.distanceTo(ctx.controls.target);
        return [
          ['Camera cách tâm', ly(d)],
          ['A<sub>V</sub> qua đầu ngựa', '≈ ' + vi(5 * 1.6 * 1.086 * ctx.params.dust) + ' mag'],
          ['Ánh sáng thấy', ctx.params.ir ? 'hồng ngoại gần–giữa' : 'khả kiến (Hα)'],
        ];
      },
    };
  },
};

export default [emission, planetary, crab, dark];
