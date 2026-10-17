// Các ngôi sao — stars rendered from their physics.
//
// One shared, parameterised star shader (compiled once per scene with a KIND define so each program
// only contains what that star needs); every scene below is a preset of it.
//
//   KIND 0  Sun               granulation, sunspots, faculae, chromosphere, prominences, corona
//   KIND 1  red dwarf         Proxima Centauri: huge starspots + flares
//   KIND 2  red supergiant    Betelgeuse: a handful of giant convection cells, lumpy limb, dusty envelope
//   KIND 3  white dwarf       Sirius B: featureless, sharp limb; Earth for scale, Sirius A in the sky
//   KIND 4  blue supergiant   Rigel: non-radial pulsations and a stellar-wind halo
//
// Units: 1 scene unit = 1 radius of the star being shown (`unitKm` per scene).
// Brightness: each star is "exposed for its own surface" like a filtered solar telescope image:
// radiance = bb(T) · (T/T_ref)^n · k, where n compresses the true Wien-side sensitivity (∂lnB/∂lnT ≈ 5–9 in
// the visible) into something readable, so cooler spots / hotter faculae and flares change both colour
// and brightness.
// Limb darkening: quadratic law I(μ)/I(1) = 1 − a(1−μ) − b(1−μ)² with separate (a,b) for R, G, B, so the
// limb is darker AND redder (blue light comes from shallower, cooler layers near the limb).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

const NS = 16, NL = 8, NF = 4, NP = 24;   // spots, loops (flare loops / 171 Å loops), flares, volumetric prominences
const R_SUN_KM = 695700, AU_KM = 1.495978707e8, C_KMS = 299792.458, G_SUN = 274.0; // m/s²

/* ============================================================================================== */
/* Shader                                                                                          */
/* ============================================================================================== */
const FRAG = COMMON_GLSL + /* glsl */`
#define HALPHA vec3(1.0, 0.15, 0.22)
uniform float uTeff, uTref, uIexp, uBright, uSkyK, uStretch;
uniform vec3 uLDa, uLDb;
uniform vec3 uOmega;            // differential rotation Ω(φ) = x + y·sin²φ + z·sin⁴φ  [rad per sim-second]
uniform vec4 uEp;               // granulation epochs: (age A, age B, seed A, seed B)  — see epochs() in JS
uniform float uEpW;             // weight of epoch A
uniform float uGranF, uGranAmp, uGranT, uRotT;
uniform float uActivity, uChromo, uCorona, uProm, uEclipse;
uniform vec3 uSpotT;            // umbra T/Teff, penumbra T/Teff, penumbral filament contrast
uniform vec4 uSpot[NS];         // xyz centre (unit vector), w radius
uniform vec4 uSpotX[NS];        // umbra fraction, seed, facular strength, facular radius
uniform vec4 uLoopA[NL];        // footpoint A, apex height
uniform vec4 uLoopB[NL];        // footpoint B, seed
uniform vec4 uLoopC[NL];        // colour temperature (0 = H-alpha), brightness, thickness, kind (0 loop, 1 sheet)
uniform vec4 uFl[NF];           // flare site (unit vector), current light-curve value
uniform vec4 uFlB[NF];          // size, ribbon half-separation, axis angle, flare temperature
uniform float uDim, uEnv, uCell, uPuls, uWind, uRad;
uniform vec4 uDust;             // Great-Dimming dust cloud: centre, radius
uniform vec3 uEarthPos; uniform float uEarthOn, uSirA; uniform vec3 uSirDir;
uniform float uNeb, uKnots, uIR;

vec3 rotY(vec3 p, float a){ float c = cos(a), s = sin(a); return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z); }
float omegaAt(float s2){ return uOmega.x + uOmega.y*s2 + uOmega.z*s2*s2; }
vec3 limbDark(float mu){ float m = 1.0 - mu; return max(vec3(1.0) - uLDa*m - uLDb*m*m, 0.0); }
vec3 emit(float T){ return bb(T) * pow(max(T, 500.0) / uTref, uIexp) * uBright; }
vec3 tangentU(vec3 c){ return normalize(cross(c, abs(c.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0))); }

// Convection cells: animated Voronoi. Hot plasma rises in the cell centres (bright), spreads, cools and
// sinks in the narrow dark intergranular lanes. Each cell also brightens and fades (finite lifetime).
float granule(vec3 p, float t, float seed){
  vec3 i = floor(p), f = fract(p);
  float d1 = 8.0, d2 = 8.0, h1 = 0.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(float(x), float(y), float(z));
    vec3 h = hash33(i + o + seed);
    vec3 r = o + 0.5 + 0.4 * sin(TAU * h + t * (0.6 + 0.8 * h.zxy)) - f;
    float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; h1 = h.x; } else if (d < d2) d2 = d;
  }
  float k = sqrt(d1 / d2);                    // 0 at the up-flow centre → 1 on the lane (rounded cells)
  float life = 0.5 + 0.5 * sin(t * 0.45 + TAU * h1);
  float body = (0.4 + 0.6 * life) * (1.0 - 0.55 * k * k);
  float lane = smoothstep(0.6, 0.95, k);
  return mix(body, -1.0 - 0.4 * smoothstep(0.35, 0.0, sqrt(d2) - sqrt(d1)) * (1.0 - life), lane) - 0.2;
}
#if KIND == 0
// Solar granulation, closer to DKIST/Hinode imagery: a multiplicatively weighted Voronoi (each cell has
// its own size, so granules range ~0.7–1.5× and edges become curved), brightness peaking in the cell
// middle (hot up-flow) and falling into lanes of varying width; some large granules "explode" with a
// dark centre; magnetic bright points sit in the lanes; mesogranular-scale brightness modulation.
float sunGranule(vec3 p, float t, float seed){
  vec3 i = floor(p), f = fract(p);
  float d1 = 8.0, d2 = 8.0;
  vec3 h1 = vec3(0.0), g1 = vec3(0.0); float w1 = 1.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(float(x), float(y), float(z));
    vec3 h = hash33(i + o + seed);
    vec3 g = hash33(i + o + seed + 71.3);
    float w = 0.75 + 0.6 * g.x;                                   // per-granule size
    vec3 r = o + 0.5 + 0.38 * sin(TAU * h + t * (0.6 + 0.8 * h.zxy)) - f;
    float d = length(r) / w;
    if (d < d1){ d2 = d1; d1 = d; h1 = h; g1 = g; w1 = w; } else if (d < d2) d2 = d;
  }
  float life = 0.5 + 0.5 * sin(t * 0.45 + TAU * h1.y);
  float prof = 1.0 - smoothstep(0.0, 0.85, d1);                  // dome: hot up-flow in the middle, cooling outward
  float body = (0.55 + 0.45 * h1.x) * (0.5 + 0.5 * life) * (0.0 + 1.0 * pow(prof, 1.4));
  if (g1.y > 0.8 && w1 > 1.1) body *= 1.0 - 0.65 * exp(-d1 * d1 / 0.048);   // exploding granule
  float sn = snoise(p * 0.7 + seed);
  float lw = 0.16 + 0.2 * (0.5 + 0.5 * sn);                                 // lanes of varying width, soft edges
  float lane = 1.0 - smoothstep(0.0, lw, d2 - d1);
  lane *= lane * (3.0 - 2.0 * lane);
  float bp = lane * smoothstep(0.55, 0.9, snoise(p * 2.7 + seed * 1.7)); // bright points in the lanes
  return mix(body, -0.8 - 0.3 * sn, lane * lane) + 1.6 * bp + 0.1;
}
// ---- prominences, rendered as volumes (see promMarch) ------------------------------------------------
uniform vec4 uPrA[NP];   // foot-point centre C (unit vector, world), type (0 hedgerow, 1 flux-rope arch, 2 loop arcade, 3 erupting rope)
uniform vec4 uPrB[NP];   // axis direction T (unit, tangent at C), seed
uniform vec4 uPrC[NP];   // half-length L, height H, sheet half-thickness / rope radius a, density scale (0 = empty slot)
uniform vec4 uPrD[NP];   // bounding box in the local frame: half-length, top, half-width, depth below the tangent plane
uniform vec4 uPrE[NP];   // twist (turns from foot to foot), number of legs, lean, unused
uniform vec4 uCme;       // CME bubble centre, radius (0 = none)
uniform float uCmeB;     // CME brightness
uniform float uView;     // 0 white light, 1 H-alpha 656.3 nm, 2 He II 30.4 nm, 3 Fe IX 17.1 nm
uniform float uLoopK;    // brightness of the 171 Å coronal loops
// Narrow-band images are monochrome; observatories and astrophotographers show them through colour tables.
// H-alpha uses a heat table like the best published Hα images: faint levels in the deep red of 656 nm (which
// sRGB can only approximate), brighter ones through orange toward white. The EUV channels use SDO/AIA's
// conventional false colours (304 Å red-orange, 171 Å gold).
vec3 lutHa(float x){ x = max(x, 0.0); return vec3(x, 0.46 * pow(x, 2.2) + 0.01 * x, 0.2 * pow(x, 3.6) + 0.012 * x); }
vec3 lut304(float x){ x = max(x, 0.0); return vec3(x, 0.42 * x * sqrt(x), 0.07 * x * x * x); }
vec3 lut171(float x){ x = max(x, 0.0); return vec3(x, 0.6 * pow(x, 1.15), 0.1 * x * x) + vec3(0.05, 0.08, 0.08) * x * x * x; }
vec3 lutView(float x){ return uView < 1.5 ? lutHa(x) : (uView < 2.5 ? lut304(x) : lut171(x)); }
#endif
vec3 warp3(vec3 p){ return p + 0.45 * vec3(snoise(p * 0.31), snoise(p * 0.31 + 17.1), snoise(p * 0.31 + 31.7)); }
// a finer second warp makes granule borders irregular and rounded instead of straight polygon edges
vec3 warpFine(vec3 p){ return p + 0.16 * vec3(snoise(p * 1.6 + 5.3), snoise(p * 1.6 + 41.2), snoise(p * 1.6 + 77.9)); }

#if KIND <= 1
// ---- sunspots / starspots ------------------------------------------------------------------------
// Strong magnetic flux tubes suppress convection: the umbra is ~1500 K cooler than the photosphere,
// the penumbra is made of radial filaments (inclined field), and the surrounding plage/faculae are hot
// walls of thin flux tubes that are seen best near the limb.
float spotField(vec3 q, float fp, inout float fac, inout float gsup){
  float tf = 1.0;
  for (int i = 0; i < NS; i++){
    vec4 s = uSpot[i], x = uSpotX[i];
    if (x.w <= 0.0) continue;
    vec3 dv = q - s.xyz;
    float d2 = dot(dv, dv);
    if (d2 > x.w * x.w * 4.0) continue;
    fac += x.z * exp(-d2 / (x.w * x.w));
    if (s.w <= 0.0) continue;
    float dist = sqrt(d2);
    float wob = 1.0 + 0.2 * snoise(q * (2.2 / s.w) + x.y) + 0.08 * snoise(q * (7.0 / s.w) + x.y);
    float r = dist / (s.w * wob);
    if (r > 1.25) continue;
    vec3 e1 = tangentU(s.xyz), e2 = cross(s.xyz, e1);
    float ang = atan(dot(dv, e2), dot(dv, e1));
    float lodF = 1.0 - smoothstep(0.012 * s.w, 0.04 * s.w, fp);
    float fil = snoise(vec3(cos(ang) * 24.0, sin(ang) * 24.0, r * 2.2 + x.y)) * lodF;
    float pen = smoothstep(1.0, uSpotT.z > 0.001 ? 0.9 : 0.7, r);   // M-dwarf spots: soft edges, little penumbra
    float umb = smoothstep(x.x + 0.05, x.x - 0.05, r);
    float tp = uSpotT.y * (1.0 + uSpotT.z * fil);
    float dots = lodF * smoothstep(0.6, 0.95, snoise(q * (60.0 / s.w) + x.y * 3.0));   // umbral dots
    float tu = uSpotT.x * (1.0 + 0.06 * dots);
    tf = min(tf, mix(1.0, mix(tp, tu, umb), pen));
    gsup = min(gsup, 1.0 - pen);
  }
  return tf;
}
#endif

#if KIND == 1
// ---- flares: two bright ribbons at the footpoints of reconnecting loops + an impulsive kernel ----
void flareField(vec3 q, inout float T, inout vec3 glow){
  for (int i = 0; i < NF; i++){
    vec4 F = uFl[i], G = uFlB[i];
    if (F.w <= 0.0) continue;
    vec3 dv = q - F.xyz; float d = length(dv);
    if (d > G.x * 6.0) continue;
    vec3 e1 = tangentU(F.xyz), e2 = cross(F.xyz, e1);
    float ca = cos(G.z), sa = sin(G.z);
    vec2 uv = vec2(dot(dv, e1) * ca + dot(dv, e2) * sa, -dot(dv, e1) * sa + dot(dv, e2) * ca) / G.x;
    float knot = 0.5 + 0.5 * snoise(vec3(uv * 9.0, float(i) * 7.3 + G.y * 2.0));
    float along = exp(-uv.x * uv.x / 0.45);
    float rib = along * (exp(-pow((uv.y - G.y) / 0.07, 2.0)) + exp(-pow((uv.y + G.y) / 0.07, 2.0))) * knot;
    float ker = exp(-dot(uv, uv) / 0.02);
    float heat = clamp((rib + 1.6 * ker) * F.w, 0.0, 1.0);
    T = mix(T, G.w, heat);
    glow += bb(G.w) * F.w * exp(-d / (G.x * 0.6)) * 0.25 * uBright;   // white-light continuum around the site
  }
}
#endif

#if KIND <= 1
vec3 photosphere(vec3 q, float mu, float fp){
  // Differential rotation shears everything attached to the surface. To keep the shear bounded the
  // granulation pattern is re-seeded every epoch and two staggered epochs are cross-faded (granules
  // only live ~10 minutes, so re-seeding is invisible).
  float om = omegaAt(q.y * q.y);
  vec3 qa = rotY(q, -om * uEp.x), qb = rotY(q, -om * uEp.y);
  float wA = uEpW, wB = 1.0 - uEpW;
  float cs = 1.0 / uGranF;
  float lodG = 1.0 - smoothstep(0.3 * cs, 1.1 * cs, fp);   // granules smaller than a pixel average out
  float g = 0.0, net = 0.0;
  if (lodG > 0.001){
#if KIND == 0
    float lodW = 1.0 - smoothstep(0.08 * cs, 0.3 * cs, fp);   // the fine warp only matters when granules are big on screen
    if (wA > 0.001){ vec3 w = warp3(qa * uGranF); if (lodW > 0.0) w = mix(w, warpFine(w), lodW); g += wA * sunGranule(w, uGranT, uEp.z * 13.0); }
    if (wB > 0.001){ vec3 w = warp3(qb * uGranF); if (lodW > 0.0) w = mix(w, warpFine(w), lodW); g += wB * sunGranule(w, uGranT, uEp.w * 13.0); }
    g *= 0.85 + 0.3 * snoise(qa * uGranF * 0.17);  // mesogranular brightness modulation
    g += 0.08 * snoise(qa * uGranF * 3.1);         // sub-granular texture
#else
    if (wA > 0.001) g += wA * granule(warp3(qa * uGranF), uGranT, uEp.z * 13.0);
    if (wB > 0.001) g += wB * granule(warp3(qb * uGranF), uGranT, uEp.w * 13.0);
    g += 0.12 * snoise(qa * uGranF * 3.1);    // sub-granular texture
#endif
    g *= lodG;
  }
  // supergranulation (~30 000 km): its boundaries collect magnetic field → bright network
  float sgF = uGranF / 28.0;
  float lodS = 1.0 - smoothstep(0.25 / sgF, 0.8 / sgF, fp);
  if (lodS > 0.001){
    if (wA > 0.001){ vec2 w = worley(qa * sgF + uEp.z * 7.31); net += wA * smoothstep(0.3, 0.0, w.y - w.x); }
    if (wB > 0.001){ vec2 w = worley(qb * sgF + uEp.w * 7.31); net += wB * smoothstep(0.3, 0.0, w.y - w.x); }
    net *= lodS * smoothstep(-0.2, 0.6, snoise(qa * sgF * 4.0));   // the network is patchy, not a continuous web
  }
  float fac = 0.0, gsup = 1.0;
  float tf = spotField(q, fp, fac, gsup);
  float T = uTeff * (1.0 + uGranAmp * 0.06 * g * gsup);
#if KIND == 0
  T *= 1.0 + uGranAmp * 0.006 * (1.0 - lodG) * gsup * (snoise(q * 160.0) + 0.6 * snoise(q * 47.0));
#endif
  T *= tf;
  // faculae: contrast grows toward the limb (we look at the hot walls of flux tubes side-on)
  float lw = pow(1.0 - mu, 1.3) * smoothstep(0.0, 0.12, mu);
  T *= 1.0 + (0.08 * min(fac, 1.5) * (0.3 + 0.7 * net) + 0.007 * net) * lw * gsup;
  vec3 c;
#if KIND == 1
  vec3 glow = vec3(0.0);
  flareField(q, T, glow);
  c = emit(T) * limbDark(mu) + glow;
#else
  c = emit(T) * limbDark(mu);
#endif
  return c;
}

// chromosphere seen edge-on above the limb: H-alpha pink, a forest of spicules (jets ~3 000–10 000 km)
vec3 chromosphere(vec3 s, float h){
  float sp = snoise(s * 240.0 + vec3(0.0, uGranT * 0.05, 0.0));
  float sp2 = snoise(s * 700.0 - vec3(uGranT * 0.03));
  float hs = 0.0045 + 0.007 * (0.5 + 0.5 * sp);
  float e = exp(-h / 0.0011) + 0.4 * smoothstep(hs, hs * 0.4, h) * (0.5 + 0.5 * sp2);
  return HALPHA * e * uChromo;
}

// loops (prominences, post-flare loops) — glowing tubes approximated by 8 segments, integrated
// analytically across the tube; hedgerow prominences are thin vertical sheets with threads.
vec3 loopsGlow(vec3 ro, vec3 rd, float tEnd){
  vec3 acc = vec3(0.0);
  for (int k = 0; k < NL; k++){
    vec4 C = uLoopC[k];
    if (C.y <= 0.0) continue;
    vec4 A = uLoopA[k], B = uLoopB[k];
    vec3 m = 0.5 * (A.xyz + B.xyz), e = 0.5 * (B.xyz - A.xyz);
    float L = max(length(e), 1e-4); vec3 eh = e / L;
    vec3 n = normalize(m - eh * dot(m, eh));
    float Hh = A.w + 1.0 - length(m);
    vec3 cen = m + n * Hh * 0.5; float rad = max(L, Hh) + C.z * 4.0;
    vec3 oc = cen - ro; float tb = dot(oc, rd);
    if (dot(oc, oc) - tb * tb > rad * rad) continue;
#if KIND == 0
    vec3 colk = lut171(1.0);          // the Sun only uses these tubes for 171 Å coronal loops
#else
    vec3 colk = C.x < 100.0 ? HALPHA : bb(C.x);
#endif
    float em = 0.0;
    if (C.w < 0.5){
      vec3 P0 = m - e; vec3 bestX = P0; float best = 0.0;
      for (int j = 1; j <= 8; j++){
        float th = PI * (1.0 - float(j) / 8.0);
        vec3 P1 = m + e * cos(th) + n * Hh * sin(th);
        vec3 u = P1 - P0, w0 = ro - P0;
        float bq = dot(rd, u), cq = dot(u, u), dq = dot(rd, w0), eq = dot(u, w0);
        float den = max(cq - bq * bq, 1e-9);
        float tt = clamp((eq - bq * dq) / den, 0.0, 1.0);
        vec3 S = P0 + u * tt;
        float sc = max(dot(S - ro, rd), 0.0);
        vec3 X = ro + rd * sc;
        float d2 = dot(X - S, X - S);
        float ej = exp(-d2 / (C.z * C.z)) / max(sqrt(den / cq), 0.2);
        if (sc < tEnd && dot(X, X) > 1.0 && ej > best){ best = ej; bestX = X; }
        P0 = P1;
      }
      if (best > 0.002) em = best * (0.55 + 0.45 * snoise(bestX * (0.6 / C.z) + B.w - vec3(uGranT * 0.08)));
    } else {
      vec3 w = cross(eh, n);
      float dn = dot(rd, w);
      dn = abs(dn) < 1e-4 ? 1e-4 : dn;
      float t = dot(m - ro, w) / dn;
      if (t > 0.0 && t < tEnd){
        vec3 p = ro + rd * t;
        if (dot(p, p) > 1.0){
          vec3 x = p - m; float u = dot(x, eh) / L, v = dot(x, n) / Hh;
          float top = sqrt(max(1.0 - u * u, 0.0)) * (0.65 + 0.35 * snoise(vec3(u * 2.5, B.w, uGranT * 0.01)));
          float inside = smoothstep(top + 0.02, top - 0.15, v) * (1.0 - smoothstep(0.85, 1.0, abs(u)));
          if (inside > 0.0){
            float thr = 0.35 + 0.65 * smoothstep(-0.3, 0.7, snoise(vec3(u * L * 110.0, v * Hh * 9.0, B.w + uGranT * 0.02)));
            em = inside * thr / max(abs(dn), 0.25) * 0.35;
          }
        }
      }
    }
    acc += colk * C.y * em;
  }
  return acc;
}
#endif

#if KIND == 0
// K+F corona: Baumbach's radial law, shaped into helmet streamers along a warped current sheet (flat at
// solar minimum, all latitudes at maximum) and polar plumes; anything that depends only on direction
// gives radial rays automatically.
vec3 corona(vec3 s, float b){
  vec3 sb = rotY(s, -uOmega.x * uRotT * 0.93);   // s: plane-of-sky direction (see main)
  float rho = max(b, 1.0);
  float K = 0.0532 * pow(rho, -2.5) + 1.425 * pow(rho, -7.0) + 2.565 * pow(rho, -17.0);
  float lat = asin(clamp(sb.y, -1.0, 1.0)), lon = atan(sb.z, sb.x);
  float A = uActivity;
  float hcs = (0.1 + 0.6 * A) * (0.65 * sin(lon + 0.7) + 0.35 * sin(2.0 * lon + 2.1));
  float wid = mix(0.2, 0.42, A) * (0.3 + 0.7 * exp(-(rho - 1.0) * 1.6));
  float helm = exp(-pow((lat - hcs) / wid, 2.0));
  helm = max(helm, A * 0.85 * exp(-pow((abs(lat) - 0.5) / wid, 2.0)) * (0.5 + 0.5 * sin(3.0 * lon + 1.3)));
  float rays = 0.72 + 0.22 * snoise(sb * 11.0) + 0.1 * snoise(sb * 37.0);
  float plume = pow(abs(sb.y), 5.0) * (1.0 - 0.8 * A) * (0.35 + 0.65 * smoothstep(-0.2, 0.6, snoise(sb * 36.0)));
  float shape = 0.22 + 1.9 * helm * rays + 0.8 * plume;
  return bb(5772.0) * pow(K / 4.04, 0.6) * shape * uCorona * 0.06;
}

// ---- the chromosphere / transition region / low corona seen on the disc through narrow-band filters ----
// Cellular noise that also returns the offset to the nearest feature point (the supergranule centre).
vec4 worleyV(vec3 p){
  vec3 i = floor(p), f = fract(p);
  float d1 = 8.0, d2 = 8.0; vec3 r1 = vec3(0.0);
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(float(x), float(y), float(z));
    vec3 r = o + hash33(i + o) - f;
    float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; r1 = r; } else if (d < d2) d2 = d;
  }
  return vec4(r1, sqrt(d2) - sqrt(d1));
}
// One epoch of the supergranular pattern (≈ 30 Mm cells). Magnetic flux swept to the cell borders makes the
// bright network; dark fibrils (mottles, the on-disc counterparts of spicules) fan out from it into the cells.
vec2 chromoNet(vec3 qq, float lodM){
  vec4 w = worleyV(qq);
  float edge = smoothstep(0.45, 0.0, w.w) * max(0.2 + 0.8 * snoise(qq * 1.7 + 4.0), 0.0);
  // dark mottles cluster around the network; below the pixel scale they average to a darker cell rim
  float rim = smoothstep(0.0, 0.12, w.w) * smoothstep(0.9, 0.1, w.w);
  float mot = 0.35 * rim * (0.6 + 0.4 * snoise(qq * 3.3 + 9.0));
  if (lodM > 0.01){
    // fibrils radiate from the cell centre: noise in the local polar frame (direction, radius) of the cell
    vec3 d = -w.xyz; float r = length(d);
    vec3 cc = qq + w.xyz;                                       // cell centre: a per-cell offset
    mot = mix(mot, smoothstep(-0.3, 0.8, snoise(d / max(r, 1e-3) * 10.0 + cc * 3.1 + r * 1.6)) * rim * 0.8, lodM);
  }
  return vec2(edge, mot);
}
// Hα fibrils: dark, field-aligned streaks fanning out of plage / around sunspots (superpenumbrae),
// tracing the canopy field that spreads from the active region.
float plageFibrils(vec3 q, float fp){
  float fib = 0.0;
  for (int i = 0; i < NS; i++){
    vec4 sp = uSpot[i], x = uSpotX[i];
    if (x.w <= 0.0 || x.z <= 0.0) continue;
    vec3 dv = q - sp.xyz;
    float r = length(dv) / (x.w * 2.0);
    if (r > 1.7 || r < 0.05) continue;
    vec3 e1 = tangentU(sp.xyz), e2 = cross(sp.xyz, e1);
    float ang = atan(dot(dv, e2), dot(dv, e1)) + 0.5 * sin(r * 3.0 + x.y);   // slightly curved
    float lodF = 1.0 - smoothstep(0.004 * x.w * 6.0, 0.02 * x.w * 6.0, fp);
    float n = snoise(vec3(cos(ang) * 9.0, sin(ang) * 9.0, r * 1.2 + x.y));
    float f = mix(0.45, smoothstep(-0.1, 0.7, n), lodF);
    fib = max(fib, f * smoothstep(0.25, 0.55, r) * smoothstep(1.7, 1.0, r) * min(x.z, 1.0));
  }
  return fib;
}
// Disc intensity (1 ≈ quiet-Sun mean) in the current narrow-band view; act = plage / active-region strength.
float chromoI(vec3 q, float mu, float fp){
  float om = omegaAt(q.y * q.y);
  float sgF = uGranF / 28.0;
  float lodN = 1.0 - smoothstep(0.1 / sgF, 0.45 / sgF, fp);
  float lodM = 1.0 - smoothstep(0.012 / sgF, 0.03 / sgF, fp);      // mottles (period ≈ 0.1/sgF) need ≥ 3 px
  vec2 n = vec2(0.0);
  if (lodN > 0.001){
    if (uEpW > 0.001) n += uEpW * chromoNet(rotY(q, -om * uEp.x) * sgF + uEp.z * 7.31, lodM);
    if (uEpW < 0.999) n += (1.0 - uEpW) * chromoNet(rotY(q, -om * uEp.y) * sgF + uEp.w * 7.31, lodM);
    n *= lodN;
  }
  float fac = 0.0, gsup = 1.0;
  float tf = spotField(q, fp, fac, gsup);
  float spot = smoothstep(0.985, 0.72, tf);                     // 0 outside, ~1 in the umbra
  float pen = smoothstep(0.995, 0.93, tf);
  float lodF = 1.0 - smoothstep(0.0003, 0.0009, fp);              // 380⁻¹ R☉ period needs ≥ 3 px
  float fine = snoise(q * 380.0) * lodF;
  float mid = snoise(q * 90.0 + 2.0);                            // plage patchiness at ~5 Mm
  float pl = min(fac, 1.4) * max(0.35 + 0.6 * n.x + 0.35 * mid + 0.2 * fine, 0.0);  // plage: patchy, brightest on the network
  float I;
  if (uView < 1.5){
    // H-alpha line centre: weak limb darkening, bright network & plage, dark mottles and fibrils, dark spots
    float fib = plageFibrils(q, fp);
    float big = snoise(q * 9.0 + 4.0);                           // enhanced / quiet network patches
    I = 0.92 + 0.32 * n.x * (1.0 + 0.5 * big) - 0.5 * n.y + 0.06 * fine + 0.05 * big + 0.6 * pl - 0.32 * fib * (1.0 - min(fac, 1.0) * 0.6);
    I *= (0.6 + 0.4 * sqrt(mu)) * (1.0 - 0.45 * pen - 0.45 * spot);
    I += 0.1 * exp(-mu / 0.03);                                // the chromosphere thickens toward the limb
  } else if (uView < 2.5){
    // He II 304 Å (~50 000–80 000 K): high-contrast network, bright active regions, limb brightening
    I = 0.42 + 0.75 * n.x * (0.7 + 0.3 * fine) - 0.12 * n.y + 1.6 * pl;
    I *= 1.0 + 0.7 * exp(-mu / 0.12);
  } else {
    // Fe IX 171 Å (~0.7 MK): faint mottled disc ("moss"), bright fuzzy active regions, dark coronal holes
    float moss = 0.5 + 0.5 * snoise(q * 45.0) * snoise(q * 13.0 + 3.0);
    float hole = smoothstep(0.72, 0.9, abs(q.y)) * (1.0 - 0.8 * uActivity);
    I = (0.2 + 0.16 * moss + 0.08 * n.x + 1.1 * pow(min(fac, 1.4), 1.3) * (0.5 + 0.5 * fine)) * (1.0 - 0.75 * hole);
    I *= 1.0 + 1.2 * exp(-mu / 0.08);
  }
  return I;
}
float viewK(){ return uView < 1.5 ? 0.6 : (uView < 2.5 ? 0.8 : 0.85); }

// The chromosphere seen edge-on through a narrow-band filter: a dense ~2 Mm layer and above it the
// forest of spicules — jets ~0.5 Mm wide, 5–10 Mm tall, tilted, each living 5–10 min ("burning prairie").
// Sampled in the plane of the sky (sp: unit vector toward the limb point, vh: toward the camera).
float spiculeForest(vec3 sp, vec3 vh, float h, float fpx){
  if (h > 0.02) return 0.0;
  vec3 tang = cross(vh, sp);
  vec3 s = sp + tang * h * (0.8 * snoise(sp * 35.0));            // jets lean in clumps
  const float F = 1150.0;
  float lodS = 1.0 - smoothstep(0.5 / F, 2.5 / F, fpx);
  float n = snoise(s * F + vh * (h * 110.0 - uGranT * 0.12));
  float n2 = snoise(s * 700.0 + vh * (uGranT * 0.05));
  float n3 = snoise(s * 70.0 + 5.0);
  float hn = 0.5 + 0.5 * n2;
  float top = (0.003 + 0.0095 * hn * hn) * (0.75 + 0.35 * n3);               // jet tops: 2–9 Mm, most short
  float jets = mix(0.3, pow(max(n, 0.0), 1.3) * 2.2, lodS) * smoothstep(top, top * 0.15, h) * (0.65 + 0.35 * n3);
  return 0.95 * exp(-h / 0.0011) + 0.45 * jets * exp(-h / 0.007);
}

// ---- prominences --------------------------------------------------------------------------------------
float wave(float x, float s){ return 0.5 * sin(x + s) + 0.3 * sin(2.3 * x + 1.7 * s) + 0.2 * sin(5.1 * x + 2.9 * s); }
// Field-aligned fine structure, built explicitly as threads: in each cell across the field sits one thread
// with a Gaussian cross-section that wanders slowly along the field and is loaded with plasma only along
// some stretches (Hinode SOT: threads ~200–300 km wide, 5–20 Mm long). Band-limiting: the profile is
// convolved with the pixel / march-step footprint bw (widths add in quadrature, the integral is kept), so a
// thread finer than the footprint turns into a smooth haze instead of sparkle.
//   u: position across the field, s: along it (R☉); sp: thread spacing, w: half-width, ka: along-field wavenumber
float threadLayer(float u, float s, float seed, float sp, float w, float bw, float ka){
  float we = sqrt(w * w + 0.3 * bw * bw);
  float f = smoothstep(0.3 * sp, 0.75 * sp, we);                 // fully blurred: just the mean
  if (f > 0.999) return 0.5;
  float x = u / sp, j0 = floor(x), sum = 0.0;
  for (int i = -1; i <= 1; i++){
    float j = j0 + float(i);
    vec3 h = hash33(vec3(j, seed, sp * 977.0));
    float c = j + 0.5 + 0.5 * (h.x - 0.5) + 0.3 * sin(s * ka * (0.5 + 0.6 * h.y) + TAU * h.z);   // gentle wander
    float load = 0.5 + 0.5 * sin(s * ka * (1.0 + 1.5 * h.z) + TAU * h.y);
    load = smoothstep(0.1, 0.9, load);                            // loaded / empty stretches along the thread
    float d = (x - c) * sp / we;
    sum += load * exp(-d * d);
  }
  return mix(sum * (w / we) * sp / (w * 1.7725), 0.5, f);
}
float threads(vec2 c, float s, float seed, float bw, float ka, float fc){
  // strands (~1–3 Mm): smooth noise stretched ×20 along the field, faded to its mean below ~2 footprints
  float n1 = snoise(vec3(c * 300.0, s * 14.0 + seed));
  float st = mix(0.55, smoothstep(-0.35, 0.75, n1), 1.0 - smoothstep(0.0015, 0.005, bw));
  // threads (~0.4 Mm FWHM, 1 Mm apart): explicit band-limited Gaussian threads
  float L2 = threadLayer(c.x + 0.37 * c.y, s, seed + 17.0, 0.0011, 0.00026, bw, ka);
  return (0.15 + 1.55 * st) * (1.0 - 0.7 * fc + 1.4 * fc * L2);                  // mean ≈ 1
}
float gStepZ = 0.0;   // march step projected across the hedgerow sheet (set by promMarch)
float gSkip = 0.0;    // hedgerow: distance (across the sheet) that is certainly empty — lets the march skip it
// Density in the prominence's frame: p.x along the axis (the filament channel), p.y height above the
// photosphere, p.z across. G = (L, H, a, ·), E = (twist, legs, lean, ·). Roughly 0…1.5.
float promDensity(vec3 p, float type, vec4 G, vec4 E, float seed, float bw, float t){
  float L = G.x, H = G.y, a = G.z;
  if (p.y < -0.002) return 0.0;
  if (type < 0.5){
    // Quiescent "hedgerow": a thin vertical sheet of mostly vertical threads, standing on a few legs
    // (barbs) rooted in the chromosphere, with arches of empty corona between them. Plasma constantly
    // drains down the threads while dark plumes rise from below (Hinode SOT, Berger et al. 2008).
    float xn = p.x / L, ax = abs(xn);
    if (ax > 1.1) return 0.0;
    float yn = p.y / H;
    float zc = L * 0.15 * wave(xn * 2.0, seed) + a * (0.8 * wave(xn * 5.0, seed + 7.0) + 2.0 * yn * wave(xn * 1.7, seed + 2.0));   // sinuous, leaning sheet
    float dz = p.z - zc;
    float del0 = a * (0.45 + 0.85 * clamp(yn, 0.0, 1.3));
    // a sheet thinner than the march step (seen obliquely) is widened to ~1 step, keeping its column density
    float del = sqrt(del0 * del0 + 0.5 * gStepZ * gStepZ);
    float mz = dz * dz / (del * del);
    if (mz > 4.5){ gSkip = abs(dz) - 2.1 * del; return 0.0; }
    float top = H * (1.0 - pow(min(ax, 1.0), 4.0)) * (0.78 + 0.28 * wave(xn * 4.0, seed + 5.0));
    if (p.y > top * 1.5) return 0.0;
    // legs at irregular intervals, of different widths; the gaps between them have different heights
    float ph = (xn * 0.5 + 0.5) * E.y + 0.3 * wave(xn * 3.1, seed + 1.0);
    float bot = H * 0.6 * pow(abs(sin(PI * ph)), 0.9 + 0.8 * (0.5 + 0.5 * wave(ph * 1.7, seed + 9.0)))
              * max(0.15 + 0.85 * (0.5 + 0.5 * wave(ph * 2.3, seed + 3.0)), 0.0) * smoothstep(1.0, 0.85, ax);
    float xu = p.x + 0.0006 * wave(p.y * 45.0 + p.x * 150.0, seed);
    float th = threads(vec2(xu, xu + 0.2 * dz), p.y + t * 0.012, seed, bw, 60.0, 0.7);
    float te = min(th, 1.4);                                     // ragged, thread-following outline
    float env = smoothstep(top * (1.0 + 0.3 * te), top * 0.55, p.y)
              * smoothstep(bot - 0.002 - 0.008 * te * yn, bot + 0.007, p.y)
              * exp(-mz) * smoothstep(1.1, 0.8, ax);
    if (env < 0.01) return 0.0;
    // one noise for both: dark plumes rising through the sheet, and brighter, denser knots
    float v = snoise(vec3(p.x * 55.0, (p.y - t * 0.02) * 36.0, seed + 11.0));
    env *= 1.0 - 0.7 * smoothstep(0.35, 0.85, v) * smoothstep(top * 0.95, top * 0.2, p.y);
    env *= 0.75 + 0.7 * smoothstep(-0.2, -0.75, v);
    return env * th * 1.35 * del0 / del;
  }
  if (type < 1.5 || type > 2.5){
    // Flux-rope arch (and the erupting rope): plasma threaded on the helical field lines of a twisted
    // tube whose axis is an elliptical arch between two feet.
    float h0 = 0.25 * H + 2.0 * a;
    float x = (p.x - E.z * p.y) / (1.0 + E.w * smoothstep(0.1, 0.9, p.y / H));   // lean; an erupting rope balloons at the top
    vec2 q = vec2(x / L, (p.y + h0) / (H + h0));
    float rho = length(q);
    float d = (rho - 1.0) * rho / max(length(vec2(q.x / L, q.y / (H + h0))), 1e-5);   // ≈ distance from the axis
    float phi = atan(q.y, q.x);                                  // 0 … π, foot to foot
    float sphi = sin(clamp(phi, 0.0, PI));
    float rad = a * (0.55 + 0.45 * sphi);
    vec2 cr = vec2(d, p.z - a * 1.2 * wave(phi * 2.0, seed));
    float rr = dot(cr, cr) / (rad * rad);
    if (rr > 3.2) return 0.0;
    float tw = TAU * E.x * phi / PI;                             // the cross-section turns along the rope
    float ct = cos(tw), st = sin(tw);
    cr = vec2(ct * cr.x - st * cr.y, st * cr.x + ct * cr.y);
    float sl = (phi - PI * 0.5) * (L + H) * 0.5;                 // ≈ arc length from the apex
    float th = threads(cr * (0.012 / max(a, 0.012)), abs(sl) - t * 0.01, seed, bw, 160.0, 1.0);   // drains down both legs
    float env = smoothstep(1.0 + 0.9 * th, 0.2, rr) * (0.45 + 0.55 * sphi);
    // only some bundles of field lines are loaded with cool plasma: gaps and bright knots along the rope;
    // an erupting rope frays into separate strands and blobs
    float bn = snoise(vec3(cr * (1.1 / max(a, 0.006)), sl * (type > 2.5 ? 7.0 : 4.0) / max(1.0, a / 0.012) + seed * 1.3 - t * 0.004));
    env *= type > 2.5 ? 0.1 + 1.4 * smoothstep(-0.05, 0.75, bn) : 0.3 + 0.95 * smoothstep(-0.35, 0.6, bn);
    return env * th * 1.3;
  }
  // Active-region (post-flare) loop arcade with coronal rain: a row of distinct loops in planes across the
  // axis, each with its own height; plasma cooling at the loop tops condenses into blobs that slide down
  // both legs (~50–100 km/s).
  float xn = p.x / L;
  if (abs(xn) > 1.1) return 0.0;
  float nl = E.y;                                                // number of loops
  float u = (xn * 0.5 + 0.5) * nl;
  float j = floor(u);
  float hj = hash13(vec3(j, seed, 1.7));
  float xj = (j + 0.5 + 0.3 * (hj - 0.5)) / nl * 2.0 - 1.0;      // loop plane
  float h0 = 0.3 * H;
  float R2 = H + h0;
  vec3 hk = hash33(vec3(j, seed, 4.1));
  float Rj = R2 * (0.45 + 0.55 * pow(hk.x, 0.7));
  // each loop has its own footpoint offset, inclination and flattening
  vec2 q = vec2(p.z - 0.12 * H * wave(xn * 2.0, seed) - (hk.y - 0.5) * 0.35 * Rj + (hk.z - 0.5) * 0.8 * p.y, (p.y + h0) / (0.8 + 0.45 * hk.z));
  float rn = length(q);
  float wj = 0.0012 + 0.0016 * hj;                               // loop half-thickness ~1–2 Mm
  float dx = (xn - xj) * L, dr = rn - Rj;
  float tube = exp(-(dx * dx + dr * dr) / (wj * wj));
  if (tube < 0.02) return 0.0;
  float phi = atan(q.y, q.x);
  float sl = abs(phi - PI * 0.5) * Rj;                           // distance from the loop top
  float th = threads(vec2(dx, dr) * 1.3, sl * 0.6 - t * 0.02, seed + j, bw, 160.0, 1.0);
  // rain: soft condensations ~2–4 Mm long along the loop, sliding toward the feet
  float bph = (sl - t * 0.03) * (220.0 + 120.0 * hk.y) + TAU * hk.z + dr * 300.0;
  float blob = pow(0.5 + 0.5 * sin(bph), 4.0) * (0.6 + 0.4 * sin(bph * 0.37 + hj * 9.0));
  float legs = 0.35 + 0.65 * smoothstep(0.2 * Rj, 0.9 * Rj, sl);  // the rain collects in the legs
  return tube * th * (0.25 + 1.6 * blob) * legs * 1.3;
}
// Front-to-back emission + absorption through every prominence bounding box the ray crosses. Returns the
// emitted radiance; "trans" is the remaining transmission (filaments on the disc absorb the light behind).
vec3 promMarch(vec3 ro, vec3 rd, float tEnd, float pixAng, inout float trans){
  vec3 acc = vec3(0.0);
  // step jitter: the densities are band-limited to the step, so a random (unstructured) jitter only leaves
  // a faint grain, which the host's film grain hides; structured (interleaved-gradient) jitter would
  // show as a regular hatch without temporal accumulation
  float jit = hash13(vec3(gl_FragCoord.xy, 3.7));
  float tm = uGranT;
  // Source functions: Hα and 304 Å S ≈ 0.25–0.6 of the disc (dark filaments on the disc); 171 Å: the cool
  // plasma barely emits, it absorbs the EUV behind it (H I / He I / He II continua); white light: faint pink
  // (Balmer lines — seen only at the limb, best during eclipses). Opacity ∝ density, κ chosen so that a dense
  // sheet is optically thick (τ ≈ 3–10) along a few Mm, as for Hα line centre.
  float kap = 190.0 * uProm;
  // Hα: the source function is well below the disc intensity (filaments are dark on the disc). Like the
  // two-exposure composites of solar photographers, light from prominences seen against the sky gets a
  // longer "exposure" (gain 2.2) — the continuous ramp sits in the bright limb band, where it isn't seen.
  float bC = length(ro + rd * max(-dot(ro, rd), 0.0));
  float gHa = mix(1.0, uView < 1.5 ? 2.2 : 3.2, smoothstep(1.0, 1.006, bC) * step(1e8, tEnd));
  for (int k = 0; k < NP; k++){
    vec4 G = uPrC[k];
    if (G.w <= 0.0) continue;
    vec4 A = uPrA[k], B = uPrB[k], D = uPrD[k], E = uPrE[k];
    vec3 N = A.xyz, X = B.xyz, Z = cross(X, N);
    vec3 o = ro - N;
    vec3 lo = vec3(dot(o, X), dot(o, N), dot(o, Z)), ld = vec3(dot(rd, X), dot(rd, N), dot(rd, Z));
    vec3 inv = (step(vec3(0.0), ld) * 2.0 - 1.0) / max(abs(ld), vec3(1e-6));
    vec3 ta = (vec3(-D.x, -D.w, -D.z) - lo) * inv, tb = (D.xyz - lo) * inv;
    vec3 tn = min(ta, tb), tf = max(ta, tb);
    float t0 = max(max(tn.x, tn.y), max(tn.z, 0.0));
    float t1 = min(min(tf.x, tf.y), min(tf.z, tEnd));
    if (t1 <= t0) continue;
    float fp = max(t0, 0.02) * pixAng;
    float len = t1 - t0;
    // fixed step for ropes and arcades; hedgerows (a thin sheet in a wide box) march finer and skip the
    // empty space on either side of the sheet
    bool sheet = A.w < 0.5;
    float dt = max(max(fp * 0.6, 0.0003), len / (sheet ? 240.0 : 72.0));
    if (!sheet) dt = max(dt, len / 72.0);
    int n = sheet ? 96 : int(ceil(len / dt));
    // band limit for the threads: ≥ 1.5 pixel footprints, and ≥ 1.5 march steps measured across the field
    // (a ray seen broadside through a hedgerow sheet runs along its threads and needs no step blur)
    float across = A.w < 0.5 ? max(abs(ld.x), 0.35) : 1.0;
    float bw = max(1.5 * fp, 1.5 * dt * across);
    float tt = t0 + dt * fract(jit + 0.618 * float(k));
    gStepZ = dt * abs(ld.z);
    for (int i = 0; i < 96; i++){
      if (i >= n || tt > t1) break;
      vec3 P = lo + ld * tt;
      vec3 p = vec3(P.x, P.y + 1.0 - sqrt(max(1.0 - P.x * P.x - P.z * P.z, 0.0)), P.z);
      gSkip = 0.0;
      float rho = promDensity(p, A.w, G, E, B.w, bw, tm);
      // skip ahead (in whole steps, so the jittered sampling grid is kept) when far from the sheet
      tt += gSkip > 0.0 ? dt * max(1.0, floor(0.6 * gSkip / max(abs(ld.z), 0.05) / dt)) : dt;
      if (rho <= 0.0) continue;
      rho *= G.w;
      float al = 1.0 - exp(-kap * rho * dt);
      float x = min(rho, 1.6);
      vec3 S = uView < 0.5 ? HALPHA * (0.1 + 0.08 * x)
             : uView < 1.5 ? lutHa((0.12 + 0.3 * x) * gHa)
             : uView < 2.5 ? lut304((0.1 + 0.3 * x) * gHa)
             : lut171(0.05 + 0.03 * x);
      acc += trans * al * S;
      trans *= 1.0 - al;
      if (trans < 0.01) break;
    }
    if (trans < 0.01) break;
  }
  return acc;
}
// CME: the expanding magnetic bubble ahead of an erupting prominence, seen by Thomson scattering of
// photospheric light (the bright front of the classic three-part CME). Column through a thick spherical shell.
float cmeShell(vec3 ro, vec3 rd, float tEnd){
  vec3 oc = uCme.xyz - ro; float tb = dot(oc, rd);
  float R = uCme.w, b2 = dot(oc, oc) - tb * tb;
  if (b2 > R * R || tb < -R) return 0.0;
  float l1 = sqrt(R * R - b2), l2 = sqrt(max(0.64 * R * R - b2, 0.0));
  // chord lengths through the outer and inner spheres, counting only t > 0
  float c1 = max(tb + l1 - max(tb - l1, 0.0), 0.0), c2 = l2 > 0.0 ? max(tb + l2 - max(tb - l2, 0.0), 0.0) : 0.0;
  float col = (c1 - c2) / R;
  vec3 pq = (ro + rd * tb - uCme.xyz) / R, pb = normalize(pq);
  col *= max(0.3 + 0.7 * (0.5 + 0.5 * snoise(pq * 2.5 + uCme.w)) + 0.25 * snoise(pq * 7.0), 0.0);   // loops and knots in the front
  col *= smoothstep(-0.2, 0.4, dot(pb, normalize(uCme.xyz)));   // open at the bottom, where it is rooted
  if (tEnd < 1e8) col *= 0.3;
  return col;
}
#endif

#if KIND == 2
// The τ = 1 surface of a red supergiant is lumpy: convective cells comparable to the radius bulge it
// by several per cent, so the silhouette is irregular. Evaluated in the direction of the point of
// closest approach, which is exactly the limb direction for the pixels that matter.
float rsgRadius(vec3 s, float b){
  return 1.0 + 0.04 * fbm(s * 1.7 + vec3(0.0, uGranT * 0.015, 0.0), 3) * smoothstep(0.2, 0.9, b);
}
vec3 rsgSurface(vec3 q, float mu){
  float t = uGranT;
  vec3 p = q * 1.35 + 0.35 * vec3(snoise(q * 1.6 + t * 0.05), snoise(q * 1.6 + 11.0 - t * 0.04), snoise(q * 1.6 + 23.0));
  float g1 = granule(p, t * 0.6, 3.0);                       // a handful of giant cells
  float g2 = granule(q * 5.0 + 0.25 * p, t * 1.1, 19.0);     // smaller, short-lived cells on top
  float f = fbm(q * 14.0 + vec3(0.0, t * 0.03, 0.0), 3);
  float d = uCell * (0.11 * g1 + 0.012 * g2 + 0.015 * f);
  // Great Dimming: a cooler patch on the southern hemisphere (Montargès et al. 2021)
  d -= uDim * 0.045 * smoothstep(0.1, -0.6, q.y);
  return emit(uTeff * (1.0 + d)) * limbDark(mu);
}
#endif

#if KIND == 4
// α Cygni variables pulsate in many non-radial modes; a few low-degree spherical harmonics, some
// travelling in longitude, make the temperature pattern.
float pulsation(vec3 q, float t){
  vec3 a = rotY(q, t * 0.31), b = rotY(q, -t * 0.19), c = rotY(q, t * 0.13);
  float m20 = (1.5 * q.y * q.y - 0.5) * sin(t * 0.9);
  float m22 = (a.x * a.x - a.z * a.z);
  float m32 = q.y * (b.x * b.z) * 2.0;
  float m33 = c.x * c.x * c.x - 3.0 * c.x * c.z * c.z;
  return 0.6 * m20 + 0.6 * m22 + 1.2 * m32 + 0.7 * m33;
}
#endif

#if KIND == 3
vec3 earthColor(vec3 p, vec3 rd){
  vec3 n = normalize(p - uEarthPos);
  vec3 nb = rotY(n, uTime * 0.01);
  float land = smoothstep(0.04, 0.1, fbm(nb * 1.7 + 3.0, 5));
  float dry = smoothstep(0.0, 0.3, fbm(nb * 3.0 + 9.0, 3));
  float ice = smoothstep(0.78, 0.86, abs(nb.y));
  float cloud = smoothstep(0.05, 0.45, fbm(nb * 3.2 + vec3(uTime * 0.002, 0.0, 0.0), 5) * 0.6 + 0.2 * snoise(nb * 9.0));
  vec3 alb = mix(vec3(0.015, 0.04, 0.11), mix(vec3(0.07, 0.11, 0.04), vec3(0.3, 0.22, 0.12), dry), land);
  alb = mix(alb, vec3(0.75), ice);
  alb = mix(alb, vec3(0.8), cloud * 0.85);
  vec3 L = normalize(-p);
  float diff = max(dot(n, L), 0.0);
  float E = 1.0 / dot(p, p);                         // irradiance from a unit-radius source ∝ 1/d²
  vec3 c = alb * diff * E * bb(uTeff) * uBright * 2.5;
  float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
  c += vec3(0.25, 0.45, 1.0) * rim * smoothstep(-0.2, 0.4, dot(n, L)) * E * uBright * 0.6;
  return c;
}
#endif

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  float tc = max(-dot(ro, rd), 0.0);
  vec3 pc = ro + rd * tc; float b = max(length(pc), 1e-5);
  vec3 s = pc / b;
  float R = uRad;
#if KIND == 2
  R = rsgRadius(s, b);
#endif
  float pxW = max(tc, 0.05) * pixAng;
  float cov = dot(ro, rd) < 0.0 ? smoothstep(R + 0.75 * pxW, R - 0.75 * pxW, b) : 0.0;
#if KIND == 5
  cov = 0.0;     // the star itself is far below a pixel at nebula scale
#endif
  vec3 surf = vec3(0.0); float tHit = 1e9;
  if (cov > 0.0){
    float hh = sqrt(max(R * R - b * b, 0.0));
    tHit = tc - hh;
    vec3 q = normalize(ro + rd * tHit);
    float mu = clamp(hh / R, 0.0, 1.0);
    float fp = tHit * pixAng / max(mu, 0.04);
#if KIND == 0
    if (uView < 0.5) surf = photosphere(q, mu, fp);
    else surf = lutView(chromoI(q, mu, fp) * viewK());
#elif KIND == 1
    surf = photosphere(q, mu, fp);
#elif KIND == 2
    surf = rsgSurface(q, mu);
#elif KIND == 3
    surf = emit(uTeff * (1.0 + 0.003 * fbm(q * 5.0, 2))) * limbDark(mu);
#elif KIND == 4
    float d = 0.03 * uPuls * pulsation(q, uGranT) + 0.006 * fbm(q * 7.0 + vec3(uGranT * 0.02), 3);
    surf = emit(uTeff * (1.0 + d)) * limbDark(mu);
#endif
  }
  // Display stretch: the host's tone curve (ACES + 1/2.2 gamma) flattens contrast the way no solar
  // telescope image is shown. Raising the radiance relative to the disc centre to uStretch (≈ 2.2 ⇒
  // displayed brightness ∝ intensity, like a linear scientific image) restores the true limb darkening and
  // granulation contrast on screen. Above the disc-centre level (flares) the response stays linear.
#if KIND == 0
  if (uView < 0.5)
#endif
  {
    vec3 ref = uBright * bb(uTeff) * pow(uTeff / uTref, uIexp);   // the star's own mean surface
    vec3 x = surf / ref;
    surf = ref * mix(pow(max(x, 0.0), vec3(uStretch)), x, step(1.0, x));
  }
  float tEnd = cov > 0.5 ? tHit : 1e9;
  float skyK = uSkyK;
#if KIND == 0
  if (uView > 0.5) skyK *= 0.25;      // a narrow-band filter passes almost nothing of the night sky
#endif
  vec3 col = mix(sky(rd, 1.0) * skyK, surf, cov);
  float h = max(b - R, 0.0);

#if KIND <= 1
  // off-limb structures are sampled in the plane of the sky through the star's centre, so radial
  // features stay exactly radial on screen whatever the perspective
  vec3 vh = normalize(ro);
  vec3 sp = pc - vh * dot(pc, vh); sp = length(sp) > 1e-5 ? normalize(sp) : s;
  #if KIND == 0
  // Optically thin layers above the limb are drawn from their plane-of-sky value (closest approach b) times
  // the fraction of the line-of-sight column that lies in front of the camera: for a layer of scale height
  // Hs the column is a Gaussian in t around the closest approach, so the fraction is ½·erfc(−t_c/√(2·Hs·b)).
  // From far away this is 1; a camera skimming the surface (limb close-ups) sees half looking horizontally
  // and almost nothing looking up.
  float tcu = -dot(ro, rd);
  vec3 pcu = ro + rd * tcu; float bu = max(length(pcu), 1e-5), hu = max(bu - R, 0.0);
  vec3 spu = pcu / bu;
  #define FRONT(Hs) (0.5 + 0.5 * erfA(tcu / sqrt(2.0 * (Hs) * bu)))
  if (tEnd > 1e8){
    if (uView < 0.5) col += chromosphere(spu, hu) * FRONT(0.002) * (1.0 + 4.0 * uEclipse) + corona(spu, bu) * FRONT(0.25) * (1.0 + 7.0 * uEclipse);
    else if (uView < 1.5) col += lutHa(0.95 * spiculeForest(spu, vh, hu, pxW) * FRONT(0.003) * viewK());
    else if (uView < 2.5){
      float I = 0.75 * spiculeForest(spu, vh, hu, pxW) * FRONT(0.003) + 0.3 * exp(-hu / 0.006) * FRONT(0.006) + 0.05 * exp(-hu / 0.08) * FRONT(0.08);
      col += lut304(I * viewK()) + corona(spu, bu) * 0.15 * FRONT(0.25);
    } else {
      // 171 Å: optically thin ~1 MK plasma, emission ∝ n², so it falls off much faster than the white-light K corona
      vec3 k = corona(spu, bu) / max(uCorona, 1e-3);
      float I = dot(k, vec3(0.3, 0.6, 0.1)) * 9.0 * exp(-(bu - 1.0) / 0.3) * FRONT(0.15) + 0.35 * exp(-hu / 0.02) * FRONT(0.02);
      col += lut171(I * viewK()) * (0.4 + 0.6 * uCorona);
    }
  }
  if (uView > 2.5 && uLoopK > 0.0) col += loopsGlow(ro, rd, tEnd) * uLoopK;
  if (uProm > 0.0 && !(uView < 0.5 && tEnd < 1e8)){
    // in white light a prominence is invisible against the disc, so those rays are skipped
    float tr = 1.0;
    vec3 pe = promMarch(ro, rd, tEnd, pixAng, tr);
    if (uView < 0.5) pe *= 1.0 + 3.0 * uEclipse;
    col = col * (uView < 0.5 ? 1.0 : tr) + pe;
  }
  if (uCme.w > 0.0 && uCmeB > 0.0 && uView != 1.0){
    float c = cmeShell(ro, rd, tEnd) * uCmeB;
    col += c * (uView < 0.5 ? bb(5772.0) * 0.035 * (1.0 + 3.0 * uEclipse) : uView < 2.5 ? lut304(0.1) : lut171(0.22));
  }
  #else
  if (tEnd > 1e8) col += chromosphere(sp, h) * (1.0 + 4.0 * uEclipse);
  if (uProm > 0.0) col += loopsGlow(ro, rd, tEnd) * uProm * (1.0 + uEclipse);
  #endif
  #if KIND == 0
  if (uEclipse > 0.0){
    // the Moon covers the photosphere exactly (angular diameters 0.53° vs 0.52–0.55°)
    float covM = dot(ro, rd) < 0.0 ? smoothstep(1.008 + 0.75 * pxW, 1.008 - 0.75 * pxW, b) : 0.0;
    col = mix(col, vec3(0.0025, 0.003, 0.004), covM * uEclipse);   // faint earthshine on the new Moon
  }
  #endif
#elif KIND == 2
  {
    // extended atmosphere: hot, dense layer right above the photosphere, the molecular "MOLsphere"
    // (~1.3–1.5 R*), then clumpy dust plumes further out (ESO VLT/VISIR-like)
    float lb = log(max(b, 1.0));
    float clump = 0.55 + 0.45 * snoise(s * 4.0 + vec3(lb * 2.0 - uGranT * 0.01));
    float clump2 = fbm(s * 7.0 + vec3(lb * 1.3), 3);
    float atm = tEnd > 1e8 ? exp(-h / 0.035) : 0.0;
    float mol = exp(-h / 0.3) * (0.5 + 0.5 * clump);
    float dust = pow(1.0 / max(b, 1.0), 3.0) * smoothstep(0.0, 0.6, clump2) * smoothstep(1.2, 2.0, b);
    vec3 cAtm = emit(uTeff * 0.93);
    col += cAtm * (0.28 * atm + uEnv * (0.05 * mol)) * (tEnd > 1e8 ? 1.0 : 0.25) + bb(1500.0) * uEnv * 0.05 * dust * uBright;
  }
  if (uDim > 0.001){
    // Great Dimming 2019–20: a dust clump formed in front of the southern hemisphere
    vec3 c = uDust.xyz; float sg = uDust.w;
    float tb = dot(c - ro, rd);
    if (tb > 0.0){
      vec3 pb = ro + rd * tb; vec3 dl = pb - c;
      float lumpy = max(0.75 + 0.5 * fbm(dl * (2.5 / sg) + 4.0, 3), 0.0);
      float tau = uDim * 2.6 * exp(-dot(dl, dl) / (sg * sg)) * lumpy;
      float behind = smoothstep(-sg, sg, tEnd - tb);
      col *= exp(-tau * behind * vec3(0.8, 1.0, 1.25));
      col += bb(3000.0) * uBright * 0.03 * (1.0 - exp(-tau));
    }
  }
#elif KIND == 3
  if (uEarthOn > 0.5){
    float eR = 1.0906;                     // Earth radius / Sirius B radius = 6371 / 5842
    vec3 oc = ro - uEarthPos; float eb = dot(oc, rd);
    float tcE = max(-eb, 0.0); float bE = length(oc + rd * tcE);
    float pxE = max(tcE, 0.05) * pixAng;
    float covE = eb < 0.0 ? smoothstep(eR + 0.75 * pxE, eR - 0.75 * pxE, bE) : 0.0;
    float tE = tcE - sqrt(max(eR * eR - bE * bE, 0.0));
    if (covE > 0.0 && tE < tEnd) col = mix(col, earthColor(ro + rd * tE, rd), covE);
  }
  if (uSirA > 0.5 && tEnd > 1e8){
    // Sirius A (A1V, 9 940 K, 25 L☉) ~20 AU away: a dazzling point, far brighter than any other star
    float c = dot(rd, uSirDir);
    float a = sqrt(max(2.0 * (1.0 - c), 0.0)) / pixAng;
    col += bb(9940.0) * (exp(-a * a * 0.05) * 8.0 + exp(-a * 0.25) * 0.25 + 0.04 / (1.0 + a * a * 0.002));
  }
#elif KIND == 4
  {
    // radiatively driven wind: ρ ∝ 1/(r² v(r)). Columns of ρ (electron-scattered starlight) and ρ²
    // (H-alpha recombination) integrated analytically from the camera along the ray (constant-v part),
    // plus the slow, dense wind base where the β-law velocity is still small.
    float B = max(b, 1.0);
    float at = PI * 0.5 + atan(tc / B);
    float c1 = at / B;
    float c2 = at / (2.0 * B * B * B) + tc / (2.0 * B * B * (B * B + tc * tc));
    if (tEnd < 1e8){ c1 *= 0.5; c2 *= 0.5; }
    float flow = log(B) * 4.0 - uGranT * 0.25;                 // clumps stream outward
    float clump = tEnd > 1e8 ? max(0.75 + 0.3 * snoise(s * 6.0 + vec3(flow)) + 0.12 * snoise(s * 27.0 + vec3(flow * 1.7)), 0.0) : 0.8;
    float base = exp(-(B - 1.0) / 0.12) * (tEnd > 1e8 ? 1.0 : 0.0);
    vec3 w = bb(uTeff) * (0.12 * c2 + 0.0015 * c1) * clump + HALPHA * 0.03 * base * clump;
    col += w * uWind * uBright;
  }
#elif KIND == 5
  {
    // M1-67 around WR 124: gas ejected in an earlier (LBV-like) phase, now ~4 ly out, broken into
    // knots with tails pointing away from the star, photo-ionised by it. Emission ~ n_e n_i with the
    // ionising flux diluted as 1/r^2; dust (seen by JWST in the infrared) is warm and smoother.
    vec2 sh = sphereHit(ro, rd, vec3(0.0), 5.0);
    float t0 = max(sh.x, 0.0), t1 = sh.y;
    if (t1 > t0){
      const int N = 48;
      float dt = (t1 - t0) / float(N);
      float t = t0 + dt * hash13(vec3(gl_FragCoord.xy, 1.0));
      vec3 acc = vec3(0.0); float tau = 0.0;
      for (int i = 0; i < N; i++){
        vec3 p = ro + rd * t; t += dt;
        vec3 q = p * vec3(0.82, 1.0, 1.0);          // mildly bipolar along x
        float r = length(q); vec3 dir = q / max(r, 1e-3);
        float shell = exp(-pow((r - 3.1) / 0.9, 2.0)) + 0.25 * exp(-pow((r - 1.6) / 0.9, 2.0));
        if (shell < 0.02) continue;
        float k1 = snoise(dir * 13.0 + vec3(r * 1.6));                      // knots, ~2:1 radially elongated
        float k2 = snoise(dir * 31.0 + vec3(r * 4.0) + 5.0);
        float knots = smoothstep(0.25, 0.8, k1 + 0.35 * k2) * 1.6;
        float n = shell * mix(0.35, knots * 1.6, uKnots);
        float ion = 1.0 / (0.6 + r * r * 0.12);
        vec3 ha = mix(vec3(1.0, 0.16, 0.22), vec3(1.0, 0.45, 0.3), smoothstep(0.3, 0.9, k2));   // H-alpha + [N II]
        vec3 dustC = mix(vec3(0.75, 0.35, 0.75), vec3(1.0, 0.6, 0.3), smoothstep(-0.2, 0.7, k2));  // warm dust (false colour)
        vec3 e = uIR > 0.5 ? dustC * (0.7 * n * n + 0.04 * shell) * (0.4 + 0.6 * ion) : ha * n * n * ion;
        acc += e * exp(-tau) * dt;
        tau += n * dt * 0.25;
      }
      col = col * exp(-tau * 0.3) + acc * uNeb * 0.22;
    }
    // the star: a point source (its pseudo-photosphere, ~17 R_sun, is far below a pixel)
    vec3 sd = normalize(-ro);
    float a = sqrt(max(2.0 * (1.0 - dot(rd, sd)), 0.0)) / pixAng;
    vec3 sc = bb(uTeff) * (exp(-a * a * 0.05) * 6.0 + exp(-a * 0.12) * 0.35);
    if (uIR > 0.5){
      // JWST's hexagonal mirror: six bright diffraction spikes (+2 faint ones from the strut)
      vec3 up = normalize(uCamUp - sd * dot(uCamUp, sd)), rt = cross(sd, up);
      vec2 d2 = vec2(dot(rd, rt), dot(rd, up)) / pixAng;
      float sp = 0.0;
      for (int k = 0; k < 4; k++){
        float ang = k == 3 ? 0.0 : float(k) * PI / 3.0 + PI / 6.0;
        vec2 ax = vec2(cos(ang), sin(ang));
        float along = abs(dot(d2, ax)), across = abs(dot(d2, vec2(-ax.y, ax.x)));
        sp += (k == 3 ? 0.25 : 1.0) * exp(-across * across * 0.6) * 1.6 / (1.0 + along * 0.08) * exp(-along / 140.0);
      }
      sc += bb(uTeff) * sp;
    }
    col += sc;
  }
#endif
  gl_FragColor = vec4(col, 1.0);
}`;

/* ============================================================================================== */
/* CPU-side helpers                                                                                 */
/* ============================================================================================== */
function hashJS(a, b, c) {
  let h = Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul((b | 0) + 0x632be5ab, 0x165667b1) ^ Math.imul((c | 0) + 0x7f4a7c15, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b); h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const D2R = Math.PI / 180;
// same convention as rotY() in the shader: rotating by +a decreases the longitude atan2(z, x)
function dirLL(lat, lon, out = new THREE.Vector3()) { return out.set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)); }
// exponential approach; jumps straight to the target on an instance's first frame (snapshots)
function ease(v, target, k, inst) {
  if (!inst.__eased) { inst.__eased = true; return target; }
  const n = v + (target - v) * Math.min(1, k);
  return Math.abs(n - target) < 1e-3 ? target : n;
}
const v4arr = n => Array.from({ length: n }, () => new THREE.Vector4());

// Granulation epochs: pattern A is re-seeded at x = 0, pattern B half an epoch later; each is only
// visible while it is "young", so the differential-rotation shear never exceeds E·dΩ/dφ.
function epochs(U, tRot, E) {
  const x = tRot / E, n = Math.floor(x), f = x - n;
  const ageA = f * E;
  const nb = Math.floor(x + 0.5), ageB = (x + 0.5 - nb) * E;
  U.uEp.value.set(ageA, ageB, n % 997, (nb % 997) + 0.5);
  U.uEpW.value = smooth(0.1, 0.25, f) * (1 - smooth(0.75, 0.9, f));
}

// Global stellar quantities for the HUD
function starNumbers(R, T, M) {
  const L = R * R * Math.pow(T / 5772, 4);
  const g = G_SUN * M / (R * R);
  const vesc = 617.7 * Math.sqrt(M / R);
  return { L, g, vesc };
}
function starHud(ctx, R, T, M) {
  const { L, g, vesc } = starNumbers(R, T, M), f = ctx.fmt;
  const rkm = R * R_SUN_KM;
  const ang = rkm < AU_KM ? 2 * Math.asin(rkm / AU_KM) / D2R : null;
  return [
    ['Độ sáng', `${f.num(L, L < 0.1 ? 4 : L < 100 ? 3 : 0)} L☉`],
    ['Hấp dẫn bề mặt', `${f.num(g, g < 1 ? 4 : 1)} m/s² · log g = ${Math.log10(g * 100).toFixed(2)}`],
    ['Vận tốc thoát', `${f.num(vesc, 0)} km/s`],
    ['Nhìn từ 1 AU', ang === null ? 'bạn ở <b>bên trong</b> ngôi sao' : (ang > 0.5 ? ang.toFixed(2) + '°' : (ang * 60).toFixed(ang * 60 < 1 ? 3 : 1) + '′') + ` (Mặt Trời: 0.53°)`],
    ['Ánh sáng đi qua đường kính', f.time(2 * rkm / C_KMS)],
  ];
}

// Planet orbits drawn as an overlay for the supergiants (dashed where hidden by / inside the star)
function makeOrbits(list, scaleRsun) {
  const group = new THREE.Group();
  const mat = (col) => new THREE.ShaderMaterial({
    uniforms: { uCol: { value: new THREE.Color(...col) }, uCam: { value: new THREE.Vector3() }, uOp: { value: 1 } },
    vertexShader: `attribute float aT; varying float vT; varying vec3 vW;
      void main(){ vT = aT; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform vec3 uCol, uCam; uniform float uOp; varying float vT; varying vec3 vW;
      void main(){
        vec3 d = vW - uCam; float L = length(d); d /= L;
        float b = dot(uCam, d); float h = b * b - dot(uCam, uCam) + 1.0;
        bool hidden = h > 0.0 && (-b - sqrt(h)) < L;
        float a = hidden ? 0.35 * step(0.5, fract(vT * 90.0)) : 1.0;
        gl_FragColor = vec4(uCol * a * uOp, 1.0);
      }`,
    transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const labels = [];
  for (const [name, au, col] of list) {
    const r = au * 215.032 / scaleRsun, N = 256, pos = [], at = [];
    for (let i = 0; i < N; i++) { const a = i / N * Math.PI * 2; pos.push(r * Math.cos(a), 0, r * Math.sin(a)); at.push(i / N); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('aT', new THREE.Float32BufferAttribute(at, 1));
    const line = new THREE.LineLoop(g, mat(col)); line.frustumCulled = false;
    group.add(line);
    const lab = makeLabel(`${name} · ${au} AU`, col);
    lab.position.set(r * Math.cos(-0.6), 0, r * Math.sin(-0.6));
    group.add(lab); labels.push(lab);
  }
  group.userData.update = (cam, op) => group.children.forEach(c => { if (c.material.uniforms) { c.material.uniforms.uCam.value.copy(cam); c.material.uniforms.uOp.value = op; } else c.material.opacity = op; });
  return group;
}
function makeLabel(text, col) {
  const cv = document.createElement('canvas'); cv.width = 512; cv.height = 64;
  const g = cv.getContext('2d');
  g.font = '500 30px system-ui, "Segoe UI", sans-serif'; g.textBaseline = 'middle';
  g.fillStyle = `rgb(${col.map(c => Math.round(Math.min(1, c) * 255)).join(',')})`;
  g.fillText(text, 14, 32);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.LinearSRGBColorSpace;
  const m = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false, opacity: 0.85 });
  const s = new THREE.Sprite(m); s.center.set(0, 0.5); s.scale.set(0.16, 0.02, 1);
  return s;
}

/* ============================================================================================== */
/* Shared instance factory                                                                          */
/* ============================================================================================== */
function makeStar(ctx, kind, cfg) {
  const U = {
    ...ctx.shared,
    uTeff: { value: cfg.T }, uTref: { value: cfg.Tref ?? cfg.T }, uIexp: { value: cfg.Iexp ?? 4 }, uBright: { value: cfg.bright ?? 1.2 },
    uSkyK: { value: cfg.sky ?? 0.4 }, uStretch: { value: cfg.stretch ?? 1.6 },
    uLDa: { value: new THREE.Vector3(...cfg.ldA) }, uLDb: { value: new THREE.Vector3(...cfg.ldB) },
    uOmega: { value: new THREE.Vector3() }, uEp: { value: new THREE.Vector4() }, uEpW: { value: 1 },
    uGranF: { value: cfg.granF ?? 600 }, uGranAmp: { value: 1 }, uGranT: { value: 0 }, uRotT: { value: 0 },
    uActivity: { value: 0.5 }, uChromo: { value: 0 }, uCorona: { value: 0 }, uProm: { value: 0 }, uEclipse: { value: 0 },
    uSpotT: { value: new THREE.Vector3(...(cfg.spotT || [0.7, 0.93, 0.04])) },
    uSpot: { value: v4arr(NS) }, uSpotX: { value: v4arr(NS) },
    uLoopA: { value: v4arr(NL) }, uLoopB: { value: v4arr(NL) }, uLoopC: { value: v4arr(NL) },
    uFl: { value: v4arr(NF) }, uFlB: { value: v4arr(NF) },
    uDim: { value: 0 }, uEnv: { value: 1 }, uCell: { value: 1 }, uPuls: { value: 1 }, uWind: { value: 1 }, uRad: { value: 1 },
    uDust: { value: new THREE.Vector4(0, 0, 0, 1) },
    uNeb: { value: 1 }, uKnots: { value: 1 }, uIR: { value: 0 },
    uEarthPos: { value: new THREE.Vector3() }, uEarthOn: { value: 0 }, uSirA: { value: 0 }, uSirDir: { value: new THREE.Vector3(0, 0, 1) },
    uPrA: { value: v4arr(NP) }, uPrB: { value: v4arr(NP) }, uPrC: { value: v4arr(NP) }, uPrD: { value: v4arr(NP) }, uPrE: { value: v4arr(NP) },
    uCme: { value: new THREE.Vector4() }, uCmeB: { value: 0 }, uView: { value: 0 }, uLoopK: { value: 0 },
  };
  const material = new THREE.ShaderMaterial({
    uniforms: U, defines: { KIND: kind, NS, NL, NF, NP },
    vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
  });
  return { U, material };
}

/* ---------- active regions (sunspot groups + faculae) ---------- */
// Region slots live for `life` sim-seconds each; whether a slot is occupied and where its group sits
// are hashes of (slot, epoch), so the spot pattern is a deterministic function of time.
function activeRegions(t, tRot, activity, omegaFn, cfg, out) {
  const regs = [];
  let si = 0;
  for (let r = 0; r < cfg.slots; r++) {
    const life = cfg.life * (0.6 + 0.8 * hashJS(r, 0, 1));
    const tt = t + hashJS(r, 0, 2) * life, k = Math.floor(tt / life), f = tt / life - k;
    if (hashJS(r, k, 3) > activity * 1.1 - 0.02) continue;
    const hemi = hashJS(r, k, 4) < 0.5 ? -1 : 1;
    const lat = hemi * (cfg.latMin + (cfg.latMax - cfg.latMin) * Math.pow(hashJS(r, k, 5), 1.3)) * D2R;
    const lon0 = hashJS(r, k, 6) * Math.PI * 2;
    const size = cfg.sMin + (cfg.sMax - cfg.sMin) * Math.pow(hashJS(r, k, 7), 2.0) * (0.6 + 0.6 * activity);
    const envS = smooth(0, 0.12, f) * (1 - smooth(0.45, 0.85, f));     // spots decay first…
    const envP = smooth(0, 0.08, f) * (1 - smooth(0.7, 1.0, f));       // …faculae linger
    const lonC = lon0 - omegaFn(Math.sin(lat) ** 2) * tRot;
    regs.push({ lat, lon: lonC, size, envS, envP, seed: hashJS(r, k, 8) * 50 });
    // Joy's law: the leading spot sits closer to the equator, tilt ≈ ½·latitude
    const tilt = 0.5 * Math.abs(lat), sep = size * cfg.sep;
    const pairs = cfg.pairs ? [[1, 1, cfg.umbra], [-1, 0.65, cfg.umbra * 0.85]] : [[0, 1, cfg.umbra]];
    for (const [sgn, sc, umb] of pairs) {
      if (si >= NS) break;
      const la = lat - sgn * hemi * Math.sin(tilt) * sep * 0.5;
      const lo = lonC - sgn * Math.cos(tilt) * sep * 0.5 / Math.cos(lat);
      dirLL(la, lo, out.spot[si]);
      out.spot[si].w = size * sc * Math.sqrt(envS) * (envS > 0.02 ? 1 : 0);
      out.spotX[si].set(umb, hashJS(r, k, 9 + sgn) * 40 + si, cfg.fac * envP, size * cfg.facR);
      si++;
    }
  }
  for (; si < NS; si++) { out.spot[si].set(0, 1, 0, 0); out.spotX[si].set(0, 0, 0, 0); }
  return regs;
}

/* ============================================================================================== */
/* 1. Sun                                                                                           */
/* ============================================================================================== */
// Snodgrass & Ulrich (1990) sidereal rotation: Ω = 14.713 − 2.396 sin²φ − 1.787 sin⁴φ  °/day
const SUN_OMEGA_DEG = [14.713, -2.396, -1.787];
const SUN_DPS = 0.05;         // days of solar rotation per sim-second (≈ 85 s per rotation at 6×)

// ---- prominences (CPU side) ----------------------------------------------------------------------------
// 39 candidate prominences (30 quiescent filament channels); each is re-born every `life` sim-seconds at a new place (a hash of its slot and
// cycle), anchored in the photosphere and carried by differential rotation. The NP that matter for the
// current view (limb first, plus the one the camera follows / that erupts) are uploaded to the shader.
const VIEWS = ['wl', 'ha', 'euv304', 'euv171'];
const VIEW_NAME = ['Ánh sáng trắng (quang cầu, ~5 800 K)', 'Hα 656,3 nm (sắc cầu, ~10 000 K)', 'EUV 30,4 nm He II (~80 000 K)', 'EUV 17,1 nm Fe IX (~0,7 triệu K)'];
const PROM_NAME = ['tai lửa yên tĩnh dạng “hàng rào”', 'tai lửa hình vòm (dây từ thông xoắn)', 'hệ vòng sau bùng nổ & mưa nhật hoa', 'tai lửa phun trào'];
// the first cycle of three candidates is placed so the default view opens with them on the limb
const PROM_HEROES = {
  0: { lat: 30, lon: -20, az: 80, L: 0.12, H: 0.06, a: 0.0042 },
  // two long quiescent filaments crossing the disc as seen from the default view (disc centre ≈ lon 70°)
  1: { lat: -30, lon: 50, az: 25, L: 0.22, H: 0.045, a: 0.007 },
  2: { lat: 47, lon: 100, az: -12, L: 0.18, H: 0.04, a: 0.006 },
  30: { lat: -24, lon: 166, az: 70, L: 0.05, H: 0.07, a: 0.009, twist: 1.6, lean: 0.25 },
  35: { lat: 12, lon: -15, az: 5, L: 0.022, H: 0.028 },
};
function promCatalog() {
  const list = [];
  for (let i = 0; i < 39; i++) {
    const type = i < 30 ? 0 : i < 35 ? 1 : 2;
    const life = type === 0 ? 700 + 500 * hashJS(i, 1, 1) : type === 1 ? 350 + 250 * hashJS(i, 1, 1) : 160 + 120 * hashJS(i, 1, 1);
    list.push({ i, type, life, off: PROM_HEROES[i] ? 0.35 : hashJS(i, 1, 2) });
  }
  return list;
}
// Sizes in R☉ (1 R☉ = 696 Mm): quiescent hedgerows 60–350 Mm long (longer and more of them at high activity), 20–50 Mm tall, sheets ~5 Mm thick;
// arches 40–90 Mm across; active-region arcades 15–35 Mm. Quiescent ones sit at mid/high latitude.
function promGeom(pr, k, activity) {
  const h = j => hashJS(pr.i, k, 20 + j);
  const sgn = h(1) < 0.5 ? -1 : 1, T = pr.type;
  const g = { type: T, seed: h(0) * 60, lon: h(9) * Math.PI * 2, twist: 0, legs: 0, lean: 0, on: true };
  let latDeg;
  if (T === 0) {
    latDeg = sgn * (18 + 45 * h(2));
    // more and longer filaments on an active Sun (polar-crown and active-region filament channels)
    Object.assign(g, { az: (h(3) - 0.5) * 1.6, L: (0.06 + 0.2 * h(4)) * (0.7 + 0.6 * activity), H: 0.025 + 0.045 * h(5), a: 0.005 + 0.004 * h(6) });   // sheet + barbs: 7–12 Mm wide filaments on the disc
    g.on = h(7) < 0.45 + 0.5 * activity;
  } else if (T === 1) {
    latDeg = sgn * (10 + 30 * h(2));
    Object.assign(g, { az: h(3) * Math.PI, L: 0.03 + 0.035 * h(4), H: 0.035 + 0.045 * h(5), a: 0.006 + 0.004 * h(6), twist: 0.7 + 1.3 * h(7), lean: (h(8) - 0.5) * 0.6 });
  } else {
    latDeg = sgn * (8 + 22 * h(2));
    Object.assign(g, { az: h(3) * Math.PI, L: 0.012 + 0.016 * h(4), H: 0.016 + 0.018 * h(5), a: 0 });
    g.on = h(7) < 0.25 + 0.85 * activity;
  }
  const hero = k === 0 && PROM_HEROES[pr.i];
  if (hero) { Object.assign(g, hero); latDeg = hero.lat; g.lon = hero.lon * D2R; g.az = hero.az * D2R; g.on = true; }
  g.lat = latDeg * D2R;
  if (T === 0) g.legs = 2 + Math.floor(g.L / 0.03);
  if (T === 2) g.legs = 5 + Math.floor(g.L / 0.003);   // number of loops in the arcade
  return g;
}
// Eruption kinematics (u = wall-clock seconds since onset): a slow rise while the rope is activated, then
// the torus instability takes over — exponential acceleration, h ∝ e^{t/τ} (Kliem & Török 2006). The rope
// expands self-similarly, its twist relaxes (it unwinds), it rotates about the vertical (writhe) and its
// density drops. 27 s on screen ≈ 45 min on the Sun.
const ERUPT_DUR = 27, REAL_PER_WALL = 100;
function eruptState(base, u) {
  const act = smooth(0, 5, u);
  let H = base.H * (1 + 0.35 * act);
  if (u > 5) H += 0.012 * (Math.exp((u - 5) / 3.2) - 1);
  const x = H / base.H;
  return {
    H, L: base.L * (1 + 2.2 * (1 - 1 / Math.sqrt(x))), a: base.a * Math.pow(x, 0.8), twist: base.twist / (1 + 0.15 * (x - 1)),
    bulge: 2.5 * (1 - 1 / Math.sqrt(x)), rot: 0.8 * (1 - 1 / Math.sqrt(x)), dens: Math.pow(x, -0.6) * (1 + 0.3 * act) * (1 - smooth(19, 26, u)),
    v: u > 5 ? 0.012 / 3.2 * Math.exp((u - 5) / 3.2) : base.H * 0.07 * (u < 5 ? 1 : 0),   // R☉ per wall second
  };
}
function promBox(it) {
  let bx, top, bz;
  if (it.type === 0) { bx = it.L * 1.12; top = it.H * 1.5 + 0.004; bz = it.L * 0.16 + it.a * 7.5 + 0.002; }
  else if (it.type === 2) { bx = it.L * 1.16; top = it.H * 1.3 + 0.002; bz = it.H * 1.3 * 1.2 + 0.15 * it.H + 0.002; }
  else { bx = it.L * (1 + (it.bulge || 0)) + Math.abs(it.lean) * it.H + 2.4 * it.a + 0.002; top = it.H + 2.2 * it.a + 0.003; bz = 3.4 * it.a + 0.002; }
  return [bx, top, bz, (bx * bx + bz * bz) / 2 + 0.004];
}

const sun = {
  id: 'sun',
  name: 'Mặt Trời',
  category: 'stars',
  order: 10,
  subtitle: 'Sao lùn vàng G2V · ngôi sao của chúng ta',
  swatch: ['#fff6e0', '#ff9a2a'],
  blurb: 'Bề mặt sôi sục: mỗi <b>hạt sáng</b> (~1 000 km) là một cột plasma nóng trồi lên rồi nguội đi, chìm xuống ở rãnh tối. Phía trên, trong vành nhật hoa triệu độ, những <b>tai lửa</b> — rèm plasma nguội ~10 000 K, đặc gấp trăm lần xung quanh — được từ trường nâng đỡ: ở rìa đĩa chúng sáng hồng trong vạch <b>Hα</b>, trên đĩa chúng thành những <b>sợi tối</b>.',
  intro: 'Nhấn <b>V</b> để đổi bước sóng (thử <b>Hα 656 nm</b>), <b>L</b> để bay tới cận cảnh <b>tai lửa</b> ở rìa đĩa, <b>R</b> để xem một tai lửa <b>phun trào</b>, <b>E</b> cho nhật thực toàn phần.',
  facts: [
    ['Phân loại', 'G2V — sao lùn vàng dãy chính'],
    ['Bán kính', '695 700 km (109 × Trái Đất)'],
    ['Khối lượng', '1,989×10³⁰ kg (333 000 × Trái Đất)'],
    ['Nhiệt độ hiệu dụng', '5 772 K'],
    ['Công suất bức xạ', '3,828×10²⁶ W'],
    ['Tự quay', '25,4 ngày (xích đạo) · ~35 ngày (cực)'],
    ['Nhiệt độ lõi / vành nhật hoa', '15,7 triệu K / 1–2 triệu K'],
    ['Hạt đối lưu', '~1 000 km, sống 8–10 phút'],
    ['Tai lửa', '~10 000 K, cao 10–100 Mm, sợi mảnh ~300 km; sống vài giờ đến vài tháng'],
    ['Tuổi', '4,6 tỷ năm'],
  ],
  unitKm: R_SUN_KM,
  camera: { distance: 3.5, polar: 82, azimuth: 20, min: 1.03, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.25,
  params: [
    { key: 'view', label: 'Bước sóng quan sát', type: 'seg', options: [['wl', 'Ánh sáng trắng'], ['ha', 'Hα 656 nm'], ['euv304', 'EUV 304 Å'], ['euv171', 'EUV 171 Å']], default: 'wl', note: 'Ánh sáng trắng: quang cầu. Hα: sắc cầu — tai lửa sáng ở rìa, sợi tối trên đĩa. 304 Å (He II): sắc cầu–vùng chuyển tiếp. 171 Å (Fe IX): vành nhật hoa ~1 triệu K với các vòng từ. Ảnh hẹp băng là đơn sắc; màu là bảng màu quy ước (như SDO).' },
    { key: 'activity', label: 'Mức hoạt động (vết đen)', type: 'range', min: 0, max: 1, step: 0.01, default: 0.6, note: 'Chu kỳ 11 năm: 0 = cực tiểu (đĩa trơn, vành nhật hoa dẹt theo xích đạo), 1 = cực đại (nhiều nhóm vết đen, các tia vành nhật hoa tỏa mọi hướng).' },
    { key: 'granulation', label: 'Độ tương phản hạt', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Hạt nóng hơn rãnh tối ~200–300 K; thấy rõ khi lại gần (khoảng cách < 1.2).' },
    { key: 'corona', label: 'Vành nhật hoa', type: 'range', min: 0, max: 2, step: 0.01, default: 0.6 },
    { key: 'prominences', label: 'Sắc cầu & tai lửa', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Mật độ plasma trong tai lửa. Ở ánh sáng trắng tai lửa gần như vô hình (chỉ lộ ra khi nhật thực); hãy xem ở Hα.' },
    { key: 'eclipse', label: 'Nhật thực toàn phần', type: 'toggle', default: false, note: 'Mặt Trăng che vừa khít quang cầu (chỉ ở ánh sáng trắng).' },
  ],
  actions: [
    { label: 'Bước sóng', key: 'v', title: 'Đổi bước sóng quan sát', run: ctx => {
      const p = ctx.params; p.view = VIEWS[(VIEWS.indexOf(p.view) + 1) % VIEWS.length];
      ctx.toast({
        wl: 'Ánh sáng trắng: <b>quang cầu</b> 5 800 K. Tai lửa sáng kém đĩa cả triệu lần nên chỉ thấy khi <b>nhật thực</b>.',
        ha: '<b>Hα 656,3 nm</b> — vạch đỏ của hydro: thấy <b>sắc cầu</b> (~10 000 K). Ở rìa, tai lửa phát sáng trên nền trời tối; trên đĩa chúng hấp thụ ánh sáng phía sau thành <b>sợi tối</b> (filament). Viền rìa lởm chởm là rừng <b>gai spicule</b>.',
        euv304: '<b>EUV 304 Å</b> (He II, ~80 000 K, như kính SDO/AIA): mạng lưới sắc cầu sáng, vùng hoạt động rực, tai lửa ở rìa sáng rõ nhất ở bước sóng này.',
        euv171: '<b>EUV 171 Å</b> (Fe IX, ~0,7 triệu K): vành nhật hoa và các <b>vòng từ</b> trên vùng hoạt động. Plasma lạnh của tai lửa ở đây lại <b>tối</b> — nó hấp thụ tia EUV.',
      }[p.view], 7000);
    } },
    { label: 'Cận cảnh rìa', key: 'l', title: 'Bay tới tai lửa ở rìa đĩa (nhấn tiếp để đổi loại, rồi quay về)', run: ctx => sun.__inst?.limbTour(ctx) },
    { label: 'Phun trào', key: 'r', title: 'Cho một tai lửa phun trào (CME)', run: ctx => sun.__inst?.erupt(ctx) },
    { label: 'Nhật thực', key: 'e', title: 'Bật/tắt nhật thực toàn phần', run: ctx => { ctx.params.eclipse = !ctx.params.eclipse; if (ctx.params.eclipse) ctx.params.view = 'wl'; ctx.toast(ctx.params.eclipse ? 'Nhật thực toàn phần: quang cầu bị che, lộ ra <b>vành nhật hoa</b> (1–2 triệu K) và các <b>tai lửa</b> hồng (vạch Balmer của hydro).' : 'Quang cầu sáng trở lại — vành nhật hoa mờ hơn ~một triệu lần nên chìm trong ánh chói.'); } },
  ],
  create(ctx) {
    const { U, material } = makeStar(ctx, 0, {
      T: 5772, Iexp: 5, bright: 0.72, stretch: 2.6, sky: 0.25, granF: 640,
      // per-channel quadratic limb darkening, fitted to Pierce & Slaughter (1977) at ~620 / 540 / 460 nm
      ldA: [0.47, 0.62, 0.80], ldB: [0.20, 0.14, 0.02], spotT: [0.69, 0.95, 0.05],
    });
    const om = SUN_OMEGA_DEG.map(v => v * D2R * SUN_DPS);
    U.uOmega.value.set(...om);
    const omegaFn = s2 => om[0] + om[1] * s2 + om[2] * s2 * s2;
    const spots = { spot: U.uSpot.value, spotX: U.uSpotX.value };
    const regCfg = { slots: 8, life: 420, latMin: 5, latMax: 35, sMin: 0.012, sMax: 0.055, sep: 3.4, pairs: true, umbra: 0.38, fac: 1.0, facR: 3.2 };
    const proms = promCatalog();
    const V3 = THREE.Vector3, Y = new V3(0, 1, 0);
    const camDir = new V3();
    const gone = new Set();
    let regs = [], items = [], erupt = null, lastT = null, nVisible = 0, nFil = 0;
    // camera: 0 free, 1 flying to the featured prominence, 2 following it, 3 flying back to the full disc
    const cam = { mode: 0, i: -1, k: -1, t0: 0, dur: 2.2, fromPos: new V3(), fromTgt: new V3(), fromUp: new V3(0, 1, 0), step: -1 };
    const defMin = sun.camera.min;

    const collect = (t, activity) => {
      const out = [];
      for (const pr of proms) {
        const tt = t + pr.off * pr.life, k = Math.floor(tt / pr.life), f = tt / pr.life - k;
        if (gone.has(pr.i + ':' + k)) continue;
        const isE = !!erupt && erupt.i === pr.i && erupt.k === k;
        const env = smooth(0, 0.12, f) * (1 - smooth(0.85, 1, f));
        if (env <= 0.01 && !isE) continue;
        const g = promGeom(pr, k, activity);
        if (!g.on) continue;
        const lonW = g.lon - omegaFn(Math.sin(g.lat) ** 2) * t;
        const C = dirLL(g.lat, lonW);
        const T = new V3(-Math.sin(lonW), 0, Math.cos(lonW)).multiplyScalar(Math.cos(g.az))
          .addScaledVector(new V3(-Math.sin(g.lat) * Math.cos(lonW), Math.cos(g.lat), -Math.sin(g.lat) * Math.sin(lonW)), Math.sin(g.az));
        const it = { pr, k, g, C, T, type: g.type, L: g.L, H: g.H * (0.55 + 0.45 * env), a: g.a, twist: g.twist, lean: g.lean, legs: g.legs, dens: env, env, erupting: isE };
        if (isE) {
          const base = g.type === 0 ? { L: g.L * 0.6, H: g.H, a: Math.max(g.a * 2.5, 0.008), twist: 1.6 } : { L: g.L, H: g.H, a: g.a, twist: g.twist };
          const e = eruptState(base, erupt.age);
          Object.assign(it, { type: 3, L: e.L, H: e.H, a: e.a, twist: e.twist, dens: e.dens, lean: g.lean * 0.5, bulge: e.bulge });
          it.T.applyAxisAngle(C, e.rot);
          erupt.cur = { C, H: e.H, v: e.v, base };
        }
        out.push(it);
      }
      return out;
    };
    const findItem = (i, k) => items.find(it => it.pr.i === i && it.k === k);
    const biggest = (type, pred = () => true) => {
      let best = null, bs = -1;
      for (const it of items) {
        if ((type >= 0 && it.type !== type) || it.env < 0.5 || !pred(it)) continue;
        const s = it.L * it.H * (1.2 - Math.abs(it.C.dot(camDir)));
        if (s > bs) { bs = s; best = it; }
      }
      return best;
    };
    // camera pose that shows prominence `it` exactly on the limb: looking along the surface, broadside for
    // sheets and ropes, along the axis for an arcade (so the loops are seen in profile)
    const framing = (it, ctx, pos, tgt) => {
      const W = new V3().crossVectors(it.T, it.C).normalize();
      const D = it.type === 2 ? it.T.clone() : W;
      if (D.dot(camDir) < 0) D.negate();
      const tanV = Math.tan(ctx.camera.fov * D2R / 2), asp = ctx.camera.aspect || 16 / 9;
      const H = Math.min(it.H, 1.6);
      const halfW = (it.type === 2 ? 1.5 * it.H + 0.3 * it.L : 1.15 * it.L) + 0.012;
      const V = Math.max((it.type === 3 ? 0.72 : 0.62) * H + 0.02, halfW / asp, 0.05);   // ≥ 0.05: keeps the camera above the spicules
      const dist = V / tanV;
      tgt.copy(it.C).multiplyScalar(1 + (it.type === 3 ? 0.48 : 0.42) * H);
      pos.copy(it.C).addScaledVector(D, dist);    // in the tangent plane at the foot: the foot sits exactly on the limb
    };
    const setFeatured = (it, ctx, instant) => {
      cam.i = it.pr.i; cam.k = it.k;
      ctx.controls.minDistance = 0.04;
      if (instant) { framing(it, ctx, ctx.camera.position, ctx.controls.target); ctx.camera.up.copy(it.C); ctx.camera.lookAt(ctx.controls.target); cam.mode = 2; return; }
      cam.mode = 1; cam.t0 = ctx.time.wall; cam.dur = 2.4;
      cam.fromPos.copy(ctx.camera.position); cam.fromTgt.copy(ctx.controls.target); cam.fromUp.copy(ctx.camera.up);
    };
    let prevView = null;
    const toHalpha = (ctx) => { if (ctx.params.view === 'wl' && !ctx.params.eclipse) { prevView = 'wl'; ctx.params.view = 'ha'; } };
    const exitLimb = (ctx) => {
      if (prevView) { ctx.params.view = prevView; prevView = null; }
      cam.mode = 3; cam.i = -1; cam.t0 = ctx.time.wall; cam.dur = 2.2; cam.step = -1;
      cam.fromPos.copy(ctx.camera.position); cam.fromTgt.copy(ctx.controls.target); cam.fromUp.copy(ctx.camera.up);
    };
    const typeToast = (it, ctx) => {
      const base = [
        'Tai lửa yên tĩnh dạng <b>“hàng rào”</b>: một tấm plasma ~10 000 K, đặc gấp ~100 lần vành nhật hoa xung quanh, nằm trong các <b>võng</b> của đường sức từ. Nó gồm hàng nghìn <b>sợi mảnh</b> ~300 km; plasma liên tục <b>rơi xuống</b> dọc các sợi, còn những bọt tối trồi lên từ bên dưới. Các “chân” cắm xuống sắc cầu. Có thể tồn tại hàng tháng.',
        'Tai lửa <b>hình vòm</b>: plasma lạnh nằm trên các đường sức <b>xoắn ốc</b> của một <b>dây từ thông</b>, hai chân cắm xuống quang cầu. Nhấn <b>R</b> để nó mất cân bằng và phun trào.',
        '<b>Hệ vòng sau bùng nổ</b>: plasma bị đốt tới hàng chục triệu K trong một vụ bùng nổ nguội dần, ngưng tụ thành giọt — <b>mưa nhật hoa</b> trượt xuống hai chân vòng với vận tốc ~50–100 km/s.',
        'Tai lửa <b>phun trào</b>: dây từ thông bị đẩy lên, nở ra và tự tháo xoắn.',
      ][it.type];
      ctx.toast(base + (prevView ? ' <br>Đã chuyển sang <b>Hα</b> — ở ánh sáng trắng tai lửa gần như vô hình.' : ''), 8000);
    };
    const pickEruptCandidate = () => {
      const f = cam.mode === 1 || cam.mode === 2 ? findItem(cam.i, cam.k) : null;
      if (f && f.type <= 1) return f;
      // the biggest arch or hedgerow close to the visible limb
      let best = null, bs = -1;
      for (const it of items) {
        if (it.type > 1 || it.env < 0.5) continue;
        const s = it.L * it.H * Math.exp(-Math.abs(it.C.dot(camDir)) * 6) * (it.type === 1 ? 1.5 : 1);
        if (s > bs) { bs = s; best = it; }
      }
      return best;
    };

    const inst = {
      material,
      limbTour(ctx) {
        if (erupt && erupt.cur && cam.i !== erupt.i) { const it = findItem(erupt.i, erupt.k); if (it) { setFeatured(it, ctx); return; } }
        // cycle hedgerow → arch → arcade → back to the full disc
        for (let s = cam.step + 1; s <= 3; s++) {
          if (s === 3) { exitLimb(ctx); ctx.toast('Quay về toàn đĩa.'); return; }
          const it = biggest(s);
          if (it) { cam.step = s; toHalpha(ctx); setFeatured(it, ctx); typeToast(it, ctx); return; }
        }
      },
      erupt(ctx, age = 0, quiet = false) {
        if (erupt) { if (!quiet) ctx.toast('Một tai lửa đang phun trào — chờ nó rời khỏi Mặt Trời đã.'); return; }
        const it = pickEruptCandidate();
        if (!it) { if (!quiet) ctx.toast('Không có tai lửa nào đủ lớn ở rìa đĩa lúc này — thử lại sau ít giây.'); return; }
        erupt = { i: it.pr.i, k: it.k, age };
        if (cam.mode !== 1 && cam.mode !== 2) cam.step = 0;
        if (!quiet) {
          toHalpha(ctx);
          setFeatured(it, ctx);
          ctx.audio?.tone?.(55, 4, 0.2, 0, 'sawtooth');
          ctx.toast('<b>Phun trào tai lửa!</b> Dây từ thông xoắn mất cân bằng (<b>bất ổn định xuyến</b>), bung lên, nở ra và tháo xoắn, đạt vận tốc hàng trăm km/s. Nó kéo theo một <b>CME</b> — bong bóng hàng tỷ tấn plasma. Nếu hướng về Trái Đất, CME gây bão địa từ và cực quang (như tháng 5/2024).', 9000);
        }
      },
      update(dt, ctx) {
        const p = ctx.params, t = ctx.time.sim;
        const dts = lastT === null ? 0 : t - lastT; lastT = t;
        U.uGranT.value = t * 0.12;
        U.uRotT.value = t;
        epochs(U, t, 160);
        U.uActivity.value = p.activity; U.uGranAmp.value = p.granulation;
        const view = Math.max(0, VIEWS.indexOf(p.view));
        U.uView.value = view;
        // like a solar telescope's exposure: high-resolution close-ups are exposed a little darker so the
        // granules don't clip, the full disc a little brighter so its centre reads white
        U.uBright.value = 0.42 + 0.3 * smooth(1.1, 2.0, ctx.camera.position.length());
        U.uCorona.value = p.corona; U.uChromo.value = 0.18 * p.prominences; U.uProm.value = p.prominences;
        U.uEclipse.value = ease(U.uEclipse.value, p.eclipse && view === 0 ? 1 : 0, dt * 2.5, this);
        regs = activeRegions(t, t, p.activity, omegaFn, regCfg, spots);
        camDir.copy(ctx.camera.position).normalize();

        // snapshot hooks for the headless checker: ?sunlimb=<0 hedgerow|1 arch|2 arcade> ?erupt=<seconds>
        if (!this.__dbg) {
          this.__dbg = true;
          const q = new URLSearchParams(location.search);
          items = collect(t, p.activity);
          if (q.get('sunlimb') !== null) { const it = biggest(+q.get('sunlimb')); if (it) { cam.step = it.type; setFeatured(it, ctx, true); } }
          // ?sunact=<keys>: run the dock actions in order (exercises the real fly-in / eruption code paths)
          for (const ch of q.get('sunact') || '') ({ l: () => this.limbTour(ctx), r: () => this.erupt(ctx), v: () => sun.actions[0].run(ctx) })[ch]?.();
          if (q.get('erupt') !== null) {
            this.erupt(ctx, +q.get('erupt'), true);
            if (erupt && q.get('sunlimb') !== null) { items = collect(t, p.activity); const it = findItem(erupt.i, erupt.k); if (it) setFeatured(it, ctx, true); }
          }
        }
        if (erupt) {
          erupt.age += dt;
          if (erupt.age > ERUPT_DUR) { gone.add(erupt.i + ':' + erupt.k); erupt = null; U.uCme.value.w = 0; U.uCmeB.value = 0; }
        }
        items = collect(t, p.activity);

        // ---- upload the prominences that matter for this view ----
        const feat = cam.mode === 1 || cam.mode === 2 ? findItem(cam.i, cam.k) : null;
        const ranked = [];
        for (const it of items) {
          const sc = it.C.dot(camDir);
          const pri = it.erupting || it === feat ? -10 : Math.abs(sc);
          if (pri > -1 && sc < -0.45) continue;
          ranked.push([pri, it, it.erupting || it === feat ? 1 : smooth(-0.45, -0.3, sc)]);
        }
        ranked.sort((a, b) => a[0] - b[0]);
        nVisible = 0; nFil = 0;
        for (let s = 0; s < NP; s++) {
          const r = ranked[s];
          if (!r) { U.uPrC.value[s].set(0, 0, 0, 0); continue; }
          const it = r[1], [bx, top, bz, depth] = promBox(it);
          if (Math.abs(it.C.dot(camDir)) < 0.25) nVisible++;
          else if (it.C.dot(camDir) > 0.25 && it.type <= 1) nFil++;
          U.uPrA.value[s].set(it.C.x, it.C.y, it.C.z, it.type);
          U.uPrB.value[s].set(it.T.x, it.T.y, it.T.z, it.g.seed);
          U.uPrC.value[s].set(it.L, it.H, it.a, it.dens * r[2] * (it.type === 2 ? 1.2 : 1));
          U.uPrD.value[s].set(bx, top, bz, depth);
          U.uPrE.value[s].set(it.twist, it.legs, it.lean, it.bulge || 0);
        }
        // CME bubble ahead of the erupting rope
        if (erupt && erupt.cur) {
          const u = erupt.age, hle = erupt.cur.H * 1.6 + 0.04, Rb = 0.42 * hle;
          U.uCme.value.set(erupt.cur.C.x, erupt.cur.C.y, erupt.cur.C.z, 0).multiplyScalar(1 + hle - Rb);
          U.uCme.value.w = Rb;
          U.uCmeB.value = smooth(5, 9, u) * (1 - smooth(20, 27, u));
        }

        // ---- 171 Å coronal loops over the active regions: arcs joining the two polarities of each group ----
        let li = 0;
        if (view === 3) {
          for (let s = 0; s + 1 < NS && li < NL; s += 2) {
            const xa = U.uSpotX.value[s], xb = U.uSpotX.value[s + 1];
            if (xa.z <= 0.05 || xb.z <= 0.05) continue;
            const A = new V3(U.uSpot.value[s].x, U.uSpot.value[s].y, U.uSpot.value[s].z), B = new V3(U.uSpot.value[s + 1].x, U.uSpot.value[s + 1].y, U.uSpot.value[s + 1].z);
            const d = A.distanceTo(B), m = A.clone().add(B).normalize(), ax = B.clone().sub(A).normalize(), perp = new V3().crossVectors(m, ax);
            for (const [off, hk, w] of [[0, 0.55, 1], [0.35, 0.8, 0.7], [-0.3, 0.42, 0.8]]) {
              if (li >= NL) break;
              const a2 = A.clone().addScaledVector(perp, off * d).normalize(), b2 = B.clone().addScaledVector(perp, -off * d * 0.6).normalize();
              U.uLoopA.value[li].set(a2.x, a2.y, a2.z, d * hk);
              U.uLoopB.value[li].set(b2.x, b2.y, b2.z, s * 3.7 + off * 10);
              U.uLoopC.value[li].set(0, 0.55 * w * Math.min(1, xa.z), 0.0022 + 0.004 * d, 0);
              li++;
            }
          }
        }
        for (; li < NL; li++) U.uLoopC.value[li].set(0, 0, 0, 0);
        U.uLoopK.value = view === 3 ? 0.6 * (0.4 + 0.6 * p.corona) : 0;

        // ---- camera: fly to / follow the featured prominence (OrbitControls still works around it) ----
        const camera = ctx.camera, target = ctx.controls.target;
        if (cam.mode === 1 || cam.mode === 2) {
          if (!feat) { if (cam.mode === 1) cam.mode = 2; }
          else {
            // co-rotate with the Sun so the prominence stays on the limb
            const w = omegaFn(Math.sin(feat.g.lat) ** 2) * dts;
            if (cam.mode === 1) {
              const k = Math.min(1, (ctx.time.wall - cam.t0) / cam.dur), e = 1 - Math.pow(1 - k, 3);
              cam.fromPos.applyAxisAngle(Y, w); cam.fromTgt.applyAxisAngle(Y, w);
              const P = new V3(), Tg = new V3();
              framing(feat, ctx, P, Tg);
              camera.position.lerpVectors(cam.fromPos, P, e); target.lerpVectors(cam.fromTgt, Tg, e);
              cam.fromUp.applyAxisAngle(Y, w);
              camera.up.lerpVectors(cam.fromUp, feat.C, e).normalize();
              camera.lookAt(target);
              if (k >= 1) cam.mode = 2;
            } else if (feat.erupting) {
              const P = new V3(), Tg = new V3();
              framing(feat, ctx, P, Tg);
              const k = 1 - Math.exp(-dt * 1.5);
              camera.position.applyAxisAngle(Y, w).lerp(P, k); target.applyAxisAngle(Y, w).lerp(Tg, k);
              camera.up.copy(feat.C);
            } else {
              camera.position.applyAxisAngle(Y, w); target.applyAxisAngle(Y, w);
              camera.up.copy(feat.C);
            }
          }
        } else if (cam.mode === 3) {
          const k = Math.min(1, (ctx.time.wall - cam.t0) / cam.dur), e = 1 - Math.pow(1 - k, 3);
          const P = camDir.clone().multiplyScalar(sun.camera.distance);
          camera.position.lerpVectors(cam.fromPos, P, e); target.lerpVectors(cam.fromTgt, new V3(), e);
          camera.up.lerpVectors(cam.fromUp, Y, e).normalize();
          camera.lookAt(target);
          if (k >= 1) { cam.mode = 0; ctx.controls.minDistance = defMin; camera.up.copy(Y); }
        }
        if (camera.position.length() < 1.004) camera.position.setLength(1.004);
      },
      hud(ctx) {
        let g = 0, s = 0;
        for (const r of regs) if (r.envS > 0.05) { g++; s += r.size > 0.016 ? 2 + Math.round(r.size * 300) : 1; }
        const lat = Math.asin(Math.max(-1, Math.min(1, ctx.camera.position.y / ctx.camera.position.length())));
        const P = 360 / (SUN_OMEGA_DEG[0] + SUN_OMEGA_DEG[1] * Math.sin(lat) ** 2 + SUN_OMEGA_DEG[2] * Math.sin(lat) ** 4);
        const rows = [
          ['Bước sóng', VIEW_NAME[Math.max(0, VIEWS.indexOf(ctx.params.view))]],
          ['Số vết đen Wolf (R = 10g + s)', `${10 * g + s} · ${g} nhóm`],
          ['Tai lửa gần rìa đĩa · sợi tối trên đĩa', `${nVisible} · ${nFil}`],
        ];
        const f = cam.mode === 1 || cam.mode === 2 ? findItem(cam.i, cam.k) : null;
        if (f && !f.erupting) rows.push(['Đang xem', `${PROM_NAME[f.type]} · dài ${Math.round(2 * f.L * 696)} Mm, cao ${Math.round(f.H * 696)} Mm`]);
        if (erupt && erupt.cur) {
          rows.push(['Phun trào: độ cao', `${Math.round(erupt.cur.H * 696).toLocaleString('vi-VN')} Mm (${(erupt.cur.H).toFixed(2)} R☉)`]);
          rows.push(['Phun trào: vận tốc', `~${Math.round(erupt.cur.v * R_SUN_KM / REAL_PER_WALL)} km/s`]);
        }
        rows.push([`Chu kỳ quay ở vĩ độ ${Math.abs(lat / D2R).toFixed(0)}°`, `${P.toFixed(1)} ngày`], ...starHud(ctx, 1, 5772, 1));
        return rows;
      },
      onClick(info, ctx) {
        const q = hitUnit(info.origin, info.dir); if (!q) return;
        const lat = Math.asin(q.y);
        const P = 360 / (SUN_OMEGA_DEG[0] + SUN_OMEGA_DEG[1] * Math.sin(lat) ** 2 + SUN_OMEGA_DEG[2] * Math.sin(lat) ** 4);
        let inSpot = false;
        for (const sp of U.uSpot.value) if (sp.w > 0 && q.distanceTo(new THREE.Vector3(sp.x, sp.y, sp.z)) < sp.w) inSpot = true;
        ctx.toast(`Vĩ độ <b>${(lat / D2R).toFixed(1)}°</b>: quay một vòng mất <b>${P.toFixed(1)} ngày</b> (vận tốc ${(2 * Math.PI * R_SUN_KM * Math.cos(lat) / (P * 86400)).toFixed(2)} km/s).` +
          (inSpot ? ' Bạn chạm vào một <b>vết đen</b>: từ trường ~0,3 tesla chặn đối lưu, nhiệt độ chỉ ~4 000 K.' : ''));
      },
      exit(ctx) {
        // leave the shared camera controls as we found them
        if (cam.mode !== 0) { cam.mode = 0; cam.i = -1; cam.step = -1; ctx.controls.target.set(0, 0, 0); ctx.camera.up.set(0, 1, 0); }
        ctx.controls.minDistance = defMin;
      },
    };
    sun.__inst = inst;
    return inst;
  },
};

/* ============================================================================================== */
/* 2. Red dwarf — Proxima Centauri                                                                  */
/* ============================================================================================== */
const PROX = { R: 0.1542, T: 3042, M: 0.1221 };
const PROX_DPS = 0.25;          // days per sim-second (rotation 83 d → 330 sim-s)
// Planck ratio in the g′ band (477 nm), used for the "how much brighter" read-out
const planck = (T, lamNm) => 1 / (Math.exp(1.4388e7 / (lamNm * T)) - 1);

const redDwarf = {
  id: 'red-dwarf',
  name: 'Sao lùn đỏ',
  category: 'stars',
  order: 20,
  subtitle: 'Proxima Centauri · M5.5Ve · sao lóe',
  swatch: ['#ffb070', '#7a1d08'],
  blurb: 'Loại sao phổ biến nhất trong Ngân Hà. Toàn bộ thân sao là một khối đối lưu, khuấy từ trường mạnh: <b>vết sao</b> phủ những vùng rộng lớn, và các vụ <b>bùng nổ</b> (flare) có thể làm ngôi sao sáng lên hàng chục lần chỉ trong vài phút — hiểm họa cho các hành tinh như Proxima b.',
  intro: 'Nhấn <b>B</b> để tái hiện siêu bùng nổ tháng 3/2016, hoặc <b>click</b> lên bề mặt để kích một vụ bùng nổ tại đó.',
  facts: [
    ['Phân loại', 'M5.5Ve — sao lùn đỏ, sao lóe (UV Ceti)'],
    ['Khoảng cách', '4,246 năm ánh sáng — gần Mặt Trời nhất'],
    ['Bán kính', '0,154 R☉ (~107 000 km, cỡ 1,5 Sao Mộc)'],
    ['Khối lượng', '0,122 M☉'],
    ['Nhiệt độ hiệu dụng', '~3 040 K'],
    ['Độ sáng', '0,0017 L☉ (phần lớn ở hồng ngoại)'],
    ['Tự quay', '~83 ngày'],
    ['Siêu bùng nổ 3/2016', 'sáng gấp ~68 lần, thấy được bằng mắt thường'],
    ['Hành tinh', 'Proxima b — 11,2 ngày, trong vùng ở được'],
  ],
  unitKm: PROX.R * R_SUN_KM,
  camera: { distance: 3.5, polar: 80, azimuth: 30, min: 1.06, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.35,
  params: [
    { key: 'spots', label: 'Vết sao', type: 'range', min: 0, max: 1, step: 0.01, default: 0.7, note: 'Vết sao trên sao lùn M lớn và có thể nằm ở mọi vĩ độ, kể cả gần cực.' },
    { key: 'flareRate', label: 'Tần suất bùng nổ ngẫu nhiên', type: 'range', min: 0, max: 1, step: 0.01, default: 0.4 },
    { key: 'granulation', label: 'Độ tương phản hạt', type: 'range', min: 0, max: 2, step: 0.01, default: 0.8 },
  ],
  actions: [
    { label: 'Bùng nổ', key: 'b', title: 'Siêu bùng nổ (như 3/2016)', run: ctx => ctx.scene.__inst?.superflare(ctx) },
  ],
  create(ctx) {
    const { U, material } = makeStar(ctx, 1, {
      T: PROX.T, Iexp: 4, bright: 0.72, sky: 0.45, granF: 520,
      // M dwarf, quadratic coefficients in the spirit of Claret (2000) PHOENIX fits, strongest in blue
      ldA: [0.50, 0.62, 0.74], ldB: [0.24, 0.22, 0.18], spotT: [0.86, 0.95, 0.0],
    });
    const om = 360 / 83 * D2R * PROX_DPS;
    U.uOmega.value.set(om, 0, 0);
    U.uChromo.value = 0.35; U.uProm.value = 1;
    const omegaFn = () => om;
    const spots = { spot: U.uSpot.value, spotX: U.uSpotX.value };
    const regCfg = { slots: 13, life: 700, latMin: 0, latMax: 75, sMin: 0.05, sMax: 0.17, sep: 0, pairs: false, umbra: 0.62, fac: 0.4, facR: 1.8 };
    const flares = [];
    const camDir = new THREE.Vector3();
    let boostShown = 1, nextRand = 0;
    const spawn = (body, o) => {
      if (flares.length >= NF) flares.shift();
      flares.push({ lat: Math.asin(body.y), lon: Math.atan2(body.z, body.x), age: 0, axis: Math.random() * Math.PI, ...o });
      ctx.audio?.click?.(0.5 * o.peak, 0, 0.9);
    };
    // body-frame direction of the visible hemisphere's centre (+ random scatter)
    const bodyFromWorld = (w, t) => { const lon = Math.atan2(w.z, w.x) + om * t; return dirLL(Math.asin(w.y), lon); };
    const inst = {
      material,
      superflare(ctx) {
        camDir.copy(ctx.camera.position).normalize();
        const w = camDir.clone().add(new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(0.7)).normalize();
        spawn(bodyFromWorld(w, ctx.time.sim), { peak: 1, size: 0.2, T: 9500, tauR: 1.5, tauD: 22, big: true });
        ctx.audio?.tone?.(70, 2.5, 0.25, 0, 'sawtooth');
        ctx.toast('<b>Siêu bùng nổ!</b> Năng lượng từ trường tích tụ trên vết sao được giải phóng trong vài phút: hai <b>dải sáng</b> ~10 000 K, vòng plasma nóng phía trên, cả ngôi sao trắng lóa. Ngày 18/3/2016 Proxima sáng gấp ~<b>68 lần</b>.', 7000);
      },
      update(dt, ctx) {
        const p = ctx.params, t = ctx.time.sim, sdt = ctx.time.dt;
        U.uGranT.value = t * 0.12; U.uRotT.value = t;
        epochs(U, t, 400);
        U.uActivity.value = p.spots; U.uGranAmp.value = p.granulation;
        activeRegions(t, t, p.spots, omegaFn, regCfg, spots);
        // random small flares (Poisson; Proxima flares many times a day)
        if (p.flareRate > 0 && t > nextRand) {
          nextRand = t + (-Math.log(Math.random() + 1e-9)) * 14 / p.flareRate;
          if (nextRand - t < 200) {
            const lat = (Math.random() * 2 - 1) * 1.2, lon = Math.random() * Math.PI * 2;
            spawn(dirLL(lat, lon), { peak: 0.5 + 0.5 * Math.random(), size: 0.035 + 0.04 * Math.random(), T: 8500, tauR: 0.6, tauD: 4 + 5 * Math.random() });
          }
        }
        // debug hook for the headless checker: ?starflare=<age> fires a superflare at load
        if (!inst.__dbg) { inst.__dbg = true; const a = new URLSearchParams(location.search).get('starflare'); if (a !== null) { inst.superflare(ctx); flares[flares.length - 1].age = +a; } }
        camDir.copy(ctx.camera.position).normalize();
        let boost = 1; const rg = planck(9500, 477) / planck(PROX.T, 477);
        for (let i = flares.length - 1; i >= 0; i--) { flares[i].age += sdt; if (flares[i].age > flares[i].tauR + flares[i].tauD * 7) flares.splice(i, 1); }
        let li = 0;
        for (let i = 0; i < NF; i++) {
          const f = flares[i];
          if (!f) { U.uFl.value[i].set(0, 1, 0, 0); continue; }
          const L = f.age < f.tauR ? (f.age / f.tauR) ** 2 : Math.exp(-(f.age - f.tauR) / f.tauD);
          const w = dirLL(f.lat, f.lon - om * t);
          U.uFl.value[i].set(w.x, w.y, w.z, L * f.peak);
          const sepGrow = 0.12 + 0.55 * (1 - Math.exp(-f.age / (f.tauD * 0.8)));
          U.uFlB.value[i].set(f.size, sepGrow, f.axis, f.T);
          boost += 1.55 * f.size * f.size * L * f.peak * Math.max(0, w.dot(camDir)) * (rg - 1);
          // post-flare loop arcade: loops rooted in the two ribbons, they brighten later (gradual phase)
          const Lg = f.age < f.tauR * 2 ? (f.age / (f.tauR * 2)) : Math.exp(-(f.age - 2 * f.tauR) / (f.tauD * 1.6));
          const e1 = new THREE.Vector3(0, 1, 0).cross(w).normalize(), e2 = w.clone().cross(e1);
          const ax = e1.clone().multiplyScalar(Math.cos(f.axis)).addScaledVector(e2, Math.sin(f.axis));
          const ay = e1.clone().multiplyScalar(-Math.sin(f.axis)).addScaledVector(e2, Math.cos(f.axis));
          const nLoops = f.big ? 3 : 1;
          for (let j = 0; j < nLoops && li < NL; j++, li++) {
            const along = (j - (nLoops - 1) / 2) * 0.45 * f.size;
            const base = w.clone().addScaledVector(ax, along);
            const A = base.clone().addScaledVector(ay, -sepGrow * f.size).normalize(), B = base.clone().addScaledVector(ay, sepGrow * f.size).normalize();
            U.uLoopA.value[li].set(A.x, A.y, A.z, f.size * (0.35 + 0.5 * sepGrow));
            U.uLoopB.value[li].set(B.x, B.y, B.z, i * 7 + j);
            U.uLoopC.value[li].set(7000, 1.6 * Lg * f.peak, f.size * 0.07, 0);
          }
        }
        for (; li < NL; li++) U.uLoopC.value[li].set(0, 0, 0, 0);
        boostShown = boost;
      },
      hud(ctx) {
        return [
          ['Độ sáng dải g′ (so với lúc yên tĩnh)', `× ${boostShown.toFixed(boostShown < 10 ? 2 : 0)}`],
          ['Số vụ bùng nổ đang diễn ra', `${flares.length}`],
          ...starHud(ctx, PROX.R, PROX.T, PROX.M),
        ];
      },
      onClick(info, ctx) {
        const q = hitUnit(info.origin, info.dir); if (!q) return;
        const b = bodyFromWorld(q, ctx.time.sim);
        spawn(b, { peak: 0.9, size: 0.08, T: 9000, tauR: 0.8, tauD: 8 });
        ctx.toast('Bùng nổ nhỏ: các đường sức từ xoắn lại rồi <b>tái kết nối</b>, đốt nóng plasma lên ~10 000 K. Với sao lùn đỏ, chỉ một vùng nhỏ đã đủ làm cả ngôi sao sáng vọt.');
      },
    };
    return inst;
  },
};
// actions get ctx only; keep a handle to the instance on the def (set on enter)
redDwarf.create = (orig => function (ctx) { const inst = orig.call(this, ctx); redDwarf.__inst = inst; return inst; })(redDwarf.create);

/* ============================================================================================== */
/* 3. Red supergiant — Betelgeuse                                                                   */
/* ============================================================================================== */
const BETEL = { R: 764, T: 3600, M: 18 };
const betelgeuse = {
  id: 'red-supergiant',
  name: 'Siêu khổng lồ đỏ',
  category: 'stars',
  order: 30,
  subtitle: 'Betelgeuse (α Orionis) · M1–2 Ia–ab',
  swatch: ['#ffb36b', '#6a1608'],
  blurb: 'Nếu đặt vào chỗ Mặt Trời, Betelgeuse sẽ nuốt cả quỹ đạo Sao Hỏa. Lớp vỏ loãng và hấp dẫn bề mặt yếu đến mức <b>thang độ cao áp suất</b> lên tới cỡ một phần mười bán kính: chỉ vài <b>tế bào đối lưu khổng lồ</b> phủ kín bề mặt, làm rìa sao lồi lõm và độ sáng thay đổi thất thường.',
  intro: 'Nhấn <b>G</b> để tái hiện <b>Đại suy giảm</b> 2019–2020 — rồi thử bay ra phía sau ngôi sao.',
  facts: [
    ['Phân loại', 'M1–M2 Ia–ab — siêu khổng lồ đỏ'],
    ['Khoảng cách', '~550 năm ánh sáng (168 pc)'],
    ['Bán kính', '~760 R☉ (≈ 3,5 AU)'],
    ['Khối lượng', '16–19 M☉'],
    ['Nhiệt độ hiệu dụng', '~3 600 K'],
    ['Độ sáng', '~10⁵ L☉'],
    ['Đường kính góc', '~42 mili-giây cung (đo được bằng giao thoa kế)'],
    ['Đại suy giảm', 'cấp sao V 0,5 → 1,6 (2/2020)'],
    ['Số phận', 'siêu tân tinh loại II, trong vòng ~100 000 năm'],
  ],
  unitKm: BETEL.R * R_SUN_KM,
  camera: { distance: 3.6, polar: 85, azimuth: 20, min: 1.15, max: 40, fov: 45 },
  exposure: 1.0,
  bloom: 0.3,
  params: [
    { key: 'cells', label: 'Tương phản tế bào đối lưu', type: 'range', min: 0, max: 2, step: 0.01, default: 1 },
    { key: 'envelope', label: 'Vỏ khí phân tử & bụi', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Sao siêu khổng lồ đỏ mất ~10⁻⁶ M☉ mỗi năm; khí nguội đi, ngưng tụ thành bụi silicat.' },
    { key: 'dimming', label: 'Đại suy giảm 2019–20', type: 'toggle', default: false },
    { key: 'orbits', label: 'Quỹ đạo các hành tinh', type: 'toggle', default: false, note: 'Kích thước Hệ Mặt Trời so với ngôi sao (nét đứt: bị che hoặc nằm bên trong sao).' },
  ],
  actions: [
    { label: 'Đại suy giảm', key: 'g', title: 'Đám bụi che nửa nam (2019–2020)', run: ctx => { ctx.params.dimming = !ctx.params.dimming; ctx.toast(ctx.params.dimming ? 'Cuối 2019 một <b>bong bóng khí</b> phóng ra rồi nguội, ngưng tụ thành <b>đám bụi</b> che nửa nam ngôi sao; cùng lúc bề mặt có một vùng nguội. Betelgeuse mờ đi ~1,2 cấp sao — không phải dấu hiệu sắp nổ.' : 'Đám bụi tan dần — Betelgeuse sáng trở lại (giữa năm 2020).', 7000); } },
  ],
  create(ctx) {
    const { U, material } = makeStar(ctx, 2, {
      T: BETEL.T, Iexp: 4, bright: 0.52, stretch: 1.4, sky: 0.4,
      // strong, very wavelength-dependent limb darkening of a cool extended atmosphere (TiO bands in blue)
      ldA: [0.62, 0.78, 0.88], ldB: [0.20, 0.13, 0.08],
    });
    // the dust cloud sits between the star and Earth (= the default viewing direction), to the south
    const c = betelgeuse.camera;
    const earth = new THREE.Vector3().setFromSphericalCoords(1, c.polar * D2R, c.azimuth * D2R);
    // 1.3 R* in front of the star, offset so that from the default view it projects onto the southern half
    const dustC = earth.clone().multiplyScalar(1.3).add(new THREE.Vector3(0, -0.3, 0)).addScaledVector(new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), earth).normalize(), -0.08);
    U.uDust.value.set(dustC.x, dustC.y, dustC.z, 0.45);
    const overlay = makeOrbits([['Trái Đất', 1, [0.35, 0.6, 1.0]], ['Sao Hỏa', 1.52, [1.0, 0.45, 0.3]], ['Sao Mộc', 5.2, [1.0, 0.8, 0.5]], ['Sao Thổ', 9.58, [0.95, 0.9, 0.6]]], BETEL.R);
    let orbOp = 0;
    return {
      material, overlay,
      update(dt, ctx) {
        const p = ctx.params, t = ctx.time.sim;
        U.uGranT.value = t * 0.05;
        U.uCell.value = p.cells; U.uEnv.value = p.envelope;
        const target = p.dimming ? 1 : 0;
        U.uDim.value = ease(U.uDim.value, target, dt * 0.35, this);
        orbOp += ((p.orbits ? 1 : 0) - orbOp) * Math.min(1, dt * 4);
        overlay.visible = orbOp > 0.01;
        overlay.userData.update(ctx.camera.position, orbOp);
      },
      hud(ctx) {
        const V = 0.5 + 1.1 * U.uDim.value;
        return [
          ['Cấp sao biểu kiến V (từ Trái Đất)', V.toFixed(2) + (U.uDim.value > 0.5 ? ' — mờ nhất từng ghi nhận' : '')],
          ...starHud(ctx, BETEL.R, BETEL.T, BETEL.M),
        ];
      },
      onClick(info, ctx) {
        const q = hitUnit(info.origin, info.dir); if (!q) return;
        ctx.toast(`Một tế bào đối lưu ở đây rộng ~${Math.round(0.6 * BETEL.R * R_SUN_KM / 1e6)} triệu km — lớn hơn khoảng cách Mặt Trời–Sao Thủy. Hấp dẫn bề mặt chỉ ~1 cm/s², nên plasma trồi lên rất cao trước khi rơi lại.`);
      },
    };
  },
};

/* ============================================================================================== */
/* 4. White dwarf — Sirius B                                                                        */
/* ============================================================================================== */
const SIRB = { R: 0.0084, M: 1.018, T: 25000 };
const whiteDwarf = {
  id: 'white-dwarf',
  name: 'Sao lùn trắng',
  category: 'stars',
  order: 50,
  subtitle: 'Sirius B · DA2 · lõi sao đã chết',
  swatch: ['#eef4ff', '#5a7cff'],
  blurb: 'Khối lượng bằng Mặt Trời nén vào một quả cầu nhỏ hơn Trái Đất: một thìa vật chất nặng vài tấn. Không còn phản ứng hạt nhân, nó chống lại hấp dẫn nhờ <b>áp suất suy biến electron</b> (nguyên lý loại trừ Pauli) và chỉ từ từ nguội đi trong hàng tỷ năm. Bầu khí quyển hydro mỏng vài chục mét nên rìa sao sắc như dao cạo.',
  intro: 'Trái Đất được đặt cạnh để so sánh kích thước. Nhấn <b>C</b> để xem sao lùn trắng <b>nguội dần</b> theo thời gian.',
  facts: [
    ['Phân loại', 'DA2 — sao lùn trắng khí quyển hydro, lõi C–O'],
    ['Khoảng cách', '8,6 năm ánh sáng (bạn đồng hành của Sirius A)'],
    ['Bán kính', '0,0084 R☉ ≈ 5 840 km (0,92 Trái Đất)'],
    ['Khối lượng', '1,02 M☉'],
    ['Nhiệt độ hiệu dụng', '~25 000 K'],
    ['Mật độ trung bình', '~2,4 tấn/cm³'],
    ['Hấp dẫn bề mặt', 'log g ≈ 8,6 (~400 000 lần Trái Đất)'],
    ['Dịch đỏ hấp dẫn', '≈ 80 km/s (Hubble, 2018) — khớp thuyết tương đối rộng'],
    ['Quỹ đạo quanh Sirius A', '50,1 năm, cách 8–31,5 AU'],
  ],
  unitKm: SIRB.R * R_SUN_KM,
  camera: { distance: 4.6, polar: 82, azimuth: 35, min: 1.1, max: 40, fov: 45, target: [0.9, 0, 0] },
  exposure: 1.0,
  bloom: 0.35,
  params: [
    { key: 'temp', label: 'Nhiệt độ bề mặt', type: 'range', min: 4000, max: 100000, step: 100, default: 25000, fmt: v => Math.round(v).toLocaleString('vi-VN') + ' K', note: 'Mới sinh: > 100 000 K, xanh trắng. Nguội dần qua trắng, vàng, rồi đỏ cam. Phơi sáng giữ cố định.' },
    { key: 'earth', label: 'Trái Đất để so sánh', type: 'toggle', default: true },
    { key: 'siriusA', label: 'Sirius A trên bầu trời', type: 'toggle', default: true, note: 'Sao chủ A1V, sáng gấp ~25 lần Mặt Trời, cách ~20 AU.' },
  ],
  actions: [
    { label: 'Nguội dần', key: 'c', title: 'Cho sao nguội từ 100 000 K xuống 4 000 K', run: ctx => { ctx.scene.__cool = { t0: ctx.time.wall }; ctx.params.temp = 100000; ctx.toast('Sao lùn trắng không có nguồn năng lượng: nó chỉ <b>tỏa nhiệt dư</b>. Càng nguội càng tỏa chậm (L ∝ T⁴) — nên phải mất hàng chục tỷ năm, lâu hơn tuổi vũ trụ, mới thành “sao lùn đen”.', 7000); } },
  ],
  create(ctx) {
    const { U, material } = makeStar(ctx, 3, {
      T: SIRB.T, Tref: 25000, Iexp: 0.5, bright: 0.68, stretch: 1.5, sky: 0.7,
      // hot DA white dwarf: modest limb darkening (radiative H atmosphere)
      ldA: [0.20, 0.24, 0.29], ldB: [0.20, 0.20, 0.19],
    });
    U.uEarthPos.value.set(2.6, 0.0, -0.9);
    U.uSirDir.value.set(-0.62, 0.22, -0.75).normalize();
    const def = whiteDwarf;
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params;
        if (def.__cool) {
          // logarithmic cooling-track playback: 100 000 K → 4 000 K in ~16 s
          const k = Math.min(1, (ctx.time.wall - def.__cool.t0) / 16);
          p.temp = Math.round(100000 * Math.pow(4000 / 100000, k) / 100) * 100;
          if (k >= 1) def.__cool = null;
        }
        U.uTeff.value = p.temp;
        U.uEarthOn.value = p.earth ? 1 : 0;
        U.uSirA.value = p.siriusA ? 1 : 0;
      },
      hud(ctx) {
        const T = ctx.params.temp, { L } = starNumbers(SIRB.R, T, SIRB.M);
        // Mestel (1952) cooling law, C/O core: t ≈ 8.8 Myr · (M/M☉)^(5/7) · (L/L☉)^(−5/7)  (rough)
        const tc = 8.8e6 * Math.pow(SIRB.M, 5 / 7) * Math.pow(L, -5 / 7);
        const vz = 0.6366 * SIRB.M / SIRB.R;   // GM/(Rc) in km/s
        return [
          ['Tuổi nguội (định luật Mestel)', tc > 1e9 ? (tc / 1e9).toFixed(1) + ' tỷ năm' : (tc / 1e6).toFixed(tc < 1e7 ? 1 : 0) + ' triệu năm'],
          ['Dịch đỏ hấp dẫn GM/Rc', `${vz.toFixed(0)} km/s`],
          ...starHud(ctx, SIRB.R, T, SIRB.M),
        ];
      },
      onClick(info, ctx) {
        const q = hitUnit(info.origin, info.dir); if (!q) return;
        ctx.toast('Một mẩu vật chất cỡ hạt đường ở đây nặng ~<b>2–3 kg</b>; trên bề mặt nó nặng như ~<b>1 000 tấn</b> trên Trái Đất. Ánh sáng thoát ra mất ~0,03% năng lượng vì giếng hấp dẫn.');
      },
    };
  },
};

/* ============================================================================================== */
/* 5. Blue supergiant — Rigel                                                                       */
/* ============================================================================================== */
const RIGEL = { R: 78.9, T: 12100, M: 21 };
const blueSupergiant = {
  id: 'blue-supergiant',
  name: 'Siêu khổng lồ xanh',
  category: 'stars',
  order: 40,
  subtitle: 'Rigel (β Orionis) · B8 Ia · biến quang α Cygni',
  swatch: ['#e8f0ff', '#3b6cff'],
  blurb: 'Nóng gấp đôi Mặt Trời và to gấp ~80 lần, Rigel tỏa sáng như <b>120 000 Mặt Trời</b>. Áp suất bức xạ lên các vạch hấp thụ đẩy khí khỏi bề mặt thành <b>gió sao</b> vài trăm km/s. Bề mặt không có hạt đối lưu, nhưng cả ngôi sao “rung” theo nhiều mode dao động phi xuyên tâm.',
  intro: 'Tăng <b>Biên độ dao động</b> để thấy các mode rung trên bề mặt; bật <b>Quỹ đạo</b> để so với Hệ Mặt Trời.',
  facts: [
    ['Phân loại', 'B8 Ia — siêu khổng lồ xanh-trắng'],
    ['Khoảng cách', '~860 năm ánh sáng'],
    ['Bán kính', '~79 R☉ (0,37 AU)'],
    ['Khối lượng', '~21 M☉'],
    ['Nhiệt độ hiệu dụng', '~12 100 K'],
    ['Độ sáng', '~120 000 L☉'],
    ['Biến quang', 'kiểu α Cygni, cấp sao 0,05–0,18, chu kỳ 1–75 ngày'],
    ['Gió sao', 'cỡ 10⁻⁷ M☉/năm, vài trăm km/s'],
    ['Tuổi', '~8 triệu năm'],
  ],
  unitKm: RIGEL.R * R_SUN_KM,
  camera: { distance: 3.6, polar: 84, azimuth: 25, min: 1.1, max: 40, fov: 45 },
  exposure: 1.0,
  bloom: 0.4,
  params: [
    { key: 'wind', label: 'Gió sao', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Tán xạ electron ánh sáng sao (∝ ρ) và phát xạ H-alpha (∝ ρ²) trong gió.' },
    { key: 'pulse', label: 'Biên độ dao động', type: 'range', min: 0, max: 6, step: 0.1, default: 1, fmt: v => '×' + v.toFixed(1), note: '×1 ≈ thực tế (nhiệt độ dao động ~1–3%, độ sáng ~0,1 cấp sao); tăng lên để phóng đại.' },
    { key: 'orbits', label: 'Quỹ đạo các hành tinh', type: 'toggle', default: false, note: 'Sao Thủy sẽ lướt sát bề mặt Rigel.' },
  ],
  actions: [
    { label: 'Phóng đại dao động', key: 'p', run: ctx => { ctx.params.pulse = ctx.params.pulse > 2 ? 1 : 5; ctx.toast(ctx.params.pulse > 2 ? 'Dao động phóng đại ×5: các vùng nóng/lạnh xen kẽ là <b>mode phi xuyên tâm</b> (l = 2, 3) chạy quanh sao.' : 'Biên độ thực tế: dao động chỉ làm Rigel thay đổi ~0,1 cấp sao.'); } },
  ],
  create(ctx) {
    const { U, material } = makeStar(ctx, 4, {
      T: RIGEL.T, Iexp: 3, bright: 0.36, stretch: 2.2, sky: 0.4,
      ldA: [0.24, 0.28, 0.33], ldB: [0.16, 0.16, 0.15],
    });
    const overlay = makeOrbits([['Sao Thủy', 0.387, [0.8, 0.75, 0.7]], ['Sao Kim', 0.723, [1.0, 0.85, 0.55]], ['Trái Đất', 1, [0.35, 0.6, 1.0]]], RIGEL.R);
    let orbOp = 0;
    return {
      material, overlay,
      update(dt, ctx) {
        const p = ctx.params, t = ctx.time.sim;
        U.uGranT.value = t * 0.1;
        U.uWind.value = p.wind; U.uPuls.value = p.pulse;
        U.uRad.value = 1 + 0.0015 * p.pulse * Math.sin(t * 0.1 * 0.9);
        orbOp += ((p.orbits ? 1 : 0) - orbOp) * Math.min(1, dt * 4);
        overlay.visible = orbOp > 0.01;
        overlay.userData.update(ctx.camera.position, orbOp);
      },
      hud(ctx) {
        const dm = 0.07 * Math.min(ctx.params.pulse, 6) * Math.sin(ctx.time.sim * 0.09);
        return [['Cấp sao V (dao động)', (0.12 + dm * 0.6).toFixed(2)], ...starHud(ctx, RIGEL.R, RIGEL.T, RIGEL.M)];
      },
    };
  },
};

/* ============================================================================================== */
/* 6. Wolf–Rayet star WR 124 and its nebula M1-67                                                   */
/* ============================================================================================== */
// One scene unit = 1 light-year here: the star is a point, the nebula is ~8 ly across.
const LY_KM = 9.4607e12;
const wolfRayet = {
  id: 'wolf-rayet',
  name: 'Sao Wolf–Rayet',
  category: 'stars',
  order: 60,
  subtitle: 'WR 124 · WN8h · tinh vân M1-67',
  swatch: ['#ffd0e0', '#8a1838'],
  blurb: 'Một ngôi sao nặng ~30 M☉ ở giai đoạn cuối đời đã thổi bay lớp vỏ hydro: gió sao ~10⁻⁵ M☉ mỗi năm, nhanh ~700 km/s, phơi ra lõi nóng ~45 000 K. Lớp khí phun ra trước đó vỡ thành hàng nghìn <b>cục khí–bụi</b> có đuôi hướng ra ngoài, bị bức xạ tử ngoại của sao ion hóa và phát sáng đỏ hồng H-alpha.',
  intro: 'Chuyển <b>Bước sóng</b> sang hồng ngoại (phím <b>W</b>) để xem tinh vân như kính James Webb (2023) — kèm các gai nhiễu xạ sáu cánh đặc trưng.',
  facts: [
    ['Phân loại', 'WN8h — sao Wolf–Rayet giàu nitơ'],
    ['Khoảng cách', '~15 000 năm ánh sáng (chòm Mũi Tên)'],
    ['Khối lượng', '~30 M☉'],
    ['Nhiệt độ', '~45 000 K'],
    ['Gió sao', '~700 km/s, ~10⁻⁵ M☉/năm'],
    ['Tinh vân M1-67', 'đường kính ~8 năm ánh sáng, nở ra ~46 km/s'],
    ['Khối lượng đã phun', '~10 M☉ (JWST, 2023)'],
    ['Vận tốc riêng', '~180 km/s — một ngôi sao “chạy trốn”'],
    ['Số phận', 'siêu tân tinh loại Ib/c, hoặc sụp thẳng thành hố đen'],
  ],
  unitKm: LY_KM,
  camera: { distance: 13, polar: 80, azimuth: 30, min: 1.5, max: 40, fov: 45 },
  exposure: 1.0,
  bloom: 0.6,
  params: [
    { key: 'band', label: 'Bước sóng', type: 'seg', options: [['ha', 'H-alpha (Hubble)'], ['ir', 'Hồng ngoại (JWST)']], default: 'ha', note: 'H-alpha: khí hydro ion hóa. Hồng ngoại: bụi ấm tự phát nhiệt (màu giả).' },
    { key: 'nebula', label: 'Độ sáng tinh vân', type: 'range', min: 0, max: 2, step: 0.01, default: 1 },
    { key: 'knots', label: 'Độ vón cục', type: 'range', min: 0, max: 1, step: 0.01, default: 0.8, note: 'Bất ổn định Rayleigh–Taylor: gió nhanh đẩy vào lớp vỏ chậm và xé nó thành các cục khí.' },
  ],
  actions: [
    { label: 'Đổi bước sóng', key: 'w', run: ctx => { ctx.params.band = ctx.params.band === 'ir' ? 'ha' : 'ir'; ctx.toast(ctx.params.band === 'ir' ? 'Hồng ngoại: <b>bụi</b> ấm phát sáng; các gai sáu cánh là nhiễu xạ của gương lục giác JWST, không phải của ngôi sao.' : 'H-alpha: hydro bị ion hóa rồi tái hợp, phát ánh sáng đỏ 656 nm.'); } },
  ],
  create(ctx) {
    const { U, material } = makeStar(ctx, 5, { T: 45000, bright: 1, sky: 0.6, ldA: [0, 0, 0], ldB: [0, 0, 0] });
    U.uRad.value = 1e-6;
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params;
        U.uNeb.value = p.nebula; U.uKnots.value = p.knots; U.uIR.value = p.band === 'ir' ? 1 : 0;
      },
      hud(ctx) {
        const age = 3.4 * LY_KM / 46 / 3.156e7;   // shell radius / expansion speed, years
        return [
          ['Tuổi động học (R / v)', `~${(Math.round(age / 1000) * 1000).toLocaleString('vi-VN')} năm`],
          ['Ánh sáng sao tới vỏ tinh vân', '~3,4 năm'],
          ['Công suất', '~2×10⁵ L☉ (chủ yếu tử ngoại)'],
        ];
      },
    };
  },
};

function hitUnit(o, d) {
  const b = o.dot(d), c = o.lengthSq() - 1, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? o.clone().addScaledVector(d, t).normalize() : null;
}

export default [sun, redDwarf, betelgeuse, blueSupergiant, whiteDwarf, wolfRayet];
