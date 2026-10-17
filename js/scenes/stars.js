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

const NS = 16, NL = 8, NF = 4;           // spots, loops (prominences / flare loops), flares
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
  float prof = 1.0 - smoothstep(0.0, 0.95, d1);                  // soft dome, brightest in the middle
  float body = (0.55 + 0.45 * h1.x) * (0.5 + 0.5 * life) * (0.1 + 0.9 * prof * prof * (3.0 - 2.0 * prof));
  if (g1.y > 0.8 && w1 > 1.1) body *= 1.0 - 0.65 * exp(-d1 * d1 / 0.048);   // exploding granule
  float lw = 0.22 + 0.22 * (0.5 + 0.5 * snoise(p * 0.7 + seed));          // lanes of varying width
  float lane = 1.0 - smoothstep(0.0, lw, d2 - d1);
  lane *= lane * (3.0 - 2.0 * lane);
  float bp = lane * smoothstep(0.55, 0.9, snoise(p * 2.7 + seed * 1.7)); // bright points in the lanes
  return mix(body, -1.3, lane) + 1.6 * bp + 0.12;
}
#endif
vec3 warp3(vec3 p){ return p + 0.45 * vec3(snoise(p * 0.31), snoise(p * 0.31 + 17.1), snoise(p * 0.31 + 31.7)); }

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
    if (wA > 0.001) g += wA * sunGranule(warp3(qa * uGranF), uGranT, uEp.z * 13.0);
    if (wB > 0.001) g += wB * sunGranule(warp3(qb * uGranF), uGranT, uEp.w * 13.0);
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
    vec3 colk = C.x < 100.0 ? HALPHA : bb(C.x);
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
#if KIND <= 1
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
  {
    vec3 ref = uBright * bb(uTeff) * pow(uTeff / uTref, uIexp);   // the star's own mean surface
    vec3 x = surf / ref;
    surf = ref * mix(pow(max(x, 0.0), vec3(uStretch)), x, step(1.0, x));
  }
  float tEnd = cov > 0.5 ? tHit : 1e9;
  vec3 col = mix(sky(rd, 1.0) * uSkyK, surf, cov);
  float h = max(b - R, 0.0);

#if KIND <= 1
  // off-limb structures are sampled in the plane of the sky through the star's centre, so radial
  // features stay exactly radial on screen whatever the perspective
  vec3 vh = normalize(ro);
  vec3 sp = pc - vh * dot(pc, vh); sp = length(sp) > 1e-5 ? normalize(sp) : s;
  if (tEnd > 1e8){
    vec3 halo = chromosphere(sp, h) * (1.0 + 4.0 * uEclipse);
  #if KIND == 0
    halo += corona(sp, b) * (1.0 + 7.0 * uEclipse);
  #endif
    col += halo;
  }
  if (uProm > 0.0) col += loopsGlow(ro, rd, tEnd) * uProm * (1.0 + uEclipse);
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
  };
  const material = new THREE.ShaderMaterial({
    uniforms: U, defines: { KIND: kind, NS, NL, NF },
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

const sun = {
  id: 'sun',
  name: 'Mặt Trời',
  category: 'stars',
  order: 10,
  subtitle: 'Sao lùn vàng G2V · ngôi sao của chúng ta',
  swatch: ['#fff6e0', '#ff9a2a'],
  blurb: 'Bề mặt sôi sục: mỗi <b>hạt sáng</b> (~1 000 km) là một cột plasma nóng trồi lên rồi nguội đi, chìm xuống ở rãnh tối. Rìa đĩa tối và đỏ hơn tâm (<b>hiệu ứng tối rìa</b>) vì ở đó ta chỉ nhìn được vào những lớp nông, nguội hơn. Xích đạo quay nhanh hơn hai cực — Mặt Trời không phải vật rắn.',
  intro: 'Cuộn lại thật gần để thấy <b>hạt đối lưu</b> và cấu trúc vết đen. Nhấn <b>E</b> để xem <b>nhật thực toàn phần</b>: vành nhật hoa và tai lửa hiện ra.',
  facts: [
    ['Phân loại', 'G2V — sao lùn vàng dãy chính'],
    ['Bán kính', '695 700 km (109 × Trái Đất)'],
    ['Khối lượng', '1,989×10³⁰ kg (333 000 × Trái Đất)'],
    ['Nhiệt độ hiệu dụng', '5 772 K'],
    ['Công suất bức xạ', '3,828×10²⁶ W'],
    ['Tự quay', '25,4 ngày (xích đạo) · ~35 ngày (cực)'],
    ['Nhiệt độ lõi / vành nhật hoa', '15,7 triệu K / 1–2 triệu K'],
    ['Hạt đối lưu', '~1 000 km, sống 8–10 phút'],
    ['Tuổi', '4,6 tỷ năm'],
  ],
  unitKm: R_SUN_KM,
  camera: { distance: 3.5, polar: 82, azimuth: 20, min: 1.03, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.25,
  params: [
    { key: 'activity', label: 'Mức hoạt động (vết đen)', type: 'range', min: 0, max: 1, step: 0.01, default: 0.6, note: 'Chu kỳ 11 năm: 0 = cực tiểu (đĩa trơn, vành nhật hoa dẹt theo xích đạo), 1 = cực đại (nhiều nhóm vết đen, các tia vành nhật hoa tỏa mọi hướng).' },
    { key: 'granulation', label: 'Độ tương phản hạt', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Hạt nóng hơn rãnh tối ~200–300 K; thấy rõ khi lại gần (khoảng cách < 1.2).' },
    { key: 'corona', label: 'Vành nhật hoa', type: 'range', min: 0, max: 2, step: 0.01, default: 0.6 },
    { key: 'prominences', label: 'Sắc cầu & tai lửa', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Phát xạ H-alpha (656 nm) màu hồng của sắc cầu, gai (spicule) và tai lửa ở rìa đĩa.' },
    { key: 'eclipse', label: 'Nhật thực toàn phần', type: 'toggle', default: false, note: 'Mặt Trăng che vừa khít quang cầu.' },
  ],
  actions: [
    { label: 'Nhật thực', key: 'e', title: 'Bật/tắt nhật thực toàn phần', run: ctx => { ctx.params.eclipse = !ctx.params.eclipse; ctx.toast(ctx.params.eclipse ? 'Nhật thực toàn phần: quang cầu bị che, lộ ra <b>vành nhật hoa</b> (1–2 triệu K) và các <b>tai lửa</b> hồng H-alpha.' : 'Quang cầu sáng trở lại — vành nhật hoa mờ hơn ~một triệu lần nên chìm trong ánh chói.'); } },
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
    // prominences: quiescent hedgerows (mid/high latitudes, long-lived) and active-region loops
    const proms = [];
    for (let i = 0; i < 22; i++) {
      const sheet = i < 13;
      proms.push({
        sheet, life: sheet ? 500 + 400 * hashJS(i, 1, 1) : 200 + 200 * hashJS(i, 1, 1), off: hashJS(i, 1, 2),
        lat: (hashJS(i, 1, 3) < 0.5 ? -1 : 1) * (sheet ? 15 + 50 * hashJS(i, 1, 4) : 6 + 28 * hashJS(i, 1, 4)) * D2R,
        lon: hashJS(i, 1, 5) * Math.PI * 2,
        span: sheet ? 0.07 + 0.2 * hashJS(i, 1, 6) : 0.025 + 0.04 * hashJS(i, 1, 6),
        height: sheet ? 0.035 + 0.07 * hashJS(i, 1, 7) : 0.02 + 0.035 * hashJS(i, 1, 7),
        az: hashJS(i, 1, 8) * Math.PI, seed: hashJS(i, 1, 9) * 60,
      });
    }
    const camDir = new THREE.Vector3(), A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3();
    let regs = [];
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params, t = ctx.time.sim;
        U.uGranT.value = t * 0.12;
        U.uRotT.value = t;
        epochs(U, t, 160);
        U.uActivity.value = p.activity; U.uGranAmp.value = p.granulation;
        // like a solar telescope's exposure: high-resolution close-ups are exposed a little darker so the
        // granules don't clip, the full disc a little brighter so its centre reads white
        U.uBright.value = 0.42 + 0.3 * smooth(1.1, 2.0, ctx.camera.position.length());
        U.uCorona.value = p.corona; U.uChromo.value = 0.18 * p.prominences; U.uProm.value = p.prominences;
        U.uEclipse.value = ease(U.uEclipse.value, p.eclipse ? 1 : 0, dt * 2.5, this);
        regs = activeRegions(t, t, p.activity, omegaFn, regCfg, spots);
        // choose the prominences nearest to the visible limb (the only place they show in white light)
        camDir.copy(ctx.camera.position).normalize();
        const cand = [];
        for (const pr of proms) {
          const tt = t + pr.off * pr.life, k = Math.floor(tt / pr.life), f = tt / pr.life - k;
          const env = smooth(0, 0.15, f) * (1 - smooth(0.8, 1, f));
          if (env <= 0.01 || hashJS(pr.seed | 0, k, 3) > 0.35 + 0.65 * (pr.sheet ? 0.8 : p.activity)) continue;
          const lon = pr.lon + 2.1 * k - omegaFn(Math.sin(pr.lat) ** 2) * t;
          dirLL(pr.lat, lon, C);
          const score = Math.abs(C.dot(camDir));
          if (score < 0.42) cand.push({ pr, lon, env, score, k });
        }
        cand.sort((a, b) => a.score - b.score);
        for (let i = 0; i < NL; i++) {
          const c = cand[i];
          if (!c) { U.uLoopC.value[i].set(0, 0, 0, 0); continue; }
          const { pr } = c;
          // footpoints ±span/2 along an azimuth in the local tangent plane
          dirLL(pr.lat, c.lon, C);
          const e1 = new THREE.Vector3(0, 1, 0).cross(C).normalize(), e2 = C.clone().cross(e1);
          const az = pr.az + c.k;
          const tv = e1.multiplyScalar(Math.cos(az)).addScaledVector(e2, Math.sin(az)).multiplyScalar(pr.span * 0.5);
          A.copy(C).sub(tv).normalize(); B.copy(C).add(tv).normalize();
          U.uLoopA.value[i].set(A.x, A.y, A.z, pr.height * (0.6 + 0.4 * c.env));
          U.uLoopB.value[i].set(B.x, B.y, B.z, pr.seed + c.k * 3.1);
          U.uLoopC.value[i].set(0, (pr.sheet ? 0.55 : 0.9) * c.env, pr.sheet ? 0.006 : 0.0035, pr.sheet ? 1 : 0);
        }
      },
      hud(ctx) {
        let g = 0, s = 0;
        for (const r of regs) if (r.envS > 0.05) { g++; s += r.size > 0.016 ? 2 + Math.round(r.size * 300) : 1; }
        const lat = Math.asin(Math.max(-1, Math.min(1, ctx.camera.position.y / ctx.camera.position.length())));
        const P = 360 / (SUN_OMEGA_DEG[0] + SUN_OMEGA_DEG[1] * Math.sin(lat) ** 2 + SUN_OMEGA_DEG[2] * Math.sin(lat) ** 4);
        return [
          ['Số vết đen Wolf (R = 10g + s)', `${10 * g + s} · ${g} nhóm`],
          [`Chu kỳ quay ở vĩ độ ${Math.abs(lat / D2R).toFixed(0)}°`, `${P.toFixed(1)} ngày`],
          ...starHud(ctx, 1, 5772, 1),
        ];
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
    };
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
