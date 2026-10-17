// Schwarzschild black hole GLSL (units: r_s = 1, so M = 1/2; c = 1).
//
// BH_GLSL provides the accretion disk and traceBH(), which integrates one null geodesic in
// Schwarzschild coordinates. Other scenes (e.g. the wormhole) can include BH_GLSL after
// COMMON_GLSL and call traceBH() themselves; BH_MAIN is the black-hole scene's own main().
//
// Requires defines MAX_STEPS (int) and STEP_K (float).

import { COMMON_GLSL } from './common.js';

export const BH_GLSL = /* glsl */`
uniform vec3 uObsBeta;
uniform float uSplit, uLensing;
uniform float uDiskOn, uDiskIn, uDiskOut, uDiskTmax, uDiskBright, uDiskDensity, uDiskTurb, uGlow;
uniform vec3 uGlowCol;
uniform float uDoppler;
uniform float uShowPS, uShowISCO;
uniform vec4 uProbe[8];
uniform vec4 uProbeCol[8];
uniform int uProbeN;

/* --- accretion disk --- */
float diskNoise(float r, float a, float seed){
  vec2 q = vec2(cos(a), sin(a));
  float lr = log(r);
  float n = 0.5  * snoise(vec3(q * 1.6, lr * 5.0 + seed))
          + 0.25 * snoise(vec3(q * 3.8, lr * 13.0 + seed * 1.31))
          + 0.14 * snoise(vec3(q * 9.0, lr * 31.0 + seed * 1.73))
          + 0.07 * snoise(vec3(q * 20.0, lr * 67.0 - seed));
  float rings = 0.5 + 0.5 * sin(lr * 55.0 + n * 4.0);
  return clamp(0.52 + 0.75 * n, 0.0, 1.0) * (0.72 + 0.28 * rings);
}
vec4 diskSample(vec3 p, float r, vec3 nPh, bool gr, float gCam){
  float inner = smoothstep(uDiskIn * 0.985, uDiskIn * 1.08, r);
  float outer = 1.0 - smoothstep(uDiskOut * 0.55, uDiskOut, r);
  float phi = atan(p.z, p.x);
  float om = sqrt(0.5 / (r * r * r));          // Keplerian Ω = sqrt(M/r³), exact in Schwarzschild coordinate time
  const float PER = 90.0;
  float t1 = mod(uTime, PER), t2 = mod(uTime + 0.5 * PER, PER);
  float w = 1.0 - abs(2.0 * t1 / PER - 1.0);
  float n = mix(diskNoise(r, phi - om * t2, 17.3), diskNoise(r, phi - om * t1, 3.1), w);
  float dens = inner * outer * mix(1.0, n, uDiskTurb);
  float alpha = clamp(dens * uDiskDensity, 0.0, 1.0);
  // Novikov–Thorne-like temperature profile, peak at r = 49/36 r_in
  float x = r / uDiskIn;
  float f = max(1.0 - sqrt(1.0 / x), 0.0) / (x * x * x);
  float T = uDiskTmax * pow(max(f, 1e-7) / 0.0566528, 0.25);
  float g = 1.0;
  if (uDoppler > 0.5){
    float beta = gr ? sqrt(0.5 / max(r - 1.0, 0.02)) : sqrt(0.5 / r);
    beta = min(beta, 0.995);
    float gam = inversesqrt(1.0 - beta * beta);
    vec3 tdir = vec3(-p.z, 0.0, p.x) / r;
    vec3 nl = nPh;
    if (gr){ vec3 rh = p / r; float nr = dot(nl, rh); nl = normalize(nl + rh * nr * (inversesqrt(1.0 - 1.0 / r) - 1.0)); }
    float dop = 1.0 / (gam * (1.0 - beta * dot(tdir, nl)));
    g = dop * (gr ? sqrt(1.0 - 1.0 / r) : 1.0) * gCam;
    if (uDoppler < 1.5) g = mix(1.0, g, 0.28);
  }
  float To = T * g;
  float I = pow(To / uDiskTmax, 4.0);
  // soft-knee only above 4× (keeps the physical g⁴ contrast but stops the beamed side from turning into a white sheet)
  I = I < 4.0 ? I : 4.0 + 4.0 * log(I / 4.0);
  vec3 e = bb(To) * I * uDiskBright * (0.45 + 0.95 * n) * 1.15;
  if (uShowISCO > 0.5){
    float l = 1.0 - smoothstep(0.0, 0.045, abs(r - 3.0));
    e += vec3(0.3, 0.9, 1.0) * l * 5.0;
    alpha = max(alpha, l * 0.7);
  }
  return vec4(e, alpha);
}

// Integrates a photon backwards from x (Schwarzschild coordinates, r_s = 1) along coordinate
// direction v, accumulating disk emission into col / trans. Returns
//   0 = escaped beyond max(r0 + 2, 60) (x, v hold the exit state),
//   1 = fell through the horizon,  2 = opaque / undetermined.
int traceBH(inout vec3 x, inout vec3 v, bool gr, float gCam, inout vec3 col, inout float trans){
  float r = length(x);
  vec3 hv = cross(x, v); float h2 = dot(hv, hv);
  float rFar = max(r + 2.0, 60.0);
  for (int i = 0; i < MAX_STEPS; i++){
    if (r > rFar && dot(x, v) > 0.0) return 0;
    float dl = clamp(STEP_K * r, 0.012, 6.0);
    float dt = dl / length(v);
    vec3 xp = x; float rp = r;
    if (gr){
      // null geodesic: x'' = -(3/2) h² x / r⁵  (exact Schwarzschild photon orbits, r_s = 1)
      vec3 a = -1.5 * h2 * x / (r * r * r * r * r);
      v += a * (0.5 * dt); x += v * dt; r = length(x);
      a = -1.5 * h2 * x / (r * r * r * r * r);
      v += a * (0.5 * dt);
    } else { x += v * dt; r = length(x); }

    if (uDiskOn > 0.5 && xp.y * x.y < 0.0){
      vec3 p = mix(xp, x, xp.y / (xp.y - x.y));
      float rr = length(p.xz);
      if (rr > uDiskIn * 0.97 && rr < uDiskOut){
        vec4 e = diskSample(p, rr, -normalize(v), gr, gCam);
        col += trans * e.rgb * e.a; trans *= 1.0 - e.a;
      }
    }
    if (uGlow > 0.0 && uDiskOn > 0.5){
      float rr = length(x.xz);
      float H = 0.06 * rr + 0.05;
      float gl = exp(-abs(x.y) / H) * smoothstep(uDiskIn * 0.9, uDiskIn * 1.4, rr) * (1.0 - smoothstep(uDiskOut * 0.4, uDiskOut, rr)) * pow(uDiskIn / max(rr, 0.5), 2.5);
      col += trans * uGlowCol * gl * dl * uGlow * 0.1;
    }
    if (uShowPS > 0.5 && (rp - 1.5) * (r - 1.5) < 0.0){
      vec3 p = mix(xp, x, (rp - 1.5) / (rp - r));
      vec3 q = p / 1.5; float lon = atan(q.z, q.x), lat = asin(clamp(q.y, -1.0, 1.0));
      float s = PI / 12.0;
      float l = max(gridLine(lon / s, 0.07), gridLine(lat / s, 0.07));
      col += trans * vec3(0.25, 0.75, 1.0) * (0.06 + l * 1.3);
    }
    for (int k = 0; k < 8; k++){
      if (k >= uProbeN) break;
      vec3 pp = uProbe[k].xyz; float s = uProbe[k].w;
      vec3 seg = x - xp; float L = length(seg); vec3 sd = seg / L;
      vec3 w0 = pp - xp; float tca = dot(w0, sd);
      if (tca < -5.0 * s || tca > L + 5.0 * s) continue;
      float d2 = dot(w0, w0) - tca * tca;
      if (d2 > 25.0 * s * s) continue;
      float e = exp(-d2 / (s * s)) * 0.5 * (erfA((L - tca) / s) + erfA(tca / s));
      col += trans * uProbeCol[k].rgb * e;
    }
    if (r < 1.0005) return 1;
    if (trans < 0.004) return 2;
  }
  return r > 3.0 ? 0 : 2;
}

// Converts a direction measured by a static observer at x into a Schwarzschild coordinate
// direction (radial component scaled by the lapse √(1 − 1/r)).
vec3 staticToCoord(vec3 x, vec3 d){
  float r = length(x), lapse = sqrt(max(1.0 - 1.0 / r, 1e-4));
  vec3 rh = x / r; return d + rh * dot(d, rh) * (lapse - 1.0);
}
`;

export const BH_MAIN = /* glsl */`
void main(){
  vec3 dObs = camRay();
  bool gr = uLensing > 0.5 && gl_FragCoord.x / uRes.x >= uSplit;

  // relativistic aberration: observer frame -> local static frame
  vec3 d = dObs; float gObs = 1.0;
  float b2 = dot(uObsBeta, uObsBeta);
  if (b2 > 1e-10){
    float b = sqrt(b2); vec3 bh = uObsBeta / b; float gam = inversesqrt(1.0 - b2);
    vec3 np = -dObs; float npar = dot(np, bh); vec3 nperp = np - npar * bh;
    float den = 1.0 + b * npar;
    vec3 n = bh * ((npar + b) / den) + nperp / (gam * den);
    d = -normalize(n);
    gObs = 1.0 / (gam * den);
  }
  vec3 x = uCamPos; float rc = length(x);
  float lapse = gr ? sqrt(max(1.0 - 1.0 / rc, 1e-4)) : 1.0;
  float gCam = gObs / lapse;
  vec3 v = gr ? staticToCoord(x, d) : d;

  vec3 col = vec3(0.0); float trans = 1.0;
  int res = traceBH(x, v, gr, gCam, col, trans);
  if (res == 0 && trans > 0.004) col += trans * sky(normalize(v), gCam);
  gl_FragColor = vec4(col, 1.0);
}`;

export const BH_FRAG = COMMON_GLSL + BH_GLSL + BH_MAIN;
