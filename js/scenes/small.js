// Thiên thể nhỏ — comets, asteroids and an interstellar visitor (see CONTRACT.md).
//
//  comet               1 unit = 10 000 km. The nucleus (a few km) is ray-marched as an SDF at its true
//                      size at the origin; the coma is integrated along each ray with an angular
//                      parametrisation that importance-samples the 1/ρ² density; the two tails are a
//                      bounded volumetric march (64 steps, geometric spacing).
//  asteroid            1 unit = 250 m. Rubble pile (Bennu / Itokawa) as an SDF with boulder domes,
//                      Hapke-style shading, free precession, and a DART impact with an ejecta cone.
//  interstellar-object 1 unit = 100 m. 'Oumuamua as a tumbling cigar or pancake with a live light curve.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';
import { t, t as tt } from '../i18n.js';  // tt: for scopes where a local `t` shadows t()

const HELPERS = /* glsl */`
float smin(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
float smax(float a, float b, float k){ return -smin(-a, -b, k); }
float sdEll(vec3 p, vec3 r){ float k0 = length(p / r), k1 = length(p / (r * r)); return k0 * (k0 - 1.0) / max(k1, 1e-6); }
// 1-D value noise in [0,1]
float n1(float x){ float i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(hash13(vec3(i, 7.13, 3.31)), hash13(vec3(i + 1.0, 7.13, 3.31)), f); }
// Henyey–Greenstein phase function (normalised to 4π)
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * c, 1.5); }
// boulders as faceted domes; 2×2×2 cell search (feature points kept in the cell centre)
float boulders(vec3 p, float cell, float dens, float seed){
  vec3 g = p / cell; vec3 b = floor(g - 0.5); float h = 0.0;
  for (int z = 0; z <= 1; z++) for (int y = 0; y <= 1; y++) for (int x = 0; x <= 1; x++){
    vec3 c = b + vec3(x, y, z);
    vec3 r = hash33(c + seed);
    if (r.x > dens) continue;
    vec3 q = c + 0.3 + 0.4 * fract(r.yzx * 31.17 + r.zxy * 17.31) - g;   // second hash derived from the first
    q *= vec3(1.0, 0.8 + 0.4 * r.z, 1.2 - 0.4 * r.z);
    vec3 aq = abs(q);
    float dist = mix(length(q), (aq.x + aq.y + aq.z) * 0.62, 0.55);
    float R = 0.2 + 0.3 * r.y * r.y;      // ≤ 0.5 cell: stays inside the 2×2×2 search support
    h = max(h, (R * R - dist * dist) / (2.0 * R) * 1.35);
  }
  return h * cell;
}
vec3 sunDisk(vec3 rd, vec3 s, float angR){
  float c = dot(rd, s);
  return bb(5772.0) * (smoothstep(cos(angR * 1.15), cos(angR * 0.85), c) * 300.0 + pow(max(c, 0.0), 3000.0) * 3.0 + pow(max(c, 0.0), 60.0) * 0.04);
}
`;

/* =============================================================================================
   COMET
   ============================================================================================= */
const COMET_FRAG = COMMON_GLSL + HELPERS + /* glsl */`
uniform vec3 uSun, uA, uLag, uNrm, uIonAx, uIonE1, uIonE2;
uniform mat3 uNucRot;          // body -> world
uniform float uNucS;           // scene units per nucleus-SDF unit (1e-4 × preset scale: SDF unit ≈ km for 67P)
uniform float uG;              // radiation-pressure acceleration for β = 1, units/day²
uniform float uOmega;          // angular rate of the anti-solar direction, rad/day
uniform float uTauMax;         // age of the oldest visible dust, days
uniform float uVej;            // dust ejection speed, units/day
uniform float uXs;             // sunward turn-around distance of the dust fountain, units
uniform float uC;              // dust coma strength: n·ρ² in radiance units (from Afρ)
uniform float uGas, uLp, uLd;  // C2 coma strength and Haser parent / daughter scale lengths
uniform float uIon, uLion, uVion;
uniform float uDays, uExpo, uSunF, uRc, uSunAng, uStretch, uStretchRef;
uniform vec4 uBound;
uniform vec4 uJetP[6];         // jet source (world, scene units) + activity
uniform vec3 uJetD[6];         // jet axis (world)

#define NUC_R 3.3
#define ION_K 0.022
uniform float uZmax, uDustK, uComaGain;
const vec3 DUST_COL = vec3(1.0, 0.84, 0.66);   // sunlight reddened by micron-sized grains
const vec3 GAS_COL  = vec3(0.22, 1.0, 0.5);    // C2 Swan bands (516 nm) + CN: the green coma
const vec3 ION_COL  = vec3(0.28, 0.52, 1.0);   // CO+ comet-tail bands (≈ 427 nm)

// ---- nucleus: two lobes like 67P (big lobe 4.1×3.3×1.8 km, small lobe 2.6×2.3×1.8 km) ----------
float nucShape(vec3 p){
  float big = sdEll(p - vec3(-0.8, 0.0, 0.0), vec3(2.05, 1.6, 1.0));
  vec3 q = p - vec3(1.6, 0.95, 0.05);
  q.xy = mat2(0.866, 0.5, -0.5, 0.866) * q.xy;
  float sml = sdEll(q, vec3(1.25, 1.05, 0.92));
  return smin(big, sml, 0.42);
}
float nucSDF(vec3 p, float detail){
  float d = nucShape(p);
  if (d > 0.7) return d;
  d += 0.17 * snoise(p * 0.55 + 1.3) + 0.07 * snoise(p * 1.4 - 2.1);
  d -= 0.06 * (1.0 - abs(snoise(p * 2.2 + 9.0)));            // ridged: sharp crests and pits
  // strata: 67P is layered like an onion; the terraces end in cliffs
  float s = dot(p, vec3(0.25, 0.94, 0.22)) * 2.1 + 0.5 * snoise(p * 0.6 + 4.0);
  float stair = floor(s) + smoothstep(0.35, 0.65, fract(s));       // continuous: flat terraces + steep risers
  d += 0.09 * (s - stair);
  if (detail > 0.5){
    d += 0.03 * snoise(p * 4.1) + 0.013 * snoise(p * 9.3);
    d -= boulders(p, 0.28, 0.45, 3.0);                          // house-sized boulders (Rosetta saw thousands)
  }
  return d;
}
vec3 nucNormal(vec3 p, float e){
  vec2 k = vec2(1.0, -1.0);
  return normalize(k.xyy * nucSDF(p + k.xyy * e, 1.0) + k.yyx * nucSDF(p + k.yyx * e, 1.0) +
                   k.yxy * nucSDF(p + k.yxy * e, 1.0) + k.xxx * nucSDF(p + k.xxx * e, 1.0));
}
float nucShadow(vec3 p, vec3 l){
  float res = 1.0, t = 0.04;
  for (int i = 0; i < 28; i++){
    float h = nucSDF(p + l * t, 0.0);
    res = min(res, 8.0 * h / t);
    t += clamp(h, 0.04, 0.6);
    if (res < 0.01 || t > 8.0) break;
  }
  return clamp(res, 0.0, 1.0);
}

// ---- dust phase function: strong forward scattering (why comets near the Sun-line blaze) ----------
float dustPhase(vec3 rd){
  float c = dot(rd, uSun);                       // cos of scattering angle (light travels along -uSun)
  return (0.75 * hg(c, 0.72) + 0.25 * hg(c, -0.25)) / 1.05;
}

// ---- coma: angular sampling t = tc + b·tanθ makes every sample carry equal 1/ρ² column ----------
vec3 comaMarch(vec3 ro, vec3 rd, float tMax, float jit, float pix){
  vec2 s = sphereHit(ro, rd, vec3(0.0), uRc);
  if (s.x > s.y || s.y < 0.0) return vec3(0.0);
  float t0 = max(s.x, 0.0), t1 = min(s.y, tMax);
  if (t1 <= t0) return vec3(0.0);
  float tc = -dot(ro, rd);
  vec3 pc = ro + rd * tc;
  float b = length(pc);
  float bE = max(b, max(0.35 * pix * max(tc, 0.0), 0.6 * uNucS));   // pixel footprint / nucleus size
  float th0 = atan((t0 - tc) / bE), th1 = atan((t1 - tc) / bE);
  float ph = dustPhase(rd);
  vec3 acc = vec3(0.0);
  const int N = 40;
  float dth = (th1 - th0) / float(N);
  for (int i = 0; i < N; i++){
    float th = th0 + dth * (float(i) + jit);
    float tt = bE * tan(th);
    vec3 p = pc + rd * tt;
    float rho2 = b * b + tt * tt, rho = sqrt(rho2);
    float w = dth * bE * (1.0 + tan(th) * tan(th)) / max(rho2, 1e-12);   // dl / ρ²
    vec3 u = p / max(rho, 1e-9);
    float X = dot(p, uA);
    float perp2 = max(rho2 - X * X, 0.0);
    // fountain model: grains launched sunward turn round at Xs → paraboloid envelope
    float env = smoothstep(-0.7, 0.5, (X + uXs - perp2 / (4.0 * uXs)) / uXs);
    float cs = dot(u, uSun);
    float day = 0.3 + 1.7 * max(cs, 0.0) + 3.0 * max(-cs, 0.0) * max(-cs, 0.0);   // sunward release, anti-sunward funnel
    float jets = 0.0;
    for (int k = 0; k < 6; k++){
      vec3 v = p - uJetP[k].xyz;
      float vl = length(v);
      float cj = dot(v, uJetD[k]) / max(vl, 1e-12);
      jets += uJetP[k].w * (exp((cj - 1.0) * 260.0) + 0.25 * exp((cj - 1.0) * 30.0)) * smoothstep(0.0, 0.5 * uNucS, vl);
    }
    jets *= 9.0 * exp(-rho / (40.0 * uNucS));                      // rotation smears jets into the coma
    float fil = 1.0 + 0.25 * snoise(u * 9.0 + vec3(0.0, 0.0, log(max(rho, 1e-9)) * 1.5));
    float outer = 1.0 - smoothstep(0.5 * uRc, uRc, rho);
    float dust = uC * (day * fil + jets) * env * outer;
    float gas = uGas * (exp(-rho / uLd) - exp(-rho / uLp)) * outer;
    acc += (DUST_COL * dust * ph + GAS_COL * gas) * w;
  }
  return acc;
}

// ---- tails --------------------------------------------------------------------------------------
// Dust: syndynes in the linearised comet frame. A grain of radiation-pressure ratio β released τ days
// ago sits at X = ½βgτ² (anti-sunward) and Y = βgωτ³/3 (lagging behind the orbital motion, because
// the anti-solar direction has rotated by ωτ since). Each syndyne is a tube widened by the ejection
// speed; grains released at the same moment form synchrones → striae when the activity flickers.
const float BETA[6] = float[6](0.06, 0.12, 0.24, 0.45, 0.8, 1.4);
const float BW[6]   = float[6](0.12, 0.16, 0.19, 0.2, 0.18, 0.15);
// sample positions: geometric spacing away from the start when the camera sits inside the volume
float segT(float t0, float L, float u, out float dt){
  float c = max(t0, 0.4);
  float k = log(L / c + 1.0);
  float e = exp(k * u);
  dt = c * e * k;
  return t0 + c * (e - 1.0);
}
// ray ∩ ion-tail cone (apex behind the head, half-angle atan(uConeK)) ∩ 0 < s < sMax
vec2 coneSeg(vec3 ro, vec3 rd, vec3 A, float apex, float k, float sMax){
  vec3 o = ro - A * apex;
  float c2 = 1.0 / (1.0 + k * k);
  float oa = dot(o, A), da = dot(rd, A);
  float a2 = da * da - c2;
  if (abs(a2) < 1e-7) a2 = 1e-7;
  float b2 = 2.0 * (oa * da - c2 * dot(o, rd)), cc = oa * oa - c2 * dot(o, o);
  float disc = b2 * b2 - 4.0 * a2 * cc;
  vec2 seg = vec2(0.0, 1e9);
  if (disc < 0.0){
    if (a2 < 0.0) return vec2(1.0, 0.0);
  } else {
    float sq = sqrt(disc);
    float r1 = (-b2 - sq) / (2.0 * a2), r2 = (-b2 + sq) / (2.0 * a2);
    float t1 = min(r1, r2), t2 = max(r1, r2);
    if (a2 < 0.0) seg = vec2(t1, t2);
    else seg = da > 0.0 ? vec2(t2, 1e9) : vec2(-1e9, t1);
  }
  // keep the forward nappe and 0 < s < sMax (s measured from the head, apex at s = apex)
  float s0 = apex;
  if (abs(da) > 1e-6){
    float ta = (0.0 - (oa + s0) + 0.0) / da;       // s = oa + s0 + da t  ... s=0 at the head plane
    float tb = (sMax - (oa + s0)) / da;
    float lo = min(ta, tb), hi = max(ta, tb);
    seg = vec2(max(seg.x, lo), min(seg.y, hi));
  } else if (oa + s0 < 0.0 || oa + s0 > sMax) return vec2(1.0, 0.0);
  seg.x = max(seg.x, 0.0);
  return seg;
}
vec3 dustMarch(vec3 ro, vec3 rd, float tMax, float jit, float pix){
  vec2 s = sphereHit(ro, rd, uBound.xyz, uBound.w);
  if (s.x > s.y || s.y < 0.0) return vec3(0.0);
  // the dust lies in a slab around the orbital plane
  float zo = dot(ro, uNrm), zd = dot(rd, uNrm);
  float t0 = max(s.x, 0.0), t1 = min(s.y, tMax);
  if (abs(zd) > 1e-5){
    float ta = (-uZmax - zo) / zd, tb = (uZmax - zo) / zd;
    t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb));
  } else if (abs(zo) > uZmax) return vec3(0.0);
  if (t1 <= t0) return vec3(0.0);
  const int N = 48;
  float ph = dustPhase(rd);
  float Sdot = 12.566 * uVej * uC;            // cross-section production that feeds the 1/ρ² coma
  vec3 acc = vec3(0.0);
  for (int i = 0; i < N; i++){
    float dt;
    float t = segT(t0, t1 - t0, (float(i) + jit) / float(N), dt);
    dt /= float(N);
    vec3 p = ro + rd * t;
    float X0 = dot(p, uA);
    float X = max(X0, 0.02);
    float rho = length(p);
    float head = smoothstep(0.2 * uRc, 0.95 * uRc, rho);
    if (head <= 0.0) continue;
    float Y = dot(p, uLag), Z = dot(p, uNrm);
    float blur0 = 0.08 * dt * dt + pix * pix * t * t;
    float dust = 0.0;
    for (int k = 0; k < 6; k++){
      float bg = BETA[k] * uG;
      float tau = sqrt(2.0 * X / bg);
      if (tau > uTauMax * 1.35) continue;
      float Yk = bg * uOmega * tau * tau * tau / 3.0;
      float sg = uVej * sqrt(BETA[k]) * tau * 0.6 + 0.3;     // big grains (small β) leave slowly: v ∝ √β
      float blur2 = min(blur0, 4.0 * sg * sg);     // anti-alias, but never smear a tube beyond 2× its width
      float sy2 = sg * sg + 0.05 * Yk * Yk + blur2;
      float sz2 = sg * sg + blur2;
      float dy = Y - Yk;
      float xs = min(X0, 0.0);
      float g = exp(-0.5 * (dy * dy / sy2 + (Z * Z + xs * xs) / sz2)) / (6.2832 * sqrt(sy2 * sz2));
      float lam = BW[k] / (bg * tau + uVej);
      float age = smoothstep(uTauMax * 1.35, uTauMax * 0.6, tau);
      float stri = 0.7 + 0.6 * smoothstep(0.2, 0.9, n1((uDays - tau) * 1.3));
      dust += lam * g * age * stri;
    }
    acc += DUST_COL * (dust * Sdot * ph * head * uDustK) * dt;
  }
  return acc;
}
// ion tail: CO+ ions picked up by the solar wind, straight along the (aberrated) anti-solar axis
vec3 ionMarch(vec3 ro, vec3 rd, float tMax, float jit, float pix){
  if (uIon <= 0.0) return vec3(0.0);
  float apex = -0.4 / ION_K;
  vec2 sg = coneSeg(ro, rd, uIonAx, apex, ION_K * 3.2, uLion * 2.4);
  float t0 = sg.x, t1 = min(sg.y, tMax);
  if (t1 <= t0) return vec3(0.0);
  const int N = 40;
  vec3 acc = vec3(0.0);
  float P = 2.6, ph2 = fract(uDays / P), sk = ph2 * uVion * P * 0.55;
  for (int i = 0; i < N; i++){
    float dt;
    float t = segT(t0, t1 - t0, (float(i) + jit) / float(N), dt);
    dt /= float(N);
    vec3 p = ro + rd * t;
    float sa = dot(p, uIonAx);
    vec3 l = p - uIonAx * sa;
    float lx = dot(l, uIonE1), ly = dot(l, uIonE2);
    float sw = sa - uVion * uDays;                 // co-moving coordinate: structures flow outward
    // disconnection event every ~2.6 days: the tail is cut near the head and the old tail drifts off
    float beyond = smoothstep(sk - 25.0, sk + 25.0, sa);
    lx -= beyond * (0.035 * (sa - sk) + 3.0 * sin((sa - sk) * 0.012));
    lx -= 0.012 * sa * sin(sw * 0.004 + 1.3);
    float gq = (sa - sk) / (0.05 * sk + 12.0);
    float gap = 1.0 - 0.85 * exp(-gq * gq);
    float w = 0.4 + ION_K * max(sa, 0.0);
    float blur2 = min(0.08 * dt * dt + pix * pix * t * t, 4.0 * w * w);
    float w2 = w * w + blur2;
    float r2 = (lx * lx + ly * ly) / w2;
    // streamers: thin rays that fold toward the axis near the head and flow outward with the plasma
    float fil = 1.0 - abs(snoise(vec3(lx / w * 1.5, ly / w * 1.5, sw * 0.0018)));
    fil = pow(clamp(fil, 0.0, 1.0), 10.0);
    float prof = exp(-0.5 * r2) * (0.25 + 1.2 * fil) + 1.6 * fil * exp(-0.12 * r2);
    float along = smoothstep(-0.5, 3.0, sa) * exp(-max(sa, 0.0) / uLion) / (1.0 + max(sa, 0.0) / 150.0);
    float rho = length(p);
    float ion = prof / (6.2832 * w2) * along * gap * mix(1.0, 0.55, beyond) * smoothstep(0.08 * uRc, 0.4 * uRc, rho);
    acc += ION_COL * ion * uIon * dt;
  }
  return acc;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  float pix = 2.0 * uTanHalfFov / uRes.y;
  float jit = hash13(vec3(gl_FragCoord.xy, 3.7));
  vec3 col = sky(rd, 1.0) * 0.6 + sunDisk(rd, uSun, uSunAng);
  float tHit = 1e9;
  float dn = length(ro);
  float nucR = NUC_R * uNucS;
  // unresolved nucleus: a point of reflected sunlight spread over the pixel footprint (kept at least
  // faintly visible, like the star-like "false nucleus" in a stretched photograph)
  vec3 point = vec3(0.0);
  if (dn * pix >= nucR * 3.0){
    float ang = acos(clamp(dot(rd, -ro / dn), -1.0, 1.0)) / pix;
    float flux = 0.05 * 0.5 * 14.0 * uSunF * pow(0.45 * nucR / (dn * pix), 2.0) * uExpo;
    point = vec3(1.0, 0.93, 0.86) * max(flux, 0.03) * exp(-0.5 * ang * ang * 0.6);
  }
  // nucleus (skipped once it is far below one pixel)
  if (dn * pix < nucR * 3.0){
    vec2 bh = sphereHit(ro, rd, vec3(0.0), nucR);
    if (bh.x < bh.y && bh.y > 0.0){
      mat3 iR = transpose(uNucRot);
      vec3 o = iR * ro / uNucS, d = iR * rd;
      float t = max(bh.x, 0.0) / uNucS, tEnd = bh.y / uNucS;
      float hitT = -1.0;
      for (int i = 0; i < 96; i++){
        vec3 p = o + d * t;
        float h = nucSDF(p, 1.0);
        if (h < max(0.0015, t * pix * 0.6)){ hitT = t; break; }
        t += h * 0.8;
        if (t > tEnd) break;
      }
      if (hitT > 0.0){
        vec3 p = o + d * hitT;
        vec3 nb = nucNormal(p, max(0.004, hitT * pix));
        vec3 n = uNucRot * nb;
        vec3 L = uSun;
        float mu0 = max(dot(n, L), 0.0), mu = max(dot(n, -rd), 1e-3);
        float sh = mu0 > 0.0 ? nucShadow(p + nb * 0.02, iR * L) : 0.0;
        float alpha = acos(clamp(dot(L, -rd), -1.0, 1.0));
        // albedo ≈ 0.05: darker than charcoal. Smooth dust plains slightly brighter, rare exposed-ice spots
        float tex = 0.75 + 0.5 * (0.5 + 0.5 * snoise(p * 2.3)) + 0.25 * snoise(p * 11.0);
        float plain = smoothstep(0.1, 0.5, snoise(p * 0.8 + 7.0));
        float ice = smoothstep(0.78, 0.86, snoise(p * 7.0 + 2.0)) * smoothstep(0.3, 0.6, snoise(p * 1.3 - 5.0));
        float alb = 0.05 * tex * (1.0 + 0.25 * plain);
        vec3 tint = mix(vec3(1.0, 0.93, 0.86), vec3(0.8, 0.9, 1.0), ice);
        alb = mix(alb, 0.35, ice);
        float ls = mu0 / (mu0 + mu);
        float opp = 1.0 + 0.9 / (1.0 + tan(alpha * 0.5) / 0.06);       // shadow-hiding opposition surge
        float phase = exp(-0.9 * alpha);                                 // rough, dark surface: steep phase curve
        col = tint * alb * ls * sh * opp * phase * 14.0 * uSunF;
        col += vec3(0.6, 0.65, 0.7) * alb * 0.004 * max(dot(n, -L) * 0.5 + 0.5, 0.0);  // light from the lit coma
        tHit = hitT * uNucS;
        col *= uExpo;
      }
    }
  }
  vec3 em = (comaMarch(ro, rd, tHit, jit, pix) * uComaGain + dustMarch(ro, rd, tHit, fract(jit + 0.37), pix)
           + ionMarch(ro, rd, tHit, fract(jit + 0.71), pix)) * uExpo + point;
  // The diffuse light of a comet spans ~5 decades from the inner coma to the tail tips. Like the asinh /
  // power stretch used for every great-comet photograph, compress it (hue and ordering preserved).
  float lum = dot(em, vec3(0.2126, 0.7152, 0.0722)) + 1e-12;
  em = mix(em, em * (0.4 * asinh(lum / uStretchRef) / lum), uStretch);
  col += em;
  gl_FragColor = vec4(col, 1.0);
}`;

// ---- comet CPU side ------------------------------------------------------------------------------
// Marsden–Sekanina–Yeomans water-sublimation law g(r), normalised to 1 at 1 AU.
function waterG(r) {
  const r0 = 2.808, m = 2.15, n = 5.093, k = 4.6142, a = 0.111262;
  return a * Math.pow(r / r0, -m) * Math.pow(1 + Math.pow(r / r0, n), -k);
}
const COMETS = {
  '67p':      { name: '67P/Churyumov–Gerasimenko', scale: 1.0, rkm: 2.0, Q1: 5e28, afrho: 25 },
  'halley':   { name: '1P/Halley', scale: 3.6, rkm: 5.5, Q1: 8e29, afrho: 300 },
  'halebopp': { name: 'C/1995 O1 Hale–Bopp', scale: 13, rkm: 30, Q1: 1.2e31, afrho: 15000 },
};
function nucShapeJS(x, y, z) {
  const ell = (px, py, pz, rx, ry, rz) => {
    const k0 = Math.hypot(px / rx, py / ry, pz / rz), k1 = Math.hypot(px / (rx * rx), py / (ry * ry), pz / (rz * rz));
    return k0 * (k0 - 1) / k1;
  };
  const big = ell(x + 0.8, y, z, 2.05, 1.6, 1.0);
  let qx = x - 1.6, qy = y - 0.95; const qz = z - 0.05;
  const rx = 0.866 * qx - 0.5 * qy, ry = 0.5 * qx + 0.866 * qy;   // matches mat2(0.866,0.5,-0.5,0.866)
  const sml = ell(rx, ry, qz, 1.25, 1.05, 0.92);
  const k = 0.42, h = Math.min(1, Math.max(0, 0.5 + 0.5 * (sml - big) / k));
  return sml + (big - sml) * h - k * h * (1 - h);
}
function surfaceAlong(f, dir, rMax = 4) {
  let lo = 0, hi = rMax;
  for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (f(dir.x * m, dir.y * m, dir.z * m) < 0) lo = m; else hi = m; }
  return dir.clone().multiplyScalar(lo);
}
function gradJS(f, p) {
  const e = 1e-3;
  return new THREE.Vector3(
    f(p.x + e, p.y, p.z) - f(p.x - e, p.y, p.z), f(p.x, p.y + e, p.z) - f(p.x, p.y - e, p.z), f(p.x, p.y, p.z + e) - f(p.x, p.y, p.z - e)).normalize();
}
// active areas (body frame); the neck region (Hapi) was 67P's most active source
const JET_DIRS = [[0.35, 0.35, 0.87], [-0.4, 0.85, 0.3], [0.85, 0.55, -0.3], [-0.9, 0.1, -0.45], [-0.2, -0.6, 0.8], [0.55, -0.8, 0.2]];
const JET_STR = [1.6, 1.0, 0.8, 0.9, 0.7, 0.6];
const JETS = JET_DIRS.map((d, i) => {
  const dir = new THREE.Vector3(...d).normalize();
  const pos = surfaceAlong(nucShapeJS, dir);
  return { pos, dir: gradJS(nucShapeJS, pos).lerp(dir, 0.3).normalize(), str: JET_STR[i] };
});

let cometParams = null;
const COMET_VIEW = {
  all: { d: 2300, tgt: [-800, 0, -300], polar: 55, az: -18 },
  head: { d: 160, tgt: [-40, 0, -10], polar: 22, az: 20 },
  nucleus: { d: null, tgt: [0, 0, 0], polar: 75, az: 35 },
};
function cometViewDist(v, preset) { return v.d ?? 0.0011 * COMETS[preset].scale * 1.0; }

const comet = {
  id: 'comet',
  name: 'Sao chổi',
  category: 'small',
  order: 10,
  subtitle: 'Hạt nhân băng bẩn · đầu sao chổi · đuôi ion và đuôi bụi',
  swatch: ['#d8f3ff', '#1d3b6a'],
  blurb: 'Một khối băng bẩn vài km, đen hơn than (suất phản chiếu ~4 %). Khi tới gần Mặt Trời, băng thăng hoa thành đầu sao chổi rộng hàng trăm nghìn km và hai cái đuôi: <b>đuôi ion</b> xanh lam bị gió Mặt Trời thổi thẳng ra xa Mặt Trời, còn <b>đuôi bụi</b> vàng nhạt bị áp suất bức xạ đẩy nhẹ nên cong, trễ lại dọc theo quỹ đạo.',
  intro: 'Hai cái đuôi <b>không</b> chỉ về phía sau chuyển động: đuôi ion xanh luôn chỉ đúng hướng ngược Mặt Trời (☀ ở bên phải), đuôi bụi cong về phía quỹ đạo đã qua. Kéo thanh <b>khoảng cách tới Mặt Trời</b> ra 3–4 AU để thấy sao chổi “tắt”, rồi chọn khung nhìn <b>Hạt nhân</b>.',
  facts: [
    ['Hạt nhân 67P', '4,1 × 3,3 × 1,8 km (thùy lớn), hai thùy'],
    ['Suất phản chiếu 67P', '0,06 (hình học)'],
    ['Chu kỳ tự quay 67P', '12,4 giờ'],
    ['Hạt nhân Halley', '15 × 8 × 8 km, chu kỳ quỹ đạo 76 năm'],
    ['Đuôi bụi Hale–Bopp', '≈ 4×10⁷ km'],
    ['Rosetta', 'quay quanh 67P 2014–2016; Philae đáp ngày 12/11/2014'],
  ],
  en: {
    name: 'Comet',
    subtitle: 'Dirty-ice nucleus · coma · ion and dust tails',
    blurb: 'A dirty ice ball a few km across, darker than charcoal (albedo ~4%). Near the Sun its ice sublimates into a coma hundreds of thousands of km wide and two tails: a blue <b>ion tail</b> blown straight away from the Sun by the solar wind, and a pale yellow <b>dust tail</b> gently pushed by radiation pressure, so it curves and lags behind along the orbit.',
    intro: 'The two tails do <b>not</b> simply trail behind the motion: the blue ion tail always points straight away from the Sun (☀ is on the right), while the dust tail curves back along the path already travelled. Drag <b>Distance from the Sun</b> out to 3–4 AU to watch the comet “switch off”, then pick the <b>Nucleus</b> view.',
    facts: [
      ['67P nucleus', '4.1 × 3.3 × 1.8 km (large lobe), two lobes'],
      ['67P albedo', '0.06 (geometric)'],
      ['67P rotation period', '12.4 h'],
      ['Halley nucleus', '15 × 8 × 8 km, orbital period 76 years'],
      ['Hale–Bopp dust tail', '≈ 4×10⁷ km'],
      ['Rosetta', 'orbited 67P 2014–2016; Philae landed on 12 November 2014'],
    ],
  },
  unitKm: 1e4,
  camera: {
    get distance() { const p = cometParams; return cometViewDist(COMET_VIEW[p?.view || 'all'], p?.preset || '67p'); },
    get target() { return COMET_VIEW[cometParams?.view || 'all'].tgt; },
    get polar() { return COMET_VIEW[cometParams?.view || 'all'].polar; },
    get azimuth() { return COMET_VIEW[cometParams?.view || 'all'].az; },
    min: 0.0003, max: 9000, fov: 45,
  },
  exposure: 1.0,
  bloom: 0.7,
  params: [
    { key: 'r', label: 'Khoảng cách tới Mặt Trời', type: 'range', min: 0.3, max: 5, step: 0.01, default: 0.9, fmt: v => v.toFixed(2) + ' AU', note: 'Băng nước chỉ thăng hoa mạnh trong khoảng ~3 AU; ngoài đó sao chổi gần như ngủ yên.',
      en: { label: 'Distance from the Sun', note: 'Water ice only sublimates strongly within ~3 AU; beyond that the comet is almost dormant.' } },
    { key: 'psi', label: 'Hướng bay so với Mặt Trời', type: 'range', min: 0, max: 180, step: 1, default: 105, fmt: v => v.toFixed(0) + '°', note: '0° = lao thẳng vào Mặt Trời, 90° = tại điểm cận nhật, 180° = bay ra xa. Thành phần ngang càng lớn, đuôi bụi càng cong.',
      en: { label: 'Direction of travel relative to the Sun', note: '0° = plunging straight toward the Sun, 90° = at perihelion, 180° = heading away. The larger the sideways component, the more the dust tail curves.' } },
    { key: 'preset', label: 'Sao chổi', type: 'seg', options: [['67p', '67P'], ['halley', 'Halley'], ['halebopp', 'Hale–Bopp']], default: 'halley', note: 'Hạt nhân lớn hơn ⇒ diện tích hoạt động lớn hơn ⇒ đầu và đuôi sáng hơn.',
      en: { label: 'Comet', options: ['67P', 'Halley', 'Hale–Bopp'], note: 'A bigger nucleus ⇒ more active area ⇒ a brighter coma and tails.' } },
    { key: 'view', label: 'Khung nhìn', type: 'seg', options: [['all', 'Toàn cảnh'], ['head', 'Đầu'], ['nucleus', 'Hạt nhân']], default: 'all', en: { label: 'View', options: ['Overview', 'Coma', 'Nucleus'] } },
    { key: 'ion', label: 'Đuôi ion', type: 'toggle', default: true, en: { label: 'Ion tail' } },
  ],
  actions: [
    { label: 'Hạt nhân', key: 'n', title: 'Bay tới hạt nhân', run: ctx => { ctx.params.view = 'nucleus'; ctx.scene.params.find(p => p.key === 'view').onChange?.('nucleus', ctx); },
      en: { label: 'Nucleus', title: 'Fly to the nucleus' } },
    { label: 'Toàn cảnh', key: 'v', title: 'Lùi ra xem toàn bộ sao chổi', run: ctx => { ctx.params.view = 'all'; ctx.scene.params.find(p => p.key === 'view').onChange?.('all', ctx); },
      en: { label: 'Overview', title: 'Pull back to see the whole comet' } },
  ],
  create(ctx) {
    cometParams = ctx.params;
    const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
    const uniforms = {
      ...ctx.shared,
      uSun: { value: V(1, 0, 0) }, uA: { value: V(-1, 0, 0) }, uLag: { value: V(0, 0, -1) }, uNrm: { value: V(0, 1, 0) },
      uIonAx: { value: V(-1, 0, 0) }, uIonE1: { value: V(0, 0, -1) }, uIonE2: { value: V(0, 1, 0) },
      uNucRot: { value: new THREE.Matrix3() }, uNucS: { value: 1e-4 },
      uG: { value: 4.43 }, uOmega: { value: 0.02 }, uTauMax: { value: 30 }, uVej: { value: 3.9 }, uXs: { value: 3 },
      uC: { value: 1e-5 }, uGas: { value: 1e-5 }, uLp: { value: 2.2 }, uLd: { value: 6.6 },
      uIon: { value: 1 }, uLion: { value: 2000 }, uVion: { value: 1700 },
      uDays: { value: 0 }, uExpo: { value: 1 }, uStretch: { value: 1 }, uStretchRef: { value: 0.03 }, uSunF: { value: 1 }, uRc: { value: 30 }, uSunAng: { value: 0.00465 },
      uBound: { value: new THREE.Vector4(0, 0, 0, 100) }, uZmax: { value: 100 }, uDustK: { value: 4 }, uComaGain: { value: 1 },
      uJetP: { value: JETS.map(() => new THREE.Vector4()) }, uJetD: { value: JETS.map(() => V()) },
    };
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: COMET_FRAG, depthTest: false, depthWrite: false });
    const U = uniforms;
    const S = V(1, 0, 0);
    const m4 = new THREE.Matrix4(), mObl = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0.55, 0.3, 0.2)), mSpin = new THREE.Matrix4();
    let fly = null;
    const st = {};
    const startFly = (view, c) => {
      const v = COMET_VIEW[view];
      const tgt = V(...v.tgt);
      const dir = c.camera.position.clone().sub(c.controls.target).normalize();
      if (view === 'all') dir.setFromSphericalCoords(1, THREE.MathUtils.degToRad(v.polar), THREE.MathUtils.degToRad(v.az));
      fly = { t0: c.time.wall, dur: 3.2, fromT: c.controls.target.clone(), toT: tgt, fromD: c.camera.position.distanceTo(c.controls.target), toD: cometViewDist(v, c.params.preset), fromDir: c.camera.position.clone().sub(c.controls.target).normalize(), toDir: dir };
    };
    comet.params.find(p => p.key === 'view').onChange = (v, c) => startFly(v, c);
    return {
      material,
      update(dt, c) {
        const p = c.params, pr = COMETS[p.preset] || COMETS.halley;
        const r = p.r;
        const days = c.time.sim * 600 / 86400;          // 1 simulated second = 10 minutes of comet time
        // geometry: Sun along +x, orbit in the xz plane
        const psi = THREE.MathUtils.degToRad(p.psi);
        const vhat = V(Math.cos(psi), 0, Math.sin(psi));
        const a = S.clone().negate();
        const lag = vhat.clone().addScaledVector(a, -vhat.dot(a)).negate();
        if (lag.lengthSq() < 1e-6) lag.set(0, 0, -1); else lag.normalize();
        const nrm = V().crossVectors(a, lag).normalize();
        const vc = 42.1 / Math.sqrt(r);                  // parabolic orbital speed, km/s
        // the ion tail is aberrated: it lines up with the solar wind seen from the moving comet (~400 km/s)
        const ionAx = a.clone().multiplyScalar(400).addScaledVector(vhat, -vc).normalize();
        const e1 = lag.clone().addScaledVector(ionAx, -lag.dot(ionAx)).normalize(), e2 = V().crossVectors(ionAx, e1);
        // activity
        const act = waterG(r) + 0.012 / (r * r);         // + weak CO/CO2-driven activity far out
        const Q = pr.Q1 * act, afrho = pr.afrho * act;   // H2O molecules/s, Afρ in metres
        const sunF = Math.pow(r, -0.5);                  // flux ∝ r⁻², camera compensates r^1.5
        const g = 4.43 / (r * r);                        // β=1 radiation-pressure accel., units/day²
        const omegaFull = 0.0243 / Math.pow(r, 1.5);
        const omega = omegaFull * Math.sin(psi);
        const tauMax = Math.min(36 * Math.pow(act, 0.35), 45, 1.1 / omegaFull);
        const vej = 3.9 / Math.sqrt(r);
        const Xs = vej * vej / (2 * 0.5 * g);
        const Lp = 2.2 * r * r, Ld = 6.6 * r * r;        // C2 Haser scale lengths (×10⁴ km)
        const Rc = Math.min(90, Math.max(6, 6 * Math.max(Ld, Xs)));
        const Lion = 2600 * Math.sqrt(Math.min(act, 2)) * Math.pow(pr.Q1 / 1e30, 0.12);
        const dustLen = 0.5 * 1.0 * g * tauMax * tauMax;
        st.act = act; st.Q = Q; st.afrho = afrho; st.Lion = Lion; st.dustLen = dustLen; st.vc = vc; st.days = days; st.tauMax = tauMax;
        U.uSun.value.copy(S); U.uA.value.copy(a); U.uLag.value.copy(lag); U.uNrm.value.copy(nrm);
        U.uIonAx.value.copy(ionAx); U.uIonE1.value.copy(e1); U.uIonE2.value.copy(e2);
        U.uG.value = g; U.uOmega.value = omega; U.uTauMax.value = tauMax; U.uVej.value = vej; U.uXs.value = Xs;
        U.uLp.value = Lp; U.uLd.value = Ld; U.uRc.value = Rc;
        // brightness: Afρ fixes the dust coma surface brightness B(b) = K·Afρ/(2b)
        const K = 4.5;
        U.uC.value = K * (afrho / 1e7) / (2 * Math.PI) * sunF;
        U.uGas.value = 0.55 * U.uC.value / Math.max(pr.afrho / pr.Q1 * 1e30 / 300, 1e-3) * Math.pow(r, -0.0);
        U.uIon.value = p.ion ? 0.6e-4 * Math.pow(Q / 1e30, 0.6) * Math.pow(act, 0.4) * sunF * (r > 3.2 ? Math.max(0, (4.2 - r)) : 1) : 0;
        U.uLion.value = Lion; U.uVion.value = 1700; U.uDays.value = days; U.uSunF.value = sunF;
        U.uSunAng.value = 0.00465 / r;
        U.uNucS.value = 1e-4 * pr.scale;
        // nucleus rotation (12.4 h for 67P; scaled by size for the others)
        const per = 0.517 * Math.sqrt(pr.scale);
        mSpin.makeRotationZ(2 * Math.PI * days / per);
        m4.multiplyMatrices(mObl, mSpin);
        U.uNucRot.value.setFromMatrix4(m4);
        JETS.forEach((j, i) => {
          const d = j.dir.clone().applyMatrix4(m4), pos = j.pos.clone().applyMatrix4(m4).multiplyScalar(1e-4 * pr.scale);
          const lit = THREE.MathUtils.smoothstep(d.dot(S), -0.15, 0.5);
          U.uJetP.value[i].set(pos.x, pos.y, pos.z, lit * j.str * Math.min(1, act * 3));
          U.uJetD.value[i].copy(d);
        });
        // bounding sphere of the tails
        const tm = tauMax * 1.35;
        const pts = [V(), ionAx.clone().multiplyScalar(Lion * 2.2)];
        for (const b of [0.06, 1.4]) {
          const bg = b * g, X = Math.min(0.5 * bg * tm * tm, 9000);
          const tau = Math.sqrt(2 * X / bg);
          pts.push(a.clone().multiplyScalar(X).addScaledVector(lag, bg * omega * tau ** 3 / 3));
        }
        const box = new THREE.Box3().setFromPoints(pts);
        const ctr = box.getCenter(V());
        let rad = 0; for (const q of pts) rad = Math.max(rad, q.distanceTo(ctr));
        rad += vej * tm * 1.8 + Rc + 20;
        U.uBound.value.set(ctr.x, ctr.y, ctr.z, rad);
        U.uZmax.value = 3.2 * (vej * 1.18 * tm * 0.6 + 0.3) + 2;
        // auto-exposure: a long exposure for the faint coma and tails, a short one next to the nucleus
        const dCam = c.camera.position.length();
        const Emax = 2.4e4 * Math.pow(300 / pr.afrho, 0.72) * Math.pow(Math.max(act, 1e-3), -0.35);   // the eye/camera adapts partly
        const q = dCam / (dCam + 1.0);
        U.uExpo.value = 1 + Emax * q * q * Math.min(1, Math.pow(dCam / 2000, 0.75));
        // photographic stretch only once we are far from the nucleus (Rosetta-like close-ups stay linear)
        const lk = THREE.MathUtils.smoothstep(Math.log10(dCam * 1e4 / pr.rkm), 1.3, 3.3);
        U.uStretch.value = lk;
        // near the nucleus the faint jets are shown contrast-stretched, as in processed Rosetta NAVCAM frames
        U.uComaGain.value = 1 + Math.min(19, 20 * Math.pow(20 / Math.max(afrho, 1e-3), 0.6)) * (1 - THREE.MathUtils.smoothstep(Math.log10(dCam * 1e4 / pr.rkm), 1.0, 2.5));
        // camera
        c.controls.minDistance = 3.3 * 1e-4 * pr.scale * 1.25;
        if (fly) {
          const k = Math.min(1, (c.time.wall - fly.t0) / fly.dur), e = k * k * (3 - 2 * k);
          const tgt = fly.fromT.clone().lerp(fly.toT, e);
          const dist = Math.exp(THREE.MathUtils.lerp(Math.log(fly.fromD), Math.log(fly.toD), e));
          const dir = fly.fromDir.clone().lerp(fly.toDir, e).normalize();
          c.controls.target.copy(tgt); c.camera.position.copy(tgt).addScaledVector(dir, dist); c.camera.lookAt(tgt);
          if (k >= 1) fly = null;
        }
      },
      hud(c) {
        const p = c.params;
        const kgs = st.Q * 18 * 1.6605e-27;
        return [
          [t('Cách hạt nhân', 'Distance from nucleus'), c.fmt.dist(c.camera.position.length() * 1e7)],
          [t('Cách Mặt Trời', 'Distance from the Sun'), p.r.toFixed(2) + ' AU · ' + c.fmt.sci(p.r * 1.496e8) + ' km'],
          [t('Tốc độ quỹ đạo', 'Orbital speed'), st.vc.toFixed(1) + ' km/s'],
          [t('Sinh khí H₂O', 'H₂O production'), c.fmt.sci(st.Q) + t(' phân tử/s (', ' molecules/s (') + (kgs >= 1000 ? (kgs / 1000).toFixed(1) + t(' tấn/s', ' t/s') : kgs.toFixed(kgs < 1 ? 2 : 0) + ' kg/s') + ')'],
          [t('Afρ (độ bụi)', 'Afρ (dust activity)'), (st.afrho * 100).toFixed(0) + ' cm'],
          [t('Đuôi ion', 'Ion tail'), p.ion ? (st.Lion * 1e4 / 1e6).toFixed(1) + t(' triệu km', ' million km') : t('tắt', 'off')],
          [t('Đuôi bụi', 'Dust tail'), t((st.dustLen * 1e4 / 1e6).toFixed(1) + ' triệu km (hạt bụi già ' + st.tauMax.toFixed(0) + ' ngày)', (st.dustLen * 1e4 / 1e6).toFixed(1) + ' million km (dust up to ' + st.tauMax.toFixed(0) + ' days old)')],
        ];
      },
      onParam(key, v, c) {
        if (key === 'preset' && c.params.view === 'nucleus') startFly('nucleus', c);
        if (key === 'r' && v > 3.2 && (this._lastR ?? 0) <= 3.2) c.toast(t('Ngoài ~3 AU, băng nước gần như không thăng hoa: chỉ còn chút CO và CO₂ — sao chổi trông như một tiểu hành tinh.', 'Beyond ~3 AU water ice barely sublimates: only a little CO and CO₂ is left — the comet looks like an asteroid.'));
        if (key === 'r') this._lastR = v;
      },
    };
  },
};

/* =============================================================================================
   ASTEROID & INTERSTELLAR OBJECT (shared shader)
   ============================================================================================= */
const ROCK_FRAG = COMMON_GLSL + HELPERS + /* glsl */`
uniform vec3 uSun;
uniform mat3 uRot;            // body -> world
uniform float uShape;         // 0 Bennu, 1 Itokawa, 2 'Oumuamua cigar, 3 'Oumuamua pancake
uniform float uAlb, uRough;
uniform vec3 uTint;
uniform float uBR;            // bounding radius
uniform highp sampler3D uNoise3;  // baked tileable smooth value noise (16 lattice cells, 8 voxels per cell)
// ~[-1,1] noise with unit feature size, like snoise but a single filtered texture fetch (no derivatives)
float tnoise(vec3 p){ return textureLod(uNoise3, p * 0.0625, 0.0).r; }
// DART
uniform vec3 uSite, uAxis, uInDir;
uniform float uTi;            // seconds since impact (negative: impactor still incoming; < -50: none)

float baseShape(vec3 p){
  if (uShape < 0.5){
    // Bennu: "spinning top" — equatorial ridge built by YORP spin-up, 565 × 535 × 508 m
    float rr = length(p.xz);
    float cone = dot(vec2(rr, abs(p.y)), vec2(0.8423, 0.539)) - 0.9434;   // Re = 1.12, Hp = 1.75
    float ell = sdEll(p, vec3(1.2, 1.0, 1.17));
    return smax(cone, ell, 0.18);
  } else if (uShape < 1.5){
    // Itokawa: contact binary "sea otter", 535 × 294 × 209 m
    float body = sdEll(p - vec3(-0.33, 0.0, 0.0), vec3(0.76, 0.42, 0.57));
    float head = sdEll(p - vec3(0.6, 0.03, 0.04), vec3(0.47, 0.38, 0.45));
    return smin(body, head, 0.22);
  } else if (uShape < 2.5){
    return sdEll(p, vec3(1.7, 0.21, 0.23));
  }
  return sdEll(p, vec3(0.575, 0.095, 0.555));
}
float craters(vec3 p){
  // 2×2×2 cell search: centres jittered in [0.3, 0.7], rim support 1.6R ≤ 0.72 < 0.8
  vec3 g = p * 2.6; vec3 b = floor(g - 0.5); float h = 0.0;
  for (int z = 0; z <= 1; z++) for (int y = 0; y <= 1; y++) for (int x = 0; x <= 1; x++){
    vec3 c = b + vec3(x, y, z); vec3 r = hash33(c + 31.0);
    if (r.z > 0.45) continue;
    float R = 0.25 + 0.2 * fract(r.x * 41.3 + r.y * 7.7);
    float d = length(c + 0.3 + 0.4 * r - g) / R;
    if (d < 1.6) h += R * ((d < 1.0 ? (d * d - 1.0) * 0.32 : 0.0) + 0.07 * exp(-(d - 1.0) * (d - 1.0) * 25.0));
  }
  return h / 2.6;
}
float roughMask(vec3 p){
  if (uShape < 0.5) return 1.0;
  // Itokawa: rough boulder-strewn highlands vs the smooth gravel "seas" (Muses-C, Sagamihara)
  float m = smoothstep(-0.3, 0.2, tnoise(p * 1.3 + 3.0));
  m *= smoothstep(0.1, 0.38, abs(p.x - 0.2) + 0.12 * (p.y + 0.4));
  return m;
}
// Hierarchical SDF: each detail layer only displaces the surface outward by a bounded amount, so far from
// the surface a cheap lower bound (coarse distance − remaining max displacement) is returned instead.
float rockSDF(vec3 p, float lod){
  float d = baseShape(p);
  if (uShape > 1.5){
    if (d > 0.03) return d - 0.014;
    return d - 0.01 * tnoise(p * 3.0) - 0.004 * tnoise(p * 9.0);
  }
  if (d > 0.22) return d - 0.19;                                   // all layers together add ≤ 0.19
  float m = roughMask(p);
  d -= 0.045 * tnoise(p * 1.1 + 2.0) + 0.02 * tnoise(p * 2.7) * mix(0.3, 1.0, m);
  float rest = 0.072 + (lod > 0.5 ? 0.034 : 0.0) + (lod > 1.5 ? 0.011 : 0.0);
  if (d > rest + 0.045) return d - rest - 0.013;                  // crater rims add ≤ 0.013
  d -= craters(p) * (uShape < 0.5 ? 1.0 : 0.5);
  if (d > rest + 0.03) return d - rest;
  float bd = uShape < 0.5 ? 0.5 : 0.32;
  d -= boulders(p, 0.21, bd * m, 1.0);
  if (lod > 0.5){
    if (d > 0.06) return d - 0.043;
    d -= boulders(p, 0.07, (0.7 * m + 0.12), 7.0);
    d -= (0.006 * tnoise(p * 42.0) + 0.003 * tnoise(p * 97.0 + 5.0)) * mix(0.25, 1.0, m);   // fractured rock and cobbles; fine gravel in the seas
  }
  if (lod > 1.5 && d < 0.02) d -= boulders(p, 0.03, 0.55 * m + 0.2, 17.0);
  return d;
}
vec3 rockNormal(vec3 p, float e, float lod){
  vec2 k = vec2(1.0, -1.0);
  return normalize(k.xyy * rockSDF(p + k.xyy * e, lod) + k.yyx * rockSDF(p + k.yyx * e, lod) +
                   k.yxy * rockSDF(p + k.yxy * e, lod) + k.xxx * rockSDF(p + k.xxx * e, lod));
}
// cheaper SDF for shadow rays: base shape, low-frequency lumps and the large boulders only
float rockSDFs(vec3 p){
  float d = baseShape(p);
  if (uShape > 1.5) return d;
  if (d > 0.15) return d - 0.12;
  float m = roughMask(p);
  d -= 0.045 * tnoise(p * 1.1 + 2.0) + 0.02 * tnoise(p * 2.7) * mix(0.3, 1.0, m);
  if (d > 0.13) return d - 0.085;
  d -= craters(p) * (uShape < 0.5 ? 1.0 : 0.5);                  // needed: rays leave from crater floors
  if (d > 0.1) return d - 0.072;
  return d - boulders(p, 0.21, (uShape < 0.5 ? 0.5 : 0.32) * m, 1.0);
}
float rockShadow(vec3 p, vec3 l){
  // shadows only need the large relief (base + craters + big boulders)
  float res = 1.0, t = 0.008;
  for (int i = 0; i < 14; i++){
    float h = rockSDFs(p + l * t);
    res = min(res, 10.0 * h / t);
    t += clamp(h * 1.2, 0.012, 0.35);
    if (res < 0.01 || t > 2.6) break;
  }
  return clamp(res, 0.0, 1.0);
}

// ---- DART ejecta: hollow cone (half-angle ~ 70°) with ray-like streamers ----------------------------
float ejecta(vec3 p, out float shade){
  shade = 1.0;
  if (uTi <= 0.0) return 0.0;
  vec3 q = p - uSite;
  float h = dot(q, uAxis);
  vec3 lat = q - uAxis * h;
  float rl = length(lat);
  float r = length(q);
  float Rf = 0.55 * pow(uTi, 0.8) + 0.02;          // the fastest ejecta lead; slow material stays low
  if (r > Rf * 1.15 || h < -0.02) return 0.0;
  float th = atan(rl, max(h, 1e-4));
  float alpha = 1.1;                               // cone half-angle (DART: opening angle ~125–140°)
  float sq = (th - alpha) / 0.2;
  float shell = exp(-sq * sq) + 0.25 * exp(-th * th * 4.0);
  vec3 e1 = normalize(cross(uAxis, abs(uAxis.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
  vec3 e2 = cross(uAxis, e1);
  float phi = atan(dot(lat, e2), dot(lat, e1));
  float rays = 0.12 + 1.9 * pow(clamp(0.5 + 0.5 * snoise(vec3(cos(phi) * 2.6, sin(phi) * 2.6, 1.7)), 0.0, 1.0), 3.0)
             + 1.2 * pow(clamp(0.5 + 0.5 * snoise(vec3(cos(phi) * 7.0, sin(phi) * 7.0, r * 3.0 - uTi * 0.3)), 0.0, 1.0), 6.0);
  float x = r / Rf;
  float radial = smoothstep(1.12, 0.85, x) * (0.35 + 1.0 / (0.15 + 6.0 * x));   // most mass is slow
  float clump = 0.35 + 1.3 * smoothstep(-0.3, 0.8, snoise(q * 7.0 + vec3(0.0, uTi * 0.2, 0.0)));
  float fade = exp(-uTi / 18.0);
  return shell * rays * radial * clump * fade / (Rf * Rf + 0.05) * 1.2;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  float pix = 2.0 * uTanHalfFov / uRes.y;
  vec3 col = vec3(0.0);
  vec2 bh = sphereHit(ro, rd, vec3(0.0), uBR);
  float tHit = 1e9;
  if (bh.x < bh.y && bh.y > 0.0){
    mat3 iR = transpose(uRot);
    vec3 o = iR * ro, d = iR * rd;
    float t = max(bh.x, 0.0), tEnd = bh.y;
    float hitT = -1.0;
    for (int i = 0; i < 90; i++){
      vec3 p = o + d * t;
      float fp = t * pix;                                   // pixel footprint
      float lod = fp < 0.0006 ? 2.0 : (fp < 0.005 ? 1.0 : 0.0);
      float h = rockSDF(p, lod);
      if (h < max(0.0004, fp * 0.9)){ hitT = t; break; }
      t += h * (h > 0.05 ? 0.95 : 0.8);                    // relaxed while far, conservative near relief
      if (t > tEnd) break;
    }
    if (hitT > 0.0){
      vec3 p = o + d * hitT;
      float lod = hitT * pix < 0.0006 ? 2.0 : (hitT * pix < 0.005 ? 1.0 : 0.0);
      vec3 nb = rockNormal(p, max(0.0015, hitT * pix * 0.8), lod);

      vec3 n = uRot * nb, L = uSun, Vv = -rd;
      float mu0 = max(dot(n, L), 0.0), mu = max(dot(n, Vv), 1e-3);
      float sh = mu0 > 0.0 ? rockShadow(p + nb * 0.004, iR * L) : 0.0;
      float cg = clamp(dot(L, Vv), -1.0, 1.0), g = acos(cg);
      // Hapke-style: Lommel–Seeliger × single-particle phase × shadow-hiding opposition surge
      float ls = mu0 / (mu0 + mu);
      // normalised so that zero phase gives the geometric albedo (surge included)
      float P = hg(-cg, -0.28) / hg(-1.0, -0.28);
      float B = (1.0 + 1.1 / (1.0 + tan(g * 0.5) / 0.05)) / 2.1;
      float tex = 0.8 + 0.25 * tnoise(p * 21.0) + 0.15 * tnoise(p * 75.0);
      // boulders differ in brightness (Bennu: dark and bright boulders); Itokawa seas are smoother/brighter
      float bv = 0.5 + 0.5 * tnoise(p * 9.0 + 3.0);
      tex *= 0.75 + 0.5 * bv * bv;
      float m = roughMask(p);
      vec3 tint = uTint * (uShape > 0.5 && uShape < 1.5 ? mix(vec3(1.06, 1.03, 0.98), vec3(0.95, 0.95, 0.97), m) : vec3(1.0));
      col = tint * uAlb * tex * ls * P * B * sh * 18.0;
      col *= pow(0.1 / uAlb, 0.45);                 // the camera exposure partly adapts (Bennu vs Itokawa: 6.6× in albedo)
      col += tint * uAlb * 0.0015;                 // faint fill (scattered starlight)
      tHit = hitT;
    }
  }
  if (tHit > 1e8) col = sky(rd, 1.0) * 0.6 + sunDisk(rd, uSun, 0.0047 / 1.13);   // background only where the body is absent
  // DART ejecta + impactor
  if (uTi > -50.0){
    float Rf = 0.55 * pow(max(uTi, 0.0), 0.8) + 0.02;
    vec2 eb = sphereHit(ro, rd, uSite + uAxis * Rf * 0.4, Rf * 1.25 + 0.05);
    if (uTi > 0.0 && eb.x < eb.y && eb.y > 0.0){
      float t0 = max(eb.x, 0.0), t1 = min(eb.y, tHit);
      float jit = hash13(vec3(gl_FragCoord.xy, 1.3));
      float dtt = (t1 - t0) / 48.0;
      vec3 sc = vec3(0.0); float T = 1.0;
      float ph = (0.6 * hg(dot(rd, uSun), 0.6) + 0.4 * hg(dot(rd, uSun), -0.2));
      for (int i = 0; i < 48; i++){
        if (t1 <= t0) break;
        float tt = t0 + dtt * (float(i) + jit);
        vec3 p = ro + rd * tt;
        float shd;
        float dens = ejecta(p, shd) * 4.0;
        if (dens <= 0.0) continue;
        // the asteroid shadows the plume where the line toward the Sun passes through the body
        vec2 sb = sphereHit(p, uSun, vec3(0.0), 0.95 * uBR * 0.85);
        float lit = (sb.x < sb.y && sb.y > 0.0) ? 0.25 : 1.0;
        float a = dens * dtt;
        sc += T * a * vec3(1.0, 0.95, 0.88) * ph * lit * 0.35;
        T *= exp(-a * 0.8);
        if (T < 0.02) break;
      }
      col = col * T + sc;
    }
    // the spacecraft (sunlit glint) and the impact flash
    vec3 ip = uSite - uInDir * max(-uTi, 0.0) * 3.0;
    vec3 v = ip - ro; float vd = length(v);
    float ang = acos(clamp(dot(rd, v / vd), -1.0, 1.0)) / pix;
    if (uTi < 0.0 && vd < tHit + 0.05) col += vec3(1.0, 0.95, 0.85) * (exp(-ang * ang * 0.5) * 6.0 + exp(-ang * 0.25) * 0.15);
    if (uTi >= 0.0){
      float fl = exp(-uTi / 0.25);
      col += vec3(1.0, 0.85, 0.65) * fl * (exp(-ang * ang / 18.0) * 60.0 + exp(-ang / 25.0) * 4.0);
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// ---- shapes on the CPU (base shape only, for the DART impact site and light curves) --------------
function ellJS(x, y, z, rx, ry, rz) {
  const k0 = Math.hypot(x / rx, y / ry, z / rz), k1 = Math.hypot(x / (rx * rx), y / (ry * ry), z / (rz * rz));
  return k0 * (k0 - 1) / Math.max(k1, 1e-6);
}
function sminJS(a, b, k) { const h = Math.min(1, Math.max(0, 0.5 + 0.5 * (b - a) / k)); return b + (a - b) * h - k * h * (1 - h); }
function rockBaseJS(shape) {
  if (shape === 0) return (x, y, z) => { const cone = Math.hypot(x, z) * 0.8423 + Math.abs(y) * 0.539 - 0.9434; return -sminJS(-cone, -ellJS(x, y, z, 1.2, 1.0, 1.17), 0.18); };
  if (shape === 1) return (x, y, z) => sminJS(ellJS(x + 0.33, y, z, 0.76, 0.42, 0.57), ellJS(x - 0.6, y - 0.03, z - 0.04, 0.47, 0.38, 0.45), 0.22);
  if (shape === 2) return (x, y, z) => ellJS(x, y, z, 1.7, 0.21, 0.23);
  return (x, y, z) => ellJS(x, y, z, 0.575, 0.095, 0.555);
}

const ROCKS = {
  bennu:    { shape: 0, alb: 0.044, tint: [0.93, 0.96, 1.0], br: 1.3, period: 4.296, GM: 6.674e-11 * 7.329e10, Rm: 245, Req: 282, dims: '565 × 535 × 508 m', name: '101955 Bennu' },
  itokawa:  { shape: 1, alb: 0.29, tint: [1.0, 0.9, 0.78], br: 1.2, period: 12.132, GM: 6.674e-11 * 3.51e10, Rm: 165, Req: 267, dims: '535 × 294 × 209 m', name: '25143 Itokawa' },
  cigar:    { shape: 2, alb: 0.1, tint: [1.0, 0.76, 0.6], br: 1.8, period: 8.67, GM: 6.674e-11 * 5e8, Rm: 44, Req: 170, dims: '≈ 340 × 42 × 46 m', name: "1I/ʻOumuamua (hình điếu xì gà)", nameEn: '1I/ʻOumuamua (cigar shape)' },
  pancake:  { shape: 3, alb: 0.1, tint: [1.0, 0.76, 0.6], br: 0.65, period: 8.67, GM: 6.674e-11 * 2e8, Rm: 31, Req: 57, dims: '≈ 115 × 111 × 19 m', name: "1I/ʻOumuamua (hình đĩa)", nameEn: '1I/ʻOumuamua (pancake shape)' },
};

// Tileable smooth value noise baked into a 128³ half-float 3D texture (period 16, 8 voxels per lattice
// cell, quintic interpolation, rescaled to ~[-1, 1]). One textureLod replaces an Ashima snoise call in
// the asteroid SDF, which is evaluated ~40× per pixel.
let NOISE3 = null;
function noiseTexture() {
  if (NOISE3) return NOISE3;
  const L = 16, R = 128, V = R / L, lat = new Float32Array(L * L * L);
  let seed = 1234567;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < lat.length; i++) lat[i] = rnd() * 2 - 1;
  const at = (x, y, z) => lat[((z & 15) * L + (y & 15)) * L + (x & 15)];
  const q = t => t * t * t * (t * (t * 6 - 15) + 10);
  const out = new Float32Array(R * R * R);
  let sum = 0, sum2 = 0;
  for (let z = 0; z < R; z++) for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
    const fx = (x + 0.5) / V, fy = (y + 0.5) / V, fz = (z + 0.5) / V;
    const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
    const u = q(fx - ix), v = q(fy - iy), w = q(fz - iz);
    const l = (a, b, t) => a + (b - a) * t;
    const val = l(l(l(at(ix, iy, iz), at(ix + 1, iy, iz), u), l(at(ix, iy + 1, iz), at(ix + 1, iy + 1, iz), u), v),
                  l(l(at(ix, iy, iz + 1), at(ix + 1, iy, iz + 1), u), l(at(ix, iy + 1, iz + 1), at(ix + 1, iy + 1, iz + 1), u), v), w);
    out[(z * R + y) * R + x] = val; sum += val; sum2 += val * val;
  }
  const n = R * R * R, mean = sum / n, sd = Math.sqrt(sum2 / n - mean * mean);
  const half = new Uint16Array(n);
  for (let i = 0; i < n; i++) half[i] = THREE.DataUtils.toHalfFloat(Math.max(-1, Math.min(1, (out[i] - mean) / sd * 0.45)));
  const t = new THREE.Data3DTexture(half, R, R, R);
  t.format = THREE.RedFormat; t.type = THREE.HalfFloatType;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.RepeatWrapping;
  t.unpackAlignment = 1; t.needsUpdate = true;
  return (NOISE3 = t);
}

function makeRockInstance(ctx, mode) {
  const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
  const uniforms = {
    ...ctx.shared,
    uSun: { value: V(1, 0, 0) }, uRot: { value: new THREE.Matrix3() }, uShape: { value: 0 },
    uAlb: { value: 0.044 }, uRough: { value: 1 }, uTint: { value: V(1, 1, 1) }, uBR: { value: 1.3 },
    uNoise3: { value: noiseTexture() }, uSite: { value: V(1, 0, 0) }, uAxis: { value: V(1, 0, 0) }, uInDir: { value: V(-1, 0, 0) }, uTi: { value: -100 },
  };
  const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: ROCK_FRAG, depthTest: false, depthWrite: false });
  const U = uniforms;
  let phase = 0.6, prec = 0.3, rate = 1, lastSim = null;
  let dart = null;                         // { t0, siteB, axisB, inB }
  const lc = [];                           // light-curve history
  const m4 = new THREE.Matrix4(), mA = new THREE.Matrix4(), mB = new THREE.Matrix4(), mC = new THREE.Matrix4(), mTilt = new THREE.Matrix4().makeRotationZ(0.35);
  const camDir = V(), tmp = V();
  // free precession of a nearly axisymmetric body: spin about a body axis (angle ph) while that axis
  // cones round the fixed angular-momentum vector (angle pr, half-angle set by the tumble amount)
  function bodyRot(out, ph, pr, tumble, shape) {
    mA.makeRotationY(pr);
    if (shape === 3) { mB.makeRotationZ(0.9 * tumble + 0.15); mC.makeRotationX(ph); }   // disk flips edge-on/face-on
    else if (shape === 2) { mB.makeRotationX(0.9 * tumble + 0.1); mC.makeRotationY(ph); } // cigar: end-on/side-on
    else { mB.makeRotationX(0.65 * tumble); mC.makeRotationY(ph); }
    return out.copy(mTilt).multiply(mA).multiply(mB).multiply(mC);
  }
  const st = { geff: 0, crit: 0, bright: 1 };
  const q = new URLSearchParams(location.search);
  let pendingDart = q.has('snap') && q.get('dart') !== null ? +q.get('dart') : null;
  const rockOf = p => ROCKS[p.shape] || (mode === 'oumuamua' ? ROCKS.cigar : ROCKS.bennu);

  function triggerDart(c) {
    const p = c.params, rk = rockOf(p);
    const f = rockBaseJS(rk.shape);
    camDir.copy(c.camera.position).normalize();
    const up = V(0, 1, 0), rt = V().crossVectors(up, camDir).normalize();
    // a sunlit spot on the visible face, off-centre so the ejecta cone is seen obliquely
    const sideSun = Math.sign(U.uSun.value.dot(rt)) || 1;
    const wdir = camDir.clone().multiplyScalar(0.6).addScaledVector(rt, 0.62 * sideSun).addScaledVector(up, 0.3).normalize();
    const Rm = U.uRot.value.clone().transpose();
    const bdir = wdir.clone().applyMatrix3(Rm).normalize();
    const siteB = surfaceAlong(f, bdir, 3);
    const nB = gradJS(f, siteB);
    // impactor comes in from the side, roughly opposite the camera-right direction
    const right = V().crossVectors(up, camDir).normalize();
    const inW = camDir.clone().multiplyScalar(-0.55).addScaledVector(right, -0.7).addScaledVector(up, -0.2).normalize();
    // make sure it actually comes from above the surface
    const nW = nB.clone().applyMatrix3(U.uRot.value);
    if (inW.dot(nW) > -0.3) inW.addScaledVector(nW, -(inW.dot(nW) + 0.6)).normalize();
    dart = { t0: c.time.wall + 1.4, siteB, axisB: nB.clone().lerp(inW.clone().applyMatrix3(Rm).negate(), 0.2).normalize(), inB: inW.clone().applyMatrix3(Rm), toasted: false };
  }

  return {
    material,
    triggerDart,
    update(dt, c) {
      const p = c.params, rk = rockOf(p);
      // phase angle measured from the camera, as in the Moon scene
      camDir.copy(c.camera.position).normalize();
      const side = V().crossVectors(V(0, 1, 0), camDir).normalize();
      const a = THREE.MathUtils.degToRad(p.sunAngle);
      U.uSun.value.copy(camDir).multiplyScalar(Math.cos(a)).addScaledVector(side, Math.sin(a)).addScaledVector(V(0, 1, 0), 0.25 * Math.sin(a)).normalize();
      // slow motion while the DART sequence plays
      let ti = dart ? c.time.wall - dart.t0 : -100;
      const want = dart && ti < 25 ? 0.04 : 1;
      rate += (want - rate) * Math.min(1, dt * 2);
      // free precession: spin about the body axis while that axis cones round the angular momentum
      const simDt = Math.max(0, Math.min(1, c.time.sim - (lastSim ?? c.time.sim))) * rate;
      lastSim = c.time.sim;
      const ts = 100;                         // 1 simulated second = 100 s of asteroid time
      const P = p.period * 3600;
      const w = 2 * Math.PI / P * ts;
      phase += w * simDt * (1 - 0.45 * p.tumble);
      prec += w * simDt * 0.62 * p.tumble;
      bodyRot(m4, phase, prec, p.tumble, rk.shape);
      U.uRot.value.setFromMatrix4(m4);
      U.uShape.value = rk.shape; U.uAlb.value = rk.alb; U.uTint.value.set(...rk.tint); U.uBR.value = rk.br;
      if (pendingDart !== null) { triggerDart(c); dart.t0 = c.time.wall - pendingDart; dart.toasted = true; pendingDart = null; }   // snapshot testing: ?dart=<s since impact>
      // DART
      if (dart) {
        ti = c.time.wall - dart.t0;
        U.uTi.value = ti;
        U.uSite.value.copy(dart.siteB).applyMatrix4(m4);
        U.uAxis.value.copy(dart.axisB).transformDirection(m4);
        U.uInDir.value.copy(dart.inB).transformDirection(m4);
        if (ti > 0 && !dart.toasted) {
          dart.toasted = true;
          c.audio.tone(70, 1.6, 0.25, 0, 'sine'); c.audio.click(0.5, 0, 0.3);
          c.toast(t('<b>Va chạm!</b> Ngày 26/9/2022 tàu DART (570 kg, 6,1 km/s) đâm vào Dimorphos. Chùm vật chất bắn ra mang thêm động lượng — gấp ~3,6 lần động lượng của chính con tàu — nên chu kỳ quỹ đạo của Dimorphos quanh Didymos ngắn đi <b>32 phút</b> (11 giờ 55 → 11 giờ 23). Lần đầu tiên loài người dịch chuyển một thiên thể.', '<b>Impact!</b> On 26 September 2022 the DART spacecraft (570 kg, 6.1 km/s) slammed into Dimorphos. The ejecta plume added extra momentum — about 3.6 times the spacecraft\'s own — so Dimorphos\'s orbital period around Didymos shrank by <b>32 minutes</b> (11 h 55 min → 11 h 23 min). The first time humanity has changed the motion of a celestial body.'), 12000);
        }
        if (ti > 60) { dart = null; U.uTi.value = -100; }
      } else U.uTi.value = -100;
      // effective gravity at the equator (centrifugal relief)
      const om = 2 * Math.PI / P;
      st.geff = rk.GM / (rk.Req * rk.Req) - om * om * rk.Req;
      st.crit = 2 * Math.PI / Math.sqrt(rk.GM / rk.Req ** 3) / 3600;
      st.vesc = Math.sqrt(2 * rk.GM / rk.Rm);
      // light curve: reflected light ∝ projected cross-section toward Sun and observer (ellipsoid approx.),
      // predicted over the next two rotations from the same free-precession model
      if (rk.shape >= 2) {
        const ax = rk.shape === 2 ? [1.7, 0.21, 0.23] : [0.575, 0.095, 0.555];
        const sun = U.uSun.value;
        const mm = new THREE.Matrix4(), Rt = new THREE.Matrix3();
        const area = (u, M) => { const b = u.clone().applyMatrix3(M); return Math.PI * ax[0] * ax[1] * ax[2] * Math.hypot(b.x / ax[0], b.y / ax[1], b.z / ax[2]); };
        lc.length = 0;
        for (let i = 0; i < 48; i++) {
          const dtv = (i / 48) * 2 * P / ts;                     // two periods, in simulated seconds
          const ph = phase + w * dtv * (1 - 0.45 * p.tumble), pr2 = prec + w * dtv * 0.62 * p.tumble;
          bodyRot(mm, ph, pr2, p.tumble, rk.shape);
          Rt.setFromMatrix4(mm).transpose();
          lc.push(Math.sqrt(area(camDir, Rt) * area(sun, Rt)));
        }
      }
    },
    hud(c) {
      const p = c.params, rk = rockOf(p);
      const rows = [
        [t('Thiên thể', 'Object'), t(rk.name, rk.nameEn)],
        [t('Kích thước', 'Size'), rk.dims],
        [t('Chu kỳ tự quay', 'Rotation period'), p.period.toFixed(2) + t(' giờ', ' h') + (p.tumble > 0.02 ? t(' · lộn nhào (NPA)', ' · tumbling (NPA)') : '')],
        [t('Vận tốc thoát', 'Escape velocity'), (st.vesc * 100).toFixed(0) + ' cm/s'],
      ];
      if (mode !== 'oumuamua') {
        rows.push([t('Trọng lực hiệu dụng ở xích đạo', 'Effective gravity at the equator'), st.geff > 0 ? (st.geff * 1e6).toFixed(1) + ' µm/s²' : t('<span style="color:#ff8a6a">âm — đá văng khỏi xích đạo!</span>', '<span style="color:#ff8a6a">negative — rocks fly off the equator!</span>')]);
        rows.push([t('Chu kỳ giới hạn (vỡ do quay)', 'Critical period (spin breakup)'), st.crit.toFixed(2) + t(' giờ', ' h')]);
      } else if (lc.length) {
        const mx = Math.max(...lc), mn = Math.min(...lc);
        const bars = '▁▂▃▄▅▆▇█';
        const spark = lc.filter((_, i) => i % 2 === 0).map(v => bars[Math.min(7, Math.floor((v - mn) / Math.max(mx - mn, 1e-6) * 7.99))]).join('');
        rows.push([t('Biên độ quang biến', 'Light-curve amplitude'), (2.5 * Math.log10(mx / Math.max(mn, 1e-6))).toFixed(1) + t(' cấp sao (×', ' mag (×') + (mx / Math.max(mn, 1e-6)).toFixed(0) + ')']);
        rows.push([t('Đường cong ánh sáng (2 vòng quay tới)', 'Light curve (next 2 rotations)'), '<span style="font-family:monospace;letter-spacing:-1px">' + spark + '</span>']);
      }
      rows.push([t('Góc pha', 'Phase angle'), p.sunAngle.toFixed(0) + '°']);
      return rows;
    },
    onClick(info, c) {
      const p = c.params, rk = rockOf(p);
      const o = info.origin, d = info.dir;
      const f = rockBaseJS(rk.shape), Rt = U.uRot.value.clone().transpose();
      let t = 0, hit = false;
      for (let i = 0; i < 200; i++) {
        tmp.copy(o).addScaledVector(d, t).applyMatrix3(Rt);
        const h = f(tmp.x, tmp.y, tmp.z);
        if (h < 1e-3) { hit = true; break; }
        t += Math.max(h * 0.8, 1e-3); if (t > 200) break;
      }
      if (!hit) return;
      const r = tmp.length() * (mode === 'oumuamua' ? 100 : 250);
      const om = 2 * Math.PI / (p.period * 3600);
      const gl = rk.GM / (r * r);
      const hop = 0.05 * 0.05 / (2 * gl);   // height of a 5 cm/s hop
      if (mode === 'oumuamua') {
        c.toast(tt(`Cách tâm ${r.toFixed(0)} m. Trọng lực ~${(gl * 1e6).toFixed(1)} µm/s² — nhỏ hơn Trái Đất hàng triệu lần. ʻOumuamua bay qua Hệ Mặt Trời với vận tốc dư 26 km/s và sẽ không bao giờ quay lại.`,
          `${r.toFixed(0)} m from the center. Gravity ~${(gl * 1e6).toFixed(1)} µm/s² — millions of times weaker than on Earth. ʻOumuamua swept through the Solar System with a hyperbolic excess speed of 26 km/s and will never return.`), 7000);
      } else {
        c.toast(tt(`Cách tâm ${r.toFixed(0)} m · trọng lực ≈ ${(gl * 1e6).toFixed(0)} µm/s² (≈ ${(gl / 9.81 * 1e6).toFixed(0)} phần triệu g). Vận tốc thoát chỉ ${(st.vesc * 100).toFixed(0)} cm/s: một bước nhảy nhẹ (~1 m/s) đưa bạn vĩnh viễn rời khỏi ${rk.name.split(' ').pop()}; một cú nhón chân 5 cm/s bay cao ${hop.toFixed(0)} m và mất ${(2 * 0.05 / gl / 60).toFixed(0)} phút mới rơi xuống.`,
          `${r.toFixed(0)} m from the center · gravity ≈ ${(gl * 1e6).toFixed(0)} µm/s² (≈ ${(gl / 9.81 * 1e6).toFixed(0)} millionths of a g). The escape velocity is only ${(st.vesc * 100).toFixed(0)} cm/s: a gentle jump (~1 m/s) would send you away from ${rk.name.split(' ').pop()} forever; a 5 cm/s tiptoe hop rises ${hop.toFixed(0)} m and takes ${(2 * 0.05 / gl / 60).toFixed(0)} minutes to come back down.`), 9000);
      }
    },
  };
}

const asteroid = {
  id: 'asteroid',
  name: 'Tiểu hành tinh',
  category: 'small',
  order: 20,
  subtitle: 'Đống đá vụn (rubble pile) · Bennu, Itokawa',
  swatch: ['#8a8378', '#1a1816'],
  blurb: 'Không phải một khối đá liền mà là một <b>đống đá vụn</b> lỏng lẻo, giữ lại với nhau bằng lực hấp dẫn yếu ớt — vận tốc thoát chỉ ~20 cm/s, bạn có thể nhảy khỏi nó. Bề mặt Bennu tối như nhựa đường và phủ kín tảng đá; Itokawa có “biển” sỏi mịn giữa hai thùy.',
  intro: 'Nhấn <b>J</b> để phóng tàu <b>DART</b> vào tiểu hành tinh. Kéo <b>chu kỳ tự quay</b> xuống dưới ~3,7 giờ để thấy vì sao Bennu có hình con quay: ở xích đạo, lực ly tâm thắng trọng lực.',
  facts: [
    ['Bennu: đường kính', '490 m (565 × 535 × 508 m)'],
    ['Bennu: khối lượng', '7,33×10¹⁰ kg · mật độ 1,19 g/cm³ (rỗng ~50 %)'],
    ['Bennu: suất phản chiếu', '0,044 — tối như nhựa đường'],
    ['Itokawa', '535 × 294 × 209 m · suất phản chiếu 0,29 (loại S)'],
    ['OSIRIS-REx', 'lấy 121,6 g mẫu Bennu, về Trái Đất 24/9/2023'],
    ['DART → Dimorphos', '26/9/2022, chu kỳ quỹ đạo giảm 32–33 phút'],
  ],
  en: {
    name: 'Asteroid',
    subtitle: 'Rubble pile · Bennu, Itokawa',
    blurb: 'Not a solid rock but a loose <b>rubble pile</b> held together by feeble gravity — the escape velocity is only ~20 cm/s, so you could jump off it. Bennu\'s surface is as dark as asphalt and covered in boulders; Itokawa has a “sea” of fine gravel between its two lobes.',
    intro: 'Press <b>J</b> to launch the <b>DART</b> spacecraft at the asteroid. Drag the <b>rotation period</b> below ~3.7 h to see why Bennu is shaped like a spinning top: at the equator, centrifugal force beats gravity.',
    facts: [
      ['Bennu: diameter', '490 m (565 × 535 × 508 m)'],
      ['Bennu: mass', '7.33×10¹⁰ kg · density 1.19 g/cm³ (porosity ~50%)'],
      ['Bennu: albedo', '0.044 — as dark as asphalt'],
      ['Itokawa', '535 × 294 × 209 m · albedo 0.29 (S-type)'],
      ['OSIRIS-REx', 'collected 121.6 g of Bennu samples, returned to Earth on 24 September 2023'],
      ['DART → Dimorphos', '26 September 2022, orbital period shortened by 32–33 minutes'],
    ],
  },
  unitKm: 0.25,
  camera: { distance: 3.6, polar: 78, azimuth: 25, min: 1.25, max: 40, fov: 40 },
  exposure: 1.1,
  bloom: 0.3,
  params: [
    { key: 'shape', label: 'Hình dạng', type: 'seg', options: [['bennu', 'Bennu (con quay)'], ['itokawa', 'Itokawa (hạt lạc)']], default: 'bennu',
      en: { label: 'Shape', options: ['Bennu (spinning top)', 'Itokawa (peanut)'] },
      onChange: (v, ctx) => { ctx.params.period = v === 'bennu' ? 4.296 : 12.132; } },
    { key: 'period', label: 'Chu kỳ tự quay', type: 'range', min: 2.0, max: 20, step: 0.01, default: 4.296, fmt: v => v.toFixed(2) + ' h', note: 'Hiệu ứng YORP (ánh nắng tái bức xạ) làm tiểu hành tinh quay nhanh dần; quá giới hạn, đá trượt về xích đạo và văng ra.',
      en: { label: 'Rotation period', note: 'The YORP effect (re-radiated sunlight) gradually spins asteroids up; past the limit, rocks slide toward the equator and are flung off.' } },
    { key: 'tumble', label: 'Lộn nhào (quay không quanh trục chính)', type: 'range', min: 0, max: 1, step: 0.01, default: 0.35, note: '0 = quay đều quanh trục chính. Thực tế Bennu và Itokawa quay quanh trục chính; Toutatis và ʻOumuamua thì lộn nhào.',
      en: { label: 'Tumbling (non-principal-axis rotation)', note: '0 = steady rotation about the principal axis. In reality Bennu and Itokawa spin about their principal axes, while Toutatis and ʻOumuamua tumble.' } },
    { key: 'sunAngle', label: 'Góc pha (Mặt Trời)', type: 'range', min: 0, max: 165, step: 1, default: 55, fmt: v => v.toFixed(0) + '°', note: '0°: hiệu ứng đối pha — bề mặt sáng bừng khi mọi bóng đổ bị che khuất.',
      en: { label: 'Phase angle (Sun)', note: '0°: the opposition effect — the surface brightens sharply as every shadow hides behind the rock that casts it.' } },
  ],
  actions: [
    { label: 'DART', key: 'j', title: 'Tàu va chạm DART lao vào tiểu hành tinh', run: ctx => { const inst = ctx.scene._inst; if (inst) { inst.triggerDart(ctx); ctx.toast(t('DART đang lao tới với vận tốc 6,1 km/s… (chiếu chậm)', 'DART closing in at 6.1 km/s… (slow motion)'), 2500); } },
      en: { title: 'The DART impactor crashes into the asteroid' } },
  ],
  create(ctx) { const inst = makeRockInstance(ctx, 'asteroid'); asteroid._inst = inst; return inst; },
};

const interstellar = {
  id: 'interstellar-object',
  name: 'ʻOumuamua',
  category: 'small',
  order: 30,
  subtitle: 'Vật thể liên sao đầu tiên được phát hiện (1I/2017 U1)',
  swatch: ['#c98c6a', '#24160f'],
  blurb: 'Vị khách đến từ một hệ sao khác, bay qua Hệ Mặt Trời năm 2017 trên quỹ đạo hyperbol (e ≈ 1,20). Độ sáng của nó dao động tới ~10 lần mỗi 8 giờ: một vật thể cực kỳ dẹt hoặc thuôn dài đang <b>lộn nhào</b>.',
  intro: 'Theo dõi <b>đường cong ánh sáng</b> trong bảng thông tin: biên độ lớn là bằng chứng duy nhất về hình dạng của ʻOumuamua. Đổi giữa dạng <b>điếu xì gà</b> và <b>đĩa</b> — cả hai đều khớp với dữ liệu.',
  facts: [
    ['Phát hiện', '19/10/2017, Pan-STARRS (Haleakalā)'],
    ['Vận tốc dư (v∞)', '26,3 km/s'],
    ['Độ lệch tâm quỹ đạo', '1,20 (hyperbol)'],
    ['Biên độ quang biến', '~2,5 cấp sao (≈ 10×)'],
    ['Chu kỳ lộn nhào', '≈ 8,67 giờ (không quanh trục chính)'],
    ['Gia tốc phi hấp dẫn', '~5×10⁻⁶ m/s² ở 1 AU, không thấy đuôi khí'],
  ],
  en: {
    name: 'ʻOumuamua',
    subtitle: 'The first interstellar object ever detected (1I/2017 U1)',
    blurb: 'A visitor from another star system that swept through the Solar System in 2017 on a hyperbolic orbit (e ≈ 1.20). Its brightness varied by up to ~10 times every 8 hours: an extremely flat or elongated body <b>tumbling</b> through space.',
    intro: 'Watch the <b>light curve</b> in the info panel: its large amplitude is the only evidence we have of ʻOumuamua\'s shape. Switch between the <b>cigar</b> and <b>pancake</b> shapes — both fit the data.',
    facts: [
      ['Discovered', '19 October 2017, Pan-STARRS (Haleakalā)'],
      ['Hyperbolic excess speed (v∞)', '26.3 km/s'],
      ['Orbital eccentricity', '1.20 (hyperbolic)'],
      ['Light-curve amplitude', '~2.5 mag (≈ 10×)'],
      ['Tumbling period', '≈ 8.67 h (non-principal-axis)'],
      ['Non-gravitational acceleration', '~5×10⁻⁶ m/s² at 1 AU, no gas tail seen'],
    ],
  },
  unitKm: 0.1,
  camera: { distance: 5.2, polar: 75, azimuth: 30, min: 0.9, max: 60, fov: 40 },
  exposure: 1.2,
  bloom: 0.3,
  params: [
    { key: 'shape', label: 'Hình dạng', type: 'seg', options: [['cigar', 'Điếu xì gà'], ['pancake', 'Cái đĩa']], default: 'cigar', en: { label: 'Shape', options: ['Cigar', 'Pancake'] } },
    { key: 'period', label: 'Chu kỳ quay', type: 'range', min: 2, max: 20, step: 0.01, default: 8.67, fmt: v => v.toFixed(2) + ' h', en: { label: 'Rotation period' } },
    { key: 'tumble', label: 'Lộn nhào', type: 'range', min: 0, max: 1, step: 0.01, default: 0.6, en: { label: 'Tumbling' } },
    { key: 'sunAngle', label: 'Góc pha (Mặt Trời)', type: 'range', min: 0, max: 165, step: 1, default: 25, fmt: v => v.toFixed(0) + '°', note: 'Khi phát hiện, góc pha của ʻOumuamua khoảng 20–25°.',
      en: { label: 'Phase angle (Sun)', note: 'At discovery, ʻOumuamua\'s phase angle was about 20–25°.' } },
  ],
  actions: [],
  create(ctx) { return makeRockInstance(ctx, 'oumuamua'); },
};

export default [comet, asteroid, interstellar];
