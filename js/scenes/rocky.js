// Hành tinh đất đá — Earth, Mars, Venus and the lava world 55 Cancri e.
// One scene unit = one planetary radius. Sunlight is a directional light whose direction is set
// relative to the camera from a "local solar time under the camera" parameter (like the Moon's
// phase angle), so the requested lighting holds while you orbit. 55 Cnc e is tidally locked: its
// star is fixed in space and you orbit around to see the day and night hemispheres.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

const D2R = Math.PI / 180;
// body-frame unit vector for (latitude, east longitude) — must match the body rotation used below:
// body = rotY(uRot) · world, and the surface moves toward −azimuth (east), so east lon = −θ_body.
const ll = (lat, lon) => [-Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R)];
const v3 = a => `vec3(${a.map(x => x.toFixed(6)).join(', ')})`;

/* ------------------------------------------------------------------------------------------------
   Shared GLSL: body rotation, the Chapman function, phase functions, the single-scattering march
   ------------------------------------------------------------------------------------------------ */
const ROCKY_GLSL = /* glsl */`
mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }

// Chapman grazing-incidence function (Schüler 2012 form, with the √(πx/2) horizontal limit):
// column density, in units of H·ρ₀, from height h (in scale heights) to space along a direction of
// zenith cosine mu, for a planet of radius X = R/H scale heights. For mu < 0 the ray first dives to
// its tangent point (where the planet itself blocks it, the column becomes enormous).
float chapman(float X, float h, float mu){
  float c = sqrt(1.5708 * (X + h));
  if (mu >= 0.0) return c / (c * mu + 1.0) * exp(-h);
  float x0 = sqrt(max(1.0 - mu * mu, 0.0)) * (X + h);
  float c0 = sqrt(1.5708 * x0);
  return 2.0 * c0 * exp(min(X - x0, 60.0)) - c / (1.0 - c * mu) * exp(-h);
}
float phaseR(float mu){ return 0.0596831 * (1.0 + mu * mu); }                 // Rayleigh, 3/(16π)(1+μ²)
vec3 phaseCS(float mu, vec3 g){                                                // Cornette–Shanks (Mie)
  vec3 g2 = g * g;
  return 0.1193662 * (1.0 - g2) * (1.0 + mu * mu) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, vec3(1.5)));
}

// An exponential two-species atmosphere: molecules (Rayleigh, plus absorption bA on the same
// profile — ozone on Earth) and aerosols (Mie / dust). Coefficients are per scene unit.
struct Atm { float R; float Hr; float Hm; vec3 bR; vec3 bA; vec3 bMs; vec3 bMe; vec3 g; };

vec3 sunTrans(Atm A, float h, float mu){
  float oR = A.Hr * chapman(A.R / A.Hr, h / A.Hr, mu);
  float oM = A.Hm * chapman(A.R / A.Hm, h / A.Hm, mu);
  return exp(-((A.bR + A.bA) * oR + A.bMe * oM));
}
// Single scattering along ro + rd·[t0, t1]: 14 view samples, light optical depth from the
// analytic Chapman function (no inner light march). Returns radiance per unit solar irradiance and
// the view transmittance T.
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
// the Sun seen from a planet at d AU (angular radius 0.00465 rad / d), plus a little glare
vec3 sunDisk(vec3 rd, vec3 L, float dAU){
  float c = dot(rd, L), a = acos(clamp(c, -1.0, 1.0)), r = 0.00465 / dAU;
  return bb(5772.0) * (smoothstep(r * 1.15, r * 0.85, a) * 400.0 + pow(max(c, 0.0), 3000.0 * dAU * dAU) * 4.0);
}
// Worley noise that also returns a random id of the nearest cell: (F1, F2, id ∈ [0,1))
vec3 worleyId(vec3 p){
  vec3 i = floor(p), f = fract(p), id = vec3(0.0);
  float d1 = 8.0, d2 = 8.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(x, y, z);
    vec3 r = o + hash33(i + o) - f;
    float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; id = i + o; } else if (d < d2){ d2 = d; }
  }
  return vec3(sqrt(d1), sqrt(d2), hash13(id + 17.31));
}
// finite-difference normal of a height field on the unit sphere (in body coordinates)
void tangents(vec3 p, out vec3 t1, out vec3 t2){
  t1 = normalize(cross(p, abs(p.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  t2 = cross(p, t1);
}
`;

const HEAD = COMMON_GLSL + ROCKY_GLSL;

/* ================================================================================================
   EARTH
   ================================================================================================ */
const EARTH_FRAG = HEAD + /* glsl */`
uniform vec3 uSunDir, uAurN, uAurS;
uniform float uRot, uCloudRot, uClouds, uCities, uAtmo, uKp, uAurora;
const float RKM = 6371.0;
const float RTOP = 1.0 + 70.0 / RKM;
const float RCL = 1.0 + 8.0 / RKM;       // effective cloud-top shell, ~8 km
const float ESUN = 4.0;                  // solar irradiance in scene radiance units

Atm atmE(){
  Atm A;
  A.R = 1.0; A.Hr = 8.0 / RKM; A.Hm = 1.2 / RKM;
  // β in 10⁻⁶ m⁻¹ × 6.371·10⁶ m per scene unit
  A.bR = vec3(5.8, 13.5, 33.1) * 6.371 * uAtmo;
  // ozone (Chappuis band) — really a layer near 25 km; put on the molecular profile ×2 to match its column
  A.bA = vec3(0.65, 1.881, 0.085) * 6.371 * 2.0 * uAtmo;
  A.bMs = vec3(21.0 * 6.371) * uAtmo; A.bMe = A.bMs / 0.9; A.g = vec3(0.76);
  return A;
}

// ---- terrain -----------------------------------------------------------------------------
vec3 warpE(vec3 p){ return 0.3 * vec3(snoise(p * 1.3 + 11.0), snoise(p * 1.3 + 23.0), snoise(p * 1.3 + 37.0)); }
// elevation: > 0 land, < 0 sea (arbitrary units, ~1 ≈ 6 km)
float elevE(vec3 p, vec3 W){
  vec3 q = p + W;
  float hills = fbm(q * 13.0, 3);
  float c = fbm(q * 1.15 + 4.0, 5) - 0.075 + 0.045 * hills;
  c += 0.75 * smoothstep(-0.84, -0.94, p.y);             // Antarctica
  c -= 0.35 * smoothstep(0.86, 0.97, p.y);               // an Arctic ocean
  float r = 1.0 - abs(snoise(q * 4.2 + 2.0)); r *= r * r;
  float land = smoothstep(0.0, 0.2, c);
  return c + land * (0.3 * r * smoothstep(0.02, 0.3, c + 0.15 * hills) + 0.06 * hills);
}

// biome albedo from latitude (temperature) and a moisture field with a Hadley-cell desert belt
vec3 landAlbedo(vec3 p, float e, out float habit){
  float alat = abs(asin(clamp(p.y, -1.0, 1.0))) * 57.2958;
  float nz = fbm(p * 7.0 + 9.0, 3);
  float temp = 1.0 - alat / 80.0 - max(e - 0.32, 0.0) * 1.0 + nz * 0.06;
  float moist = 0.55 + 0.5 * fbm(p * 2.3 + 17.0, 3)
              - 0.65 * exp(-pow((alat - 23.0) / 8.0, 2.0))          // subtropical highs → deserts
              + 0.35 * exp(-pow(alat / 9.0, 2.0))                    // ITCZ rain belt → rainforest
              + 0.3 * (1.0 - smoothstep(0.0, 0.12, e));               // coasts are wetter
  vec3 desert = mix(vec3(0.42, 0.30, 0.17), vec3(0.40, 0.21, 0.09), smoothstep(-0.3, 0.4, nz));
  vec3 savanna = vec3(0.17, 0.14, 0.065), rain = vec3(0.018, 0.045, 0.016);
  vec3 steppe = vec3(0.19, 0.16, 0.10), grass = vec3(0.08, 0.10, 0.04), forest = vec3(0.03, 0.06, 0.022);
  vec3 tundra = vec3(0.10, 0.09, 0.07), boreal = vec3(0.022, 0.038, 0.024), rock = vec3(0.13, 0.115, 0.10);
  vec3 warm = mix(desert, savanna, smoothstep(0.15, 0.35, moist));
  warm = mix(warm, rain, smoothstep(0.5, 0.75, moist));
  vec3 temperate = mix(mix(steppe, grass, smoothstep(0.2, 0.42, moist)), forest, smoothstep(0.45, 0.65, moist));
  vec3 cold = mix(tundra, boreal, smoothstep(0.3, 0.5, moist));
  vec3 c = mix(cold, temperate, smoothstep(0.3, 0.42, temp));
  c = mix(c, warm, smoothstep(0.56, 0.68, temp));
  c = mix(c, tundra, smoothstep(0.2, 0.1, temp));
  c = mix(c, rock, smoothstep(0.2, 0.45, e) * 0.7);
  float snow = max(smoothstep(0.08, 0.0, temp), smoothstep(-0.82, -0.88, p.y));
  c = mix(c, vec3(0.8, 0.83, 0.88), snow);
  c *= 0.85 + 0.3 * fbm(p * 40.0, 2);
  habit = smoothstep(0.25, 0.45, temp) * smoothstep(0.08, 0.3, moist) * (1.0 - snow) * (1.0 - smoothstep(0.25, 0.5, e));
  return c;
}

// night lights: population follows climate and coasts, clustered into cities (Worley cells)
vec3 cityLights(vec3 p, float e, float habit){
  float pop = habit * smoothstep(0.0, 0.45, fbm(p * 3.2 + 31.0, 3));
  pop *= 0.35 + 1.0 * (1.0 - smoothstep(0.0, 0.08, e));                // coasts and lowlands
  if (pop < 0.01) return vec3(0.0);
  vec3 w = worleyId(p * 70.0);
  float big = step(1.0 - pop * 0.9, w.z);                                // larger cities where people are
  float city = exp(-w.x * w.x * 40.0) * big * (0.5 + 1.5 * fract(w.z * 13.7));
  vec3 w2 = worleyId(p * 260.0);
  float town = exp(-w2.x * w2.x * 30.0) * step(1.0 - pop * 0.9, w2.z);
  float roads = (1.0 - smoothstep(0.0, 0.035, w.y - w.x)) * smoothstep(0.3, 0.8, pop) * 0.12;
  float l = 2.0 * city + 0.7 * town + roads;
  return mix(bb(2100.0), bb(3600.0), smoothstep(0.5, 0.95, fract(w.z * 7.13))) * l * 0.6;
}

// ---- clouds ------------------------------------------------------------------------------------
// extratropical cyclones: twist the sample domain around nearby (hashed) storm centres
vec3 cyclones(vec3 p){
  vec3 g = p * 2.6, i = floor(g), f = fract(g);
  float best = 9.0, second = 9.0, hid = 0.0; vec3 bc = vec3(0.0, 1.0, 0.0);
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(x, y, z), c = o + hash33(i + o + 71.0);
    float d = length(c - f);
    if (d < best){ second = best; best = d; bc = (i + c) / 2.6; hid = hash13(i + o + 5.0); }
    else if (d < second) second = d;
  }
  vec3 ax = normalize(bc);
  float s = sign(ax.y) * smoothstep(0.35, 0.6, abs(ax.y)) * step(0.45, hid);   // mid-latitude storm tracks
  // the twist fades to zero at the cell borders so the warped domain stays continuous
  float ang = s * 4.0 * exp(-best * best * 7.0) * smoothstep(0.0, 0.35, second - best);
  float ca = cos(ang), sa = sin(ang);
  return p * ca + cross(ax, p) * sa + ax * dot(ax, p) * (1.0 - ca);
}
float cloudCov(vec3 pw, int oct){
  vec3 p = rotY(uCloudRot) * normalize(pw);
  float lat = p.y;
  vec3 q = cyclones(p);
  float t = uTime * 0.0015;
  vec3 w = vec3(snoise(q * 1.7 + vec3(t, 3.0, 0.0)), snoise(q * 1.7 + vec3(7.0, t, 1.0)), snoise(q * 1.7 + vec3(13.0, 0.0, t)));
  q = q * vec3(1.0, 1.3, 1.0);                                      // zonal stretching by the jets
  float n = fbm(q * 2.4 + w * 0.45 + vec3(0.0, 0.0, t), oct) * 0.5 + 0.5;
  // convective detail (cumulus fields, ragged edges) only for the directly viewed layer
  if (oct >= 5) n += 0.07 * fbm(q * 34.0 + w * 3.0, 2) + 0.04 * (abs(snoise(q * 90.0)) - 0.4);
  // general circulation: ITCZ (cloudy), subtropical highs (clear), storm tracks (cloudy)
  float band = 0.09 * exp(-pow(lat / 0.1, 2.0)) - 0.13 * exp(-pow((abs(lat) - 0.42) / 0.12, 2.0))
             + 0.08 * exp(-pow((abs(lat) - 0.8) / 0.14, 2.0));
  float thr = 1.0 - uClouds;
  return smoothstep(thr - 0.01, thr + 0.14, n + band) * (0.6 + 0.4 * smoothstep(0.4, 0.75, n));
}

// ---- aurora: oxygen 557.7 nm (green, ~100–150 km) and 630 nm (red, 200–300 km), N₂⁺ 427.8 nm fringe
vec3 aurora(vec3 ro, vec3 rd, float tEnd){
  const float R1 = 1.0 + 85.0 / RKM, R2 = 1.0 + 330.0 / RKM;
  vec2 h = sphereHit(ro, rd, vec3(0.0), R2);
  if (h.x > h.y || h.y < 0.0) return vec3(0.0);
  float t0 = max(h.x, 0.0), t1 = min(h.y, tEnd);
  vec2 hi = sphereHit(ro, rd, vec3(0.0), R1);
  if (hi.x < hi.y && hi.x > t0) t1 = min(t1, hi.x);        // only the near part of the shell
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
    // a bright, thin arc at the equatorward edge over a diffuse poleward glow
    float ring = (exp(-x * x) * 0.2 + exp(-pow((x + 0.55) * 6.0, 2.0)) + 0.5 * exp(-pow((x - 0.35) * 7.0, 2.0))) * (0.4 + 0.6 * smoothstep(-0.2, 0.6, snoise(d * 5.0 + 20.0 + uWall * 0.02)));
    float rays = 0.2 + 0.8 * pow(0.5 + 0.5 * snoise(d * 160.0 + vec3(uWall * 0.3, 0.0, 0.0)), 3.0);
    float hk = (r - 1.0) * RKM;
    ring *= 0.1 + 0.9 * smoothstep(0.15, -0.25, dot(d, uSunDir));   // dayside (cusp) aurora is far weaker
    float green = smoothstep(90.0, 104.0, hk) * exp(-max(hk - 104.0, 0.0) / 28.0);
    float red = exp(-pow((hk - 230.0) / 70.0, 2.0)) * 0.22;
    float blue = exp(-pow((hk - 96.0) / 5.0, 2.0)) * 0.25 * smoothstep(5.0, 8.0, uKp);
    acc += ring * rays * (vec3(0.12, 1.0, 0.32) * green + vec3(1.0, 0.08, 0.12) * red * (0.5 + 0.12 * uKp) + vec3(0.5, 0.25, 1.0) * blue) * dt;
  }
  return acc * 16.0 * uAurora;
}

vec3 cloudShade(vec3 pc, float a, vec3 L, Atm A){
  vec3 nc = normalize(pc);
  float mu0 = dot(nc, L);
  vec3 Ts = sunTrans(A, RCL - 1.0, mu0);
  float ahead = cloudCov(pc + L * 0.006, 2);                 // cloud mass toward the sun: self-shadow
  float self = 1.0 - 0.5 * clamp(ahead - 0.6 * a, 0.0, 1.0);
  float alb = mix(0.45, 0.8, a);
  float lit = max(mu0 + 0.06, 0.0) / 1.06 * self;     // tall cloud tops still catch the sun after ground sunset
  vec3 skyAmb = vec3(0.05, 0.08, 0.14) * smoothstep(-0.1, 0.3, mu0) * uAtmo;
  return alb / PI * ESUN * (Ts * lit + skyAmb);
}

vec3 earthSurface(vec3 pw, vec3 rd, vec3 L, Atm A){
  vec3 n0 = normalize(pw);
  mat3 R = rotY(uRot);
  vec3 p = R * n0;
  vec3 W = warpE(p);
  float e = elevE(p, W);
  float mu0s = dot(n0, L);
  vec3 Ts = sunTrans(A, 0.0, mu0s);
  float shadow = 1.0;
  if (uClouds > 0.001 && mu0s > -0.1){
    float b = dot(pw, L);
    float tc = -b + sqrt(max(b * b - dot(pw, pw) + RCL * RCL, 0.0));
    shadow = 1.0 - 0.75 * cloudCov(pw + L * tc, 3);
  }
  vec3 skyAmb = vec3(0.05, 0.085, 0.16) * smoothstep(-0.15, 0.4, mu0s) * uAtmo;
  vec3 col;
  float alat = abs(p.y);
  if (e <= 0.0){
    float shelf = smoothstep(-0.08, 0.0, e);
    vec3 alb = mix(vec3(0.004, 0.011, 0.03), vec3(0.012, 0.05, 0.065), shelf);
    float ice = smoothstep(0.95, 0.975, alat + 0.02 * fbm(p * 9.0, 3)) + smoothstep(-0.9, -0.93, p.y);
    ice = clamp(ice, 0.0, 1.0);
    alb = mix(alb, vec3(0.6, 0.65, 0.7), ice);
    col = alb / PI * ESUN * (Ts * max(mu0s, 0.0) * shadow + skyAmb);
    // sun glint: GGX microfacet ocean (wind-roughened, α ≈ 0.12) with Schlick Fresnel, F0 = 0.02
    vec3 V = -rd, H = normalize(L + V);
    float nh = max(dot(n0, H), 0.0), nv = max(dot(n0, V), 1e-3), nl = max(mu0s, 0.0);
    float a2 = 0.0144;
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float F = 0.02 + 0.98 * pow(1.0 - max(dot(V, H), 0.0), 5.0);
    float k = 0.06;
    float G = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
    col += (1.0 - ice) * D * F * G / (4.0 * nv) * ESUN * Ts * shadow;
  } else {
    vec3 t1, t2; tangents(p, t1, t2);
    float eps = 0.0015, kb = 0.02;
    float hx = max(elevE(normalize(p + t1 * eps), W), 0.0), hy = max(elevE(normalize(p + t2 * eps), W), 0.0);
    vec3 nb = normalize(p - (t1 * (hx - e) + t2 * (hy - e)) * kb / eps);
    vec3 n = transpose(R) * nb;
    float habit;
    vec3 alb = landAlbedo(p, e, habit);
    float mu0 = max(dot(n, L), 0.0) * smoothstep(-0.05, 0.05, mu0s);
    col = alb / PI * ESUN * (Ts * mu0 * shadow + skyAmb);
    float night = smoothstep(0.03, -0.12, mu0s);
    if (uCities > 0.5 && night > 0.0) col += cityLights(p, e, habit) * night;
  }
  return col;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  Atm A = atmE();
  vec3 col = sky(rd, 1.0) * 0.6;
  vec2 hg = sphereHit(ro, rd, vec3(0.0), 1.0);
  bool ground = hg.x < hg.y && hg.x > 0.0;
  if (!ground) col += sunDisk(rd, L, 1.0);
  vec2 ha = sphereHit(ro, rd, vec3(0.0), RTOP);
  if (ha.x < ha.y && ha.y > 0.0){
    float t0 = max(ha.x, 0.0), t1 = ground ? hg.x : ha.y;
    vec3 bg = ground ? earthSurface(ro + rd * hg.x, rd, L, A) : col;
    if (uClouds > 0.001){
      vec2 hc = sphereHit(ro, rd, vec3(0.0), RCL);
      if (hc.x < hc.y && hc.x > 0.0){
        vec3 pc = ro + rd * hc.x;
        float a = cloudCov(pc, 5);
        if (a > 0.002) bg = mix(bg, cloudShade(pc, a, L, A), a);
      }
    }
    vec3 T;
    vec3 S = atmosphere(ro, rd, t0, t1, L, A, T) * ESUN;
    col = S + T * bg;
  }
  if (uAurora > 0.0) col += aurora(ro, rd, ground ? hg.x : 1e9);
  gl_FragColor = vec4(col, 1.0);
}`;

/* ================================================================================================
   MARS
   ================================================================================================ */
const MARS_SITES = {
  oly: ll(18.65, -133.8), asc: ll(11.8, -104.5), pav: ll(0.8, -113.4), ars: ll(-8.3, -120.1),
  alba: ll(40.5, -109.6), tharsis: ll(2.0, -108.0), elysium: ll(25.0, 147.0),
  vmA: ll(-6.5, -95.0), vmB: ll(-10.5, -42.0), hellas: ll(-42.4, 70.5), argyre: ll(-49.7, -43.4),
  capS: ll(-87.0, -45.0), syrtis: ll(8.4, 69.5),
};
const MARS_FRAG = HEAD + /* glsl */`
uniform vec3 uSunDir;
uniform float uRot, uTau, uHd, uAtmo, uCaps;
const float RKM = 3389.5;
const float RTOP = 1.0 + 150.0 / RKM;
const float ESUN = 4.0;
const vec3 OLY = ${v3(MARS_SITES.oly)}, ASC = ${v3(MARS_SITES.asc)}, PAV = ${v3(MARS_SITES.pav)}, ARS = ${v3(MARS_SITES.ars)};
const vec3 ALBA = ${v3(MARS_SITES.alba)}, THAR = ${v3(MARS_SITES.tharsis)}, ELYS = ${v3(MARS_SITES.elysium)};
const vec3 VMA = ${v3(MARS_SITES.vmA)}, VMB = ${v3(MARS_SITES.vmB)}, HELLAS = ${v3(MARS_SITES.hellas)}, ARGYRE = ${v3(MARS_SITES.argyre)};
const vec3 CAPS = ${v3(MARS_SITES.capS)}, SYRTIS = ${v3(MARS_SITES.syrtis)};
const vec3 DUST_ALB = vec3(0.50, 0.27, 0.12);     // multiply-scattering albedo of a thick dust veil

Atm atmM(){
  Atm A;
  A.R = 1.0; A.Hr = 11.1 / RKM; A.Hm = uHd;
  // CO₂ Rayleigh: Earth's coefficients × (ρ ratio 0.016) × (CO₂ cross-section ≈ 2.4× air)
  A.bR = vec3(5.8, 13.5, 33.1) * 0.039 * 3.3895 * uAtmo;
  A.bA = vec3(0.0);
  // dust: vertical optical depth uTau; single-scattering albedo falls toward the blue (ferric oxide)
  A.bMe = vec3(uTau / uHd) * uAtmo;
  A.bMs = A.bMe * vec3(0.97, 0.88, 0.70);
  // ~1.5 µm grains: the size parameter is larger in the blue, so blue light is scattered more
  // strongly forward — the reason Martian sunsets are blue
  A.g = vec3(0.58, 0.68, 0.84);
  return A;
}

float ang(vec3 a, vec3 b){ return acos(clamp(dot(a, b), -1.0, 1.0)); }
float shield(vec3 p, vec3 c, float R, float H){
  float d = ang(p, c) / R;
  if (d > 1.15) return 0.0;
  float h = H * pow(clamp(1.0 - d, 0.0, 1.0), 1.5);
  h -= 0.18 * H * exp(-pow(d / 0.1, 2.0));                  // summit caldera
  return h;
}
float craters(vec3 p, float seed){
  vec3 i = floor(p), f = fract(p); float h = 0.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(x, y, z), r = hash33(i + o + seed);
    if (r.z > 0.4) continue;
    float R = 0.15 + 0.3 * hash13(i + o + seed + 7.1);
    float d = length(o + r - f) / R;
    if (d < 1.8) h += R * ((d < 1.0 ? (d * d - 1.0) * 0.6 : 0.0) + 0.2 * exp(-pow((d - 1.0) * 4.0, 2.0)));
  }
  return h;
}
// Valles Marineris: a 4000 km trough system along a great-circle arc east of Tharsis
float canyon(vec3 p){
  vec3 nrm = normalize(cross(VMA, VMB));
  float tot = ang(VMA, VMB);
  vec3 bx = normalize(cross(nrm, VMA));
  float s = atan(dot(p, bx), dot(p, VMA)) / tot;
  if (s < -0.12 || s > 1.12) return 0.0;
  float off = asin(clamp(dot(p, nrm), -1.0, 1.0)) + 0.01 * snoise(p * 14.0) + 0.003 * snoise(p * 45.0);
  float w = 0.018 + 0.035 * smoothstep(0.1, 0.45, s) * smoothstep(0.95, 0.6, s);
  float trough = smoothstep(w, w * 0.45, abs(off));
  float para = smoothstep(w * 0.6, w * 0.25, abs(off - 0.045)) * smoothstep(0.2, 0.35, s) * smoothstep(0.75, 0.55, s);
  float ends = smoothstep(-0.1, 0.06, s) * smoothstep(1.1, 0.9, s);
  // Noctis Labyrinthus: a maze of troughs at the west end
  return max(trough, para) * ends;
}
// elevation in km
float elevM(vec3 p, out float low){
  float n = fbm(p * 1.5 + 2.0, 4);
  low = smoothstep(-0.05, 0.25, p.y * 0.9 + 0.22 * n + 0.12 - 0.15 * dot(p, THAR));  // crustal dichotomy
  float e = 1.5 + 2.0 * n - 5.5 * low;
  e += 7.0 * exp(-pow(ang(p, THAR) / 0.5, 2.0));                      // Tharsis bulge
  e += 3.0 * exp(-pow(ang(p, ELYS) / 0.25, 2.0));
  e += shield(p, OLY, 0.095, 21.0) + shield(p, ASC, 0.06, 15.0) + shield(p, PAV, 0.055, 11.0) + shield(p, ARS, 0.065, 13.0);
  e += shield(p, ALBA, 0.25, 4.0) + shield(p, ELYS, 0.04, 9.0);
  e -= 7.0 * canyon(p);
  e -= 8.0 * smoothstep(0.36, 0.12, ang(p, HELLAS)) - 1.5 * exp(-pow((ang(p, HELLAS) - 0.4) / 0.06, 2.0));
  e -= 4.5 * smoothstep(0.2, 0.07, ang(p, ARGYRE));
  float hi = 1.0 - low;
  e += 1.6 * craters(p * 4.0, 0.0) * (0.3 + 0.7 * hi) + 0.6 * craters(p * 11.0, 13.0) * (0.2 + 0.8 * hi);
  e += 0.25 * fbm(p * 20.0, 2);
  return e;
}

vec3 marsAlbedo(vec3 p, float e, float low, float cany){
  float a = fbm(p * 2.1 + 7.0, 5);
  float dark = smoothstep(0.04, 0.2, a + 0.04 * fbm(p * 12.0, 2)) * (1.0 - 0.4 * low);
  dark = max(dark, 0.9 * exp(-pow(ang(p, SYRTIS) / 0.16, 2.0)) * smoothstep(-0.3, 0.2, a + 0.3));
  vec3 bright = vec3(0.50, 0.235, 0.095), mid = vec3(0.36, 0.16, 0.068), basalt = vec3(0.085, 0.058, 0.045);
  vec3 c = mix(mid, bright, smoothstep(0.0, -0.25, a));
  c = mix(c, bright, 0.8 * exp(-pow(ang(p, THAR) / 0.45, 2.0)));          // dust-mantled Tharsis
  c = mix(c, basalt, dark * 0.85 * (1.0 - 0.7 * exp(-pow(ang(p, THAR) / 0.4, 2.0))));
  c = mix(c, vec3(0.15, 0.085, 0.06), cany * 0.75);                       // dark canyon floors, layered walls
  c = mix(c, vec3(0.5, 0.33, 0.24), smoothstep(0.25, 0.1, ang(p, HELLAS)) * 0.5);   // frosty Hellas floor
  c *= 0.92 + 0.16 * fbm(p * 18.0, 2);
  // polar caps: the north residual water-ice cap with its spiral troughs, the small offset south CO₂ cap
  float colN = acos(clamp(p.y, -1.0, 1.0));
  float phi = atan(p.z, p.x);
  float capR = 0.17 * uCaps;
  float capN = smoothstep(capR, capR - 0.03, colN + 0.02 * fbm(p * 14.0, 3));
  float trough = smoothstep(0.55, 0.95, sin(phi * 2.0 - colN * 34.0 + 2.0 * snoise(p * 6.0)));
  vec3 ice = mix(vec3(0.78, 0.74, 0.68), vec3(0.32, 0.2, 0.13), trough * 0.6 * smoothstep(0.02, 0.1, colN));
  c = mix(c, ice, capN);
  float capS = smoothstep(0.1 * uCaps + 0.01, 0.07 * uCaps, ang(p, CAPS) + 0.015 * fbm(p * 16.0, 3));
  c = mix(c, vec3(0.8, 0.78, 0.76), capS);
  return c;
}

vec3 marsSurface(vec3 pw, vec3 L, Atm A){
  vec3 n0 = normalize(pw);
  mat3 R = rotY(uRot);
  vec3 p = R * n0;
  float low, lx, ly;
  float e = elevM(p, low);
  vec3 t1, t2; tangents(p, t1, t2);
  float eps = 0.0016, kb = 3.5 / RKM;                     // relief exaggerated ×3.5
  float hx = elevM(normalize(p + t1 * eps), lx), hy = elevM(normalize(p + t2 * eps), ly);
  vec3 nb = normalize(p - (t1 * (hx - e) + t2 * (hy - e)) * kb / eps);
  vec3 n = transpose(R) * nb;
  vec3 alb = marsAlbedo(p, e, low, canyon(p));
  float mu0s = dot(n0, L);
  vec3 Ts = sunTrans(A, 0.0, mu0s);
  float mu0 = max(dot(n, L), 0.0) * smoothstep(-0.05, 0.05, mu0s);
  vec3 skyAmb = vec3(0.10, 0.06, 0.035) * smoothstep(-0.1, 0.4, mu0s) * (1.0 - exp(-uTau));
  return alb / PI * ESUN * (Ts * mu0 + skyAmb);
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  Atm A = atmM();
  vec3 col = sky(rd, 1.0) * 0.6;
  vec2 hg = sphereHit(ro, rd, vec3(0.0), 1.0);
  bool ground = hg.x < hg.y && hg.x > 0.0;
  if (!ground) col += sunDisk(rd, L, 1.524);
  vec2 ha = sphereHit(ro, rd, vec3(0.0), RTOP);
  if (ha.x < ha.y && ha.y > 0.0){
    float t0 = max(ha.x, 0.0), t1 = ground ? hg.x : ha.y;
    vec3 bg = ground ? marsSurface(ro + rd * hg.x, L, A) : col;
    vec3 T;
    vec3 S = atmosphere(ro, rd, t0, t1, L, A, T) * ESUN;
    // multiple scattering in the dust (single scattering alone would leave a thick storm black):
    // a Lambertian veil lit at the reference point, weighted by the fraction of light the dust removed
    float tc = ground ? hg.x : max(-dot(ro, rd), 0.0);
    vec3 pr = normalize(ro + rd * tc);
    float muR = dot(pr, L);
    float lit = max(muR, 0.0) * 0.8 + 0.2 * smoothstep(-0.15, 0.25, muR) * max(muR + 0.15, 0.0);
    float ext = 1.0 - dot(T, vec3(0.3333));
    vec3 ms = DUST_ALB / PI * ESUN * lit * ext * (1.0 - exp(-0.4 * uTau)) * uAtmo;
    col = S + T * bg + ms;
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ================================================================================================
   VENUS
   ================================================================================================ */
const VENUS_FRAG = HEAD + /* glsl */`
uniform vec3 uSunDir;
uniform float uView, uCloudRot, uSurfRot, uAtmo;
const float RKM = 6051.8;
const float RC = 1.0 + 70.0 / RKM;        // cloud tops
const float RTOP = RC + 45.0 / RKM;
const float ESUN = 4.0;

Atm atmV(){
  Atm A;
  A.R = RC; A.Hr = 5.0 / RKM; A.Hm = 4.0 / RKM;
  // CO₂ above the cloud tops (~0.03 bar at 70 km), Rayleigh ≈ Earth's × 0.07 per unit volume at RC
  A.bR = vec3(5.8, 13.5, 33.1) * 0.07 * 6.0518 * uAtmo;
  A.bA = vec3(0.0);
  // upper haze of submicron H₂SO₄ droplets, vertical τ ≈ 0.15 (stronger in the UV)
  float tau = uView > 0.5 ? 0.3 : 0.15;
  A.bMe = vec3(tau / A.Hm) * uAtmo;
  A.bMs = A.bMe * vec3(1.0, 0.99, 0.97);
  A.g = vec3(0.7);
  return A;
}

float wrapPi(float a){ return a - TAU * floor((a + PI) / TAU); }
// UV albedo of the cloud tops: the unknown UV absorber forms the dark horizontal "Y" (a planetary
// Kelvin/Rossby wave pattern) carried around the planet every ~4 days by the super-rotating winds
float uvAlbedo(vec3 p){
  float lat = asin(clamp(p.y, -1.0, 1.0)), lon = atan(p.x, p.z);
  float lonS = lon + 0.75 * lat * lat;                       // mid-latitude jets bend streaks poleward
  float dl = wrapPi(lonS - 0.4);
  float stem = exp(-pow(lat / 0.24, 2.0)) * smoothstep(-2.5, -1.3, dl) * smoothstep(0.35, -0.05, dl);
  float armLat = 0.6 * pow(max(dl, 0.0), 0.8);
  float arms = exp(-pow((abs(lat) - armLat) / (0.17 + 0.1 * dl), 2.0)) * smoothstep(-0.15, 0.15, dl) * smoothstep(2.3, 1.2, dl);
  vec3 q = vec3(cos(lonS) * 1.3, lat * 6.0, sin(lonS) * 1.3);
  float streak = fbm(q * 1.7 + vec3(0.0, 0.0, 0.0), 4);
  float cells = fbm(p * 9.0, 3);
  float Y = max(stem, arms) * (0.8 + 0.2 * smoothstep(-0.4, 0.4, streak));
  float a = 0.64 - 0.42 * Y - 0.09 * smoothstep(-0.1, 0.5, streak) + 0.05 * cells * (1.0 - smoothstep(0.4, 0.9, abs(lat)));
  a += 0.16 * exp(-pow((abs(lat) - 1.08) / 0.14, 2.0));          // bright polar collars
  a -= 0.10 * smoothstep(1.2, 1.45, abs(lat));                    // darker polar vortex
  return clamp(a, 0.0, 1.0);
}

// ---- Magellan radar view of the surface -----------------------------------------------------
float elevV(vec3 p, out float rough){
  float big = fbm(p * 0.9 + 3.0, 4);
  float hl = smoothstep(0.28, 0.45, big);                         // Ishtar/Aphrodite-like highlands (~10 %)
  float r = 1.0 - abs(snoise(p * 16.0 + vec3(0.0, 0.0, big * 4.0))); r = r * r * r;
  float r2 = 1.0 - abs(snoise(p * 41.0 + 5.0)); r2 *= r2;
  float tess = (0.6 * r + 0.4 * r2) * hl;                         // tesserae: deformed, rough terrain
  vec3 w = worleyId(p * 2.6);
  float rr = 0.26 + 0.1 * w.z + 0.03 * snoise(p * 12.0);
  float corona = exp(-pow((w.x - rr) / 0.035, 2.0)) * step(0.8, w.z) * (1.0 - hl) * (0.5 + 0.5 * fbm(p * 20.0, 2));
  float flows = smoothstep(0.15, 0.55, fbm(p * 4.0 + 9.0, 5)) * (1.0 - hl);   // young, radar-bright lava flows
  float lines = pow(1.0 - abs(snoise(p * 22.0 + 40.0)), 14.0) * 0.25 * (1.0 - hl);   // graben / wrinkle ridges
  rough = 0.45 + 1.6 * tess + 0.9 * flows + 1.2 * corona + lines + 0.25 * hl;
  return 0.55 * hl + 0.15 * big + 0.05 * tess + 0.05 * corona + 0.012 * fbm(p * 30.0, 2);
}
// Muhleman backscatter law σ₀(θ), the empirical fit to Venus radar echoes
float muhleman(float c){ float s = sqrt(max(1.0 - c * c, 0.0)); return 0.0118 * c / pow(s + 0.111 * c, 3.0); }

vec3 radarView(vec3 pw, vec3 rd){
  vec3 n0 = normalize(pw);
  mat3 R = rotY(uSurfRot);
  vec3 p = R * n0;
  float rough, rx, ry;
  float e = elevV(p, rough);
  vec3 t1, t2; tangents(p, t1, t2);
  float eps = 0.0015, kb = 0.012;
  float hx = elevV(normalize(p + t1 * eps), rx), hy = elevV(normalize(p + t2 * eps), ry);
  vec3 n = transpose(R) * normalize(p - (t1 * (hx - e) + t2 * (hy - e)) * kb / eps);
  // the radar is its own light source (from the spacecraft); like the Magellan mosaics we
  // normalise by the backscatter of a smooth sphere so only slopes and roughness remain
  float c = max(dot(n, -rd), 0.02), c0 = max(dot(n0, -rd), 0.02);
  float s = muhleman(c) / muhleman(c0);
  float b = 0.2 * pow(s, 0.6) * rough * (0.85 + 0.3 * fbm(p * 70.0, 2));
  return vec3(1.0, 0.6, 0.24) * b;
}

vec3 deck(vec3 pc, vec3 rd, vec3 L, Atm A){
  vec3 n = normalize(pc);
  vec3 p = rotY(uCloudRot) * n;
  float mu0 = dot(n, L), mu = max(dot(n, -rd), 1e-3);
  float uv = uvAlbedo(p);
  vec3 alb;
  if (uView > 0.5) alb = vec3(0.78, 0.8, 1.0) * uv;
  else alb = vec3(0.92, 0.82, 0.58) * mix(1.0, uv / 0.62, 0.05) * (0.98 + 0.04 * fbm(p * 6.0, 2));
  // a deep, conservative scattering cloud: half Lommel–Seeliger (the flat look), half Lambert
  float m0 = max(mu0, 0.0);
  float f = 0.5 * 2.0 * m0 / (m0 + mu) + 0.5 * m0;
  vec3 Ts = sunTrans(A, 0.0, mu0);
  return alb / PI * ESUN * f * Ts;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  vec3 col = sky(rd, 1.0) * 0.6;
  if (uView > 1.5){
    vec2 hs = sphereHit(ro, rd, vec3(0.0), 1.0);
    if (hs.x < hs.y && hs.x > 0.0) col = radarView(ro + rd * hs.x, rd);
    gl_FragColor = vec4(col, 1.0);
    return;
  }
  Atm A = atmV();
  vec2 hg = sphereHit(ro, rd, vec3(0.0), RC);
  bool ground = hg.x < hg.y && hg.x > 0.0;
  if (!ground) col += sunDisk(rd, L, 0.723);
  vec2 ha = sphereHit(ro, rd, vec3(0.0), RTOP);
  if (ha.x < ha.y && ha.y > 0.0){
    float t0 = max(ha.x, 0.0), t1 = ground ? hg.x : ha.y;
    vec3 bg = ground ? deck(ro + rd * hg.x, rd, L, A) : col;
    vec3 T;
    vec3 S = atmosphere(ro, rd, t0, t1, L, A, T) * ESUN;
    if (uView > 0.5) S *= vec3(0.9, 0.95, 1.1);
    col = S + T * bg;
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ================================================================================================
   LAVA WORLD — 55 Cancri e
   ================================================================================================ */
const STAR_DIR = new THREE.Vector3(0.93, 0.16, 0.33).normalize();
const LAVA_FRAG = HEAD + /* glsl */`
uniform vec3 uStar;
uniform float uTss, uCrust, uView;
// 55 Cnc A: K0 dwarf, 0.943 R☉, 5196 K, 0.01544 AU away = 193 planetary radii; its disk spans ~32°
const float STAR_D = 193.0, STAR_R = 54.7, TSTAR = 5196.0;
const float SINA = 0.2834;                  // sine of the star's angular radius seen from the planet
const float ESTAR = 4.0;

// starlight on a surface with zenith cosine mu, for a star of finite angular size (smooth horizon)
float irr(float mu){
  if (mu > SINA) return mu;
  if (mu < -SINA) return 0.0;
  return (mu + SINA) * (mu + SINA) / (4.0 * SINA);
}
// radiative-equilibrium temperature with no heat redistribution, floored by the observed night side
float Teq(vec3 n){ return max(uTss * pow(irr(dot(n, uStar)), 0.25), 0.5 * uTss); }
// visible glow of a blackbody: Planck's law at 555 nm relative to 2700 K, compressed with a 0.55
// power (the eye / a camera adapts — the true contrast between 1400 K and 2700 K is ~10⁴)
vec3 glow(float T){
  float rel = (exp(25940.0 / 2700.0) - 1.0) / (exp(min(25940.0 / T, 80.0)) - 1.0);
  return bb(T) * 0.9 * pow(rel, 0.62);
}
vec3 inferno(float x){
  x = clamp(x, 0.0, 1.0);
  vec3 a = vec3(0.0, 0.0, 0.02), b = vec3(0.34, 0.06, 0.43), c = vec3(0.87, 0.27, 0.16), d = vec3(0.99, 0.75, 0.15), e = vec3(0.98, 1.0, 0.65);
  vec3 col = x < 0.25 ? mix(a, b, x / 0.25) : x < 0.5 ? mix(b, c, (x - 0.25) / 0.25) : x < 0.75 ? mix(c, d, (x - 0.5) / 0.25) : mix(d, e, (x - 0.75) / 0.25);
  return col * col;
}

// crust: Worley plates whose coverage grows as the local temperature drops below the solidus
void lavaField(vec3 p, out float T, out float h, out float crustF){
  float te = Teq(p);
  float t = uTime * 0.002;
  vec3 q = p * 5.5 + 0.35 * vec3(snoise(p * 2.0 + t), snoise(p * 2.0 + 7.0 - t), snoise(p * 2.0 + 13.0));
  vec3 w = worleyId(q);
  float edge = w.y - w.x;
  float cover = smoothstep(2650.0, 1700.0, te - 700.0 * (uCrust - 0.5));   // how much of the surface has frozen
  // convecting magma: hot upwellings and cooler, darker skin dragged into streaks
  float conv = fbm(p * 8.0 + vec3(0.0, t * 3.0, 0.0), 4);
  float skin = 1.0 - abs(snoise(q * 1.3 + vec3(t * 2.0, 0.0, 0.0))); skin = skin * skin * skin;
  float Tm = max(te * (1.0 + 0.08 * conv - 0.07 * skin) + 60.0, 1650.0 + 80.0 * conv);    // magma never below the liquidus
  // plates are Worley cells inset from their borders: thin glowing cracks when the crust is
  // nearly complete, rounded floating rafts as it thins toward the substellar point
  float inset = mix(0.62, 0.025, cover) + 0.12 * (w.z - 0.5) * (1.0 - cover) + 0.02 * snoise(q * 4.0);
  float solid = smoothstep(inset - 0.012, inset + 0.012, edge) * step(0.02, cover);
  float Tc = 0.6 * te + 140.0 * fbm(q * 3.0, 2) + 260.0 * (1.0 - cover);        // crust skin
  float rim = 1.0 - smoothstep(inset, inset + 0.05, edge);                         // hot rafts' edges
  crustF = solid;
  T = mix(Tm, mix(Tc, Tm * 0.85, rim * 0.6), solid);
  h = solid * smoothstep(inset, inset + 0.1, edge) * (0.6 + 0.4 * fbm(q * 2.0, 2));
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, S = uStar;
  vec3 col = sky(rd, 1.0) * 0.6;
  // the star: a real sphere 193 radii away, quadratic limb darkening and granulation
  vec2 hsS = sphereHit(ro, rd, S * STAR_D, STAR_R);
  if (hsS.x < hsS.y && hsS.x > 0.0){
    vec3 ps = normalize(ro + rd * hsS.x - S * STAR_D);
    float mu = max(dot(ps, -rd), 0.0);
    float ld = 1.0 - 0.62 * (1.0 - mu) - 0.12 * (1.0 - mu) * (1.0 - mu);
    float gran = 0.92 + 0.08 * snoise(ps * 90.0 + uWall * 0.05) + 0.04 * snoise(ps * 220.0);
    float spot = 1.0 - 0.5 * smoothstep(0.55, 0.75, snoise(ps * 6.0 + 4.0));
    col = bb(TSTAR * (0.94 + 0.06 * mu)) * 2.2 * ld * gran * spot;   // dimmed for display (true contrast vs. the lava ≈ 15×)
  } else {
    float cs = dot(rd, S);
    float a = acos(clamp(cs, -1.0, 1.0)) - asin(STAR_R / STAR_D);
    col += bb(TSTAR) * (0.3 * exp(-max(a, 0.0) * 14.0) + 0.04 * exp(-max(a, 0.0) * 3.0));
  }
  vec2 hg = sphereHit(ro, rd, vec3(0.0), 1.0);
  if (hg.x < hg.y && hg.x > 0.0){
    vec3 n0 = normalize(ro + rd * hg.x);
    float T, h0, cf;
    lavaField(n0, T, h0, cf);
    vec3 t1, t2; tangents(n0, t1, t2);
    float eps = 0.002, kb = 0.004;
    float Tx, hx, Ty, hy, c2;
    lavaField(normalize(n0 + t1 * eps), Tx, hx, c2);
    lavaField(normalize(n0 + t2 * eps), Ty, hy, c2);
    vec3 n = normalize(n0 - (t1 * (hx - h0) + t2 * (hy - h0)) * kb / eps);
    if (uView > 0.5){
      col = inferno((T - 900.0) / 2600.0) * 2.0;
    } else {
      float mu0 = dot(n, S);
      // reflected starlight: dark glassy basalt crust; smooth magma has a broad specular glint
      vec3 alb = mix(vec3(0.03, 0.028, 0.026), vec3(0.05, 0.045, 0.04), cf);
      col = alb / PI * ESTAR * irr(mu0) * bb(TSTAR);
      vec3 V = -rd, H = normalize(S + V);
      float nh = max(dot(n, H), 0.0), nv = max(dot(n, V), 1e-3);
      float a2 = 0.09, dd = nh * nh * (a2 - 1.0) + 1.0;
      float spec = a2 / (PI * dd * dd) * (0.04 + 0.96 * pow(1.0 - max(dot(V, H), 0.0), 5.0)) / (4.0 * nv);
      col += (1.0 - cf) * spec * irr(mu0) * ESTAR * bb(TSTAR) * 0.8;
      col += glow(T);
    }
    // thin rock-vapour haze at the limb
    float mu = max(dot(n0, -rd), 0.0);
    col += bb(1800.0) * pow(1.0 - mu, 6.0) * 0.4 * smoothstep(-0.3, 0.6, dot(n0, S));
  } else {
    // vapour atmosphere seen edge-on: forward-scattered starlight + thermal glow of the hot day side
    float b = length(cross(ro, rd));
    if (dot(ro, rd) < 0.0 && b < 1.06){
      vec3 pt = normalize(ro - rd * dot(ro, rd));
      float hz = exp(-(b - 1.0) / 0.006);
      float fwd = pow(max(dot(rd, S), 0.0), 8.0);
      col += hz * (bb(TSTAR) * 2.5 * fwd * smoothstep(-0.4, 0.1, dot(pt, S)) + bb(2200.0) * 0.5 * smoothstep(-0.2, 0.6, dot(pt, S)));
    }
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ================================================================================================
   JS helpers shared by the scene definitions
   ================================================================================================ */
// Sun direction for a given local solar time at the point under the camera, and solar declination.
// Afternoon puts the Sun west (= +azimuth here, since the surface moves toward −azimuth).
function sunFromHour(cam, hour, declDeg, out) {
  const az = Math.atan2(cam.x, cam.z) + (hour - 12) * Math.PI / 12, d = declDeg * D2R;
  return out.set(Math.cos(d) * Math.sin(az), Math.sin(d), Math.cos(d) * Math.cos(az));
}
function hitSphere(o, d, R = 1) {
  const b = o.dot(d), c = o.lengthSq() - R * R, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
const wrap180 = a => ((a + 180) % 360 + 360) % 360 - 180;
function bodyLatLon(p, rot) {
  const r = p.length();
  return { lat: Math.asin(p.y / r) / D2R, lon: wrap180(-(Math.atan2(p.x, p.z) + rot) / D2R) };
}
function localHour(p, L) {
  const h = 12 + (Math.atan2(L.x, L.z) - Math.atan2(p.x, p.z)) / D2R / 15;
  return ((h % 24) + 24) % 24;
}
const hhmm = h => `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`;
const latStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'B' : 'N'}`;
const lonStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'Đ' : 'T'}`;
function orbitRows(ctx, Rkm, refR = 1) {
  const d = ctx.camera.position.length(), hKm = (d - refR) * Rkm;
  const horizon = Math.sqrt(Math.max(d * d - refR * refR, 0)) * Rkm;
  return [['Độ cao', ctx.fmt.dist(hKm * 1000)], ['Khoảng cách tới đường chân trời', ctx.fmt.dist(horizon * 1000)]];
}
const hourParam = (def, note) => ({
  key: 'hour', label: 'Giờ địa phương (dưới camera)', type: 'range', min: 0, max: 24, step: 0.1, default: def,
  fmt: v => hhmm(v), note,
});
function makeMaterial(frag, uniforms) {
  return new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: frag, depthTest: false, depthWrite: false });
}
function sunElevStr(n, L) {
  const e = Math.asin(Math.max(-1, Math.min(1, n.dot(L)))) / D2R;
  const tag = e > 6 ? 'ban ngày' : e > -0.8 ? 'bình minh/hoàng hôn' : e > -18 ? 'chạng vạng' : 'ban đêm';
  return `Mặt Trời cao <b>${e.toFixed(1)}°</b> (${tag})`;
}

/* ================================================================================================
   Scene definitions
   ================================================================================================ */
export default [
{
  id: 'earth',
  name: 'Trái Đất',
  category: 'planets',
  order: 20,
  subtitle: 'Hành tinh đất đá · ngôi nhà của chúng ta',
  swatch: ['#9fd0ff', '#0b2a5a'],
  blurb: 'Hành tinh duy nhất được biết có sự sống, 71 % bề mặt là đại dương. Viền khí quyển xanh lam và bầu trời xanh đến từ <b>tán xạ Rayleigh</b> (∝ 1/λ⁴): ánh sáng xanh bị tán xạ mạnh gấp ~5 lần ánh sáng đỏ — cũng vì thế mà dải phân chia ngày–đêm nhuốm màu cam.',
  intro: 'Kéo <b>Giờ địa phương</b> về khoảng 18:00 để thấy hoàng hôn cam trên đường phân chia, hoặc 00:00 để thấy ánh đèn thành phố và cực quang. Nhấn <b>N</b> cho ban đêm.',
  facts: [
    ['Bán kính trung bình', '6 371 km'], ['Khối lượng', '5,97×10²⁴ kg'], ['Trọng lực bề mặt', '9,81 m/s²'],
    ['Độ nghiêng trục quay', '23,44°'], ['Ngày Mặt Trời / ngày sao', '24 h / 23 h 56 min 4 s'],
    ['Suất phản chiếu Bond', '0,30'], ['Khí quyển', 'N₂ 78 %, O₂ 21 %, Ar 0,9 % · 1 013 hPa'],
    ['Độ cao thang (Rayleigh / Mie)', '8 km / 1,2 km'],
  ],
  unitKm: 6371,
  camera: { distance: 3.2, polar: 72, azimuth: 20, min: 1.03, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.3,
  params: [
    hourParam(13.5, '12:00 = Mặt Trời sau lưng bạn (Trái Đất tròn đầy); 18:00 = đường phân chia đi qua giữa đĩa; 00:00 = mặt đêm.'),
    { key: 'decl', label: 'Mùa (xích vĩ Mặt Trời)', type: 'range', min: -23.44, max: 23.44, step: 0.1, default: 8, fmt: v => v.toFixed(1) + '°', note: '+23,44° = hạ chí Bắc bán cầu: vùng cực Bắc có ngày dài 24 giờ.' },
    { key: 'clouds', label: 'Độ che phủ mây', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5 },
    { key: 'atmo', label: 'Mật độ khí quyển', type: 'range', min: 0, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), note: 'Nhân hệ số tán xạ Rayleigh, Mie và hấp thụ ozone.' },
    { key: 'cities', label: 'Đèn thành phố', type: 'toggle', default: true },
    { key: 'kp', label: 'Bão địa từ (chỉ số Kp)', type: 'range', min: 0, max: 9, step: 0.1, default: 4, fmt: v => 'Kp ' + v.toFixed(1), note: 'Kp càng lớn, vành cực quang càng mở rộng về phía xích đạo và ánh đỏ 630 nm ở độ cao 200–300 km càng rõ.' },
    { key: 'aurora', label: 'Cực quang', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Ban đêm', key: 'n', title: 'Nhìn mặt đêm: đèn thành phố và cực quang', run: ctx => { ctx.params.hour = 0.5; ctx.toast('Nửa đêm dưới camera: ánh đèn đô thị bám theo bờ biển và vùng ôn đới; vành cực quang xanh lục (oxy 557,7 nm) bao quanh cực từ.'); } },
    { label: 'Hoàng hôn', key: 'j', title: 'Đặt đường phân chia ngày–đêm vào giữa đĩa', run: ctx => { ctx.params.hour = 17.8; ctx.toast('Ánh nắng chiều đi xuyên qua ~40 lần bề dày khí quyển: ánh xanh bị tán xạ hết, chỉ còn đỏ cam.'); } },
    { label: 'Bão địa từ', key: 'g', title: 'Bão địa từ cực mạnh (Kp 8,7 như tháng 5/2024)', run: ctx => { ctx.params.kp = 8.7; ctx.params.aurora = true; ctx.params.hour = 0.5; ctx.toast('Bão địa từ G5 (5/2024, Kp ≈ 9): vành cực quang mở rộng xuống vĩ độ ~45°, ánh đỏ oxy 630 nm ở độ cao lớn hiện rõ.'); } },
  ],
  create(ctx) {
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) }, uAurN: { value: new THREE.Vector3(0, 1, 0) }, uAurS: { value: new THREE.Vector3(0, -1, 0) },
      uRot: { value: 0 }, uCloudRot: { value: 0 }, uClouds: { value: 0.5 }, uCities: { value: 1 }, uAtmo: { value: 1 }, uKp: { value: 4 }, uAurora: { value: 1 },
    };
    const material = makeMaterial(EARTH_FRAG, uniforms);
    // geomagnetic north pole ≈ 80.7°N 72.7°W (body frame); the oval is displaced ~4° toward the night side
    const magB = new THREE.Vector3(...ll(80.7, -72.7)), mW = new THREE.Vector3(), anti = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0);
    const OMEGA = 0.004;   // visual spin rate (rad per sim-second)
    const rotOf = t => t * OMEGA;
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params, U = uniforms, L = U.uSunDir.value;
        sunFromHour(ctx.camera.position, p.hour, p.decl, L);
        const rot = rotOf(ctx.time.sim);
        U.uRot.value = rot;
        U.uCloudRot.value = rot * 0.97;
        U.uClouds.value = p.clouds; U.uCities.value = p.cities ? 1 : 0; U.uAtmo.value = p.atmo;
        U.uKp.value = p.kp; U.uAurora.value = p.aurora ? 1 : 0;
        // world = rotYᵀ(rot) · body ; rotYᵀ(a) = rotY(−a)
        mW.copy(magB).applyAxisAngle(Y, -rot);
        anti.copy(L).addScaledVector(mW, -L.dot(mW)).normalize();
        U.uAurN.value.copy(mW).addScaledVector(anti, -0.07).normalize();
        U.uAurS.value.copy(mW).negate().addScaledVector(anti, -0.07).normalize();
      },
      hud(ctx) {
        const c = ctx.camera.position, ll2 = bodyLatLon(c, rotOf(ctx.time.sim));
        return [
          ...orbitRows(ctx, 6371),
          ['Điểm dưới camera', `${latStr(ll2.lat)} ${lonStr(ll2.lon)}`],
          ['Giờ Mặt Trời tại đó', hhmm(ctx.params.hour)],
          ['Chu kỳ quỹ đạo tròn ở độ cao này', ctx.fmt.time(2 * Math.PI * Math.sqrt(Math.pow(c.length() * 6.371e6, 3) / 3.986e14))],
        ];
      },
      onClick(info, ctx) {
        const t = hitSphere(info.origin, info.dir);
        if (t === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, t).normalize(), L = uniforms.uSunDir.value;
        const g = bodyLatLon(p, rotOf(ctx.time.sim));
        ctx.toast(`Vĩ độ <b>${latStr(g.lat)}</b>, kinh độ <b>${lonStr(g.lon)}</b> · giờ địa phương <b>${hhmm(localHour(p, L))}</b> · ${sunElevStr(p, L)}.`);
      },
    };
  },
},
{
  id: 'mars',
  name: 'Sao Hỏa',
  category: 'planets',
  order: 26,
  subtitle: 'Hành tinh đất đá · hành tinh đỏ',
  swatch: ['#f0a070', '#5a1e0c'],
  blurb: 'Bụi oxit sắt (gỉ sắt) phủ khắp bề mặt tạo nên màu đỏ cam. Khí quyển CO₂ chỉ bằng ~0,6 % áp suất Trái Đất nhưng đầy bụi mịn: ban ngày bầu trời màu bơ caramel, còn hoàng hôn lại <b>xanh lam</b> — hạt bụi cỡ 1,5 µm tán xạ ánh sáng xanh mạnh về phía trước, quanh đĩa Mặt Trời.',
  intro: 'Nhấn <b>B</b> để nổi bão bụi toàn cầu như năm 2018, hoặc <b>X</b> để nhìn viền khí quyển ngược sáng: hoàng hôn xanh của Sao Hỏa.',
  facts: [
    ['Bán kính trung bình', '3 389,5 km'], ['Khối lượng', '6,42×10²³ kg'], ['Trọng lực bề mặt', '3,72 m/s²'],
    ['Một sol (ngày Mặt Trời)', '24 h 39 min 35 s'], ['Độ nghiêng trục quay', '25,19°'],
    ['Áp suất bề mặt', '~6 mbar (CO₂ 95 %)'], ['Độ cao thang khí quyển', '~11 km'],
    ['Olympus Mons', 'cao 21,9 km, rộng ~600 km'], ['Valles Marineris', 'dài ~4 000 km, sâu tới 7 km'],
  ],
  unitKm: 3389.5,
  camera: { distance: 3.2, polar: 70, azimuth: 115, min: 1.03, max: 30, fov: 45 },
  exposure: 1.35,
  bloom: 0.3,
  params: [
    hourParam(14.5, '12:00 = Mặt Trời sau lưng bạn; 23:00 = Mặt Trời gần như ngay sau hành tinh — viền ngược sáng hiện màu xanh.'),
    { key: 'decl', label: 'Mùa (xích vĩ Mặt Trời)', type: 'range', min: -25.19, max: 25.19, step: 0.1, default: 14, fmt: v => v.toFixed(1) + '°', note: '+25,2° = hạ chí phương Bắc: chỏm băng Bắc được chiếu sáng suốt ngày đêm.' },
    { key: 'dust', label: 'Bão bụi', type: 'range', min: 0, max: 1, step: 0.01, default: 0, fmt: v => 'τ ≈ ' + (0.45 + 6.5 * v * v).toFixed(1), note: 'Độ sâu quang học thẳng đứng của bụi: ~0,5 bình thường, > 6 khi bão bụi toàn cầu (tàu Opportunity "chết" vì thiếu nắng năm 2018).' },
    { key: 'caps', label: 'Chỏm băng cực', type: 'range', min: 0.3, max: 1.6, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), note: 'Chỏm băng nở ra vào mùa đông khi ~25 % CO₂ của khí quyển đóng băng thành băng khô.' },
    { key: 'atmo', label: 'Khí quyển', type: 'range', min: 0, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2) },
  ],
  actions: [
    { label: 'Bão bụi toàn cầu', key: 'b', title: 'Bão bụi bao phủ toàn hành tinh (6–7/2018)', run: ctx => { const on = ctx.params.dust < 0.5; ctx.params.dust = on ? 1 : 0; ctx.toast(on ? 'Bão bụi toàn cầu 2018: τ > 6, bụi bốc lên tới 60 km — bề mặt biến mất sau màn bụi màu bơ.' : 'Bụi lắng xuống trong vài tuần; τ trở về ~0,5.'); } },
    { label: 'Hoàng hôn xanh', key: 'x', title: 'Mặt Trời sau hành tinh: tán xạ thuận', run: ctx => {
      // put the Sun just behind the planet as seen from the camera (declination = −camera elevation)
      // (just inside the limb: exactly behind, the visible limb would sit in the planet's shadow)
      const c = ctx.camera.position; ctx.params.hour = Math.asin(1 / c.length()) / D2R * 0.8 / 15; ctx.params.decl = Math.max(-25.19, Math.min(25.19, -Math.asin(c.y / c.length()) / D2R));
      ctx.params.dust = Math.max(ctx.params.dust, 0.25); ctx.toast('Ngược sáng: bụi tán xạ thuận ánh sáng xanh mạnh hơn đỏ (g ≈ 0,84 so với 0,58), nên viền khí quyển gần Mặt Trời ánh xanh — giống hoàng hôn mà Curiosity chụp được.'); } },
  ],
  create(ctx) {
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) },
      uRot: { value: 0 }, uTau: { value: 0.45 }, uHd: { value: 11.1 / 3389.5 }, uAtmo: { value: 1 }, uCaps: { value: 1 },
    };
    const material = makeMaterial(MARS_FRAG, uniforms);
    const OMEGA = 0.0039;
    let storm = null;
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params, U = uniforms;
        sunFromHour(ctx.camera.position, p.hour, p.decl, U.uSunDir.value);
        U.uRot.value = ctx.time.sim * OMEGA;
        // a storm builds/decays smoothly (~ seconds here, weeks on Mars)
        storm = storm === null ? p.dust : storm + (p.dust - storm) * Math.min(1, dt * 0.8);
        U.uTau.value = 0.45 + 6.5 * storm * storm;
        U.uHd.value = (11.1 + 14 * storm) / 3389.5;     // dust lofted higher in a storm
        U.uAtmo.value = p.atmo; U.uCaps.value = p.caps;
      },
      hud(ctx) {
        const c = ctx.camera.position, g = bodyLatLon(c, ctx.time.sim * OMEGA);
        return [
          ...orbitRows(ctx, 3389.5),
          ['Điểm dưới camera', `${latStr(g.lat)} ${lonStr(g.lon)}`],
          ['Giờ Mặt Trời tại đó', hhmm(ctx.params.hour)],
          ['Độ sâu quang học bụi τ', uniforms.uTau.value.toFixed(2)],
          ['Nắng xuống tới mặt đất (giữa trưa)', (100 * Math.exp(-uniforms.uTau.value)).toFixed(0) + ' % (trực xạ)'],
        ];
      },
      onClick(info, ctx) {
        const t = hitSphere(info.origin, info.dir);
        if (t === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, t).normalize(), L = uniforms.uSunDir.value;
        const g = bodyLatLon(p, ctx.time.sim * OMEGA);
        ctx.toast(`Vĩ độ <b>${latStr(g.lat)}</b>, kinh độ <b>${lonStr(g.lon)}</b> · giờ địa phương <b>${hhmm(localHour(p, L))}</b> · ${sunElevStr(p, L)}. Trọng lực 3,72 m/s² — bạn nặng chỉ bằng 38 % trên Trái Đất.`);
      },
    };
  },
},
{
  id: 'venus',
  name: 'Sao Kim',
  category: 'planets',
  order: 15,
  subtitle: 'Hành tinh đất đá · nhà kính mất kiểm soát',
  swatch: ['#fff1c8', '#a07a3a'],
  blurb: 'Lớp mây axit sulfuric phản xạ ~76 % ánh nắng và che kín bề mặt. Bên dưới là địa ngục: <b>465 °C</b> và <b>92 bar</b> — hiệu ứng nhà kính CO₂ mất kiểm soát. Trong tia cực tím, các vệt hấp thụ hình chữ Y lộ ra, chạy vòng quanh hành tinh chỉ trong ~4 ngày (<b>siêu tự quay</b>), trong khi bản thân Sao Kim cần 243 ngày để tự quay một vòng.',
  intro: 'Nhấn <b>V</b> để chuyển giữa ánh sáng khả kiến, tử ngoại (như camera UVI của tàu Akatsuki) và radar xuyên mây (như tàu Magellan).',
  facts: [
    ['Bán kính trung bình', '6 051,8 km'], ['Khối lượng', '4,87×10²⁴ kg'], ['Trọng lực bề mặt', '8,87 m/s²'],
    ['Nhiệt độ bề mặt', '~465 °C (737 K)'], ['Áp suất bề mặt', '92 bar (CO₂ 96,5 %)'],
    ['Chu kỳ tự quay', '243,0 ngày (ngược chiều)'], ['Ngày Mặt Trời', '116,75 ngày'],
    ['Gió đỉnh mây', '~100 m/s — vòng quanh hành tinh trong ~4 ngày'], ['Suất phản chiếu Bond', '0,76'], ['Đỉnh mây', '~70 km'],
  ],
  unitKm: 6051.8,
  camera: { distance: 3.2, polar: 80, azimuth: 30, min: 1.03, max: 30, fov: 45 },
  exposure: 0.85,
  bloom: 0.3,
  params: [
    { key: 'view', label: 'Bước sóng quan sát', type: 'seg', options: [['vis', 'Khả kiến'], ['uv', 'Tử ngoại 365 nm'], ['radar', 'Radar']], default: 'vis', note: 'Radar 12,6 cm xuyên qua mây như Magellan (1990–94); ảnh đã chuẩn hoá theo góc tới, màu cam mô phỏng ảnh Venera.' },
    hourParam(14.5, '12:00 = Mặt Trời sau lưng bạn. Sao Kim quay ngược chiều nên Mặt Trời mọc ở phía tây.'),
    { key: 'spin', label: 'Siêu tự quay của mây', type: 'range', min: 0, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2) },
    { key: 'atmo', label: 'Lớp sương mù trên mây', type: 'range', min: 0, max: 4, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2) },
  ],
  actions: [
    { label: 'Đổi bước sóng', key: 'v', run: ctx => {
      const order = ['vis', 'uv', 'radar'], nx = order[(order.indexOf(ctx.params.view) + 1) % 3];
      ctx.params.view = nx;
      ctx.toast({ vis: 'Khả kiến: một quả cầu màu kem gần như không có chi tiết.', uv: 'Tử ngoại: chất hấp thụ UV chưa rõ bản chất vẽ nên chữ Y khổng lồ, bị gió 100 m/s cuốn quanh hành tinh.', radar: 'Radar: mây trong suốt với sóng 12,6 cm — lộ ra cao nguyên, vùng tessera gồ ghề, corona và dòng dung nham.' }[nx]);
    } },
  ],
  create(ctx) {
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) },
      uView: { value: 0 }, uCloudRot: { value: 0 }, uSurfRot: { value: 0 }, uAtmo: { value: 1 },
    };
    const material = makeMaterial(VENUS_FRAG, uniforms);
    let cloudRot = 0;
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params, U = uniforms;
        sunFromHour(ctx.camera.position, p.hour, 0, U.uSunDir.value);
        U.uView.value = p.view === 'uv' ? 1 : p.view === 'radar' ? 2 : 0;
        // retrograde: clouds circle in ~4 days, the surface in 243 days (both sped up for display)
        cloudRot -= ctx.time.dt * 0.012 * p.spin;
        U.uCloudRot.value = cloudRot;
        U.uSurfRot.value = -ctx.time.sim * 0.012 * (4 / 243);
        U.uAtmo.value = p.atmo;
      },
      hud(ctx) {
        return [
          ...orbitRows(ctx, 6051.8, 1 + 70 / 6051.8).map(([k, v], i) => i === 0 ? ['Độ cao trên đỉnh mây', v] : [k, v]),
          ['Giờ Mặt Trời dưới camera', hhmm(ctx.params.hour)],
          ['Mây đã quay', (-uniforms.uCloudRot.value / (2 * Math.PI)).toFixed(2) + ' vòng'],
        ];
      },
      onClick(info, ctx) {
        const t = hitSphere(info.origin, info.dir, 1 + 70 / 6051.8);
        if (t === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, t).normalize(), L = uniforms.uSunDir.value;
        ctx.toast(`Giờ địa phương <b>${hhmm(localHour(p, L))}</b> · ${sunElevStr(p, L)}. Ở đỉnh mây (70 km) áp suất ~0,03 bar, nhiệt độ ~−40 °C; ở độ cao ~55 km điều kiện gần giống mặt đất Trái Đất.`);
      },
    };
  },
},
{
  id: 'lava-world',
  name: 'Hành tinh dung nham',
  category: 'planets',
  order: 70,
  subtitle: '55 Cancri e (Janssen) · siêu Trái Đất khóa thủy triều',
  swatch: ['#ffd070', '#3a0800'],
  blurb: '55 Cancri e quay quanh sao chỉ trong 17,7 giờ, gần đến mức bị <b>khóa thủy triều</b>: một mặt mãi mãi là ban ngày. Mặt ngày nóng ~2 700 K, đủ để đá silicat nóng chảy thành biển dung nham tự phát sáng như một <b>vật đen</b>; ngôi sao chiếm ~32° bầu trời.',
  intro: 'Hành tinh không tự quay so với sao — hãy xoay camera ra phía sau để thấy mặt đêm với những vết nứt còn âm ỉ, và ngôi sao khổng lồ sau lưng nó. Nhấn <b>I</b> cho bản đồ nhiệt hồng ngoại.',
  facts: [
    ['Bán kính', '1,88 R⊕ (~12 000 km)'], ['Khối lượng', '7,99 M⊕'], ['Trọng lực bề mặt', '~22 m/s²'],
    ['Chu kỳ quỹ đạo', '17,7 giờ'], ['Bán trục lớn', '0,0154 AU (2,3 triệu km)'],
    ['Nhiệt độ mặt ngày (Spitzer)', '~2 700 K'], ['Nhiệt độ mặt đêm', '~1 400 K'],
    ['Sao chủ', '55 Cancri A (Copernicus), K0, 5 200 K'], ['Khoảng cách', '41 năm ánh sáng'],
    ['Khí quyển', 'có dấu hiệu CO/CO₂ thứ cấp (JWST, 2024)'],
  ],
  unitKm: 12000,
  camera: { distance: 3.4, polar: 78, azimuth: 5, min: 1.03, max: 40, fov: 45 },
  exposure: 0.9,
  bloom: 0.4,
  params: [
    { key: 'tss', label: 'Nhiệt độ điểm cận sao', type: 'range', min: 1500, max: 3500, step: 10, default: 2700, fmt: v => v.toFixed(0) + ' K', note: 'Cân bằng bức xạ không phân phối nhiệt: T = T₀·cos^¼(góc thiên đỉnh). Độ sáng khả kiến tăng cực nhanh theo T (định luật Planck).' },
    { key: 'crust', label: 'Lớp vỏ nguội', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5, note: 'Đá silicat đông đặc dưới ~1 600 K thành các mảng vỏ trôi trên biển dung nham.' },
    { key: 'view', label: 'Chế độ xem', type: 'seg', options: [['vis', 'Khả kiến'], ['ir', 'Bản đồ nhiệt']], default: 'vis', note: 'Bản đồ nhiệt: 900 K (tím đen) → 3 500 K (vàng trắng).' },
  ],
  actions: [
    { label: 'Hồng ngoại', key: 'i', run: ctx => { ctx.params.view = ctx.params.view === 'ir' ? 'vis' : 'ir'; } },
    { label: 'Ngược sáng', key: 'o', title: 'Bay ra phía đêm, nhìn về ngôi sao', run: ctx => {
      ctx.camera.position.setFromSphericalCoords(3.4, 84 * D2R, 215 * D2R); ctx.camera.lookAt(0, 0, 0); ctx.controls.target.set(0, 0, 0); ctx.controls.update();
      ctx.toast('Mặt đêm (~1 400 K): lớp vỏ đá đen với các khe nứt dung nham còn âm ỉ. Phía sau là 55 Cancri A — chiếm 32° bầu trời, to gấp ~60 lần Mặt Trời nhìn từ Trái Đất.');
    } },
  ],
  create(ctx) {
    const uniforms = {
      ...ctx.shared,
      uStar: { value: STAR_DIR.clone() }, uTss: { value: 2700 }, uCrust: { value: 0.5 }, uView: { value: 0 },
    };
    const material = makeMaterial(LAVA_FRAG, uniforms);
    const SINA = 0.2834;
    const irr = mu => mu > SINA ? mu : mu < -SINA ? 0 : (mu + SINA) ** 2 / (4 * SINA);
    const Teq = (n, tss) => Math.max(tss * Math.pow(irr(n.dot(STAR_DIR)), 0.25), 0.5 * tss);
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params;
        uniforms.uTss.value = p.tss; uniforms.uCrust.value = p.crust; uniforms.uView.value = p.view === 'ir' ? 1 : 0;
      },
      hud(ctx) {
        const c = ctx.camera.position, n = c.clone().normalize();
        const starAng = 2 * Math.asin(54.7 / c.clone().sub(STAR_DIR.clone().multiplyScalar(193)).length()) / D2R;
        return [
          ...orbitRows(ctx, 12000),
          ['Nhiệt độ cân bằng dưới camera', Teq(n, ctx.params.tss).toFixed(0) + ' K'],
          ['Đường kính biểu kiến của sao', starAng.toFixed(1) + '° (Mặt Trời từ Trái Đất: 0,53°)'],
        ];
      },
      onClick(info, ctx) {
        const t = hitSphere(info.origin, info.dir);
        if (t === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, t).normalize();
        const T = Teq(p, ctx.params.tss), z = Math.acos(Math.max(-1, Math.min(1, p.dot(STAR_DIR)))) / D2R;
        const state = T > 1900 ? 'biển dung nham lỏng' : T > 1500 ? 'dung nham với các mảng vỏ trôi' : 'vỏ đá đông đặc, nứt nẻ còn âm ỉ';
        ctx.toast(`Góc thiên đỉnh của sao <b>${z.toFixed(0)}°</b> · nhiệt độ cân bằng <b>${T.toFixed(0)} K</b> (${(T - 273).toFixed(0)} °C): ${state}. Bức xạ nhiệt ∝ T⁴ = ${(5.67e-8 * T ** 4 / 1e6).toFixed(1)} MW/m².`);
      },
    };
  },
},
];
