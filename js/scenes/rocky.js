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
   EARTH — real data: NASA Blue Marble NG (colour), NOAA ETOPO 2022 (relief + bathymetry),
   NASA Black Marble 2016 (night lights), NASA Blue Marble cloud composite.
   ================================================================================================ */
// High-resolution Himalaya / Karakoram / Tibet patch (1 arc-minute): lon 66°E … +2048′, lat 22°N … +1024′
const PATCH = { lon0: 66, lat0: 22, w: 2048, h: 1024 };
const EARTH_FRAG = HEAD + /* glsl */`
uniform vec3 uSunDir, uAurN, uAurS;
uniform float uRot, uClouds, uCities, uAtmo, uKp, uAurora;
uniform float uExag, uHMax, uTexOK, uCloudT, uTexel, uCloudTexel;
uniform sampler2D uColTex, uElevTex, uNightTex, uCloudTex, uPatchCol, uPatchElev;
const float RKM = 6371.0;
const float RTOP = 1.0 + 70.0 / RKM;
const float HCL = 8.0 / RKM;             // cloud deck: ~8 km above the (smoothed) ground
const float ESUN = 4.0;                  // solar irradiance in scene radiance units
// the 1′ patch, in global uv; PRATIO = patch texels per global texel
const vec2 P0 = vec2(${((PATCH.lon0 + 180) / 360).toFixed(7)}, ${((PATCH.lat0 + 90) / 180).toFixed(7)});
const vec2 PS = vec2(${(PATCH.w / 60 / 360).toFixed(7)}, ${(PATCH.h / 60 / 180).toFixed(7)});
const float PRATIO = ${(PATCH.w / (PATCH.w / 60 / 360) / 4096).toFixed(4)};

Atm atmE(){
  Atm A;
  A.R = 1.0; A.Hr = 8.0 / RKM; A.Hm = 1.2 / RKM;
  // β in 10⁻⁶ m⁻¹ × 6.371·10⁶ m per scene unit
  A.bR = vec3(5.8, 13.5, 33.1) * 6.371 * uAtmo;
  // ozone (Chappuis band) — really a layer near 25 km; put on the molecular profile ×2 to match its column
  A.bA = vec3(0.65, 1.881, 0.085) * 6.371 * 2.0 * uAtmo;
  // continental-average aerosol: vertical optical depth ≈ 0.08 at 550 nm (global mean AOD ≈ 0.1)
  A.bMs = vec3(60.0 * 6.371) * uAtmo; A.bMe = A.bMs / 0.9; A.g = vec3(0.76);
  return A;
}

// ---- data access -----------------------------------------------------------------------------
// body-fixed unit vector → equirectangular uv (u = east longitude, v = latitude; north up)
vec2 uvOf(vec3 p){ return vec2(atan(p.x, p.z) * (0.5 / PI) + 0.5, asin(clamp(p.y, -1.0, 1.0)) / PI + 0.5); }
// seam-free uv derivatives: analytic Jacobian of uvOf applied to the screen derivatives of p
vec2 uvD(vec3 p, vec3 dp){
  float q = max(p.x * p.x + p.z * p.z, 1e-6);
  return vec2((p.z * dp.x - p.x * dp.z) / q * (0.5 / PI), dp.y / (PI * sqrt(q)));
}
float patchW(vec2 uv, out vec2 q){
  q = (uv - P0) / PS;
  vec2 m = min(q, 1.0 - q);
  return smoothstep(0.0, 0.06, min(m.x, m.y));
}
// (elevation km, water fraction) at a global-texture LOD (the patch is sampled PRATIO× finer)
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
  return c;
}
float hgtKm(vec3 pb, float lod){ return max(elevAt(uvOf(pb), lod).r, 0.0); }

// ---- relief ray march ------------------------------------------------------------------------
// First crossing of the ray with the exaggerated height field r = 1 + k·max(e, 0) inside the shell,
// between tA (shell entry) and tB (sea-level sphere, or shell exit for rays that miss the planet).
// Uniform steps whose samples are prefiltered (mip LOD) to the lateral step length, then two secant
// refinements. Returns −1 for no hit.
float marchRelief(vec3 ro, vec3 rd, float tA, float tB, bool onSphere, mat3 R, float k, float pixAng){
  vec3 pm = normalize(ro + rd * (0.5 * (tA + tB)));
  float sinT = max(length(cross(rd, pm)), 0.02);
  float lat = (tB - tA) * sinT;                                   // lateral distance covered
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

// Soft shadow of the relief toward the Sun: geometric steps up to the shell top. The penumbra is
// widened beyond the Sun's 0.27° radius to the data resolution (no sharper detail exists).
float reliefShadow(vec3 pw, vec3 L, mat3 R, float k, float ds, float shellTop){
  // ds: the scale the surface was resolved at (≥ one data texel, ≥ the pixel footprint); finer
  // samples than that would make the surface shadow itself
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

// ridged fractal for sub-texel relief (sharp crests, rounded valleys — like eroded mountains)
float ridged(vec3 x){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++){ float n = 1.0 - abs(snoise(x)); s += a * n * n; x = x * 2.07 + vec3(3.1, -1.7, 5.3); a *= 0.5; }
  return s;
}

// ---- clouds (NASA composite + flow-map advection: trades westward, westerlies eastward) ---------
vec2 windUV(vec2 uv){
  float lat = (uv.y - 0.5) * PI;
  float a = abs(lat);
  float zonal = -0.8 * exp(-pow((a - 0.25) / 0.17, 2.0)) + 1.0 * exp(-pow((a - 0.8) / 0.22, 2.0));   // trades / westerlies
  return vec2(zonal / max(cos(lat), 0.15), 0.12 * sin(uv.x * TAU * 6.0 + lat * 9.0)) * 0.0045;
}
float cloudTex(vec2 uv, vec2 gx, vec2 gy){
  vec2 w = windUV(uv);
  float f0 = fract(uCloudT), f1 = fract(uCloudT + 0.5);
  float a = textureGrad(uCloudTex, uv - w * (f0 - 0.5), gx, gy).r;
  float b = textureGrad(uCloudTex, uv - w * (f1 - 0.5), gx, gy).r;
  return mix(a, b, abs(2.0 * f0 - 1.0));
}
// coverage 0.5 = the composite as observed; lower keeps only the thickest cloud, 1 = overcast
float cloudAlpha(float v){
  float lo = 0.1 + (0.5 - uClouds) * 1.7;
  return smoothstep(lo, lo + 0.52, v) * smoothstep(0.0, 0.04, uClouds);
}
// intersection with a cloud deck draped over the smoothed relief (fixed-point iteration)
float cloudHit(vec3 ro, vec3 rd, mat3 R, float k, out float lift){
  lift = 0.0;
  // the deck height is exaggerated half as much as the relief, so it keeps clear of most peaks
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

// city lights (Black Marble, stored with its display stretch): low-pressure/high-pressure sodium
// orange in the suburbs, whiter (LED, metal halide, saturation) in the brightest cores
vec3 cityLight(float v){
  float l = pow(v, 2.4);
  return mix(bb(2100.0), bb(3900.0), smoothstep(0.35, 0.95, v)) * l * 1.2;
}

// ---- surface -------------------------------------------------------------------------------
// pw: world hit, pb: body direction, fp: pixel footprint (scene units), gx/gy: uv derivatives
vec3 earthSurface(vec3 pw, vec3 pb, vec3 rd, vec3 L, Atm A, mat3 R, float k, float fp, vec2 gx, vec2 gy, float shellTop, float cloudDim){
  vec3 n0 = normalize(pw);
  vec2 uv = uvOf(pb);
  float tex = texelAt(uv);
  float ds = max(fp, tex);                                   // finite-difference step ≥ one data texel
  float lod = log2(ds / uTexel);
  float cphi = max(length(pb.xz), 0.02);
  float du = ds / (TAU * cphi), dv = ds / PI;
  vec2 c0 = elevAt(uv, lod);
  float e = c0.r, water = smoothstep(0.25, 0.75, c0.g);
  float hE = max(elevAt(uv + vec2(du, 0.0), lod).r, 0.0), hW = max(elevAt(uv - vec2(du, 0.0), lod).r, 0.0);
  float hN = max(elevAt(uv + vec2(0.0, dv), lod).r, 0.0), hS = max(elevAt(uv - vec2(0.0, dv), lod).r, 0.0);
  vec3 east = vec3(pb.z, 0.0, -pb.x) / cphi, north = cross(pb, east);
  vec2 g = vec2(hE - hW, hN - hS) * k / (2.0 * ds);          // exaggerated slope (east, north)
  float slope = length(g);
  vec3 nb = pb - g.x * east - g.y * north;
  // sub-texel relief: a ridged fractal continuing the measured slope spectrum (amplitude ∝ slope ×
  // texel size), only where the pixel footprint is finer than the data
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
  // sun transmittance: true altitude for the air mass, but the horizon of the exaggerated relief
  float muE = mu0s + sqrt(2.0 * hGeo) - sqrt(2.0 * hTrue);
  vec3 Ts = sunTrans(A, hTrue, muE);
  float sh = cloudDim;
  if (hGeo > 0.0 && muE > -0.02 && dot(Ts, vec3(1.0)) > 0.003){
    // long shadows only where they are wider than a pixel (always near the terminator)
    float tanZ = sqrt(max(1.0 - mu0s * mu0s, 0.0)) / max(mu0s, 0.02);
    if (hGeo * tanZ > 0.4 * fp) sh *= reliefShadow(pw, L, R, k, ds, shellTop);
  }
  // ambient sky light with a cavity term (valleys see less sky)
  float eC = max(elevAt(uv, lod + 3.0).r, 0.0);
  float ao = clamp(1.0 - (eC - max(e, 0.0)) * k / (ds * 10.0), 0.55, 1.0) * (0.75 + 0.25 * dot(n, n0));
  vec3 skyAmb = vec3(0.05, 0.085, 0.16) * smoothstep(-0.15, 0.4, mu0s) * uAtmo * ao;
  // water fraction: the 4k mask, sharpened at the coast by the 8k colour (Blue Marble's ocean fill is
  // a near-black navy, rgb 2,5,20)
  float wc = max((1.0 - smoothstep(0.006, 0.02, max(cl.r, cl.g))) * step(cl.r, cl.b) * smoothstep(0.0, 0.15, c0.g),
                 (1.0 - smoothstep(0.02, 0.06, dot(cl, vec3(0.2126, 0.7152, 0.0722)))) * smoothstep(0.1, 0.4, c0.g)) * (1.0 - smoothstep(0.25, 0.9, e));
  water = clamp(max(water, wc), 0.0, 1.0);
  vec3 land = vec3(0.0), sea = vec3(0.0);
  if (water < 0.999){
    // Blue Marble reflectance (linear ≈ 1.25 × albedo); its snow is sharpened at close range by
    // the sub-texel relief: snow holds on crests and gentle slopes, slides off steep faces
    vec3 alb = cl * 0.8;
    float mx = max(cl.r, max(cl.g, cl.b)), mn = min(cl.r, min(cl.g, cl.b));
    float snow = smoothstep(0.32, 0.6, mn) * smoothstep(0.75, 0.9, mn / max(mx, 1e-4));
    float fs = smoothstep(1.6, 0.6, fp / tex) * step(0.01, snow);
    if (fs > 0.0){
      // below the data resolution, a snowfield is broken by rock: snow sheds from slopes steeper than
      // ~35° and survives in hollows and on gentle crests (ridged noise stands in for the gullies)
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
    // open water: absorbing deep ocean, brighter shelves (bottom reflectance + sediment) from the
    // bathymetry; the Blue Marble colour wins where it shows real shallow-water colour
    float depth = max(-e, 0.0);
    vec3 alb = mix(vec3(0.0035, 0.011, 0.032), vec3(0.012, 0.045, 0.058), exp(-depth / 0.06));
    alb = max(alb, cl * 0.8 * (1.0 - wc));
    // seasonal sea ice (June: Arctic pack north of ~75°, Antarctic winter pack to ~62°S)
    float nz = fbm(pb * 9.0, 3);
    float ice = max(smoothstep(0.955, 0.975, pb.y + 0.03 * nz), smoothstep(-0.875, -0.895, pb.y + 0.03 * nz));
    alb = mix(alb, vec3(0.55, 0.6, 0.66) * (0.85 + 0.2 * nz), ice);
    float shw = cloudDim;
    sea = alb / PI * ESUN * (Ts * max(mu0s, 0.0) * shw + skyAmb);
    // sun glint: GGX microfacet ocean (wind-roughened, α ≈ 0.12) with Schlick Fresnel, F0 = 0.02
    vec3 V = -rd, H = normalize(L + V);
    float nh = max(dot(n0, H), 0.0), nv = max(dot(n0, V), 1e-3), nl = max(mu0s, 0.0);
    float a2 = 0.0144;
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float F = 0.02 + 0.98 * pow(1.0 - max(dot(V, H), 0.0), 5.0);
    float kk = 0.06;
    float G = nl / (nl * (1.0 - kk) + kk) * nv / (nv * (1.0 - kk) + kk);
    sea += (1.0 - ice) * D * F * G / (4.0 * nv) * ESUN * Ts * shw;
    // skylight reflected at grazing angles
    float Fv = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
    sea += (1.0 - ice) * Fv * vec3(0.06, 0.11, 0.22) * ESUN * 0.25 * smoothstep(-0.1, 0.3, mu0s) * uAtmo;
  }
  vec3 col = mix(land, sea, water);
  // night: city lights (dimmed by overhead cloud)
  float night = smoothstep(0.03, -0.12, mu0s);
  if (uCities > 0.5 && night > 0.0){
    float v = textureGrad(uNightTex, uv, gx, gy).r;
    col += cityLight(v) * night * (0.35 + 0.65 * cloudDim);
  }
  return col;
}

vec3 cloudShade(vec3 pcw, vec3 pcb, vec2 uvc, float v, float a, vec3 L, Atm A, mat3 R, float lod, float fp){
  vec3 nc = normalize(pcw);
  float mu0 = dot(nc, L);
  // cloud-top relief: thicker (brighter) cloud stands taller (≈ 2 → 12 km); bump from the gradient
  float cphi = max(length(pcb.xz), 0.02);
  float ds = uCloudTexel * exp2(max(lod, 0.0)) * 1.5;
  float vE = textureLod(uCloudTex, uvc + vec2(ds / (TAU * cphi), 0.0), lod).r;
  float vN = textureLod(uCloudTex, uvc + vec2(0.0, ds / PI), lod).r;
  vec3 east = vec3(pcb.z, 0.0, -pcb.x) / cphi, north = cross(pcb, east);
  float hk = 10.0 / RKM;
  vec3 nb = normalize(pcb - ((vE - v) * east + (vN - v) * north) * hk / ds);
  vec3 n = transpose(R) * nb;
  // self-shadowing: cloud mass a little toward the sun
  vec3 Lb = R * L;
  vec3 lt = normalize(Lb - pcb * dot(Lb, pcb));
  vec2 duvL = vec2(dot(lt, east) / (TAU * cphi), dot(lt, north) / PI) * ds * 2.0;
  float ahead = cloudAlpha(textureLod(uCloudTex, uvc + duvL, lod + 1.0).r);
  float self = 1.0 - 0.45 * clamp(ahead - 0.7 * a, 0.0, 1.0);
  vec3 Ts = sunTrans(A, HCL, mu0);
  float alb = mix(0.5, 0.86, smoothstep(0.25, 0.85, v));
  // wrapped Lambert (multiple scattering in the deck), tall tops catch the sun after ground sunset
  float lit = max(dot(n, L) * 0.75 + 0.25 * mu0 + 0.06, 0.0) / 1.06 * self;
  vec3 skyAmb = vec3(0.05, 0.08, 0.14) * smoothstep(-0.1, 0.3, mu0) * uAtmo;
  return alb / PI * ESUN * (Ts * lit + skyAmb);
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  Atm A = atmE();
  mat3 R = rotY(uRot);
  float k = uExag / RKM * uTexOK;
  float shellTop = 1.0 + k * uHMax + 1e-5;
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  // derivatives for texture filtering, from a seam-free and silhouette-continuous direction field:
  // the sea-level hit, or the closest approach for rays that miss
  vec2 hs = sphereHit(ro, rd, vec3(0.0), 1.0);
  bool onSphere = hs.x < hs.y && hs.x > 0.0;
  float tS = onSphere ? hs.x : max(-dot(ro, rd), 0.0);
  vec3 pS = R * normalize(ro + rd * tS);
  vec3 dpx = dFdx(pS), dpy = dFdy(pS);

  vec3 col = sky(rd, 1.0) * 0.6;
  // ---- relief ----
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
  if (ground && tC > tG) tC = -1.0;                          // a peak in front of the deck
  vec3 pcw = ro + rd * max(tC, 0.0), pcb = R * normalize(pcw);
  vec2 uvc = uvOf(pcb);
  vec2 gxc = uvD(pcb, dpx), gyc = uvD(pcb, dpy);
  float cv = 0.0, ca = 0.0;
  if (tC > 0.0){
    cv = cloudTex(uvc, gxc, gyc);
    // finer structure at close range (the composite is ~10 km per pixel)
    float fpc = tC * pixAng;
    float fc = smoothstep(1.0, 0.25, fpc / uCloudTexel);
    if (fc > 0.0) cv += fc * fbm(pcb / uCloudTexel * 0.3 + vec3(0.0, uCloudT * 0.2, 0.0), 3) * 0.15 * smoothstep(0.05, 0.4, cv) * (1.2 - cv);
    ca = cloudAlpha(cv);
  }
  // ---- surface ----
  vec3 bg = col;
  if (ground){
    vec3 pw = ro + rd * tG, pb = R * normalize(pw);
    // cloud shadow: the deck where the sun ray from this point crosses it
    float cd = 1.0;
    if (uClouds > 0.001){
      float rDeck = 1.0 + HCL * (1.0 + 0.5 * max(uExag - 1.0, 0.0)) + k * hgtKm(pb, 3.5);
      vec2 hc = sphereHit(pw, L, vec3(0.0), rDeck);
      if (hc.y > 0.0 && dot(pw, pw) < rDeck * rDeck){       // (peaks above the deck are in the clear)
        vec3 pq = R * normalize(pw + L * hc.y);
        vec2 uq = uvOf(pq);
        cd = 1.0 - 0.8 * cloudAlpha(cloudTex(uq, uvD(pq, dpx) * 2.0, uvD(pq, dpy) * 2.0));
      }
    }
    bg = earthSurface(pw, pb, rd, L, A, R, k, tG * pixAng, uvD(pb, dpx), uvD(pb, dpy), shellTop, cd);
  }
  if (ca > 0.002){
    float lodc = log2(max(tC * pixAng, 1e-7) / uCloudTexel);
    vec3 cc = cloudShade(pcw, pcb, uvc, cv, ca, L, A, R, lodc, tC * pixAng);
    // city glow scattered through the deck from below
    float nightC = smoothstep(0.0, -0.15, dot(normalize(pcw), L));
    if (uCities > 0.5 && nightC > 0.0) cc += cityLight(textureLod(uNightTex, uvOf(R * normalize(pcw)), max(lodc + 1.5, 2.0)).r) * 0.35 * nightC;
    bg = mix(bg, cc, ca);
  }
  // ---- atmosphere: single scattering from the entry point to whatever the ray hits ----
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

/* Earth uses a right-handed geography (east = counter-clockwise seen from the north, like the real
   planet and the NASA maps): body = rotY(−spin)·world, east longitude = atan2(x, z). */
const llE = (lat, lon) => [Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R)];
// afternoon puts the Sun west of the meridian (= −azimuth for this spin sense)
function sunFromHourE(cam, hour, declDeg, out) {
  const az = Math.atan2(cam.x, cam.z) - (hour - 12) * Math.PI / 12, d = declDeg * D2R;
  return out.set(Math.cos(d) * Math.sin(az), Math.sin(d), Math.cos(d) * Math.cos(az));
}
const bodyLatLonE = (p, spin) => ({ lat: Math.asin(p.y / p.length()) / D2R, lon: wrap180((Math.atan2(p.x, p.z) - spin) / D2R) });
const localHourE = (p, L) => (((12 + (Math.atan2(p.x, p.z) - Math.atan2(L.x, L.z)) / D2R / 15) % 24) + 24) % 24;
/* ================================================================================================
   Earth assets (assets/earth/, see CREDITS.txt)
   ================================================================================================ */
const EARTH_URL = f => new URL(`../../assets/earth/${f}`, import.meta.url).href;
// float32 → IEEE half (round-to-nearest, tiny values flushed to 0)
const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
function toHalf(v) {
  _f32[0] = v; const x = _u32[0], s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7bff;
  return s | ((e << 10) + (((x & 0x7fffff) + 0x1000) >>> 13));
}
// exact 8-bit pixels of a (colour-profile-free) lossless image
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
// Elevation PNG/WebP: R,G = (metres + 11000) as a 16-bit big-endian integer, B = water fraction.
// → RG16F texture (km, water) with a CPU-built box-filtered mip chain; rows flipped so v = 0 is south.
// Also returns a coarse float copy for CPU look-ups (click read-out, camera clearance).
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
// colour / grey JPEGs through the browser decoder; downscaled on a canvas if the GPU cannot hold them
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
// a busy static server occasionally drops a request: retry once with a cache-busting query
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
// 1×1 stand-ins until the data arrive (an ocean planet)
function solidTex(data, format, type) {
  const t = new THREE.DataTexture(data, 1, 1, format, type); t.needsUpdate = true; return t;
}
// bilinear look-up in a CPU elevation copy (lat, lon in degrees; rows south → north)
function sampleElev(cpu, lat, lon, lon0 = -180, lat0 = -90, spanLon = 360, spanLat = 180) {
  const { data, W, H } = cpu;
  const x = (lon - lon0) / spanLon * W - 0.5, y = (lat - lat0) / spanLat * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.max(0, Math.min(H - 2, Math.floor(y))), fx = x - x0, fy = Math.max(0, Math.min(1, y - y0));
  const at = (i, j) => data[(j * W + ((i % W) + W) % W) * 2];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
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
  blurb: 'Hành tinh duy nhất được biết có sự sống, 71 % bề mặt là đại dương. Bề mặt ở đây là <b>dữ liệu thật</b>: ảnh vệ tinh Blue Marble của NASA, địa hình và đáy biển ETOPO 2022, ánh đèn đêm Black Marble. Viền khí quyển xanh lam đến từ <b>tán xạ Rayleigh</b> (∝ 1/λ⁴): ánh sáng xanh bị tán xạ mạnh gấp ~5 lần ánh sáng đỏ — cũng vì thế mà dải phân chia ngày–đêm nhuốm màu cam.',
  intro: 'Nhấn <b>Y</b> để bay tới dãy Himalaya lúc bình minh. Kéo <b>Phóng đại địa hình</b> để thấy núi nổi lên; <b>N</b> cho ban đêm, <b>J</b> cho hoàng hôn.',
  facts: [
    ['Bán kính trung bình', '6 371 km'], ['Khối lượng', '5,97×10²⁴ kg'], ['Trọng lực bề mặt', '9,81 m/s²'],
    ['Độ nghiêng trục quay', '23,44°'], ['Ngày Mặt Trời / ngày sao', '24 h / 23 h 56 min 4 s'],
    ['Suất phản chiếu Bond', '0,30'], ['Khí quyển', 'N₂ 78 %, O₂ 21 %, Ar 0,9 % · 1 013 hPa'],
    ['Độ cao thang (Rayleigh / Mie)', '8 km / 1,2 km'],
    ['Đỉnh cao nhất / điểm sâu nhất', 'Everest 8 849 m / vực Challenger −10 935 m'],
    ['Dữ liệu', 'NASA Blue Marble NG (6/2004), NOAA ETOPO 2022, NASA Black Marble 2016, mây NASA Blue Marble'],
  ],
  unitKm: 6371,
  camera: { distance: 3.2, polar: 72, azimuth: 20, min: 1.03, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.3,
  params: [
    hourParam(13.5, '12:00 = Mặt Trời sau lưng bạn (Trái Đất tròn đầy); 18:00 = đường phân chia đi qua giữa đĩa; 00:00 = mặt đêm.'),
    { key: 'decl', label: 'Mùa (xích vĩ Mặt Trời)', type: 'range', min: -23.44, max: 23.44, step: 0.1, default: 20, fmt: v => v.toFixed(1) + '°', note: '+23,44° = hạ chí Bắc bán cầu: vùng cực Bắc có ngày dài 24 giờ. (Ảnh bề mặt và mây chụp vào tháng 6.)' },
    { key: 'exag', label: 'Phóng đại địa hình', type: 'range', min: 1, max: 40, step: 0.5, default: 8, fmt: v => '×' + v.toFixed(1), note: '×1 = tỉ lệ thật: Everest chỉ cao 0,14 % bán kính, gần như vô hình từ quỹ đạo. Phóng đại theo chiều đứng để núi nổi lên, đổ bóng và in bóng trên đường chân trời.' },
    { key: 'clouds', label: 'Độ che phủ mây', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5, note: '0,5 = đúng ảnh mây tổng hợp của NASA; mây trôi theo gió mậu dịch (về phía tây) và gió tây ôn đới (về phía đông).' },
    { key: 'atmo', label: 'Mật độ khí quyển', type: 'range', min: 0, max: 3, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2), note: 'Nhân hệ số tán xạ Rayleigh, Mie và hấp thụ ozone.' },
    { key: 'cities', label: 'Đèn thành phố', type: 'toggle', default: true },
    { key: 'kp', label: 'Bão địa từ (chỉ số Kp)', type: 'range', min: 0, max: 9, step: 0.1, default: 4, fmt: v => 'Kp ' + v.toFixed(1), note: 'Kp càng lớn, vành cực quang càng mở rộng về phía xích đạo và ánh đỏ 630 nm ở độ cao 200–300 km càng rõ.' },
    { key: 'aurora', label: 'Cực quang', type: 'toggle', default: true },
  ],
  actions: [
    { label: 'Himalaya', key: 'y', title: 'Bay tới dãy Himalaya lúc bình minh (bấm lần nữa để quay lại)', run: ctx => ctx.scene._earthShot?.(ctx) },
    { label: 'Ban đêm', key: 'n', title: 'Nhìn mặt đêm: đèn thành phố và cực quang', run: ctx => { ctx.params.hour = 0.5; ctx.toast('Nửa đêm dưới camera: ánh đèn đô thị (ảnh Black Marble 2016 của NASA) bám theo bờ biển, sông ngòi và đường sá; vành cực quang xanh lục (oxy 557,7 nm) bao quanh cực từ.'); } },
    { label: 'Hoàng hôn', key: 'j', title: 'Đặt đường phân chia ngày–đêm vào giữa đĩa', run: ctx => { ctx.params.hour = 17.8; ctx.toast('Ánh nắng chiều đi xuyên qua ~40 lần bề dày khí quyển: ánh xanh bị tán xạ hết, chỉ còn đỏ cam. Gần đường phân chia, núi đổ bóng dài hàng trăm km.'); } },
    { label: 'Bão địa từ', key: 'g', title: 'Bão địa từ cực mạnh (Kp 8,7 như tháng 5/2024)', run: ctx => { ctx.params.kp = 8.7; ctx.params.aurora = true; ctx.params.hour = 0.5; ctx.toast('Bão địa từ G5 (5/2024, Kp ≈ 9): vành cực quang mở rộng xuống vĩ độ ~45°, ánh đỏ oxy 630 nm ở độ cao lớn hiện rõ.'); } },
  ],
  create(ctx) {
    const R = ctx.renderer, caps = R.capabilities;
    const maxTex = caps.maxTextureSize, aniso = Math.min(8, caps.getMaxAnisotropy());
    const ocean = () => solidTex(new Uint16Array([toHalf(-4), toHalf(1)]), THREE.RGFormat, THREE.HalfFloatType);
    const black = () => solidTex(new Uint8Array([0]), THREE.RedFormat, THREE.UnsignedByteType);
    const navy = () => solidTex(new Uint8Array([2, 5, 20, 255]), THREE.RGBAFormat, THREE.UnsignedByteType);
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) }, uAurN: { value: new THREE.Vector3(0, 1, 0) }, uAurS: { value: new THREE.Vector3(0, -1, 0) },
      uRot: { value: 0 }, uClouds: { value: 0.5 }, uCities: { value: 1 }, uAtmo: { value: 1 }, uKp: { value: 4 }, uAurora: { value: 1 },
      uExag: { value: 8 }, uHMax: { value: 8.2 }, uTexOK: { value: 0 }, uCloudT: { value: 0 },
      uTexel: { value: 2 * Math.PI / 4096 }, uCloudTexel: { value: 2 * Math.PI / 4096 },
      uColTex: { value: navy() }, uElevTex: { value: ocean() }, uNightTex: { value: black() }, uCloudTex: { value: black() },
      uPatchCol: { value: navy() }, uPatchElev: { value: ocean() },
    };
    const material = makeMaterial(EARTH_FRAG, uniforms);
    const U = uniforms;
    let elevCPU = null, patchCPU = null;
    const ready = Promise.all([
      retry(loadImageTex, EARTH_URL('earth_color_8k.jpg'), { srgb: true, aniso, maxSize: maxTex }),
      retry(loadElevation, EARTH_URL('earth_elev_4k.webp'), 2),
      retry(loadImageTex, EARTH_URL('earth_night_8k.jpg'), { red: true, aniso: 4, maxSize: maxTex }),
      retry(loadImageTex, EARTH_URL('earth_clouds_4k.jpg'), { red: true, aniso: 4, maxSize: maxTex }),
      retry(loadImageTex, EARTH_URL('earth_himalaya_color.jpg'), { srgb: true, aniso, repeat: false }),
      retry(loadElevation, EARTH_URL('earth_himalaya_elev.webp'), 1, false),
    ]).then(([col, el, night, cl, pcol, pel]) => {
      U.uColTex.value = col; U.uElevTex.value = el.tex; U.uNightTex.value = night; U.uCloudTex.value = cl;
      U.uPatchCol.value = pcol; U.uPatchElev.value = pel.tex;
      U.uTexel.value = 2 * Math.PI / el.tex.image.width;
      U.uCloudTexel.value = 2 * Math.PI / cl.image.width;
      U.uHMax.value = Math.max(el.max, pel.max);
      elevCPU = el.cpu; patchCPU = pel.cpu;
      for (const t of [col, el.tex, night, cl, pcol, pel.tex]) R.initTexture(t);
      U.uTexOK.value = 1;
    });
    // elevation (m) and water at a body-frame lat/lon from the CPU copies
    const elevAt = (lat, lon) => {
      if (!elevCPU) return null;
      const pl = (lon - PATCH.lon0 + 360) % 360;
      if (patchCPU && pl < PATCH.w / 60 && lat > PATCH.lat0 && lat < PATCH.lat0 + PATCH.h / 60)
        return 1000 * sampleElev(patchCPU, lat, PATCH.lon0 + pl, PATCH.lon0, PATCH.lat0, PATCH.w / 60, PATCH.h / 60);
      return 1000 * sampleElev(elevCPU, lat, lon);
    };

    // geomagnetic north pole ≈ 80.7°N 72.7°W (body frame); the oval is displaced ~4° toward the night side
    const magB = new THREE.Vector3(...llE(80.7, -72.7)), mW = new THREE.Vector3(), anti = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0);
    const OMEGA = 0.004;   // visual spin rate (rad per sim-second)
    const rotOf = t => t * OMEGA;
    let lastRot = null;

    /* ---- the Himalaya close-up: a camera that rides with the rotating Earth ---------------- */
    const shot = { on: false, tw: null, pending: false };
    const V = () => new THREE.Vector3();
    const bodyDir = (lat, lon) => new THREE.Vector3(...llE(lat, lon));
    const toWorld = (v, rot) => v.clone().applyAxisAngle(Y, rot);
    const toBody = (v, rot) => v.clone().applyAxisAngle(Y, -rot);
    const def = ctx.scene;
    function frameOf(cam, ctl) { return { pos: cam.position.clone(), tgt: ctl.target.clone(), up: cam.up.clone() }; }
    function startTween(c, toBodyFrame, dur, onDone) {
      const rot = rotOf(c.time.sim);
      const from = frameOf(c.camera, c.controls);
      shot.tw = {
        t0: performance.now(), dur: dur * 1000, onDone,
        a: { pos: toBody(from.pos, rot), tgt: toBody(from.tgt, rot), up: toBody(from.up, rot) },
        b: toBodyFrame,
      };
    }
    const SHOT = (new URLSearchParams(location.search).get('shotGeo') || '23.5,87.5,330,28.6,86.0').split(',').map(Number);
    function himalayaFrame() {
      // like ISS031-E-67020: from ~380 km over the Ganges plain, looking north across the Himalaya to Tibet
      const tgt = bodyDir(SHOT[3], SHOT[4]);
      const pos = bodyDir(SHOT[0], SHOT[1]).multiplyScalar(1 + SHOT[2] / 6371);
      return { pos, tgt, up: pos.clone().normalize().add(tgt).normalize() };
    }
    function globalFrame(c) {
      const cam = def.camera, rot = rotOf(c.time.sim);
      const pos = new THREE.Vector3().setFromSphericalCoords(cam.distance, cam.polar * D2R, cam.azimuth * D2R);
      return { pos: toBody(pos, rot), tgt: V(), up: toBody(Y, rot) };
    }
    function goShot(c, instant) {
      const p = c.params, keep = instant ? (new URLSearchParams(location.search).get('params') || '') : '';
      const set = (k, v) => { if (!keep.includes(k + ':')) p[k] = v; };
      set('exag', 3); set('hour', 7.4); set('decl', 21); set('clouds', Math.min(p.clouds, 0.22));
      shot.on = true;
      c.controls.minDistance = 0.004;
      if (instant) { applyFrame(c, himalayaFrame(), rotOf(c.time.sim)); shot.tw = null; }
      else startTween(c, himalayaFrame(), 4.5);
      c.toast?.(`Buổi sáng trên dãy Himalaya, nhìn từ độ cao ~330 km (như từ Trạm Vũ trụ Quốc tế) qua đồng bằng sông Hằng về phía cao nguyên Tây Tạng. <b>Everest cao 8 849 m — chỉ bằng 0,14 % bán kính Trái Đất (6 371 km)</b>, nên ở đây địa hình được phóng đại <b>×${p.exag}</b> theo chiều đứng (kéo thanh <b>Phóng đại địa hình</b> về ×1 để thấy tỉ lệ thật). Nắng sớm từ phía đông chiếu xiên làm các sườn núi đổ bóng dài về phía tây; mây gió mùa trôi bên dưới các đỉnh tuyết.`, 11000);
    }
    function leaveShot(c) {
      c.params.exag = 8; c.params.hour = 13.5;
      startTween(c, globalFrame(c), 3.5, () => { shot.on = false; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); c.controls.target.set(0, 0, 0); });
    }
    function applyFrame(c, f, rot) {
      c.camera.position.copy(toWorld(f.pos, rot));
      c.controls.target.copy(toWorld(f.tgt, rot));
      c.camera.up.copy(toWorld(f.up, rot)).normalize();
      c.camera.lookAt(c.controls.target);
    }
    def._earthShot = c => { if (!shot.tw) (shot.on ? leaveShot(c) : goShot(c, false)); };
    // deep link / snapshot hook: ?earthShot=himalaya
    shot.pending = new URLSearchParams(location.search).get('earthShot') === 'himalaya';

    return {
      material,
      ready,
      exit(c) {
        if (shot.on) { shot.on = false; shot.tw = null; c.controls.minDistance = def.camera.min; c.camera.up.set(0, 1, 0); }
      },
      update(dt, c) {
        const p = c.params, L = U.uSunDir.value;
        const rot = rotOf(c.time.sim);
        if (shot.pending && U.uTexOK.value > 0) { shot.pending = false; goShot(c, true); }
        // camera that co-rotates with the ground while in the close-up
        if (shot.tw) {
          const tw = shot.tw, k0 = Math.min(1, (performance.now() - tw.t0) / tw.dur), e = k0 * k0 * (3 - 2 * k0);
          const ra = tw.a.pos.length(), rb = tw.b.pos.length();
          const dir = tw.a.pos.clone().normalize().lerp(tw.b.pos.clone().normalize(), e);
          if (dir.lengthSq() < 1e-6) dir.copy(tw.b.pos).normalize();
          // log-radius ease, lifted mid-way so the path clears the planet
          const r = Math.exp(Math.log(ra) * (1 - e) + Math.log(rb) * e) * (1 + 0.35 * Math.sin(Math.PI * e) * Math.min(1, Math.abs(ra - rb)));
          applyFrame(c, { pos: dir.normalize().multiplyScalar(r), tgt: tw.a.tgt.clone().lerp(tw.b.tgt, e), up: tw.a.up.clone().lerp(tw.b.up, e) }, rot);
          if (k0 >= 1) { shot.tw = null; tw.onDone?.(); }
        } else if (shot.on && lastRot !== null && rot !== lastRot) {
          const d = rot - lastRot;
          c.camera.position.applyAxisAngle(Y, d); c.controls.target.applyAxisAngle(Y, d); c.camera.up.applyAxisAngle(Y, d);
          c.camera.lookAt(c.controls.target);
        }
        lastRot = rot;
        // keep the camera above the (exaggerated) relief
        const top = 1 + (U.uHMax.value * p.exag + 12) / 6371;
        if (c.camera.position.length() < top) c.camera.position.setLength(top);

        sunFromHourE(c.camera.position, p.hour, p.decl, L);
        U.uRot.value = -rot;          // body = rotY(−spin)·world
        U.uCloudT.value = c.time.sim / 40;
        U.uClouds.value = p.clouds; U.uCities.value = p.cities ? 1 : 0; U.uAtmo.value = p.atmo;
        U.uKp.value = p.kp; U.uAurora.value = p.aurora ? 1 : 0;
        // the exaggeration eases toward the slider so mountains rise smoothly
        U.uExag.value += (p.exag - U.uExag.value) * Math.min(1, dt * 4);
        if (Math.abs(p.exag - U.uExag.value) < 0.01) U.uExag.value = p.exag;
        // world = rotY(spin) · body
        mW.copy(magB).applyAxisAngle(Y, rot);
        anti.copy(L).addScaledVector(mW, -L.dot(mW)).normalize();
        U.uAurN.value.copy(mW).addScaledVector(anti, -0.07).normalize();
        U.uAurS.value.copy(mW).negate().addScaledVector(anti, -0.07).normalize();
      },
      hud(c) {
        const cp = c.camera.position, ll2 = bodyLatLonE(cp, rotOf(c.time.sim));
        const e = elevAt(ll2.lat, ll2.lon);
        return [
          ...orbitRows(c, 6371),
          ['Điểm dưới camera', `${latStr(ll2.lat)} ${lonStr(ll2.lon)}` + (e === null ? '' : e >= 0 ? ` · cao ${Math.round(e).toLocaleString('vi-VN')} m` : ` · biển sâu ${Math.round(-e).toLocaleString('vi-VN')} m`)],
          ['Giờ Mặt Trời tại đó', hhmm(c.params.hour)],
          ['Phóng đại địa hình', '×' + U.uExag.value.toFixed(1)],
          ['Chu kỳ quỹ đạo tròn ở độ cao này', c.fmt.time(2 * Math.PI * Math.sqrt(Math.pow(cp.length() * 6.371e6, 3) / 3.986e14))],
        ];
      },
      onClick(info, c) {
        const t = hitSphere(info.origin, info.dir);
        if (t === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, t).normalize(), L = U.uSunDir.value;
        const g = bodyLatLonE(p, rotOf(c.time.sim));
        const e = elevAt(g.lat, g.lon);
        const hs = e === null ? '' : e >= 0 ? ` · độ cao <b>${Math.round(e).toLocaleString('vi-VN')} m</b>` : ` · độ sâu đáy biển <b>${Math.round(-e).toLocaleString('vi-VN')} m</b>`;
        c.toast(`Vĩ độ <b>${latStr(g.lat)}</b>, kinh độ <b>${lonStr(g.lon)}</b>${hs} · giờ địa phương <b>${hhmm(localHourE(p, L))}</b> · ${sunElevStr(p, L)}.`);
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
