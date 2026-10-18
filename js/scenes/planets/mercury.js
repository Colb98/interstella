// Sao Thủy — Mercury from real MESSENGER data, moving on its real orbit.
//
// One scene unit = Mercury's mean radius (2 439,4 km). The surface is the MESSENGER MDIS global
// basemaps (USGS Astrogeology / NASA PDS, public domain) with relief from the USGS MESSENGER stereo
// DEM; see assets/mercury/CREDITS.txt.
//
// Motion is computed from a real ephemeris, not animated by hand:
//   · heliocentric orbit from the JPL mean Keplerian elements (Standish), Kepler's equation solved
//     every frame → distance from the Sun, true anomaly, radial velocity;
//   · body orientation from the IAU 2015 rotation model (pole α₀, δ₀; prime meridian
//     W = 329,5988° + 6,1385108°·d) → the sub-solar point in MESSENGER map coordinates.
// Together they reproduce the 3:2 spin–orbit resonance (the hot poles 0° / 180° face the Sun at every
// perihelion, the warm poles 90° / 270° at aphelion), the 175,94-day solar day, the 1,5× change of
// the Sun's apparent size, and the ~8-day retrograde loop of the Sun near perihelion (when the orbital
// angular velocity, 6,35°/day, beats the spin, 6,14°/day).
//
// The scene's "world" frame has y = Mercury's spin axis. It is an inertial frame rotated about y by
// ψ (chosen per viewing mode: Sun fixed, stars fixed, or ground fixed); body = rotY(−β)·world with
// β = W − ψ, east longitude = atan2(x, z) in the body frame (as in the NASA/USGS maps).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';

const D2R = Math.PI / 180;
const RKM = 2439.4;
const AU_KM = 149597870.7;
const SNAP = new URLSearchParams(location.search).has('snap');
const QS = new URLSearchParams(location.search);

/* ================================================================================================
   Ephemeris
   ================================================================================================ */
const SPIN = 6.1385108;                  // °/day (IAU 2015) — sidereal day 58,6462 d
const PERIOD = 87.9691;                  // sidereal orbital period, days
const SOLAR_DAY = 1 / (1 / 58.6462 - 1 / PERIOD);   // 175,94 d
const jdOf = ms => ms / 86400000 + 2440587.5;
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
// Mercury's state at Julian date jd: distance r (AU), true/mean anomaly, the sub-solar point in body
// coordinates (planetocentric, east-positive, degrees) and the prime-meridian angle W (degrees).
function mercuryGeom(jd) {
  const d = jd - 2451545.0, T = d / 36525;
  // JPL "Keplerian elements for approximate positions of the major planets" (1800–2050 AD)
  const a = 0.38709927, e = 0.20563593 + 0.00001906 * T, I = (7.00497902 - 0.00594749 * T) * D2R;
  const L = 252.25032350 + 149472.67411175 * T, varpi = 77.45779628 + 0.16047689 * T, Om = (48.33076593 - 0.12534081 * T) * D2R;
  const M = ((((L - varpi) % 360) + 360) % 360) * D2R, w = varpi * D2R - Om;
  let E = M + e * Math.sin(M);
  for (let i = 0; i < 6; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const xp = a * (Math.cos(E) - e), yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
  const nu = Math.atan2(yp, xp), r = Math.hypot(xp, yp);
  const cw = Math.cos(w), sw = Math.sin(w), cO = Math.cos(Om), sO = Math.sin(Om), cI = Math.cos(I), sI = Math.sin(I);
  const xe = (cw * cO - sw * sO * cI) * xp + (-sw * cO - cw * sO * cI) * yp;
  const ye = (cw * sO + sw * cO * cI) * xp + (-sw * sO + cw * cO * cI) * yp;
  const ze = (sw * sI) * xp + (cw * sI) * yp;
  const eps = 23.43928 * D2R;                         // ecliptic → ICRF equator
  const s = [-xe / r, -(ye * Math.cos(eps) - ze * Math.sin(eps)) / r, -(ye * Math.sin(eps) + ze * Math.cos(eps)) / r];
  // IAU 2015 (Archinal et al. 2018) rotation model; the libration terms (≤ 0,011°) are omitted
  const a0 = (281.0103 - 0.0328 * T) * D2R, d0 = (61.4155 - 0.0049 * T) * D2R;
  const W = (((329.5988 + SPIN * d) % 360) + 360) % 360;
  const pole = [Math.cos(d0) * Math.cos(a0), Math.cos(d0) * Math.sin(a0), Math.sin(d0)];
  const node = [-Math.sin(a0), Math.cos(a0), 0], yq = cross(pole, node);
  const bx = node.map((v, i) => Math.cos(W * D2R) * v + Math.sin(W * D2R) * yq[i]), by = cross(pole, bx);
  // heliocentric radial velocity (km/s): drives the sodium tail through the Swings effect
  const vr = Math.sqrt(1.32712440018e11 / (a * (1 - e * e) * AU_KM)) * e * Math.sin(nu);
  return { r, nu, M, e, vr, W, lon: Math.atan2(dot(s, by), dot(s, bx)) / D2R, lat: Math.asin(dot(s, pole)) / D2R };
}
// Mercury's V-band phase curve (Mallama, Wang & Howard 2002), magnitudes vs phase angle in degrees
const mallamaV = a => -0.613 + 6.3280e-2 * a - 1.6336e-3 * a * a + 3.3644e-5 * a ** 3 - 3.4265e-7 * a ** 4 + 1.6893e-9 * a ** 5 - 3.0334e-12 * a ** 6;
const wrap180 = a => ((a + 180) % 360 + 360) % 360 - 180;
// body-frame unit vector for (lat, east lon) — east longitude = atan2(x, z)
const llB = (lat, lon) => new THREE.Vector3(Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R));
const Y = new THREE.Vector3(0, 1, 0);

// Surface temperature model (shared idea with the shader): radiative equilibrium on the day side,
// T = T_ss·μ₀^¼ with T_ss = 700 K·(0,3075 AU / r)^½ (≈ 700 K at perihelion, 570 K at aphelion,
// ε ≈ 0,95, bolometric Bond albedo ≈ 0,07); on the night side the regolith (very low thermal inertia)
// drops from ~170 K at dusk toward ~95 K by dawn (Vasavada et al. 1999), colder at high latitude.
function surfTempK(mu0, hourAngleDeg, latDeg, rAU) {
  const Tss = 700 * Math.sqrt(0.3075 / rAU);
  const Td = mu0 > 0 ? Tss * Math.pow(mu0, 0.25) : 0;
  const H = wrap180(hourAngleDeg);
  const since = H > 90 ? H - 90 : H < -90 ? H + 270 : 0;          // degrees of rotation since sunset
  const Tn = (95 + 75 * Math.exp(-since / 20)) * (0.8 + 0.2 * Math.cos(latDeg * D2R));
  return Math.max(Td, Tn);
}

/* ================================================================================================
   Shader
   ================================================================================================ */
const FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir, uTailDir;
uniform float uRot, uSkyRot, uSunAng, uFlux, uExag, uHMin, uHMax, uTexOK, uTexel, uLumTexel;
uniform float uMode, uTss, uExo, uExoK, uTrackN, uAutoExp;
uniform vec3 uTrack[24];
uniform sampler2D uLum, uChroma, uEnh, uElev;
const float RKM = 2439.4;
const float ESUN = 10.0;                     // solar irradiance at 0,387 AU in scene radiance units
const float LUM_MEAN = 0.391;                // area-weighted mean of the MDIS luminance map
// natural colour: Mercury is a slightly brownish grey (red-sloped reflectance spectrum)
const vec3 TINT = vec3(1.07, 0.99, 0.89);
// Hapke parameters for Mercury's regolith (cf. Domingue et al. 2016, 750 nm): single-scattering
// albedo w, backscattering Henyey–Greenstein ξ, shadow-hiding opposition surge B₀, h
const float HW = 0.25, HXI = -0.25, HB0 = 1.0, HH = 0.065;

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
// body-fixed unit vector → equirectangular uv (u = east longitude −180°…180°, v = latitude, north up)
vec2 uvOf(vec3 p){ return vec2(atan(p.x, p.z) * (0.5 / PI) + 0.5, asin(clamp(p.y, -1.0, 1.0)) / PI + 0.5); }
// seam-free uv derivatives: analytic Jacobian of uvOf applied to screen derivatives of p
vec2 uvD(vec3 p, vec3 dp){
  float q = max(p.x * p.x + p.z * p.z, 1e-6);
  return vec2((p.z * dp.x - p.x * dp.z) / q * (0.5 / PI), dp.y / (PI * sqrt(q)));
}
float elev(vec2 uv, float lod){ return textureLod(uElev, uv, max(lod, 0.0)).r; }         // km
float hKm(vec3 pb, float lod){ return elev(uvOf(pb), lod); }

// ---- relief ray march --------------------------------------------------------------------------
// First crossing of the ray with r = 1 + k·e(lat, lon) between the shell entry tA and the floor
// sphere (1 + k·eMin) or shell exit tB. Steps prefiltered (mip LOD) to their lateral length, then
// two secant refinements. Returns −1 for no hit.
float marchRelief(vec3 ro, vec3 rd, float tA, float tB, bool onFloor, mat3 R, float k, float pixAng){
  if (tA <= 0.0){
    // camera inside the relief shell (standing on the ground): steps growing with distance
    float tP = 0.0, dP = length(ro) - 1.0 - k * hKm(R * normalize(ro), 0.0), t = max(length(ro) - 1.0, 2e-5) * 0.5;
    for (int i = 0; i < 96; i++){
      t = min(t, tB);
      vec3 q = ro + rd * t; float r = length(q);
      float lod = log2(max(t * pixAng * 2.0, 1e-7) / uTexel);
      float d = r - 1.0 - k * hKm(R * (q / r), lod);
      if (d <= 0.0){
        for (int j = 0; j < 3; j++){
          float tc = tP + (t - tP) * dP / max(dP - d, 1e-9);
          vec3 qc = ro + rd * tc; float rc = length(qc);
          float dc = rc - 1.0 - k * hKm(R * (qc / rc), lod);
          if (dc <= 0.0){ t = tc; d = dc; } else { tP = tc; dP = dc; }
        }
        return tP + (t - tP) * dP / max(dP - d, 1e-9);
      }
      if (t >= tB) return -1.0;
      tP = t; dP = d;
      t += max(t * 0.07, 1e-5);
    }
    return -1.0;
  }
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
    float lod = max(lodS, log2(max(t * pixAng, 1e-7) / uTexel));
    float d = r - 1.0 - k * hKm(R * (q / r), lod);
    if (d <= 0.0 || (onFloor && float(i) >= n)){
      if (i == 0) return t;
      for (int j = 0; j < 2; j++){
        float tc = tP + (t - tP) * dP / max(dP - d, 1e-9);
        vec3 qc = ro + rd * tc; float rc = length(qc);
        float dc = rc - 1.0 - k * hKm(R * (qc / rc), lod);
        if (dc <= 0.0){ t = tc; d = dc; } else { tP = tc; dP = dc; }
      }
      return tP + (t - tP) * dP / max(dP - d, 1e-9);
    }
    tP = t; dP = d;
  }
  return -1.0;
}
// Soft shadow of the relief toward the Sun. The penumbra is the Sun's real angular radius (0,87° at
// perihelion — three times wider than seen from Earth), widened to the data resolution.
float reliefShadow(vec3 pw, vec3 L, mat3 R, float k, float ds, float shellTop){
  float t = ds * 1.5, res = 1.0;
  for (int i = 0; i < 22; i++){
    vec3 q = pw + L * t; float r = length(q);
    if (r > shellTop) break;
    float h = 1.0 + k * hKm(R * (q / r), log2(max(t * 0.25, ds) / uTexel));
    res = min(res, (r - h) / (t * uSunAng + ds * 0.25));
    if (res < -0.5) break;
    t *= 1.3;
  }
  return smoothstep(-0.5, 1.0, res);
}

// ---- sub-texel relief: small simple craters below the DEM resolution --------------------------
// One octave of a cellular crater population. Accumulates the slope (east, north) of bowl-shaped
// craters with raised rims, the analytic shadow cast by each crater's sunward rim onto its own floor
// (exact for a paraboloid bowl), and a "freshness" that brightens young craters and their ejecta
// (immature regolith). Most craters are degraded (d/D ≈ 0,03); fresh ones reach d/D ≈ 0,2.
// p: body point in cell units; e1/e2: tangent axes; ls: Sun azimuth in the tangent plane;
// tanE: tangent of the Sun's elevation divided by the vertical exaggeration.
void craterOctave(vec3 p, float seed, vec3 e1, vec3 e2, vec2 ls, float tanE, float wgt, float pxc, float dens, inout vec2 grad, inout float lit, inout float fresh){
  vec3 i = floor(p), f = fract(p);
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(x, y, z);
    vec3 r = hash33(i + o + seed);
    if (r.z > dens) continue;
    vec3 r2 = hash33(i + o + seed + 19.7);
    float Rc = 0.1 + 0.38 * r2.x * r2.x;
    // each crater fades in once its radius spans a few pixels (no sub-pixel shadow speckle)
    float wc = wgt * smoothstep(1.5, 4.0, Rc * pxc);
    if (wc <= 0.0) continue;
    vec3 dv = f - o - r;
    vec2 q = vec2(dot(dv, e1), dot(dv, e2)) / Rc;
    float d = length(q);
    if (d > 1.7) continue;
    float fr = r2.y * r2.y * r2.y * r2.y;
    float k = 0.06 + 0.34 * fr;                            // depth / radius
    if (d < 1.0){
      grad += wc * 2.0 * k * q;
      float xs = dot(q, ls);
      float t = -xs + sqrt(max(xs * xs - d * d + 1.0, 0.0));   // distance to the rim toward the Sun
      lit = min(lit, mix(1.0, smoothstep(-0.03, 0.03, t * tanE - k * (1.0 - d * d)), wc));
    }
    float rimE = exp(-pow((d - 1.0) * 4.0, 2.0));
    grad -= wc * k * 11.0 * rimE * (d - 1.0) * q / max(d, 1e-3);
    fresh = max(fresh, wc * fr * (d < 1.0 ? 0.8 : exp(-(d - 1.0) * 4.0)));
  }
}

// ---- Hapke (1981/1984) bidirectional reflectance, without macroscopic roughness ----------------
float hapke(float mu0, float mu, float cg, float w){
  float g = acos(clamp(cg, -1.0, 1.0));
  float B = HB0 / (1.0 + tan(0.5 * min(g, 3.1)) / HH);               // opposition surge
  float P = (1.0 - HXI * HXI) / pow(1.0 + 2.0 * HXI * cg + HXI * HXI, 1.5);
  float gam = sqrt(1.0 - w);
  float H0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * gam * mu0), H = (1.0 + 2.0 * mu) / (1.0 + 2.0 * gam * mu);
  return w / (4.0 * PI) * mu0 / (mu0 + mu) * ((1.0 + B) * P + H0 * H - 1.0);
}

// ---- temperature (same model as surfTempK in the JS) -------------------------------------------
float surfTemp(float mu0, float H, float latC){
  float Td = mu0 > 0.0 ? uTss * pow(mu0, 0.25) : 0.0;
  float since = H > 0.5 * PI ? H - 0.5 * PI : (H < -0.5 * PI ? H + 1.5 * PI : 0.0);
  float Tn = (95.0 + 75.0 * exp(-degrees(since) / 20.0)) * (0.8 + 0.2 * latC);
  return max(Td, Tn);
}
vec3 inferno(float t){
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.000219, 0.001651, -0.019481), c1 = vec3(0.106513, 0.563956, 3.932712);
  const vec3 c2 = vec3(11.602493, -3.972854, -15.942394), c3 = vec3(-41.703996, 17.436399, 44.354145);
  const vec3 c4 = vec3(77.162936, -33.402359, -81.807309), c5 = vec3(-71.319428, 32.626064, 73.209520);
  const vec3 c6 = vec3(25.131126, -12.242669, -23.070325);
  vec3 c = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
  return pow(clamp(c, 0.0, 1.0), vec3(2.2));
}

// ---- surface --------------------------------------------------------------------------------------
vec3 surface(vec3 pw, vec3 pb, vec3 rd, vec3 L, mat3 R, float k, float fp, vec2 gx, vec2 gy, float shellTop){
  vec3 n0 = normalize(pw);
  vec2 uv = uvOf(pb);
  float ds = max(fp, uTexel);
  float lod = log2(ds / uTexel);
  float cphi = max(length(pb.xz), 0.02);
  float du = ds / (TAU * cphi), dv = ds / PI;
  float e = elev(uv, lod);
  float hE = elev(uv + vec2(du, 0.0), lod), hW = elev(uv - vec2(du, 0.0), lod);
  float hN = elev(uv + vec2(0.0, dv), lod), hS = elev(uv - vec2(0.0, dv), lod);
  vec3 east = vec3(pb.z, 0.0, -pb.x) / cphi, north = cross(pb, east);
  vec2 g = vec2(hE - hW, hN - hS) * k / (2.0 * ds);
  vec3 nb = pb - g.x * east - g.y * north;
  // below the DEM resolution: three octaves of small craters (4,5 km, 1,4 km, 0,45 km cells), each
  // faded in as the pixel footprint gets finer than it
  float cLit = 1.0, cFresh = 0.0;
  float cell = uTexel * 0.6;
  if (fp < cell * 0.5 && uTexOK > 0.0){
    vec3 Lb = R * L;
    vec3 nd = normalize(nb);
    float sE = dot(nd, Lb);
    vec2 ls = vec2(dot(Lb, east), dot(Lb, north));
    ls /= max(length(ls), 1e-4);
    float exf = 1.0 + 0.25 * (uExag - 1.0);
    float tanE = sE / max(sqrt(max(1.0 - sE * sE, 0.0)), 1e-3) / exf;
    vec2 grad = vec2(0.0);
    // crater density varies regionally (smooth volcanic plains vs. heavily cratered terrain)
    float dens = 0.1 + 0.2 * smoothstep(-0.5, 0.5, snoise(pb * 7.0) + 0.5 * snoise(pb * 23.0));
    for (int o = 0; o < 3; o++){
      float c = cell / pow(3.2, float(o));
      float w = smoothstep(0.6, 0.2, fp / c);
      if (w <= 0.0) break;
      craterOctave(pb / c, float(o) * 13.0, east, north, ls, tanE, w, c / fp, dens, grad, cLit, cFresh);
    }
    nb -= (east * grad.x + north * grad.y) * exf;
  }
  vec3 n = transpose(R) * normalize(nb);

  float mu0s = dot(n0, L);
  float mu0 = dot(n, L), mu = max(dot(n, -rd), 1e-3);
  mu0 *= cLit;
  // shadows cast by the relief, wherever an obstacle could reach higher than a pixel
  // (a peak of height Δh stays lit up to √(2Δh) beyond the terminator)
  float sh = 1.0;
  float kRange = max(k * (uHMax - e), 0.0);
  if (mu0s < -sqrt(2.0 * kRange) - 0.002) sh = 0.0;
  else if (mu0 > 0.0 && kRange > 0.0){
    float tanZ = sqrt(max(1.0 - mu0s * mu0s, 0.0)) / max(mu0s, 0.02);
    if (kRange * tanZ > 0.4 * fp) sh = reliefShadow(pw, L, R, k, ds, shellTop);
  }

  // ---- albedo: MDIS luminance (4k) with the colour of the MDIS 3-colour (1k) or enhanced (2k) map
  float v = textureGrad(uLum, uv, gx, gy).r;
  vec3 col;
  if (uMode < 1.5){
    // the MDIS map is display-stretched; real albedo contrast is milder (rays ~1,6× the mean)
    float a = pow(max(v, 0.0) / LUM_MEAN, 0.75);
    // close up: immature regolith of young small craters is brighter; faint sub-pixel mottling
    float fl = smoothstep(1.0, 0.3, fp / uLumTexel);
    a *= (1.0 + 0.45 * cFresh) * (1.0 + fl * 0.07 * snoise(pb / (uLumTexel * 0.35)));
    vec3 alb;
    if (uMode < 0.5){
      // natural: the 1000/750/430 nm colour, desaturated to what the eye would see, on Mercury's tint
      vec3 c = textureGrad(uChroma, uv, gx, gy).rgb;
      vec3 q = c / max(dot(c, vec3(1.0 / 3.0)), 1e-3);
      alb = TINT * max(1.0 + 0.32 * (q - 1.0), 0.0) * a;
    } else {
      // MESSENGER enhanced colour (R = PC2, G = PC1, B = 430/1000 nm ratio), sharpened by the luminance
      vec3 c = pow(textureGrad(uEnh, uv, gx, gy).rgb, vec3(2.2));
      float lf = log2(max(fp, 1e-7) / uLumTexel);
      float vc = textureLod(uLum, uv, max(lf + 1.0, 1.0)).r;
      alb = c * 3.6 * clamp(v / max(vc, 0.02), 0.5, 2.0);
    }
    float r = hapke(max(mu0, 0.0), mu, dot(L, -rd), HW);
    col = alb * r * ESUN * uFlux * sh * uAutoExp;
  } else {
    // surface temperature map (inferno: 90 K → 720 K)
    vec3 Lb = R * L;
    float H = atan(pb.x, pb.z) - atan(Lb.x, Lb.z);
    H = mod(H + PI, TAU) - PI;
    float T = surfTemp(max(mu0s, 0.0) * mix(1.0, sh, 0.8), H, cphi);
    col = inferno((T - 40.0) / 680.0) * 0.55 * (0.75 + 0.25 * clamp(v / LUM_MEAN, 0.4, 1.6)) * (0.85 + 0.3 * max(mu0, 0.0));
  }
  // sub-solar point track (the Sun's path over the ground in the last days), and today's sub-solar point
  if (uTrackN > 0.5){
    vec3 Lb = R * L;
    float dS = length(pb - Lb);
    col += vec3(1.0, 0.75, 0.25) * 1.2 * (smoothstep(0.022, 0.017, dS) - smoothstep(0.014, 0.010, dS));
    for (int i = 0; i < 24; i++){
      if (float(i) >= uTrackN) break;
      float d = length(pb - uTrack[i]);
      float w = 1.0 - float(i) / uTrackN;
      col += vec3(0.25, 0.75, 1.0) * 0.6 * w * smoothstep(0.0055, 0.0025, d);
    }
  }
  return col;
}

// ---- sodium exosphere and tail (Na D 589 nm resonance fluorescence) ---------------------------
const vec3 NA_COL = vec3(1.0, 0.62, 0.05);
vec3 sodium(vec3 ro, vec3 rd, float tEnd, vec3 L){
  vec2 hb = sphereHit(ro, rd, vec3(0.0), 60.0);
  if (hb.x > hb.y || hb.y < 0.0) return vec3(0.0);
  float jit = hash13(vec3(gl_FragCoord.xy, 3.7));
  float acc = 0.0;
  // 1) the bound exosphere: inside r < 1,8
  vec2 hn = sphereHit(ro, rd, vec3(0.0), 1.8);
  if (hn.x < hn.y && hn.y > 0.0){
    float t0 = max(hn.x, 0.0), t1 = min(hn.y, tEnd);
    const int N = 16;
    float dt = (t1 - t0) / float(N);
    for (int i = 0; i < N; i++){
      vec3 p = ro + rd * (t0 + (float(i) + jit) * dt);
      float r = length(p);
      float s = dot(p, -L), rho = length(p + L * s);
      float lit = 1.0 - step(0.0, s) * (1.0 - smoothstep(0.97, 1.03, rho));     // no fluorescence in the shadow
      acc += exp(-(r - 1.0) / 0.1) * (0.35 + 0.65 * smoothstep(-0.4, 0.6, dot(p, L) / r))
           * (1.0 + 1.4 * smoothstep(0.35, 0.8, abs(p.y) / r) * smoothstep(-0.2, 0.4, dot(p, L) / r)) * 3.0 * lit * dt;
    }
  }
  // 2) the tail: march inside a cylinder (radius 7) around the tail axis
  vec3 A = uTailDir;
  vec3 od = ro - A * dot(ro, A), dd = rd - A * dot(rd, A);
  float qa = dot(dd, dd), qb = dot(od, dd), qc = dot(od, od) - 49.0;
  float disc = qb * qb - qa * qc;
  if (disc > 0.0 && qa > 1e-8){
    float sq = sqrt(disc);
    float t0 = max(max((-qb - sq) / qa, hb.x), 0.0), t1 = min(min((-qb + sq) / qa, hb.y), tEnd);
    if (t1 > t0){
      const int N = 24;
      float dt = (t1 - t0) / float(N);
      for (int i = 0; i < N; i++){
        vec3 p = ro + rd * (t0 + (float(i) + jit) * dt);
        float s = dot(p, A), rho = length(p - A * s);
        float w = 0.9 + 0.11 * max(s, 0.0);
        float sL = dot(p, -L), rL = length(p + L * sL);
        float lit = 1.0 - step(0.0, sL) * (1.0 - smoothstep(0.97, 1.05, rL));
        acc += 1.6 * exp(-rho * rho / (w * w)) / (w * w) * smoothstep(-1.0, 1.5, s) * exp(-max(s, 0.0) / 90.0) * uExoK * lit * dt;
      }
    }
  }
  // brightness hugely exaggerated: the real emission is ~10⁵× fainter than the sunlit surface
  return NA_COL * acc * 0.05 * uFlux;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos, L = uSunDir;
  mat3 R = rotY(uRot);
  float k = uExag / RKM * uTexOK;
  float shellTop = 1.0 + k * uHMax + 1e-5, shellBot = 1.0 + k * uHMin - 1e-5;
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  // texture-filter derivatives from a seam-free, silhouette-continuous direction field
  vec2 hs = sphereHit(ro, rd, vec3(0.0), 1.0);
  bool onSphere = hs.x < hs.y && hs.x > 0.0;
  float tS = onSphere ? hs.x : max(-dot(ro, rd), 0.0);
  vec3 pS = R * normalize(ro + rd * tS);
  vec3 dpx = dFdx(pS), dpy = dFdy(pS);

  // ---- sky: stars (fixed to the inertial frame) and the Sun at its true angular size ----
  vec3 col = sky(rotY(uSkyRot) * rd, 1.0) * 0.6;
  // ---- relief ----
  float tG = -1.0;
  vec2 hO = sphereHit(ro, rd, vec3(0.0), shellTop);
  if (hO.x < hO.y && hO.y > 0.0){
    vec2 hF = sphereHit(ro, rd, vec3(0.0), shellBot);
    bool onFloor = hF.x < hF.y && hF.x > 0.0;
    float tA = max(hO.x, 0.0), tB = onFloor ? hF.x : hO.y;
    tG = marchRelief(ro, rd, tA, tB, onFloor, R, k, pixAng);
  }
  bool ground = tG > 0.0;
  if (!ground){
    float c = dot(rd, L), a = acos(clamp(c, -1.0, 1.0));
    // photosphere with limb darkening, plus the inner corona / glare
    float x = clamp(a / uSunAng, 0.0, 1.0);
    float limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - x * x, 0.0)));
    col += bb(5772.0) * (smoothstep(uSunAng * 1.06, uSunAng * 0.94, a) * 300.0 * limb
          + 0.5 * pow(uSunAng / max(a, uSunAng), 3.0) + pow(max(c, 0.0), 6000.0) * 0.5 * uFlux);
  }
  if (ground){
    vec3 pw = ro + rd * tG, pb = R * normalize(pw);
    col = surface(pw, pb, rd, L, R, k, tG * pixAng, uvD(pb, dpx), uvD(pb, dpy), shellTop);
  }
  if (uExo > 0.5) col += sodium(ro, rd, ground ? tG : 1e9, L);
  gl_FragColor = vec4(col, 1.0);
}`;

/* ================================================================================================
   Assets (assets/mercury/, see CREDITS.txt)
   ================================================================================================ */
const URL_OF = f => new URL(`../../../assets/mercury/${f}`, import.meta.url).href;
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
// DEM in two files (to fit the size budget): a 2048×1024 lossless WebP base, R,G = (metres + 12000) / 8
// as a 16-bit big-endian integer, plus a 4096×2048 JPEG of the residual against the bilinearly
// upsampled base (128 + metres / 6; rms error 15 m). → R16F (km) at 4096×2048 with a CPU-built
// box-filtered mip chain; rows flipped so v = 0 is the south pole. Keeps a float copy for the CPU.
async function loadDEM(baseUrl, detailUrl) {
  const [B, D] = await Promise.all([decodeImage(baseUrl), decodeImage(detailUrl)]);
  const bw = B.W, bh = B.H, W = D.W, H = D.H;
  const base = new Float32Array(bw * bh);            // rows north → south, metres
  for (let i = 0, si = 0; i < bw * bh; i++, si += 4) base[i] = ((B.px[si] << 8) | B.px[si + 1]) * 8 - 12000;
  let lv = new Float32Array(W * H), min = 1e9, max = -1e9;
  const sx = bw / W, sy = bh / H;
  for (let y = 0; y < H; y++) {
    const fyy = (y + 0.5) * sy - 0.5, y0 = Math.floor(fyy), fy = fyy - y0;
    const ya = Math.max(0, Math.min(bh - 1, y0)) * bw, yb = Math.max(0, Math.min(bh - 1, y0 + 1)) * bw;
    let si = y * W * 4, di = (H - 1 - y) * W;
    for (let x = 0; x < W; x++, si += 4, di++) {
      const fxx = (x + 0.5) * sx - 0.5, x0 = Math.floor(fxx), fx = fxx - x0;
      const xa = (x0 + bw) % bw, xb = (x0 + 1) % bw;
      const up = (base[ya + xa] * (1 - fx) + base[ya + xb] * fx) * (1 - fy) + (base[yb + xa] * (1 - fx) + base[yb + xb] * fx) * fy;
      const e = (up + (D.px[si] - 128) * 6) / 1000;
      lv[di] = e; if (e < min) min = e; if (e > max) max = e;
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
      const a = (y * sy) * w + x * sx, b = a + sx - 1, d = a + (sy - 1) * w, f = d + sx - 1;
      nx[y * w2 + x] = 0.25 * (lv[a] + lv[b] + lv[d] + lv[f]);
    }
    lv = nx; w = w2; h = h2;
  }
  const tex = new THREE.DataTexture(mips[0].data, W, H, THREE.RedFormat, THREE.HalfFloatType);
  tex.mipmaps = mips;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return { tex, min, max, cpu };
}
function loadImageTex(url, { srgb = false, red = false, aniso = 1 }) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    if (red) tex.format = THREE.RedFormat;
    tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = aniso;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(`không tải được ${url}`))));
}
const retry = (fn, url, ...a) => fn(url, ...a).catch(() => new Promise(r => setTimeout(r, 500)).then(() => fn(url + '?r=' + Date.now(), ...a)));
function solidTex(data, format) { const t = new THREE.DataTexture(data, 1, 1, format, THREE.UnsignedByteType); t.needsUpdate = true; return t; }
function sampleDEM(cpu, lat, lon) {
  const { data, W, H } = cpu;
  const x = (lon + 180) / 360 * W - 0.5, y = (lat + 90) / 180 * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.max(0, Math.min(H - 2, Math.floor(y))), fx = x - x0, fy = Math.max(0, Math.min(1, y - y0));
  const at = (i, j) => data[j * W + ((i % W) + W) % W];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

/* ================================================================================================
   UI helpers
   ================================================================================================ */
const latStr = v => `${Math.abs(v).toFixed(1).replace('.', ',')}°${v >= 0 ? 'B' : 'N'}`;
const lonStr = v => `${Math.abs(v).toFixed(1).replace('.', ',')}°${v >= 0 ? 'Đ' : 'T'}`;
const num = (v, d = 1) => v.toFixed(d).replace('.', ',');
const hhmm = h => `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`;
function dateStr(jd) {
  const d = new Date((jd - 2440587.5) * 86400000);
  const p = n => String(n).padStart(2, '0');
  return `${p(d.getUTCDate())}/${p(d.getUTCMonth() + 1)}/${d.getUTCFullYear()} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`;
}
function hitSphere(o, d, R = 1) {
  const b = o.dot(d), c = o.lengthSq() - R * R, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
const CALORIS = { lat: 31.5, lon: 162.7 };     // basin centre (Fassett et al. 2009), Ø ≈ 1 550 km
const SUN_AZ = 95 * D2R;                       // world azimuth of the Sun in the "Sun fixed" view

/* ================================================================================================
   Scene definition
   ================================================================================================ */
export default [{
  id: 'mercury',
  name: 'Sao Thủy',
  category: 'planets',
  order: 10,
  subtitle: 'Hành tinh đất đá trong cùng · cộng hưởng quay–quỹ đạo 3:2',
  swatch: ['#c9c2b8', '#3b3631'],
  blurb: 'Hành tinh nhỏ nhất, gần Mặt Trời nhất, không có khí quyển — bề mặt ở đây là <b>ảnh thật của tàu MESSENGER</b> (NASA) và bản đồ độ cao USGS. Sao Thủy tự quay đúng <b>3 vòng trong 2 năm</b> của nó (cộng hưởng 3:2), nên một ngày Mặt Trời dài 176 ngày Trái Đất; quỹ đạo lệch tâm khiến Mặt Trời trên bầu trời to lên rồi nhỏ lại 1,5 lần, và gần điểm cận nhật còn <b>dừng lại rồi đi ngược</b> vài ngày.',
  intro: 'Thời gian chạy theo lịch thật: Mặt Trời và chuyển động quay được tính từ quỹ đạo và mô hình quay IAU. Nhấn <b>R</b> để đứng trên mặt đất xem <b>Mặt Trời mọc hai lần</b>; <b>C</b> để bay tới lòng chảo Caloris lúc bình minh; đổi <b>Màu sắc</b> sang “Tăng cường” để thấy thành phần khoáng vật.',
  facts: [
    ['Bán kính trung bình', '2 439,4 km'], ['Khối lượng', '3,301×10²³ kg (khối lượng riêng 5,43 g/cm³)'],
    ['Trọng lực bề mặt', '3,70 m/s²'],
    ['Bán trục lớn · tâm sai', '0,3871 AU (57,9 triệu km) · 0,2056'],
    ['Cận nhật / viễn nhật', '46,0 / 69,8 triệu km'],
    ['Chu kỳ quỹ đạo', '87,969 ngày'],
    ['Ngày sao · ngày Mặt Trời', '58,646 ngày · 175,94 ngày (cộng hưởng 3:2)'],
    ['Độ nghiêng trục quay', '0,034°'],
    ['Nhiệt độ bề mặt', '≈ −180 °C (đêm) … 430 °C (trưa ở cận nhật)'],
    ['Suất phản chiếu hình học / Bond', '0,142 / 0,088'],
    ['Lòng chảo Caloris', 'Ø ≈ 1 550 km, 31,5°B 162,7°Đ, ~3,8 tỉ năm'],
    ['Ngoại quyển', 'Na, K, Ca, Mg, O, H, He — áp suất < 10⁻¹⁰ Pa; đuôi natri dài > 2 triệu km'],
    ['Dữ liệu', 'NASA/JHUAPL/Carnegie MESSENGER MDIS (bản đồ 3 màu & màu tăng cường 665 m/px), USGS DEM toàn cầu 665 m (2016); quỹ đạo JPL, mô hình quay IAU 2015'],
  ],
  unitKm: RKM,
  camera: { distance: 3.4, polar: 80, azimuth: 20, min: 1.01, max: 60, fov: 45 },
  exposure: 1.0,
  bloom: 0.3,
  params: [
    { key: 'color', label: 'Màu sắc', type: 'seg', options: [['nat', 'Tự nhiên'], ['enh', 'Tăng cường'], ['temp', 'Nhiệt độ']], default: 'nat', note: 'Tự nhiên: độ sáng MDIS với màu thật đã giảm bão hoà. Tăng cường: ảnh màu MESSENGER (thành phần chính + tỉ số 430/1000 nm) — xanh lam = vật chất phản xạ thấp, cam = đồng bằng núi lửa, xanh nhạt sáng = hố trẻ và “hollows”. Nhiệt độ: mô hình cân bằng bức xạ, 90 K (tím sẫm) … 720 K (vàng).' },
    { key: 'speed', label: 'Tốc độ thời gian', type: 'range', min: 0, max: 3, step: 0.01, default: 0.3, fmt: v => num(v, 2) + ' ngày/giây', note: 'Số ngày Trái Đất trôi qua mỗi giây mô phỏng (nhân với thanh tốc độ thời gian chung). Một ngày Mặt Trời trên Sao Thủy kéo dài 175,94 ngày.' },
    { key: 'frame', label: 'Giữ cố định', type: 'seg', options: [['sun', 'Mặt Trời'], ['stars', 'Các sao'], ['ground', 'Mặt đất']], default: 'sun', note: 'Mặt Trời: hành tinh quay chậm dưới đường phân chia ngày–đêm. Các sao: hệ quán tính — thấy cả Mặt Trời di chuyển. Mặt đất: camera bay cùng bề mặt.' },
    { key: 'exag', label: 'Phóng đại địa hình', type: 'range', min: 1, max: 12, step: 0.5, default: 2.5, fmt: v => '×' + num(v), note: 'Biên độ địa hình ~ 10 km = 0,4 % bán kính. ×1 = tỉ lệ thật.' },
    { key: 'exo', label: 'Đuôi natri', type: 'toggle', default: false, note: 'Nguyên tử natri bật khỏi bề mặt bị áp suất bức xạ Mặt Trời thổi thành đuôi dài hàng triệu km, phát sáng vàng 589 nm. Độ sáng phóng đại ~10⁵ lần. Đuôi mạnh nhất khi Sao Thủy chuyển động nhanh ra/vào Mặt Trời (hiệu ứng Swings), yếu ở cận nhật và viễn nhật.' },
    { key: 'track', label: 'Vết điểm dưới Mặt Trời', type: 'toggle', default: false, note: 'Vòng vàng: điểm có Mặt Trời ở thiên đỉnh; chấm xanh: vị trí của nó mỗi ngày trong 24 ngày qua.' },
  ],
  actions: [
    { label: 'Mặt Trời quay ngược', key: 'r', title: 'Đứng trên mặt đất ở “cực ấm”, xem Mặt Trời mọc, lặn rồi mọc lại (bấm lần nữa để quay lại)', run: ctx => ctx.scene._mercury?.reverse(ctx) },
    { label: 'Caloris', key: 'c', title: 'Bay tới lòng chảo Caloris lúc bình minh (bấm lần nữa để quay lại)', run: ctx => ctx.scene._mercury?.caloris(ctx) },
    { label: 'Cận nhật', key: 'p', title: 'Nhảy tới điểm cận nhật kế tiếp', run: ctx => ctx.scene._mercury?.perihelion(ctx) },
  ],
  create(ctx) {
    const Rr = ctx.renderer, caps = Rr.capabilities, aniso = Math.min(8, caps.getMaxAnisotropy());
    const grey = () => solidTex(new Uint8Array([100, 100, 100, 255]), THREE.RGBAFormat);
    const flat = () => { const t = new THREE.DataTexture(new Uint16Array([0]), 1, 1, THREE.RedFormat, THREE.HalfFloatType); t.needsUpdate = true; return t; };
    const track = Array.from({ length: 24 }, () => new THREE.Vector3());
    const U = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) }, uTailDir: { value: new THREE.Vector3(-1, 0, 0) },
      uRot: { value: 0 }, uSkyRot: { value: 0 }, uSunAng: { value: 0.012 }, uFlux: { value: 1 },
      uExag: { value: 2.5 }, uHMin: { value: -11 }, uHMax: { value: 9 }, uTexOK: { value: 0 },
      uTexel: { value: 2 * Math.PI / 4096 }, uLumTexel: { value: 2 * Math.PI / 4096 },
      uMode: { value: 0 }, uTss: { value: 650 }, uAutoExp: { value: 1 }, uExo: { value: 0 }, uExoK: { value: 1 }, uTrackN: { value: 0 },
      uTrack: { value: track },
      uLum: { value: grey() }, uChroma: { value: grey() }, uEnh: { value: grey() }, uElev: { value: flat() },
    };
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    let dem = null;
    const ready = Promise.all([
      retry(loadImageTex, URL_OF('mercury_lum_4k.jpg'), { red: true, aniso }),
      retry(loadImageTex, URL_OF('mercury_md3_1k.jpg'), { aniso: 2 }),
      retry(loadImageTex, URL_OF('mercury_enh_2k.jpg'), { srgb: false, aniso: 4 }),
      loadDEM(URL_OF('mercury_dem_2k.webp'), URL_OF('mercury_dem_detail_4k.jpg')).catch(() => new Promise(r => setTimeout(r, 500)).then(() => loadDEM(URL_OF('mercury_dem_2k.webp') + '?r=1', URL_OF('mercury_dem_detail_4k.jpg') + '?r=1'))),
    ]).then(([lum, md3, enh, d]) => {
      U.uLum.value = lum; U.uChroma.value = md3; U.uEnh.value = enh; U.uElev.value = d.tex;
      U.uTexel.value = 2 * Math.PI / d.tex.image.width; U.uLumTexel.value = 2 * Math.PI / lum.image.width;
      U.uHMin.value = d.min; U.uHMax.value = d.max;
      dem = d.cpu;
      for (const t of [lum, md3, enh, d.tex]) Rr.initTexture(t);
      U.uTexOK.value = 1;
    });
    const elevKm = (lat, lon) => dem ? sampleDEM(dem, lat, lon) : 0;

    /* ---- time: a real calendar date that advances at `speed` days per simulation second -------- */
    const md = QS.get('mercDate');
    const jd0 = jdOf(md ? Date.parse(md) : SNAP ? Date.UTC(2026, 9, 18) : Date.now());
    const T = { days: 0, lastSim: SNAP ? 0 : null, geo: mercuryGeom(jd0), frame: 'sun', psi: 0, beta: 0, jd: jd0 };
    // frame state: ψ = rotation of the world frame against the inertial one; β = W − ψ (body → world)
    function frameAngles(g) {
      const W = g.W * D2R, thSun = (g.lon * D2R) + W;      // Sun's inertial azimuth
      // the first frame always starts from the Sun-fixed orientation (the default view), whatever the mode
      if (T.frame === 'sun' || !T.init) { T.psi = thSun - SUN_AZ; T.init = true; }
      else if (T.frame === 'ground') T.psi = W - T.beta;
      T.beta = W - T.psi;
    }
    // world ↔ body
    const toWorld = (v, beta = T.beta) => v.clone().applyAxisAngle(Y, beta);
    const toBody = (v, beta = T.beta) => v.clone().applyAxisAngle(Y, -beta);

    /* ---- camera shots (Caloris, double sunrise): tween in the world frame with the ground frozen -- */
    const def = ctx.scene;
    const shot = { on: null, tw: null, pending: QS.get('mercShot') };
    const V = () => new THREE.Vector3();
    function setFrame(c, f) {
      // the current angles carry over (frameAngles), so switching frames never jumps
      T.frame = f; c.params.frame = f;
    }
    function jumpTo(c, jd) { T.days = jd - jd0; update(0, c, true); }
    function tweenTo(c, pos, tgt, up, dur, onDone) {
      shot.tw = { t0: performance.now(), dur: dur * 1000, onDone,
        a: { pos: c.camera.position.clone(), tgt: c.controls.target.clone(), up: c.camera.up.clone() }, b: { pos, tgt, up } };
    }
    function applyCam(c, pos, tgt, up) {
      c.camera.position.copy(pos); c.controls.target.copy(tgt); c.camera.up.copy(up).normalize(); c.camera.lookAt(tgt);
    }
    function leave(c, instant, exiting) {
      const cam = def.camera;
      const pos = new THREE.Vector3().setFromSphericalCoords(cam.distance, cam.polar * D2R, cam.azimuth * D2R);
      const done = () => { shot.on = null; c.controls.minDistance = cam.min; c.controls.autoRotateSpeed = shot.rotSpeed ?? 0.18; c.camera.up.set(0, 1, 0); c.controls.target.set(0, 0, 0); };
      if (shot.on === 'reverse' && !exiting) c.setFov?.(cam.fov);     // back from the telephoto view
      if (instant) { applyCam(c, pos, V(), Y); done(); return; }
      setFrame(c, 'sun'); c.params.speed = 0.3; c.params.exag = 2.5;
      tweenTo(c, pos, V(), Y.clone(), 3.5, done);
    }
    function enterShot(c, name) {
      shot.on = name; c.controls.minDistance = 0.0005;
      if (shot.rotSpeed === undefined) shot.rotSpeed = c.controls.autoRotateSpeed;
      c.controls.autoRotateSpeed = 0;           // no auto-orbit while standing on the ground
    }
    // when is the Sun at elevation h (deg) in the morning at east longitude lon (equator-ish)?
    function findSunElev(jdStart, lon, h) {
      const target = lon + 90 - h;                // sub-solar longitude giving that elevation
      let prev = wrap180(mercuryGeom(jdStart).lon - target);
      for (let k = 0.25; k < 200; k += 0.25) {
        const cur = wrap180(mercuryGeom(jdStart + k).lon - target);
        if (prev > 0 && cur <= 0 && prev - cur < 90) {
          let lo = k - 0.25, hi = k;
          for (let i = 0; i < 30; i++) { const m = 0.5 * (lo + hi); if (wrap180(mercuryGeom(jdStart + m).lon - target) > 0) lo = m; else hi = m; }
          return jdStart + hi;
        }
        prev = cur;
      }
      return jdStart;
    }
    function nextPerihelion(jd) {
      const g = mercuryGeom(jd);
      return jd + ((2 * Math.PI - g.M) % (2 * Math.PI)) / (2 * Math.PI) * PERIOD;
    }
    function goCaloris(c, instant) {
      const p = c.params;
      // morning at Caloris: the Sun ~10° above the eastern horizon, long shadows to the west
      const jd = findSunElev(T.jd, CALORIS.lon, 10);
      jumpTo(c, jd);
      setFrame(c, 'ground'); update(0, c, true);
      p.speed = 0.02; p.exag = 3;
      enterShot(c, 'caloris');
      // from ~2 400 km up, south of the basin, looking north: sunlight from the right (east)
      const tgt = toWorld(llB(CALORIS.lat, CALORIS.lon));
      const pos = toWorld(llB(CALORIS.lat - 14, CALORIS.lon + 2).multiplyScalar(1 + 2400 / RKM));
      const up = pos.clone().normalize();
      if (instant) applyCam(c, pos, tgt, up); else tweenTo(c, pos, tgt, up, 4.5);
      c.toast?.(`Bình minh trên <b>lòng chảo Caloris</b> (Ø ≈ 1 550 km), nhìn từ độ cao ~2 400 km phía nam. Mặt Trời thấp ở phía đông nên vành núi và các nếp gãy đổ bóng dài về phía tây. Ở đây buổi sáng kéo dài ~44 ngày Trái Đất. Caloris là di tích của một va chạm khổng lồ ~3,8 tỉ năm trước; nền lòng chảo là dung nham núi lửa sau đó (màu cam ở chế độ <b>Tăng cường</b>). Địa hình phóng đại ×3.`, 11000);
    }
    function goReverse(c, instant) {
      const p = c.params;
      // the next perihelion (at least 3 days ahead, so the whole loop is still to come)
      const jdP = nextPerihelion(T.jd + 3);
      // the Sun's retrograde loop around this perihelion: local min / max of the sub-solar longitude
      let lmin = 1e9, tmin = jdP;
      const lon0 = mercuryGeom(jdP).lon;
      for (let k = -8; k <= 0; k += 0.05) { const l = lon0 + wrap180(mercuryGeom(jdP + k).lon - lon0); if (l < lmin) { lmin = l; tmin = jdP + k; } }
      // Observer ~90° west of the sub-solar point: there the Sun is on the eastern horizon. Among
      // latitudes within ±8° pick the most open eastern horizon (from the DEM, true scale, curvature
      // included), then place the observer so the Sun's centre culminates 0,25° above that horizon at
      // the start of the loop.
      const horizon = (lat, lon) => {
        const e0 = elevKm(lat, lon) + 0.05, cl = Math.cos(lat * D2R);
        let hz = -1e9;
        for (let s = 1; s < 500; s *= 1.06) {
          const dz = elevKm(lat, lon + s / (RKM * cl * D2R)) - e0;
          hz = Math.max(hz, (Math.atan2(dz, s) - s / RKM / 2) / D2R);
        }
        return hz;
      };
      // candidates: latitudes within ±8°, longitudes near the nominal spot; keep the one whose own
      // horizon is consistent with its longitude, preferring an open (low) horizon
      let obs = wrap180(lmin - 90), obsLat = 0, best = 1e9;
      for (let la = -8; la <= 8; la += 0.5) {
        for (let dl = -1.5; dl <= 3; dl += 0.05) {
          const lo = wrap180(lmin - 90 + dl), hz = horizon(la, lo);
          const err = Math.abs(dl - (hz + 0.25)) + 0.2 * Math.abs(hz);
          if (err < best) { best = err; obsLat = la; obs = lo; }
        }
      }
      jumpTo(c, tmin - 4.5);
      setFrame(c, 'ground'); update(0, c, true);
      p.speed = 0.25; p.exag = 1; p.track = false; U.uExag.value = 1;
      enterShot(c, 'reverse');
      c.setFov?.(22);                            // telephoto: the 1,7° solar disc spans ~80 px at 1080p
      const eye = llB(obsLat, obs).multiplyScalar(1 + (elevKm(obsLat, obs) + 0.05) / RKM);
      // look east along the horizon, ~2° up
      const east = new THREE.Vector3(Math.cos(obs * D2R), 0, -Math.sin(obs * D2R));   // due east (the Sun rises there: declination ≈ 0)
      const look = east.clone().multiplyScalar(Math.cos(2 * D2R)).addScaledVector(eye.clone().normalize(), Math.sin(2 * D2R));
      const pos = toWorld(eye), tgt = toWorld(eye.clone().addScaledVector(look, 0.004)), up = toWorld(eye.clone().normalize());
      if (instant) applyCam(c, pos, tgt, up); else tweenTo(c, pos, tgt, up, 4.5);
      c.toast?.(`Bạn đang đứng ở <b>${latStr(obsLat)} ${lonStr(obs)}</b> (gần “cực ấm”), nhìn về phía đông, vài ngày trước cận nhật. Gần cận nhật, Sao Thủy chạy trên quỹ đạo nhanh hơn nó tự quay (6,35°/ngày so với 6,14°/ngày), nên trong ~8 ngày Mặt Trời <b>đi ngược</b> trên bầu trời: nó ló lên khỏi chân trời, lặn xuống lại, rồi mới mọc hẳn — <b>bình minh kép</b>. Đĩa Mặt Trời ở đây rộng 1,7°, gấp 3,2 lần nhìn từ Trái Đất.`, 13000);
    }
    def._mercury = {
      caloris: c => { if (shot.tw) return; if (shot.on) leave(c); else goCaloris(c, false); },
      reverse: c => { if (shot.tw) return; if (shot.on) leave(c); else goReverse(c, false); },
      perihelion: c => {
        if (shot.on) leave(c, true);
        jumpTo(c, nextPerihelion(T.jd + 1));
        setFrame(c, 'sun'); c.params.speed = 0.3;
        const g = T.geo;
        c.toast?.(`Cận nhật: Sao Thủy chỉ cách Mặt Trời <b>46,0 triệu km</b>; ánh nắng mạnh gấp <b>2,3 lần</b> so với ở viễn nhật. Mặt Trời đứng trên đỉnh đầu ở kinh độ <b>${Math.abs(wrap180(g.lon)) < 90 ? '0°' : '180°'}</b> — một trong hai “cực nóng”; cộng hưởng 3:2 đưa đúng hai kinh độ này về phía Mặt Trời ở mọi lần cận nhật. Trưa ở đây đạt ~430 °C.`, 10000);
      },
    };

    /* ---- per-frame state ------------------------------------------------------------------------ */
    const Lw = V();
    function update(dt, c, silent) {
      const p = c.params;
      if (!silent) {
        if (T.lastSim === null) T.lastSim = c.time.sim;
        T.days += (c.time.sim - T.lastSim) * p.speed;
        T.lastSim = c.time.sim;
        if (p.frame !== T.frame) T.frame = p.frame;
      }
      T.jd = jd0 + T.days;
      const g = T.geo = mercuryGeom(T.jd);
      frameAngles(g);
      // Sun in the world frame
      Lw.copy(llB(g.lat, g.lon)).applyAxisAngle(Y, T.beta);
      U.uSunDir.value.copy(Lw);
      U.uRot.value = -T.beta;
      U.uSkyRot.value = T.psi;
      U.uSunAng.value = Math.asin(695700 / (g.r * AU_KM));
      U.uFlux.value = Math.pow(0.387098 / g.r, 2);
      U.uTss.value = 700 * Math.sqrt(0.3075 / g.r);
      // sodium tail: anti-sunward, aberrated slightly behind the orbital motion; Na D g-factor rises as the
      // Doppler shift (heliocentric radial velocity) moves the lines out of the solar Fraunhofer cores
      const vt = new THREE.Vector3(Lw.z, 0, -Lw.x).normalize();          // dL/dθ: direction the Sun moves (CCW)
      U.uTailDir.value.copy(Lw).negate().addScaledVector(vt, 0.12).normalize();
      const gf = 0.15 + 0.85 * (1 - Math.exp(-Math.pow(g.vr / 6, 2)));
      U.uExoK.value = gf * Math.pow(0.387 / g.r, 2) * 1.4;
      U.uExo.value = p.exo ? 1 : 0;
      U.uMode.value = p.color === 'enh' ? 1 : p.color === 'temp' ? 2 : 0;
      // camera auto-exposure (half-compensating, like a spacecraft camera): Mercury's disc is ~10×
      // brighter at full phase than at 70° (Mallama 2002 phase curve) and 2,3× brighter at perihelion
      // than at aphelion; we keep part of both differences visible
      if (!silent) {
        const cp = c.camera.position, dist = cp.length();
        const alpha = Math.acos(Math.max(-1, Math.min(1, cp.dot(Lw) / dist))) / D2R;
        const ratio = Math.pow(10, -0.4 * (mallamaV(alpha) - mallamaV(70)));
        const far = THREE.MathUtils.smoothstep(dist, 1.3, 2.2);
        U.uAutoExp.value = Math.pow(ratio, -0.55 * far) * Math.pow(U.uFlux.value, -0.4);
      }
      // relief exaggeration eases toward the slider
      U.uExag.value += (p.exag - U.uExag.value) * (SNAP ? 1 : Math.min(1, (dt || 1) * 4));
      if (Math.abs(p.exag - U.uExag.value) < 0.01) U.uExag.value = p.exag;
      // track of the sub-solar point: one dot per Earth day over the last 24 days
      if (p.track) {
        for (let i = 0; i < 24; i++) { const gi = mercuryGeom(T.jd - (i + 1)); track[i].copy(llB(gi.lat, gi.lon)); }
        U.uTrackN.value = 24;
      } else U.uTrackN.value = 0;
      if (silent) return;

      // deep-link / snapshot hook (?mercShot=caloris|reverse|perihelion[&mercDt=days]) once the data are in
      if (shot.pending && U.uTexOK.value > 0) {
        const s = shot.pending; shot.pending = null;
        if (s === 'caloris') goCaloris(c, true); else if (s === 'reverse') goReverse(c, true); else if (s === 'perihelion') def._mercury.perihelion(c);
        T.days += +(QS.get('mercDt') || 0);        // snapshot hook: days after the shot starts
        U.uExag.value = c.params.exag;
      }
      // camera tween
      if (shot.tw) {
        const tw = shot.tw, k0 = Math.min(1, (performance.now() - tw.t0) / tw.dur), e = k0 * k0 * (3 - 2 * k0);
        const ra = tw.a.pos.length(), rb = tw.b.pos.length();
        const dir = tw.a.pos.clone().normalize().lerp(tw.b.pos.clone().normalize(), e);
        if (dir.lengthSq() < 1e-6) dir.copy(tw.b.pos).normalize();
        const r = Math.exp(Math.log(ra) * (1 - e) + Math.log(rb) * e) * (1 + 0.35 * Math.sin(Math.PI * e) * Math.min(1, Math.abs(ra - rb)));
        applyCam(c, dir.normalize().multiplyScalar(r), tw.a.tgt.clone().lerp(tw.b.tgt, e), tw.a.up.clone().lerp(tw.b.up, e));
        if (k0 >= 1) { shot.tw = null; tw.onDone?.(); }
      }
      // keep the camera above the ground (DEM under the camera × exaggeration + margin)
      if (dem) {
        const b = toBody(c.camera.position), r = b.length();
        const lat = Math.asin(b.y / r) / D2R, lon = Math.atan2(b.x, b.z) / D2R;
        const top = 1 + (elevKm(lat, lon) * U.uExag.value + 0.03) / RKM;
        if (r < top) c.camera.position.setLength(top);
      }
    }

    /* ---- HUD --------------------------------------------------------------------------------- */
    function underCamera(c) {
      const b = toBody(c.camera.position), r = b.length();
      return { lat: Math.asin(b.y / r) / D2R, lon: Math.atan2(b.x, b.z) / D2R, n: b.normalize() };
    }
    function localSun(n, g) {
      const Lb = llB(g.lat, g.lon);
      const mu0 = n.dot(Lb);
      const H = wrap180((Math.atan2(n.x, n.z) - Math.atan2(Lb.x, Lb.z)) / D2R);   // hour angle, + = afternoon
      return { mu0, H, elev: Math.asin(Math.max(-1, Math.min(1, mu0))) / D2R };
    }

    return {
      material,
      ready,
      update,
      exit(c) {
        // (the host restores the next scene's fov itself)
        if (shot.on) leave(c, true, true);
        shot.tw = null;
      },
      hud(c) {
        const g = T.geo;
        const g2 = mercuryGeom(T.jd + 0.05);
        const rate = wrap180(g2.lon - g.lon) / 0.05;                 // °/day, negative = westward (normal)
        const motion = Math.abs(rate) < 0.04 ? '<b>gần như đứng yên</b>' : rate < 0 ? `đi về phía tây ${num(-rate, 2)}°/ngày` : `<b style="color:#ffb347">ĐI NGƯỢC về phía đông</b> ${num(rate, 2)}°/ngày`;
        const u = underCamera(c), s = localSun(u.n, g);
        const lat = u.lat, lon = u.lon;
        const T_K = surfTempK(s.mu0, s.H, lat, g.r);
        const hour = 12 + s.H / 15;
        const toSet = s.H < 90 && s.H > -90 ? (90 - s.H) / 360 * SOLAR_DAY : null;
        const sunDiam = 2 * Math.asin(695700 / (g.r * AU_KM)) / D2R;
        const alt = (c.camera.position.length() - 1 - elevKm(lat, lon) * U.uExag.value / RKM) * RKM;
        const dayInYear = g.M / (2 * Math.PI) * PERIOD;
        const wall = (c.bhGlobals?.timeScale ?? 6) * c.params.speed;
        return [
          ['Ngày mô phỏng', dateStr(T.jd)],
          ['Khoảng cách tới Mặt Trời', `${num(g.r, 4)} AU · ${num(g.r * AU_KM / 1e6, 1)} triệu km`],
          ['Vị trí trên quỹ đạo', `ngày ${num(dayInYear, 1)} / 87,97 sau cận nhật`],
          ['Đường kính biểu kiến Mặt Trời', `${num(sunDiam, 2)}° (gấp ${num(sunDiam / 0.533, 1)} lần nhìn từ Trái Đất)`],
          ['Điểm dưới Mặt Trời', `${lonStr(g.lon)} · ${motion}`],
          ['Điểm dưới camera', `${latStr(lat)} ${lonStr(lon)}` + (dem ? ` · cao ${Math.round(elevKm(lat, lon) * 1000).toLocaleString('vi-VN')} m` : '')],
          ['Giờ Mặt Trời tại đó', `${hhmm(((hour % 24) + 24) % 24)} · Mặt Trời cao ${num(s.elev, 1)}°` + (toSet !== null ? ` · lặn sau ~${num(toSet, 0)} ngày` : '')],
          ['Nhiệt độ bề mặt', `${Math.round(T_K - 273.15)} °C (${Math.round(T_K)} K)`],
          ['Độ cao camera', c.fmt.dist(Math.max(alt, 0) * 1000)],
          ['Nhịp thời gian', `1 giây thực = ${num(wall, 2)} ngày Trái Đất`],
        ];
      },
      onClick(info, c) {
        const t = hitSphere(info.origin, info.dir);
        if (t === null) return;
        const pw = info.origin.clone().addScaledVector(info.dir, t);
        const n = toBody(pw).normalize();
        const lat = Math.asin(n.y) / D2R, lon = Math.atan2(n.x, n.z) / D2R;
        const s = localSun(n, T.geo), TK = surfTempK(s.mu0, s.H, lat, T.geo.r);
        const e = dem ? ` · độ cao <b>${Math.round(elevKm(lat, lon) * 1000).toLocaleString('vi-VN')} m</b>` : '';
        c.toast(`Vĩ độ <b>${latStr(lat)}</b>, kinh độ <b>${lonStr(lon)}</b>${e} · giờ Mặt Trời <b>${hhmm((((12 + s.H / 15) % 24) + 24) % 24)}</b> · Mặt Trời cao <b>${num(s.elev, 1)}°</b> · nhiệt độ ≈ <b>${Math.round(TK - 273.15)} °C</b>.`);
      },
      onParam(key, v, c) {
        if (key === 'frame') T.frame = v;
        if (key === 'speed') { T.lastSim = c.time.sim; }
      },
    };
  },
}];
