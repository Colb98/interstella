// Sao neutron — a radio pulsar (Vela) and a magnetar (SGR 1806−20), rendered by tracing photons
// backwards along exact Schwarzschild null geodesics (same integrator as js/glsl/blackhole.js).
//
// Units: r_s = 2GM/c² = 1 (so M = ½, c = 1). For M = 1.4 M☉, r_s = 4.134 km and the 12 km star
// sits at R ≈ 2.90 r_s — outside the photon sphere (1.5 r_s) but compact enough that light leaving
// the far hemisphere is bent round to the observer: more than half the surface is visible.
//
// Everything that should be lensed (surface, lighthouse beams, dipole field lines, the magnetar's
// trapped fireball) is evaluated inside the geodesic march.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';
import { t, dec, locale } from '../i18n.js';

const M_SUN = 1.989e30, G_N = 6.674e-11, C_L = 2.998e8;
const MASS = 1.4;
const RS_KM = 2.953 * MASS;                  // Schwarzschild radius of 1.4 M☉ in km (4.134)
const BEAM_LEN = 26;                         // drawn extent of the radio beams, r_s

const FRAG = COMMON_GLSL + /* glsl */`
uniform float uR;                    // stellar radius in r_s
uniform float uGR, uSplit, uGrid;
uniform mat3 uBody;                  // body-fixed frame -> world (rotation about the spin axis y)
uniform vec3 uMag, uMagE1, uMagE2;   // magnetic axis and two perpendicular directions (world)
uniform float uTloc, uTcap, uTinf, uCap, uChi, uCrust;
uniform float uBeam, uOpenK, uBeamLen;
uniform vec3 uBeamCol;
uniform float uLines, uNphi, uTwist, uFlow;
uniform vec4 uLs;                    // field-line shells r = L sin²θ (L in r_s)
uniform vec3 uLineCol;
uniform float uQuake, uQuakeSpread;
uniform vec3 uQuakeDir;              // body frame
uniform vec4 uFire;                  // world position (co-rotating), intensity
uniform float uFireR, uFlash;
uniform vec3 uFireCol;
uniform vec3 uCamDirN;               // unit vector centre -> camera
uniform float uCosFlat;              // R / D: cosine of the flat-space visibility limit

vec3 staticToCoord(vec3 x, vec3 d){
  float r = length(x), lapse = sqrt(max(1.0 - 1.0 / r, 1e-4));
  vec3 rh = x / r; return d + rh * dot(d, rh) * (lapse - 1.0);
}
vec3 coordToStatic(vec3 x, vec3 v){
  float r = length(x), lapse = sqrt(max(1.0 - 1.0 / r, 1e-4));
  vec3 rh = x / r; return v + rh * dot(v, rh) * (1.0 / lapse - 1.0);
}
// q = sin²θ_m / r = 1/L : the dipole field-line label (smooth everywhere, 0 on the magnetic axis)
float qShell(vec3 x){ float r = length(x); float c = dot(x, uMag) / r; return (1.0 - c * c) / r; }

// Radio emission from the open-field-line bundle above each magnetic pole. The bundle's
// half-opening angle follows the last open dipole line, sin²θ = r·uOpenK (θ ∝ √r), with a
// hollow cone + core profile (Rankin's empirical beam model). Emission is beamed along the local
// field direction, so it only flashes when the beam is pointed at the camera.
vec3 beamEm(vec3 p, vec3 vdir){
  float r = length(p);
  if (r > uBeamLen || r < uR) return vec3(0.0);
  vec3 ph = p / r;
  float cs = dot(ph, uMag), c = abs(cs);
  float s2 = 1.0 - c * c;
  float so2 = min(uOpenK * r, 0.3);
  if (s2 > so2 * 3.0) return vec3(0.0);
  float th = asin(sqrt(s2)), tho = asin(sqrt(so2));
  float x = th / tho;
  float prof = exp(-pow((x - 0.72) / 0.26, 2.0)) + 0.6 * exp(-x * x / 0.08);
  float rad = smoothstep(uR, uR + 0.8, r) * (uR / r) * (1.0 - smoothstep(uBeamLen * 0.4, uBeamLen, r));
  float dens = prof * rad;
  if (dens < 1e-3) return vec3(0.0);
  // streaming plasma: structure flowing outward along the bundle
  float az = atan(dot(ph, uMagE2), dot(ph, uMagE1));
  dens *= 0.7 + 0.45 * snoise(vec3(x * 2.5 + cs * 7.0, log(r) * 5.0 - uTime * 0.6, az * 0.8));
  vec3 B = normalize(3.0 * cs * ph - uMag) * sign(cs);
  float fw = max(dot(-vdir, B), 0.0);
  fw = pow(fw, 160.0);
  return uBeamCol * dens * (0.12 + 1.6 * fw);
}

// photosphere: anisotropic heat conduction in the magnetised crust (Greenstein & Hartke 1983:
// heat flows along B, so T⁴ ∝ cos²Θ + χ sin²Θ, Θ = angle between B and the normal) + hot polar
// caps heated by returning pair plasma + crust inhomogeneity + starquake cracks.
vec3 surface(vec3 p, vec3 v, bool gr, float gS, out float Tobs){
  vec3 n = p / uR;
  vec3 pb = transpose(uBody) * n;
  float c = dot(n, uMag);
  float cos2 = 4.0 * c * c / (1.0 + 3.0 * c * c);
  float T = uTloc * pow(cos2 + uChi * (1.0 - cos2), 0.25);
  float ac = 1.0 - abs(c);                       // ≈ θ²/2 from the nearest magnetic pole
  float cap = exp(-2.0 * ac / (uCap * uCap));
  float halo = exp(-2.0 * ac / (9.0 * uCap * uCap));
  T = mix(T, uTcap, cap) * (1.0 + 0.15 * halo);
  // crust: a solid ion lattice broken into plates; small-scale crustal fields make the temperature
  // patchy, with slightly warmer plate boundaries (where starquake cracks open)
  vec3 wp = pb * 7.0 + 0.35 * vec3(snoise(pb * 3.0 + 1.7), snoise(pb * 3.0 + 9.2), snoise(pb * 3.0 - 4.1));
  vec2 w = worley(wp);
  float edge = 1.0 - smoothstep(0.0, 0.12, w.y - w.x);
  float cr = fbm(pb * 3.0 + 2.0, 3);
  T *= 1.0 + uCrust * (cr + 0.5 * edge + 0.35 * (w.x - 0.5));
  if (uQuake > 0.002){
    float reg = exp(-(1.0 - dot(pb, uQuakeDir)) * uQuakeSpread);
    if (reg > 0.01){
      float crack = 1.0 - smoothstep(0.0, 0.05, w.y - w.x);
      vec2 w2 = worley(wp * 2.7 + 5.0);
      crack = max(crack, 0.55 * (1.0 - smoothstep(0.0, 0.04, w2.y - w2.x)));
      T *= 1.0 + uQuake * reg * (0.2 + 1.7 * crack);
    }
  }
  vec3 vs = gr ? coordToStatic(p, v) : v;
  float mu = abs(dot(normalize(vs), n));         // cosine of the emission angle in the local static frame
  Tobs = T * gS;
  float I = pow(Tobs / uTinf, 4.0) * (0.35 + 0.65 * mu) * 0.34;   // mild limb darkening (H atmosphere)
  I = I < 3.0 ? I : 3.0 + 3.0 * log(I / 3.0);
  vec3 col = bb(Tobs) * I;
  if (uGrid > 0.5){
    // spin-frame latitude / longitude every 15°, magnetic poles ringed, far hemisphere tinted amber
    float lat = asin(clamp(pb.y, -1.0, 1.0)), lon = atan(pb.z, pb.x);
    float s = PI / 12.0;
    float gl = max(gridLine(lat / s, 0.04), gridLine(lon / s, 0.04 / max(cos(lat), 0.2)));
    float far = smoothstep(uCosFlat + 0.012, uCosFlat - 0.012, dot(n, uCamDirN));   // hidden in flat space
    vec3 lc = mix(vec3(0.35, 0.75, 1.0), vec3(1.0, 0.6, 0.15), far);
    col = mix(col, vec3(1.0, 0.45, 0.08) * (0.25 + 0.6 * dot(col, vec3(0.33))), far * 0.75);
    col = mix(col, col * 0.35 + lc * 0.9, gl * 0.85);
    float ring = 1.0 - smoothstep(0.0, 0.009, abs(ac - 0.0996));   // 26° from each magnetic pole
    col += vec3(1.0, 0.25, 0.6) * ring * 0.7;
  }
  return col;
}

void main(){
  vec3 d = camRay();
  bool gr = uGR > 0.5;
  float sx = gl_FragCoord.x / uRes.x;
  if (uSplit > 0.5) gr = sx >= 0.5;                 // left: straight rays, right: Schwarzschild
  vec3 x = uCamPos; float r = length(x);
  float lapseC = sqrt(max(1.0 - 1.0 / r, 1e-4));
  float gS = sqrt(1.0 - 1.0 / uR) / lapseC;        // surface -> static camera frequency ratio (kept in both modes)
  vec3 v = gr ? staticToCoord(x, d) : d;
  vec3 hv = cross(x, v); float h2 = dot(hv, hv);
  float rFar = max(r + 1.0, uBeamLen + 0.5);
  vec3 col = vec3(0.0);
  float rmin = r;
  float qp = qShell(x);
  bool hit = false;
  vec3 hitV = v;

  for (int i = 0; i < MAX_STEPS; i++){
    if (r > rFar && dot(x, v) > 0.0) break;
    float dl = clamp(STEP_K * r, 0.01, 3.0);
    float dt = dl / length(v);
    vec3 xp = x;
    if (gr){
      // Schwarzschild null geodesic, x'' = -(3/2) h² x / r⁵ (velocity Verlet)
      vec3 a = -1.5 * h2 * x / (r * r * r * r * r);
      v += a * (0.5 * dt); x += v * dt; r = length(x);
      a = -1.5 * h2 * x / (r * r * r * r * r);
      v += a * (0.5 * dt);
    } else { x += v * dt; r = length(x); }
    float seg = dl;
    if (r < uR){
      vec3 dd = x - xp; float A = dot(dd, dd), Bq = dot(xp, dd), Cq = dot(xp, xp) - uR * uR;
      float s = clamp((-Bq - sqrt(max(Bq * Bq - A * Cq, 0.0))) / A, 0.0, 1.0);
      x = xp + dd * s; r = uR; seg = dl * s; hit = true; hitV = v;
    }
    rmin = min(rmin, r);
    vec3 vn = normalize(v);
    if (uBeam > 0.0) col += beamEm(0.5 * (xp + x), vn) * seg * uBeam;
    if (uFire.w > 0.0){
      vec3 q = 0.5 * (xp + x) - uFire.xyz; float d2 = dot(q, q), s2 = uFireR * uFireR;
      if (d2 < 9.0 * s2){
        // optically thick e± pair-photon plasma (T ~ 10⁹ K) held by closed field lines: white core,
        // violet halo (false colour for the non-thermal scattering envelope)
        float nz = snoise(q * (2.0 / uFireR) + vec3(0.0, uWall * 1.1, 0.0)) + 0.5 * snoise(q * (5.0 / uFireR) - uWall * 0.7);
        float core = exp(-d2 / (0.2 * s2));
        col += (uFireCol * exp(-d2 / s2) * max(0.55 + 0.45 * nz, 0.0) + vec3(0.85, 0.9, 1.0) * core * 1.6) * uFire.w * seg;
      }
    }
    // dipole field lines: ribbons on the shells r = L sin²θ, found by sign changes of q - 1/L
    if (uLines > 0.0){
      float q = qShell(x);
      vec3 xm = 0.5 * (xp + x); float qm = qShell(xm);   // two sub-intervals: catches grazing double crossings
      for (int k = 0; k < 8; k++){
        float qk = 1.0 / uLs[k & 3];
        float qa = k < 4 ? qp : qm, qb = k < 4 ? qm : q;
        vec3 xa = k < 4 ? xp : xm, xb = k < 4 ? xm : x;
        if ((qa - qk) * (qb - qk) < 0.0){
          float f = (qa - qk) / (qa - qb);
          vec3 pc = mix(xa, xb, f); float rc = length(pc);
          if (rc < uR * 1.002) continue;
          vec3 ph = pc / rc; float c = dot(ph, uMag), st = sqrt(max(1.0 - c * c, 1e-6));
          float phi = atan(dot(ph, uMagE2), dot(ph, uMagE1)) - uTwist * c + float(k & 3) * 0.37;
          float cell = phi * uNphi / TAU;
          float dphi = abs(fract(cell + 0.5) - 0.5) * TAU / uNphi;
          float dist = rc * st * dphi;
          float w = 0.022 + 0.009 * rc;
          float gl = exp(-dist * dist / (w * w));
          if (gl < 0.01) continue;
          vec3 th = (c * ph - uMag) / st;
          vec3 nrm = normalize(st * ph - 2.0 * c * th);
          float inc = min(1.0 / max(abs(dot(nrm, vn)), 1e-3), 4.0);
          float flow = 1.0 + uFlow * (0.6 * sin(c * 16.0 - uTime * 1.6 + floor(cell) * 2.1) - 0.1);
          float foot = 1.0 + 1.2 * exp(-(rc - uR) * 3.0);
          col += uLineCol * gl * inc * flow * foot * uLines * 0.11;
        }
      }
      qp = q;
    }
    if (hit) break;
  }

  if (hit){
    float To;
    col += surface(x, hitV, gr, gS, To);
  } else {
    // remaining weak-field bending beyond rFar: Δα = (r_s / b)(1 − cos), toward the centre
    vec3 vn = normalize(v);
    if (gr){
      vec3 xh = normalize(x);
      vec3 perp = x - vn * dot(x, vn); float b = length(perp);
      if (b > 1e-3){ float da = (1.0 - dot(xh, vn)) / b; vn = normalize(vn * cos(da) - perp / b * sin(da)); }
    }
    col += sky(vn, 1.0 / lapseC) * 0.8;
  }
  // giant-flare initial spike: a lensed glare whose brightness follows how close each ray came to the star
  if (uFlash > 0.0) col += vec3(0.85, 0.9, 1.0) * uFlash * (pow(uR / rmin, 3.0) * 4.0 + 0.15);
  if (uSplit > 0.5){
    float px = abs(gl_FragCoord.x - 0.5 * uRes.x);
    col = mix(col, vec3(0.9, 0.95, 1.0), (1.0 - smoothstep(0.5, 1.5, px)) * 0.6);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// ---------------------------------------------------------------------------------------------
// CPU physics helpers

const vn = (x, d = 2) => dec(x, d);

// Largest angle ψ (seen from the star's centre) between the camera direction and a surface point
// whose light still reaches a camera at distance D. GR: the limiting photon leaves the surface
// tangentially (b = R/√(1−1/R)) and sweeps φ = ∫ du / √(1/b² − u² + u³) from u = 1/D to 1/R.
// The substitution u = 1/R − s² removes the turning-point singularity. Flat space: arccos(R/D).
function psiMax(R, D, gr) {
  if (D <= R) return 0;
  if (!gr) return Math.acos(R / D);
  if (R <= 1.5) return Math.PI;
  const b = R / Math.sqrt(1 - 1 / R), u0 = 1 / R, S = Math.sqrt(u0 - 1 / D), N = 240;
  let sum = 0;
  for (let i = 0; i <= N; i++) {
    const s = S * i / N, u = u0 - s * s;
    const val = i === 0 ? 2 / Math.sqrt(2 * u0 - 3 * u0 * u0) : 2 * s / Math.sqrt(Math.max(1 / (b * b) - u * u + u * u * u, 1e-14));
    sum += val * (i === 0 || i === N ? 1 : (i % 2 ? 4 : 2));
  }
  return sum * S / N / 3;
}
const visFrac = psi => psi >= Math.PI ? 1 : (1 - Math.cos(psi)) / 2;

// one photon traced backwards on the CPU (for clicks)
function traceJS(origin, dir, R, gr) {
  const x = origin.clone(); let r = x.length();
  const lap = Math.sqrt(Math.max(1 - 1 / r, 1e-4)), rh = x.clone().divideScalar(r);
  const v = gr ? dir.clone().addScaledVector(rh, dir.dot(rh) * (lap - 1)) : dir.clone();
  const h2 = x.clone().cross(v).lengthSq(), rFar = Math.max(r + 1, 40);
  const a = new THREE.Vector3(), xp = new THREE.Vector3();
  for (let i = 0; i < 6000; i++) {
    if (r > rFar && x.dot(v) > 0) return null;
    const dl = Math.max(0.003, 0.008 * r), dt = dl / v.length();
    xp.copy(x);
    if (gr) {
      a.copy(x).multiplyScalar(-1.5 * h2 / r ** 5); v.addScaledVector(a, 0.5 * dt); x.addScaledVector(v, dt); r = x.length();
      a.copy(x).multiplyScalar(-1.5 * h2 / r ** 5); v.addScaledVector(a, 0.5 * dt);
    } else { x.addScaledVector(v, dt); r = x.length(); }
    if (r < R) {
      const dd = x.clone().sub(xp), A = dd.lengthSq(), B = xp.dot(dd), C = xp.lengthSq() - R * R;
      const s = Math.min(1, Math.max(0, (-B - Math.sqrt(Math.max(B * B - A * C, 0))) / A));
      return xp.addScaledVector(dd, s);
    }
  }
  return null;
}

// bulk properties of a 1.4 M☉ star of radius Rkm
function bulk(Rkm) {
  const M = MASS * M_SUN, R = Rkm * 1e3, rs = RS_KM / Rkm;
  return {
    g: G_N * M / (R * R) / Math.sqrt(1 - rs),          // proper surface gravity
    vesc: Math.sqrt(rs),                                // in units of c (exact in Schwarzschild)
    rho: M / (4 / 3 * Math.PI * R ** 3),
    z: 1 / Math.sqrt(1 - rs) - 1,
  };
}

// ---------------------------------------------------------------------------------------------

const COMMON_FACTS = [
  ['Khối lượng', '≈ 1,4 M☉ (điển hình)'],
  ['Bán kính', '≈ 12 km = 2,9 r_s'],
  ['Hấp dẫn bề mặt', '≈ 1,6×10¹¹ g'],
  ['Vận tốc thoát', '≈ 0,59 c'],
  ['Mật độ', '≈ 4×10¹⁷ kg/m³'],
  ['Một thìa cà phê', 'nặng ~2 tỉ tấn'],
  ['Dịch đỏ bề mặt', 'z ≈ 0,24'],
];
const COMMON_FACTS_EN = [
  ['Mass', '≈ 1.4 M☉ (typical)'],
  ['Radius', '≈ 12 km = 2.9 r_s'],
  ['Surface gravity', '≈ 1.6×10¹¹ g'],
  ['Escape velocity', '≈ 0.59 c'],
  ['Density', '≈ 4×10¹⁷ kg/m³'],
  ['One teaspoon', 'weighs ~2 billion tonnes'],
  ['Surface redshift', 'z ≈ 0.24'],
];

function makeScene(cfg) {
  return {
    id: cfg.id, name: cfg.name, category: 'compact', order: cfg.order,
    subtitle: cfg.subtitle, swatch: cfg.swatch, blurb: cfg.blurb, intro: cfg.intro,
    facts: cfg.facts, en: cfg.en,
    unitKm: RS_KM,
    camera: { distance: 16, polar: cfg.camPolar, azimuth: 30, min: 3.25, max: 80, fov: 45 },
    exposure: cfg.exposure ?? 1.0,
    bloom: 0.7,
    params: cfg.params,
    actions: cfg.actions,
    create(ctx) { return createInstance(ctx, cfg); },
  };
}

function createInstance(ctx, cfg) {
  const q0 = ctx.quality;
  const uniforms = {
    ...ctx.shared,
    uR: { value: 2.9 }, uGR: { value: 1 }, uSplit: { value: 0 }, uGrid: { value: 0 },
    uBody: { value: new THREE.Matrix3() },
    uMag: { value: new THREE.Vector3(0, 1, 0) }, uMagE1: { value: new THREE.Vector3(1, 0, 0) }, uMagE2: { value: new THREE.Vector3(0, 0, 1) },
    uTloc: { value: 1e6 }, uTcap: { value: 2e6 }, uTinf: { value: cfg.Tinf }, uCap: { value: cfg.capSize }, uChi: { value: cfg.chi }, uCrust: { value: cfg.crust },
    uBeam: { value: 1 }, uOpenK: { value: cfg.openK }, uBeamLen: { value: BEAM_LEN },
    uBeamCol: { value: new THREE.Vector3(...cfg.beamCol) },
    uLines: { value: 1 }, uNphi: { value: cfg.nphi }, uTwist: { value: 0 }, uFlow: { value: cfg.flow },
    uLs: { value: new THREE.Vector4() }, uLineCol: { value: new THREE.Vector3(...cfg.lineCol) },
    uQuake: { value: 0 }, uQuakeSpread: { value: 6 }, uQuakeDir: { value: new THREE.Vector3(1, 0, 0) },
    uFire: { value: new THREE.Vector4() }, uFireR: { value: 1.5 }, uFlash: { value: 0 },
    uFireCol: { value: new THREE.Vector3(0.85, 0.42, 1.0) },
    uCamDirN: { value: new THREE.Vector3(0, 0, 1) }, uCosFlat: { value: 0.2 },
  };
  const defines = q => ({ MAX_STEPS: Math.round(q.steps * 0.6), STEP_K: (q.k * 0.75).toFixed(4) });
  const material = new THREE.ShaderMaterial({
    uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, defines: defines(q0),
    depthTest: false, depthWrite: false,
  });

  const st = {
    phase: 0, lastSim: 0, rtPhase: 0, inBeam: [false, false],
    flareT: -1e9, burstNext: 0, burstT: -1e9, burstAmp: 0,
    fireDirB: new THREE.Vector3(1, 0, 0), pendingFlare: null,
  };
  // debug / snapshot hook: ?flare=<seconds since the quake> shows the giant flare at that age
  try { const f = new URLSearchParams(location.search).get('flare'); if (f !== null && cfg.magnetar) st.pendingFlare = +f; } catch (e) { /* no location */ }

  const m4 = new THREE.Matrix4(), body = new THREE.Matrix3();
  const mag = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3();
  const camDir = new THREE.Vector3(), right = new THREE.Vector3(), tmp = new THREE.Vector3();

  const radiusRS = p => (p.radius ?? 12) / RS_KM;
  const timeScale = () => (ctx.bhGlobals && ctx.bhGlobals.timeScale) || 6;

  function frame(p) {
    const a = THREE.MathUtils.degToRad(p.alpha), sa = Math.sin(a), ca = Math.cos(a);
    const cp = Math.cos(st.phase), sp = Math.sin(st.phase);
    // body vectors m=(sa,ca,0), e1=(ca,-sa,0), e2=(0,0,1) rotated about y by the spin phase
    mag.set(sa * cp, ca, -sa * sp); e1.set(ca * cp, -sa, -ca * sp); e2.set(sp, 0, cp);
    m4.makeRotationY(st.phase); body.setFromMatrix4(m4);
  }

  function beamHalf(p) { return Math.asin(Math.sqrt(Math.min(cfg.openK * 14, 0.3))) * 0.8 + 0.03; }

  function triggerFlare(ctx2, age = 0) {
    st.flareT = ctx2.time.wall - age;
    // the fireball sits ~1.9 r_s above the crack, on closed field lines that trap the pair plasma;
    // put the crack ~40° from the centre of the visible disk (seen on the surface, fireball near the limb);
    // the star's rotation then carries it round — and GR keeps it in view well past the limb
    const cb = ctx2.camera.position.clone().normalize().applyMatrix3(body.clone().transpose());
    const side = new THREE.Vector3(0, 1, 0).cross(cb).normalize();
    if (side.lengthSq() < 0.5) side.set(1, 0, 0);
    const ang = THREE.MathUtils.degToRad(40 + 10 * Math.random()) * (Math.random() < 0.5 ? 1 : -1);
    st.fireDirB.copy(cb).multiplyScalar(Math.cos(ang)).addScaledVector(side, Math.sin(ang));
    st.fireDirB.y *= 0.6; st.fireDirB.normalize();
    uniforms.uQuakeDir.value.copy(st.fireDirB);
  }

  const inst = {
    material,
    setQuality(q) { Object.assign(material.defines, defines(q)); material.needsUpdate = true; },
    update(dt, ctx) {
      const p = ctx.params, U = uniforms;
      const R = radiusRS(p);
      // rotation: `spin` is revolutions per second at the default 6× time speed
      const dSim = ctx.time.sim - st.lastSim; st.lastSim = ctx.time.sim;
      st.phase = (st.phase + 2 * Math.PI * p.spin * dSim / 6) % (2 * Math.PI);
      frame(p);
      ctx.controls.minDistance = R * 1.1;

      U.uR.value = R; U.uGR.value = p.gr ? 1 : 0; U.uSplit.value = p.split ? 1 : 0; U.uGrid.value = p.grid ? 1 : 0;
      U.uBody.value.copy(body); U.uMag.value.copy(mag); U.uMagE1.value.copy(e1); U.uMagE2.value.copy(e2);
      // local (emitted) temperatures; uTinf is the observed base temperature at infinity for R = 12 km
      const g12 = Math.sqrt(1 - RS_KM / 12);
      U.uTloc.value = cfg.Tinf / g12; U.uTcap.value = cfg.Tcap / g12;
      U.uBeam.value = p.beam; U.uLines.value = p.lines;
      U.uTwist.value = p.twist ?? 0;
      U.uLs.value.set(R * cfg.Ls[0], R * cfg.Ls[1], R * cfg.Ls[2], R * cfg.Ls[3]);
      camDir.copy(ctx.camera.position).normalize();
      U.uCamDirN.value.copy(camDir); U.uCosFlat.value = Math.min(1, R / ctx.camera.position.length());

      // ---- magnetar: giant flare + small SGR bursts ----
      if (cfg.magnetar) {
        if (st.pendingFlare !== null) { triggerFlare(ctx, st.pendingFlare); st.pendingFlare = null; }
        const age = ctx.time.wall - st.flareT;
        const on = age >= 0 && age < 90;
        U.uQuake.value = on ? Math.min(1, age / 0.15) * (0.25 + 0.75 * Math.exp(-age / 6)) * Math.exp(-age / 40) : 0;
        U.uQuakeSpread.value = on ? 14 / (1 + age * 0.6) + 2.5 : 6;
        // initial spike (0.2 s, ~10³⁹ J) then the trapped fireball, decaying with a ~30 s (display) e-folding
        let flash = on ? 6 * Math.exp(-age / 0.22) + 0.25 * Math.exp(-age / 1.2) : 0;
        const fireI = on ? Math.min(1, age / 0.4) * (0.15 + 0.85 * Math.exp(-age / 30)) * (1 - THREE.MathUtils.smoothstep(age, 60, 90)) : 0;
        tmp.copy(st.fireDirB).applyMatrix3(body).multiplyScalar(R + 1.9);
        U.uFire.value.set(tmp.x, tmp.y, tmp.z, fireI * 0.9);
        U.uLines.value = p.lines * (1 + 1.5 * fireI);   // the released twist lights up the magnetosphere
        // occasional short SGR bursts (~0.1 s) between giant flares
        if (ctx.time.wall > st.burstNext) {
          if (st.burstNext > 0) { st.burstT = ctx.time.wall; st.burstAmp = 0.12 + Math.random() * 0.25; ctx.audio.click(0.12 * st.burstAmp / 0.3, 0, 0.3); }
          st.burstNext = ctx.time.wall + 7 + Math.random() * 14;
        }
        flash += st.burstAmp * Math.exp(-(ctx.time.wall - st.burstT) / 0.12);
        U.uFlash.value = flash;
      }

      // ---- audio: a click each time a beam sweeps across the camera ----
      right.set(1, 0, 0).applyQuaternion(ctx.camera.quaternion);
      const half = beamHalf(p);
      const zeta = Math.acos(THREE.MathUtils.clamp(camDir.y, -1, 1)), al = THREE.MathUtils.degToRad(p.alpha);
      if (p.rtAudio) {
        // true rotation rate: Vela 11.2 Hz is a buzz, the magnetar's 7.56 s a slow heartbeat
        const prev = st.rtPhase; st.rtPhase += dt / cfg.period;
        if (Math.floor(st.rtPhase) !== Math.floor(prev)) {
          const n1 = Math.abs(zeta - al) < half, n2 = Math.abs(zeta - (Math.PI - al)) < half;
          if (n1 || n2) ctx.audio.click(cfg.clickGain * (n1 && n2 ? 1.4 : 1), 0, cfg.clickBright);
        }
        st.inBeam = [false, false];
      } else {
        for (let k = 0; k < 2; k++) {
          const ax = k ? -1 : 1;
          const ang = Math.acos(THREE.MathUtils.clamp(ax * mag.dot(camDir), -1, 1));
          const inside = ang < half;
          if (inside && !st.inBeam[k]) ctx.audio.click(cfg.clickGain, THREE.MathUtils.clamp(ax * mag.dot(right) * 1.5, -1, 1), cfg.clickBright);
          st.inBeam[k] = inside;
        }
      }
    },
    hud(ctx) {
      const p = ctx.params, R = radiusRS(p), D = ctx.camera.position.length();
      camDir.copy(ctx.camera.position).normalize();
      const psiG = psiMax(R, D, true), psiF = psiMax(R, D, false);
      const fG = visFrac(psiG), fF = visFrac(psiF);
      const cosM = mag.dot(camDir);
      const angN = Math.acos(THREE.MathUtils.clamp(cosM, -1, 1)), angS = Math.PI - angN;
      const nVis = psi => (angN < psi ? 1 : 0) + (angS < psi ? 1 : 0);
      const poleTxt = n => [t('không cực nào', 'neither pole'), t('một cực', 'one pole'), t('cả hai cực', 'both poles')][n];
      const grNow = p.gr && !p.split;
      const b = bulk(p.radius ?? 12);
      const half = beamHalf(p);
      const zeta = Math.acos(THREE.MathUtils.clamp(camDir.y, -1, 1)), al = THREE.MathUtils.degToRad(p.alpha);
      const sweeps = (Math.abs(zeta - al) < half ? 1 : 0) + (Math.abs(zeta - (Math.PI - al)) < half ? 1 : 0);
      const slow = (1 / cfg.period) / Math.max(p.spin * timeScale() / 6, 1e-6);
      const rows = [
        [t('Bề mặt nhìn thấy', 'Visible surface'), p.split ? `GR ${vn(fG * 100, 0)} % · ${t('phẳng', 'flat')} ${vn(fF * 100, 0)} %`
          : grNow ? `${vn(fG * 100, 0)} % (${t('không gian phẳng', 'flat space')}: ${vn(fF * 100, 0)} %)` : `${vn(fF * 100, 0)} % (${t('với GR', 'with GR')}: ${vn(fG * 100, 0)} %)`],
        [t('Cực từ trong tầm nhìn', 'Magnetic poles in view'), `${poleTxt(nVis(grNow || p.split ? psiG : psiF))}${(grNow || p.split) && nVis(psiG) !== nVis(psiF) ? ` (${t('phẳng', 'flat')}: ${poleTxt(nVis(psiF))})` : ''}`],
        [t('Chùm tia quét qua bạn', 'Beam sweeps over you'), sweeps ? (sweeps === 2 ? t('có — 2 xung mỗi vòng', 'yes — 2 pulses per turn') : t('có — 1 xung mỗi vòng', 'yes — 1 pulse per turn')) : t('không (ngoài đường quét)', 'no (outside the beam path)')],
        [t('Tốc độ quay hiển thị', 'Displayed spin rate'), p.spin <= 0 ? t('đứng yên', 'stopped') : (Math.abs(slow - 1) < 0.03 ? t('đúng thời gian thực', 'real time') : slow > 1 ? t(`chậm hơn thật ×${fmtSlow(slow)}`, `slower than real ×${fmtSlow(slow)}`) : t(`nhanh hơn thật ×${fmtSlow(1 / slow)}`, `faster than real ×${fmtSlow(1 / slow)}`))],
        [t('Bán kính · độ đặc', 'Radius · compactness'), `${vn(p.radius ?? 12, 1)} km = ${vn(R, 2)} r_s · z = ${vn(b.z, 2)}`],
        [t('Độ cao trên bề mặt', 'Altitude above the surface'), ctx.fmt.dist(Math.max(D - R, 0) * RS_KM * 1e3)],
      ];
      if (cfg.magnetar) {
        const age = ctx.time.wall - st.flareT;
        rows.push([t('Siêu bùng nổ', 'Giant flare'), age >= 0 && age < 90 ? (age < 1 ? t('ĐỈNH — 10³⁹ J trong 0,2 s', 'PEAK — 10³⁹ J in 0.2 s') : t(`đuôi dao động theo chu kỳ quay · ${vn(age, 0)} s`, `tail pulsing at the spin period · ${vn(age, 0)} s`)) : t('yên tĩnh — nhấn Q', 'quiet — press Q')]);
      }
      return rows;
    },
    onClick(info, ctx) {
      const p = ctx.params, R = radiusRS(p);
      const gr = p.split ? info.ndcX >= 0 : p.gr;
      const hit = traceJS(info.origin, info.dir, R, gr);
      if (!hit) return;
      const n = hit.clone().normalize();
      const c = n.dot(mag), magLat = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(c, -1, 1)));
      const back = n.dot(camDir.copy(ctx.camera.position).normalize());
      const behind = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(back, -1, 1)));
      const cos2 = 4 * c * c / (1 + 3 * c * c);
      const Tinf = cfg.Tinf * Math.pow(cos2 + cfg.chi * (1 - cos2), 0.25);
      const D = ctx.camera.position.length(), horizon = THREE.MathUtils.radToDeg(Math.acos(Math.min(1, R / D)));
      let msg = t(`Vĩ độ từ <b>${vn(magLat, 0)}°</b> · nhiệt độ quan sát ≈ <b>${ctx.fmt.sci(Tinf, 1)} K</b>. Điểm này cách hướng nhìn của bạn <b>${vn(behind, 0)}°</b> quanh tâm sao.`,
        `Magnetic latitude <b>${vn(magLat, 0)}°</b> · observed temperature ≈ <b>${ctx.fmt.sci(Tinf, 1)} K</b>. This point is <b>${vn(behind, 0)}°</b> from your line of sight, measured around the star's centre.`);
      if (behind > horizon + 0.5) msg += t(` Trong không gian phẳng, mọi điểm quá ${vn(horizon, 0)}° đều bị khuất — bạn thấy được nó chỉ vì <b>ánh sáng bị bẻ cong</b> quanh sao.`,
        ` In flat space every point beyond ${vn(horizon, 0)}° would be hidden — you can see it only because <b>light bends</b> around the star.`);
      ctx.toast(msg, 7000);
    },
    enter(ctx) { st.lastSim = ctx.time.sim; st.inBeam = [false, false]; if (cfg.magnetar) st.burstNext = ctx.time.wall + 8; },
    exit(ctx) { ctx.controls.minDistance = 1.2; },
  };
  inst._trigger = c2 => triggerFlare(c2, 0);
  inst._flareAge = c2 => c2.time.wall - st.flareT;
  return inst;
}

function fmtSlow(x) { return x >= 100 ? Math.round(x).toLocaleString(locale()) : vn(x, x >= 10 ? 0 : 1); }

function beamPathAction(ctx) {
  const c = ctx.camera.position, d = c.length();
  const az = Math.atan2(c.x, c.z);
  const pol = THREE.MathUtils.degToRad(ctx.params.alpha + 1.5);
  c.setFromSphericalCoords(d, pol, az); ctx.camera.lookAt(0, 0, 0); ctx.controls.update();
  const mg = ctx.scene && ctx.scene.id === 'magnetar';
  ctx.toast(t(`Bạn đang đứng trên đường quét của cực từ (cách trục quay ${vn(ctx.params.alpha, 0)}°): mỗi vòng quay, chùm tia lướt qua bạn một lần — đó là một <b>xung</b>. ${mg ? 'Với magnetar, xung tia X từ điểm nóng đến mỗi 7,56 s.' : 'Với kính thiên văn vô tuyến thật, xung Vela đến đều đặn mỗi 89,3 ms — ổn định ngang đồng hồ nguyên tử.'}`,
    `You are now on the path swept by the magnetic pole (${vn(ctx.params.alpha, 0)}° from the spin axis): once per rotation the beam sweeps across you — that is a <b>pulse</b>. ${mg ? 'For a magnetar, X-ray pulses from the hot spot arrive every 7.56 s.' : 'With a real radio telescope, Vela’s pulses arrive every 89.3 ms like clockwork — rivalling an atomic clock for stability.'}`), 8000);
}

const GR_PARAMS = [
  { key: 'gr', label: 'Bẻ cong ánh sáng (GR)', type: 'toggle', default: true, note: 'Tắt = tia sáng thẳng (vẫn giữ dịch đỏ) để so sánh: bạn chỉ còn thấy dưới một nửa sao.',
    en: { label: 'Light bending (GR)', note: 'Off = straight light rays (redshift kept) for comparison: you then see less than half the star.' } },
  { key: 'split', label: 'So sánh chia đôi', type: 'toggle', default: false, note: 'Trái: không gian phẳng · Phải: Schwarzschild.',
    en: { label: 'Split-screen comparison', note: 'Left: flat space · Right: Schwarzschild.' } },
  { key: 'grid', label: 'Lưới & nửa sau', type: 'toggle', default: false, note: 'Lưới kinh–vĩ độ. Phần tô cam là vùng bề mặt mà trong không gian phẳng bạn không thể thấy — chỉ hiện ra nhờ ánh sáng bị bẻ cong. Vòng hồng bao quanh hai cực từ.',
    en: { label: 'Grid & far side', note: 'Latitude–longitude grid. The orange-shaded region is surface you could not see in flat space — it appears only because light bends. Pink rings circle the two magnetic poles.' } },
];

const pulsar = makeScene({
  id: 'pulsar', name: 'Sao xung (Pulsar)', order: 10,
  subtitle: 'Sao neutron quay · PSR J0835−4510 (Vela)',
  swatch: ['#d6e8ff', '#18265a'],
  blurb: 'Lõi còn lại của một vụ nổ siêu tân tinh: 1,4 lần khối lượng Mặt Trời nén trong quả cầu 12 km, quay 11 vòng mỗi giây và quét hai chùm sóng vô tuyến như ngọn hải đăng. Hấp dẫn mạnh tới mức ánh sáng bị bẻ cong quanh sao — bạn nhìn thấy <b>hơn một nửa</b> bề mặt, có lúc thấy cả hai cực từ cùng lúc.',
  intro: 'Bật <b>Lưới & nửa sau</b> rồi tắt <b>Bẻ cong ánh sáng</b>: phần tô cam — nửa sau của sao — biến mất. Nhấn <b>B</b> để đứng vào đường quét của chùm tia và nghe tiếng “tích” của xung.',
  camPolar: 75,
  Tinf: 6.8e5, Tcap: 1.3e6, capSize: 0.17, chi: 0.3, crust: 0.07, openK: 0.0042,
  beamCol: [0.42, 0.68, 1.0], lineCol: [0.4, 0.62, 1.0], flow: 0, nphi: 6, Ls: [1.35, 1.9, 2.8, 4.2],
  period: 0.0893, clickGain: 0.28, clickBright: 0.85,
  facts: [
    ['Thiên thể', 'PSR J0835−4510 (Vela)'],
    ['Khoảng cách', '≈ 940 năm ánh sáng'],
    ['Chu kỳ quay', '89,3 ms (11,2 vòng/s)'],
    ['Tuổi đặc trưng', '~11 000 năm'],
    ['Từ trường bề mặt', '≈ 3,4×10¹² G'],
    ['Nhiệt độ bề mặt', '≈ 6,8×10⁵ K'],
    ['Chỏm cực thật', 'r ~ 0,6 km (đã phóng to)'],
    ['Mặt trụ ánh sáng', 'cP/2π ≈ 4 260 km'],
    ...COMMON_FACTS,
    ['Glitch', 'ΔΩ/Ω ~ 10⁻⁶, ~3 năm/lần'],
  ],
  en: {
    name: 'Pulsar',
    subtitle: 'Rotating neutron star · PSR J0835−4510 (Vela)',
    blurb: 'The leftover core of a supernova: 1.4 solar masses crushed into a 12 km ball, spinning 11 times a second and sweeping two radio beams around like a lighthouse. Gravity is so strong that light bends around the star — you see <b>more than half</b> of its surface, sometimes both magnetic poles at once.',
    intro: 'Turn on <b>Grid & far side</b>, then turn off <b>Light bending</b>: the orange-shaded area — the far half of the star — disappears. Press <b>B</b> to step into the beam’s path and hear the “tick” of each pulse.',
    facts: [
      ['Object', 'PSR J0835−4510 (Vela)'],
      ['Distance', '≈ 940 light-years'],
      ['Rotation period', '89.3 ms (11.2 turns/s)'],
      ['Characteristic age', '~11,000 years'],
      ['Surface magnetic field', '≈ 3.4×10¹² G'],
      ['Surface temperature', '≈ 6.8×10⁵ K'],
      ['Actual polar cap', 'r ~ 0.6 km (enlarged here)'],
      ['Light cylinder', 'cP/2π ≈ 4,260 km'],
      ...COMMON_FACTS_EN,
      ['Glitches', 'ΔΩ/Ω ~ 10⁻⁶, about every 3 years'],
    ],
  },
  params: [
    ...GR_PARAMS,
    { key: 'spin', label: 'Tốc độ quay hiển thị', type: 'range', min: 0, max: 2, step: 0.01, default: 0.3, fmt: v => vn(v, 2) + t(' vòng/s', ' turns/s'), note: 'Đã làm chậm rất nhiều: Vela thật quay 11,2 vòng mỗi giây.',
      en: { label: 'Displayed spin rate', note: 'Slowed down enormously: the real Vela turns 11.2 times per second.' } },
    { key: 'alpha', label: 'Góc trục từ – trục quay', type: 'range', min: 0, max: 90, step: 1, default: 55, fmt: v => v.toFixed(0) + '°', en: { label: 'Magnetic–spin axis angle' } },
    { key: 'beam', label: 'Chùm tia vô tuyến', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Sóng vô tuyến vô hình với mắt — vẽ bằng màu giả. Chùm chỉ chói khi chĩa thẳng vào bạn.',
      en: { label: 'Radio beams', note: 'Radio waves are invisible to the eye — drawn in false colour. A beam is only dazzling when it points straight at you.' } },
    { key: 'lines', label: 'Đường sức từ lưỡng cực', type: 'range', min: 0, max: 2, step: 0.01, default: 0.6, note: 'r = L·sin²θ, cũng bị thấu kính hấp dẫn bẻ cong.',
      en: { label: 'Dipole field lines', note: 'r = L·sin²θ, also bent by gravitational lensing.' } },
    { key: 'radius', label: 'Bán kính sao', type: 'range', min: 6.5, max: 16, step: 0.1, default: 12, fmt: v => vn(v, 1) + ' km', note: 'Cùng 1,4 M☉. Sao càng đặc, ánh sáng càng bị bẻ cong; dưới 6,2 km (mặt cầu photon) bạn sẽ thấy toàn bộ bề mặt. Sao neutron thật: ~10–14 km.',
      en: { label: 'Star radius', note: 'Same 1.4 M☉. The more compact the star, the more light bends; below 6.2 km (the photon sphere) you would see the entire surface. Real neutron stars: ~10–14 km.' } },
    { key: 'rtAudio', label: 'Âm thanh thời gian thực', type: 'toggle', default: false, note: 'Tiếng “tích” ở đúng tốc độ thật 11,2 Hz — nghe như tiếng rè. Chỉ khi bạn nằm trên đường quét.',
      en: { label: 'Real-time audio', note: 'Ticks at the true rate of 11.2 Hz — it sounds like a buzz. Only when you are on the beam path.' } },
  ],
  actions: [
    { label: 'Vào đường quét chùm tia', key: 'b', title: 'Đặt camera lên vĩ độ mà cực từ quét qua', run: beamPathAction,
      en: { label: 'Step into the beam path', title: 'Put the camera at the latitude swept by the magnetic pole' } },
  ],
});

const magnetar = makeScene({
  id: 'magnetar', name: 'Từ sao (Magnetar)', order: 11, magnetar: true,
  subtitle: 'Sao neutron từ trường cực mạnh · SGR 1806−20',
  swatch: ['#ffd2f2', '#3a0c3c'],
  blurb: 'Sao neutron có từ trường mạnh nhất vũ trụ đã biết, ~10¹⁵ G — gấp nghìn lần sao xung thường và vượt ngưỡng mà chân không tự nó trở nên lưỡng chiết. Từ trường xoắn đè nén vỏ sắt tới mức nứt vỡ: một “động đất sao” giải phóng siêu bùng nổ tia gamma đủ làm nhiễu tầng điện ly Trái Đất từ hàng chục nghìn năm ánh sáng.',
  intro: 'Nhấn <b>Q</b> để gây <b>động đất sao</b> và xem siêu bùng nổ năm 2004 của SGR 1806−20. Quả cầu plasma bị từ trường giam giữ, quay cùng sao — và nhờ GR vẫn nhìn thấy được khi đã ra sau sao.',
  camPolar: 70, exposure: 0.95,
  Tinf: 7e6, Tcap: 1.15e7, capSize: 0.22, chi: 0.18, crust: 0.11, openK: 0.0035,
  beamCol: [0.75, 0.5, 1.0], lineCol: [1.0, 0.25, 0.8], flow: 1, nphi: 7, Ls: [1.25, 1.75, 2.6, 4.0],
  period: 7.56, clickGain: 0.22, clickBright: 0.25,
  facts: [
    ['Thiên thể', 'SGR 1806−20'],
    ['Khoảng cách', '≈ 28 000 năm ánh sáng'],
    ['Chu kỳ quay', '7,56 s'],
    ['Từ trường', '≈ 2×10¹⁵ G'],
    ['So với ngưỡng QED', '≈ 45 × 4,4×10¹³ G'],
    ['Nhiệt độ bề mặt', '≈ 7×10⁶ K (0,6 keV)'],
    ['Siêu bùng nổ', '27/12/2004 · ~2×10³⁹ J'],
    ['Diễn biến', 'đỉnh 0,2 s + đuôi ~380 s'],
    ['QPO vỏ sao', '18 → 1 840 Hz'],
    ...COMMON_FACTS,
  ],
  en: {
    name: 'Magnetar',
    subtitle: 'Neutron star with an extreme magnetic field · SGR 1806−20',
    blurb: 'The neutron stars with the strongest magnetic fields known, ~10¹⁵ G — a thousand times a normal pulsar’s and beyond the point where the vacuum itself becomes birefringent. The twisted field strains the iron crust until it cracks: a “starquake” that unleashes a gamma-ray giant flare strong enough to disturb Earth’s ionosphere from tens of thousands of light-years away.',
    intro: 'Press <b>Q</b> to trigger a <b>starquake</b> and watch the 2004 giant flare of SGR 1806−20. The plasma fireball is trapped by the magnetic field and co-rotates with the star — and thanks to GR it stays visible even after it has rotated behind the star.',
    facts: [
      ['Object', 'SGR 1806−20'],
      ['Distance', '≈ 28,000 light-years'],
      ['Rotation period', '7.56 s'],
      ['Magnetic field', '≈ 2×10¹⁵ G'],
      ['Relative to the QED limit', '≈ 45 × 4.4×10¹³ G'],
      ['Surface temperature', '≈ 7×10⁶ K (0.6 keV)'],
      ['Giant flare', '27 Dec 2004 · ~2×10³⁹ J'],
      ['Time profile', '0.2 s spike + ~380 s tail'],
      ['Crustal QPOs', '18 → 1,840 Hz'],
      ...COMMON_FACTS_EN,
    ],
  },
  params: [
    ...GR_PARAMS,
    { key: 'spin', label: 'Tốc độ quay hiển thị', type: 'range', min: 0, max: 1, step: 0.002, default: 0.132, fmt: v => vn(v, 3) + t(' vòng/s', ' turns/s'), note: 'Mặc định ≈ thời gian thực (7,56 s mỗi vòng ở tốc độ thời gian 6×).',
      en: { label: 'Displayed spin rate', note: 'Default ≈ real time (7.56 s per turn at the 6× time speed).' } },
    { key: 'alpha', label: 'Góc trục từ – trục quay', type: 'range', min: 0, max: 90, step: 1, default: 35, fmt: v => v.toFixed(0) + '°', en: { label: 'Magnetic–spin axis angle' } },
    { key: 'twist', label: 'Độ xoắn từ trường', type: 'range', min: 0, max: 2, step: 0.01, default: 0.9, fmt: v => vn(v, 2) + ' rad', note: 'Dòng điện chảy dọc đường sức làm từ trường xoắn lại; ứng suất này làm vỏ sao nứt.',
      en: { label: 'Magnetic twist', note: 'Currents flowing along the field lines twist the field; that stress cracks the crust.' } },
    { key: 'lines', label: 'Đường sức từ', type: 'range', min: 0, max: 2, step: 0.01, default: 1, note: 'Màu giả: vầng phát xạ từ plasma tán xạ cộng hưởng trong từ quyển xoắn.',
      en: { label: 'Field lines', note: 'False colour: glow from resonant scattering by plasma in the twisted magnetosphere.' } },
    { key: 'beam', label: 'Chùm tia', type: 'range', min: 0, max: 2, step: 0.01, default: 0.35, note: 'Phần lớn magnetar không phát xung vô tuyến; xung tia X của chúng đến từ điểm nóng trên bề mặt.',
      en: { label: 'Beams', note: 'Most magnetars emit no radio pulses; their X-ray pulses come from hot spots on the surface.' } },
    { key: 'radius', label: 'Bán kính sao', type: 'range', min: 6.5, max: 16, step: 0.1, default: 12, fmt: v => vn(v, 1) + ' km', en: { label: 'Star radius' } },
    { key: 'rtAudio', label: 'Âm thanh thời gian thực', type: 'toggle', default: false, note: 'Một xung mỗi 7,56 s, như nhịp tim chậm.',
      en: { label: 'Real-time audio', note: 'One pulse every 7.56 s, like a slow heartbeat.' } },
  ],
  actions: [
    {
      label: 'Động đất sao', key: 'q', title: 'Vỏ sao nứt vỡ → siêu bùng nổ tia gamma',
      en: { label: 'Starquake', title: 'The crust cracks → gamma-ray giant flare' },
      run(ctx) {
        const inst = ctx.scene && ctx.scene.id === 'magnetar' ? currentMagnetar : null;
        if (!inst) return;
        if (inst._flareAge(ctx) < 3) return;
        inst._trigger(ctx);
        const A = ctx.audio;
        A.tone(41, 5, 0.12, 0, 'sawtooth');
        A.tone(92.5, 7, 0.06, -0.3, 'sine'); A.tone(150, 6, 0.045, 0.3, 'sine'); A.tone(625, 3.5, 0.02, 0, 'sine');
        for (let i = 0; i < 9; i++) setTimeout(() => A.click(0.3 * Math.exp(-i / 4), (Math.random() - 0.5) * 1.4, 0.4 + 0.6 * Math.random()), 30 + i * (40 + Math.random() * 90));
        ctx.toast(t('<b>Động đất sao!</b> Ứng suất của từ trường 10¹⁵ G làm vỡ lớp vỏ sắt. Ngày 27/12/2004, SGR 1806−20 phóng ra ~10³⁹ J chỉ trong 0,2 giây; dù ở cách hàng chục nghìn năm ánh sáng (ước tính lúc đó ~50 000), tia gamma vẫn ion hóa tầng điện ly của Trái Đất. Quả cầu plasma bị từ trường giam lại, quay cùng sao, nên “đuôi” bùng nổ nhấp nháy theo chu kỳ 7,56 s suốt ~6 phút. Âm thanh: dao động xoắn của vỏ sao ở 92,5 · 150 · 625 Hz — các tần số đã đo được thật.',
          '<b>Starquake!</b> Stress from the 10¹⁵ G field shatters the iron crust. On 27 December 2004, SGR 1806−20 released ~10³⁹ J in just 0.2 seconds; although it lies tens of thousands of light-years away (estimated at ~50,000 at the time), its gamma rays still ionised Earth’s ionosphere. The plasma fireball is trapped by the magnetic field and co-rotates with the star, so the flare’s “tail” pulsed with the 7.56 s period for ~6 minutes. Sound: torsional oscillations of the crust at 92.5 · 150 · 625 Hz — frequencies that were actually measured.'), 12000);
      },
    },
    { label: 'Vào đường quét chùm tia', key: 'b', title: 'Đặt camera lên vĩ độ mà cực từ quét qua', run: beamPathAction,
      en: { label: 'Step into the beam path', title: 'Put the camera at the latitude swept by the magnetic pole' } },
  ],
});

// the action needs the magnetar's instance; wrap create to remember it
let currentMagnetar = null;
{
  const create = magnetar.create;
  magnetar.create = ctx => (currentMagnetar = create(ctx));
}

export default [pulsar, magnetar];
