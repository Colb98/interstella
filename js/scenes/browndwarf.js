// Sao lùn nâu — Luhman 16B, a self-luminous T dwarf (split out of the old giants.js).
//
// One scene unit = the dwarf's equatorial radius. Emission shader: grey-atmosphere limb law,
// blackbody × alkali absorption, fast-rotating banded and patchy clouds.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';
import { t, dec, int } from '../i18n.js';

const V3 = THREE.Vector3;
const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAU = Math.PI * 2;
// 1 unit of ctx.time.sim = 60 s on the planet (≈ 360× real time at the host's default 6× slider):
// Jupiter turns once in ~100 s of wall time.
const SECS = 60;

/* ------------------------------------------------------------------------------------------------ */
/* small helpers                                                                                      */
/* ------------------------------------------------------------------------------------------------ */
const ff = v => (+v).toFixed(4);
// piecewise-linear table → GLSL function (x in degrees of latitude)
function tab1(name, pts) {
  const n = pts.length;
  return `float ${name}(float x){
  const float X[${n}] = float[${n}](${pts.map(p => ff(p[0])).join(',')});
  const float Y[${n}] = float[${n}](${pts.map(p => ff(p[1])).join(',')});
  if (x <= X[0]) return Y[0];
  for (int i = 1; i < ${n}; i++){ if (x < X[i]) return mix(Y[i-1], Y[i], (x - X[i-1]) / (X[i] - X[i-1])); }
  return Y[${n - 1}];
}`;
}
function tab3(name, pts) {
  const n = pts.length;
  return `vec3 ${name}(float x){
  const float X[${n}] = float[${n}](${pts.map(p => ff(p[0])).join(',')});
  const vec3 Y[${n}] = vec3[${n}](${pts.map(p => `vec3(${p[1].map(ff).join(',')})`).join(',')});
  if (x <= X[0]) return Y[0];
  for (int i = 1; i < ${n}; i++){ if (x < X[i]){ float u = (x - X[i-1]) / (X[i] - X[i-1]); return mix(Y[i-1], Y[i], u * u * (3.0 - 2.0 * u)); } }
  return Y[${n - 1}];
}`;
}
function lerpTab(pts, x) {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) if (x < pts[i][0]) return pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * (x - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
  return pts[pts.length - 1][1];
}
const latlonJS = (latD, lonE) => { const la = latD * D2R; return new V3(Math.cos(la) * Math.cos(lonE), Math.sin(la), -Math.cos(la) * Math.sin(lonE)); };
const vn = (x, d = 1) => dec(x, d);
const latStr = d => `${vn(Math.abs(d))}° ${d >= 0 ? t('B', 'N') : t('N', 'S')}`;
const windStr = u => `${u >= 0 ? '+' : '−'}${Math.abs(u).toFixed(0)} m/s ${u >= 0 ? t('(về đông)', '(eastward)') : t('(về tây)', '(westward)')}`;
const wrap = a => ((a % TAU) + TAU) % TAU;
const fmtInt = n => int(n);

/* ------------------------------------------------------------------------------------------------ */
/* zonal wind profiles (planetographic latitude °, m/s, relative to the interior rotation)            */
/* ------------------------------------------------------------------------------------------------ */
// Jupiter: after Voyager/Cassini cloud tracking (Limaye 1986, Porco+ 2003), smoothed. Belts sit where
// the shear is cyclonic, zones where it is anticyclonic — the 23.7°N jet is the fastest (~165 m/s).
const HEAD = /* glsl */`
uniform vec3 uSunDir;      // world, toward the Sun
uniform mat3 uW2B;         // world → body frame (y = spin axis, not spinning)
uniform float uFlat, uSpin, uRm, uSunI;
uniform vec3 uPh;          // two-phase wind clock: x = t1, y = t2 (s), z = weight of phase 1
uniform vec2 uSeed;        // noise seeds of the two phases
uniform vec4 uDrift;       // longitudes (rad) of drifting features
uniform float uAxis, uRingOn, uRingGain, uRingRot;
uniform vec4 uMoonA;       // Jupiter: orbital angles of the Galilean moons
#define R2D 57.2957795
#define D2R 0.0174532925
float gMu, gMu0;           // cos(view), cos(sun) at the current surface point
vec3 gSunQ;                // Sun direction in body-fixed (spinning) coordinates

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
// body-fixed unit vector at planetographic latitude (deg) and east longitude (rad)
vec3 latlon(float latD, float lonE){ float la = latD * D2R; return vec3(cos(la) * cos(lonE), sin(la), -cos(la) * sin(lonE)); }
void tframe(vec3 c, out vec3 e, out vec3 nN){ e = normalize(vec3(c.z, 0.0, -c.x) + vec3(0.0, 0.0, 1e-6)); nN = cross(c, e); }
// elliptical radius around a feature centre c (semi-axes ax, radians); l = local (east, north) coords
float vrad(vec3 q, vec3 c, vec2 ax, out vec2 l){
  vec3 e, nN; tframe(c, e, nN); vec3 d = q - c;
  l = vec2(dot(d, e) / ax.x, dot(d, nN) / ax.y);
  return dot(q, c) > 0.0 ? length(l) : 99.0;
}
// vortex advection: rotate the noise domain about c (elliptical streamlines) by ang (+ = counter-clockwise
// seen from outside) plus a static spiral twist. Returns the source point and the vortex mask.
vec4 vortex(vec3 q, vec3 c, vec2 ax, float ang, float tw, float rIn, float rOut){
  vec3 e, nN; tframe(c, e, nN); vec3 d = q - c;
  vec2 l = vec2(dot(d, e) / ax.x, dot(d, nN) / ax.y);
  float r = length(l);
  float m = smoothstep(rOut, rIn, r) * step(0.0, dot(q, c));
  float a = -(ang + tw * max(1.0 - r / rOut, 0.0)) * m;
  float cs = cos(a), sn = sin(a);
  vec2 l2 = vec2(cs * l.x - sn * l.y, sn * l.x + cs * l.y);
  return vec4(normalize(q + e * ((l2.x - l.x) * ax.x) + nN * ((l2.y - l.y) * ax.y)), m);
}
vec3 seedOff(float s){ return vec3(s * 13.7, s * 5.3, s * 9.1); }
float bump(float x, float c, float w){ float t = (x - c) / w; return exp(-t * t); }
// Henyey–Greenstein phase function normalised to mean 1 over the sphere (c = cos scattering angle)
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * c, 1.5); }
// shadow of the (oblate) planet on point p (body frame) toward the Sun s
float bodyShadow(vec3 p, vec3 s, vec3 E){
  vec3 o = p * E, d = s * E;
  float tc = -dot(o, d) / dot(d, d);
  if (tc < 0.0) return 1.0;
  return smoothstep(0.996, 1.004, length(o + d * tc));
}
// radial value noise for ring structure, band-limited by the pixel footprint fw
float rh1(float i){ return hash13(vec3(i, 7.31, 1.97)); }
float rn1(float x){ float i = floor(x), f = fract(x); return mix(rh1(i), rh1(i + 1.0), f * f * (3.0 - 2.0 * f)); }
float rband(float r, float freq, float fw){ return (rn1(r * freq) - 0.5) * (1.0 - smoothstep(0.25, 0.8, fw * freq)); }
// narrow ring / gap of width w at r0, anti-aliased: keeps the integrated optical depth when sub-pixel
float rbox(float r, float r0, float w, float fw){ float we = max(w, fw); return (w / we) * (1.0 - smoothstep(0.35 * we, 0.65 * we, abs(r - r0))); }
`;

const BD_FRAG = COMMON_GLSL + HEAD + /* glsl */`
uniform float uTeff, uMode, uClouds, uGlow;
uniform vec3 uEye, uEyeCool; // chromaticity of the alkali-absorbed spectrum at Teff and at 0.85 Teff (luminance 1)
uniform vec3 uCompDir, uCompCol;
float bdWind(float lat){ return 650.0 * cos(lat) * cos(lat) * cos(5.0 * lat); }
vec3 bdField(vec3 p, float seed){
  vec3 Q = vec3(p.x, p.y * 2.4, p.z) + seedOff(seed);
  float a = fbm(Q * 1.7, 4);
  float b = fbm(Q * 4.5 + vec3(a * 1.8, 0.0, -a * 1.8), 4);
  float c = fbm(Q * 13.0 + vec3(b * 1.2), 3);
  return vec3(a, b, c);
}
void main(){
  vec3 rdW = camRay(), roW = uCamPos;
  vec3 ro = uW2B * roW, rd = uW2B * rdW;
  vec3 E = vec3(1.0, 1.0 / (1.0 - uFlat), 1.0);
  vec3 roE = ro * E, rdE = rd * E;
  float A = dot(rdE, rdE), B = dot(roE, rdE), C = dot(roE, roE) - 1.0, H = B * B - A * C;
  vec3 col;
  float tP = H > 0.0 ? (-B - sqrt(H)) / A : -1.0;
  if (tP > 0.0){
    vec3 p = ro + rd * tP;
    vec3 n = normalize(p * E * E);
    vec3 q = rotY(-uSpin) * (p * E);
    float lat = asin(clamp(n.y, -1.0, 1.0)), lonE = atan(-q.z, q.x);
    float mu = max(dot(n, -rd), 0.0);
    float w1 = uPh.z, w2 = 1.0 - w1;
    float om = bdWind(lat * R2D * D2R) / (uRm * max(cos(lat), 0.08));
    vec3 f = (bdField(rotY(-om * uPh.x) * q, uSeed.x) * w1 + bdField(rotY(-om * uPh.y) * q, uSeed.y) * w2) * inversesqrt(w1 * w1 + w2 * w2);
    float latD = lat * R2D;
    // cloud thickness: zonal bands modulated by planetary-scale waves (Apai+ 2017) plus patchy clouds (Doppler map, Crossfield+ 2014)
    float bands = sin(latD * 0.13 + 1.1 * f.x + 0.7 * sin(lonE * 2.0 + latD * 0.05));
    float cl = 0.5 + 0.42 * bands + 0.5 * f.y + 0.25 * f.z;
    cl = mix(0.5, cl, uClouds);
    // thin clouds reveal deeper, hotter layers
    float Tloc = uTeff * (1.0 + 0.065 * (0.5 - cl));
    // grey (Eddington) atmosphere: T(τ)^4 = ¾ Teff^4 (τ + ⅔); at slant μ we see down to τ ≈ μ
    float Tb = Tloc * pow(0.75 * (mu + 2.0 / 3.0), 0.25);
    if (uMode < 0.5){                 // what an eye would see: Wien tail at ~0.75 µm × alkali-absorbed spectrum
      float I = exp(19184.0 * (1.0 / uTeff - 1.0 / Tb));
      col = mix(uEye, uEyeCool, clamp((uTeff - Tb) / (0.15 * uTeff), 0.0, 1.0)) * I * uGlow;
    } else if (uMode < 1.5){          // J band (1.25 µm): the window where cloud holes stand out (false colour)
      float I = exp(11510.0 * (1.0 / uTeff - 1.0 / Tb)) * 0.55;
      col = mix(vec3(0.25, 0.05, 0.02), vec3(1.0, 0.55, 0.18), smoothstep(0.1, 0.8, I)) * I + vec3(1.0, 0.9, 0.7) * smoothstep(0.8, 1.6, I) * 0.6;
    } else {                          // a pure blackbody of the same temperature (no molecular / alkali absorption)
      float I = exp(19184.0 * (1.0 / uTeff - 1.0 / Tb));
      col = bb(Tb) * I * uGlow;
    }
  } else {
    col = sky(rdW, 1.0) * 0.6;
    float c = dot(rdW, uCompDir);   // the companion Luhman 16A, 3.5 AU away: a brilliant point
    col += uCompCol * (smoothstep(0.999994, 0.999999, c) * 14.0 + pow(max(c, 0.0), 40000.0) * 1.5 + pow(max(c, 0.0), 3000.0) * 0.03);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

/* ------------------------------------------------------------------------------------------------ */
/* JS side: orientation, clocks, picking                                                              */
/* ------------------------------------------------------------------------------------------------ */
// Body frame for obliquity ob (deg) with the Sun (world azimuth sunAz) at sub-solar latitude B (deg).
function orient(st, obDeg, BDeg, sunAzDeg) {
  const az = sunAzDeg * D2R, ob = obDeg * D2R;
  st.sun.set(Math.sin(az), 0, Math.cos(az));
  const c = THREE.MathUtils.clamp(Math.sin(BDeg * D2R) / Math.max(Math.sin(ob), 1e-6), -1, 1);
  const paz = az + Math.acos(c);
  const P = st.pole.set(Math.sin(ob) * Math.sin(paz), Math.cos(ob), Math.sin(ob) * Math.cos(paz));
  const a = Math.abs(P.y) < 0.99 ? new V3(0, 1, 0) : new V3(1, 0, 0);
  const xb = new V3().crossVectors(a, P).normalize(), zb = new V3().crossVectors(xb, P);
  st.W2B.set(xb.x, xb.y, xb.z, P.x, P.y, P.z, zb.x, zb.y, zb.z);
}
function hitBody(o, d, W2B, flat) {
  const ro = o.clone().applyMatrix3(W2B), rd = d.clone().applyMatrix3(W2B);
  const k = 1 / (1 - flat);
  const A = rd.x * rd.x + rd.y * rd.y * k * k + rd.z * rd.z, B = ro.x * rd.x + ro.y * rd.y * k * k + ro.z * rd.z;
  const C = ro.x * ro.x + ro.y * ro.y * k * k + ro.z * ro.z - 1, H = B * B - A * C;
  let t = null;
  if (H >= 0) { const t0 = (-B - Math.sqrt(H)) / A; if (t0 > 0) t = t0; }
  return { t, ro, rd, k };
}
// body point → planetographic latitude / body-fixed east longitude
function bodyLatLon(p, k, spin) {
  const n = new V3(p.x, p.y * k * k, p.z).normalize();
  const c = Math.cos(spin), s = Math.sin(spin);
  const qx = c * p.x - s * p.z, qz = s * p.x + c * p.z;     // Ry(−spin)
  return { lat: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) * R2D, lonE: Math.atan2(-qz, qx), n };
}

function dwarfColour(T, alkali) {
  const g = (x, mu, s1, s2) => { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
  const h = 6.62607015e-34, c = 2.99792458e8, k = 1.380649e-23;
  const s = Math.min(1.6, Math.max(0.15, (2300 - T) / 1100));   // alkali wings strengthen as clouds sink in cooler dwarfs
  let X = 0, Y = 0, Z = 0;
  for (let lam = 380; lam <= 780; lam += 2.5) {
    const l = lam * 1e-9;
    const B = 1 / (Math.pow(l, 5) * (Math.exp(h * c / (l * k * T)) - 1));
    const tr = alkali ? Math.exp(-s * (10 / (1 + ((lam - 589) / 80) ** 2) + 10 / (1 + ((lam - 768) / 35) ** 2))) : 1;
    X += B * tr * (1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2));
    Y += B * tr * (0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1));
    Z += B * tr * (1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8));
  }
  X /= Y; Z /= Y;
  let r = 3.2406 * X - 1.5372 - 0.4986 * Z, gg = -0.9689 * X + 1.8758 + 0.0415 * Z, b = 0.0557 * X - 0.2040 + 1.0570 * Z;
  const m = Math.min(r, gg, b); if (m < 0) { r -= m; gg -= m; b -= m; }
  const lum = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
  return [r / lum, gg / lum, b / lum];
}
const specType = T => T > 2300 ? t('M muộn', 'late M') : T > 1300 ? 'L' + Math.min(9, Math.max(0, Math.round((2300 - T) / 110))) : T > 550 ? 'T' + Math.min(9, Math.max(0, Math.round((1300 - T) / 85))) : 'Y';

const BROWN_DWARF = {
  id: 'brown-dwarf', name: 'Sao lùn nâu', category: 'stars', order: 60,
  subtitle: 'Luhman 16B · kiểu phổ T0,5 · 6,5 năm ánh sáng',
  swatch: ['#d0507a', '#2a0614'],
  blurb: 'Quá nặng để là hành tinh, quá nhẹ để là ngôi sao: sao lùn nâu (13–80 lần khối lượng Sao Mộc) chỉ đốt được <b>deuterium</b> trong thời gian ngắn rồi nguội dần mãi mãi. Luhman 16B nóng ~1 200 K, tự phát sáng đều cả hai mặt; Na và K hấp thụ ánh vàng–cam nên mắt người sẽ thấy nó màu <b>đỏ tía (magenta)</b>.',
  intro: 'Đổi <b>Bước sóng</b> sang hồng ngoại J để thấy các khe hở giữa mây — đúng thứ mà phép chụp Doppler (Crossfield 2014) và JWST dùng để vẽ bản đồ thời tiết của nó.',
  facts: [
    ['Khối lượng', '~28 M_Sao Mộc (Garcia 2017)'], ['Bán kính', '~0,85 – 1 R_Sao Mộc'], ['Nhiệt độ hiệu dụng', '~1 210 K'],
    ['Chu kỳ tự quay', '5,28 giờ'], ['Khoảng cách', '2,0 pc (6,5 năm ánh sáng) — sao lùn nâu gần nhất'],
    ['Bạn đồng hành', 'Luhman 16A (L7,5, ~34 M_J), cách 3,5 AU'],
    ['Ngưỡng đốt deuterium', '≈ 13 M_J'], ['Ngưỡng đốt hydro (thành sao)', '≈ 75 – 80 M_J'],
  ],
  en: {
    name: 'Brown dwarf',
    subtitle: 'Luhman 16B · spectral type T0.5 · 6.5 light-years',
    blurb: 'Too massive to be a planet, too light to be a star: a brown dwarf (13–80 Jupiter masses) can fuse only <b>deuterium</b>, and only briefly, then cools forever. Luhman 16B is ~1,200 K and glows evenly on both hemispheres; sodium and potassium absorb the yellow–orange light, so to the human eye it would look <b>magenta</b>.',
    intro: 'Switch <b>Wavelength</b> to the J-band infrared to see the gaps between the clouds — exactly what Doppler imaging (Crossfield 2014) and JWST use to map its weather.',
    facts: [
      ['Mass', '~28 M_Jup (Garcia 2017)'], ['Radius', '~0.85 – 1 R_Jup'], ['Effective temperature', '~1,210 K'],
      ['Rotation period', '5.28 h'], ['Distance', '2.0 pc (6.5 light-years) — the nearest brown dwarf'],
      ['Companion', 'Luhman 16A (L7.5, ~34 M_J), 3.5 AU away'],
      ['Deuterium-burning limit', '≈ 13 M_J'], ['Hydrogen-burning limit (becoming a star)', '≈ 75 – 80 M_J'],
    ],
  },
  unitKm: 70000,
  camera: { distance: 3.2, polar: 80, azimuth: 20, min: 1.05, max: 60, fov: 45 },
  exposure: 1.0, bloom: 0.5,
  params: [
    { key: 'teff', label: 'Nhiệt độ hiệu dụng', type: 'range', min: 500, max: 2400, step: 10, default: 1210, fmt: v => v.toFixed(0) + ' K', note: 'Kéo để đi qua dãy L → T → Y: sao lùn nâu nguội dần theo tuổi.',
      en: { label: 'Effective temperature', note: 'Drag through the L → T → Y sequence: brown dwarfs cool as they age.' } },
    { key: 'view', label: 'Bước sóng', type: 'seg', options: [['eye', 'Mắt người'], ['ir', 'Hồng ngoại J'], ['bb', 'Vật đen thuần']], default: 'eye', note: '“Vật đen thuần” bỏ đi sự hấp thụ của Na, K, H₂O, CH₄ — so sánh để thấy vì sao nó không “nâu”.',
      en: { label: 'Wavelength', options: ['Human eye', 'J-band infrared', 'Pure blackbody'], note: '“Pure blackbody” removes the absorption by Na, K, H₂O and CH₄ — compare to see why it isn’t “brown”.' } },
    { key: 'clouds', label: 'Độ loang lổ của mây', type: 'range', min: 0, max: 1, step: 0.01, default: 0.7, en: { label: 'Cloud patchiness' } },
    { key: 'windx', label: 'Tua nhanh gió', type: 'range', min: 0, max: 20, step: 0.5, default: 1, fmt: v => '×' + v.toFixed(1), en: { label: 'Wind speed-up' } },
    { key: 'spin', label: 'Tự quay', type: 'toggle', default: true, en: { label: 'Rotation' } },
  ],
  actions: [
    { label: 'Hồng ngoại', key: 'i', title: 'Chuyển giữa mắt người và hồng ngoại J', run: ctx => { ctx.params.view = ctx.params.view === 'ir' ? 'eye' : 'ir'; ctx.toast(ctx.params.view === 'ir' ? t('Hồng ngoại J (1,25 µm): ánh sáng thoát ra từ sâu hơn qua các khe mây — chỗ sáng là nơi mây mỏng, nhìn thấy lớp khí nóng hơn.', 'J-band infrared (1.25 µm): light escapes from deeper down through the cloud gaps — bright patches are where the clouds are thin and the hotter gas below shows through.') : t('Ánh sáng khả kiến (mô phỏng mắt người).', 'Visible light (simulated human eye).')); },
      en: { label: 'Infrared', title: 'Toggle between the human eye and J-band infrared' } },
  ],
  create(ctx) {
    const R = 7.0e7, prot = 5.28 * 3600, phaseP = 4e4, flat = 0.012;
    const U = {
      ...ctx.shared,
      uW2B: { value: new THREE.Matrix3() }, uFlat: { value: flat }, uSpin: { value: 0 }, uRm: { value: R },
      uPh: { value: new V3(0, 0, 1) }, uSeed: { value: new THREE.Vector2() },
      uTeff: { value: 1210 }, uMode: { value: 0 }, uClouds: { value: 0.7 }, uGlow: { value: 0.13 },
      uEye: { value: new V3(1, 0.3, 0.4) }, uEyeCool: { value: new V3(1, 0.3, 0.4) }, uCompDir: { value: new V3(-0.55, 0.32, -0.77).normalize() }, uCompCol: { value: new V3(1, 0.5, 0.3) },
      // unused HEAD uniforms
      uSunDir: { value: new V3(0, 0, 1) }, uSunI: { value: 0 }, uDrift: { value: new THREE.Vector4() }, uAxis: { value: 0 },
      uRingOn: { value: 0 }, uRingGain: { value: 0 }, uRingRot: { value: 0 }, uMoonA: { value: new THREE.Vector4() },
    };
    // slight obliquity so the poles show (unknown for Luhman 16B; its rotation axis is thought to be near edge-on)
    const ob = 12 * D2R, P = new V3(Math.sin(ob), Math.cos(ob), 0);
    const xb = new V3().crossVectors(new V3(1, 0, 0), P).normalize(), zb = new V3().crossVectors(xb, P);
    U.uW2B.value.set(xb.x, xb.y, xb.z, P.x, P.y, P.z, zb.x, zb.y, zb.z);
    // place Luhman 16A in the sky, up and to the left of the default view
    { const c = this.camera, v = new V3().setFromSphericalCoords(1, c.polar * D2R, c.azimuth * D2R).negate();
      const rt = new V3().crossVectors(v, new V3(0, 1, 0)).normalize(), up = new V3().crossVectors(rt, v);
      U.uCompDir.value.copy(v).addScaledVector(rt, -0.42).addScaledVector(up, 0.22).normalize(); }
    const comp = dwarfColour(1350, true);
    U.uCompCol.value.set(...comp).multiplyScalar(1.4);
    const material = new THREE.ShaderMaterial({ uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: BD_FRAG, depthTest: false, depthWrite: false });
    const st = { lastSim: null, frozen: 0, T: null, lastTeff: -1 };
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params, sim = ctx.time.sim;
        if (st.lastSim === null) st.lastSim = sim;
        const dsim = Math.max(0, sim - st.lastSim); st.lastSim = sim;
        if (!p.spin) st.frozen += dsim;
        const tp = (sim - st.frozen) * SECS;
        if (st.T === null) st.T = tp * p.windx; else if (p.spin) st.T += dsim * SECS * p.windx;
        U.uSpin.value = (TAU * tp / prot) % TAU;
        const T = st.T, t1 = T % phaseP;
        U.uPh.value.set(t1, (T + phaseP / 2) % phaseP, 1 - Math.abs(2 * t1 / phaseP - 1));
        U.uSeed.value.set(Math.floor(T / phaseP) % 29, (Math.floor((T + phaseP / 2) / phaseP) % 29) + 0.5);
        U.uTeff.value = p.teff; U.uClouds.value = p.clouds;
        U.uMode.value = p.view === 'ir' ? 1 : p.view === 'bb' ? 2 : 0;
        // displayed brightness grows gently with Teff (true visible surface brightness spans ~10⁵ over this range)
        U.uGlow.value = 0.13 * Math.pow(p.teff / 1210, 1.5);
        if (p.teff !== st.lastTeff) { st.lastTeff = p.teff; U.uEye.value.set(...dwarfColour(p.teff, true)); U.uEyeCool.value.set(...dwarfColour(p.teff * 0.85, true)); }
        st.tp = tp;
      },
      hud(ctx) {
        const T = ctx.params.teff;
        return [
          [t('Kiểu phổ (theo T)', 'Spectral type (from T)'), specType(T)],
          [t('Đỉnh phát xạ (Wien)', 'Emission peak (Wien)'), vn(2898 / T, 2) + t(' µm — hồng ngoại', ' µm — infrared')],
          [t('Tốc độ quay ở xích đạo', 'Equatorial rotation speed'), vn(TAU * R / prot / 1000) + ' km/s'],
          [t('Độ sáng so với Mặt Trời', 'Luminosity relative to the Sun'), '~' + ctx.fmt.sci(Math.pow(T / 5772, 4) * 0.01) + ' L☉'],
        ];
      },
      onClick(info, ctx) {
        const W2B = U.uW2B.value;
        const h = hitBody(info.origin, info.dir, W2B, flat);
        if (h.t === null) return;
        const p = h.ro.clone().addScaledVector(h.rd, h.t);
        const ll = bodyLatLon(p, h.k, 0);
        const mu = Math.max(0, -ll.n.dot(h.rd));
        const Tb = ctx.params.teff * Math.pow(0.75 * (mu + 2 / 3), 0.25);
        ctx.toast(t(`Nhiệt độ sáng ở đây ≈ <b>${fmtInt(Tb)} K</b> (μ = ${vn(mu, 2)}). Ở rìa đĩa ta nhìn xiên vào các lớp khí cao hơn, lạnh hơn — nên rìa tối và đỏ hơn tâm (khí quyển xám Eddington).`,
          `Brightness temperature here ≈ <b>${fmtInt(Tb)} K</b> (μ = ${vn(mu, 2)}). Near the limb we look obliquely into higher, cooler layers — so the limb is darker and redder than the centre (Eddington grey atmosphere).`), 6000);
      },
    };
  },
};


export default [BROWN_DWARF];
