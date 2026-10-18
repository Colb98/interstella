// Hành tinh dung nham — 55 Cancri e (Janssen), a tidally locked lava world.
// One scene unit = one planetary radius. The star is fixed in space; orbit around to see day and night.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';
import { t } from '../i18n.js';

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
const latStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? t('B', 'N') : t('N', 'S')}`;
const lonStr = v => `${Math.abs(v).toFixed(1)}°${v >= 0 ? t('Đ', 'E') : t('T', 'W')}`;
function orbitRows(ctx, Rkm, refR = 1) {
  const d = ctx.camera.position.length(), hKm = (d - refR) * Rkm;
  const horizon = Math.sqrt(Math.max(d * d - refR * refR, 0)) * Rkm;
  return [[t('Độ cao', 'Altitude'), ctx.fmt.dist(hKm * 1000)], [t('Khoảng cách tới đường chân trời', 'Distance to the horizon'), ctx.fmt.dist(horizon * 1000)]];
}
const hourParam = (def, note) => ({
  key: 'hour', label: 'Giờ địa phương (dưới camera)', type: 'range', min: 0, max: 24, step: 0.1, default: def,
  fmt: v => hhmm(v), note, en: { label: 'Local time (below the camera)' },
});
function makeMaterial(frag, uniforms) {
  return new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: frag, depthTest: false, depthWrite: false });
}
function sunElevStr(n, L) {
  const e = Math.asin(Math.max(-1, Math.min(1, n.dot(L)))) / D2R;
  const tag = e > 6 ? t('ban ngày', 'daytime') : e > -0.8 ? t('bình minh/hoàng hôn', 'sunrise/sunset') : e > -18 ? t('chạng vạng', 'twilight') : t('ban đêm', 'night');
  return t(`Mặt Trời cao <b>${e.toFixed(1)}°</b> (${tag})`, `Sun elevation <b>${e.toFixed(1)}°</b> (${tag})`);
}

/* Earth uses a right-handed geography (east = counter-clockwise seen from the north, like the real
   planet and the NASA maps): body = rotY(−spin)·world, east longitude = atan2(x, z). */
const llE = (lat, lon) => [Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R), Math.cos(lat * D2R) * Math.cos(lon * D2R)];
// afternoon puts the Sun west of the meridian (= −azimuth for this spin sense)
export default [
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
  en: {
    name: 'Lava world',
    subtitle: '55 Cancri e (Janssen) · tidally locked super-Earth',
    blurb: '55 Cancri e orbits its star in just 17.7 hours, so close that it is <b>tidally locked</b>: one hemisphere is in permanent daylight. The dayside reaches ~2,700 K, hot enough to melt silicate rock into a lava ocean that glows on its own like a <b>blackbody</b>; the star fills ~32° of the sky.',
    intro: 'The planet does not rotate relative to its star — swing the camera round the back to see the night side with its still-smouldering cracks, and the giant star behind it. Press <b>I</b> for the infrared heat map.',
    facts: [
      ['Radius', '1.88 R⊕ (~12,000 km)'], ['Mass', '7.99 M⊕'], ['Surface gravity', '~22 m/s²'],
      ['Orbital period', '17.7 h'], ['Semi-major axis', '0.0154 AU (2.3 million km)'],
      ['Dayside temperature (Spitzer)', '~2,700 K'], ['Nightside temperature', '~1,400 K'],
      ['Host star', '55 Cancri A (Copernicus), K0, 5,200 K'], ['Distance', '41 light-years'],
      ['Atmosphere', 'signs of a secondary CO/CO₂ atmosphere (JWST, 2024)'],
    ],
  },
  unitKm: 12000,
  camera: { distance: 3.4, polar: 78, azimuth: 5, min: 1.03, max: 40, fov: 45 },
  exposure: 0.9,
  bloom: 0.4,
  params: [
    { key: 'tss', label: 'Nhiệt độ điểm cận sao', type: 'range', min: 1500, max: 3500, step: 10, default: 2700, fmt: v => v.toFixed(0) + ' K', note: 'Cân bằng bức xạ không phân phối nhiệt: T = T₀·cos^¼(góc thiên đỉnh). Độ sáng khả kiến tăng cực nhanh theo T (định luật Planck).',
      en: { label: 'Substellar-point temperature', note: 'Radiative equilibrium with no heat redistribution: T = T₀·cos^¼(zenith angle). Visible brightness climbs extremely steeply with T (Planck’s law).' } },
    { key: 'crust', label: 'Lớp vỏ nguội', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5, note: 'Đá silicat đông đặc dưới ~1 600 K thành các mảng vỏ trôi trên biển dung nham.',
      en: { label: 'Cooled crust', note: 'Below ~1,600 K silicate rock solidifies into crust plates drifting on the lava ocean.' } },
    { key: 'view', label: 'Chế độ xem', type: 'seg', options: [['vis', 'Khả kiến'], ['ir', 'Bản đồ nhiệt']], default: 'vis', note: 'Bản đồ nhiệt: 900 K (tím đen) → 3 500 K (vàng trắng).',
      en: { label: 'View', options: ['Visible', 'Heat map'], note: 'Heat map: 900 K (black-violet) → 3,500 K (yellow-white).' } },
  ],
  actions: [
    { label: 'Hồng ngoại', key: 'i', run: ctx => { ctx.params.view = ctx.params.view === 'ir' ? 'vis' : 'ir'; }, en: { label: 'Infrared' } },
    { label: 'Ngược sáng', key: 'o', title: 'Bay ra phía đêm, nhìn về ngôi sao', run: ctx => {
      ctx.camera.position.setFromSphericalCoords(3.4, 84 * D2R, 215 * D2R); ctx.camera.lookAt(0, 0, 0); ctx.controls.target.set(0, 0, 0); ctx.controls.update();
      ctx.toast(t('Mặt đêm (~1 400 K): lớp vỏ đá đen với các khe nứt dung nham còn âm ỉ. Phía sau là 55 Cancri A — chiếm 32° bầu trời, to gấp ~60 lần Mặt Trời nhìn từ Trái Đất.', 'Night side (~1,400 K): a black rock crust with lava cracks still smouldering. Behind it is 55 Cancri A — spanning 32° of the sky, ~60 times wider than the Sun seen from Earth.'));
    }, en: { label: 'Backlit', title: 'Fly to the night side and look back at the star' } },
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
          [t('Nhiệt độ cân bằng dưới camera', 'Equilibrium temperature below the camera'), Teq(n, ctx.params.tss).toFixed(0) + ' K'],
          [t('Đường kính biểu kiến của sao', 'Apparent diameter of the star'), starAng.toFixed(1) + t('° (Mặt Trời từ Trái Đất: 0,53°)', '° (the Sun from Earth: 0.53°)')],
        ];
      },
      onClick(info, ctx) {
        const th = hitSphere(info.origin, info.dir);
        if (th === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, th).normalize();
        const T = Teq(p, ctx.params.tss), z = Math.acos(Math.max(-1, Math.min(1, p.dot(STAR_DIR)))) / D2R;
        const state = T > 1900 ? t('biển dung nham lỏng', 'a molten lava ocean') : T > 1500 ? t('dung nham với các mảng vỏ trôi', 'lava with drifting crust plates') : t('vỏ đá đông đặc, nứt nẻ còn âm ỉ', 'solid rock crust, cracked and still smouldering');
        ctx.toast(t(`Góc thiên đỉnh của sao <b>${z.toFixed(0)}°</b> · nhiệt độ cân bằng <b>${T.toFixed(0)} K</b> (${(T - 273).toFixed(0)} °C): ${state}. Bức xạ nhiệt ∝ T⁴ = ${(5.67e-8 * T ** 4 / 1e6).toFixed(1)} MW/m².`,
          `Stellar zenith angle <b>${z.toFixed(0)}°</b> · equilibrium temperature <b>${T.toFixed(0)} K</b> (${(T - 273).toFixed(0)} °C): ${state}. Thermal emission ∝ T⁴ = ${(5.67e-8 * T ** 4 / 1e6).toFixed(1)} MW/m².`));
      },
    };
  },
},
];
