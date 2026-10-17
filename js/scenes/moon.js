// Mặt Trăng — reference implementation of the scene contract (see CONTRACT.md).
// One scene unit = one lunar radius (1 737.4 km). Sunlight is a directional light.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../glsl/common.js';

const FRAG = COMMON_GLSL + /* glsl */`
uniform vec3 uSunDir;
uniform float uCraters, uEarthshine, uRot;

// impact craters: bowl + raised rim, random size per cell
float craterField(vec3 p, float seed){
  vec3 i = floor(p), f = fract(p); float h = 0.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(x, y, z);
    vec3 r = hash33(i + o + seed);
    if (r.z > uCraters) continue;
    float R = 0.18 + 0.32 * hash13(i + o + seed + 7.1);
    float d = length(o + r - f) / R;
    if (d < 1.8) h += R * ((d < 1.0 ? (d * d - 1.0) * 0.55 : 0.0) + 0.22 * exp(-pow((d - 1.0) * 4.5, 2.0)));
  }
  return h;
}
float mare(vec3 p){ return smoothstep(0.05, 0.32, fbm(p * 1.4 + 3.0, 4)); }
float height(vec3 p){
  float m = mare(p);
  float h = 0.010 * fbm(p * 6.0, 4);
  h += 0.030 * craterField(p * 3.0, 0.0) * (1.0 - 0.7 * m);
  h += 0.012 * craterField(p * 8.0, 11.0) * (1.0 - 0.5 * m);
  h += 0.005 * craterField(p * 21.0, 23.0);
  h += 0.002 * craterField(p * 55.0, 37.0);
  return h;
}
mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  vec2 hit = sphereHit(ro, rd, vec3(0.0), 1.0);
  vec3 col;
  if (hit.x < hit.y && hit.y > 0.0){
    vec3 n0 = normalize(ro + rd * hit.x);
    vec3 p = rotY(uRot) * n0;                    // body-fixed coordinates (the Moon rotates)
    // surface normal from the height field by finite differences in the tangent plane
    vec3 t1 = normalize(cross(p, abs(p.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 t2 = cross(p, t1);
    float e = 0.0025, h0 = height(p);
    float hx = height(normalize(p + t1 * e)), hy = height(normalize(p + t2 * e));
    vec3 nb = normalize(p - t1 * (hx - h0) / e - t2 * (hy - h0) / e);
    vec3 n = transpose(rotY(uRot)) * nb;          // back to world space
    float m = mare(p);
    float albedo = mix(0.15, 0.075, m) * (0.85 + 0.3 * fbm(p * 30.0, 3)) * (1.0 + 2.5 * max(h0, 0.0));
    vec3 tint = mix(vec3(1.0, 0.97, 0.92), vec3(0.85, 0.86, 0.9), m);
    // Lommel–Seeliger law: why the full Moon looks flat instead of shaded like a ball
    float mu0 = max(dot(n, uSunDir), 0.0), mu = max(dot(n, -rd), 1e-3);
    float ls = mu0 / (mu0 + mu);
    float phase = dot(uSunDir, -rd);
    float opposition = 1.0 + 0.4 * pow(max(phase, 0.0), 40.0);   // brightening when the Sun is behind you
    col = tint * albedo * ls * 9.0 * opposition;
    col += vec3(0.35, 0.5, 0.8) * albedo * uEarthshine * max(dot(n, -uSunDir), 0.0) * 0.08;
  } else {
    col = sky(rd, 1.0) * 0.6;
    // the Sun itself
    float c = dot(rd, uSunDir);
    col += bb(5772.0) * (smoothstep(0.99996, 0.99999, c) * 400.0 + pow(max(c, 0.0), 3000.0) * 4.0);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

export default [{
  id: 'moon',
  name: 'Mặt Trăng',
  category: 'planets',
  order: 30,
  subtitle: 'Vệ tinh tự nhiên của Trái Đất',
  swatch: ['#e6e2da', '#3a3a40'],
  blurb: 'Bề mặt không có khí quyển, phủ lớp bụi regolith và hàng triệu hố va chạm. Nhờ quy luật phản xạ Lommel–Seeliger, trăng tròn trông phẳng như một chiếc đĩa thay vì một quả cầu được chiếu sáng.',
  facts: [
    ['Bán kính', '1 737,4 km'], ['Khối lượng', '7,35×10²² kg'], ['Khoảng cách tới Trái Đất', '384 400 km'],
    ['Suất phản chiếu', '0,12'], ['Chu kỳ tự quay', '27,3 ngày (khóa thủy triều)'],
  ],
  unitKm: 1737.4,
  camera: { distance: 3.4, polar: 82, azimuth: 20, min: 1.05, max: 30, fov: 45 },
  exposure: 1.0,
  bloom: 0.35,
  params: [
    { key: 'sunAngle', label: 'Góc pha (Mặt Trời)', type: 'range', min: 0, max: 180, step: 1, default: 70, fmt: v => v.toFixed(0) + '°', note: '0° = trăng tròn (Mặt Trời sau lưng bạn), 180° = trăng non.' },
    { key: 'craters', label: 'Mật độ hố va chạm', type: 'range', min: 0, max: 1, step: 0.01, default: 0.55 },
    { key: 'earthshine', label: 'Ánh Trái Đất', type: 'toggle', default: true, note: 'Ánh sáng Trái Đất phản chiếu lên phần tối của Mặt Trăng.' },
  ],
  actions: [
    { label: 'Trăng tròn', key: 'o', run: ctx => { ctx.params.sunAngle = 0; ctx.toast('Góc pha 0°: bóng đổ biến mất, các hố gần như vô hình — hiệu ứng đối pha.'); } },
  ],
  create(ctx) {
    const uniforms = {
      ...ctx.shared,
      uSunDir: { value: new THREE.Vector3(1, 0, 0) },
      uCraters: { value: 0.55 }, uEarthshine: { value: 1 }, uRot: { value: 0 },
    };
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    const camDir = new THREE.Vector3();
    return {
      material,
      update(dt, ctx) {
        const p = ctx.params;
        // phase angle is measured from the camera direction, so the requested phase holds while you orbit
        camDir.copy(ctx.camera.position).normalize();
        const side = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), camDir).normalize();
        const a = THREE.MathUtils.degToRad(p.sunAngle);
        uniforms.uSunDir.value.copy(camDir).multiplyScalar(Math.cos(a)).addScaledVector(side, Math.sin(a)).normalize();
        uniforms.uCraters.value = p.craters;
        uniforms.uEarthshine.value = p.earthshine ? 1 : 0;
        uniforms.uRot.value = ctx.time.sim * 0.002;
      },
      hud(ctx) {
        const d = ctx.camera.position.length();
        const ang = 2 * Math.asin(Math.min(1, 1 / d));
        return [
          ['Độ cao', ctx.fmt.dist((d - 1) * 1737.4e3)],
          ['Đường kính góc', THREE.MathUtils.radToDeg(ang).toFixed(1) + '°'],
          ['Phần được chiếu sáng', (50 * (1 + Math.cos(THREE.MathUtils.degToRad(ctx.params.sunAngle)))).toFixed(0) + '%'],
        ];
      },
      onClick(info, ctx) {
        const h = sphereHitJS(info.origin, info.dir);
        if (h === null) return;
        const p = info.origin.clone().addScaledVector(info.dir, h);
        const lat = THREE.MathUtils.radToDeg(Math.asin(p.y)), lon = THREE.MathUtils.radToDeg(Math.atan2(p.z, p.x));
        ctx.toast(`Điểm chạm: vĩ độ <b>${lat.toFixed(1)}°</b>, kinh độ <b>${lon.toFixed(1)}°</b>. Trọng lực bề mặt 1,62 m/s² — một bước nhảy cao gấp 6 lần trên Trái Đất.`);
      },
    };
  },
}];

function sphereHitJS(o, d) {
  const b = o.dot(d), c = o.lengthSq() - 1, h = b * b - c;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
