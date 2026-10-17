// Shared GLSL for every scene shader.
//
// COMMON_GLSL declares the uniforms the host fills every frame (camera, time, blackbody table, sky
// settings) and provides: simplex noise, fbm, hashes, a physically based blackbody colour lookup,
// the camera ray for the current pixel, and the procedural celestial sphere.

export const FULLSCREEN_VERT = /* glsl */`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

export const COMMON_GLSL = /* glsl */`
precision highp float;
#define PI 3.14159265359
#define TAU 6.28318530718

// ---- filled by the host every frame -------------------------------------------------------
uniform vec2 uRes;              // render-target size in pixels
uniform float uTime;            // simulation time (scaled by the global time-speed slider)
uniform float uWall;            // wall-clock seconds since start (unscaled)
uniform vec3 uCamPos;           // camera position, scene units
uniform vec3 uCamRight, uCamUp, uCamFwd;  // orthonormal camera basis
uniform float uTanHalfFov;
uniform sampler2D uBB;          // blackbody chromaticity table (luminance-normalised linear sRGB)
uniform float uSkyMode, uStarBright, uSkyBright;
varying vec2 vUv;

// ---- camera ray for this pixel ---------------------------------------------------------------
vec3 camRay(){
  vec2 ndc = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  return normalize(uCamFwd + uCamRight * ndc.x * uTanHalfFov * aspect + uCamUp * ndc.y * uTanHalfFov);
}

// ---- Ashima 3D simplex noise, range ~[-1, 1] -------------------------------------------------
vec4 permute(vec4 x){ return mod(((x*34.0)+1.0)*x, 289.0); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 1.0/7.0;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
// fractal sum of simplex noise, roughly [-1, 1]
float fbm(vec3 p, int oct){
  float a = 0.5, s = 0.0, n = 0.0;
  for (int i = 0; i < 8; i++){
    if (i >= oct) break;
    s += a * snoise(p); n += a;
    p = p * 2.03 + vec3(1.7, -3.1, 2.3); a *= 0.5;
  }
  return s / n;
}
vec3 hash33(vec3 p){
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.xxy + p.yxx) * p.zyx);
}
float hash13(vec3 p){ return hash33(p).x; }
// 3D cellular noise: x = distance to nearest feature point, y = to second nearest
vec2 worley(vec3 p){
  vec3 i = floor(p), f = fract(p);
  float d1 = 8.0, d2 = 8.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
    vec3 o = vec3(x, y, z);
    vec3 r = o + hash33(i + o) - f;
    float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; } else if (d < d2){ d2 = d; }
  }
  return sqrt(vec2(d1, d2));
}

// ---- physics helpers -------------------------------------------------------------------------
// Blackbody colour for temperature T (kelvin, 500 K – 100 000 K): Planck spectrum integrated
// against CIE 1931 colour-matching functions, normalised to luminance 1. Multiply by your own
// intensity (∝ T⁴ for bolometric brightness).
vec3 bb(float T){
  float u = (log(clamp(T, 500.0, 100000.0)) / log(10.0) - 2.69897) / (5.0 - 2.69897);
  // explicit LOD: no derivatives needed, so the compiler can keep bb() inside divergent branches/loops
  return textureLod(uBB, vec2(u, 0.5), 0.0).rgb;
}
float erfA(float x){ return tanh(1.2025 * x); }
// ray-sphere: returns (tNear, tFar); tNear > tFar means miss
vec2 sphereHit(vec3 ro, vec3 rd, vec3 c, float r){
  vec3 oc = ro - c; float b = dot(oc, rd); float h = b * b - dot(oc, oc) + r * r;
  if (h < 0.0) return vec2(1e9, -1e9);
  h = sqrt(h); return vec2(-b - h, -b + h);
}
float gridLine(float x, float w){ return 1.0 - smoothstep(0.0, w, abs(fract(x + 0.5) - 0.5)); }

// ---- celestial sphere ------------------------------------------------------------------------
// gUni selects which universe's sky is drawn (1 = ours, -1 = the far side of the wormhole).
float gUni = 1.0;
vec3 starLayer(vec3 dir, float scale, float prob, float sz, float seed, float gs){
  vec3 a = abs(dir); float face; vec2 uv;
  if (a.x >= a.y && a.x >= a.z){ face = dir.x > 0.0 ? 0.0 : 1.0; uv = dir.yz / a.x; }
  else if (a.y >= a.z){ face = dir.y > 0.0 ? 2.0 : 3.0; uv = dir.xz / a.y; }
  else { face = dir.z > 0.0 ? 4.0 : 5.0; uv = dir.xy / a.z; }
  vec2 g = uv * scale; vec2 id = floor(g); vec2 f = g - id;
  seed += gUni < 0.0 ? 17.0 : 0.0;
  vec3 h = hash33(vec3(id, face * 13.1 + seed));
  if (h.x > prob) return vec3(0.0);
  vec2 c = 0.25 + 0.5 * h.yz;
  vec2 dd = f - c;
  float d2 = dot(dd, dd);
  vec3 h2 = hash33(vec3(id.yx + seed * 3.7, face + seed * 1.9));
  float m = 0.025 + pow(h2.x, 14.0) * 7.0;
  float T = 2700.0 * pow(11.0, pow(h2.y, 1.7));
  return bb(T * gs) * m * exp(-d2 / (sz * sz));
}
vec3 galaxyBand(vec3 d, out float band){
  bool other = gUni < 0.0;
  vec3 pole = other ? normalize(vec3(-0.6, 0.5, 0.62)) : normalize(vec3(0.22, 1.0, -0.38));
  vec3 ctr = normalize(cross(pole, vec3(0.0, 0.0, 1.0)));
  float o = other ? 40.0 : 0.0;
  float b = dot(d, pole);
  float lc = dot(normalize(d - pole * b + 1e-5), ctr);
  band = exp(-b * b * 20.0);
  float core = exp(-b * b * 55.0) * pow(max(lc, 0.0), 5.0);
  float n1 = snoise(d * 2.3 + o) * 0.5 + snoise(d * 5.1 + o) * 0.25 + snoise(d * 11.0 + o) * 0.125;
  float dust = snoise(d * 4.0 + 7.3 + o) * 0.5 + snoise(d * 9.0 + 3.1 + o) * 0.3 + snoise(d * 21.0 + o) * 0.15;
  float lane = smoothstep(0.0, 0.5, dust) * exp(-b * b * 85.0);
  vec3 warm = other ? vec3(1.0, 0.62, 0.78) : vec3(1.0, 0.78, 0.55);
  vec3 cool = other ? vec3(0.45, 0.85, 0.95) : vec3(0.55, 0.66, 1.0);
  vec3 col = mix(cool, warm, clamp(core * 2.0 + 0.3 * max(lc, 0.0), 0.0, 1.0)) * (band * (0.35 + 0.6 * (n1 * 0.5 + 0.5)) + core * 1.7);
  col *= 1.0 - 0.82 * lane;
  float neb = snoise(d * 1.3 + 11.0 + o) * 0.6 + snoise(d * 3.2 - 4.0 + o) * 0.3;
  col += vec3(0.42, 0.12, 0.5) * pow(max(neb, 0.0), 2.2) * 0.4 + vec3(0.05, 0.28, 0.4) * pow(max(-neb, 0.0), 2.5) * 0.3;
  return col * 0.2;
}
// Background sky in direction d. gs = frequency shift factor of the observer (1 = none).
vec3 sky(vec3 d, float gs){
  float gI = clamp(gs * gs * gs * gs, 0.01, 30.0);
  vec3 col;
  if (uSkyMode < 0.5){
    float band;
    vec3 tint = bb(5800.0 * gs) / bb(5800.0);
    col = galaxyBand(d, band) * uSkyBright * tint;
    vec3 st = starLayer(d, 45.0, 0.07, 0.07, 1.0, gs) * 1.6
            + starLayer(d, 120.0, 0.10, 0.09, 2.0, gs) * 0.7
            + starLayer(d, 300.0, 0.06 + 0.5 * band, 0.12, 3.0, gs) * (0.08 + 0.3 * band);
    col += st * uStarBright;
  } else {
    float lon = atan(d.z, d.x), lat = asin(clamp(d.y, -1.0, 1.0));
    if (uSkyMode < 1.5){
      float s = PI / 12.0;
      float gl = max(gridLine(lon / s, 0.05), gridLine(lat / s, 0.05));
      float eq = 1.0 - smoothstep(0.0, 0.012, abs(lat));
      float pm = 1.0 - smoothstep(0.0, 0.012, abs(lon));
      col = vec3(0.012, 0.02, 0.04) + vec3(0.35, 0.65, 1.0) * gl * 0.35 + vec3(1.0, 0.55, 0.2) * max(eq, pm) * 0.9;
    } else {
      float s = PI / 8.0;
      float ch = mod(floor(lon / s) + floor(lat / s), 2.0);
      vec3 q = lon > 0.0 ? (lat > 0.0 ? vec3(0.95, 0.32, 0.22) : vec3(0.22, 0.5, 0.95))
                         : (lat > 0.0 ? vec3(0.95, 0.82, 0.28) : vec3(0.3, 0.85, 0.42));
      col = q * (0.06 + 0.22 * ch);
    }
    col *= bb(6500.0 * gs) / bb(6500.0);
  }
  return col * gI;
}
`;
