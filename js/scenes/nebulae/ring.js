// Tinh vân Chiếc Nhẫn (M57, NGC 6720) — image-based 3D reconstruction of a planetary nebula.
//
// One scene unit = one light-year. Default view = looking down the nebula's symmetry axis, which is
// how we see M57 from Earth (the polar axis lies within ~10–30° of the line of sight: O'Dell et al.
// 2013, AJ 145, 92; Wesson et al. 2024, MNRAS 528, 3392).
//
// How it works
//   The emission is taken from the real Hubble (2013) and JWST (2023) images, registered to one sky
//   frame (field-star astrometry, see tools in assets/ring/CREDITS.txt) with stars removed, and spread
//   along the line of sight by a physical 3D model of the shell:
//       ε_c(x, y, z) = I_c(x, y) · f_c(x, y, z) / ∫ f_c dz          (c = R, G, B channel)
//   so the projection along the axis reproduces the photograph exactly, while from any other angle
//   you see the 3D structure: a barrel ("super-ellipsoid", straight walls, rounded ends) with
//   ionisation stratification — each colour channel lives on its own nested shell (He II / [O III]
//   on the inside and filling the cavity, Hα / [N II] / PAH on the outer edge) — plus the
//   low-density halo of earlier AGB mass loss. The integrals ∫ f dz are tabulated on the CPU.
//   Procedural 3D structure (radially stretched clumps, the halo "spikes", the ~10 concentric arcs,
//   and ~20 000 cometary knots with bright cusps and shadow tails) adds depth and close-up detail
//   beyond the photographs' resolution; knots fade in only when they would be larger than a pixel.
//
// Colours: the photo's sRGB value P is inverted through the host's display chain
// (display = aces(c · exposure)^(1/2.2)), so the projected radiance displays as the original image.
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';

const ASSET = f => new URL(`../../../assets/ring/${f}`, import.meta.url).href;
const DIST_LY = 2570;                                  // Gaia DR3 parallax distance (≈ 790 pc)
const AS = DIST_LY * Math.PI / 180 / 3600;             // ly per arcsecond at that distance (0.01246)
const EXPO = 1.0;                                      // scene default exposure; the photo inversion assumes it
const D0 = 9.0, FOV0 = 12;                             // default "telescope" framing
const LUT_N = 256, LUT_MAX = 6.0;

// Per-palette layering of the three colour channels (R, G, B) on nested shells, in units of the
// ring's ridge radius: s0 = radius of the emitting layer, dw = its thickness, fill = emission filling
// the cavity inside it, halo = relative density of the AGB halo for that channel.
const PALETTES = [
  { key: 'hubble', name: 'Hubble (2013)', img: 'ring_hubble.jpg', wide: 'ring_hubble_wide.jpg', texel: 128 / 3072,
    // R = [N II] 658 + Hα, G = [O III] 502 + Hβ, B = He II 469: the classic ionisation sequence
    s0: [1.0, 0.86, 0.66], dw: [0.13, 0.17, 0.27], dwo: [0.3, 0.28, 0.34], kz: [0.65, 0.85, 1.0], cap: [0.06, 0.3, 1.0], fill: [0.0, 0.06, 0.3], halo: [0.003, 0.0015, 0.001], ring: 0,
    kAbs: [0.93, 0.96, 1.0], kGlow: [1.0, 0.62, 0.45], kHead: 0.0, kTail: [0.3, 0.06], spikes: [[31, 1], [-59, 1]], spikeLen: 7, spikeR: [0.55, 0.95],
    star: [0.72, 0.84, 1.0],
    credit: 'NASA, ESA, C. R. O’Dell (Vanderbilt) — heic1310a',
    filters: 'He II 469 · [O III] 502 · Hβ · Hα · [N II] 658 · [S II]' },
  { key: 'nircam', name: 'JWST NIRCam', img: 'ring_nircam.jpg', wide: 'ring_nircam_wide.jpg', texel: 128 / 3072,
    // R = F335M (PAH 3.3 µm, a thin ring at the outer edge), G = F300M, B = F162M/F212N (H₂ 2.12 µm, Paα wing)
    s0: [1.04, 0.9, 0.7], dw: [0.12, 0.16, 0.27], dwo: [0.3, 0.28, 0.32], kz: [0.68, 0.85, 1.0], cap: [0.06, 0.3, 1.0], fill: [0.0, 0.05, 0.22], halo: [0.004, 0.002, 0.003], ring: 0,
    kAbs: [0.5, 0.55, 0.6], kGlow: [1.0, 0.72, 0.3], kHead: 0.2, kTail: [0.08, 0.3], spikes: [[1.9, 1], [61.9, 1], [121.9, 1], [151.9, 0.3]], spikeLen: 12, spikeR: [0.85, 1.25],
    star: [0.95, 0.95, 1.0],
    credit: 'ESA/Webb, NASA, CSA, M. Barlow, N. Cox, R. Wesson — weic2320b',
    filters: 'F162M · F212N (H₂) · F300M · F335M (PAH)' },
  { key: 'miri', name: 'JWST MIRI', img: 'ring_miri.jpg', wide: 'ring_miri_wide.jpg', texel: 128 / 1024,
    // R = 21–25 µm ([O IV] 25.9 µm, hot dust: fills the cavity), G = 12–15 µm, B = 5.6–7.7 µm (PAH, H₂: outer shell + halo)
    s0: [0.55, 0.9, 1.05], dw: [0.35, 0.17, 0.14], dwo: [0.4, 0.3, 0.32], kz: [1.0, 0.85, 0.7], cap: [1.0, 0.3, 0.08], fill: [0.45, 0.05, 0.0], halo: [0.001, 0.002, 0.006], ring: 2,
    kAbs: [0.4, 0.4, 0.45], kGlow: [1.0, 0.55, 0.25], kHead: 0.15, kTail: [0.08, 0.25], spikes: [[-5.4, 1], [54.6, 1], [114.6, 1], [144.6, 0.3]], spikeLen: 6, spikeR: [1.3, 2.2],
    star: [0.75, 0.6, 0.95],
    credit: 'ESA/Webb, NASA, CSA, M. Barlow, N. Cox, R. Wesson — weic2320c',
    filters: 'F560W · F770W (PAH) · F1130W · F1800W · F2100W · F2550W' },
];

// ------------------------------------------------------------------------------------------
// GLSL
// ------------------------------------------------------------------------------------------
const FRAG = COMMON_GLSL + /* glsl */`
#ifndef NIN
#define NIN 56
#endif
#ifndef NOUT
#define NOUT 10
#endif
uniform sampler2D uImg, uWide, uKnotMap, uLUT, uRidge;
uniform highp sampler3D uNoise;
uniform float uSc, uKz, uExpo, uTexIn, uKnots, uCmax, uBright, uArcs, uProjD, uKHead;
float gModel = 0.0;                   // 0 looking down the axis (photo) → 1 off-axis (shell model)
float sq(float x){ return x * x; }
uniform vec3 uS0, uDw, uDwo, uKzc, uRingSel, uFill, uHal, uKAbs, uKGlow, uWn, uCap;
uniform float uKTailA, uKTailG;
// AGB halo, in units of the ring radius: the inner halo of loops (≲ 2×) and the thin, nearly round
// outer halo at ≈ 3.5× (115″, O'Dell+2013), with faint gas between. Must match haloProf() in JS.
float haloProf(float sh){
  return smoothstep(0.95, 1.4, sh) * (exp(-sq((sh - 1.45) / 0.45)) + 0.03 * (1.0 - smoothstep(4.1, 4.8, sh)) + 0.7 * exp(-sq((sh - 3.5) / 0.2)));
}

const float AS = ${AS.toFixed(7)};   // ly per arcsecond
const float RB = 33.0;                // mean ridge radius of the main ring, arcsec (halo reference)
const float HALO_R = 160.0;           // outer edge of the marched volume, arcsec

// per-pixel white-noise jitter of the march start (the host's film grain hides the residual noise)
float ign(){ return fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453); }
vec2 ellipsoidHit(vec3 ro, vec3 rd, vec3 ax){
  vec3 o = ro / ax, d = rd / ax;
  float a = dot(d, d), b = dot(o, d), c = dot(o, o) - 1.0, h = b * b - a * c;
  if (h < 0.0) return vec2(1e9, -1e9);
  h = sqrt(h); return vec2((-b - h) / a, (-b + h) / a);
}
// inverse of the host's ACES fit: the linear radiance that displays as y (y = sRGB^2.2)
vec3 acesInv(vec3 y){
  y = clamp(y, 0.0, 0.985);
  vec3 A = 2.51 - 2.43 * y, B = 0.03 - 0.59 * y, C = -0.14 * y;
  return (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
}
// the photograph at sky position qa (arcsec from the central star), as linear HDR radiance
vec3 photo(vec2 qa, float lod){
  float m = max(abs(qa.x), abs(qa.y));
  if (m > 159.0) return vec3(0.0);
  float lw = max(lod - log2(0.3125 / uTexIn), 0.0);
  vec3 c;
  if (m < 62.0){
    c = textureLod(uImg, qa / 128.0 + 0.5, max(lod, 0.0)).rgb;
    if (m > 56.0) c = mix(c, textureLod(uWide, qa / 320.0 + 0.5, lw).rgb, smoothstep(56.0, 62.0, m));
  } else c = textureLod(uWide, qa / 320.0 + 0.5, lw).rgb;
  return acesInv(pow(c, vec3(2.2))) * uExpo;
}

// The photograph heavily smoothed (~1.3″, wide texture mip 2), as linear HDR radiance
vec3 photoBlur(vec2 qa){
  if (max(abs(qa.x), abs(qa.y)) > 159.0) return vec3(0.0);
  return acesInv(pow(textureLod(uWide, qa / 320.0 + 0.5, 2.0).rgb, vec3(2.2))) * uExpo;
}

// Emissivity per arcsec at q (arcsec, comoving); s = shell coordinate of the dense ring's layer.
// Two descriptions of the same nebula, blended by viewing angle (gModel) and magnification:
//  • photo-column model — the photograph spread along each line of sight in proportion to the shell
//    profile, ∫ε dz = photo: exact when looking down the axis, but from other angles it would only
//    smear the photo along the axis;
//  • shell model — a true volume emissivity: each channel's layer brightness is read from the photo
//    on the equator at that layer's own radius and azimuth (so colours and their azimuthal structure
//    carry over), the AGB halo is mapped onto spheres, and isotropic 3D clumps (2:1 stretched along
//    the radius) plus the knotty inner edge of the ring give structure seen from any angle.
vec3 emis(vec3 q, float lod, out float s){
  float r = length(q.xy);
  float R = textureLod(uRidge, vec2(atan(q.y, q.x) / TAU + 0.5 / 72.0, 0.5), 0.0).r;
  float rho = r / R;
  // each channel's layer: a super-ellipsoid (s³ = ρ³ + ζ³: barrel walls, rounded ends) whose length along
  // the axis differs — the dense low-ionisation ring ([N II], H₂, PAH) is a short equatorial band,
  // the high-ionisation gas (He II, [O III]) extends into the polar lobes (O'Dell+2013)
  vec3 zc = q.z / (R * uKz * uKzc);
  vec3 az = abs(zc);
  vec3 sv = pow(vec3(rho * rho * rho) + az * az * az, vec3(1.0 / 3.0));
  s = dot(sv, uRingSel);
  float d = length(q), sh = d / RB;
  vec3 x = (sv - uS0) / mix(uDw, uDwo, step(uS0, sv));
  vec3 fs = exp(-x * x) + uFill * (1.0 - smoothstep(uS0 - 0.3, uS0 + 0.04, sv));
  float halo = haloProf(sh);
  float wm = max(gModel, smoothstep(-0.2, -1.0, lod));
  vec3 e = vec3(0.0);
  if (wm < 0.999){
    vec3 Wsh = textureLod(uLUT, vec2((min(rho, LUT_MAX) / LUT_MAX) * (1.0 - 1.0 / 256.0) + 0.5 / 256.0, 0.5), 0.0).rgb;
    float Wh = textureLod(uLUT, vec2((min(r / RB, LUT_MAX) / LUT_MAX) * (1.0 - 1.0 / 256.0) + 0.5 / 256.0, 0.5), 0.0).a;
    vec3 W = R * uKz * uKzc * Wsh + uHal * RB * Wh;
    // the photo is a projection from Earth's direction: sample it where the line of sight from the
    // default viewpoint (distance uProjD on the axis) crosses the sky plane, so the default view is exact
    vec3 ph = photo(q.xy * (uProjD / max(uProjD - q.z, 1.0)), lod);
    e = ph * (fs + uHal * halo) / max(W, vec3(1e-4)) * (1.0 - wm);
  }
  if (wm > 0.001){
    // where each layer point is read from the photo: on the equator at the layer's radius, sliding to
    // the projected position towards the poles (continuous across the axis)
    vec3 sl = min(sv, vec3(1.5));
    vec3 wq = rho * rho / max(sv * sv, vec3(1e-6));
    vec3 k = mix(vec3(1.0), sl / max(rho, 1e-3), wq);
    vec3 A = vec3(photoBlur(q.xy * k.r).r, photoBlur(q.xy * k.g).g, photoBlur(q.xy * k.b).b);
    // dense low-ionisation gas sits in the equatorial band, the polar caps of its layer are thin;
    // the high-ionisation gas fills the lobes (cap weight per channel, μ = sine of the latitude)
    vec3 mu3 = az * az * az / max(sv * sv * sv, vec3(1e-6));
    vec3 lat = mix(uCap, vec3(1.0), (1.0 - mu3) * (1.0 - mu3));
    vec3 em = A * fs * lat / (R * uKz * uKzc * uWn);
    // AGB halo: the photo beyond the bright shell, mapped onto spheres (its radial spikes become
    // radial cones, its loops become shells), plus the ~10 concentric arcs of Wesson+2024
    // (thin spherical shells every ≈1.55″, broken up in azimuth)
    float hw = smoothstep(1.3, 1.75, sh) * halo;
    if (hw > 0.0){
      if (uArcs > 0.0 && sh < 2.0){
        float w = fract(d / 1.55);
        float arc = exp(-sq((w - 0.5) / 0.12)) * smoothstep(1.3, 1.45, sh) * smoothstep(2.0, 1.6, sh);
        arc *= smoothstep(0.6, 1.4, textureLod(uNoise, q / d * 0.2 + 0.3, 0.0).r * uCmax);
        hw *= 1.0 + 2.5 * uArcs * arc;
      }
      float Wb = textureLod(uLUT, vec2((min(sh, LUT_MAX) / LUT_MAX) * (1.0 - 1.0 / 256.0) + 0.5 / 256.0, 0.5), 0.0).a;
      // the halo photo at the sphere's own radius; near the axis the azimuth is undefined, so blend to
      // the azimuthal mean there
      vec2 u2 = r > 1e-3 ? q.xy / r : vec2(1.0, 0.0);
      vec3 hv = photoBlur(u2 * d);
      float ax = smoothstep(0.35, 0.05, r / d);
      if (ax > 0.0) hv = mix(hv, 0.25 * (photoBlur(vec2(d, 0.0)) + photoBlur(vec2(-d, 0.0)) + photoBlur(vec2(0.0, d)) + photoBlur(vec2(0.0, -d))), ax);
      em += hv * hw / (RB * max(Wb, 0.05));
    }
    // isotropic clumps, only mildly (2:1) elongated along the radius
    vec3 dir = q / max(d, 1e-3);
    vec3 qr = q - dir * dot(q, dir) * 0.5;
    float n1 = textureLod(uNoise, qr / 48.0 + 0.11, 0.0).r * uCmax;
    float n2 = textureLod(uNoise, qr / 17.0 + 0.53, 0.0).r * uCmax;
    float cl = mix(1.0, n1, 0.6) * mix(1.0, n2, 0.4);
    if (lod < 0.0){
      float mg = smoothstep(0.0, -2.0, lod);
      cl *= mix(1.0, textureLod(uNoise, qr / 5.0 + 0.29, 0.0).r * uCmax, 0.6 * mg);
      if (lod < -1.0) cl *= mix(1.0, textureLod(uNoise, qr / 1.6 + 0.61, 0.0).r * uCmax, 0.45 * smoothstep(-1.0, -3.0, lod));
    }
    // the knotty inner edge of the dense ring as texture, until the individual knots are resolved
    float s0 = dot(uS0, uRingSel);
    float kz = uKnots * smoothstep(s0 - 0.3, s0 - 0.18, s) * smoothstep(s0 + 0.05, s0 - 0.04, s);
    float fpk = uTexIn * exp2(lod + 0.5);
    if (kz > 0.0 && fpk > 0.03){
      float fp = fpk;
      float spot = smoothstep(1.5, 2.6, textureLod(uNoise, q / 14.0 + 0.77, 0.0).r * uCmax);
      cl *= 1.0 - 0.6 * spot * kz * smoothstep(0.03, 0.09, fp);
    }
    e += em * cl * wm;
  }
  return e;
}

// Cometary knots (Wesson+2024: ~20 000 globules, ~0.2″ = 150 au across, n ~ 10⁵–10⁶ cm⁻³; O'Dell+2013:
// dark heads with ionised cusps facing the star and tails pointing away from it). They are far
// smaller than a march step, so the ray walks the voxel grid (DDA, 0.9″ cells) through the knot
// layer and intersects each knot analytically; the nearest few hits are composited into the march.
#define KMAX 3
float kT[KMAX], kA[KMAX]; vec3 kG[KMAX]; int kN = 0;
float ringS(vec3 q){
  float R = textureLod(uRidge, vec2(atan(q.y, q.x) / TAU + 0.5 / 72.0, 0.5), 0.0).r;
  float rho = length(q.xy) / R, z = q.z / (R * uKz * dot(uKzc, uRingSel));
  z = abs(z);
  return pow(rho * rho * rho + z * z * z, 1.0 / 3.0);
}
void knotsDDA(vec3 ro, vec3 rd, float t0, float t1, float pixA){
  const float C = 0.9;
  float t = t0;
  vec3 p0 = ro + rd * t0, cell = floor(p0 / C), st = sign(rd);
  vec3 ird = 1.0 / max(abs(rd), vec3(1e-6));
  vec3 tMax = t0 + ((cell + max(st, 0.0)) * C - p0) * st * ird;
  vec3 tDel = C * ird;
  float s0 = dot(uS0, uRingSel);
  for (int i = 0; i < 40; i++){
    vec3 h = hash33(cell * 1.31 + 7.7);
    if (h.x < uKnots * 0.66){
      vec3 c = (cell + 0.3 + 0.4 * hash33(cell + 3.1)) * C;
      float sc = ringS(c);
      if (sc > s0 - 0.28 && sc < s0 + 0.06){
        float km = textureLod(uKnotMap, c.xy / 128.0 + 0.5, 0.0).r;
        float occ = uKnots * (0.04 + 0.42 * km) * smoothstep(s0 - 0.28, s0 - 0.18, sc) * smoothstep(s0 + 0.06, s0 - 0.02, sc);
        if (h.x < occ){
          vec3 n = normalize(c);                                   // away from the star
          float kr = 0.07 + 0.12 * h.y * h.y;                      // head radius 0.07–0.19″
          vec3 oc = c - ro; float tc = dot(oc, rd);
          float b = length(oc - rd * tc);                          // impact parameter
          float fp = max(tc, 0.1) * pixA;                          // pixel footprint at the knot
          float vis = smoothstep(0.09, 0.03, fp);
          // tail: a tapering shadow cylinder along n, 6–20 head radii long
          float L = kr * (6.0 + 14.0 * h.z);
          vec3 w0 = ro - c; float a12 = dot(rd, n), d1 = dot(rd, w0), d2 = dot(n, w0);
          float den = max(1.0 - a12 * a12, 1e-4);
          float u = clamp((d2 - a12 * d1) / den, 0.0, L);
          float tr = max(dot(c + n * u - ro, rd), 0.0);
          float dl = length(ro + rd * tr - (c + n * u));
          float rt = kr * 0.85 * (1.0 - u / L);
          float tail = (1.0 - smoothstep(0.0, rt + fp, dl)) * smoothstep(kr * 0.3, kr * 1.5, u) * (1.0 - u / L);
          // irregular, lumpy heads: radius modulated around the limb
          vec3 pcl = ro + rd * tc - c;
          vec3 e1 = normalize(cross(rd, abs(rd.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
          float ph = atan(dot(pcl, cross(rd, e1)), dot(pcl, e1));
          float kre = kr * (1.0 + 0.09 * sin(2.0 * ph + 6.3 * h.z) + 0.07 * sin(3.0 * ph + 17.0 * h.x) + 0.05 * sin(5.0 * ph + 11.0 * h.y));
          float bb = clamp(b / (kre + fp), 0.0, 1.0);
          float head = 1.0 - bb * bb * bb;                         // dense core, soft photo-evaporating edge
          float a = vis * max(head * 0.95, tail * uKTailA);
          if (a > 0.01 && tc > t0 - 1.0){
            // bright cusp on the star-facing limb of the head
            vec3 pc = ro + rd * tc;
            // the ionised skin: a sharp crescent on the side facing the white dwarf
            float face = smoothstep(-0.15, 0.8, dot(normalize(pc - c + 1e-5), -n));
            float rim = smoothstep(kre * 0.6, kre * 0.95, b) * (1.0 - smoothstep(kre * 1.0, kre * 1.3 + fp, b));
            // photo-evaporation flow: a soft glow streaming off the lit face
            float flow = exp(-max(b - kre, 0.0) / (0.7 * kre + fp)) * smoothstep(kre * 0.7, kre, b);
            vec3 gas = photoBlur(c.xy);
            vec3 glow = gas * uKGlow * ((rim * 3.2 + flow * 0.9) * face + uKHead * head + uKTailG * tail * (1.0 - head)) * vis;
            kT[kN] = tc; kA[kN] = a; kG[kN] = glow; kN++;
            if (kN >= KMAX) return;
          }
        }
      }
    }
    if (tMax.x < tMax.y && tMax.x < tMax.z){ cell.x += st.x; t = tMax.x; tMax.x += tDel.x; }
    else if (tMax.y < tMax.z){ cell.y += st.y; t = tMax.y; tMax.y += tDel.y; }
    else { cell.z += st.z; t = tMax.z; tMax.z += tDel.z; }
    if (t > t1) return;
  }
}

void march(vec3 ro, vec3 rd, float a, float b, int n, float j, float pixA, inout vec3 I, inout vec3 T, inout int ki){
  if (b <= a) return;
  float dt = (b - a) / float(n);
  float t = a + j * dt;
  for (int i = 0; i < NIN; i++){
    if (i >= n || max(T.r, max(T.g, T.b)) < 0.01) break;
    vec3 q = ro + rd * t;
    float fp = max(t, 0.5) * pixA;                          // pixel footprint, arcsec
    float lod = log2(fp / uTexIn) - 0.5;
    float s;
    vec3 e = emis(q, lod, s);
    // knots passed during this step (opaque heads, dusty shadow tails, glowing cusps)
    float tn = t + dt;
    for (int k = 0; k < KMAX; k++){
      if (k < ki || k >= kN) continue;
      if (kT[k] > tn) break;
      float f = clamp((kT[k] - t) / dt, 0.0, 1.0);
      I += T * e * dt * f;  e *= 1.0 - f;
      I += T * kG[k];
      T *= 1.0 - kA[k] * uKAbs;
      ki = k + 1;
    }
    I += T * e * dt;
    t = tn;
  }
}

void main(){
  vec3 rd = camRay();
  // the camera's own direction (not the pixel's), so a wide field of view keeps the exact photo
  gModel = smoothstep(0.03, 0.28, 1.0 - abs(uCamFwd.z));
  float qs = AS * uSc;                                     // ly per comoving arcsec
  vec3 ro = uCamPos / qs;
  float pixA = 2.0 * uTanHalfFov / uRes.y;
  vec3 I = vec3(0.0), T = vec3(1.0);
  vec2 hb = sphereHit(ro, rd, vec3(0.0), HALO_R);
  if (hb.x < hb.y && hb.y > 0.0){
    float t0 = max(hb.x, 0.0), t1 = hb.y;
    float zm = 1.32 * uKz * 41.0;
    vec2 hi = ellipsoidHit(ro, rd, vec3(68.0, 68.0, zm));
    float j = ign();
    int ki = 0;
    if (hi.x < hi.y && hi.y > 0.0){
      float a = max(hi.x, t0), b = min(hi.y, t1);
      // knots only once they would be resolved (the photographs already contain them otherwise)
      if (uKnots > 0.0 && a * pixA < 0.09) knotsDDA(ro, rd, a, b, pixA);
      march(ro, rd, t0, a, NOUT, j, pixA, I, T, ki);
      march(ro, rd, a, b, NIN, j, pixA, I, T, ki);
      march(ro, rd, b, t1, NOUT, j, pixA, I, T, ki);
    } else {
      march(ro, rd, t0, t1, 2 * NOUT, j, pixA, I, T, ki);
    }
  }
  I *= uBright;
  // the procedural Milky Way and its stars, kept faint over the field the telescopes imaged (their
  // stars are drawn by the overlay at their real positions)
  vec3 toC = normalize(-ro);
  float rad = 175.0 / max(length(ro), 1.0);
  float cosA = dot(rd, toC);
  float mask = mix(0.1, 1.0, smoothstep(cos(min(rad * 0.9, 1.5)), cos(min(rad * 1.8, 3.0)), cosA));
  if (length(ro) < 175.0) mask = 0.08;
  I += sky(rd, 1.0) * 0.35 * mask * T;
  gl_FragColor = vec4(I, 1.0);
}`.replace(/LUT_MAX/g, LUT_MAX.toFixed(1));

// Stars: point sprites with the telescope's own PSF — Hubble WFC3: 4 spikes from the secondary-mirror
// spider; JWST: 6 spikes from the hexagonal segments + 2 fainter ones from the vertical strut.
// Spikes are a camera artefact, so they are drawn in screen space at the angle seen in each release.
const STAR_VERT = /* glsl */`
attribute vec3 col; attribute float sz;
uniform float uPx, uSpread; uniform vec2 uSpkR;
varying vec3 vCol; varying float vSig, vSize, vSpk;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float m = max(col.r, max(col.g, col.b));
  vSig = 0.8 + 0.12 * sz * uPx;                       // diffraction core + the saturated disc of bright stars
  vSpk = smoothstep(uSpkR.x, uSpkR.y, sz);             // only bright (saturated, large-disc) stars show spikes
  vSize = clamp(vSig * 5.0 + vSpk * sz * uPx * uSpread, 6.0, 320.0);
  gl_PointSize = vSize;
  vCol = col;
}`;
const STAR_FRAG = /* glsl */`
uniform vec4 uSpk[4];       // xy = spike direction (screen), z = amplitude
varying vec3 vCol; varying float vSig, vSize, vSpk;
void main(){
  vec2 p = (gl_PointCoord - 0.5) * vec2(1.0, -1.0) * vSize;
  float r2 = dot(p, p);
  float core = exp(-0.5 * r2 / (vSig * vSig)) + 0.03 * exp(-sqrt(r2) / (vSig * 2.5));
  float sp = 0.0, L = vSize * 0.5;
  for (int k = 0; k < 4; k++){
    float along = dot(p, uSpk[k].xy), perp = dot(p, vec2(-uSpk[k].y, uSpk[k].x));
    sp += uSpk[k].z * exp(-perp * perp / (0.35 * vSig * vSig + 0.25)) * pow(max(1.0 - abs(along) / L, 0.0), 2.5);
  }
  float edge = 1.0 - smoothstep(0.8 * L, L, sqrt(r2));
  gl_FragColor = vec4(vCol * (core + 0.12 * vSpk * sp * edge), 1.0);
}`;

// ------------------------------------------------------------------------------------------
// JS helpers
// ------------------------------------------------------------------------------------------
const toHalf = THREE.DataUtils.toHalfFloat;
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// inverse of the host's display chain for a catalogue peak value (sRGB 0–1)
function acesInv(y) {
  y = Math.min(Math.max(y, 0), 0.985);
  const A = 2.51 - 2.43 * y, B = 0.03 - 0.59 * y, C = -0.14 * y;
  return (-B + Math.sqrt(B * B - 4 * A * C)) / (2 * A);
}

// ∫ f dζ along the line of sight for every channel (shell + fill, in ζ units) and for the halo
// (in units of RB). Must match emis() in the shader.
function buildLUT(tex, pal, thick) {
  const d = tex.image.data, NZ = 360, ZM = 3.0;
  const dw = pal.dw.map(x => x * thick), wn = [1, 1, 1];
  for (let i = 0; i < LUT_N; i++) {
    const rho = i / (LUT_N - 1) * LUT_MAX;
    const acc = [0, 0, 0];
    for (let k = 0; k < NZ; k++) {
      const z = (k + 0.5) / NZ * ZM, s = Math.cbrt(rho * rho * rho + z * z * z);
      for (let c = 0; c < 3; c++) {
        const x = (s - pal.s0[c]) / (s < pal.s0[c] ? dw[c] : pal.dwo[c] * thick);
        acc[c] += Math.exp(-x * x) + pal.fill[c] * (1 - smooth(pal.s0[c] - 0.3, pal.s0[c] + 0.04, s));
      }
    }
    let ah = 0;
    for (let k = 0; k < NZ; k++) {
      const u = (k + 0.5) / NZ * 5.0, sh = Math.sqrt(rho * rho + u * u);
      ah += haloProf(sh);
    }
    for (let c = 0; c < 3; c++) d[i * 4 + c] = toHalf(2 * acc[c] * ZM / NZ);
    d[i * 4 + 3] = toHalf(2 * ah * 5.0 / NZ);
    for (let c = 0; c < 3; c++) if (Math.abs(rho - pal.s0[c]) <= LUT_MAX / (LUT_N - 1) / 2) wn[c] = 2 * acc[c] * ZM / NZ;
  }
  tex.needsUpdate = true;
  return wn;                       // ∫f dζ through each layer's tangent point (shell-model normalisation)
}

const haloProf = sh => smooth(0.95, 1.4, sh) * (Math.exp(-(((sh - 1.45) / 0.45) ** 2)) + 0.03 * (1 - smooth(4.1, 4.8, sh)) + 0.7 * Math.exp(-(((sh - 3.5) / 0.2) ** 2)));

// tileable value-noise volume with unit mean (lognormal-ish clumps), 64³
function buildNoise() {
  const N = 64, rnd = mulberry(7);
  const lat = (P) => { const a = new Float32Array(P * P * P); for (let i = 0; i < a.length; i++) a[i] = rnd(); return a; };
  const L1 = lat(16), L2 = lat(32);
  const sample = (L, P, x, y, z) => {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    let fx = x - xi, fy = y - yi, fz = z - zi;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    const at = (a, b, c) => L[((a % P + P) % P) + P * (((b % P + P) % P) + P * ((c % P + P) % P))];
    const l = (a, b, t) => a + (b - a) * t;
    return l(l(l(at(xi, yi, zi), at(xi + 1, yi, zi), fx), l(at(xi, yi + 1, zi), at(xi + 1, yi + 1, zi), fx), fy),
             l(l(at(xi, yi, zi + 1), at(xi + 1, yi, zi + 1), fx), l(at(xi, yi + 1, zi + 1), at(xi + 1, yi + 1, zi + 1), fx), fy), fz);
  };
  const v = new Float32Array(N * N * N);
  let sum = 0, mx = 0;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const g = 0.66 * sample(L1, 16, x / 4, y / 4, z / 4) + 0.34 * sample(L2, 32, x / 2, y / 2, z / 2);
    const e = Math.exp(5.0 * (g - 0.5));
    v[x + N * (y + N * z)] = e; sum += e; if (e > mx) mx = e;
  }
  const mean = sum / v.length, data = new Uint8Array(v.length);
  for (let i = 0; i < v.length; i++) data[i] = Math.round(v[i] / mx * 255);
  const tex = new THREE.Data3DTexture(data, N, N, N);
  tex.format = THREE.RedFormat; tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping;
  tex.unpackAlignment = 1; tex.needsUpdate = true;
  return { tex, cmax: mx / mean };
}
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function loadTex(url, mip = true) {
  const attempt = () => new Promise((res, rej) => new THREE.TextureLoader().load(url, t => {
    t.colorSpace = THREE.NoColorSpace;            // the shader linearises with the display gamma (2.2)
    t.generateMipmaps = mip;
    t.minFilter = mip ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.anisotropy = 4;
    res(t);
  }, undefined, () => rej(new Error('không tải được ' + url))));
  return attempt().catch(() => attempt());
}

// smooth camera flight (spherical interpolation around a moving target)
function makeFlyer() {
  let fly = null;
  const sph = new THREE.Spherical(), off = new THREE.Vector3(), tgt = new THREE.Vector3();
  return {
    to(ctx, dist, polar, azim, target = null, dur = 2.6) {
      off.copy(ctx.camera.position).sub(ctx.controls.target); sph.setFromVector3(off);
      const from = { r: sph.radius, p: sph.phi, a: sph.theta, t: ctx.controls.target.clone() };
      const to = { r: dist ?? sph.radius, p: THREE.MathUtils.degToRad(polar), a: THREE.MathUtils.degToRad(azim), t: target ? target.clone() : from.t.clone() };
      let da = to.a - from.a; da = Math.atan2(Math.sin(da), Math.cos(da)); to.a = from.a + da;
      fly = { from, to, t: 0, dur };
    },
    finish(ctx) { if (fly) { fly.t = fly.dur; this.update(ctx, 0); } },
    update(ctx, dt) {
      if (!fly) return;
      fly.t += dt; const k = Math.min(1, fly.t / fly.dur), e = k * k * (3 - 2 * k);
      // distance interpolates logarithmically so long zooms feel even
      const r = Math.exp(Math.log(fly.from.r) + (Math.log(fly.to.r) - Math.log(fly.from.r)) * e);
      sph.set(r, fly.from.p + (fly.to.p - fly.from.p) * e, fly.from.a + (fly.to.a - fly.from.a) * e);
      tgt.lerpVectors(fly.from.t, fly.to.t, e);
      ctx.controls.target.copy(tgt);
      ctx.camera.position.setFromSpherical(sph).add(tgt);
      ctx.camera.lookAt(tgt);
      if (k >= 1) fly = null;
    },
  };
}

const vi = (x, d = 1) => x.toFixed(d).replace('.', ',');
const ly = (x, d = 1) => vi(x, d) + ' năm ánh sáng';
const qualSteps = q => (q.steps <= 200 ? [40, 8] : q.steps <= 320 ? [52, 10] : q.steps <= 480 ? [68, 12] : [88, 14]);

// ------------------------------------------------------------------------------------------
// scene
// ------------------------------------------------------------------------------------------
const ring = {
  id: 'planetary-nebula',
  name: 'Tinh vân Chiếc Nhẫn',
  category: 'nebulae',
  order: 20,
  subtitle: 'Tinh vân hành tinh M57 (NGC 6720) · dựng 3D từ ảnh Hubble & JWST',
  swatch: ['#3f8fd8', '#e0672a'],
  blurb: 'Một ngôi sao cỡ Mặt Trời ở cuối đời thổi bay lớp vỏ ngoài; lõi trơ lại thành sao lùn trắng ~120 000 K, tia UV của nó ion hóa lớp vỏ. Vỏ có dạng <b>cái thùng</b> mà ta nhìn gần dọc trục nên thấy một chiếc nhẫn. Các ion <b>phân tầng</b> theo năng lượng: He II, [O III] (lam-lục) ở trong, Hα, [N II] (đỏ) ở mép ngoài, nơi có ~20 000 <b>nút khí</b> đặc. Ở đây ảnh thật của Hubble và JWST được trải ra theo chiều sâu bằng một mô hình 3D của vỏ.',
  intro: 'Đây đúng là ảnh của <b>Hubble</b>/<b>JWST</b> khi nhìn dọc trục. Bấm <b>V</b> để xoay sang ngang — chiếc nhẫn thật ra là một cái thùng rỗng — hoặc <b>N</b> để bay sát vào các nút khí.',
  facts: [
    ['Khoảng cách', '≈ 2 570 năm ánh sáng (Gaia)'], ['Chòm sao', 'Thiên Cầm (Lyra)'],
    ['Vòng sáng chính', '≈ 1,3 × 1,0 năm ánh sáng (≈ 100″ × 80″)'], ['Quầng ngoài', '≈ 2,9 năm ánh sáng (bán kính ≈ 115″)'],
    ['Sao trung tâm', 'lùn trắng ≈ 120 000 – 135 000 K, ≈ 300 L☉'], ['Tốc độ giãn nở', '≈ 20 – 35 km/s'],
    ['Tuổi', '≈ 4 000 năm'], ['Nút khí (JWST)', '≈ 20 000 · đường kính ≈ 0,2″ (150 AU)'],
    ['Cung đồng tâm', '~10 cung, cách nhau ≈ 1,5″ ↔ sao đồng hành chu kỳ ≈ 280 năm'],
    ['Ảnh gốc (CC BY 4.0)', 'Hubble: NASA, ESA, C. R. O’Dell · JWST: ESA/Webb, NASA, CSA, M. Barlow, N. Cox, R. Wesson · LBT: D. Thompson'],
  ],
  camera: { distance: D0, polar: 90, azimuth: 0, min: 0.25, max: 40, fov: FOV0, target: [0, 0, 0] },
  exposure: EXPO,
  bloom: 0.22,
  params: [
    { key: 'palette', label: 'Ảnh nguồn', type: 'seg', options: [[0, 'Hubble 2013'], [1, 'JWST NIRCam'], [2, 'JWST MIRI']], default: 0,
      note: 'Màu là màu gán cho từng bộ lọc trong ảnh công bố. Hubble: He II → lam, [O III] → lục, [N II]/Hα → đỏ. NIRCam: H₂, PAH 3,3 µm. MIRI: bụi nóng, [O IV] 25,9 µm.' },
    { key: 'elong', label: 'Độ dài theo trục', type: 'range', min: 0.6, max: 2.6, step: 0.01, default: 1.6, fmt: v => '×' + vi(v, 2),
      note: 'Từ Trái Đất không đo được chiều sâu: mọi giá trị đều cho cùng một tấm ảnh khi nhìn dọc trục. Phổ vận tốc (O’Dell 2013) cho thấy trục cực dài hơn vòng xích đạo.' },
    { key: 'thick', label: 'Độ dày vỏ', type: 'range', min: 0.5, max: 2, step: 0.01, default: 1, fmt: v => '×' + vi(v, 2),
      note: 'Độ dày các tầng ion hóa — chỉ thấy khác khi nhìn nghiêng.' },
    { key: 'knots', label: 'Nút khí', type: 'range', min: 0, max: 1, step: 0.01, default: 0.6, fmt: v => Math.round(v * 100) + '%',
      note: 'Khối khí trung hòa đặc ở mặt ion hóa: đầu tối, viền sáng hướng về sao, đuôi bóng hướng ra ngoài. Hiện khi đủ gần.' },
    { key: 'age', label: 'Tuổi (giãn nở)', type: 'range', min: 1500, max: 8000, step: 50, default: 4000, fmt: v => Math.round(v).toLocaleString('vi-VN') + ' năm',
      note: 'Vỏ nở đồng dạng; mật độ giảm nên độ sáng bề mặt giảm, sau ~20 000 năm tinh vân tan vào môi trường liên sao.' },
  ],
  actions: [
    { label: 'Nhìn dọc trục', key: 'r', title: 'Hướng nhìn từ Trái Đất: chiếc nhẫn', run: ctx => ring._fly?.(ctx, 0) },
    { label: 'Nhìn ngang', key: 'v', title: 'Nhìn vuông góc trục: cái thùng', run: ctx => ring._fly?.(ctx, 1) },
    { label: 'Cận cảnh nút khí', key: 'n', title: 'Bay sát mép trong của vòng', run: ctx => ring._fly?.(ctx, 2) },
    { label: 'Đổi ảnh', key: 'p', title: 'Hubble → NIRCam → MIRI', run: ctx => {
      ctx.params.palette = (ctx.params.palette + 1) % 3;
      ctx.toast(['<b>Hubble 2013</b> (ánh sáng khả kiến): lõi lam-lục là He II và [O III]; vòng đỏ-cam là [N II] và Hα; chấm tối là nút khí.',
        '<b>JWST NIRCam</b> (1,6 – 3,4 µm): nút khí sáng vì H₂; viền đỏ ngoài cùng là PAH; các “gai” toả ra trong quầng.',
        '<b>JWST MIRI</b> (5,6 – 25,5 µm): lõi đỏ-cam là bụi nóng và [O IV]; quầng xanh là PAH và H₂ — lộ rõ các cung đồng tâm.'][ctx.params.palette]);
    } },
  ],
  create(ctx) {
    const [nin, nout] = qualSteps(ctx.quality);
    const lutTex = new THREE.DataTexture(new Uint16Array(LUT_N * 4), LUT_N, 1, THREE.RGBAFormat, THREE.HalfFloatType);
    lutTex.minFilter = lutTex.magFilter = THREE.LinearFilter; lutTex.wrapS = lutTex.wrapT = THREE.ClampToEdgeWrapping;
    const ridgeTex = new THREE.DataTexture(new Uint16Array(72).fill(toHalf(33)), 72, 1, THREE.RedFormat, THREE.HalfFloatType);
    ridgeTex.minFilter = ridgeTex.magFilter = THREE.LinearFilter; ridgeTex.wrapS = THREE.RepeatWrapping; ridgeTex.needsUpdate = true;
    const noise = buildNoise();
    const black = new THREE.DataTexture(new Uint8Array(4), 1, 1); black.needsUpdate = true;

    const U = {
      ...ctx.shared,
      uImg: { value: black }, uWide: { value: black }, uKnotMap: { value: black }, uLUT: { value: lutTex }, uRidge: { value: ridgeTex },
      uNoise: { value: noise.tex }, uCmax: { value: noise.cmax },
      uSc: { value: 1 }, uKz: { value: 1.6 }, uExpo: { value: 1 / EXPO }, uTexIn: { value: 128 / 2048 }, uKnots: { value: 0.6 },
      uBright: { value: 1 }, uArcs: { value: 1 }, uProjD: { value: D0 / AS }, uKHead: { value: 0 }, uKTailA: { value: 0.3 }, uKTailG: { value: 0 }, uWn: { value: new THREE.Vector3(1, 1, 1) }, uCap: { value: new THREE.Vector3(0.1, 0.3, 1) },
      uS0: { value: new THREE.Vector3() }, uDw: { value: new THREE.Vector3() }, uDwo: { value: new THREE.Vector3() }, uKzc: { value: new THREE.Vector3() }, uRingSel: { value: new THREE.Vector3(1, 0, 0) }, uFill: { value: new THREE.Vector3() }, uHal: { value: new THREE.Vector3() },
      uKAbs: { value: new THREE.Vector3() }, uKGlow: { value: new THREE.Vector3() },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
      defines: { NIN: nin, NOUT: nout },
    });

    // ---- overlay: field stars + the central white dwarf, one sprite set per telescope ----
    const overlay = new THREE.Group();
    const starSets = [];
    const camDefault = new THREE.Vector3(0, 0, D0);
    const spkU = PALETTES.map(p => {
      const a = [];
      for (let k = 0; k < 4; k++) {
        const s = p.spikes[k];
        a.push(s ? new THREE.Vector4(Math.cos(s[0] * Math.PI / 180), Math.sin(s[0] * Math.PI / 180), s[1], 0) : new THREE.Vector4(1, 0, 0, 0));
      }
      return a;
    });
    const starMat = (pi) => new THREE.ShaderMaterial({
      uniforms: { uPx: { value: 4 }, uSpread: { value: PALETTES[pi].spikeLen }, uSpk: { value: spkU[pi] }, uSpkR: { value: new THREE.Vector2(...PALETTES[pi].spikeR) } },
      vertexShader: STAR_VERT, fragmentShader: STAR_FRAG,
      blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
    });
    const buildStars = (meta) => {
      PALETTES.forEach((p, pi) => {
        // catalogue rows: [x″, y″, peak sRGB (r,g,b), disc radius ″, traced spike length ″, local sky/nebula level]
        const list = (meta.stars[p.key === 'hubble' ? 'hst' : p.key === 'nircam' ? 'nir' : 'miri'] || [])
          .map(s => ({ x: s[0], y: s[1], pk: s[2], r: s[3], bg: s[5] ?? 0 }));
        // stars beyond the telescope field, from the wide ground-based (LBT) frame
        for (const w of meta.stars.wide || []) if (Math.max(Math.abs(w[0]), Math.abs(w[1])) > 62) {
          const v = Math.min(w[2] + 0.2, 0.88);
          list.push({ x: w[0], y: w[1], pk: [v, v, v], r: Math.min(w[3] * 0.42, 0.95), bg: 0 });
        }
        // drop detections that are pieces of a brighter star's diffraction pattern
        const big = list.filter(s => s.r > 0.9);
        const stars = list.filter(s => !big.some(b => b !== s && b.r > s.r && Math.hypot(b.x - s.x, b.y - s.y) < 5 * b.r));
        stars.push({ x: 0, y: 0, pk: p.star.map(x => x * 0.92), r: 0.45, bg: 0, centre: true });   // the white dwarf
        const pos = [], col = [], sz = [];
        const v = new THREE.Vector3();
        const lin = x => acesInv(Math.pow(Math.min(Math.max(x, 0), 0.99), 2.2)) / EXPO;
        for (const s of stars) {
          // fixed direction on the sky: where the star appears from the default (Earth) viewpoint
          if (s.centre) v.set(0, 0, 0);
          else v.set(s.x * AS, s.y * AS, 0).sub(camDefault).normalize().multiplyScalar(1500).add(camDefault);
          pos.push(v.x, v.y, v.z);
          // radiance the photo had on top of the (inpainted) background, through the inverse display chain
          const c = s.pk.map(x => Math.max(lin(x) - lin(s.bg), 0));
          const sat = Math.max(...s.pk) > 0.9;
          // the photos were resampled coarser than the telescopes' pixels: small (faint) stars lose peak
          // brightness; saturated stars were clipped, so they are brighter than the photo says
          const kf = s.centre ? 0.9 : Math.min(Math.max(Math.pow(s.r / 1.1, 2.0), 0.1), 1.0) * (sat ? 1 + 3 * smooth(0.6, 1.6, s.r) : 1);
          col.push(...c.map(x => x * 1.6 * kf));
          sz.push(Math.max(s.r, 0.12));
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setAttribute('col', new THREE.Float32BufferAttribute(col, 3));
        g.setAttribute('sz', new THREE.Float32BufferAttribute(sz, 1));
        const pts = new THREE.Points(g, starMat(pi));
        pts.frustumCulled = false; pts.visible = pi === 0;
        overlay.add(pts); starSets.push(pts);
      });
    };

    // ---- assets ----
    const texs = PALETTES.map(() => ({ img: null, wide: null }));
    let meta = null;
    const ready = Promise.all([
      fetch(ASSET('ring_meta.json')).then(r => { if (!r.ok) throw new Error('ring_meta.json ' + r.status); return r.json(); }),
      loadTex(ASSET('ring_knots.png')),
      ...PALETTES.flatMap((p, i) => [
        loadTex(ASSET(p.img)).then(t => { texs[i].img = t; }),
        loadTex(ASSET(p.wide)).then(t => { texs[i].wide = t; }),
      ]),
    ]).then(([m, km]) => {
      meta = m;
      const rd = ridgeTex.image.data;
      for (let i = 0; i < 72; i++) rd[i] = toHalf(m.ridge[i]);
      ridgeTex.needsUpdate = true;
      U.uKnotMap.value = km;
      buildStars(m);
      for (const t of texs) { ctx.renderer.initTexture(t.img); ctx.renderer.initTexture(t.wide); }
      applyPalette(ctx.params.palette ?? 0, true);
      const dbg = new URLSearchParams(location.search).get('ringAction');   // snapshot testing of the camera actions
      if (dbg !== null) { ring._fly(ctx, +dbg); flyer.finish?.(ctx); }
    });

    let curPal = -1, lutKey = '';
    const applyPalette = (pi, force = false) => {
      if (pi === curPal && !force) return;
      curPal = pi;
      const p = PALETTES[pi];
      if (texs[pi].img) { U.uImg.value = texs[pi].img; U.uWide.value = texs[pi].wide; }
      U.uTexIn.value = p.texel;
      U.uS0.value.fromArray(p.s0); U.uKzc.value.fromArray(p.kz); U.uCap.value.fromArray(p.cap); U.uRingSel.value.set(+(p.ring === 0), +(p.ring === 1), +(p.ring === 2)); U.uFill.value.fromArray(p.fill); U.uHal.value.fromArray(p.halo);
      U.uKAbs.value.fromArray(p.kAbs); U.uKHead.value = p.kHead; U.uKTailA.value = p.kTail[0]; U.uKTailG.value = p.kTail[1]; U.uKGlow.value.fromArray(p.kGlow);
      starSets.forEach((s, i) => { s.visible = i === pi; });
      lutKey = '';
    };

    const flyer = makeFlyer();
    ring._fly = (c, mode) => {
      const d = c.camera.position.distanceTo(c.controls.target);
      if (mode === 0) { flyer.to(c, Math.max(d, 4), 90, 0, new THREE.Vector3()); c.toast('Nhìn dọc trục — đúng hướng ta thấy từ Trái Đất: thành thùng chồng lên nhau theo đường nhìn → một chiếc nhẫn.'); }
      else if (mode === 1) { flyer.to(c, Math.max(d, 7), 90, 90, new THREE.Vector3()); c.toast('Nhìn ngang trục: hai bức tường sáng của “cái thùng”; hai đầu mỏng và lam vì chứa khí bị ion hóa cao (He II).'); }
      else {
        const sc = c.params.age / 4000, R = 31 * AS * sc;
        flyer.to(c, 0.55 * sc, 72, 18, new THREE.Vector3(-R * 0.93, R * 0.36, 0), 3.2);
        c.toast('Mép trong của vòng: các nút khí đặc có đầu tối, viền sáng phía sao lùn trắng và đuôi bóng tối hướng ra ngoài.');
      }
    };

    const camDir = new THREE.Vector3(), sz = new THREE.Vector2();
    const sm = { kz: 1.6, thick: 1 };
    return {
      material, overlay, ready,
      setQuality(q) { const [a, b] = qualSteps(q); material.defines.NIN = a; material.defines.NOUT = b; material.needsUpdate = true; },
      update(dt, c) {
        const p = c.params, k = 1 - Math.exp(-dt * 5);
        applyPalette(p.palette);
        sm.kz += (p.elong - sm.kz) * k; sm.thick += (p.thick - sm.thick) * k;
        const pal = PALETTES[curPal];
        const key = curPal + ':' + sm.thick.toFixed(3);
        if (key !== lutKey) { U.uWn.value.fromArray(buildLUT(lutTex, pal, sm.thick)); lutKey = key; }
        U.uDw.value.set(pal.dw[0] * sm.thick, pal.dw[1] * sm.thick, pal.dw[2] * sm.thick);
        U.uDwo.value.set(pal.dwo[0] * sm.thick, pal.dwo[1] * sm.thick, pal.dwo[2] * sm.thick);
        U.uKz.value = sm.kz;
        const sc = p.age / 4000;
        U.uSc.value = sc;
        U.uProjD.value = D0 / (AS * sc);
        // fixed shell mass: n ∝ R⁻³, emission measure ∝ n²·L ∝ R⁻⁵ — strongly softened (R⁻¹·⁵) for display;
        // in reality the shell is also more ionisation-bounded when young, which caps its brightness
        U.uBright.value = Math.pow(sc, -1.5);
        U.uKnots.value = p.knots;
        // star sprites: PSF size in screen pixels as in the release images at the default framing
        c.renderer.getDrawingBufferSize(sz);
        const px = sz.y / (2 * Math.tan(FOV0 * Math.PI / 360) * D0) * AS;
        for (const s of starSets) s.material.uniforms.uPx.value = px;
        flyer.update(c, dt);
      },
      hud(c) {
        const p = c.params, sc = p.age / 4000;
        camDir.copy(c.camera.position);
        const d = camDir.length();
        const ang = THREE.MathUtils.radToDeg(Math.acos(Math.min(1, Math.abs(camDir.normalize().z))));
        const R = 33 * AS * sc, v = R * 9.461e12 / (p.age * 3.156e7);
        const pal = PALETTES[curPal < 0 ? 0 : curPal];
        return [
          ['Camera cách sao lùn trắng', ly(d, 2)],
          ['Góc giữa đường nhìn và trục', ang.toFixed(0) + '° ' + (ang < 25 ? '(thấy nhẫn)' : ang > 60 ? '(thấy thùng)' : '')],
          ['Bán kính vòng sáng', ly(R, 2) + ' (' + Math.round(33 * sc) + '″)'],
          ['Tốc độ giãn nở trung bình', '≈ ' + Math.round(v) + ' km/s'],
          ['Ảnh ' + pal.name, pal.filters],
          ['Nguồn ảnh', pal.credit],
        ];
      },
    };
  },
};

export default [ring];
