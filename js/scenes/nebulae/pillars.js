// Những cột trụ của sự sáng tạo (M16, Tinh vân Đại Bàng) — an image-based volume.
//
// The canonical view reproduces the real releases: the star-free Hubble 2014 WFC3 mosaic (heic1501a) and the
// JWST 2022 NIRCam mosaic (weic2216b, resampled onto the Hubble frame with the published WCS, refined by
// matching stars) are the emission source, extruded into 3D along the rays of the canonical camera — the
// "2.5D" image-based technique NASA/STScI visualisers use for flights into Hubble images:
//
//   * every point p of the volume projects from the canonical eye C0 = (0, 0, D0) onto the image → uv;
//   * a per-pixel dust column τ(uv) = −ln(1 − α) (α = dust matte cut from the photo's colour and
//     brightness) is spread across a slab |z − zc| < h, concentrated in ~0.05 ly skins on both faces;
//     zc (depth of each pillar) follows McLeod et al. 2015 (MUSE): P1a behind NGC 6611 with its tip tilted
//     toward us, P1b / P2 / P3 in front of the cluster, P3 nearest; h comes from the in-plane distance to
//     the pillar edge (round cross-sections, ≤ 1.3 ly; the base cloud thickens downward, ≤ 1.8 ly; depth and
//     thickness are smoothed over ~0.4 ly so junctions ramp instead of forming ledges); below the frame a
//     procedural parent molecular cloud takes over (off-axis views only);
//   * the photo is split into three canonical layers: the far wall of the H II cavity (photo where the sky is
//     clear, inpainted where the pillars hide it), an emission-only photo-evaporation glow in a 0.3 ly shell
//     around the dust, and the dust source function E, solved per pixel so that
//     E·α + G·(1 − α/2) + S·(1 − α) = photo exactly from C0;
//   * off the canonical line of sight the hidden sides of the slabs use the locally averaged dust colour,
//     blurred photo and 3D turbulence (zero mean along the canonical ray, so the photograph is unchanged);
//     sub-texel noise takes over when the camera flies closer than the image resolution, and the head of
//     the left pillar (EGGs) has a native-resolution crop (0.04″/px ≈ 0.0011 ly/px).
//
// Stars are not in the textures: they were detected, fitted (disc + spikes, as each release renders them)
// and removed, and are drawn as instanced sprites at 3D positions — Hubble 4-spike / JWST 6+2-spike —
// dimmed by the same dust column between camera and star; the NIRCam's embedded protostars sit inside
// the pillars. Fainter NIRCam stars stay in the texture (beyond the mosaic they are synthesised).
//
// One scene unit = 1 light-year. Image scale: heic1501a 4.53′ × 4.72′ at 5 700 ly → 7.51 × 7.84 ly.
// Assets and licences: assets/pillars/CREDITS.txt (ESA/Hubble, ESA/Webb — CC BY 4.0).
import * as THREE from 'three';
import { COMMON_GLSL, FULLSCREEN_VERT } from '../../glsl/common.js';

const ASSET = f => new URL(`../../../assets/pillars/${f}`, import.meta.url).href;

// geometry of the image frame (must match tools that produced assets/pillars, see CREDITS.txt)
const FW = 7.5131;                  // frame width, ly (4.53′ at 5 700 ly)
const FH = FW * 2136 / 2048;        // frame height, ly
const LYPX = FW / 2048;             // ly per texel of the 2048-px textures
const D0 = 12.4;                    // canonical eye: (0, 0, D0), looking at −z
const BRECT = [-1.5 * FW, -FH, 3 * FW, 2 * FH];   // extent of the background sheets (x0, y0, w, h)
// JWST NIRCam footprint inside the Hubble frame (centre, axis angle, half sizes, ly)
const JW_C = [-0.185, -1.492], JW_ANG = 37.54 * Math.PI / 180, JW_HALF = [3.49, 6.05];

// ------------------------------------------------------------------------------------------
// fullscreen volume
// ------------------------------------------------------------------------------------------
const FRAG = COMMON_GLSL + /* glsl */`
#ifndef STEPS
#define STEPS 88
#endif
uniform sampler2D tP;        // palette photo (star-free), sRGB display values
uniform sampler2D tX;        // r = canonical dust alpha, g = ionisation-front mask, b = empty-space distance/0.6 ly
uniform sampler2D tB;        // background sheet, extended 3W × 2H, display values
uniform sampler2D tD;        // dust colour: the dust source function averaged over dusty pixels (for off-axis sides)
uniform sampler2D tPd;       // native-resolution crop of the head of the left pillar (EGGs), same palette
uniform vec4 uDRect;         // that crop in image uv: (u0, v0, du, dv)
uniform sampler2D tG;        // quarter resolution, half float (ly): r = slab half-thickness h, g = slab centre zc,
                             // b, a = z-range occupied by any slab within 0.3 ly (for empty-space skipping)
uniform float uPal;          // 0 Hubble 2014, 1 natural colour, 2 JWST NIRCam
uniform float uKap;          // dust optical-depth scale of this palette (1 visible, 0.3 near-IR)
uniform float uDust, uIon, uExpo;
uniform vec3 uDustTint, uFringe;   // per palette, luminance 1: shadowed dust, ionisation fringe (off-axis surfaces)
uniform float uSideCap;            // brightest dust surface brightness used on reconstructed sides
uniform vec4 uBRect;
const float FW = ${FW.toFixed(5)}, FH = ${FH.toFixed(5)}, D0 = ${D0.toFixed(2)}, TEXEL = ${LYPX.toFixed(6)};
const float GZ = 0.3, RW = 22.0;
const vec3 NGC6611 = vec3(2.5, 7.8, 0.0);   // the ionising O stars (see stars.json), ly        // the photo-evaporation glow lives within this distance (ly) of the dust

float ign(){ return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))); }
// photo display value → scene radiance: undo the host's display transform (gamma 2.2 ∘ ACES ∘ exposure)
vec3 toLin(vec3 c){
  vec3 y = pow(clamp(c, 0.0, 0.985), vec3(2.2));
  vec3 A = y * 2.43 - 2.51, B = y * 0.59 - 0.03, C = y * 0.14;
  return (-B - sqrt(max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A) / uExpo;
}
// natural colour from the Hubble narrow bands (R = [S II] 673, G = Hα+[N II] 657, B = [O III] 502 nm):
// Hα, [N II], [S II] → deep red, Hβ = Hα/2.86 (case B) → blue, [O III] → teal
vec3 natural(vec3 l){
  return l.g * vec3(1.0, 0.03, 0.05) * 1.15 + l.r * vec3(1.0, 0.03, 0.04) * 0.4
       + l.g / 2.86 * vec3(0.05, 0.38, 1.0) * 0.95 + l.b * vec3(0.03, 0.85, 0.62) * 0.5;
}
vec3 palLin(vec3 c){ vec3 l = toLin(c); return uPal > 0.5 && uPal < 1.5 ? natural(l) : l; }
// photograph at image position uv; lr = mip level of the 2048-px texture (negative when magnified):
// close to the head of P1 switch to the native-resolution crop (3.3× finer)
vec3 photo(vec2 uv, float lr){
  vec3 c = textureLod(tP, uv, max(lr, 0.0)).rgb;
  if (lr < 1.0){
    vec2 d = (uv - uDRect.xy) / uDRect.zw;
    if (d.x > 0.0 && d.x < 1.0 && d.y > 0.0 && d.y < 1.0){
      float w = smoothstep(0.0, 0.05, min(min(d.x, 1.0 - d.x), min(d.y, 1.0 - d.y))) * smoothstep(1.0, 0.2, lr);
      c = mix(c, textureLod(tPd, d, max(lr + 1.727, 0.0)).rgb, w);
    }
  }
  return c;
}
vec2 projUV(vec3 p){ return p.xy * (D0 / (D0 - p.z)) / vec2(FW, FH) + 0.5; }
// smooth background of the H II cavity at frame-plane position q (ly): the inpainted sheet, extended
// beyond the photographed frame with large-scale nebular structure that fades away from the pillars
vec3 sheetBg(vec2 q, float lod){
  vec2 ub = (q - uBRect.xy) / uBRect.zw;
  vec3 b = palLin(textureLod(tB, ub, max(lod - 2.0, 0.0)).rgb);
  vec2 uv = q / vec2(FW, FH) + 0.5;
  float o = max(max(-uv.x, uv.x - 1.0), max(-uv.y, uv.y - 1.0));
  if (o > 0.0){
    float outF = smoothstep(0.0, 0.08, o);
    float n = fbm(vec3(q * 0.33, 3.7), 4), n2 = snoise(vec3(q * 1.4, 9.1));
    b *= max(1.0 + outF * (1.1 * n + 0.18 * n2), 0.15);
    b *= 1.0 - 0.5 * smoothstep(0.1, 1.3, o);
  }
  return b;
}
// Split the photograph at image position uv into the three canonical layers:
//   S  far wall of the cavity (photo where the sky is clear, inpainted background behind dust),
//   G  emission-only photo-evaporation glow around the dust (photo minus background near the pillars),
//   E  source function of the dust, solved so that  E·aC + G·(1 − aC/2) + S·(1 − aC) = photo.
void decomp(vec2 uv, float lr, vec4 X, out vec3 S, out vec3 G, out vec3 E, out float aC, out float front){
  float lod = max(lr, 0.0);
  vec3 P = palLin(photo(uv, lr));
  vec3 B = sheetBg((uv - 0.5) * vec2(FW, FH), lod);
  float a = min(X.r, 0.993);
  aC = 1.0 - pow(1.0 - a, uKap);
  float wz = smoothstep(GZ, 0.05, X.b * 0.6) * (1.0 - smoothstep(0.3, 0.85, a));
  G = wz * max(P - B, 0.0);
  S = mix(P - G, B, smoothstep(0.02, 0.35, aC));
  E = clamp((P - G * (1.0 - 0.5 * aC) - S * (1.0 - aC)) / max(aC, 0.03), 0.0, 8.0);
  front = X.g;
}

// The parent molecular cloud below the photographed frame (never seen from the canonical view): a diffuse,
// clumpy dust body under the base, densest a little below the frame and fading out with noise, lit from above
// by the H II region. Returns the dust density (τ_V per ly, before uDust·uKap) and its source function.
float parentCloud(vec3 p, out vec3 Ep){
  vec3 q = p - vec3(-0.3, -0.5 * FH, 0.6);
  float db = -q.y;                                           // ly below the frame
  Ep = vec3(0.0);
  if (abs(q.x) > 4.2 || abs(q.z) > 3.7 || db > 4.2 || db < -0.95) return -1.0;   // outside even with the noise
  float sh = smoothstep(3.4, 1.6, abs(q.x) + 0.7 * snoise(p * 0.45 + 3.0))
           * smoothstep(3.0, 1.0, abs(q.z) + 0.6 * snoise(p * 0.45 + 9.0))
           * smoothstep(3.5, 1.4, db + 0.5 * snoise(p * 0.6 + 4.0));
  Ep = vec3(0.0);
  if (sh < 0.01) return -1.0;
  float n = fbm(vec3(p.x * 0.8, p.y * 0.55, p.z * 0.8) + 61.0, 3);
  float n2 = snoise(vec3(p.x * 2.4, p.y * 0.6, p.z * 2.4) + 5.0);            // streaks along the flow
  sh *= smoothstep(-0.7, 0.35, db + 0.6 * n + 0.15 * n2);                     // ragged top, rising into the frame
  float dens = sh * smoothstep(-0.15, 0.5, n + 0.3 * smoothstep(0.0, 1.2, db) + 0.15 * n2) * 4.0;
  float lit = 0.12 + 0.95 * exp(-max(db + 0.3, 0.0) * 1.4);
  Ep = uDustTint * uSideCap * lit * max(0.5 + 0.45 * n2 + 0.5 * n, 0.1) * pow(max(uIon, 0.0), 0.8);
  return dens;
}

void main(){
  vec3 rd = camRay(), ro = uCamPos;
  float pixAng = 2.0 * uTanHalfFov / uRes.y;
  vec3 col = vec3(0.0); float T = 1.0;
  float ionG = pow(max(uIon, 0.0), 0.8), ionF = pow(max(uIon, 0.0), 1.6);
  vec3 S, G, E; float aC, front;

  // the dusty volume lives in this box (frame stretched by the extrusion, base cloud continuing below)
  vec3 bmin = vec3(-0.5 * FW * 1.3, -0.5 * FH - 3.6, -3.9), bmax = vec3(0.5 * FW * 1.3, 0.5 * FH * 1.3, 3.4);
  vec3 ib = 1.0 / (rd + vec3(equal(rd, vec3(0.0))) * 1e-6);
  vec3 ta = (bmin - ro) * ib, tb = (bmax - ro) * ib;
  vec3 t1v = min(ta, tb), t2v = max(ta, tb);
  float tN = max(max(max(t1v.x, t1v.y), t1v.z), 0.0), tF = min(min(t2v.x, t2v.y), t2v.z);
  if (tN < tF){
    float t = tN + 0.03 * ign();
    for (int i = 0; i < STEPS; i++){
      if (t > tF || T < 0.01) break;
      vec3 p = ro + rd * t;
      vec2 uv = projUV(p);
      // lateral: outside the frame (left/right/top) there is no dust; below it the base cloud continues
      float outX = max(-uv.x, uv.x - 1.0), outY = uv.y - 1.0;
      if (outX > 0.0 || outY > 0.0){ t += max(max(outX * FW, outY * FH) * 0.8, 0.05); continue; }
      vec4 g = textureLod(tG, uv, 0.0);
      vec3 Ep; float pcd = -1.0;
      if (uv.y < 0.12){
        // inside the frame only off-axis (the canonical view keeps the photograph)
        float ob0 = uv.y < 0.0 ? 1.0 : smoothstep(0.004, 0.09, 1.0 - dot(rd, normalize(p - vec3(0.0, 0.0, D0))));
        if (ob0 > 0.01){ pcd = parentCloud(p, Ep); if (pcd > 0.0) pcd *= ob0; }
      }
      // empty-space skipping, conservative for any ray direction: within 0.2 ly sideways every slab lies in
      // [g.b, g.a]; outside that z-range the ray may advance until it moves 0.2 ly sideways or reaches it
      float dlo = g.b - p.z, dhi = p.z - g.a;
      if (dlo > 0.0 || dhi > 0.0){
        bool toward = dlo > 0.0 ? rd.z > 0.0 : rd.z < 0.0;
        float tz = toward ? max(dlo, dhi) / max(abs(rd.z), 1e-3) : 1e3;
        float stp = clamp(min(0.2 / max(length(rd.xy), 1e-3), tz), 0.02, 1.5);
        if (pcd >= 0.0){ stp = min(stp, 0.12); float a = 1.0 - exp(-pcd * uDust * uKap * stp); col += T * Ep * a; T *= 1.0 - a; }
        t += stp;
        continue;
      }
      // angle between this ray and the canonical ray through the same point: 0 → the photograph, 1 → ≳ 25° off-axis.
      // Everything procedural below is weighted by it, so the canonical view stays exactly the photograph.
      float oc = 1.0 - dot(rd, normalize(p - vec3(0.0, 0.0, D0)));
      float obl = smoothstep(0.004, 0.09, oc);
      // off-axis: an extruded outline would be identical at every depth (straight edges running away from the
      // camera). Warp the image lookup with depth-dependent turbulence, stretched along the photo-evaporation
      // flow (≈ vertical, toward NGC 6611), so outlines wander and fray through the depth of each pillar.
      if (obl > 0.05){
        float dzc = min(abs(p.z - g.g), 2.5);
        vec3 sq = vec3(p.x * 1.6, p.y * 0.3, p.z * 1.6);
        float wn = snoise(sq + 5.0);
        vec2 w = vec2(wn, 0.3 * sin(6.2832 * wn + p.y * 2.0));
        uv += obl * w * (0.04 + 0.1 * dzc) / vec2(FW, FH);
        g = textureLod(tG, uv, 0.0);
      }
      // below the photographed frame the foot of the pillars merges into the parent molecular cloud: the
      // bottom row's slab continues downward, thinning out with large-scale noise over ~3 ly (never seen
      // from the canonical view, whose rays all stay inside the frame)
      float below = 1.0;
      if (uv.y < 0.03){
        float nb = fbm(vec3(p.x * 0.55, p.y * 0.4, p.z * 0.55) + 31.0, 3);
        float db = max(-uv.y, 0.0) * FH;                          // ly below the frame
        if (db > 0.0){
          // lateral wander + clumps so the clamped bottom row does not extrude into straight columns
          uv.x += (snoise(vec3(p.x * 0.6, p.y * 0.25, p.z * 0.6) + 51.0) * 0.6 + 0.3 * snoise(p * 1.4 + 7.0)) * min(db, 2.0) / FW;
          g = textureLod(tG, uv, 0.0);
          nb += 0.5 * snoise(p * 0.9 + 13.0);
        }
        below = smoothstep(-0.42, 0.03, uv.y + 0.16 * nb) * (db > 0.0 ? 0.5 + 0.5 * smoothstep(1.5, 0.0, db + 0.6 * nb) : 1.0);
      }
      float h = g.r * below, zc = g.g;
      float lr = log2(max(t * pixAng, 1e-6) / TEXEL) - 0.5, lod = max(lr, 0.0);
      lr += 4.0 * smoothstep(0.0, -0.12, uv.y); lod = max(lr, 0.0);   // below the frame: only the blurred bottom row
      vec4 X = textureLod(tX, uv, lod);
      float dq = X.b * 0.6;
      float shell = dq < GZ ? GZ : 0.0;
      float zone = max(h * 1.25, h + shell);
      float dz = abs(p.z - zc) - zone;
      if (zone < 0.004 || dz > 0.0){
        float skip = zone < 0.004 ? (dq - GZ) * 0.8 : dz * mix(0.3, 0.8, abs(rd.z));
        float stp = clamp(skip, 0.015, 0.25);
        if (pcd >= 0.0){ stp = min(stp, 0.12); float a = 1.0 - exp(-pcd * uDust * uKap * stp); col += T * Ep * a; T *= 1.0 - a; }
        t += stp;
        continue;
      }
      // seen off-axis, the sides of an extruded photo would show streaks: away from the front face use
      // reconstructed dust (shadowed dust colour, fibrous striations, cluster-facing fringe) instead
      float n1 = snoise(p * 1.7 + vec3(3.1, 0.0, 7.7));
      // lumpy slab faces; off-axis also large-scale relief so the broad base is not a flat-faced plinth
      float relief = (0.3 + 0.45 * obl) * snoise(vec3(p.x * 0.8, p.y * 0.55, p.z * 0.8) + 11.0);
      float hsk = h * (1.0 + (0.22 + 0.25 * obl) * n1);
      float u = (p.z - zc + relief * h) / max(hsk, 1e-3);
      float side = obl * (1.0 - smoothstep(0.35, 0.95, u));
      // fibrous striations parallel to the evaporation flow (≈ vertical): ridged noise stretched along y
      float stri = 1.0;
      if (side > 0.05){
        float r1 = 1.0 - abs(snoise(vec3(p.x * 13.0, p.y * 1.3, p.z * 13.0) + 3.0));
        stri = 0.2 + 1.6 * r1 * r1 * r1;
      }
      float dskin = h > 0.004 ? abs(1.0 - abs(u)) * hsk : 0.3;
      float dt = clamp((0.35 * dskin + 0.01) / max(abs(rd.z), 0.35), 0.01, 0.06);
      if (pcd >= 0.0){ float a = 1.0 - exp(-pcd * uDust * uKap * dt); col += T * Ep * a; T *= 1.0 - a; }
      // off the canonical line of sight, everything behind the front skin is seen through it at another angle:
      // use a blurred photo there (otherwise small features, e.g. faint stars in the NIRCam frame, smear in depth)
      float blr = 1.5 * side + 2.0 * smoothstep(0.0004, 0.01, oc) * (1.0 - smoothstep(0.7, 0.95, u));
      decomp(uv, lr + blr, blr > 0.05 ? textureLod(tX, uv, lod + blr) : X, S, G, E, aC, front);
      // which way does the surface here face? (only needed off-axis) — outward normal from the in-plane
      // gradient of the slab thickness (inside the outline) or of the distance to the dust (in the glow shell),
      // and the position across the slab; lit = facing the O stars of NGC 6611
      float facing = 1.0;
      if (obl > 0.05 && (side > 0.01 || front > 0.05 || shell > 0.0)){
        vec2 e = vec2(16.0 / 2048.0, 16.0 / 2136.0);
        float a0 = textureLod(tX, uv, 4.0).r, ax = textureLod(tX, uv + vec2(e.x, 0.0), 4.0).r, ay = textureLod(tX, uv + vec2(0.0, e.y), 4.0).r;
        vec2 gr = -vec2(ax - a0, ay - a0);                      // points away from the dust
        vec2 nxy = gr / max(length(gr), 1e-4) * clamp(length(gr) * 12.0, 0.0, 1.0);
        float nz = dq > 0.01 ? 0.6 * clamp((p.z - zc) / max(h + GZ, 0.05), -1.0, 1.0) : clamp(u, -1.0, 1.0);
        vec3 n = normalize(vec3(nxy * sqrt(max(1.0 - nz * nz, 0.0)), nz) + vec3(0.0, 0.0, 1e-4));
        vec3 L = normalize(NGC6611 - p);
        // also shadowed if dust lies between this point and the cluster (a head shading its own neck)
        vec2 uvL = projUV(p + L * 0.9);
        float shade = uvL.y < 1.0 ? smoothstep(0.65, 0.2, textureLod(tX, uvL, 3.0).r) : 1.0;
        facing = smoothstep(-0.05, 0.55, dot(n, L)) * mix(1.0, shade, 0.7);
      }
      if (obl > 0.05 && front > 0.05){
        // off-axis, the photo's ionisation fronts keep their brightness but take the fringe colour and fade where
        // the surface turns away from the cluster
        float le = dot(E, vec3(0.2126, 0.7152, 0.0722));
        E = mix(E, uFringe * le * (0.35 + 0.8 * facing), obl * front * 0.85);
      }
      if (side > 0.01){
        // reconstructed side: shadowed dust colour (no rim colours from the photo), brighter where it faces
        // the cluster, with a thin pale ionisation fringe there
        vec3 Ed = palLin(textureLod(tD, uv, 0.0).rgb);
        float ld = min(dot(Ed, vec3(0.2126, 0.7152, 0.0722)), uSideCap);
        vec3 Es = uDustTint * ld * (0.45 + 0.9 * facing) * stri + uFringe * uSideCap * 1.6 * facing * front;
        E = mix(E, Es, side);
      }
      float kc = (D0 - p.z) / length(vec3(0.0, 0.0, D0) - p);   // canonical rays are not exactly along z
      float ion = mix(ionG, ionF, front);
      // emission-only glow: a shell around the slab, normalised so the canonical column equals G
      // (seen off-axis a glow that is constant along z would smear into streaks: give it 3D clumps — zero mean
      // along the canonical line of sight, so the photograph is unchanged)
      float clump = obl > 0.05 ? 1.0 + obl * 0.85 * snoise(vec3(p.x * 9.0, p.y * 2.5, p.z * 9.0) + 21.0) : 1.0;
      if (obl > 0.05){
        // off-axis the photo-evaporation glow keeps its brightness but takes the real fringe colour, and only
        // on surfaces that face the cluster (the flow streams off the lit side toward the stars)
        float lg = dot(G, vec3(0.2126, 0.7152, 0.0722));
        G = mix(G, uFringe * lg * 1.4 * facing, obl);
      }
      if (shell > 0.0 && max(G.r, max(G.g, G.b)) > 1e-4){
        float hg = 0.6 * h + 0.6 * GZ, ug = (p.z - zc) / hg;
        col += T * G * (max(1.0 - ug * ug, 0.0) * 0.75 / hg * kc * dt * ion * clump);
      }
      if (h > 0.004 && aC > 0.0005){
        // procedural 3D structure: lumpy slab surfaces + sub-texel detail when the camera is close
        // dust profile across the slab: most of the column sits in a ~0.05 ly skin on both faces (so a close or
        // oblique camera sees a surface, not a smear of the photo through the depth) + a weak interior
        float hs = hsk;
        float du = clamp(0.05 / max(hs, 1e-3), 0.04, 0.6);
        float prof = abs(u) < 1.0 ? (0.1 + 0.8 * exp(-(1.0 - abs(u)) / du) / (2.0 * du * (1.0 - exp(-1.0 / du)))) / hs : 0.0;
        float mag = TEXEL / max(t * pixAng, 1e-5);               // >1: the photo is magnified on screen
        float det = smoothstep(1.2, 5.0, mag);
        float fine = 1.0;
        if (det > 0.0){
          vec3 q = p * 70.0;
          fine = 1.0 + det * (0.75 * snoise(q) + 0.35 * snoise(q * 2.7 + 4.0));
        }
        float tau = -log(1.0 - min(1.0 - pow(1.0 - aC, 1.0 / uKap), 0.993)) * below;   // canonical column (visible)
        float rho = tau * prof * kc * uDust * max(fine, 0.0) * (0.6 + 0.4 * clump) * mix(1.0, stri, obl * 0.85);
        float a = 1.0 - exp(-rho * uKap * dt);
        if (a > 1e-5){
          // the photo shows the faces turned toward C0; the far faces of P2/P3 (in front of NGC 6611)
          // face the O stars and are brighter, the far face of P1a (behind them) is in shadow
          vec3 Ee = E * ion;
          if (side > 0.01) Ee *= mix(1.0, 0.55 + 0.6 * stri, side);
          if (det > 0.0) Ee *= 1.0 + det * 0.3 * snoise(p * 160.0 + 2.0);
          col += T * Ee * a;
          T *= 1.0 - a;
        }
      }
      t += dt;
    }
  }
  // the emitting wall of the H II cavity: a sphere of radius RW around the pillars, painted with the
  // photo's sky as seen from the canonical eye (so it lines up exactly behind the pillars from C0)
  if (T > 0.003){
    vec2 hs = sphereHit(ro, rd, vec3(0.0), RW);
    vec3 hp = ro + rd * hs.y;
    vec3 s;
    if (hp.z < D0 - 1.0){
      vec2 q = hp.xy * (D0 / (D0 - hp.z));
      float pz = D0 / (D0 - hp.z);
      float lod = log2(max(hs.y * pixAng / (TEXEL / pz), 1.0));
      vec2 uv = q / vec2(FW, FH) + 0.5;
      if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0){
        decomp(uv, max(lod - 0.5, 0.0), textureLod(tX, uv, max(lod - 0.5, 0.0)), S, G, E, aC, front);
        float inF = smoothstep(0.0, 0.004, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
        s = mix(sheetBg(q, lod), S, inF);
      } else s = sheetBg(q, lod);
      s = mix(s, sheetBg(vec2(1e3), 0.0), smoothstep(D0 - 8.0, D0 - 1.0, hp.z));
    } else s = sheetBg(vec2(1e3), 0.0);
    col += T * s * pow(max(uIon, 0.0), 0.9);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// stars: instanced screen-aligned quads, PSF + diffraction spikes, dust occlusion in the vertex shader
// ------------------------------------------------------------------------------------------
const STAR_VERT = /* glsl */`
attribute vec3 aPos, aS;                 // position (ly); spike strength per palette
attribute vec4 aH, aN, aJ;               // per palette: peak radiance × colour (rgb), disc width σ (photo px)
attribute float aKind;
uniform float uPal, uKap, uDust, uPxW;
uniform vec2 uScreen;
uniform sampler2D tA, tG;   // aux (r = canonical dust alpha), geometry
varying vec2 vOff; varying vec3 vA; varying vec3 vSS;
const float FW = ${FW.toFixed(5)}, FH = ${FH.toFixed(5)}, D0 = ${D0.toFixed(2)};
vec2 projUV(vec3 p){ return p.xy * (D0 / (D0 - p.z)) / vec2(FW, FH) + 0.5; }
float dens(vec3 p){
  vec2 uv = projUV(p);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  float below = uv.y < 0.0 ? smoothstep(-0.3, -0.01, uv.y) : 1.0;
  vec4 g = textureLod(tG, uv, 0.0);
  float h = g.r * below;
  if (h < 0.004) return 0.0;
  float u = (p.z - g.g) / h;
  float a = min(textureLod(tA, uv, 1.0).r, 0.993) * below;
  return -log(1.0 - a) * max(1.0 - u * u, 0.0) * 0.75 / h;
}
// dust transmittance along camera → star (same density model as the volume, without the fine noise)
float transmit(vec3 a, vec3 b){
  vec3 d = b - a; float L = length(d); vec3 rd = d / L;
  vec3 bmin = vec3(-0.5 * FW * 1.3, -0.5 * FH - 3.6, -3.9), bmax = vec3(0.5 * FW * 1.3, 0.5 * FH * 1.3, 3.4);
  vec3 ib = 1.0 / (rd + vec3(equal(rd, vec3(0.0))) * 1e-6);
  vec3 ta = (bmin - a) * ib, tb = (bmax - a) * ib;
  float t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
  float t1 = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), L);
  if (t1 <= t0) return 1.0;
  float dt = (t1 - t0) / 28.0, tau = 0.0;
  for (int i = 0; i < 28; i++) tau += dens(a + rd * (t0 + (float(i) + 0.5) * dt));
  return exp(-tau * dt * uKap * uDust);
}
void main(){
  vec4 F = uPal < 0.5 ? aH : (uPal < 1.5 ? aN : aJ);
  float S = uPal < 0.5 ? aS.x : (uPal < 1.5 ? aS.y : aS.z);
  vec4 c = projectionMatrix * viewMatrix * vec4(aPos, 1.0);
  float A = max(F.r, max(F.g, F.b));
  if (A <= 0.0 || c.w <= 0.02){ gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  // brightness ∝ 1/d² relative to the canonical eye (only nearby stars change), times dust transmittance
  vec3 C0 = vec3(0.0, 0.0, D0);
  float d0 = length(aPos - C0), d = length(aPos - cameraPosition);
  float k = clamp(d0 * d0 / (d * d), 0.0, 400.0) * transmit(cameraPosition, aPos);
  vec3 col = F.rgb * k;
  float A0 = A; A *= k;
  if (A < 2e-4){ gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float sig = F.a;
  S = min(S, 0.25 + 1.5 * smoothstep(0.8, 3.0, A0));    // spikes only where the photo shows them
  float sat = smoothstep(2.0, 2.9, A0) * smoothstep(2.5, 7.0, sig);   // only big saturated stars show long faint spike tails
  vA = col; vSS = vec3(sig, S, sat);
  // quad radius in photo px: where the disc, the wing or the spikes fall below ~0.003 (scene radiance)
  float thr = 0.003;
  float se = sqrt(sig * sig + uPxW * uPxW / 12.0);
  float rd = se * sqrt(2.0 * log(max(A / thr, 1.0)));
  float rw = 3.0 * sig * sqrt(max(pow(0.03 * A / thr, 0.667) - 1.0, 0.0));
  float lq = uPal > 1.5 ? 18.0 : 9.0, q = uPal > 1.5 ? 0.1 : 0.12;    // long faint tail of the spikes
  float rs = S > 0.0 ? max(1.6 * sig * log(max(S * A / thr, 1.0)), lq * sig * log(max(q * sat * S * A / thr, 1.0))) : 0.0;
  float r = clamp(max(max(rd, rw), rs) / uPxW + 1.5, 2.0, uScreen.y * 0.4);
  vOff = position.xy * r;
  c.xy += position.xy * r * 2.0 / uScreen * c.w;
  gl_Position = c;
}`;

const STAR_FRAG = /* glsl */`
uniform float uPxW, uSpk;     // photo px per screen px; spike pattern 0 = Hubble ×, 1 = JWST 6+2
varying vec2 vOff; varying vec3 vA; varying vec3 vSS;
// one diffraction spike pair through the centre: exponential core + long faint tail, Gaussian across
float spike(vec2 v, float ang, float sig, float w, float q, float lq){
  vec2 d = vec2(cos(ang), sin(ang));
  float a = abs(dot(v, d)), c = v.x * d.y - v.y * d.x;
  return (exp(-a / (1.6 * sig)) + q * exp(-a / (lq * sig))) * exp(-c * c / (2.0 * w * w));
}
void main(){
  vec2 v = vOff * uPxW;                       // offset in photo pixels
  float r2 = dot(v, v);
  float sig = vSS.x, S = vSS.y, sat = vSS.z;
  bool jw = uSpk > 0.5;
  // the release's star disc (its size grows with brightness), averaged over the screen pixel
  float pv = uPxW * uPxW / 12.0;
  float se2 = sig * sig + pv;
  float I = sig * sig / se2 * exp(-r2 / (2.0 * se2)) + 0.03 * pow(1.0 + r2 / (9.0 * sig * sig), -1.5);
  float w0 = jw ? 0.7 : 0.8, we = sqrt(w0 * w0 + pv);
  float sp = 0.0;
  if (jw){
    // JWST: six spikes from the hexagonal primary (37° + k·60° in this frame) + two fainter strut spikes
    for (int k = 0; k < 3; k++) sp += spike(v, radians(37.0 + 60.0 * float(k)), sig, we, 0.1 * sat, 18.0);
    sp += 0.35 * spike(v, radians(127.0), sig, we, 0.1 * sat, 10.0);
  } else {
    // Hubble WFC3: four spikes from the secondary-mirror spider (±45° in this frame)
    sp = spike(v, radians(46.0), sig, we, 0.12 * sat, 9.0) + spike(v, radians(134.0), sig, we, 0.12 * sat, 9.0);
  }
  sp *= S * w0 / we;
  vec3 col = vA * I;
  col += vA * sp * (jw ? vec3(0.85, 0.95, 1.2) : vec3(1.0));     // NIRCam spikes look bluer: sharper at short λ
  gl_FragColor = vec4(col, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------
const stepsFor = q => (q.steps <= 200 ? 64 : q.steps <= 320 ? 80 : q.steps <= 480 ? 96 : 112);
const vi = (x, d = 1) => x.toFixed(d).replace('.', ',');
// Colours for the reconstructed (off-axis) surfaces, measured in the photos (linear, luminance 1): median dust
// away from the fronts; the ionisation fringe — Hubble/JWST: the pale blue-white / teal haze off the lit
// tips; natural colour: the same narrow-band fluxes through natural(). cap = 1.6 × median dust brightness.
const lum3 = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
const norm3 = c => c.map(x => x / lum3(c));
const naturalJS = ([r, g, b]) => [g * 1.15 + r * 0.4 + g / 2.86 * 0.05 + b * 0.015, g * 0.0345 + r * 0.012 + g / 2.86 * 0.361 + b * 0.425,
  g * 0.0575 + r * 0.016 + g / 2.86 * 0.95 + b * 0.31];
const DUST_H = [0.0817, 0.0491, 0.0372], FRINGE_H = [0.62, 0.9, 1.15];
const PAL_SURF = [
  { dust: norm3(DUST_H), fringe: norm3(FRINGE_H), cap: 1.6 * lum3(DUST_H) },
  { dust: norm3(naturalJS(DUST_H)), fringe: norm3(naturalJS([1.0, 1.03, 0.9])), cap: 1.6 * lum3(naturalJS(DUST_H)) },
  { dust: norm3([0.128, 0.0789, 0.0822]), fringe: norm3([0.7, 0.85, 1.4]), cap: 1.6 * 0.0896 },
];
const PAL = [
  { name: 'Hubble 2014 (màu tổng hợp)', short: 'Hubble · [S II] đỏ, Hα lục, [O III] lam', kap: 1, spk: 0 },
  { name: 'Màu tự nhiên', short: 'Hα đỏ thẫm · [O III] lục lam · Hβ lam', kap: 1, spk: 0 },
  { name: 'JWST NIRCam (hồng ngoại gần)', short: 'JWST 0,9–4,4 µm · bụi ấm và PAH', kap: 0.3, spk: 1 },
];
// tip of the left pillar (P1) — where Hester et al. (1996) found the EGGs — and a viewpoint in the gap
const P1_TIP = new THREE.Vector3(-0.785, 2.89, -1.0);
const TIP_DIR = new THREE.Vector3(0, 0, D0).sub(P1_TIP).normalize();
const VIEWS = {
  home: { pos: new THREE.Vector3(0, 0, D0), tgt: new THREE.Vector3(0, 0, 0) },
  gap: { pos: new THREE.Vector3(1.6, 0.4, 4.2), tgt: new THREE.Vector3(-0.6, 1.2, -1.3) },
  tip: { pos: P1_TIP.clone().addScaledVector(TIP_DIR, 1.9).add(new THREE.Vector3(0.22, 0.1, 0)), tgt: P1_TIP.clone().add(new THREE.Vector3(0.05, -0.22, 0)) },
};

// all textures are plain RGB (no alpha channel: browsers decode RGBA images premultiplied)
function loadTex(url) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, tex => {
    tex.colorSpace = THREE.NoColorSpace;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true; tex.anisotropy = 4;
    tex.needsUpdate = true;
    resolve(tex);
  }, undefined, () => reject(new Error(`không tải được ${url}`))));
}
// slab geometry: 512 × 534 × (h, zc, zlo, zhi) as little-endian uint16 over [0,2.5], [−4,4], [−9,9], [−9,9] ly
async function loadGeo(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const W = 512, H = 534, u16 = new Uint16Array(buf), out = new Uint16Array(W * H * 4);
  const toH = THREE.DataUtils.toHalfFloat;
  for (let i = 0; i < W * H; i++) {
    out[i * 4] = toH(u16[i * 4] / 65535 * 2.5);
    out[i * 4 + 1] = toH(u16[i * 4 + 1] / 65535 * 8 - 4);
    out[i * 4 + 2] = toH(u16[i * 4 + 2] / 65535 * 18 - 9);
    out[i * 4 + 3] = toH(u16[i * 4 + 3] / 65535 * 18 - 9);
  }
  const tex = new THREE.DataTexture(out, W, H, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = false;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function bbApprox(T) {   // luminance-normalised blackbody chromaticity (linear sRGB)
  const t = T / 100;
  const r = T < 6600 ? 1 : Math.min(1.3, 1.29 * Math.pow(t - 60, -0.133));
  const g = T < 6600 ? Math.max(0, 0.39 * Math.log(t) - 0.63) : Math.min(1.3, 1.13 * Math.pow(t - 60, -0.0755));
  const b = T < 1900 ? 0 : T < 6600 ? Math.max(0, 0.54 * Math.log(t - 10) - 1.2) : 1;
  const c = [r, g, b].map(x => Math.pow(x, 2.2)); const L = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  return c.map(x => x / L);
}

// real catalogue + synthetic field stars beyond the photographed data (same brightness distribution).
// Row layout (assets/pillars/stars.json): x y z | Hubble A·rgb σ S | natural A·rgb σ S | JWST A·rgb σ S | kind
function buildStars(data) {
  const R = rng(6611);
  const rows = data.stars.slice();
  const inFrame = (x, y) => Math.abs(x) < FW / 2 && Math.abs(y) < FH / 2;
  const inJW = (x, y) => {
    const dx = x - JW_C[0], dy = y - JW_C[1];
    const a = dx * Math.cos(JW_ANG) + dy * Math.sin(JW_ANG), b = -dx * Math.sin(JW_ANG) + dy * Math.cos(JW_ANG);
    return Math.abs(a) < JW_HALF[0] && Math.abs(b) < JW_HALF[1];
  };
  const X0 = -1.15 * FW, X1 = 1.15 * FW, Y0 = -0.8 * FH, Y1 = 0.8 * FH, area = (X1 - X0) * (Y1 - Y0);
  const place = () => {
    const x = X0 + R() * (X1 - X0), y = Y0 + R() * (Y1 - Y0);
    const z = R() < 0.75 ? -Math.exp(Math.log(30) + R() * Math.log(20)) : -4.5 + R() * 8.5;
    return [x, y, z, (D0 - z) / D0];
  };
  const NONE = [0, 0, 0, 1, 0];
  // Hubble / natural
  const nH = Math.round(data.stars.filter(r => r[3] + r[4] + r[5] > 0 && r[18] < 3).length / (FW * FH) * area);
  for (let i = 0; i < nH; i++) {
    const [x, y, z, s] = place();
    if (inFrame(x, y)) continue;
    const f = data.poolH[Math.floor(R() * data.poolH.length)];
    const c = bbApprox(Math.exp(Math.log(3200) + R() * Math.log(4))), L = (f[0] + f[1] + f[2]) / 3 * 0.9;
    rows.push([x * s, y * s, z, ...f, c[0] * L, c[1] * L, c[2] * L, f[3], f[4], ...NONE, 0]);
  }
  // JWST: the faint stars stay in the NIRCam texture, so synthesise them only where there is no data
  const covered = 0.83 * FW * FH;
  const nJ = Math.round(data.poolJ.length * data.poolRatio / covered * area);
  for (let i = 0; i < nJ; i++) {
    const [x, y, z, s] = place();
    if (inFrame(x, y) && inJW(x, y)) continue;
    rows.push([x * s, y * s, z, ...NONE, ...NONE, ...data.poolJ[Math.floor(R() * data.poolJ.length)], 0]);
  }
  // …and the brighter NIRCam sprites need a population outside the mosaic too
  const bright = data.stars.filter(r => r[13] + r[14] + r[15] > 0 && r[18] < 2);
  const nB = Math.round(bright.length / covered * area);
  for (let i = 0; i < nB; i++) {
    const [x, y, z, s] = place();
    if (inFrame(x, y) && inJW(x, y)) continue;
    rows.push([x * s, y * s, z, ...NONE, ...NONE, ...bright[Math.floor(R() * bright.length)].slice(13, 18), 0]);
  }
  const n = rows.length;
  const pos = new Float32Array(n * 3), aS = new Float32Array(n * 3), kind = new Float32Array(n);
  const aH = new Float32Array(n * 4), aN = new Float32Array(n * 4), aJ = new Float32Array(n * 4);
  rows.forEach((r, i) => {
    pos.set(r.slice(0, 3), i * 3);
    aH.set(r.slice(3, 7), i * 4); aN.set(r.slice(8, 12), i * 4); aJ.set(r.slice(13, 17), i * 4);
    aS[i * 3] = r[7]; aS[i * 3 + 1] = r[12]; aS[i * 3 + 2] = r[17];
    kind[i] = r[18];
  });
  const geo = new THREE.InstancedBufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  geo.setAttribute('aPos', new THREE.InstancedBufferAttribute(pos, 3));
  geo.setAttribute('aH', new THREE.InstancedBufferAttribute(aH, 4));
  geo.setAttribute('aN', new THREE.InstancedBufferAttribute(aN, 4));
  geo.setAttribute('aJ', new THREE.InstancedBufferAttribute(aJ, 4));
  geo.setAttribute('aS', new THREE.InstancedBufferAttribute(aS, 3));
  geo.setAttribute('aKind', new THREE.InstancedBufferAttribute(kind, 1));
  geo.instanceCount = n;
  return geo;
}

// ------------------------------------------------------------------------------------------
// scene definition
// ------------------------------------------------------------------------------------------
const def = {
  id: 'emission-nebula',
  name: 'Những cột trụ của sự sáng tạo',
  category: 'nebulae',
  order: 10,
  subtitle: 'Vùng H II · M16, Tinh vân Đại Bàng — dựng 3D từ ảnh Hubble & JWST',
  swatch: ['#e9c58a', '#1d4a5c'],
  blurb: 'Ba cột khí và bụi lạnh cao vài năm ánh sáng trong Tinh vân Đại Bàng, bị bức xạ cực tím của các sao O trong cụm NGC 6611 gọt dần. Ở mặt hướng về các sao, khí bị ion hóa và <b>bốc hơi</b> thành một viền sáng mảnh; ở đỉnh cột còn sót lại những “ngón tay” và <b>EGG</b> — khối khí đặc đang tự co lại thành sao. Cảnh này dựng khối 3D trực tiếp từ ảnh thật của Hubble và JWST.',
  intro: 'Góc nhìn mặc định chính là ảnh thật. Kéo để xoay — các cột tách ra theo chiều sâu (P3 gần ta nhất, đỉnh P1 nằm sau cụm sao). Đổi <b>Bảng màu</b> sang <b>JWST</b> để thấy bụi trong suốt và các sao mới sinh.',
  facts: [
    ['Khoảng cách', '≈ 5 700 năm ánh sáng (1,74 kpc, Gaia)'], ['Chòm sao', 'Cự Xà (Serpens Cauda)'],
    ['Chiều cao cột trái (P1)', '≈ 4–5 năm ánh sáng (hình chiếu)'], ['Cụm sao ion hóa', 'NGC 6611 · ~1–2 triệu năm'],
    ['Sao nóng nhất', 'HD 168076 · O3.5 V, ~45 000 K'], ['Dòng quang bốc hơi', '≈ 8 km/s · ~70 M☉ mỗi triệu năm'],
    ['Tuổi thọ ước tính', '≈ 3 triệu năm (McLeod et al. 2015, MUSE)'],
    ['Ảnh gốc', 'NASA, ESA/Hubble & Hubble Heritage Team (2014); NASA, ESA, CSA, STScI — J. DePasquale, A. Koekemoer, A. Pagan (JWST 2022) · CC BY 4.0'],
  ],
  camera: { distance: D0, polar: 90, azimuth: 0, min: 0.25, max: 60, fov: 35, target: [0, 0, 0] },
  exposure: 1.0,
  bloom: 0.15,
  params: [
    { key: 'palette', label: 'Bảng màu', type: 'seg', options: [[0, 'Hubble 2014'], [1, 'Màu tự nhiên'], [2, 'JWST NIRCam']], default: 0,
      note: 'Hubble: [S II] → đỏ, Hα → lục, [O III] → lam (màu tổng hợp). Tự nhiên: Hα đỏ thẫm, [O III] lục lam. JWST: bụi chỉ tắt sáng ~10 % so với ánh sáng nhìn thấy (A<sub>K</sub>/A<sub>V</sub> ≈ 0,1).' },
    { key: 'dust', label: 'Mật độ bụi', type: 'range', min: 0, max: 2.5, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: '×1 khớp đúng ảnh thật. Bụi dày hơn che khuất cả các sao phía sau; mỏng đi thì lộ ra thành phía xa của khoang H II.' },
    { key: 'ion', label: 'Cường độ cụm sao ion hóa', type: 'range', min: 0, max: 2.5, step: 0.01, default: 1, fmt: v => '×' + v.toFixed(2),
      note: 'Thông lượng photon ion hóa (λ < 91,2 nm) từ NGC 6611. Viền mặt ion hóa sáng ∝ Φ, tốc độ mất khối ∝ Φ<sup>1/2</sup>.' },
  ],
  actions: [
    { label: 'Đổi bảng màu', key: 'p', run: ctx => {
      ctx.params.palette = (ctx.params.palette + 1) % 3;
      ctx.toast(['<b>Hubble 2014</b>: ba vạch hẹp gán vào ba kênh màu — màu “tổng hợp” giúp tách các nguyên tố.',
        '<b>Màu tự nhiên</b>: mắt người sẽ thấy Hα đỏ thẫm pha Hβ và [O III] — tinh vân hồng đỏ, các cột tối nâu.',
        '<b>JWST NIRCam</b>: hồng ngoại gần xuyên qua bụi — cột trở nên trong mờ, lộ ra hàng nghìn sao và các sao mới sinh đỏ rực.'][ctx.params.palette]);
    } },
    { label: 'Bay vào khe', key: 'g', title: 'Bay vào khoảng trống giữa các cột', run: ctx => ctx.scene._go?.(ctx, 'gap') },
    { label: 'Đỉnh cột & EGG', key: 'e', title: 'Đến đỉnh cột trái, nơi có các khối EGG', run: ctx => ctx.scene._go?.(ctx, 'tip') },
    { label: 'Góc ảnh gốc', key: 'r', title: 'Về đúng góc nhìn của kính thiên văn', run: ctx => ctx.scene._go?.(ctx, 'home') },
    { label: 'Tắt cụm sao', key: 'o', title: 'Tắt / bật nguồn UV của NGC 6611', run: ctx => {
      ctx.params.ion = ctx.params.ion > 0.05 ? 0 : 1;
      ctx.toast(ctx.params.ion ? 'NGC 6611 sáng trở lại: khí bị ion hóa, các viền bốc hơi bừng sáng.' : 'Không còn tia UV: khí H II tái hợp và tối dần (~10⁵ năm) — chỉ còn bóng bụi trước nền sao.');
    } },
  ],
  create(ctx) {
    const U = {
      ...ctx.shared,
      tP: { value: null }, tX: { value: null }, tB: { value: null }, tG: { value: null }, tD: { value: null }, tPd: { value: null },
      uDRect: { value: new THREE.Vector4(555.80 / 2048, 1 - 502.06 / 2136, (1019.77 - 555.80) / 2048, (502.06 - 226.56) / 2136) },
      uPal: { value: 0 }, uKap: { value: 1 }, uDust: { value: 1 }, uIon: { value: 1 }, uExpo: { value: 1 },
      uDustTint: { value: new THREE.Vector3(1, 1, 1) }, uFringe: { value: new THREE.Vector3(1, 1, 1) }, uSideCap: { value: 0.1 },
      uBRect: { value: new THREE.Vector4(...BRECT) },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: U, vertexShader: FULLSCREEN_VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
      defines: { STEPS: stepsFor(ctx.quality) },
    });
    const SU = {
      uPal: { value: 0 }, uKap: { value: 1 }, uDust: { value: 1 }, uPxW: { value: 3 }, uSpk: { value: 0 },
      uScreen: { value: new THREE.Vector2(1280, 720) }, tA: { value: null }, tG: { value: null },
    };
    const starMat = new THREE.ShaderMaterial({
      uniforms: SU, vertexShader: STAR_VERT, fragmentShader: STAR_FRAG,
      blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
    });
    const overlay = new THREE.Group();
    const tex = {};
    const ready = Promise.all([
      loadTex(ASSET('hst.webp')), loadTex(ASSET('jwst.webp')), loadTex(ASSET('matte.webp')), loadGeo(ASSET('geo.bin')),
      loadTex(ASSET('bg_hst.jpg')), loadTex(ASSET('bg_jw.jpg')), loadTex(ASSET('dust_hst.jpg')), loadTex(ASSET('dust_jw.jpg')),
      loadTex(ASSET('hst_head.webp')), loadTex(ASSET('jwst_head.webp')),
      fetch(ASSET('stars.json')).then(r => r.json()),
    ]).then(([h, j, x, g, bh, bj, dh, dj, hh, jh, stars]) => {
      Object.assign(tex, { h, j, g, bh, bj, dh, dj, hh, jh });
      U.tG.value = g; U.tX.value = x; SU.tG.value = g; SU.tA.value = x;
      const mesh = new THREE.Mesh(buildStars(stars), starMat);
      mesh.frustumCulled = false;
      overlay.add(mesh);
    });

    // camera flights (position + orbit target), smoothstep in time
    let fly = null;
    const go = (c, which, instant = false) => {
      const v = VIEWS[which];
      if (instant) { c.camera.position.copy(v.pos); c.controls.target.copy(v.tgt); c.camera.lookAt(v.tgt); fly = null; return; }
      fly = { p0: c.camera.position.clone(), t0: c.controls.target.clone(), p1: v.pos, t1: v.tgt, k: 0, dur: which === 'home' ? 2.5 : 4 };
      c.toast({ gap: 'Giữa các cột: P3 và P2 ở phía trước cụm sao, đỉnh P1 lùi về phía sau — điều chỉ thấy được khi đổi góc nhìn.',
        tip: 'Đỉnh cột trái: những “ngón tay” khí đặc (EGG) còn sót lại khi khí xung quanh đã bốc hơi — bên trong có thể có sao đang hình thành.',
        home: 'Góc nhìn của Hubble/JWST: dựng khối 3D khớp đúng từng điểm ảnh của ảnh thật.' }[which], 7000);
    };
    def._go = go;
    const sm = { ion: 1, dust: 1 };
    let saved = null, pending = null;
    const tmp = new THREE.Vector2();
    return {
      material, overlay, ready,
      setQuality(q) { material.defines.STEPS = stepsFor(q); material.needsUpdate = true; },
      enter(c) {
        // the volume is reconstructed from one line of sight: keep the orbit within ±35° of it
        const k = c.controls;
        saved = [k.minAzimuthAngle, k.maxAzimuthAngle, k.minPolarAngle, k.maxPolarAngle];
        k.minAzimuthAngle = -0.61; k.maxAzimuthAngle = 0.61; k.minPolarAngle = 0.96; k.maxPolarAngle = 2.18;
        pending = new URLSearchParams(location.search).get('pilfly');   // checker hook: start at a named view
      },
      exit(c) {
        if (saved) [c.controls.minAzimuthAngle, c.controls.maxAzimuthAngle, c.controls.minPolarAngle, c.controls.maxPolarAngle] = saved;
        fly = null;
      },
      update(dt, c) {
        const p = c.params, k = 1 - Math.exp(-dt * 4);
        sm.ion += (p.ion - sm.ion) * k; sm.dust += (p.dust - sm.dust) * k;
        const pal = { ...(PAL[p.palette] || PAL[0]), ...(PAL_SURF[p.palette] || PAL_SURF[0]) };
        if (tex.h) {
          U.tP.value = p.palette === 2 ? tex.j : tex.h;
          U.tB.value = p.palette === 2 ? tex.bj : tex.bh;
          U.tD.value = p.palette === 2 ? tex.dj : tex.dh;
          U.tPd.value = p.palette === 2 ? tex.jh : tex.hh;
        }
        U.uPal.value = p.palette; U.uKap.value = pal.kap; U.uDust.value = sm.dust; U.uIon.value = sm.ion;
        U.uDustTint.value.fromArray(pal.dust); U.uFringe.value.fromArray(pal.fringe); U.uSideCap.value = pal.cap;
        SU.uPal.value = p.palette; SU.uKap.value = pal.kap; SU.uDust.value = sm.dust; SU.uSpk.value = pal.spk;
        c.renderer.getDrawingBufferSize(tmp);
        SU.uScreen.value.copy(tmp);
        // working (photo) px per screen px: photo texel subtends LYPX/D0 rad
        const tanH = Math.tan(THREE.MathUtils.degToRad(c.camera.fov) / 2);
        SU.uPxW.value = (2 * tanH / tmp.y) / (LYPX / D0);
        if (pending && VIEWS[pending]) { go(c, pending, true); pending = null; }
        if (fly) {
          fly.k = Math.min(1, fly.k + dt / fly.dur);
          const e = fly.k * fly.k * (3 - 2 * fly.k);
          c.camera.position.lerpVectors(fly.p0, fly.p1, e);
          c.controls.target.lerpVectors(fly.t0, fly.t1, e);
          c.camera.lookAt(c.controls.target);
          if (fly.k >= 1) fly = null;
        }
      },
      hud(c) {
        const p = c.params;
        const dTip = c.camera.position.distanceTo(P1_TIP);
        const ion = Math.max(p.ion, 0);
        const mdot = 70 * Math.sqrt(ion);
        return [
          ['Bảng màu', (PAL[p.palette] || PAL[0]).short],
          ['Camera cách đỉnh cột trái', vi(dTip, dTip < 2 ? 2 : 1) + ' năm ánh sáng'],
          ['A<sub>V</sub> qua lõi cột', '≈ ' + vi(5.4 * p.dust) + ' mag' + (p.palette === 2 ? ' (ở 2 µm chỉ ≈ ' + vi(0.55 * p.dust) + ')' : '')],
          ['Dòng bốc hơi', ion > 0.02 ? '≈ 8 km/s · ' + Math.round(mdot) + ' M☉/triệu năm' : 'ngừng (không còn UV)'],
          ['Tuổi thọ còn lại', ion > 0.02 ? '≈ ' + vi(200 / mdot, 1) + ' triệu năm' : 'không bị bào mòn'],
        ];
      },
    };
  },
};

export default [def];
