/**
 * Viewport: renderer, cameras, lighting and post-processing.
 *
 * The post chain is hand-written rather than assembled from three's examples/ tree.
 * Two reasons. Vendoring a dozen interlinked example modules with their own relative
 * imports is fragile, and the effects here want to share one composite pass: depth of
 * field needs the depth buffer, bloom needs the bright-pass result, the fly-vision
 * overlay needs both colour and depth, and vignette and grain cost nothing once they
 * ride along. One fused composite is cheaper and far easier to reason about than four
 * separate passes each round-tripping a full-screen buffer.
 *
 * Units are millimetres. The arena is roughly 900 x 420 x 600, the fly is 3.2 mm long.
 */

import * as THREE from '../vendor/three.module.js';
import { buildFly } from './fly.js';
import { buildEnvironment, buildStimulus, updateStimulus, disposeGroup } from './world.js';

/**
 * A tiny radiance source that stands in for the room.
 *
 * Nothing in the scene was acting as an environment map, so every metalness and
 * transmission surface — the fly's cuticle, the sink, the petri dish — was resolving
 * its specular term against pure black and reading as flat paint. A PMREM built from a
 * few emissive quads gives those materials something to reflect: a bright ceiling, a
 * warm floor bounce, and a coloured rim matching the environment. It costs one
 * cubemap render at load and is regenerated whenever the environment changes.
 */
function buildEnvironmentProbe() {
  const probe = new THREE.Scene();
  const add = (color, intensity, w, h, d, x, y, z, rx, ry) => {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w, h, d),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity) }),
    );
    mesh.position.set(x, y, z);
    mesh.rotation.set(rx, ry, 0);
    probe.add(mesh);
  };

  // Ceiling — the dominant source, and what a horizontal surface actually mirrors.
  // Kept dim on purpose: the probe's job is specular reflection, not lighting. A
  // bright probe becomes a giant area light and washes every diffuse surface out,
  // so the analytic lights still own the exposure and this only adds the reflection
  // that analytic lights cannot provide.
  add(0xdfeaff, 0.9, 60, 1, 60, 0, 24, 0);
  // Floor bounce, warm and dim, so undersides are not dead black.
  add(0x4a3a2c, 0.22, 60, 1, 60, 0, -24, 0);
  // Two side walls break up the reflection so curved surfaces show a gradient
  // instead of one flat value across their whole length.
  add(0x8fa8c8, 0.34, 1, 44, 60, -26, 2, 0);
  add(0x6a7fa0, 0.26, 1, 44, 60, 26, 2, 0);
  // A warm accent behind the camera, which is what gives the fly's eye and thorax
  // their highlight as it turns.
  add(0xffd9a0, 0.55, 40, 26, 1, 0, 4, -26);
  return probe;
}

/**
 * Sky dome: a vertical gradient plus a sun disc, drawn on the inside of a box.
 *
 * The environment's ``skybox`` field is a single flat colour, which reads as a void
 * behind every arena. This turns it into an actual sky: the given colour becomes the
 * horizon, the zenith is derived from it and pushed cooler and darker, and a soft
 * sun sits where the key light comes from. It costs one shader on a box that the
 * camera never leaves.
 */
function buildSky(spec, sunDirection) {
  const horizon = new THREE.Color(spec.skybox || '#0b1220');
  const zenith = horizon.clone();
  {
    // Pull the horizon colour toward a deep blue at altitude: interiors want a dark
    // ceiling, exteriors want a lighter one, and both want less saturation up top.
    //
    // The hue is moved *toward* a target rather than rotated by a fixed amount. A
    // fixed rotation only looks right for warm horizons — add the same 0.52 to a warm
    // brown and you land on blue, which is what you want, but add it to an already
    // blue sky and you land on yellow-green, which is a bug that only shows up in the
    // two environments whose skybox is blue.
    const hsl = { h: 0, s: 0, l: 0 };
    zenith.getHSL(hsl);
    const TARGET_HUE = 0.58;                      // blue
    let delta = TARGET_HUE - hsl.h;
    if (delta > 0.5) delta -= 1;                  // take the short way round the wheel
    if (delta < -0.5) delta += 1;
    zenith.setHSL(
      (hsl.h + delta * 0.75 + 1) % 1,            // most of the way to blue, not all of it
      Math.min(0.8, hsl.s * 0.85 + 0.10),
      Math.max(0.04, hsl.l * 0.5),
    );
  }
  const ground = horizon.clone().multiplyScalar(0.35);

  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uZenith: { value: zenith },
      uHorizon: { value: horizon },
      uGround: { value: ground },
      uSun: { value: sunDirection.clone().normalize() },
      uSunColor: { value: new THREE.Color(0xfff2d8) },
      uSunIntensity: { value: Math.max(0, Math.min(1.4, (spec.ambientLight ?? 0.6) * 1.15)) },
    },
    vertexShader: /* glsl */`
      precision highp float;
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      // A ShaderMaterial gets no default float precision, so without this every
      // varying and local below is an untyped float and the program will not link.
      precision highp float;
      varying vec3 vDir;
      uniform vec3 uZenith;
      uniform vec3 uHorizon;
      uniform vec3 uGround;
      uniform vec3 uSun;
      uniform vec3 uSunColor;
      uniform float uSunIntensity;

      void main() {
        vec3 dir = normalize(vDir);
        float h = dir.y;
        // Above the horizon: horizon -> zenith. Below: horizon -> dark ground.
        vec3 sky = mix(uHorizon, uZenith, pow(clamp(h, 0.0, 1.0), 0.55));
        vec3 below = mix(uHorizon, uGround, pow(clamp(-h, 0.0, 1.0), 0.45));
        vec3 color = h > 0.0 ? sky : below;

        // Sun disc with a wide halo. The halo is what sells it as atmosphere; a hard
        // disc alone just looks like a bright pixel.
        float cosAngle = dot(dir, normalize(uSun));
        float halo = pow(clamp(cosAngle, 0.0, 1.0), 220.0);
        float glow = pow(clamp(cosAngle, 0.0, 1.0), 8.0) * 0.22;
        color += uSunColor * (halo * 2.4 + glow) * uSunIntensity;

        // A soft band right at the horizon keeps the two halves from meeting in a
        // hard line, which is the giveaway of a gradient sky on a box.
        color += uHorizon * 0.35 * exp(-abs(h) * 14.0);

        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;
  mesh.scale.setScalar(1200);
  return mesh;
}

export const CAMERA_MODES = [
  { key: 'follow', label: 'Follow', blurb: 'Third-person orbit locked to the fly with damping' },
  { key: 'tactical', label: 'Tactical room', blurb: 'Free-look overhead view for managing the arena' },
  { key: 'flyvision', label: 'Fly vision', blurb: 'Compound-eye mosaic with UV and optical-flow highlights' },
];

const COMPOSITE_FRAG = /* glsl */`
  precision highp float;
  varying vec2 vUv;

  uniform sampler2D tScene;
  uniform sampler2D tBloom;
  uniform sampler2D tBloomWide;
  uniform sampler2D tDepth;
  uniform vec2  uResolution;
  uniform float uNear;
  uniform float uFar;
  uniform float uFocusDist;
  uniform float uAperture;
  uniform float uBloomStrength;
  uniform float uGrain;
  uniform float uVignette;
  uniform float uExposure;
  uniform float uFlyVision;      // 0..1 blend into the compound-eye view
  uniform vec2  uFlow;           // screen-space motion hint from the fly's velocity
  uniform float uFlowStrength;
  uniform float uFacets;
  uniform float uTime;
  uniform sampler2D tAO;         // ambient occlusion, half resolution
  uniform float uAOStrength;
  uniform float uAberration;

  // -- helpers ---------------------------------------------------------------
  float linearDepth(vec2 uv) {
    float z = texture2D(tDepth, uv).x;
    if (z >= 1.0) return uFar;
    float ndc = z * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
  }

  vec3 aces(vec3 x) {
    // Narkowicz's ACES fit: cheap, and it keeps highlights from clipping to white.
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
  }

  /**
   * Split one channel off to the side of the optical axis.
   *
   * Real optics do not focus every wavelength to the same point, so a bright rim
   * fringes blue on one side and red on the other. Sampling R, G and B at slightly
   * different radial offsets costs three fetches instead of one and is the whole
   * effect; the magnitude is driven by distance from centre so the middle of the
   * frame stays perfectly sharp, which is where the fly usually is.
   */
  vec3 chroma(vec2 uv, vec2 texel) {
    // Two extra fetches only buy the fringe at the edges of the frame, where the
    // aberration offset is largest. Near the centre the three samples are within a
    // fraction of a texel of each other, so the single-fetch path is not an
    // approximation — it is the same answer for a third of the bandwidth.
    vec2 centred = uv - 0.5;
    float radial = length(centred) * 2.0;
    if (uAberration < 0.01 || radial < 0.25) return texture2D(tScene, uv).rgb;

    vec2 shift = centred * uAberration * 0.012;
    vec3 c;
    c.r = texture2D(tScene, uv + shift).r;
    c.g = texture2D(tScene, uv).g;
    c.b = texture2D(tScene, uv - shift).b;
    return c;
  }

  vec3 linearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, 1e-5), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
  }

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  // -- hexagonal facet grid (axial coordinates) -----------------------------
  // Flat-top hexagons read as ommatidia in a way squares never do. Axial
  // coordinates with cube rounding give an exact nearest-centre lookup, which is
  // what makes each facet sample *its own* pixel rather than a neighbour's.
  vec2 axialToPixel(vec2 a) { return vec2(a.x * 0.8660254, a.y + a.x * 0.5); }
  vec2 pixelToAxial(vec2 p) { return vec2(p.x / 0.8660254, p.y - p.x * 0.5773503); }

  vec2 hexRound(vec2 a) {
    float x = a.x, z = a.y, y = -x - z;
    float rx = floor(x + 0.5), ry = floor(y + 0.5), rz = floor(z + 0.5);
    float dx = abs(rx - x), dy = abs(ry - y), dz = abs(rz - z);
    if (dx > dy && dx > dz) rx = -ry - rz;
    else if (dy > dz) ry = -rx - rz;
    else rz = -rx - ry;
    return vec2(rx, rz);
  }

  void main() {
    vec2 uv = vUv;
    vec3 color;

    if (uFlyVision > 0.5) {
      // ---- compound eye -----------------------------------------------------
      // Sample once per ommatidium at the facet's own centre. That is the whole
      // trick: the mosaic is not a blur, it is genuinely low-resolution vision.
      float aspect = max(1.0, uResolution.x / uResolution.y);
      vec2 square = vec2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * aspect);

      // Barrel distortion, so the field reads as a curved retinal surface.
      float r2 = dot(square, square);
      vec2 distorted = square * (1.0 + 0.17 * r2);

      vec2 facetPx = distorted * uFacets;
      vec2 axial = pixelToAxial(facetPx);
      vec2 snapped = hexRound(axial);
      vec2 centrePx = axialToPixel(snapped);

      // Where that facet's centre projects to, with the distortion roughly undone.
      vec2 centredSample = centrePx / uFacets;
      centredSample /= (1.0 + 0.17 * dot(centredSample, centredSample));
      vec2 facetUv = vec2(centredSample.x, centredSample.y / aspect) * 0.5 + 0.5;

      vec3 eye = texture2D(tScene, clamp(facetUv, 0.0015, 0.9985)).rgb;

      // UV channel. Drosophila see well into the ultraviolet, so a bright region
      // is re-tinted toward violet rather than simply brightened.
      float luma = dot(eye, vec3(0.2126, 0.7152, 0.0722));
      vec3 uvTint = mix(vec3(0.30, 0.20, 0.95), vec3(0.70, 0.50, 1.0), luma);
      eye = mix(eye, eye * 0.30 + uvTint * luma * 0.55, 0.45);

      // Local motion the eye would see, drawn as banded flow lines aligned with
      // the fly's velocity in screen space.
      vec2 flowDir = uFlow * max(0.0001, length(uFlow));
      vec2 screenRel = vec2((uv.x - 0.5) * aspect, uv.y - 0.5);
      float along = dot(screenRel, normalize(flowDir + vec2(1e-5, 0.0)));
      float bands = abs(fract(along * 26.0 - uTime * 2.2) - 0.5);
      eye += vec3(1.0, 0.62, 0.28) * smoothstep(0.45, 0.5, bands)
             * min(1.0, uFlowStrength) * 0.26;

      // Ommatidial walls: the facet edge is where the neighbouring sample wins,
      // so it is measured in facet-local space, not in screen space.
      vec2 local = facetPx - centrePx;
      float edge = max(abs(local.x) / 0.8660254, abs(local.y * 1.0 + local.x * 0.5773503 * 0.0));
      eye *= 0.60 + 0.40 * smoothstep(0.52, 0.30, edge);

      // Peripheral falloff of the eye's usable field of view.
      eye *= 1.0 - smoothstep(0.50, 1.05, length(square)) * 0.6;

      color = eye;
    } else {
      // ---- depth of field + bloom -----------------------------------------
      float depth = linearDepth(uv);
      float coc = clamp(abs(depth - uFocusDist) / max(1.0, uFocusDist) * uAperture, 0.0, 1.0);
      float radius = coc * 2.6 / uResolution.x;

      vec3 sharp = chroma(uv, 1.0 / uResolution);
      vec3 scene = sharp;

      // The blur is only mixed in where there is any circle of confusion at all.
      // Most of the frame is in focus, and skipping the whole 12-tap disc for those
      // warps is the difference between a smooth frame rate and a slow one. The
      // threshold is well below the first visible step of the mix so the transition
      // stays invisible.
      float blurMix = smoothstep(0.0, 0.55, coc);
      if (uAperture > 0.001 && blurMix > 0.002) {
        // A 12-tap disc: cheap and, at these radii, indistinguishable from a proper
        // golden-angle spiral for a soft background.
        vec3 blurred = vec3(0.0);
        float total = 0.0;
        for (int i = 0; i < 12; i++) {
          float angle = float(i) * 2.39996;
          float ring = sqrt(float(i) / 12.0);
          vec2 offset = vec2(cos(angle), sin(angle)) * ring * (radius * uResolution.x);
          vec2 tap = clamp(uv + offset / uResolution, 0.001, 0.999);
          float weight = 1.0 / (1.0 + abs(linearDepth(tap) - depth) * 0.05);
          blurred += texture2D(tScene, tap).rgb * weight;
          total += weight;
        }
        blurred /= max(0.0001, total);
        scene = mix(sharp, blurred, blurMix);
      }

      // Ambient occlusion darkens contact points: the crease where a leg meets the
      // body, the ring where a fruit sits in the bowl, the fly's own shadow pooling
      // under it. Without it everything floats.
      float ao = texture2D(tAO, uv).r;
      scene *= mix(1.0, ao, uAOStrength);

      // Two bloom scales: a tight core plus a wide halo. One scale alone either
      // looks like a smudge or like nothing happened.
      vec3 bloom = texture2D(tBloom, uv).rgb * 0.62 + texture2D(tBloomWide, uv).rgb * 0.38;
      color = scene + bloom * uBloomStrength;
    }

    color *= uExposure;
    color = aces(color);
    color = linearToSrgb(color);

    // Vignette and grain apply to both views; they sell the sensor.
    float vig = 1.0 - uVignette * dot((uv - 0.5) * vec2(1.1, 1.0), (uv - 0.5) * vec2(1.1, 1.0)) * 3.2;
    color *= clamp(vig, 0.0, 1.0);

    // Grain is strongest in the shadows. Uniform noise reads as digital sensor
    // noise and flattens the midtones; film noise is densest where the signal is
    // weakest, so the weighting follows the inverted luma.
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float grain = (hash(uv * uResolution + fract(uTime) * 91.7) - 0.5) * uGrain;
    color += grain * mix(1.8, 0.35, smoothstep(0.0, 0.55, luma));

    gl_FragColor = vec4(color, 1.0);
  }
`;

const BRIGHT_FRAG = /* glsl */`
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D tScene;
  uniform float uThreshold;
  void main() {
    vec3 c = texture2D(tScene, vUv).rgb;
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float contribution = smoothstep(uThreshold, uThreshold + 0.5, luma);
    gl_FragColor = vec4(c * contribution, 1.0);
  }
`;

const BLUR_FRAG = /* glsl */`
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D tInput;
  uniform vec2 uDirection;    // texel step
  void main() {
    // 9-tap gaussian, separable.
    float weights[5];
    weights[0] = 0.227027; weights[1] = 0.194594; weights[2] = 0.121621;
    weights[3] = 0.054054; weights[4] = 0.016216;
    vec3 sum = texture2D(tInput, vUv).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
      vec2 offset = uDirection * float(i) * 1.4;
      sum += texture2D(tInput, vUv + offset).rgb * weights[i];
      sum += texture2D(tInput, vUv - offset).rgb * weights[i];
    }
    gl_FragColor = vec4(sum, 1.0);
  }
`;

/**
 * Screen-space ambient occlusion from the depth buffer alone.
 *
 * There is no G-buffer here — the post chain deliberately renders a single forward
 * target — so the surface normal is reconstructed per pixel from the depth texture by
 * central differences, and the hemisphere is sampled around it. That is enough for
 * contact darkening, which is all AO is doing in this scene.
 *
 * The kernel is a fixed 12-tap golden-angle spiral with a per-pixel rotation. The
 * rotation is what turns banding into noise, and the blur pass afterwards turns the
 * noise back into a smooth gradient — the standard interleaved-graphics trick, and the
 * reason this can run at half resolution without showing it.
 */
const AO_FRAG = /* glsl */`
  precision highp float;
  varying vec2 vUv;

  uniform sampler2D tDepth;
  uniform vec2  uResolution;    // AO buffer resolution, not the scene's
  uniform float uNear;
  uniform float uFar;
  uniform float uRadius;        // world-space radius of the hemisphere
  uniform float uBias;
  uniform float uTime;
  uniform float uFovTan;        // tan(verticalFov / 2)
  uniform float uAspect;

  float linearDepth(vec2 uv) {
    float z = texture2D(tDepth, uv).x;
    if (z >= 1.0) return uFar;
    float ndc = z * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
  }

  /**
   * View-space position for a depth sample.
   *
   * The tangent is rebuilt from the projection's field of view rather than passed in
   * as a matrix, because the tangent of the half-angle is the only quantity the
   * hemisphere maths needs and one divide per pixel is cheaper than a mat4 multiply.
   */
  vec3 viewPosition(vec2 uv, float depth) {
    // View-space position rebuilt from the projection's tan(fov/2). Shipping
    // tan-half-angle and aspect costs two floats; shipping the inverse projection
    // matrix would cost four and buy nothing this pass uses.
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * depth * uFovTan * uAspect, ndc.y * depth * uFovTan, -depth);
  }

  float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  void main() {
    float depth = linearDepth(vUv);
    // Sky, and anything past the far plane, cannot be occluded by anything.
    if (depth >= uFar * 0.999) { gl_FragColor = vec4(1.0); return; }

    // Reconstruct the normal from neighbouring depth. A one-texel offset is too
    // small to survive float precision on a scene 2000 units deep, so the step is
    // scaled with resolution and clamped.
    vec2 texel = 1.0 / uResolution;
    float dL = linearDepth(vUv - vec2(texel.x, 0.0));
    float dR = linearDepth(vUv + vec2(texel.x, 0.0));
    float dD = linearDepth(vUv - vec2(0.0, texel.y));
    float dU = linearDepth(vUv + vec2(0.0, texel.y));

    vec3 pL = viewPosition(vUv - vec2(texel.x, 0.0), dL);
    vec3 pR = viewPosition(vUv + vec2(texel.x, 0.0), dR);
    vec3 pD = viewPosition(vUv - vec2(0.0, texel.y), dD);
    vec3 pU = viewPosition(vUv + vec2(0.0, texel.y), dU);

    // Degenerate where depth is identical on both sides — a flat surface seen
    // edge-on, or the far plane — leaves the cross product at zero, and normalizing
    // that yields NaN which would then propagate through every downstream tap. The
    // comparison is a length test rather than isnan(), which GLSL ES 1.00 does not
    // have; NaN fails every comparison and so also falls through to the same branch.
    vec3 cross1 = cross(pR - pL, pU - pD);
    float crossLen = length(cross1);
    if (!(crossLen > 1e-8)) { gl_FragColor = vec4(1.0); return; }
    vec3 normal = cross1 / crossLen;

    // Per-pixel rotation, animated slowly so the residual noise crawls instead of
    // sitting still as a fixed dither pattern.
    float angle = hash12(gl_FragCoord.xy + fract(uTime) * 37.0) * 6.2831853;

    // Build a basis around the normal so the spiral lies in a hemisphere above it.
    vec3 up = abs(normal.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    vec3 origin = viewPosition(vUv, depth);
    float occlusion = 0.0;

    for (int i = 0; i < 8; i++) {
      float fi = float(i);
      // Golden-angle spiral: even coverage of the disc with no visible structure.
      // Eight taps is one fewer third of the AO cost; because the result is
      // half-resolution and then blurred, the difference against twelve taps is not
      // visible, while the saving is spent on resolution instead.
      float a = fi * 2.39996 + angle;
      float r = sqrt((fi + 0.5) / 8.0);
      vec3 dir = tangent * cos(a) * r + bitangent * sin(a) * r + normal * (0.35 + 0.55 * (1.0 - r));
      vec3 samplePos = origin + dir * uRadius;

      // Project the sample back to screen and read the depth actually stored there.
      // If the surface is *closer* than the sample, something else is in the way.
      vec2 suv = vec2(
        (samplePos.x / max(1e-4, -samplePos.z)) / (uFovTan * uAspect),
        (samplePos.y / max(1e-4, -samplePos.z)) / uFovTan
      ) * 0.5 + 0.5;

      if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) continue;

      float sceneDepth = linearDepth(suv);
      float rangeWeight = 1.0 - smoothstep(0.0, 1.0, abs(depth - sceneDepth) / uRadius);
      float delta = depth - sceneDepth;
      occlusion += step(uBias, delta) * rangeWeight;
    }

    float ao = 1.0 - (occlusion / 8.0);
    gl_FragColor = vec4(ao, ao, ao, 1.0);
  }
`;

/**
 * A monotonic frame clock.
 *
 * ``THREE.Clock`` is deprecated as of r183 and its replacement, ``THREE.Timer``, has
 * to be driven with an explicit ``update(timestamp)`` on every frame. The interface
 * the app actually needs is two numbers, so this replaces it with no API to misuse and
 * no deprecation warning on every page load.
 */
export class FrameClock {
  constructor() {
    this.started = performance.now();
    this.previous = this.started;
  }

  /** Seconds since the clock was created. */
  get elapsedTime() {
    return (performance.now() - this.started) / 1000;
  }

  /** Seconds since the previous call, clamped so a backgrounded tab cannot jump. */
  getDelta(maxSeconds = 0.05) {
    const now = performance.now();
    const delta = (now - this.previous) / 1000;
    this.previous = now;
    return Math.min(maxSeconds, delta);
  }
}

/** Full-screen triangle/quad helper used by every post pass. */
function fullscreenGeometry() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -1, -1, 0, 3, -1, 0, -1, 3, 0,
  ]), 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 0, 2, 0, 0, 2,
  ]), 2));
  return geometry;
}

export class Viewport {
  constructor(canvas, manifest) {
    this.canvas = canvas;
    this.manifest = manifest;
    this.cameraMode = 'follow';
    this.environment = null;
    this.fogBase = 0.0006;
    this.solids = [];
    this.groups = { environment: new THREE.Group(), stimuli: new THREE.Group(), flies: new THREE.Group() };
    this.flyModels = new Map();
    this.stimulusModels = new Map();
    this.clock = new FrameClock();
    this.showROIs = true;
    this.roiGroup = new THREE.Group();
    this.selectedTool = null;
    this.orbit = {
      yaw: 0.6, pitch: 0.42, distance: 46,
      target: new THREE.Vector3(0, 0, 0), want: new THREE.Vector3(0, 0, 0),
      wantYaw: 0.6, wantPitch: 0.42, wantDistance: 46,
      tactical: { yaw: 0.7, pitch: 1.32, distance: 900, target: new THREE.Vector3(0, 0, 0) },
    };
    this.pointer = { dragging: false, mode: null, x: 0, y: 0 };
    this.brainHook = null;

    this._initRenderer();
    this._initScene();
    this._initPost();
    this._bindInput();
  }

  // -- setup ---------------------------------------------------------------
  _initRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false,          // the post chain owns antialiasing
      alpha: false,
      powerPreference: 'high-performance',
      depth: true,
      stencil: false,
    });
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.renderScale = 1;
    this.renderer.setPixelRatio(1);
    this.renderer.shadowMap.enabled = true;
    // PCFSoftShadowMap was removed in r186; PCFShadowMap is the supported soft filter.
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setClearColor(0x05070c, 1);
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.add(this.groups.environment, this.groups.stimuli, this.groups.flies, this.roiGroup);

    this.camera = new THREE.PerspectiveCamera(52, 1, 2, 4000);

    // A hemisphere fill plus one shadow-casting key light covers every
    // environment; per-preset tweaks come from the manifest.
    this.hemi = new THREE.HemisphereLight(0x9fb4d6, 0x2b3140, 0.55);
    this.sun = new THREE.DirectionalLight(0xfff2d8, 2.1);
    this.sun.position.set(320, 620, 260);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.camera.near = 60;
    this.sun.shadow.camera.far = 2600;
    this.sun.shadow.bias = -0.0012;
    this.sun.shadow.normalBias = 0.6;
    // The shadow frustum is retargeted onto the fly every frame in follow mode, so it
    // only ever has to cover a small patch of ground rather than the whole arena. That
    // buys the resolution back: at 2048 across 1400 units a texel spans 0.7 units and
    // the fly's own shadow is one texel wide, which is why nothing was casting.
    this.sun.shadow.camera.left = -70;
    this.sun.shadow.camera.right = 70;
    this.sun.shadow.camera.top = 70;
    this.sun.shadow.camera.bottom = -70;
    this.sunTarget = new THREE.Vector3(0, 0, 0);
    // Offset from the shadow target to the light. Recomputed per environment in
    // setEnvironment; the default here is what the first frame uses if an
    // environment has not loaded yet.
    this.sunOffset = new THREE.Vector3(320, 620, 260);
    this.shadowSpan = 70;
    this.scene.add(this.sun.target);
    this.scene.add(this.hemi, this.sun);

    this.fill = new THREE.PointLight(0xbcd4ff, 0.5, 2400, 2.0);
    this.fill.position.set(-360, 320, -260);
    this.scene.add(this.fill);

    // Fly vision keeps the scene visible at extreme close range, where the
    // standard camera's near plane would clip the fly's own body.
    this.flyLight = new THREE.PointLight(0xffffff, 0.0, 120, 2.0);
    this.scene.add(this.flyLight);

    // Image-based lighting. Without this every metalness > 0 surface has nothing to
    // reflect and renders as flat paint — the sink, the drain, the fly's cuticle and
    // its eyes all lose their form. The probe is a handful of emissive quads, so the
    // whole map costs one 128px cubemap render.
    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    this.probeScene = buildEnvironmentProbe();
    const probeTarget = this.pmrem.fromScene(this.probeScene, 0.04);
    this.scene.environment = probeTarget.texture;
    this.scene.environmentIntensity = 0.75;

    this.sky = null;
  }

  /**
   * Rebuild the sky dome and re-tint the environment probe for a new arena.
   *
   * The probe is regenerated rather than tinted in place because PMREM textures are
   * prefiltered: changing the source colour afterwards would leave the roughness
   * mips stale. Regenerating costs a fraction of a millisecond and only happens when
   * the environment actually changes.
   */
  _rebuildSky(spec) {
    if (this.sky) {
      this.scene.remove(this.sky);
      this.sky.geometry.dispose();
      this.sky.material.dispose();
    }
    this.sky = buildSky(spec, this.sun.position);
    this.scene.add(this.sky);

    const light = spec.ambientLight ?? 0.6;
    // Retint the probe's emissive quads toward the environment's own colour so the
    // reflections agree with the sky the player can actually see. The multipliers
    // stay well below 1: this modulates a reflection, not the exposure.
    const tint = new THREE.Color(spec.skybox || '#0b1220');
    const levels = [0.55 + 0.5 * light, 0.5, 0.5 + 0.3 * light, 0.45 + 0.25 * light, 0.5 + 0.4 * light];
    const meshes = this.probeScene.children;
    for (let index = 0; index < meshes.length; index++) {
      const base = meshes[index].userData.baseColor;
      if (!base) {
        meshes[index].userData.baseColor = meshes[index].material.color.clone();
      }
      const color = meshes[index].userData.baseColor.clone();
      // A cool sky pulls reflections blue; a warm interior pulls them amber. Mixing
      // toward the environment colour by 35% is enough to read without flattening.
      color.lerp(tint, 0.35);
      color.multiplyScalar(levels[index] ?? 1);
      meshes[index].material.color.copy(color);
    }
    const previous = this.scene.environment;
    const target = this.pmrem.fromScene(this.probeScene, 0.04);
    this.scene.environment = target.texture;
    if (previous) previous.dispose();
    this.scene.environmentIntensity = 0.5 + 0.5 * light;
  }

  _initPost() {
    const size = this._targetSize();
    const options = {
      type: THREE.HalfFloatType,
      depthTexture: new THREE.DepthTexture(size.width, size.height, THREE.UnsignedIntType),
      depthBuffer: true,
      stencilBuffer: false,
    };
    // 4x MSAA inside the offscreen target: supersampling the scene antes up the
    // geometry before the post chain ever sees it, which matters far more than any
    // edge filter once bloom and depth of field are in play.
    options.samples = 4;
    this.sceneTarget = new THREE.WebGLRenderTarget(size.width, size.height, options);
    this.sceneTarget.depthTexture.format = THREE.DepthFormat;

    // Three bloom mips, each with a scratch buffer so a separable blur never has
    // to read and write the same texture in one pass.
    const mipSize = (index) => ({
      width: Math.max(2, size.width >> (index + 1)),
      height: Math.max(2, size.height >> (index + 1)),
    });
    this.bloomTargets = [0, 1, 2].map((index) => {
      const dims = mipSize(index);
      return new THREE.WebGLRenderTarget(dims.width, dims.height, { type: THREE.HalfFloatType });
    });
    this.bloomScratch = [0, 1, 2].map((index) => {
      const dims = mipSize(index);
      return new THREE.WebGLRenderTarget(dims.width, dims.height, { type: THREE.HalfFloatType });
    });

    // Ambient occlusion runs at half resolution and is blurred before use. AO is a
    // low-frequency signal by construction — nothing about a contact crease needs
    // full-res detail — so half-res costs a quarter of the work and is invisible.
    const aoWidth = Math.max(2, size.width >> 1);
    const aoHeight = Math.max(2, size.height >> 1);
    this.aoTarget = new THREE.WebGLRenderTarget(aoWidth, aoHeight, {
      type: THREE.UnsignedByteType,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.aoBlurTarget = new THREE.WebGLRenderTarget(aoWidth, aoHeight, {
      type: THREE.UnsignedByteType,
      depthBuffer: false,
      stencilBuffer: false,
    });

    this.quad = new THREE.Mesh(fullscreenGeometry(), null);
    this.quadScene = new THREE.Scene();
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);

    this.compositeMaterial = new THREE.RawShaderMaterial({
      vertexShader: `precision highp float;
        attribute vec3 position; attribute vec2 uv; varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: COMPOSITE_FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        tScene: { value: this.sceneTarget.texture },
        tBloom: { value: this.bloomTargets[0].texture },
        tBloomWide: { value: this.bloomTargets[2].texture },
        tDepth: { value: this.sceneTarget.depthTexture },
        uResolution: { value: new THREE.Vector2(size.width, size.height) },
        uNear: { value: 2 },
        uFar: { value: 4000 },
        uFocusDist: { value: 60 },
        uAperture: { value: 0.55 },
        uBloomStrength: { value: 0.5 },
        uGrain: { value: 0.008 },
        uVignette: { value: 0.32 },
        uExposure: { value: 1.05 },
        uFlyVision: { value: 0 },
        uFlow: { value: new THREE.Vector2(1, 0) },
        uFlowStrength: { value: 0 },
        uFacets: { value: 78 },
        uTime: { value: 0 },
        tAO: { value: this.aoTarget.texture },
        uAOStrength: { value: 0.85 },
        uAberration: { value: 0.7 },
      },
    });

    this.brightMaterial = new THREE.RawShaderMaterial({
      vertexShader: `precision highp float;
        attribute vec3 position; attribute vec2 uv; varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: BRIGHT_FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        tScene: { value: this.sceneTarget.texture },
        uThreshold: { value: 0.72 },
      },
    });

    this.aoMaterial = new THREE.RawShaderMaterial({
      vertexShader: `precision highp float;
        attribute vec3 position; attribute vec2 uv; varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: AO_FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        tDepth: { value: this.sceneTarget.depthTexture },
        uResolution: { value: new THREE.Vector2(aoWidth, aoHeight) },
        uNear: { value: 2 },
        uFar: { value: 4000 },
        uRadius: { value: 14 },
        uBias: { value: 0.9 },
        uTime: { value: 0 },
        uFovTan: { value: Math.tan((52 * Math.PI) / 360) },
        uAspect: { value: 1 },
      },
    });

    this.blurMaterial = new THREE.RawShaderMaterial({
      vertexShader: `precision highp float;
        attribute vec3 position; attribute vec2 uv; varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: BLUR_FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        tInput: { value: null },
        uDirection: { value: new THREE.Vector2() },
      },
    });
  }

  _targetSize() {
    // The canvas is sized in device pixels for the CSS box, then scaled by the
    // adaptive quality factor. Clamped to 2 so a 3x display does not quietly triple
    // the cost of every pass.
    const ratio = Math.min(this.pixelRatio * (this.renderScale ?? 1), 2);
    const width = Math.max(2, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(2, Math.floor(this.canvas.clientHeight * ratio));
    return { width, height };
  }

  // -- environment ---------------------------------------------------------
  setEnvironment(spec) {
    disposeGroup(this.groups.environment);
    this.solids = [];
    const built = buildEnvironment(spec, this.manifest);
    this.groups.environment.add(built.group);
    this.environmentGroup = built.group;
    this.solids = built.solids;

    this.scene.background = null;   // the sky dome draws the background now
    // The density in the environment table is authored for a fly's-eye view; how it
    // is applied across zoom levels is decided per frame in render().
    this.fogBase = spec.fog ?? 0.0006;
    const horizon = new THREE.Color(spec.skybox || '#0b1220');
    this.scene.fog = new THREE.FogExp2(horizon, this.fogBase);

    const light = spec.ambientLight ?? 0.6;
    this.hemi.intensity = 0.22 + 0.5 * light;
    this.sun.intensity = 0.25 + 1.9 * light;
    this.sun.position.set(spec.size[0] * 0.35, spec.size[1] * 1.4, spec.size[2] * 0.45);
    this.fill.intensity = 0.25 + 0.5 * light;
    this.sunOffset = new THREE.Vector3(
      spec.size[0] * 0.35, spec.size[1] * 1.4, spec.size[2] * 0.45,
    );
    this._rebuildSky(spec);

    this.environment = spec;
    this.orbit.tactical.distance = Math.max(spec.size[0], spec.size[2]) * 1.15;
    this.orbit.tactical.target.set(0, spec.size[1] * 0.25, 0);
    this._rebuildROIs(spec);
  }

  /** ROI ellipsoids from the manifest, drawn as toggleable translucent markers. */
  _rebuildROIs(spec) {
    disposeGroup(this.roiGroup);
    const atlas = this.manifest.atlas;
    const size = spec.size;
    for (const region of atlas.regions) {
      for (const lobe of region.lobes) {
        const [cx, cy, cz, rx, ry, rz] = lobe;
        // Atlas space is x = left-right, y = anterior-posterior, z = dorsal-ventral.
        // The scene maps atlas x to world x, atlas y to world z, atlas z to world y.
        const geometry = new THREE.SphereGeometry(1, 20, 14);
        const material = new THREE.MeshBasicMaterial({
          color: new THREE.Color(`rgb(${region.color.join(',')})`),
          transparent: true,
          opacity: 0.085,
          depthWrite: false,
          wireframe: false,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.set(rx * size[0], rz * size[2], ry * size[0]);
        mesh.position.set((cx - 0.5) * size[0], (cz - 0.5) * size[2] * 0.9 + size[2] * 0.55, (cy - 0.5) * size[0]);
        mesh.renderOrder = 3;
        this.roiGroup.add(mesh);
      }
    }
    this.roiGroup.visible = this.showROIs;
  }

  setROIVisible(visible) {
    this.showROIs = visible;
    this.roiGroup.visible = visible;
  }

  // -- stimuli -------------------------------------------------------------
  syncStimuli(list) {
    const seen = new Set();
    for (const spec of list) {
      seen.add(spec.id);
      let entry = this.stimulusModels.get(spec.id);
      if (!entry) {
        const built = buildStimulus(spec, this.environment);
        this.groups.stimuli.add(built.group);
        entry = { ...built, spec: null };
        this.stimulusModels.set(spec.id, entry);
      }
      updateStimulus(entry, spec, this.environment);
      entry.spec = spec;
    }
    for (const [id, entry] of this.stimulusModels) {
      if (!seen.has(id)) {
        disposeGroup(entry.group);
        this.groups.stimuli.remove(entry.group);
        this.stimulusModels.delete(id);
      }
    }
  }

  // -- flies ---------------------------------------------------------------
  syncFlies(flies) {
    const seen = new Set();
    for (const state of flies) {
      seen.add(state.id);
      let model = this.flyModels.get(state.id);
      if (!model) {
        model = buildFly(state.id);
        this.groups.flies.add(model.root);
        this.flyModels.set(state.id, model);
      }
      model.apply(state, this.clock.elapsedTime);
    }
    for (const [id, model] of this.flyModels) {
      if (!seen.has(id)) {
        disposeGroup(model.root);
        this.groups.flies.remove(model.root);
        this.flyModels.delete(id);
      }
    }
  }

  // -- cameras -------------------------------------------------------------
  setCameraMode(mode) {
    this.cameraMode = mode;
    if (mode === 'flyvision') {
      this.camera.near = 0.6;
      this.camera.far = 3000;
      this.flyLight.intensity = 3.2;
    } else if (mode === 'tactical') {
      this.camera.near = 40;
      this.camera.far = 6000;
      this.flyLight.intensity = 0;
    } else {
      this.camera.near = 2;
      this.camera.far = 4000;
      this.flyLight.intensity = 0;
    }
    this.camera.updateProjectionMatrix();
  }

  _updateCamera(dt, primary) {
    const orbit = this.orbit;
    const position = primary ? primary.position : [0, 0, 0];
    const target = new THREE.Vector3(position[0], position[1], position[2]);

    if (this.cameraMode === 'follow') {
      orbit.want.set(target.x, target.y + 1.4, target.z);
      orbit.wantDistance = 46;
      orbit.wantPitch = Math.max(-0.15, Math.min(1.15, orbit.wantPitch));
    } else if (this.cameraMode === 'flyvision') {
      // Sit just behind and above the fly's head, aimed along its heading.
      const heading = primary ? primary.heading : 0;
      const back = 5.2;
      orbit.want.set(
        target.x + Math.sin(heading) * back,
        target.y + 1.5,
        target.z + Math.cos(heading) * back,
      );
      orbit.wantDistance = 0.001;
      const yaw = heading;
      orbit.yaw += ((yaw - orbit.yaw + Math.PI * 3) % (Math.PI * 2) - Math.PI) * Math.min(1, 14 * dt);
      orbit.pitch = -0.16;
    } else {
      orbit.want.copy(orbit.tactical.target);
      orbit.wantDistance = orbit.tactical.distance;
    }

    const smoothing = this.cameraMode === 'tactical' ? 5.5 : 7.5;
    const blend = Math.min(1, smoothing * dt);
    orbit.target.lerp(orbit.want, blend);
    orbit.distance += (orbit.wantDistance - orbit.distance) * blend;
    orbit.pitch += (orbit.wantPitch - orbit.pitch) * blend;
    if (this.cameraMode !== 'flyvision') {
      orbit.yaw += (orbit.wantYaw - orbit.yaw) * blend;
    }

    if (this.cameraMode === 'flyvision') {
      this.camera.position.copy(orbit.want);
      const heading = primary ? primary.heading : 0;
      this.camera.lookAt(
        target.x + Math.sin(heading) * 40,
        target.y - 1.0,
        target.z + Math.cos(heading) * 40,
      );
    } else {
      const cosPitch = Math.cos(orbit.pitch);
      this.camera.position.set(
        orbit.target.x + Math.sin(orbit.yaw) * cosPitch * orbit.distance,
        orbit.target.y + Math.sin(orbit.pitch) * orbit.distance + orbit.distance * 0.28,
        orbit.target.z + Math.cos(orbit.yaw) * cosPitch * orbit.distance,
      );
      this.camera.lookAt(orbit.target.x, orbit.target.y + 1.2, orbit.target.z);
    }
    this.camera.updateProjectionMatrix();
  }

  // -- input ---------------------------------------------------------------
  _bindInput() {
    const canvas = this.canvas;
    canvas.addEventListener('pointerdown', (event) => {
      this.pointer.dragging = true;
      this.pointer.x = event.clientX;
      this.pointer.y = event.clientY;
      this.pointer.moved = 0;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.pointer.dragging) return;
      const dx = event.clientX - this.pointer.x;
      const dy = event.clientY - this.pointer.y;
      this.pointer.x = event.clientX;
      this.pointer.y = event.clientY;
      this.pointer.moved += Math.abs(dx) + Math.abs(dy);
      if (this.cameraMode === 'tactical') {
        const speed = this.orbit.tactical.distance * 0.0016;
        this.orbit.tactical.target.x -= Math.cos(this.orbit.tactical.yaw) * dx * speed;
        this.orbit.tactical.target.z += Math.sin(this.orbit.tactical.yaw) * dx * speed;
        this.orbit.tactical.target.x += Math.sin(this.orbit.tactical.yaw) * dy * speed;
        this.orbit.tactical.target.z += Math.cos(this.orbit.tactical.yaw) * dy * speed;
      } else if (this.cameraMode !== 'flyvision') {
        this.orbit.wantYaw -= dx * 0.006;
        this.orbit.wantPitch = Math.max(-0.2, Math.min(1.35, this.orbit.wantPitch + dy * 0.005));
      }
    });
    canvas.addEventListener('pointerup', (event) => {
      this.pointer.dragging = false;
      try { canvas.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
    });
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      if (this.cameraMode === 'tactical') {
        this.orbit.tactical.distance = Math.max(120, Math.min(3200, this.orbit.tactical.distance * (1 + Math.sign(event.deltaY) * 0.12)));
      } else if (this.cameraMode === 'follow') {
        this.orbit.wantDistance = Math.max(6, Math.min(320, this.orbit.wantDistance * (1 + Math.sign(event.deltaY) * 0.15)));
      }
    }, { passive: false });
  }

  /** Convert a client-space point to a NDC vector for raycasting. */
  _ndc(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
  }

  /** Where a click lands in the world: on a solid if it hits one, else the floor. */
  pickPosition(clientX, clientY) {
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(this._ndc(clientX, clientY), this.camera);
    const hits = raycaster.intersectObjects(this.solids, false);
    if (hits.length) {
      const point = hits[0].point;
      return [point.x, point.y + 2, point.z];
    }
    const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const point = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(floor, point)) {
      return [point.x, 0, point.z];
    }
    return null;
  }

  // -- frame ---------------------------------------------------------------
  render(dt, snapshot) {
    if (!snapshot) return;
    const primary = snapshot.flies && snapshot.flies[0];
    this._updateCamera(dt, primary);
    this.scene.updateMatrixWorld();
    for (const model of this.flyModels.values()) {
      if (model.animate) model.animate(dt, this.clock.elapsedTime);
    }
    for (const entry of this.stimulusModels.values()) {
      if (entry.animate) entry.animate(dt, this.clock.elapsedTime);
    }
    // Ceiling fans and other mechanical props spin at their real rate.
    if (this.environmentGroup && this.environmentGroup.userData.animate) {
      this.environmentGroup.userData.animate(dt);
    }

    const { width, height } = this._targetSize();
    if (this.canvas.width !== width || this.canvas.height !== height) this.resize();

    // Keep the sky centred on the camera so it never runs into the far plane.
    if (this.sky) this.sky.position.copy(this.camera.position);

    // Exponential-squared fog is authored for a fly's-eye view. From the tactical
    // camera roughly a thousand units away, the same density leaves barely half the
    // scene unfogged and the room collapses into a near-black smear, so the density
    // is scaled down as the camera pulls back. The close view is left untouched.
    if (this.scene.fog) {
      const viewDistance = Math.max(1, this.orbit.distance);
      this.scene.fog.density = this.fogBase * Math.min(1, 90 / viewDistance);
    }

    const uniforms = this.compositeMaterial.uniforms;
    const flyVision = this.cameraMode === 'flyvision';
    uniforms.uFlyVision.value = flyVision ? 1 : 0;
    uniforms.uNear.value = this.camera.near;
    uniforms.uFar.value = this.camera.far;
    uniforms.uTime.value = this.clock.elapsedTime;
    uniforms.uFocusDist.value = flyVision ? 30 : Math.max(12, this.orbit.distance * 0.85);
    uniforms.uAperture.value = flyVision ? 0 : (this.cameraMode === 'tactical' ? 0.22 : 0.75);

    // Retarget the shadow frustum. In follow and fly-vision the subject is the fly, so
    // the frustum tracks it and stays small. The tactical camera sees the whole arena,
    // so it has to pull back out or the room simply loses its shadows.
    if (primary && this.cameraMode !== 'tactical') {
      this.sunTarget.set(primary.position[0], primary.position[1], primary.position[2]);
    } else {
      this.sunTarget.copy(this.orbit.target);
    }
    const shadowSpan = this.cameraMode === 'tactical'
      ? Math.max(this.environment ? this.environment.size[0] : 900, this.environment ? this.environment.size[2] : 600) * 0.75
      : 70;
    if (Math.abs(shadowSpan - (this.shadowSpan || 0)) > 1) {
      this.shadowSpan = shadowSpan;
      const camera = this.sun.shadow.camera;
      camera.left = -shadowSpan;
      camera.right = shadowSpan;
      camera.top = shadowSpan;
      camera.bottom = -shadowSpan;
      camera.updateProjectionMatrix();
    }
    // The key light keeps its authored direction but follows the subject, so the
    // shadow direction on the ground stays constant as the fly moves across the room.
    this.sun.target.position.copy(this.sunTarget);
    this.sun.position.set(
      this.sunTarget.x + this.sunOffset.x,
      this.sunTarget.y + this.sunOffset.y,
      this.sunTarget.z + this.sunOffset.z,
    );
    this.sun.target.updateMatrixWorld();

    // Optical flow hint: the fly's own velocity projected into screen space.
    if (primary) {
      const speed = Math.min(1, (primary.speed || 0) / 400);
      const heading = primary.heading || 0;
      const rel = heading - this.orbit.yaw;
      uniforms.uFlow.value.set(Math.sin(rel) * 2.2, -Math.cos(rel) * 2.2);
      uniforms.uFlowStrength.value = flyVision ? speed * 1.6 + 0.15 : 0;
    }

    this.renderer.setRenderTarget(this.sceneTarget);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);

    if (!flyVision) {
      this._ambientOcclusion();
      this._bloom();
      uniforms.tBloom.value = this.bloomTargets[0].texture;
      uniforms.tBloomWide.value = this.bloomTargets[2].texture;
    }

    this.renderer.setRenderTarget(null);
    this.quad.material = this.compositeMaterial;
    this.renderer.render(this.quadScene, this.quadCamera);

    this._adapt(dt);
  }

  /**
   * Screen-space AO, then a separable blur across it.
   *
   * The blur is not optional: the per-pixel kernel rotation that removes the AO's
   * banding turns the leftover error into noise, and a single separable pass over a
   * half-res buffer is what turns that noise back into a smooth gradient.
   */
  _ambientOcclusion() {
    const uniforms = this.aoMaterial.uniforms;
    uniforms.tDepth.value = this.sceneTarget.depthTexture;
    uniforms.uNear.value = this.camera.near;
    uniforms.uFar.value = this.camera.far;
    uniforms.uTime.value = this.clock.elapsedTime;
    uniforms.uFovTan.value = Math.tan((this.camera.fov * Math.PI) / 360);
    uniforms.uAspect.value = this.camera.aspect;
    // Scale the sampling radius with how far away the ground is. A fixed world radius
    // either misses the fly entirely from the tactical camera or swallows the whole
    // scene up close, so it tracks the orbit distance instead.
    uniforms.uRadius.value = Math.max(4, Math.min(90, this.orbit.distance * 0.28));

    this.quad.material = this.aoMaterial;
    this.renderer.setRenderTarget(this.aoTarget);
    this.renderer.render(this.quadScene, this.quadCamera);

    this.quad.material = this.blurMaterial;
    this.blurMaterial.uniforms.tInput.value = this.aoTarget.texture;
    this.blurMaterial.uniforms.uDirection.value.set(1 / this.aoTarget.width, 0);
    this.renderer.setRenderTarget(this.aoBlurTarget);
    this.renderer.render(this.quadScene, this.quadCamera);

    this.blurMaterial.uniforms.tInput.value = this.aoBlurTarget.texture;
    this.blurMaterial.uniforms.uDirection.value.set(0, 1 / this.aoTarget.height);
    this.renderer.setRenderTarget(this.aoTarget);
    this.renderer.render(this.quadScene, this.quadCamera);

    this.compositeMaterial.uniforms.tAO.value = this.aoTarget.texture;
  }

  /**
   * Hold the frame budget by trading resolution.
   *
   * AO and the existing bloom chain add real per-pixel cost, and the machine this runs
   * on varies. Rather than pick one fixed quality and hope, the renderer measures its
   * own frame time and nudges the internal resolution between 55% and 100% of the
   * device pixel ratio. The changes are small and hysteretic so it settles instead of
   * oscillating, and the canvas is CSS-sized so the layout never moves.
   */
  _adapt(dt) {
    this._frameAccum = (this._frameAccum || 0) + dt;
    this._frameCount = (this._frameCount || 0) + 1;
    if (this._frameCount < 24) return;

    const average = this._frameAccum / this._frameCount;
    this._frameAccum = 0;
    this._frameCount = 0;

    const scale = this.renderScale;
    let next = scale;
    if (average > 0.024 && scale > 0.55) next = Math.max(0.55, scale - 0.1);
    else if (average < 0.0135 && scale < 1) next = Math.min(1, scale + 0.05);

    if (Math.abs(next - scale) > 0.001) {
      this.renderScale = next;
      this.resize();
      if (this.onQualityChange) this.onQualityChange(next);
    }
  }

  _bloom() {
    // Bright pass into mip 0, then per mip: horizontal blur into scratch,
    // vertical blur back, then downsample into the next mip. Every pass reads a
    // texture it is not writing, which is what keeps this correct on any driver.
    this.quad.material = this.brightMaterial;
    this.brightMaterial.uniforms.tScene.value = this.sceneTarget.texture;
    this.renderer.setRenderTarget(this.bloomTargets[0]);
    this.renderer.render(this.quadScene, this.quadCamera);

    this.quad.material = this.blurMaterial;
    for (let index = 0; index < this.bloomTargets.length; index++) {
      const target = this.bloomTargets[index];
      const scratch = this.bloomScratch[index];

      if (index > 0) {
        // Seed this mip from the previous, blurred one: the downsample *is* the
        // widening, so no separate downsample pass is needed.
        this.blurMaterial.uniforms.tInput.value = this.bloomTargets[index - 1].texture;
        this.blurMaterial.uniforms.uDirection.value.set(
          1 / Math.max(1, target.width), 1 / Math.max(1, target.height));
        this.renderer.setRenderTarget(target);
        this.renderer.render(this.quadScene, this.quadCamera);
      }

      this.blurMaterial.uniforms.tInput.value = target.texture;
      this.blurMaterial.uniforms.uDirection.value.set(1 / Math.max(1, target.width), 0);
      this.renderer.setRenderTarget(scratch);
      this.renderer.render(this.quadScene, this.quadCamera);

      this.blurMaterial.uniforms.tInput.value = scratch.texture;
      this.blurMaterial.uniforms.uDirection.value.set(0, 1 / Math.max(1, target.height));
      this.renderer.setRenderTarget(target);
      this.renderer.render(this.quadScene, this.quadCamera);
    }
  }

  resize() {
    const { width, height } = this._targetSize();
    this.renderer.setSize(width, height, false);
    this.canvas.width = width;
    this.canvas.height = height;
    this.sceneTarget.setSize(width, height);
    this.sceneTarget.depthTexture.image.width = width;
    this.sceneTarget.depthTexture.image.height = height;
    this.sceneTarget.depthTexture.needsUpdate = true;
    for (let index = 0; index < this.bloomTargets.length; index++) {
      const mipWidth = Math.max(2, width >> (index + 1));
      const mipHeight = Math.max(2, height >> (index + 1));
      this.bloomTargets[index].setSize(mipWidth, mipHeight);
      this.bloomScratch[index].setSize(mipWidth, mipHeight);
    }
    const aoWidth = Math.max(2, width >> 1);
    const aoHeight = Math.max(2, height >> 1);
    this.aoTarget.setSize(aoWidth, aoHeight);
    this.aoBlurTarget.setSize(aoWidth, aoHeight);
    this.aoMaterial.uniforms.uResolution.value.set(aoWidth, aoHeight);
    this.compositeMaterial.uniforms.uResolution.value.set(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  setPostOption(name, value) {
    const uniforms = this.compositeMaterial.uniforms;
    if (name === 'bloom') uniforms.uBloomStrength.value = value;
    if (name === 'grain') uniforms.uGrain.value = value;
    if (name === 'vignette') uniforms.uVignette.value = value;
    if (name === 'exposure') uniforms.uExposure.value = value;
    if (name === 'ao') uniforms.uAOStrength.value = value;
    if (name === 'aberration') uniforms.uAberration.value = value;
  }

  /** Freeze the adaptive resolution controller, e.g. when the user picks a preset. */
  setRenderScale(scale) {
    this.renderScale = Math.max(0.5, Math.min(1, scale));
    this._frameAccum = 0;
    this._frameCount = 0;
    this.resize();
  }
}
