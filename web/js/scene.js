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

      vec3 sharp = texture2D(tScene, uv).rgb;
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
      vec3 scene = mix(sharp, blurred, smoothstep(0.0, 0.55, coc));

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
    float grain = (hash(uv * uResolution + fract(uTime) * 91.7) - 0.5) * uGrain;
    color += grain;

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
    this.sun.shadow.camera.left = -700;
    this.sun.shadow.camera.right = 700;
    this.sun.shadow.camera.top = 700;
    this.sun.shadow.camera.bottom = -700;
    this.scene.add(this.hemi, this.sun, this.sun.target);

    this.fill = new THREE.PointLight(0xbcd4ff, 0.5, 2400, 2.0);
    this.fill.position.set(-360, 320, -260);
    this.scene.add(this.fill);

    // Fly vision keeps the scene visible at extreme close range, where the
    // standard camera's near plane would clip the fly's own body.
    this.flyLight = new THREE.PointLight(0xffffff, 0.0, 120, 2.0);
    this.scene.add(this.flyLight);
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
    const width = Math.max(2, Math.floor(this.canvas.clientWidth * this.pixelRatio));
    const height = Math.max(2, Math.floor(this.canvas.clientHeight * this.pixelRatio));
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

    this.scene.background = new THREE.Color(spec.skybox || '#0b1220');
    // The density in the environment table is authored for a fly's-eye view; how it
    // is applied across zoom levels is decided per frame in render().
    this.fogBase = spec.fog ?? 0.0006;
    this.scene.fog = new THREE.FogExp2(spec.skybox || '#0b1220', this.fogBase);

    const light = spec.ambientLight ?? 0.6;
    this.hemi.intensity = 0.22 + 0.5 * light;
    this.sun.intensity = 0.25 + 1.9 * light;
    this.sun.position.set(spec.size[0] * 0.35, spec.size[1] * 1.4, spec.size[2] * 0.45);
    this.fill.intensity = 0.25 + 0.5 * light;

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
      this._bloom();
      uniforms.tBloom.value = this.bloomTargets[0].texture;
      uniforms.tBloomWide.value = this.bloomTargets[2].texture;
    }

    this.renderer.setRenderTarget(null);
    this.quad.material = this.compositeMaterial;
    this.renderer.render(this.quadScene, this.quadCamera);
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
  }
}
