/**
 * Live connectome HUD: a raymarched 3-D brain built from the real VFB volume.
 *
 * The intensity volume is the JRC2018Unisex template shipped with this workspace,
 * reduced to a 303 x 142 x 87 uint8 texture by the Python backend. A second channel
 * carries a per-voxel neuropil id, so the shader can look up a region's live
 * activation and brighten exactly that part of the anatomy. The glow is therefore
 * spatial: it sits where the antennal lobes actually are, not on a sphere standing in
 * for them. A third channel is the real registered VFB_00102212 mask, drawn as a
 * distinct accent so you can see genuine registered annotation next to the ROIs.
 *
 * Geometry is one box. Everything visible is the fragment shader walking the volume
 * until the tissue ends, so the silhouette is the real brain rather than a convex
 * hull of blobs.
 *
 * Two details that are easy to get wrong and were:
 *
 * * The camera is orthographic, so every ray shares one direction. Deriving the ray
 *   from the fragment's own position (the usual trick for a perspective raymarch)
 *   produces a radial fan that looks plausible and is wrong.
 * * The shaders are GLSL ES 3.00. Sampling a `sampler3D` needs `texture()`, which
 *   does not exist in GLSL ES 1.00, so `glslVersion: GLSL3` is mandatory and the
 *   syntax has to be consistent with it throughout.
 *
 * Atlas orientation, established on the Python side by a mirror-symmetry test:
 * x is left-right, y is anterior-posterior, z is dorsal-ventral.
 */

import * as THREE from '../vendor/three.module.js';

const VERT = /* glsl */`
  precision highp float;
  in vec3 position;
  in vec3 normal;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  out vec3 vLocal;
  out vec3 vNormal;
  void main() {
    vLocal = position;
    vNormal = normal;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG = /* glsl */`
  precision highp float;
  // GLSL ES 3.00 requires an explicit precision for opaque sampler types; without
  // this the whole program fails to link and the HUD silently draws nothing.
  precision highp sampler3D;

  in vec3 vLocal;
  in vec3 vNormal;
  out vec4 fragColor;

  uniform sampler3D tVolume;
  uniform sampler3D tRegions;
  uniform sampler3D tMask;
  uniform vec3  uVolumeSize;
  uniform vec3  uRayDir;        // camera-space ray direction, world normalised
  uniform float uMaxSteps;
  uniform float uThreshold;
  uniform float uTime;
  uniform float uMaskStrength;
  uniform float uActivation[12];
  uniform vec3  uRegionColor[12];
  uniform float uRegionCount;

  float regionIdAt(vec3 p) {
    // Region ids are stored as (index + 1) / 255; 0 means unassigned tissue.
    return floor(texture(tRegions, p).r * 255.0 + 0.5) - 1.0;
  }

  void main() {
    // The box spans -0.5..0.5, so the entry point in texture space is vLocal + 0.5.
    vec3 entry = vLocal + 0.5;
    vec3 raydir = normalize(uRayDir);
    vec3 normalized_raydir = raydir;

    // A step should be roughly isotropic in *voxel* terms, so the step is scaled per
    // axis by the volume's aspect. Total path is about the box diagonal.
    vec3 anisotropy = max(uVolumeSize.x, max(uVolumeSize.y, uVolumeSize.z)) / uVolumeSize;
    vec3 stepSize = raydir * anisotropy * (1.9 / uMaxSteps);

    // Advance to the box surface if we are outside it (the box is drawn as a solid,
    // so entry is on the surface, but the guard costs nothing).
    vec3 position = entry;

    vec4 accumulated = vec4(0.0);
    float maskAccent = 0.0;

    for (int index = 0; index < 512; index++) {
      if (float(index) >= uMaxSteps) break;
      if (accumulated.a > 0.95) break;

      if (any(lessThan(position, vec3(0.0))) || any(greaterThan(position, vec3(1.0)))) break;

      float density = texture(tVolume, position).r;
      float tissue = smoothstep(uThreshold, uThreshold + 0.30, density);

      if (tissue > 0.008) {
        float id = regionIdAt(position);
        vec3 tint = vec3(0.42, 0.50, 0.62);
        float glow = 0.0;
        if (id > -0.5 && id < uRegionCount) {
          int region = int(id + 0.5);
          tint = uRegionColor[region];
          glow = uActivation[region];
        }

        // Anatomy is always visible; activity brightens it. A region at rest still
        // reads, which is what makes this look like a brain and not a light show.
        float mask = texture(tMask, position).r;
        maskAccent = max(maskAccent, step(0.35, mask) * tissue);

        vec3 base = mix(vec3(0.30, 0.36, 0.46), tint, 0.58);
        vec3 colour = base * (0.34 + 0.58 * density) + tint * glow * 1.6;
        // A slow pulse separates a sustained drive from a single transient.
        colour *= 1.0 + glow * 0.20 * sin(uTime * 3.1 + id * 1.7);

        float alpha = tissue * (0.050 + 0.155 * glow + 0.05 * density);
        accumulated.rgb += (1.0 - accumulated.a) * colour * alpha;
        accumulated.a += (1.0 - accumulated.a) * alpha;
      }

      position += stepSize;
    }

    // The real registered mask, drawn as a violet accent layer so genuine annotation
    // is visually distinct from the approximate ROIs.
    if (maskAccent > 0.0) {
      accumulated.rgb += vec3(0.55, 0.36, 1.0) * maskAccent * uMaskStrength * (1.0 - accumulated.a * 0.6);
      accumulated.a = min(1.0, accumulated.a + maskAccent * uMaskStrength * 0.22);
    }

    if (accumulated.a < 0.008) discard;

    // A faint rim keeps the volume's edge legible on a near-black panel.
    float rim = pow(1.0 - abs(dot(normalize(vNormal), normalized_raydir)), 2.0);
    accumulated.rgb += vec3(0.16, 0.42, 0.52) * rim * 0.20;

    fragColor = vec4(accumulated.rgb, min(1.0, accumulated.a * 1.35));
  }
`;

export class BrainView {
  constructor(canvas, manifest) {
    this.canvas = canvas;
    this.manifest = manifest;
    this.regions = manifest.atlas.regions;
    this.shape = manifest.atlas.shape;
    this.regionShape = manifest.atlas.regionShape;
    this.activation = new Float32Array(12);
    this.autoRotate = true;
    this.yaw = 0.55;
    this.pitch = 0.22;
    this.dragging = false;
    this.ready = false;
    this.maxSteps = 104;

    this.renderer = new THREE.WebGLRenderer({
      canvas, alpha: true, antialias: true, powerPreference: 'low-power',
    });
    this.renderer.setClearColor(0x05070d, 0);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-0.62, 0.62, 0.46, -0.46, -2, 2);

    const colors = Array.from({ length: 12 }, (_, index) => {
      const region = this.regions[index];
      if (!region) return new THREE.Vector3(0.5, 0.5, 0.5);
      return new THREE.Vector3(region.color[0] / 255, region.color[1] / 255, region.color[2] / 255);
    });

    this.uniforms = {
      tVolume: { value: null },
      tRegions: { value: null },
      tMask: { value: null },
      uVolumeSize: { value: new THREE.Vector3(this.shape[0], this.shape[1], this.shape[2]) },
      uRayDir: { value: new THREE.Vector3(0, 0, -1) },
      uMaxSteps: { value: this.maxSteps },
      uThreshold: { value: 0.12 },
      uTime: { value: 0 },
      uMaskStrength: { value: 0.85 },
      uActivation: { value: this.activation },
      uRegionColor: { value: colors },
      uRegionCount: { value: Math.min(12, this.regions.length) },
    };

    this.material = new THREE.RawShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: this.uniforms,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      glslVersion: THREE.GLSL3,
    });

    this.mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);

    // A neutral 1x1x1 texture stands in for the mask when none is loaded, so the
    // shader never samples an unbound sampler.
    const blank = new Uint8Array([0]);
    this.blankTexture = new THREE.Data3DTexture(blank, 1, 1, 1);
    this.blankTexture.format = THREE.RedFormat;
    this.blankTexture.type = THREE.UnsignedByteType;
    this.blankTexture.needsUpdate = true;
    this.uniforms.tMask.value = this.blankTexture;

    this._bindInput();
  }

  /** Fetch the volume blobs and upload them as 3-D textures. */
  async load() {
    const blobs = this.manifest.atlas.blobs;
    const [templateBuffer, regionBuffer] = await Promise.all([
      fetch(blobs.template).then((response) => response.arrayBuffer()),
      fetch(blobs.regionIds).then((response) => response.arrayBuffer()),
    ]);

    this.templateTexture = this._texture(new Uint8Array(templateBuffer), this.shape);
    this.regionTexture = this._texture(new Uint8Array(regionBuffer), this.regionShape);
    this.uniforms.tVolume.value = this.templateTexture;
    this.uniforms.tRegions.value = this.regionTexture;

    if (blobs.mask) {
      try {
        const maskBuffer = await fetch(blobs.mask).then((response) => response.arrayBuffer());
        this.maskTexture = this._texture(new Uint8Array(maskBuffer), this.shape);
        this.uniforms.tMask.value = this.maskTexture;
      } catch (error) {
        console.warn('registered mask unavailable, continuing without it', error);
      }
    }

    this.ready = true;
    const meta = this.manifest.atlas.meta || {};
    const source = meta.sources && meta.sources.template
      ? meta.sources.template.path.split('/').pop()
      : 'template';
    const megavoxels = (this.shape[0] * this.shape[1] * this.shape[2]) / 1e6;
    const midline = (meta.orientation && meta.orientation.midlineAxis) || '?';
    return `${source} · ${this.shape.join('×')} · ${megavoxels.toFixed(2)} M voxels · midline ${midline}`;
  }

  _texture(bytes, shape) {
    // Data3DTexture wants x-fastest ordering, which is exactly what the NRRD reader
    // and its decimator produce.
    const texture = new THREE.Data3DTexture(bytes, shape[0], shape[1], shape[2]);
    texture.format = THREE.RedFormat;
    texture.type = THREE.UnsignedByteType;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.wrapR = THREE.ClampToEdgeWrapping;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;
    return texture;
  }

  _bindInput() {
    const canvas = this.canvas;
    canvas.addEventListener('pointerdown', (event) => {
      this.dragging = true;
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      this.autoRotate = false;
      try { canvas.setPointerCapture(event.pointerId); } catch { /* ignore */ }
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.dragging) return;
      this.yaw += (event.clientX - this.lastX) * 0.009;
      this.pitch = Math.max(-1.25, Math.min(1.25, this.pitch + (event.clientY - this.lastY) * 0.007));
      this.lastX = event.clientX;
      this.lastY = event.clientY;
    });
    canvas.addEventListener('pointerup', (event) => {
      this.dragging = false;
      try { canvas.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
      window.setTimeout(() => { this.autoRotate = true; }, 3000);
    });
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const scale = event.deltaY > 0 ? 1.06 : 0.94;
      const halfHeight = Math.abs(this.camera.top) * scale;
      if (halfHeight < 0.16 || halfHeight > 1.6) return;
      const aspect = this.camera.right / this.camera.top;
      this.camera.top = halfHeight;
      this.camera.bottom = -halfHeight;
      this.camera.left = -halfHeight * aspect * -1;
      this.camera.right = halfHeight * aspect;
      this.camera.updateProjectionMatrix();
    }, { passive: false });
  }

  /** Feed live region activation from the kernel snapshot. */
  setActivation(regionActivation) {
    if (!regionActivation) return;
    const count = Math.min(12, this.regions.length);
    for (let index = 0; index < count; index++) {
      const value = regionActivation[this.regions[index].key];
      const target = Math.max(0, Math.min(1, typeof value === 'number' ? value : 0));
      // Ease so a single-tick spike does not strobe the panel.
      this.activation[index] += (target - this.activation[index]) * 0.24;
    }
  }

  render(dt, time) {
    if (!this.ready) return;
    if (this.autoRotate && !this.dragging) this.yaw += dt * 0.2;
    this.uniforms.uTime.value = time;

    const cosPitch = Math.cos(this.pitch);
    this.camera.position.set(
      Math.sin(this.yaw) * cosPitch, Math.sin(this.pitch), Math.cos(this.yaw) * cosPitch,
    );
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld();

    // Orthographic: one shared ray direction for every fragment.
    this.camera.getWorldDirection(this.uniforms.uRayDir.value);

    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(2, Math.floor(rect.width));
    const height = Math.max(2, Math.floor(rect.height));
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(width, height, false);
    this.canvas.width = Math.floor(width * ratio);
    this.canvas.height = Math.floor(height * ratio);

    const aspect = width / Math.max(1, height);
    const halfHeight = Math.abs(this.camera.top);
    this.camera.left = -halfHeight * aspect;
    this.camera.right = halfHeight * aspect;
    this.camera.updateProjectionMatrix();
  }

  setQuality(steps) {
    this.maxSteps = Math.max(40, Math.min(220, steps));
    this.uniforms.uMaxSteps.value = this.maxSteps;
  }
}
