/**
 * Environments, stimuli and odour plumes.
 *
 * Geometry is generated from the same prop list the kernel uses for physics, so the
 * surface the fly walks on is the surface you see. Two things are worth calling out:
 *
 * **Plumes are drawn from the kernel's own puff list.** The fly senses analytic
 * Gaussian puffs in Python; each puff arrives here as a position, a sigma and a
 * strength, and is drawn as a soft additive billboard. There is no second plume
 * simulation, so what you see cannot drift out of sync with what the fly smells.
 *
 * **Thermal zones are drawn as a floor patch**, because that is what a hot surface
 * physically is: a region of the floor with an elevated temperature field above it.
 */

import * as THREE from '../vendor/three.module.js';

// Materials are cached so that repeated props (every grass blade's instanced mesh,
// every petri dish) share one GPU program. The key must therefore include anything
// that varies, or the first prop to be built would hand its colour to all the rest.
const MATERIALS = {};
const CACHED = new Set();

function material(key, build) {
  if (!MATERIALS[key]) {
    MATERIALS[key] = build();
    CACHED.add(MATERIALS[key]);
  }
  return MATERIALS[key];
}

function surface(color, roughness, metalness) {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(color), roughness, metalness,
    side: THREE.DoubleSide,
  });
}

/** A soft radial billboard used for plumes, heat shimmer and light pools. */
function radialTexture(inner = 'rgba(255,255,255,0.95)', mid = 'rgba(255,255,255,0.35)') {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, inner);
  gradient.addColorStop(0.45, mid);
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

const SOFT_TEXTURE = { value: null };
function softTexture() {
  if (!SOFT_TEXTURE.value) SOFT_TEXTURE.value = radialTexture();
  return SOFT_TEXTURE.value;
}

function addProp(group, solids, prop, spec) {
  const size = spec.size;
  const push = (mesh, solid = true) => {
    group.add(mesh);
    if (solid) solids.push(mesh);
    return mesh;
  };

  switch (prop.type) {
    case 'counter':
    case 'bench': {
      const geometry = new THREE.BoxGeometry(prop.sx, prop.sy, prop.sz, 12, 2, 12);
      const mesh = new THREE.Mesh(geometry, material(`surface:${prop.type}`, () =>
        surface(prop.type === 'bench' ? '#c9cfd6' : '#8d7f6e', 0.62, 0.06)));
      mesh.position.set(prop.x, prop.y + prop.sy * 0.5, prop.z);
      mesh.receiveShadow = true;
      push(mesh);
      // Front skirt so the counter reads as a solid volume from the tactical view.
      const skirt = new THREE.Mesh(
        new THREE.BoxGeometry(prop.sx, 220, 8),
        material('skirt', () => surface('#2f2a24', 0.9, 0.02)),
      );
      skirt.position.set(prop.x, prop.y - 100, prop.z + prop.sz * 0.5);
      skirt.receiveShadow = true;
      group.add(skirt);
      break;
    }
    case 'ground': {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(prop.sx, prop.sz, 40, 40),
        material('ground', () => surface('#4a5a3a', 0.95, 0.0)),
      );
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(prop.x, prop.y, prop.z);
      mesh.receiveShadow = true;
      push(mesh);
      break;
    }
    case 'bin_floor': {
      const mesh = new THREE.Mesh(new THREE.CircleGeometry(prop.r, 40), material('binfloor', () => surface('#23252a', 0.95, 0.05)));
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(prop.x, prop.y + 0.5, prop.z);
      mesh.receiveShadow = true;
      push(mesh);
      break;
    }
    case 'bin_wall': {
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(prop.r, prop.r * 0.94, prop.h, 44, 1, true),
        material('binwall', () => new THREE.MeshStandardMaterial({
          color: 0x2a2d33, roughness: 0.72, metalness: 0.35, side: THREE.BackSide,
        })),
      );
      mesh.position.set(prop.x, prop.y, prop.z);
      mesh.receiveShadow = true;
      push(mesh, false);
      break;
    }
    case 'wall': {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(prop.sx, prop.sy, prop.sz), material('wall', () => surface('#1c1e22', 0.95, 0.02)));
      mesh.position.set(prop.x, prop.y, prop.z);
      mesh.receiveShadow = true;
      push(mesh, false);
      break;
    }
    case 'fruit_bowl': {
      const bowl = new THREE.Mesh(
        new THREE.SphereGeometry(prop.r, 32, 18, 0, Math.PI * 2, Math.PI * 0.62, Math.PI * 0.38),
        material('bowl', () => surface('#d8d3c8', 0.35, 0.08)),
      );
      bowl.scale.y = 0.82;
      bowl.position.set(prop.x, prop.y + 26, prop.z);
      bowl.castShadow = true;
      bowl.receiveShadow = true;
      group.add(bowl);
      break;
    }
    case 'rotting_fruit':
    case 'fruit_chunk': {
      const fruitColor = prop.color || '#8a6b2f';
      const fruit = new THREE.Mesh(new THREE.SphereGeometry(prop.r, 22, 16), material(`fruit:${fruitColor}`, () => surface(fruitColor, 0.85, 0.02)));
      fruit.scale.set(1.0, 0.82, 1.0);
      fruit.position.set(prop.x, prop.y + prop.r * 0.4, prop.z);
      fruit.castShadow = true;
      fruit.receiveShadow = true;
      push(fruit, false);
      break;
    }
    case 'sink': {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(prop.sx, prop.sy, prop.sz), material('sink', () => surface('#b9bfc6', 0.28, 0.62)));
      mesh.position.set(prop.x, prop.y, prop.z);
      mesh.receiveShadow = true;
      push(mesh, false);
      break;
    }
    case 'drain': {
      const mesh = new THREE.Mesh(new THREE.TorusGeometry(prop.r, 5, 10, 28), material('drain', () => surface('#6a6f76', 0.4, 0.8)));
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(prop.x, prop.y, prop.z);
      group.add(mesh);
      break;
    }
    case 'mug': {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(prop.r, prop.r * 0.9, prop.r * 1.6, 24, 1, true), material('mug', () => surface('#e6e2da', 0.4, 0.05)));
      mesh.position.set(prop.x, prop.y + prop.r * 0.8, prop.z);
      mesh.castShadow = true;
      push(mesh, false);
      break;
    }
    case 'ceiling_fan': {
      const fan = new THREE.Group();
      fan.position.set(prop.x, prop.y, prop.z);
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(18, 22, 16, 20), material('fanhub', () => surface('#3d4149', 0.5, 0.55)));
      fan.add(hub);
      const blades = new THREE.Group();
      for (let index = 0; index < 4; index++) {
        const blade = new THREE.Mesh(new THREE.BoxGeometry(prop.r * 0.92, 4, 34), material('blade', () => surface('#5a6068', 0.5, 0.4)));
        blade.position.x = prop.r * 0.48;
        blade.rotation.z = 0.16;
        const arm = new THREE.Group();
        arm.add(blade);
        arm.rotation.y = (index / 4) * Math.PI * 2;
        blades.add(arm);
      }
      fan.add(blades);
      fan.userData.spin = (prop.rpm || 40) * (Math.PI * 2 / 60) * 0.16;
      fan.userData.blades = blades;
      group.add(fan);
      break;
    }
    case 'lamp': {
      const shade = new THREE.Mesh(new THREE.ConeGeometry(prop.r, prop.r * 0.9, 24, 1, true), material('lampShade', () => surface('#d8d4cc', 0.45, 0.3)));
      shade.position.set(prop.x, prop.y, prop.z);
      shade.rotation.x = Math.PI;
      group.add(shade);
      const bulb = new THREE.Mesh(new THREE.SphereGeometry(prop.r * 0.4, 16, 12), new THREE.MeshBasicMaterial({ color: 0xfff3c4 }));
      bulb.position.set(prop.x, prop.y - prop.r * 0.5, prop.z);
      group.add(bulb);
      const light = new THREE.SpotLight(0xfff2d0, 220, 1500, 0.85, 0.55, 1.6);
      light.position.set(prop.x, prop.y - prop.r * 0.5, prop.z);
      light.target.position.set(prop.x, 0, prop.z);
      light.castShadow = true;
      light.shadow.mapSize.set(1024, 1024);
      light.shadow.bias = -0.002;
      group.add(light, light.target);
      break;
    }
    case 'lamp_bar': {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(prop.sx, 14, prop.sz), material('lampBar', () => surface('#e2e6ea', 0.4, 0.4)));
      bar.position.set(prop.x, prop.y, prop.z);
      group.add(bar);
      // Three point lights rather than a RectAreaLight: a rect light needs its own
      // LTC lookup tables initialised, and three point lights read identically here.
      for (const offset of [-0.34, 0, 0.34]) {
        const light = new THREE.PointLight(0xffffff, 260, 900, 1.7);
        light.position.set(prop.x + offset * prop.sx, prop.y - 10, prop.z);
        light.castShadow = offset === 0;
        if (offset === 0) light.shadow.mapSize.set(1024, 1024);
        group.add(light);
      }
      break;
    }
    case 'sun': {
      const light = new THREE.DirectionalLight(0xfff6e0, 2.6);
      light.position.set(prop.x, prop.y, prop.z);
      light.castShadow = true;
      light.shadow.mapSize.set(2048, 2048);
      light.shadow.camera.left = -600;
      light.shadow.camera.right = 600;
      light.shadow.camera.top = 600;
      light.shadow.camera.bottom = -600;
      light.shadow.bias = -0.0015;
      group.add(light, light.target);
      break;
    }
    case 'flower':
    case 'flowers': {
      const count = prop.count || 8;
      for (let index = 0; index < count; index++) {
        const angle = (index / count) * Math.PI * 2 + Math.random() * 0.4;
        const radius = prop.r * (0.35 + Math.random() * 0.65);
        const x = prop.x + Math.cos(angle) * radius;
        const z = prop.z + Math.sin(angle) * radius;
        const stem = new THREE.Mesh(new THREE.CylinderGeometry(2, 3, 60 + Math.random() * 40, 6), material('stem', () => surface('#3f5a2a', 0.9, 0)));
        const height = 60 + Math.random() * 40;
        stem.position.set(x, prop.y + height * 0.5, z);
        const petals = new THREE.Mesh(
          new THREE.SphereGeometry(14 + Math.random() * 6, 10, 8),
          new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(0.09 + Math.random() * 0.08, 0.75, 0.62), roughness: 0.75 }),
        );
        petals.scale.set(1, 0.55, 1);
        petals.position.set(x, prop.y + height, z);
        stem.castShadow = true;
        petals.castShadow = true;
        group.add(stem, petals);
      }
      break;
    }
    case 'grass': {
      const count = prop.count || 200;
      const geometry = new THREE.ConeGeometry(2.4, 46, 3);
      const mesh = new THREE.InstancedMesh(geometry, material('grass', () => surface('#4f6b34', 0.95, 0)), count);
      const matrix = new THREE.Matrix4();
      const quat = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      const position = new THREE.Vector3();
      for (let index = 0; index < count; index++) {
        position.set(
          prop.x + (Math.random() - 0.5) * prop.sx,
          prop.y + 23,
          prop.z + (Math.random() - 0.5) * prop.sz,
        );
        quat.setFromEuler(new THREE.Euler((Math.random() - 0.5) * 0.5, 0, (Math.random() - 0.5) * 0.5));
        const height = 0.55 + Math.random() * 0.9;
        scale.set(1, height, 1);
        matrix.compose(position, quat, scale);
        mesh.setMatrixAt(index, matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      break;
    }
    case 'puddle': {
      const mesh = new THREE.Mesh(new THREE.CircleGeometry(prop.r, 36), material('puddle', () => new THREE.MeshPhysicalMaterial({
        color: 0x1b2a33, roughness: 0.06, metalness: 0.1, transparent: true, opacity: 0.92,
        clearcoat: 1.0, clearcoatRoughness: 0.05,
      })));
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(prop.x, prop.y + 0.6, prop.z);
      mesh.receiveShadow = true;
      group.add(mesh);
      break;
    }
    case 'rock': {
      const mesh = new THREE.Mesh(
        new THREE.DodecahedronGeometry(prop.r, 1),
        material('rock', () => surface('#6b6a63', 0.95, 0.05)),
      );
      mesh.position.set(prop.x, prop.y + prop.r * 0.5, prop.z);
      mesh.scale.set(1, 0.72, 1.1);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      push(mesh, false);
      break;
    }
    case 'trash_pile': {
      for (let index = 0; index < 7; index++) {
        const trashColor = prop.color || '#4a4a3a';
        const chunk = new THREE.Mesh(
          new THREE.IcosahedronGeometry(prop.r * (0.35 + Math.random() * 0.5), 0),
          material(`trash:${trashColor}`, () => surface(trashColor, 0.95, 0.02)),
        );
        chunk.position.set(
          prop.x + (Math.random() - 0.5) * prop.r * 1.4,
          prop.y + Math.random() * prop.r * 0.9,
          prop.z + (Math.random() - 0.5) * prop.r * 1.4,
        );
        chunk.rotation.set(Math.random() * 3, Math.random() * 3, Math.random() * 3);
        chunk.castShadow = true;
        chunk.receiveShadow = true;
        group.add(chunk);
      }
      break;
    }
    case 'petri': {
      const dish = new THREE.Mesh(
        new THREE.CylinderGeometry(prop.r, prop.r * 0.96, 14, 36),
        material('petri', () => new THREE.MeshPhysicalMaterial({
          color: 0xdfe6ea, roughness: 0.1, metalness: 0.0, transparent: true,
          opacity: 0.34, transmission: 0.55, thickness: 4, side: THREE.DoubleSide,
        })),
      );
      dish.position.set(prop.x, prop.y + 7, prop.z);
      dish.receiveShadow = true;
      group.add(dish);
      const agarColor = prop.color || '#d8c98a';
      const agar = new THREE.Mesh(new THREE.CylinderGeometry(prop.r * 0.94, prop.r * 0.94, 7, 36), material(`agar:${agarColor}`, () => surface(agarColor, 0.55, 0.02)));
      agar.position.set(prop.x, prop.y + 4, prop.z);
      agar.receiveShadow = true;
      group.add(agar);
      break;
    }
    case 'arena_grid':
    case 'microscope': {
      if (prop.type === 'arena_grid') {
        const grid = new THREE.GridHelper(prop.sx, 22, 0x3b6ea5, 0x243449);
        grid.position.set(prop.x, prop.y + 0.4, prop.z);
        grid.material.transparent = true;
        grid.material.opacity = 0.5;
        group.add(grid);
      } else {
        const base = new THREE.Mesh(new THREE.CylinderGeometry(prop.r * 0.5, prop.r * 0.66, 40, 20), material('scope', () => surface('#20242b', 0.5, 0.5)));
        base.position.set(prop.x, prop.y + 20, prop.z);
        base.castShadow = true;
        group.add(base);
        const arm = new THREE.Mesh(new THREE.BoxGeometry(24, 130, 24), material('scope', () => surface('#20242b', 0.5, 0.5)));
        arm.position.set(prop.x, prop.y + 90, prop.z - 20);
        arm.rotation.x = 0.25;
        arm.castShadow = true;
        group.add(arm);
      }
      break;
    }
    default:
      break;
  }
}

export function buildEnvironment(spec, manifest) {
  const group = new THREE.Group();
  const solids = [];
  for (const prop of spec.props || []) addProp(group, solids, prop, spec);

  // Ceiling fan blade rotation is advanced per frame from a stored omega.
  const spinners = [];
  group.traverse((child) => {
    if (child.userData && child.userData.spin) spinners.push(child);
  });
  group.userData.spinners = spinners;
  group.userData.animate = (dt) => {
    for (const fan of spinners) {
      if (!fan.userData.fanPhase) fan.userData.fanPhase = 0;
      fan.userData.fanPhase += dt * fan.userData.spin;
      fan.userData.blades.rotation.y = fan.userData.fanPhase;
    }
  };

  void manifest;
  return { group, solids };
}

/** Build the visual for one placed stimulus. */
export function buildStimulus(spec, environment) {
  const group = new THREE.Group();
  const entry = { group, spec, plumes: [], markers: {} };

  const color = new THREE.Color(spec.color);

  if (spec.kind === 'odor') {
    // The source itself: a small emissive marker so you can see where it is.
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(5, (spec.sigma || 40) * 0.16), 16, 12),
      new THREE.MeshBasicMaterial({ color }),
    );
    marker.position.set(spec.x, spec.y, spec.z);
    group.add(marker);
    entry.markers.source = marker;

    // A pool of soft billboards, reused as puffs come and go.
    const plumeMaterial = new THREE.SpriteMaterial({
      map: softTexture(),
      color,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      fog: true,
    });
    for (let index = 0; index < 10; index++) {
      const sprite = new THREE.Sprite(plumeMaterial.clone());
      sprite.visible = false;
      group.add(sprite);
      entry.plumes.push(sprite);
    }
  } else if (spec.kind === 'thermal') {
    const hot = spec.deltaC > 0;
    const patch = new THREE.Mesh(
      new THREE.CircleGeometry(Math.max(20, (spec.sigma || 80) * 1.1), 40),
      new THREE.MeshBasicMaterial({
        map: softTexture(),
        color: hot ? (spec.deltaC > 40 ? 0xff5722 : 0xfbbf24) : 0x7dd3fc,
        transparent: true,
        opacity: hot ? 0.42 : 0.3,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    patch.rotation.x = -Math.PI / 2;
    patch.position.set(spec.x, 1.2, spec.z);
    group.add(patch);
    entry.markers.patch = patch;

    if (spec.deltaC > 40) {
      // A hot burner gets an actual flame so the threat is unmistakable.
      for (let index = 0; index < 9; index++) {
        const flame = new THREE.Sprite(new THREE.SpriteMaterial({
          map: softTexture(), color: 0xff8a3d, transparent: true, opacity: 0.5,
          depthWrite: false, blending: THREE.AdditiveBlending,
        }));
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * (spec.sigma || 80) * 0.9;
        flame.position.set(spec.x + Math.cos(angle) * radius, 6 + Math.random() * 26, spec.z + Math.sin(angle) * radius);
        flame.scale.setScalar(22 + Math.random() * 26);
        flame.userData.baseY = flame.position.y;
        flame.userData.phase = Math.random() * 6;
        group.add(flame);
      }
    }
  } else if (spec.kind === 'light') {
    // Positioned groups: children sit at the local origin and the group is placed.
    entry.placed = true;
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: softTexture(), color, transparent: true, opacity: 0.75,
      depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    glow.scale.setScalar(Math.max(40, Math.min(220, (spec.intensity || 1) * 90)));
    group.add(glow);
    entry.markers.glow = glow;

    const light = new THREE.PointLight(color, (spec.intensity || 1) * 260, (spec.intensity || 1) * 1400, 1.8);
    group.add(light);
  } else if (spec.kind === 'threat') {
    entry.placed = true;
    const radius = spec.radius || 60;
    if (spec.key === 'swatter') {
      // A swatter is a disc on a handle. The disc is what looms.
      const disc = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, 8, 30),
        material('swatter', () => surface('#e8edf3', 0.4, 0.05)),
      );
      disc.rotation.z = Math.PI / 2;
      disc.castShadow = true;
      group.add(disc);
      const handle = new THREE.Mesh(
        new THREE.CylinderGeometry(6, 6, radius * 3.2, 12),
        material('handle', () => surface('#8c3f2a', 0.65, 0.1)),
      );
      handle.position.set(0, radius * 1.7, 0);
      handle.castShadow = true;
      group.add(handle);
    } else if (spec.key === 'hand') {
      const palm = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 24, 18),
        material('hand', () => surface('#e8b892', 0.72, 0.02)),
      );
      palm.scale.set(1.0, 0.42, 1.25);
      palm.castShadow = true;
      group.add(palm);
      for (let index = 0; index < 4; index++) {
        const finger = new THREE.Mesh(
          new THREE.CapsuleGeometry(radius * 0.17, radius * 0.8, 6, 10),
          material('hand', () => surface('#e8b892', 0.72, 0.02)),
        );
        finger.position.set((index - 1.5) * radius * 0.4, -radius * 0.1, radius * 0.75);
        finger.rotation.x = Math.PI / 2;
        finger.castShadow = true;
        group.add(finger);
      }
    } else {
      const body = new THREE.Mesh(
        new THREE.SphereGeometry(radius * 0.7, 20, 14),
        material('predator', () => surface('#5b1f22', 0.6, 0.15)),
      );
      body.scale.set(1.3, 0.8, 1.0);
      body.castShadow = true;
      group.add(body);
      const aura = new THREE.Sprite(new THREE.SpriteMaterial({
        map: softTexture(), color: 0xef4444, transparent: true, opacity: 0.32,
        depthWrite: false, blending: THREE.AdditiveBlending,
      }));
      aura.scale.setScalar(radius * 4);
      group.add(aura);
      entry.markers.aura = aura;
    }
  }

  updateStimulus(entry, spec, environment);
  return entry;
}

/** Reposition a stimulus and refresh the parts that depend on live state. */
export function updateStimulus(entry, spec, environment) {
  const { group, markers } = entry;

  // Two placement conventions, chosen by what has to move independently:
  //   odour and thermal children carry absolute world coordinates (a plume puff
  //   drifts away from its source, so the source cannot own their transform),
  //   while lights and threats are rigid bodies and simply move the group.
  if (entry.placed) {
    group.position.set(spec.x, spec.y, spec.z);
  }

  for (const key of Object.keys(markers)) {
    const marker = markers[key];
    if (key === 'source') marker.position.set(spec.x, spec.y, spec.z);
    if (key === 'patch') marker.position.set(spec.x, 1.2, spec.z);
  }

  if (spec.kind === 'odor') {
    const puffs = spec.puffs || [];
    for (let index = 0; index < entry.plumes.length; index++) {
      const sprite = entry.plumes[index];
      const puff = puffs[index];
      if (!puff) { sprite.visible = false; continue; }
      sprite.visible = true;
      sprite.position.set(puff.x, puff.y, puff.z);
      // A Gaussian of sigma reads as roughly 1.4 sigma of visible radius.
      sprite.scale.setScalar(puff.sigma * 2.8);
      sprite.material.opacity = Math.min(0.42, puff.strength * 0.30);
    }
  }
}

export function disposeGroup(group) {
  group.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const item of materials) {
        // Cached materials are shared with other groups, so disposing one here
        // would blank out live geometry elsewhere in the scene.
        if (CACHED.has(item)) continue;
        if (item.map && item.map.dispose) item.map.dispose();
        item.dispose();
      }
    }
  });
  group.clear();
}
