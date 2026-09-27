/**
 * Procedural Drosophila melanogaster.
 *
 * Built entirely from code — no mesh files, no build step. Body length is 3.2 mm,
 * which is the real size, so the camera genuinely has to zoom to see the fly.
 *
 * Everything visible is a pure function of kernel state: the kernel publishes
 * ``gaitPhase``, ``wingPhase``, ``proboscis``, ``antennaSwipe``, ``groomVariant``,
 * ``roll``, ``pitch`` and ``speed``, and this module maps those onto joint
 * rotations. There is no animation timeline and no keyframing, so an animation can
 * never disagree with what the simulation is actually doing.
 *
 * The wing beat is 218 Hz. At 60 fps that is wildly above Nyquist, so the wings
 * strobe — exactly as they do on real high-speed footage. Rendering two faint
 * ghosts at offset phases turns that aliasing into readable motion blur instead of
 * a flicker.
 */

import * as THREE from '../vendor/three.module.js';

const BODY_LENGTH = 3.2;         // mm
const WING_LENGTH = 2.15;
const LEG_SEGMENTS = [0.42, 0.46, 0.30];   // coxa+femur, tibia, tarsus
const TRIPOD_A = ['L1', 'R2', 'L3'];
const TRIPOD_B = ['R1', 'L2', 'R3'];

const LEG_ANCHORS = {
  L1: [-0.30, -0.05, 0.30], R1: [0.30, -0.05, 0.30],
  L2: [-0.34, -0.05, 0.02], R2: [0.34, -0.05, 0.02],
  L3: [-0.30, -0.05, -0.28], R3: [0.30, -0.05, -0.28],
};

/** Body-surface textures generated on a canvas: cheaper and more honest than a file. */
function makeEyeTexture(size = 128) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1a0a06';
  ctx.fillRect(0, 0, size, size);
  const radius = 5.2;
  const dx = radius * 1.732;
  for (let row = -1; row * radius * 1.5 < size + radius; row++) {
    for (let col = -1; col * dx < size + dx; col++) {
      const x = col * dx + (row % 2 ? dx / 2 : 0);
      const y = row * radius * 1.5;
      const gradient = ctx.createRadialGradient(x, y, 0.5, x, y, radius * 0.95);
      gradient.addColorStop(0, '#8a3b26');
      gradient.addColorStop(0.72, '#5c2114');
      gradient.addColorStop(1, '#210a06');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius * 0.92, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 1);
  return texture;
}

function makeWingTexture(size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size * 0.44;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Membrane: very faint, with a slight haze toward the tip.
  const haze = ctx.createLinearGradient(0, 0, canvas.width, 0);
  haze.addColorStop(0, 'rgba(216,228,240,0.30)');
  haze.addColorStop(0.6, 'rgba(200,214,232,0.20)');
  haze.addColorStop(1, 'rgba(186,204,226,0.30)');
  ctx.fillStyle = haze;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Veins. Longitudinal first, then a couple of cross veins, which is enough to
  // read as a wing at any distance the app actually renders.
  ctx.strokeStyle = 'rgba(150,168,192,0.85)';
  ctx.lineWidth = 1.5;
  const veins = [0.30, 0.44, 0.58, 0.72];
  for (const v of veins) {
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.06, canvas.height * v);
    ctx.bezierCurveTo(
      canvas.width * 0.35, canvas.height * (v - 0.06),
      canvas.width * 0.72, canvas.height * (v + 0.03),
      canvas.width * 0.94, canvas.height * (v - 0.02),
    );
    ctx.stroke();
  }
  ctx.lineWidth = 1.0;
  for (const x of [0.30, 0.44, 0.62]) {
    ctx.beginPath();
    ctx.moveTo(canvas.width * x, canvas.height * 0.24);
    ctx.lineTo(canvas.width * (x + 0.05), canvas.height * 0.78);
    ctx.stroke();
  }
  // Thickened anterior margin, as on a real wing.
  ctx.strokeStyle = 'rgba(170,186,208,0.95)';
  ctx.lineWidth = 3.0;
  ctx.beginPath();
  ctx.moveTo(canvas.width * 0.04, canvas.height * 0.24);
  ctx.bezierCurveTo(canvas.width * 0.4, canvas.height * 0.06, canvas.width * 0.8, canvas.height * 0.14, canvas.width * 0.96, canvas.height * 0.30);
  ctx.stroke();

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

/** A fly wing outline as a shape, so the silhouette is right rather than a quad. */
function wingGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(0.12, 0.20, 0.45, 0.26, 0.72, 0.15);
  shape.bezierCurveTo(0.92, 0.07, 1.0, 0.02, 1.02, -0.06);
  shape.bezierCurveTo(0.86, -0.20, 0.52, -0.30, 0.26, -0.24);
  shape.bezierCurveTo(0.12, -0.20, 0.02, -0.10, 0, 0);
  const geometry = new THREE.ShapeGeometry(shape, 22);
  geometry.scale(WING_LENGTH, WING_LENGTH * 0.92, 1);
  // Give the wing a slight camber so it catches specular light.
  const position = geometry.attributes.position;
  for (let index = 0; index < position.count; index++) {
    const x = position.getX(index);
    const y = position.getY(index);
    position.setZ(index, Math.sin((x / WING_LENGTH) * Math.PI) * 0.04 + Math.abs(y) * 0.03);
  }
  geometry.computeVertexNormals();
  return geometry;
}

function legSegment(radius, length) {
  const geometry = new THREE.CylinderGeometry(radius * 0.72, radius, length, 7, 1);
  geometry.translate(0, length / 2, 0);
  geometry.rotateX(Math.PI / 2);
  return geometry;
}

function buildLeg(name, materials) {
  const root = new THREE.Group();
  const [coxa, femur, tarsus] = LEG_SEGMENTS;

  const upper = new THREE.Mesh(legSegment(0.032, coxa + femur), materials.leg);
  const mid = new THREE.Mesh(legSegment(0.026, femur * 0.85), materials.leg);
  mid.position.z = coxa + femur;
  const lower = new THREE.Mesh(legSegment(0.020, tarsus), materials.leg);
  lower.position.z = coxa + femur + femur * 0.7;

  root.add(upper, mid, lower);
  root.userData = { upper, mid, lower, name };
  return root;
}

export function buildFly(id) {
  const eyeTexture = makeEyeTexture();
  const wingTexture = makeWingTexture();

  const materials = {
    body: new THREE.MeshStandardMaterial({ color: 0x2b2f38, roughness: 0.55, metalness: 0.32 }),
    thorax: new THREE.MeshStandardMaterial({ color: 0x39404b, roughness: 0.42, metalness: 0.44 }),
    abdomen: new THREE.MeshStandardMaterial({ color: 0x262a32, roughness: 0.5, metalness: 0.34 }),
    leg: new THREE.MeshStandardMaterial({ color: 0x1e2128, roughness: 0.6, metalness: 0.3 }),
    bristle: new THREE.MeshStandardMaterial({ color: 0x11141a, roughness: 0.8, metalness: 0.1 }),
    eye: new THREE.MeshPhysicalMaterial({
      color: 0x6d2416,
      roughness: 0.24,
      metalness: 0.1,
      iridescence: 0.85,
      iridescenceIOR: 1.6,
      map: eyeTexture,
      emissive: 0x3a0f06,
      emissiveIntensity: 0.35,
    }),
    wing: new THREE.MeshPhysicalMaterial({
      color: 0xd6e2f0,
      transparent: true,
      opacity: 0.34,
      roughness: 0.16,
      metalness: 0.0,
      side: THREE.DoubleSide,
      map: wingTexture,
      transmission: 0.35,
      depthWrite: false,
    }),
    wingGhost: new THREE.MeshBasicMaterial({
      color: 0xa9bdd6,
      transparent: true,
      opacity: 0.09,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  };

  const root = new THREE.Group();
  root.name = `fly:${id}`;
  const body = new THREE.Group();
  root.add(body);

  // ── head ────────────────────────────────────────────────────────────────
  const head = new THREE.Group();
  head.position.set(0, 0.02, 0.86);
  const headShell = new THREE.Mesh(new THREE.SphereGeometry(0.24, 20, 14), materials.body);
  headShell.scale.set(1.0, 0.86, 0.82);
  head.add(headShell);

  const eyeGeometry = new THREE.SphereGeometry(0.185, 18, 14);
  const eyes = [];
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(eyeGeometry, materials.eye);
    eye.position.set(side * 0.145, 0.045, 0.045);
    eye.scale.set(0.86, 1.0, 0.94);
    head.add(eye);
    eyes.push(eye);
  }

  // Three ocelli between the eyes, the classic Drosophila arrangement.
  for (const [ox, oy] of [[-0.06, 0.015], [0.06, 0.015], [0.0, 0.075]]) {
    const ocellus = new THREE.Mesh(new THREE.SphereGeometry(0.033, 10, 8), materials.eye);
    ocellus.position.set(ox, 0.24, -0.02);
    ocellus.position.y += oy * 0.2;
    head.add(ocellus);
  }

  // ── antennae, each with an arista ───────────────────────────────────────
  const antennae = [];
  for (const side of [-1, 1]) {
    const antenna = new THREE.Group();
    antenna.position.set(side * 0.09, 0.10, 0.20);
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.012, 0.16, 6), materials.bristle);
    shaft.rotation.x = Math.PI / 2.3;
    shaft.position.z = 0.06;
    const arista = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.003, 0.20, 5), materials.bristle);
    arista.rotation.set(-0.5, 0, side * 0.4);
    arista.position.set(side * 0.02, 0.01, 0.14);
    antenna.add(shaft, arista);
    head.add(antenna);
    antennae.push(antenna);
  }

  // ── proboscis (extensible) ──────────────────────────────────────────────
  const proboscis = new THREE.Group();
  proboscis.position.set(0, -0.13, 0.10);
  const proboscisBase = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.062, 0.14, 10), materials.body);
  proboscisBase.rotation.x = Math.PI / 2;
  proboscisBase.position.z = 0.06;
  const labellum = new THREE.Mesh(new THREE.SphereGeometry(0.085, 12, 9), materials.body);
  labellum.scale.set(1.0, 0.7, 0.7);
  labellum.position.z = 0.15;
  proboscis.add(proboscisBase, labellum);
  head.add(proboscis);
  body.add(head);

  // ── thorax ──────────────────────────────────────────────────────────────
  const thorax = new THREE.Group();
  const thoraxShell = new THREE.Mesh(new THREE.SphereGeometry(0.4, 22, 16), materials.thorax);
  thoraxShell.scale.set(0.86, 0.9, 1.24);
  thoraxShell.position.set(0, 0.05, 0.28);
  thorax.add(thoraxShell);

  // Dorsal bristles: a scattering of thin cones. Cheap, and instantly reads as
  // "bristly insect" rather than "smooth toy".
  const bristleGeometry = new THREE.ConeGeometry(0.011, 0.10, 4);
  for (let index = 0; index < 26; index++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * 0.9;
    const bristle = new THREE.Mesh(bristleGeometry, materials.bristle);
    bristle.position.set(
      Math.cos(theta) * 0.28 * Math.sin(phi),
      0.05 + Math.cos(phi) * 0.34,
      0.28 + Math.sin(theta) * 0.28 * Math.sin(phi) * 1.2,
    );
    bristle.rotation.set(Math.cos(theta) * 0.7, 0, -Math.sin(theta) * 0.7);
    thorax.add(bristle);
  }

  // Halteres: the modified hindwings, which counter-rotate with the wings.
  const halteres = [];
  for (const side of [-1, 1]) {
    const haltere = new THREE.Group();
    haltere.position.set(side * 0.22, 0.0, -0.06);
    const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.010, 0.010, 0.18, 5), materials.bristle);
    stalk.rotation.z = side * 0.9;
    stalk.position.x = side * 0.08;
    const knob = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), materials.body);
    knob.position.x = side * 0.17;
    haltere.add(stalk, knob);
    thorax.add(haltere);
    halteres.push(haltere);
  }
  body.add(thorax);

  // ── abdomen, as five tapered segments ───────────────────────────────────
  const abdomen = new THREE.Group();
  const abdomenSegments = [];
  for (let index = 0; index < 5; index++) {
    const t = index / 4;
    const segment = new THREE.Mesh(
      new THREE.SphereGeometry(0.30 * (1 - t * 0.42), 16, 11),
      materials.abdomen,
    );
    segment.scale.set(0.95 - t * 0.12, 0.8 - t * 0.14, 0.72);
    segment.position.set(0, 0.02 - t * 0.03, -0.28 - index * 0.24);
    abdomen.add(segment);
    abdomenSegments.push(segment);
  }
  body.add(abdomen);

  // ── wings ───────────────────────────────────────────────────────────────
  const wings = [];
  for (const side of [-1, 1]) {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.22, 0.24, 0.14);

    const wing = new THREE.Mesh(wingGeometry(), materials.wing);
    wing.rotation.z = side * -0.18;
    if (side < 0) wing.scale.x = -1;
    pivot.add(wing);

    // Motion-blur ghosts for the sub-frame wing positions.
    const ghosts = [];
    for (let index = 0; index < 2; index++) {
      const ghost = new THREE.Mesh(wingGeometry(), materials.wingGhost);
      ghost.rotation.z = side * -0.18;
      if (side < 0) ghost.scale.x = -1;
      pivot.add(ghost);
      ghosts.push(ghost);
    }

    body.add(pivot);
    wings.push({ pivot, wing, ghosts, side });
  }

  // ── legs ────────────────────────────────────────────────────────────────
  const legs = {};
  for (const [name, anchor] of Object.entries(LEG_ANCHORS)) {
    const leg = buildLeg(name, materials);
    leg.position.set(anchor[0], anchor[1], anchor[2]);
    leg.rotation.z = Math.sign(anchor[0]) * -0.5;
    body.add(leg);
    legs[name] = leg;
  }

  const state = {
    mode: 'REST',
    groomVariant: 'antenna',
    gait: 0,
    wingPhase: 0,
    wingAmp: 0,
    proboscis: 0,
    antennaSwipe: 0,
    flip: 0,
    speed: 0,
  };

  root.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = false;
    }
  });
  // Wings are translucent: casting shadows from them produces a solid black wedge.
  for (const { pivot } of wings) pivot.traverse((child) => { if (child.isMesh) child.castShadow = false; });

  const model = {
    root,
    body,
    head,
    thorax,
    wings,
    legs,
    antennae,
    halteres,
    proboscis,
    abdomenSegments,
    eyes,
    state,

    apply(snapshot) {
      Object.assign(state, {
        mode: snapshot.mode,
        groomVariant: snapshot.groomVariant,
        gait: snapshot.gaitPhase,
        wingPhase: snapshot.wingPhase,
        wingAmp: snapshot.wingAmp,
        proboscis: snapshot.proboscis,
        antennaSwipe: snapshot.antennaSwipe,
        flip: snapshot.flip,
        speed: snapshot.speed,
      });
      root.position.set(snapshot.position[0], snapshot.position[1], snapshot.position[2]);
      root.rotation.set(0, snapshot.heading, 0);
      body.rotation.set(snapshot.pitch, 0, snapshot.roll + state.flip * Math.PI * 1.6);
      body.position.y = state.mode === 'FLIGHT' || state.mode === 'ESCAPE' ? 0.3 : 0.0;
    },

    animate(dt, time) {
      const flying = state.mode === 'FLIGHT' || state.mode === 'ESCAPE';
      const escape = state.mode === 'ESCAPE';

      // ── wings ───────────────────────────────────────────────────────────
      const amplitude = state.wingAmp;
      for (const { pivot, wing, ghosts, side } of wings) {
        const angle = Math.sin(state.wingPhase * Math.PI * 2) * (0.15 + 0.85 * amplitude);
        pivot.rotation.z = side * (0.28 * amplitude + angle * 1.15);
        pivot.rotation.y = Math.cos(state.wingPhase * Math.PI * 2) * 0.28 * amplitude;
        // Slight twist so the wing looks like it is generating force.
        wing.rotation.x = Math.sin(state.wingPhase * Math.PI * 2 + 0.6) * 0.22 * amplitude;

        // A ghost must be the *same* mesh lagged in time, not a second mesh at a
        // different angle: the delta between this frame's sweep and the lagged
        // frame's is added on top of the pivot's rotation, so the ghosts trail the
        // real wing exactly as a shutter would have captured them.
        const ghostAlpha = amplitude * 0.9;
        for (let index = 0; index < ghosts.length; index++) {
          const lead = (index + 1) * 0.12;
          const lagged = Math.sin((state.wingPhase - lead) * Math.PI * 2) * (0.15 + 0.85 * amplitude);
          ghosts[index].rotation.z = side * -0.18 + side * (lagged - angle) * 1.15;
          ghosts[index].rotation.x = Math.sin((state.wingPhase - lead) * Math.PI * 2 + 0.6) * 0.22 * amplitude;
          ghosts[index].material.opacity = 0.10 * ghostAlpha;
          ghosts[index].visible = ghostAlpha > 0.05;
        }
      }

      // ── halteres: counter-rotate against the wings ──────────────────────
      for (let index = 0; index < halteres.length; index++) {
        const side = index === 0 ? -1 : 1;
        halteres[index].rotation.z = -Math.sin(state.wingPhase * Math.PI * 2) * 0.6 * amplitude * side;
      }

      // ── legs ────────────────────────────────────────────────────────────
      const walking = state.mode === 'WALK' && !flying;
      const groomPhase = time * 3.4;
      for (const [name, leg] of Object.entries(legs)) {
        const { upper, mid, lower } = leg.userData;
        const trippedA = TRIPOD_A.includes(name);
        const phase = (state.gait + (trippedA ? 0 : 0.5)) * Math.PI * 2;
        const lift = Math.max(0, Math.sin(phase)) * (walking ? 0.42 : 0);
        const swing = Math.cos(phase) * (walking ? 0.30 : 0);

        let target = { upper: 0.0, mid: -0.55, lower: -0.35 };

        if (walking) {
          target = {
            upper: -swing * 0.9 + lift * 0.4,
            mid: -0.55 - lift * 0.7,
            lower: -0.35 + lift * 0.5,
          };
        } else if (flying) {
          // Legs tuck against the body in flight.
          target = { upper: -0.5, mid: -1.35, lower: -0.9 };
        } else if (state.mode === 'FEED') {
          target = { upper: 0.1, mid: -0.7, lower: -0.5 };
        }

        // Grooming overrides the front legs only, and only on the relevant variant.
        if (state.mode === 'GROOM') {
          const isFront = name === 'L1' || name === 'R1';
          const isHind = name === 'L3' || name === 'R3';
          if (state.groomVariant === 'antenna' || state.groomVariant === 'front_legs') {
            if (isFront) {
              const reach = 0.5 + 0.5 * Math.sin(groomPhase * 2.0);
              target = {
                upper: -1.25 - reach * 0.35,
                mid: -1.5 + reach * 0.5,
                lower: -1.05 + reach * 0.55,
              };
            }
          } else if (state.groomVariant === 'wings') {
            if (name === 'L3' || name === 'R3') {
              const reach = 0.5 + 0.5 * Math.sin(groomPhase * 1.7);
              target = { upper: -0.9 - reach * 0.6, mid: -1.2 + reach * 0.4, lower: -0.6 };
            }
          } else if (state.groomVariant === 'abdomen') {
            if (isHind) {
              const reach = 0.5 + 0.5 * Math.sin(groomPhase * 1.4);
              target = { upper: -0.4 + reach * 0.5, mid: -1.5 + reach * 0.7, lower: -0.8 };
            }
          }
        }

        const blend = Math.min(1, 11 * dt);
        upper.rotation.x += (target.upper - upper.rotation.x) * blend;
        mid.rotation.x += (target.mid - mid.rotation.x) * blend;
        lower.rotation.x += (target.lower - lower.rotation.x) * blend;
      }

      // ── antennae and proboscis ──────────────────────────────────────────
      const swipe = state.antennaSwipe;
      for (let index = 0; index < antennae.length; index++) {
        const side = index === 0 ? -1 : 1;
        antennae[index].rotation.x = -0.25 - swipe * 0.9 + Math.sin(time * 2.3 + index) * 0.05;
        antennae[index].rotation.z = side * (0.25 + swipe * 0.5);
        if (state.mode === 'GROOM' && (state.groomVariant === 'antenna' || state.groomVariant === 'front_legs')) {
          antennae[index].rotation.y = Math.sin(groomPhase * 2.4 + index * 1.4) * 0.4 * swipe;
        }
      }
      proboscis.scale.set(1.0, 1.0, 0.45 + state.proboscis * 1.55);
      proboscis.rotation.x = 0.15 + state.proboscis * 0.35;

      // ── abdomen: breathing, and a curl during escape ────────────────────
      const curl = escape ? Math.sin(time * 22.0) * 0.08 : 0;
      for (let index = 0; index < abdomenSegments.length; index++) {
        const t = index / Math.max(1, abdomenSegments.length - 1);
        abdomenSegments[index].rotation.x = -0.06 - t * 0.06 + curl * t;
      }

      // ── root motion touches ─────────────────────────────────────────────
      if (state.mode === 'REST' || state.mode === 'SLEEP' || state.mode === 'GROOM') {
        // Idle: a slow breathing bob, plus body sway while grooming.
        const sway = state.mode === 'GROOM' ? 0.05 : 0.012;
        body.position.y += Math.sin(time * 1.7) * sway * 0.02;
        body.rotation.z += Math.sin(time * 1.1) * (state.mode === 'GROOM' ? 0.06 : 0.012);
      }

    },
  };

  model.animate(1 / 60, 0);
  return model;
}
