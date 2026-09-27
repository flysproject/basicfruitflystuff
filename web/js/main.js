/**
 * Entry point: wires the backend, the viewport, the brain HUD and the interface
 * together and runs the frame loop.
 *
 * The loop is deliberately thin. Physics runs on the Python side at 120 Hz; this
 * only interpolates, draws, and pushes the newest state into the panels. Nothing
 * here decides anything about the fly.
 */

import * as THREE from '../vendor/three.module.js';
import { backend } from './net.js';
import { Viewport, CAMERA_MODES, FrameClock } from './scene.js';
import { BrainView } from './brain.js';
import { UI } from './ui.js';

const bootStatus = document.getElementById('boot-status');
const bootFill = document.getElementById('boot-fill');
const bootLog = document.getElementById('boot-log');

function bootSay(message, warn = false) {
  const item = document.createElement('li');
  item.textContent = message;
  if (warn) item.className = 'warn';
  bootLog.appendChild(item);
  bootLog.scrollTop = bootLog.scrollHeight;
  bootStatus.textContent = message;
}

function bootProgress(fraction) {
  bootFill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}

async function main() {
  bootProgress(0.05);
  bootSay('Requesting the bootstrap manifest…');
  const manifest = await backend.bootstrap();
  bootProgress(0.2);
  bootSay(`Atlas: ${manifest.atlas.sourceShape.join('×')} source → ${manifest.atlas.shape.join('×')} texture`);
  if (manifest.atlas.synthetic) {
    bootSay('No VFB volumes found — running on a synthetic CNS stand-in', true);
  }
  bootSay(`Regions: ${manifest.atlas.regions.length} neuropils · ${manifest.environments.length} environments · ${manifest.stimuli.length} stimuli`);

  // ── viewport ──────────────────────────────────────────────────────────
  bootProgress(0.35);
  const canvas = document.getElementById('gl');
  const viewport = new Viewport(canvas, manifest);
  const showError = (error) => {
    console.error(error);
    bootSay(String(error.message || error), true);
  };
  // Declared before the listener below so a resize during boot cannot trip the
  // temporal dead zone; the HUD is only resized once it exists.
  let brain = null;
  window.addEventListener('resize', () => {
    viewport.resize();
    if (brain) brain.resize();
  });
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    viewport.isDisposed = true;
    showError(new Error('WebGL context lost — reload the page to continue'));
  });
  viewport.resize();

  // ── brain HUD ─────────────────────────────────────────────────────────
  bootProgress(0.5);
  brain = new BrainView(document.getElementById('brain'), manifest);
  try {
    const description = await brain.load();
    document.getElementById('brain-source').textContent = description;
    bootSay(`Brain volume uploaded (${description.split('·')[0].trim()})`);
  } catch (error) {
    console.warn(error);
    document.getElementById('brain-source').textContent = 'volume unavailable';
    bootSay('Brain volume could not be loaded; the HUD will stay dark', true);
  }
  brain.resize();

  // ── interface ─────────────────────────────────────────────────────────
  bootProgress(0.7);
  const environmentByKey = Object.fromEntries(manifest.environments.map((entry) => [entry.key, entry]));
  let currentEnvironment = null;
  let paused = false;

  /**
   * Build the room, but only when it actually changes.
   *
   * Rebuilding geometry is the most expensive thing the viewport does, so this is
   * driven from the kernel's own `world.key` rather than from button presses alone:
   * the scene stays correct after a preset swaps environment, and the boot room
   * appears on the first painted frame instead of waiting for a pill to be clicked.
   */
  function syncEnvironment(key) {
    if (!key || key === currentEnvironment) return;
    const spec = environmentByKey[key];
    if (!spec) return;
    viewport.setEnvironment(spec);
    ui._updateEnvironments(key);
    currentEnvironment = key;
  }

  const actions = {
    loadEnvironment(key) {
      // Swap locally for an immediate response, then let the kernel confirm.
      syncEnvironment(key);
      backend.cmd('load_environment', { key }).catch(() => {});
    },
    loadPreset(key) {
      backend.cmd('preset', { name: key }).catch(() => {});
      const preset = manifest.presets.find((entry) => entry.key === key);
      if (preset) {
        // The kernel owns the real environment load; mirror it immediately so the
        // scene swaps without waiting for the round trip.
        syncEnvironment(preset.environment);
        ui.toast(`${preset.label} — ${preset.watch}`, 'good');
      }
    },
    togglePause() {
      paused = !paused;
      backend.cmd('set_paused', { value: paused }).catch(() => {});
    },
    reset() {
      backend.cmd('reset', {}).catch(() => {});
    },
    setCamera(key) {
      viewport.setCameraMode(key);
      ui._updateCamera(key);
    },
    setROIs(visible) {
      viewport.setROIVisible(visible);
    },
    pair(kind) {
      backend.cmd('condition', { cs: 'peppermint', us: kind }).catch(() => {});
      ui.toast(kind === 'shock'
        ? 'Peppermint paired with a shock'
        : 'Peppermint paired with sugar', kind === 'shock' ? '' : 'good');
    },
  };

  const ui = new UI(manifest, backend, actions);
  ui.setCameraModes(CAMERA_MODES, actions.setCamera);
  ui._updateCamera('follow');

  // Open on the room the kernel is already running, so the first frame is furnished.
  syncEnvironment(manifest.environments[0] && manifest.environments[0].key);

  // Place a stimulus where the user clicks, if the toolbelt has one armed.
  const viewportEl = document.getElementById('viewport');
  const dropHint = document.getElementById('drop-hint');
  let pendingTool = null;
  // The interface owns arming; the scene only needs to know what is armed.
  ui.onToolChange = (key) => {
    pendingTool = key;
    if (!key) dropHint.hidden = true;
  };

  canvas.addEventListener('pointerdown', (event) => {
    if (!pendingTool || event.button !== 0) return;
    // Only treat a click as placement if the pointer did not drag the camera.
    const startX = event.clientX;
    const startY = event.clientY;
    const onUp = (upEvent) => {
      canvas.removeEventListener('pointerup', onUp);
      if (Math.hypot(upEvent.clientX - startX, upEvent.clientY - startY) > 5) return;
      const position = viewport.pickPosition(upEvent.clientX, upEvent.clientY);
      if (!position) return;
      const stimulus = manifest.stimuli.find((entry) => entry.key === pendingTool);
      const spawnHeight = stimulus && stimulus.kind === 'odor' ? position[1] + 18 : position[1] + 6;
      backend.cmd('add_stimulus', {
        key: pendingTool,
        x: position[0],
        y: Math.max(4, spawnHeight),
        z: position[2],
      }).catch(() => {});
      ui.setSelectedTool(null);
      dropHint.hidden = true;
    };
    canvas.addEventListener('pointerup', onUp);
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!pendingTool) return;
    dropHint.hidden = false;
    const rect = viewportEl.getBoundingClientRect();
    dropHint.style.left = `${event.clientX - rect.left}px`;
    dropHint.style.top = `${event.clientY - rect.top}px`;
    const stimulus = manifest.stimuli.find((entry) => entry.key === pendingTool);
    dropHint.textContent = `Place ${stimulus ? stimulus.label : ''} — Esc to cancel`;
  });
  canvas.addEventListener('pointerleave', () => { dropHint.hidden = true; });

  // ── stream wiring ─────────────────────────────────────────────────────
  bootProgress(0.85);
  const stats = {
    kernel: document.getElementById('stat-kernel'),
    step: document.getElementById('stat-step'),
    atlas: document.getElementById('stat-atlas'),
    conn: document.getElementById('stat-conn'),
    net: document.getElementById('stat-net'),
  };
  stats.atlas.textContent =
    `atlas ${manifest.atlas.shape.join('×')} · ${manifest.atlas.regions.length} ROIs · ` +
    `${manifest.atlas.synthetic ? 'synthetic' : 'VFB NRRD'}`;
  stats.conn.textContent =
    `connectome ${manifest.sources.connectome.name} ` +
    `(${manifest.sources.connectome.available ? 'loaded' : 'unavailable'})`;

  let lastSlowWorld = null;
  backend.on('state', (snapshot) => {
    if (snapshot.world) lastSlowWorld = snapshot.world;
    const merged = lastSlowWorld ? { ...snapshot, world: lastSlowWorld } : snapshot;
    if (merged.world && merged.world.key) syncEnvironment(merged.world.key);
    viewport.syncFlies(merged.flies || []);
    if (merged.world) viewport.syncStimuli(merged.world.stimuli || []);
    brain.setActivation(snapshot.regionActivation);
    ui.updateControls(merged, null);
    ui.updateAudio(snapshot);

    stats.kernel.textContent = `kernel ${snapshot.performance.hz.toFixed(0)} Hz · load ${(snapshot.performance.load * 100).toFixed(1)}%`;
    stats.step.textContent = `step ${snapshot.step} · t ${snapshot.t.toFixed(2)}s · ${snapshot.flyCount} fly`;
    paused = snapshot.paused;
    document.getElementById('btn-play').textContent = paused ? '▶' : '❚❚';
  });

  backend.on('events', (entries) => ui.addEvents(entries));
  backend.on('glance', (payload) => {
    ui.setGraphData(payload.graphs);
    // Memory is only refreshed at ~5 Hz, and it always rides the same snapshot the
    // 60 Hz channel just delivered. Guarding on the snapshot avoids feeding the
    // control panel a payload with no endocrine block during the first ticks.
    const latest = backend.stats().snapshot;
    if (payload.memory && latest) ui.updateControls({ ...latest, memory: payload.memory }, null);
  });

  backend.on('status', (status) => {
    stats.net.textContent = status.connected ? 'stream live' : `stream ${status.message || 'offline'}`;
    stats.net.className = `stat-net ${status.connected ? 'ok' : 'bad'}`;
  });

  backend.connect();

  // The very first state can take a moment; pull one snapshot so the scene is never
  // empty on the first painted frame.
  try {
    const first = await backend.snapshot();
    if (first.world) {
      syncEnvironment(first.world.key);
      viewport.syncStimuli(first.world.stimuli || []);
    }
    viewport.syncFlies(first.flies || []);
    ui.updateControls(first, null);
  } catch (error) {
    console.warn('first snapshot unavailable', error);
  }

  // ── frame loop ────────────────────────────────────────────────────────
  const clock = new FrameClock();
  let graphTimer = 0;

  function frame() {
    const dt = clock.getDelta();
    const snapshot = backend.interpolated(1);
    if (snapshot) viewport.render(dt, snapshot);
    brain.render(dt, clock.elapsedTime);
    graphTimer += dt;
    if (graphTimer > 0.2) {
      graphTimer = 0;
      ui.drawGraphs();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  bootProgress(1);
  bootSay('Ready.');
  window.setTimeout(() => {
    document.getElementById('boot').classList.add('done');
    ui.maybeShowGuide();
  }, 320);

  // Expose a small handle for debugging from the console.
  window.flybrain = { backend, viewport, brain, ui, manifest, THREE };
}

main().catch((error) => {
  console.error(error);
  bootStatus.textContent = `Startup failed: ${error.message || error}`;
  bootFill.style.background = '#f87171';
  bootFill.style.width = '100%';
  const item = document.createElement('li');
  item.className = 'warn';
  item.textContent = 'Check that the Python kernel is still running and reload.';
  bootLog.appendChild(item);
});
