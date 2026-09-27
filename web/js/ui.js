/**
 * The interface: panels, controls, graphs, log and the first-run guide.
 *
 * Built from the backend manifest rather than hard-coded, so the endocrine sliders,
 * the toolbelt, the mutation toggles and the environment pills all come from the
 * Python side's own definitions. Adding a neuromodulator to MODULATORS in neuro.py
 * makes it appear here with its real tooltip text — there is no second list to keep
 * in sync.
 *
 * Every control is optimistic: it updates locally and posts the command, so a slider
 * never waits on a round trip to feel like it moved.
 */

const TOOL_GLYPHS = {
  banana: '🍌', drop: '💧', flask: '⚗', leaf: '🌿', skull: '☠', spark: '✦',
  light: '◉', lamp: '⌾', sun: '☀', flame: '🔥', snow: '❄', swatter: '🏓',
  hand: '✋', paw: '🐾',
};

const LOG_CHANNELS = [
  { key: 'all', label: 'all' },
  { key: 'behaviour', label: 'behaviour' },
  { key: 'reflex', label: 'reflex' },
  { key: 'memory', label: 'memory' },
  { key: 'control', label: 'control' },
  { key: 'system', label: 'system' },
];

export class UI {
  constructor(manifest, backend, actions) {
    this.manifest = manifest;
    this.backend = backend;
    this.actions = actions;
    this.elements = {};
    this.logFilter = 'all';
    this.entries = [];
    this.graphKeys = manifest.series.filter((series) => series.default).map((series) => series.key);
    this.hiddenRegions = new Set();
    this.audio = null;
    this.audioOn = false;
    this.graphBuffer = null;
    this._buildSliders();
    this._buildToolbelt();
    this._buildMutations();
    this._buildEnvironments();
    this._buildLegends();
    this._bindChrome();
    this._bindGuide();
    this._bindTools();
  }

  $(id) {
    if (!this.elements[id]) this.elements[id] = document.getElementById(id);
    return this.elements[id];
  }

  // ── endocrine sliders ───────────────────────────────────────────────────
  _buildSliders() {
    const container = this.$('sliders');
    container.innerHTML = '';
    this.sliders = {};
    for (const spec of this.manifest.modulators) {
      const wrap = document.createElement('div');
      wrap.className = 'slider';
      wrap.innerHTML = `
        <div class="slider-head">
          <span class="slider-name">${spec.label}</span>
          <span class="slider-acr">${spec.short}</span>
          <span class="slider-val" style="margin-left:auto;color:${spec.color}">--</span>
        </div>
        <div class="slider-track">
          <span class="slider-phasic" style="background:${spec.color}"></span>
          <input type="range" min="0" max="100" step="1" value="${Math.round(spec.default * 100)}"
                 aria-label="${spec.label}" style="--thumb:${spec.color}" />
          <span class="slider-spike" style="background:${spec.color}"></span>
        </div>`;
      const input = wrap.querySelector('input');
      const value = wrap.querySelector('.slider-val');
      const phasic = wrap.querySelector('.slider-phasic');
      const spike = wrap.querySelector('.slider-spike');

      input.addEventListener('input', () => {
        const normalised = Number(input.value) / 100;
        value.textContent = `${input.value}%`;
        this.backend.cmd('set_slider', { key: spec.key, value: normalised }).catch(() => {});
      });
      // Show the biological role and the simulation effect on hover: this is the
      // tooltip the app is supposed to have, not a bare number.
      this._attachTip(wrap.querySelector('.slider-head'), spec.label, spec.blurb, spec.effect);
      container.appendChild(wrap);

      this.sliders[spec.key] = { spec, input, value, phasic, spike, lastSpike: 0 };
    }
  }

  _updateSliders(levels, tonic, phasic, spikes) {
    for (const [key, entry] of Object.entries(this.sliders)) {
      const level = levels[key] ?? 0;
      const base = tonic[key] ?? 0;
      entry.value.textContent = `${Math.round(level * 100)}%`;
      entry.value.style.color = entry.spec.color;
      // The phasic bar shows the transient riding on top of the tonic level.
      const transient = Math.max(0, level - base);
      entry.phasic.style.width = `${(base + transient) * 100}%`;
      entry.phasic.style.background = transient > 0.02
        ? `linear-gradient(90deg, ${entry.spec.color}, #ffb347)`
        : entry.spec.color;
      entry.phasic.style.opacity = 0.28;
      const spike = spikes[key] ?? 1;
      if (spike > 1.08 && performance.now() - entry.lastSpike > 120) {
        entry.lastSpike = performance.now();
        entry.spike.classList.add('on');
        window.setTimeout(() => entry.spike.classList.remove('on'), 240);
      }
      // Keep the slider in step when a preset or profile moves the target.
      if (document.activeElement !== entry.input) {
        const target = Math.round((tonic[key] ?? 0) * 100);
        if (Math.abs(Number(entry.input.value) - target) > 1) entry.input.value = String(target);
      }
    }
  }

  // ── toolbelt ────────────────────────────────────────────────────────────
  _buildToolbelt() {
    const container = this.$('toolbelt');
    container.innerHTML = '';
    this.tools = {};
    const groups = [
      { kind: 'odor', label: 'Odour' },
      { kind: 'light', label: 'Light' },
      { kind: 'thermal', label: 'Thermal' },
      { kind: 'threat', label: 'Threat' },
    ];
    for (const group of groups) {
      const items = this.manifest.stimuli.filter((stimulus) => stimulus.kind === group.kind);
      if (!items.length) continue;
      const row = document.createElement('div');
      row.className = 'tool-group';
      for (const stimulus of items) {
        const button = document.createElement('button');
        button.className = 'tool';
        button.type = 'button';
        button.dataset.key = stimulus.key;
        button.setAttribute('aria-pressed', 'false');
        button.innerHTML = `
          <span class="tool-glyph" style="color:${stimulus.color}">${TOOL_GLYPHS[stimulus.icon] || '◆'}</span>
          <span class="tool-name">${stimulus.label}</span>
          <span class="tool-swatch" style="background:${stimulus.color}"></span>`;
        this._attachTip(button, stimulus.label, stimulus.blurb, stimulus.effect);
        row.appendChild(button);
        this.tools[stimulus.key] = { stimulus, button };
      }
      container.appendChild(row);
    }
  }

  _bindTools() {
    for (const [key, entry] of Object.entries(this.tools)) {
      entry.button.addEventListener('click', () => {
        const already = entry.button.classList.contains('on');
        this.setSelectedTool(already ? null : key);
        if (!already) {
          this.toast(`${entry.stimulus.label} armed — click in the scene to place it`, 'good');
        }
      });
    }
    this.$('btn-log-clear').addEventListener('click', () => {
      this.entries = [];
      this._renderLog();
    });
  }

  /**
   * Arm or disarm a toolbelt stimulus. The host page passes an ``onToolChange``
   * callback so it can react to arming without re-wrapping this method.
   */
  setSelectedTool(key) {
    this.selectedKey = key;
    for (const [toolKey, entry] of Object.entries(this.tools)) {
      const on = toolKey === key;
      entry.button.classList.toggle('on', on);
      entry.button.setAttribute('aria-pressed', String(on));
    }
    document.getElementById('viewport').classList.toggle('placing', Boolean(key));
    if (typeof this.onToolChange === 'function') this.onToolChange(key);
  }

  // ── mutations ───────────────────────────────────────────────────────────
  _buildMutations() {
    const container = this.$('mutations');
    container.innerHTML = '';
    this.mutations = {};
    for (const mutation of this.manifest.mutations) {
      const row = document.createElement('label');
      row.className = 'switch';
      row.innerHTML = `
        <input type="checkbox" />
        <span class="knob"></span>
        <span>${mutation.label} <small>${mutation.gene || ''}</small></span>`;
      const input = row.querySelector('input');
      input.addEventListener('change', () => {
        this.backend.cmd('set_mutation', { key: mutation.key, value: input.checked }).catch(() => {});
        this.toast(`${mutation.label}: ${mutation.effect}`, input.checked ? 'good' : '');
      });
      this._attachTip(row, `${mutation.label} mutant`, mutation.blurb, mutation.effect);
      container.appendChild(row);
      this.mutations[mutation.key] = { mutation, input };
    }
  }

  _updateMutations(state) {
    for (const [key, entry] of Object.entries(this.mutations)) {
      const wanted = Boolean(state[key]);
      if (entry.input.checked !== wanted) entry.input.checked = wanted;
    }
  }

  // ── environments ────────────────────────────────────────────────────────
  _buildEnvironments() {
    const container = this.$('env-pills');
    container.innerHTML = '';
    for (const environment of this.manifest.environments) {
      const pill = document.createElement('button');
      pill.className = 'pill';
      pill.type = 'button';
      pill.textContent = environment.label;
      this._attachTip(pill, environment.label, environment.blurb, environment.notes);
      pill.addEventListener('click', () => this.actions.loadEnvironment(environment.key));
      container.appendChild(pill);
      this.environmentPills = this.environmentPills || {};
      this.environmentPills[environment.key] = pill;
    }
    this._updateEnvironments(this.manifest.environments[0].key);
  }

  _updateEnvironments(activeKey) {
    for (const [key, pill] of Object.entries(this.environmentPills || {})) {
      pill.classList.toggle('on', key === activeKey);
    }
  }

  // ── cameras ─────────────────────────────────────────────────────────────
  /** Camera modes live in the viewport, so their pills are built from there. */
  setCameraModes(modes, onSelect) {
    const container = this.$('camera-pills');
    container.innerHTML = '';
    for (const mode of modes) {
      const pill = document.createElement('button');
      pill.className = 'pill';
      pill.type = 'button';
      pill.textContent = mode.label;
      this._attachTip(pill, mode.label, mode.blurb, '');
      pill.addEventListener('click', () => onSelect(mode.key));
      container.appendChild(pill);
      (this.cameraPills ||= {})[mode.key] = pill;
    }
  }

  _updateCamera(activeKey) {
    for (const [key, pill] of Object.entries(this.cameraPills || {})) {
      pill.classList.toggle('on', key === activeKey);
    }
  }

  // ── legends ─────────────────────────────────────────────────────────────
  _buildLegends() {
    const brain = this.$('brain-legend');
    brain.innerHTML = '';
    this.regionChips = {};
    for (const region of this.manifest.atlas.regions) {
      const item = document.createElement('button');
      item.className = 'legend-item on';
      item.type = 'button';
      item.innerHTML = `<i class="legend-swatch" style="background:rgb(${region.color.join(',')})"></i>${region.key}`;
      const coverage = region.tissueOverlap && region.tissueOverlap.onTissuePct;
      this._attachTip(
        item,
        region.label,
        region.blurb,
        coverage != null ? `ROI placement check: ${coverage}% of this region sits on real tissue.` : '',
      );
      item.addEventListener('click', () => {
        const off = item.classList.toggle('off');
        item.classList.toggle('on', !off);
        if (off) this.hiddenRegions.add(region.key); else this.hiddenRegions.delete(region.key);
      });
      brain.appendChild(item);
    }

    const graphs = this.$('graph-legend');
    graphs.innerHTML = '';
    this.seriesChips = {};
    for (const series of this.manifest.series) {
      const item = document.createElement('button');
      item.className = `legend-item${this.graphKeys.includes(series.key) ? ' on' : ''}`;
      item.type = 'button';
      item.innerHTML = `<i class="legend-swatch" style="background:${series.color}"></i>${series.label}`;
      item.addEventListener('click', () => {
        const on = !item.classList.contains('on');
        item.classList.toggle('on', on);
        if (on) this.graphKeys.push(series.key);
        else this.graphKeys = this.graphKeys.filter((key) => key !== series.key);
      });
      graphs.appendChild(item);
      this.seriesChips[series.key] = item;
    }
  }

  // ── chrome bindings ─────────────────────────────────────────────────────
  _bindChrome() {
    this.$('btn-play').addEventListener('click', () => this.actions.togglePause());
    this.$('btn-reset').addEventListener('click', () => {
      this.actions.reset();
      this.entries = [];
      this._renderLog();
      this.toast('Session reset');
    });
    this.$('time-scale').addEventListener('input', (event) => {
      const value = Number(event.target.value);
      this.$('time-scale-out').textContent = value.toFixed(1);
      this.backend.cmd('set_time_scale', { value }).catch(() => {});
    });
    this.$('fly-count').addEventListener('input', (event) => {
      const value = Number(event.target.value);
      this.$('fly-count-out').textContent = String(value);
      this.backend.cmd('set_fly_count', { value }).catch(() => {});
      if (value > 1) this.toast(`Swarming mode: ${value} flies competing for food`, 'good');
    });
    this.$('day-length').addEventListener('input', (event) => {
      const value = Number(event.target.value);
      this.$('day-length-out').textContent = `${value}s`;
      this.backend.cmd('set_day_length', { value }).catch(() => {});
    });

    this.$('btn-presets').addEventListener('click', (event) => {
      event.stopPropagation();
      this._toggleMenu('menu-presets');
    });
    this.$('btn-export').addEventListener('click', (event) => {
      event.stopPropagation();
      this._toggleMenu('menu-export');
    });
    document.addEventListener('click', () => this._closeMenus());

    this._buildPresetMenu();
    this._buildExportMenu();

    this.$('btn-audio').addEventListener('click', () => this.toggleAudio());
    this.$('btn-rois').addEventListener('click', () => {
      const button = this.$('btn-rois');
      const on = button.getAttribute('aria-pressed') !== 'true';
      button.setAttribute('aria-pressed', String(on));
      button.textContent = `ROI overlay: ${on ? 'on' : 'off'}`;
      this.actions.setROIs(on);
    });

    this.$('btn-pair-shock').addEventListener('click', () => this.actions.pair('shock'));
    this.$('btn-pair-sugar').addEventListener('click', () => this.actions.pair('sugar'));
    this.$('btn-forget').addEventListener('click', () => {
      this.backend.cmd('forget', {}).catch(() => {});
      this.toast('All learned odour values cleared');
    });
    this.$('chk-auto-condition').addEventListener('change', (event) => {
      this.backend.cmd('set_conditioning', { enabled: event.target.checked }).catch(() => {});
      this.toast(event.target.checked
        ? 'Auto-conditioning on: real shocks and real meals now write memory'
        : 'Auto-conditioning off');
    });

    this.$('btn-profile-save').addEventListener('click', () => {
      this.backend.cmd('export_profile', { name: `profile-${Date.now()}` }).catch(() => {});
      this.toast('Endocrine profile saved to profiles/', 'good');
    });
    this.$('profile-file').addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const payload = JSON.parse(String(reader.result));
          for (const [key, value] of Object.entries(payload)) {
            if (this.sliders[key]) {
              this.sliders[key].input.value = String(Math.round(Number(value) * 100));
              this.backend.cmd('set_slider', { key, value: Number(value) }).catch(() => {});
            }
          }
          this.toast(`Loaded profile with ${Object.keys(payload).length} modulators`, 'good');
        } catch (error) {
          this.toast('That file is not a valid endocrine profile', 'bad');
        }
      };
      reader.readAsText(file);
    });

    // Collapsible rails: the layout has to survive a small window gracefully.
    for (const button of document.querySelectorAll('.rail-collapse')) {
      button.addEventListener('click', () => {
        const rail = document.getElementById(button.dataset.target);
        const layout = document.getElementById('layout');
        rail.classList.toggle('collapsed');
        const collapsed = rail.classList.contains('collapsed');
        if (button.dataset.target === 'rail-left') layout.classList.toggle('no-left', collapsed);
        else layout.classList.toggle('no-right', collapsed);
        button.textContent = collapsed
          ? (button.dataset.target === 'rail-left' ? '›' : '‹')
          : (button.dataset.target === 'rail-left' ? '‹' : '›');
      });
    }
    for (const toggle of document.querySelectorAll('.hud-toggle')) {
      toggle.addEventListener('click', () => {
        const body = document.getElementById(toggle.dataset.collapse);
        const hidden = body.style.display === 'none';
        body.style.display = hidden ? '' : 'none';
        toggle.textContent = hidden ? '–' : '+';
      });
    }

    // Keyboard shortcuts: the ones a user reaches for without being told.
    document.addEventListener('keydown', (event) => {
      if (event.target.closest('input, textarea, select, [contenteditable]')) return;
      if (event.code === 'Space') { event.preventDefault(); this.actions.togglePause(); }
      if (event.key === 'h' || event.key === 'H') this.openGuide();
      if (event.key === 'e' || event.key === 'E') this._toggleMenu('menu-presets');
      if (event.key === 'Escape') this.setSelectedTool(null);
      if (event.key === '1') this.actions.setCamera('follow');
      if (event.key === '2') this.actions.setCamera('tactical');
      if (event.key === '3') this.actions.setCamera('flyvision');
    });
  }

  _buildPresetMenu() {
    const menu = this.$('menu-presets');
    menu.innerHTML = '';
    for (const preset of this.manifest.presets) {
      const button = document.createElement('button');
      button.type = 'button';
      button.innerHTML = `
        <strong>${preset.label}</strong>
        <span>${preset.blurb}</span>
        <span class="menu-watch">Watch for: ${preset.watch}</span>`;
      button.addEventListener('click', () => {
        this.actions.loadPreset(preset.key);
        this._closeMenus();
      });
      menu.appendChild(button);
    }
  }

  _buildExportMenu() {
    const menu = this.$('menu-export');
    menu.innerHTML = '';
    const items = [
      ['Telemetry CSV', 'One row per 20 Hz sample, ready for pandas', '/api/export/csv'],
      ['Session JSON', 'Full log, states, memory and atlas metadata', '/api/export/json'],
      ['Endocrine profile', 'Just the modulator setpoints', '/api/profile'],
      ['Snapshot JSON', 'The exact state the renderer is drawing', '/api/snapshot'],
    ];
    for (const [label, blurb, url] of items) {
      const button = document.createElement('button');
      button.type = 'button';
      button.innerHTML = `<strong>${label}</strong><span>${blurb}</span>`;
      button.addEventListener('click', () => {
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = '';
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        this._closeMenus();
        this.toast(`Downloading ${label}`, 'good');
      });
      menu.appendChild(button);
    }
  }

  _toggleMenu(id) {
    const menu = this.$(id);
    const wasHidden = menu.hidden;
    this._closeMenus();
    menu.hidden = !wasHidden;
  }

  _closeMenus() {
    this.$('menu-presets').hidden = true;
    this.$('menu-export').hidden = true;
  }

  // ── tooltips ────────────────────────────────────────────────────────────
  _attachTip(target, title, blurb, effect) {
    if (!title && !blurb) return;
    target.addEventListener('mouseenter', () => this._showTip(target, title, blurb, effect));
    target.addEventListener('mouseleave', () => this._hideTip());
    target.addEventListener('focus', () => this._showTip(target, title, blurb, effect));
    target.addEventListener('blur', () => this._hideTip());
  }

  _showTip(target, title, blurb, effect) {
    const tip = this.$('tip');
    tip.innerHTML = `<b>${title}</b>${blurb || ''}${effect ? `<br><em>${effect}</em>` : ''}`;
    tip.hidden = false;
    const rect = target.getBoundingClientRect();
    const width = tip.offsetWidth;
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(10, Math.min(window.innerWidth - width - 10, left));
    let top = rect.bottom + 8;
    if (top + tip.offsetHeight > window.innerHeight - 10) top = rect.top - tip.offsetHeight - 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(10, top)}px`;
  }

  _hideTip() {
    this.$('tip').hidden = true;
  }

  toast(message, kind = '') {
    const stack = this.$('toast');
    const item = document.createElement('div');
    item.className = `toast ${kind}`;
    item.textContent = message;
    stack.appendChild(item);
    window.setTimeout(() => item.remove(), 3400);
  }

  // ── onboarding ──────────────────────────────────────────────────────────
  _bindGuide() {
    this.$('btn-help').addEventListener('click', () => this.openGuide());
    this.$('guide-close').addEventListener('click', () => this.closeGuide());
    this.$('guide-tour').addEventListener('click', () => {
      this.closeGuide();
      this.actions.loadPreset('pavlovian');
    });
  }

  openGuide() { this.$('guide').hidden = false; }
  closeGuide() {
    this.$('guide').hidden = true;
    try { window.localStorage.setItem('flybrain.guideSeen', '1'); } catch { /* private mode */ }
  }

  maybeShowGuide() {
    let seen = null;
    try { seen = window.localStorage.getItem('flybrain.guideSeen'); } catch { /* private mode */ }
    if (!seen) this.openGuide();
  }

  // ── live updates ────────────────────────────────────────────────────────
  updateControls(snapshot, previous) {
    const endocrine = snapshot.endocrine;
    this._updateSliders(
      endocrine.levels, endocrine.tonic, endocrine.phasic, endocrine.spikes,
    );
    this._updateMutations(snapshot.mutations || {});
    this._updateDerived(endocrine);
    this._updateVitals(endocrine);
    this._updateMode(snapshot);
    this._updateStatus(snapshot);
    this._updateRegions(snapshot.regionActivation);
    this._updateMemory(snapshot.memory);
    this._updateStimuli(snapshot.world ? snapshot.world.stimuli : null, previous);
    const paused = snapshot.paused;
    this.$('btn-play').textContent = paused ? '▶' : '❚❚';
    this.$('btn-play').setAttribute('aria-pressed', String(paused));
    if (snapshot.world && snapshot.world.key) this._updateEnvironments(snapshot.world.key);
  }

  _updateDerived(endocrine) {
    const derived = endocrine.derived;
    const rows = [
      ['motorGain', 'Motor gain', derived.motorGain, 2.4, '#38bdf8'],
      ['persistence', 'Persistence', derived.persistence, 1.5, '#f9d423'],
      ['reactivity', 'Reactivity', derived.reactivity, 2.0, '#ff7a45'],
      ['riskTolerance', 'Risk tolerance', derived.riskTolerance, 1.0, '#a3e635'],
      ['flightReadiness', 'Flight readiness', derived.flightReadiness, 1.5, '#f59e0b'],
      ['feedingDrive', 'Feeding drive', derived.feedingDrive, 1.5, '#7ad1ff'],
    ];
    const container = this.$('derived');
    if (!this._derivedBuilt) {
      container.innerHTML = rows.map(([key, label]) => `
        <div class="metric" data-key="${key}">
          <span class="metric-label">${label}</span>
          <span class="metric-value">0</span>
          <span class="metric-bar"><span style="width:0%"></span></span>
        </div>`).join('');
      this._derivedBuilt = {};
      for (const [key] of rows) {
        const node = container.querySelector(`[data-key="${key}"]`);
        this._derivedBuilt[key] = {
          value: node.querySelector('.metric-value'),
          bar: node.querySelector('.metric-bar span'),
        };
      }
    }
    for (const [key, , value, max, color] of rows) {
      const entry = this._derivedBuilt[key];
      entry.value.textContent = value.toFixed(2);
      entry.bar.style.width = `${Math.min(100, (value / max) * 100)}%`;
      entry.bar.style.background = color;
    }
  }

  _updateVitals(endocrine) {
    const state = endocrine.state;
    const rows = [
      ['Energy', state.energy, '#c084fc'],
      ['Hunger', state.hunger, '#a3e635'],
      ['Arousal', state.arousal, '#ff7a45'],
      ['Fear', state.fear, '#f87171'],
      ['Sleep pressure', state.sleepPressure, '#5eead4'],
    ];
    const container = this.$('vitals');
    if (!this._vitalsBuilt) {
      container.innerHTML = rows.map(([label]) => `
        <div class="metric">
          <span class="metric-label">${label}</span>
          <span class="metric-value">0</span>
          <span class="metric-bar"><span style="width:0%"></span></span>
        </div>`).join('');
      this._vitalsBuilt = [...container.querySelectorAll('.metric')].map((node) => ({
        value: node.querySelector('.metric-value'),
        bar: node.querySelector('.metric-bar span'),
      }));
    }
    rows.forEach(([, value, color], index) => {
      const entry = this._vitalsBuilt[index];
      entry.value.textContent = value.toFixed(2);
      entry.bar.style.width = `${Math.min(100, value * 100)}%`;
      entry.bar.style.background = color;
    });

    const phase = state.timeOfDay;
    this.$('day-fill').style.left = `${phase * 100}%`;
    this.$('day-night').textContent = state.sleep ? '☾' : (phase > 0.25 && phase < 0.75 ? '☀' : '◐');
  }

  _updateMode(snapshot) {
    const mode = snapshot.primary ? snapshot.primary.mode : 'REST';
    const info = this.manifest.modes[mode] || { label: mode, color: '#5eead4', blurb: '' };
    this.$('mode-name').textContent = info.label;
    this.$('mode-hint').textContent = info.blurb;
    const chip = this.$('mode-chip');
    chip.classList.toggle('escape', mode === 'ESCAPE');
    chip.querySelector('.mode-dot').style.background = info.color;
    chip.querySelector('.mode-dot').style.color = info.color;
  }

  _updateStatus(snapshot) {
    const primary = snapshot.primary;
    if (!primary) return;
    const senses = primary.senses;
    const chips = [
      ['odour', `${(senses.attract + senses.aversive).toFixed(2)}`, senses.aversive > senses.attract ? 'hot' : 'cool'],
      ['temp', `${senses.tempC.toFixed(1)}°C`, senses.tempC > 34 ? 'hot' : senses.tempC < 20 ? 'warm' : ''],
      ['threat', senses.threat.toFixed(2), senses.threat > 0.6 ? 'hot' : senses.threat > 0.2 ? 'warm' : ''],
      ['loom', `${senses.loomRate >= 0 ? '+' : ''}${senses.loomRate.toFixed(2)}`, senses.loomRate > 0.05 ? 'hot' : ''],
      ['speed', `${primary.speed.toFixed(0)} mm/s`, ''],
      ['wings', `${primary.wingHz.toFixed(0)} Hz`, primary.wingAmp > 0.4 ? 'cool' : ''],
      ['learned', `${senses.learned >= 0 ? '+' : ''}${senses.learned.toFixed(2)}`, senses.learned < -0.05 ? 'warm' : senses.learned > 0.05 ? 'cool' : ''],
      ['casting', primary.casting > 0.4 ? 'yes' : 'no', primary.casting > 0.4 ? 'warm' : ''],
    ];
    const container = this.$('status-chips');
    if (!this._statusBuilt) {
      container.innerHTML = chips.map(([label]) => `<span class="status-chip">${label} <b>–</b></span>`).join('');
      this._statusBuilt = [...container.querySelectorAll('.status-chip')];
    }
    chips.forEach(([, value, kind], index) => {
      const node = this._statusBuilt[index];
      node.querySelector('b').textContent = value;
      node.className = `status-chip ${kind}`;
    });
  }

  _updateRegions(activation) {
    if (!activation) return;
    const container = this.$('region-readout');
    if (!this._regionsBuilt) {
      container.innerHTML = this.manifest.atlas.regions
        .map((region) => `<span class="region-pill" data-key="${region.key}">
            <i style="background:rgb(${region.color.join(',')})"></i>${region.key} <b>0.00</b></span>`)
        .join('');
      this._regionsBuilt = {};
      for (const region of this.manifest.atlas.regions) {
        this._regionsBuilt[region.key] = container.querySelector(`[data-key="${region.key}"]`);
      }
    }
    for (const region of this.manifest.atlas.regions) {
      const node = this._regionsBuilt[region.key];
      if (!node) continue;
      const value = activation[region.key] ?? 0;
      node.querySelector('b').textContent = value.toFixed(2);
      const hidden = this.hiddenRegions.has(region.key);
      node.style.opacity = hidden ? '0.25' : '1';
      node.style.borderColor = value > 0.35 && !hidden
        ? `rgba(${region.color.join(',')},0.8)` : '';
      node.style.background = value > 0.35 && !hidden
        ? `rgba(${region.color.join(',')},${Math.min(0.28, value * 0.34)})` : '';
    }
  }

  _updateMemory(memory) {
    const container = this.$('memory');
    const empty = this.$('memory-empty');
    if (!memory || !memory.entries.length) {
      container.innerHTML = '';
      empty.hidden = false;
      return;
    }
    empty.hidden = true;
    container.innerHTML = memory.entries.map((entry) => {
      const value = entry.value;
      const magnitude = Math.abs(value) * 50;
      const positive = value >= 0;
      const color = positive ? '#4ade80' : '#f87171';
      const style = positive
        ? `left:50%;width:${magnitude}%`
        : `left:${50 - magnitude}%;width:${magnitude}%`;
      return `
        <div class="mem-item">
          <span class="mem-name">${entry.name}</span>
          <span class="mem-val" style="color:${color}">${value >= 0 ? '+' : ''}${value.toFixed(2)}</span>
          <div class="mem-bar">
            <span class="mid"></span>
            <span style="${style};background:${color}"></span>
          </div>
        </div>`;
    }).join('');
  }

  _updateStimuli(stimuli, previous) {
    if (!stimuli || stimuli === this._lastStimuli) return;
    this._lastStimuli = stimuli;
    const container = this.$('stimulus-list');
    if (!stimuli.length) {
      container.innerHTML = '<span class="stim-empty">Nothing placed yet. Use the toolbelt below the arena.</span>';
      return;
    }
    container.innerHTML = '';
    for (const stimulus of stimuli) {
      const item = document.createElement('div');
      item.className = 'stim-item';
      item.innerHTML = `
        <span class="stim-swatch" style="background:${stimulus.color}"></span>
        <span class="stim-label">${stimulus.label}</span>
        <span class="stim-pos">${stimulus.x.toFixed(0)}, ${stimulus.z.toFixed(0)}</span>
        <button class="stim-del" title="Remove">×</button>`;
      item.querySelector('.stim-del').addEventListener('click', (event) => {
        event.stopPropagation();
        this.backend.cmd('remove_stimulus', { id: stimulus.id }).catch(() => {});
      });
      // Dragging a list item repositions the stimulus in the world directly, which
      // is quicker than deleting and re-placing when tuning an experiment.
      item.draggable = true;
      item.addEventListener('dragstart', (event) => {
        event.dataTransfer.setData('text/plain', stimulus.id);
        item.classList.add('dragging');
        this.actions.beginDragStimulus?.(stimulus.id);
      });
      item.addEventListener('dragend', () => item.classList.remove('dragging'));
      container.appendChild(item);
    }
    void previous;
  }

  // ── log ─────────────────────────────────────────────────────────────────
  addEvents(entries) {
    for (const entry of entries) this.entries.push(entry);
    if (this.entries.length > 260) this.entries = this.entries.slice(-260);
    this._renderLog();
    const last = entries[entries.length - 1];
    if (last) {
      const ticker = this.$('ticker-text');
      ticker.textContent = `[${last.t.toFixed(1)}s] ${last.message}`;
      ticker.className = last.level === 'alert' ? 'alert' : last.level === 'learn' ? 'learn' : '';
    }
  }

  _renderLog() {
    const list = this.$('log');
    if (!this._filtersBuilt) {
      const filters = this.$('log-filters');
      filters.innerHTML = LOG_CHANNELS.map((channel) => `
        <button data-channel="${channel.key}" class="${channel.key === 'all' ? 'on' : ''}">${channel.label}</button>`).join('');
      filters.addEventListener('click', (event) => {
        const button = event.target.closest('button');
        if (!button) return;
        this.logFilter = button.dataset.channel;
        for (const sibling of filters.querySelectorAll('button')) {
          sibling.classList.toggle('on', sibling === button);
        }
        this._renderLog();
      });
      this._filtersBuilt = true;
    }

    const visible = this.logFilter === 'all'
      ? this.entries
      : this.entries.filter((entry) => entry.channel === this.logFilter);

    if (!visible.length) {
      list.innerHTML = `<li class="log-empty">Nothing logged on this channel yet.</li>`;
      return;
    }
    const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 24;
    list.innerHTML = visible.slice(-140).map((entry) => `
      <li class="${entry.level}">
        <time>${entry.t.toFixed(1)}s</time>
        <span>${this._escape(entry.message)}</span>
      </li>`).join('');
    if (atBottom) list.scrollTop = list.scrollHeight;
  }

  _escape(text) {
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── graphs ──────────────────────────────────────────────────────────────
  setGraphData(graphs) {
    this.graphs = graphs;
  }

  drawGraphs() {
    const canvas = this.$('graphs');
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(2, Math.floor(rect.width));
    const height = Math.max(2, Math.floor(rect.height));
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const graphs = this.graphs;
    const keys = this.graphKeys.filter((key) => graphs && graphs[key]);
    if (!keys.length) {
      ctx.fillStyle = '#4a5568';
      ctx.font = '11px system-ui';
      ctx.fillText('Select a channel below to plot it', 12, height / 2);
      return;
    }

    // Each channel is normalised to its own window and labelled at the right edge, so
    // the gutter has to be measured from the widest label rather than guessed, or the
    // longest readout gets sliced off by the canvas edge.
    ctx.font = '9.5px ui-monospace, monospace';
    const readouts = keys.map((key) => {
      const series = graphs[key];
      const points = series.v || [];
      const last = points.length ? points[points.length - 1] : 0;
      const text = `${series.label.split(' ')[0]} ${last.toFixed(Math.abs(last) > 10 ? 0 : 2)}`;
      return { text, color: series.color, width: ctx.measureText(text).width };
    });
    const gutter = Math.ceil(Math.max(...readouts.map((entry) => entry.width))) + 12;

    const pad = { left: 6, right: gutter, top: 10, bottom: 12 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;

    // A shared time axis, since every series is sampled on the same clock.
    let minT = Infinity;
    let maxT = -Infinity;
    for (const key of keys) {
      const series = graphs[key];
      if (series.t.length) {
        minT = Math.min(minT, series.t[0]);
        maxT = Math.max(maxT, series.t[series.t.length - 1]);
      }
    }
    if (!Number.isFinite(minT) || maxT <= minT) return;

    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let index = 0; index <= 4; index++) {
      const y = pad.top + (plotHeight * index) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotWidth, y);
      ctx.stroke();
    }

    const placed = [];
    keys.forEach((key, order) => {
      const series = graphs[key];
      const points = series.v;
      if (!points.length) return;
      // Each channel is normalised to its own window and labelled at the right, so
      // a 400 mm/s velocity trace and a 0..1 modulator can share one panel.
      let low = Math.min(...points);
      let high = Math.max(...points);
      if (high - low < 1e-6) { high = low + 0.5; low -= 0.5; }
      const span = high - low;

      ctx.beginPath();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = series.color;
      ctx.globalAlpha = 0.95;
      points.forEach((value, index) => {
        const t = series.t[index] ?? minT;
        const x = pad.left + ((t - minT) / (maxT - minT)) * plotWidth;
        const y = pad.top + plotHeight - ((value - low) / span) * plotHeight;
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;

      const last = points[points.length - 1];
      const y = pad.top + plotHeight - ((last - low) / span) * plotHeight;
      ctx.fillStyle = series.color;
      ctx.beginPath();
      ctx.arc(pad.left + plotWidth, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
      placed.push({ y, color: series.color, ...readouts[order] });
    });

    // Labels ride their own trace, but several channels pinned near 0 would land on
    // top of each other, so the stack is relaxed apart and the winner keeps the spot.
    placed.sort((a, b) => a.y - b.y);
    const lineHeight = 11;
    for (let index = 1; index < placed.length; index++) {
      const overlap = placed[index - 1].y + lineHeight - placed[index].y;
      if (overlap > 0) placed[index].y += overlap;
    }
    const overflow = placed.length
      ? placed[placed.length - 1].y - (pad.top + plotHeight)
      : 0;
    if (overflow > 0) for (const entry of placed) entry.y -= overflow;

    ctx.textAlign = 'right';
    ctx.font = '9.5px ui-monospace, monospace';
    for (const entry of placed) {
      ctx.fillStyle = entry.color;
      ctx.fillText(entry.text, width - 5, entry.y + 3);
    }
    ctx.textAlign = 'left';

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotHeight);
    ctx.lineTo(pad.left + plotWidth, pad.top + plotHeight);
    ctx.stroke();
  }

  // ── audio ───────────────────────────────────────────────────────────────
  toggleAudio() {
    const button = this.$('btn-audio');
    if (this.audioOn) {
      if (this.audio) this.audio.output.gain.value = 0;
      this.audioOn = false;
      button.textContent = 'Audio: off';
      button.setAttribute('aria-pressed', 'false');
      return;
    }
    if (!this.audio) this.audio = createWingbeatSynth();
    this.audio.context.resume();
    this.audio.output.gain.value = 0.06;
    this.audioOn = true;
    button.textContent = 'Audio: on';
    button.setAttribute('aria-pressed', 'true');
    this.toast('Wing-beat synthesis on: pitch tracks octopamine and airspeed', 'good');
  }

  /** Drive the synth from the fly's real wing frequency. */
  updateAudio(snapshot) {
    if (!this.audioOn || !this.audio) return;
    const primary = snapshot.primary;
    if (!primary) return;
    const intensity = primary.wingAmp;
    this.audio.setFrequency(primary.wingHz, intensity);
    const flare = primary.mode === 'ESCAPE' ? 1.0 : 0.0;
    this.audio.setFlare(flare, primary.roll || 0);
  }
}

/**
 * A small wing-beat synthesiser.
 *
 * Two sawtooth oscillators at the fundamental and one harmonic through a bandpass,
 * with a noise component for the air hiss, which is what a wingbeat actually sounds
 * like: a buzzy fundamental plus broadband turbulence.
 */
function createWingbeatSynth() {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  const context = new Ctor();

  const output = context.createGain();
  output.gain.value = 0;
  output.connect(context.destination);

  // A stereo panner sits between the filter and the master gain so a turning fly
  // sweeps across the stereo image instead of always sitting dead centre.
  const panner = context.createStereoPanner();
  panner.pan.value = 0;
  panner.connect(output);

  const filter = context.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.value = 240;
  filter.Q.value = 2.4;
  filter.connect(panner);

  const fundamental = context.createOscillator();
  fundamental.type = 'sawtooth';
  const fundamentalGain = context.createGain();
  fundamentalGain.gain.value = 0.55;
  fundamental.connect(fundamentalGain).connect(filter);

  const harmonic = context.createOscillator();
  harmonic.type = 'square';
  const harmonicGain = context.createGain();
  harmonicGain.gain.value = 0.16;
  harmonic.connect(harmonicGain).connect(filter);

  // Air noise: a short looping buffer of white noise through the same filter.
  const noiseLength = context.sampleRate * 0.5;
  const noiseBuffer = context.createBuffer(1, noiseLength, context.sampleRate);
  const data = noiseBuffer.getChannelData(0);
  for (let index = 0; index < noiseLength; index++) data[index] = (Math.random() * 2 - 1) * 0.5;
  const noise = context.createBufferSource();
  noise.buffer = noiseBuffer;
  noise.loop = true;
  const noiseGain = context.createGain();
  noiseGain.gain.value = 0.10;
  const noiseFilter = context.createBiquadFilter();
  noiseFilter.type = 'highpass';
  noiseFilter.frequency.value = 1800;
  noise.connect(noiseFilter).connect(noiseGain).connect(panner);

  fundamental.start();
  harmonic.start();
  noise.start();

  return {
    context,
    output,
    panner,
    setFrequency(hz, amplitude) {
      const clamped = Math.max(80, Math.min(420, hz || 218));
      fundamental.frequency.setTargetAtTime(clamped, context.currentTime, 0.03);
      harmonic.frequency.setTargetAtTime(clamped * 2.02, context.currentTime, 0.03);
      filter.frequency.setTargetAtTime(clamped * 1.1, context.currentTime, 0.05);
      // Wings only make noise while they are moving.
      output.gain.setTargetAtTime(amplitude > 0.05 ? 0.05 + 0.07 * amplitude : 0.0,
        context.currentTime, 0.08);
    },
    /** Pan the sound by up to half the image, and whine on escape. */
    setFlare(amount, roll) {
      fundamental.detune.setTargetAtTime(amount * 260, context.currentTime, 0.05);
      const pan = Math.max(-1, Math.min(1, (roll || 0) * 0.75));
      panner.pan.setTargetAtTime(pan, context.currentTime, 0.12);
    },
  };
}
