/**
 * Backend client: bootstrap manifest, SSE state stream, command channel.
 *
 * The server publishes at 60 Hz. Rendering does not consume that stream directly:
 * the scene interpolates between the two most recent snapshots, so a dropped or
 * late frame cannot produce a stutter in the visual, and the physics rate stays
 * independent of the display rate.
 */

const state = {
  connected: false,
  lastSnapshot: null,
  previous: null,
  lastMessageAt: 0,
  messages: 0,
  dropped: 0,
  latencyMs: 0,
};

export class Backend {
  constructor() {
    this.stream = null;
    this.handlers = { state: [], events: [], glance: [], status: [] };
    this.retry = 0;
    this.sentAt = 0;
  }

  on(kind, fn) {
    (this.handlers[kind] ||= []).push(fn);
    return this;
  }

  emit(kind, payload) {
    for (const fn of this.handlers[kind] || []) {
      try { fn(payload); } catch (error) { console.error(`[${kind}] handler failed`, error); }
    }
  }

  async bootstrap() {
    const response = await fetch('/api/bootstrap');
    if (!response.ok) throw new Error(`bootstrap failed: HTTP ${response.status}`);
    return response.json();
  }

  /** Open the SSE stream. Reconnection is handled by EventSource itself. */
  connect() {
    if (this.stream) this.stream.close();
    const source = new EventSource('/api/stream');
    this.stream = source;

    source.onopen = () => {
      this.retry = 0;
      state.connected = true;
      this.emit('status', { connected: true });
    };

    source.onmessage = (message) => {
      let payload;
      try {
        payload = JSON.parse(message.data);
      } catch (error) {
        state.dropped += 1;
        return;
      }
      this.receivedAt = performance.now();
      state.previous = state.lastSnapshot;
      state.lastSnapshot = payload;
      state.lastMessageAt = this.receivedAt;
      state.messages += 1;
      if (payload.events) this.emit('events', payload.events);
      if (payload.glance) this.emit('glance', payload.glance);
      this.emit('state', payload);
    };

    source.onerror = () => {
      const now = performance.now();
      // A momentary gap is normal when the tab is backgrounded; only report it
      // once it has actually lasted long enough for a user to notice.
      if (state.connected && now - state.lastMessageAt > 2500) {
        state.connected = false;
        this.emit('status', { connected: false, message: 'stream stalled, reconnecting' });
      }
    };

    // Browsers throttle background tabs, so on return poll for immediate state
    // rather than waiting for the next push.
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) this.snapshot().then((payload) => this.emit('state', payload)).catch(() => {});
    });

    return source;
  }

  async snapshot() {
    const response = await fetch('/api/snapshot');
    if (!response.ok) throw new Error(`snapshot failed: HTTP ${response.status}`);
    const payload = await response.json();
    state.previous = state.lastSnapshot;
    state.lastSnapshot = payload;
    return payload;
  }

  cmd(name, payload = {}) {
    this.sentAt = performance.now();
    return fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, payload }),
    }).then((response) => {
      if (!response.ok) throw new Error(`command ${name} failed: HTTP ${response.status}`);
      return response.json();
    }).catch((error) => {
      console.error(error);
      throw error;
    });
  }

  stats() {
    return { ...state, snapshot: state.lastSnapshot, previous: state.previous };
  }

  /** The most recent snapshot, interpolated toward the newest one. */
  interpolated(alpha = 1) {
    const a = state.lastSnapshot;
    const b = state.previous;
    if (!a) return null;
    if (!b || alpha >= 1) return a;
    const out = { ...a };
    const mesh = (from, to, t) => from + (to - from) * t;
    out.flies = (a.flies || []).map((fly, index) => {
      const earlier = b.flies && b.flies[index];
      if (!earlier) return fly;
      return {
        ...fly,
        position: fly.position.map((v, axis) => mesh(earlier.position[axis], v, alpha)),
        heading: faceAngle(earlier.heading, fly.heading, alpha),
        pitch: mesh(earlier.pitch, fly.pitch, alpha),
        roll: mesh(earlier.roll, fly.roll, alpha),
        speed: mesh(earlier.speed, fly.speed, alpha),
        wingPhase: fly.wingPhase,
      };
    });
    return out;
  }
}

/** Shortest-arc angular interpolation, so a heading wrap does not spin the model. */
function faceAngle(from, to, t) {
  let delta = to - from;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return from + delta * t;
}

export const backend = new Backend();

/** Format a simulation time as mm:ss.t, which is what the log and HUD use. */
export function formatTime(seconds) {
  const safe = Math.max(0, seconds || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${rest.toFixed(1).padStart(4, '0')}`;
}

export function download(url) {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = '';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}
