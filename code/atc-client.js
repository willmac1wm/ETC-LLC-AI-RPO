/**
 * ATC AI RPO — JavaScript Client SDK
 * ====================================
 * Drop this file into your Vercel / Netlify game app to connect to
 * the hosted ATC AI RPO API server.
 *
 * Usage (ESM / browser):
 *   import { ATCClient } from './atc-client.js';
 *
 *   const atc = new ATCClient({
 *     baseUrl: 'https://your-api.onrender.com',   // your deployed API URL
 *     apiKey:  'your-api-key',                     // from .env / Render dashboard
 *   });
 *
 *   await atc.loadFacility('N90');
 *   const result = await atc.parseText('delta 452 turn right heading 180 descend 3000');
 *   console.log(result.readback);    // "turn right heading one eight zero, ..."
 *   console.log(result.stars_keys);  // ["DAL452", "H180", "D030"]
 *
 * Real-time via WebSocket:
 *   atc.onResult = (result) => updateGameUI(result);
 *   atc.onStateUpdate = (states) => renderTrafficPanel(states);
 *   atc.connect();
 *   atc.sendText('united 731 squawk 4521');
 */

export class ATCClient {
  /**
   * @param {object} opts
   * @param {string} opts.baseUrl  - Base URL of the ATC API server (no trailing slash)
   * @param {string} [opts.apiKey] - API key (Bearer token). Omit if server has auth disabled.
   */
  constructor({ baseUrl, apiKey = '' }) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey  = apiKey;
    this._socket = null;

    /** Callbacks — override these */
    this.onResult      = null;   // (result: ParseResult) => void
    this.onStateUpdate = null;   // (states: AircraftStateMap) => void
    this.onConnected   = null;   // () => void
    this.onError       = null;   // (err: string) => void
  }

  // ── HTTP helpers ────────────────────────────────────────────────────────────

  _headers(extra = {}) {
    const h = { 'Content-Type': 'application/json', ...extra };
    if (this.apiKey) h['Authorization'] = `Bearer ${this.apiKey}`;
    return h;
  }

  async _get(path) {
    const res = await fetch(`${this.baseUrl}${path}`, { headers: this._headers() });
    if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
    return res.json();
  }

  async _post(path, body) {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method:  'POST',
      headers: this._headers(),
      body:    JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `POST ${path} → ${res.status}`);
    }
    return res.json();
  }

  // ── REST API ────────────────────────────────────────────────────────────────

  /** Check server health. Returns { status, model, facility, aircraft, ts }. */
  async health() {
    return this._get('/api/health');
  }

  /** List available facility packs. Returns [{ id, name }]. */
  async listFacilities() {
    return this._get('/api/facilities');
  }

  /**
   * Load a facility data pack on the server.
   * @param {string} facilityId  e.g. 'N90', 'PCT', 'SCT'
   */
  async loadFacility(facilityId) {
    return this._post('/api/load_facility', { facility_id: facilityId });
  }

  /**
   * Parse a plain-text ATC instruction.
   * @param {string} text  e.g. "delta 452 turn right heading 180 descend 3000"
   * @returns {ParseResult}
   *
   * ParseResult shape:
   * {
   *   ts:               '14:32:01',
   *   transcript:       'delta 452 turn right heading 180 descend 3000',
   *   callsign:         'DAL452',
   *   command:          'DAL452 TR H180 A30',
   *   readback:         'turn right heading one eight zero, descend and maintain three thousand, Delta four five two',
   *   stars_keys:       ['DAL452', 'H180', 'D030'],
   *   confident:        true,
   *   confidence_reason:'2 token(s)'
   * }
   */
  async parseText(text) {
    return this._post('/api/parse_text', { text });
  }

  /**
   * Transcribe an audio file and parse the result.
   * @param {File|Blob} audioBlob  WAV file (16 kHz mono preferred)
   * @returns {ParseResult}
   */
  async transcribeAudio(audioBlob) {
    const form = new FormData();
    form.append('audio', audioBlob, 'audio.wav');
    const res = await fetch(`${this.baseUrl}/api/transcribe`, {
      method:  'POST',
      headers: this.apiKey ? { 'Authorization': `Bearer ${this.apiKey}` } : {},
      body:    form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `transcribe → ${res.status}`);
    }
    return res.json();
  }

  /** Get all tracked aircraft states. Returns { [callsign]: AircraftState }. */
  async getState() {
    return this._get('/api/state');
  }

  /**
   * Remove an aircraft from the state tracker.
   * @param {string} callsign  e.g. 'DAL452'
   */
  async removeAircraft(callsign) {
    const res = await fetch(`${this.baseUrl}/api/state/${callsign}`, {
      method:  'DELETE',
      headers: this._headers(),
    });
    if (!res.ok) throw new Error(`DELETE state/${callsign} → ${res.status}`);
    return res.json();
  }

  // ── WebSocket (real-time) ────────────────────────────────────────────────────

  /**
   * Open a Socket.IO connection for real-time parsing.
   * Requires socket.io-client in your project:
   *   npm install socket.io-client
   *   import { io } from 'socket.io-client';
   *
   * Set this.onResult and this.onStateUpdate before calling connect().
   */
  connect() {
    if (this._socket) return;

    // Dynamically import socket.io-client so this file works without it
    // when only REST is needed.
    const url    = this.baseUrl;
    const apiKey = this.apiKey;

    import('socket.io-client').then(({ io }) => {
      this._socket = io(url, {
        query:             { api_key: apiKey },
        transports:        ['websocket'],
        reconnectionDelay: 2000,
      });

      this._socket.on('connect', () => {
        console.log('[ATCClient] WebSocket connected');
        this.onConnected?.();
      });

      this._socket.on('connected', (data) => {
        console.log('[ATCClient] Server ack:', data);
      });

      this._socket.on('result', (data) => {
        this.onResult?.(data);
      });

      this._socket.on('state_update', (states) => {
        this.onStateUpdate?.(states);
      });

      this._socket.on('facility_loaded', (data) => {
        if (data.error) console.warn('[ATCClient] Facility load error:', data.error);
        else console.log('[ATCClient] Facility loaded:', data.name);
      });

      this._socket.on('disconnect', () => {
        console.warn('[ATCClient] WebSocket disconnected');
      });

      this._socket.on('connect_error', (err) => {
        console.error('[ATCClient] Connection error:', err.message);
        this.onError?.(err.message);
      });
    }).catch(err => {
      console.error('[ATCClient] socket.io-client not available. Use REST instead.', err);
    });
  }

  /** Disconnect the WebSocket. */
  disconnect() {
    this._socket?.disconnect();
    this._socket = null;
  }

  /**
   * Send a text instruction over WebSocket (real-time).
   * Result fires via this.onResult callback.
   * @param {string} text
   */
  sendText(text) {
    if (!this._socket?.connected) {
      console.warn('[ATCClient] Not connected. Call connect() first.');
      return;
    }
    this._socket.emit('parse_text', { text });
  }

  /**
   * Load a facility pack over WebSocket.
   * @param {string} facilityId
   */
  sendLoadFacility(facilityId) {
    this._socket?.emit('load_facility', { facility_id: facilityId });
  }
}

// ── Convenience: record mic audio and send to API ─────────────────────────────

/**
 * Record from the browser microphone for `durationMs` milliseconds,
 * then call client.transcribeAudio() and return the ParseResult.
 *
 * @param {ATCClient} client
 * @param {number}    [durationMs=4000]
 * @returns {Promise<ParseResult>}
 *
 * Example:
 *   const result = await recordAndParse(atc, 4000);
 *   console.log(result.readback);
 */
export async function recordAndParse(client, durationMs = 4000) {
  const stream  = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks   = [];

  return new Promise((resolve, reject) => {
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob   = new Blob(chunks, { type: 'audio/wav' });
      try {
        const result = await client.transcribeAudio(blob);
        resolve(result);
      } catch (err) {
        reject(err);
      }
    };
    recorder.start();
    setTimeout(() => recorder.stop(), durationMs);
  });
}
