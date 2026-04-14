"""
ATC AI RPO — Cloud API Server
==============================
REST + WebSocket API designed to be called from a web game app
hosted on Vercel / Netlify (or any frontend).

Endpoints
---------
GET  /api/health                   — liveness check
GET  /api/facilities               — list available facility packs
POST /api/load_facility            — {"facility_id": "N90"}
POST /api/parse_text               — {"text": "delta 452 heading 180 descend 3000"}
POST /api/transcribe               — multipart/form-data: audio file (WAV/MP3)
GET  /api/state                    — current tracked aircraft states
DELETE /api/state/<callsign>       — remove aircraft from state
WS   socket.io                     — real-time bidirectional channel

Authentication
--------------
Set API_KEY in environment (or .env).
Clients must send:   Authorization: Bearer <API_KEY>
Omit or leave blank in dev to disable auth.

CORS
----
Set ALLOWED_ORIGINS in environment (comma-separated).
Defaults to * for development.

Deploy
------
  Railway / Render:  see render.yaml / railway.json at project root
  Local:             python code/atc_api_server.py
"""

import os
import io
import re
import json
import time
import tempfile
import threading
import datetime
import numpy as np
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
FACILITY_DIR = BASE_DIR / "data" / "facilities"
MODEL_DIR    = BASE_DIR / "models" / "atc-whisper"

# ── Config from environment ────────────────────────────────────────────────────
API_KEY         = os.getenv("API_KEY", "")           # blank = auth disabled
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
PORT            = int(os.getenv("PORT", 8766))
CLOUD_MODEL     = os.getenv("CLOUD_MODEL", "openai/whisper-base.en")  # HF fallback

# ── Shared engine (no hardware deps) ──────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR / "code"))

from atc_engine import (
    parse_atc_command, generate_pilot_readback, map_to_stars,
    update_aircraft_state, get_aircraft_states, remove_aircraft, get_confidence,
    load_facility, list_facilities, get_loaded_facility,
    process_text, _aircraft_states,
)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "atc-api-secret-change-me")

CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS, async_mode="eventlet")

# ── ASR pipeline (lazy-loaded on first request) ────────────────────────────────
_asr_pipe  = None
_asr_lock  = threading.Lock()

def get_asr_pipeline():
    global _asr_pipe
    if _asr_pipe is not None:
        return _asr_pipe
    with _asr_lock:
        if _asr_pipe is not None:
            return _asr_pipe
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if MODEL_DIR.exists():
            model_id = str(MODEL_DIR)
            local    = True
            print(f"[*] Loading fine-tuned model from: {model_id}")
        else:
            model_id = CLOUD_MODEL
            local    = False
            print(f"[*] Fine-tuned model not found — downloading {model_id}")

        device = ("cuda" if torch.cuda.is_available()
                  else "mps"  if torch.backends.mps.is_available()
                  else "cpu")
        dtype  = torch.float16 if device != "cpu" else torch.float32
        print(f"[*] Compute device: {device.upper()}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=local,
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=local
        )

        _asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )
        print("[*] ASR pipeline ready.")
        return _asr_pipe


# ── Auth middleware ────────────────────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            return f(*args, **kwargs)   # auth disabled
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if token != API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Helpers ────────────────────────────────────────────────────────────────────
def _process_text(text):
    """Parse ATC text → full response dict (delegates to engine)."""
    result = process_text(text)
    # After state update, push to all WS clients
    socketio.emit("state_update", get_aircraft_states())
    return result


def _transcribe_audio(audio_bytes, sample_rate=16000):
    """Run Whisper on raw bytes. Returns transcript string or None."""
    pipe = get_asr_pipeline()
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < sample_rate * 0.3:
        return None
    result = pipe(audio, generate_kwargs={"language": "english"})
    text   = result["text"].strip()
    junk   = {"you", "thank you.", "subtitles by", "bye.", ""}
    return text if text and len(text) > 1 and text.lower() not in junk else None


# ── REST routes ────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status":   "ok",
        "model":    str(MODEL_DIR) if MODEL_DIR.exists() else CLOUD_MODEL,
        "facility": _facility_name(),
        "aircraft": len(_aircraft_states),
        "ts":       datetime.datetime.now().isoformat(),
    })


def _facility_name():
    try:
        from atc_whisper_server import _facility
        return _facility.get("facility_name", "none")
    except Exception:
        return "none"


@app.route("/api/facilities")
@require_api_key
def api_list_facilities():
    packs = []
    for fid in list_facilities():
        path = FACILITY_DIR / f"{fid}.json"
        try:
            with open(path) as f:
                d = json.load(f)
            packs.append({"id": fid, "name": d.get("facility_name", fid)})
        except Exception:
            packs.append({"id": fid, "name": fid})
    return jsonify(packs)


@app.route("/api/load_facility", methods=["POST"])
@require_api_key
def api_load_facility():
    body = request.get_json(silent=True) or {}
    fid  = body.get("facility_id", "")
    if not re.match(r"^[A-Za-z0-9_]{1,10}$", fid):
        return jsonify({"error": "invalid facility_id"}), 400
    fac = load_facility(fid)
    if not fac:
        return jsonify({"error": "facility not found"}), 404
    return jsonify({
        "id":         fid,
        "name":       fac.get("facility_name"),
        "airports":   fac.get("airports", []),
        "fixes_count": len(fac.get("fixes", [])),
    })


@app.route("/api/parse_text", methods=["POST"])
@require_api_key
def api_parse_text():
    """
    Accept ATC instruction as plain text, return parsed result.

    Request body (JSON):
      { "text": "delta 452 turn right heading 180 descend maintain 3000" }

    Response:
      {
        "ts":         "14:32:01",
        "transcript": "delta 452 turn right heading 180 descend maintain 3000",
        "callsign":   "DAL452",
        "command":    "DAL452 TR H180 A30",
        "readback":   "turn right heading one eight zero, descend and maintain three thousand, Delta four five two",
        "stars_keys": ["DAL452", "H180", "D030"],
        "confident":  true,
        "confidence_reason": "2 token(s)"
      }
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    return jsonify(_process_text(text))


@app.route("/api/transcribe", methods=["POST"])
@require_api_key
def api_transcribe():
    """
    Accept an audio file, transcribe with Whisper, parse, and return result.

    Request: multipart/form-data
      audio: WAV file (16kHz mono preferred; other formats auto-converted)

    Response: same shape as /api/parse_text
    """
    if "audio" not in request.files:
        return jsonify({"error": "audio file required (field name: audio)"}), 400

    audio_file = request.files["audio"]
    suffix     = Path(audio_file.filename or "audio.wav").suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        import scipy.io.wavfile as wav
        sr, data = wav.read(tmp_path)
        if data.dtype != np.int16:
            # Normalise and convert to int16
            if np.issubdtype(data.dtype, np.floating):
                data = (data * 32767).astype(np.int16)
            else:
                data = data.astype(np.int16)
        if data.ndim > 1:
            data = data[:, 0]   # mono
        if sr != 16000:
            from scipy.signal import resample
            data = resample(data, int(len(data) * 16000 / sr)).astype(np.int16)
        audio_bytes = data.tobytes()
    except Exception as e:
        return jsonify({"error": f"audio decode failed: {e}"}), 422
    finally:
        os.unlink(tmp_path)

    text = _transcribe_audio(audio_bytes)
    if not text:
        return jsonify({"error": "no speech detected"}), 422

    return jsonify(_process_text(text))


@app.route("/api/state")
@require_api_key
def api_state():
    """Return current aircraft state dict."""
    return jsonify(get_aircraft_states())


@app.route("/api/state/<callsign>", methods=["DELETE"])
@require_api_key
def api_remove_aircraft(callsign):
    """Remove an aircraft from the state tracker."""
    cs = callsign.upper()
    if remove_aircraft(cs):
        return jsonify({"removed": cs})
    return jsonify({"error": "not found"}), 404


# ── WebSocket events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def ws_connect():
    token = request.args.get("api_key", "")
    if API_KEY and token != API_KEY:
        disconnect()
        return False
    emit("connected", {
        "facility": _facility_name(),
        "aircraft": len(_aircraft_states),
    })


@socketio.on("parse_text")
def ws_parse_text(data):
    """
    Real-time parse over WebSocket.
    Send: { "text": "..." }
    Receive: same dict as REST /api/parse_text
    """
    text = (data or {}).get("text", "").strip()
    if not text:
        return
    result = _process_text(text)
    emit("result", result)
    # Broadcast state update to all connected clients
    socketio.emit("state_update", dict(_aircraft_states))


@socketio.on("load_facility")
def ws_load_facility(data):
    fid = (data or {}).get("facility_id", "")
    if not re.match(r"^[A-Za-z0-9_]{1,10}$", fid):
        emit("facility_loaded", {"error": "invalid id"})
        return
    fac = load_facility(fid)
    if not fac:
        emit("facility_loaded", {"error": "not found"})
        return
    emit("facility_loaded", {
        "id":   fid,
        "name": fac.get("facility_name"),
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[*] ATC API Server starting on port {PORT}")
    print(f"[*] Auth: {'enabled' if API_KEY else 'DISABLED (dev mode)'}")
    print(f"[*] CORS origins: {ALLOWED_ORIGINS}")
    print(f"[*] Facilities: {list_facilities()}")
    # Warm up the ASR pipeline in a background thread so first request is fast
    threading.Thread(target=get_asr_pipeline, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False)
