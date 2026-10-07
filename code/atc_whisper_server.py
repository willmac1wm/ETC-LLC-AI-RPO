import os
import sys
import json
import time
import queue
import threading
import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd
import webrtcvad
import glob as _glob
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, Response
from flask_socketio import SocketIO, emit

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # moving mouse to any corner aborts injection
    pyautogui.PAUSE = 0.05  # 50 ms between keystrokes
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("[!] pyautogui not installed — keyboard injection disabled")

try:
    import pyttsx3 as _pyttsx3
    _tts_engine = _pyttsx3.init()
    _tts_engine.setProperty('rate', 175)  # slightly faster than default (natural pilot cadence)
    _tts_lock = threading.Lock()  # pyttsx3 is not thread-safe
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("[!] pyttsx3 unavailable — TTS readback disabled")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path.home() / "Desktop" / "ETC-LLC-AI-RPO"
CODE_DIR = BASE_DIR / "code"
MODEL_DIR = BASE_DIR / "models" / "atc-whisper"
LOG_DIR = BASE_DIR / "logs"
FACILITY_DIR = BASE_DIR / "data" / "facilities"
LOG_DIR.mkdir(exist_ok=True)

# ─── Engine (shared parser / state / facility logic) ─────────────────────────
import sys as _sys
_sys.path.insert(0, str(CODE_DIR))
from atc_engine import (
    parse_atc_command, generate_pilot_readback, map_to_stars,
    update_aircraft_state, get_aircraft_states, get_confidence,
    load_facility, list_facilities, get_loaded_facility,
    process_text as _engine_process_text,
    _aircraft_states,
)

# ─── Keyboard Injection ───────────────────────────────────────────────────────
def map_to_stars(command_str):
    """Convert internal command string into an ordered list of STARS keystroke strings.

    Internal token → STARS keystroke:
      H180          → H180           (heading)
      A30           → D030           (descend/maintain 3000)
      A350          → A350           (flight level 350)
      S250          → S250           (speed)
      SQ4521        → SQ4521         (squawk)
      APPR_ILS_27L  → CI27L          (cleared ILS 27L)
      APPR_RNAV_28R → CR28R          (cleared RNAV 28R)
      APPR_VIS_23   → CV23           (cleared visual 23)
      DCT_KEYED     → DKEYED         (direct-to fix)
      FREQ_119.1    → (skipped — freq changes are verbal/coordination only)
      TL / TR / FH  → (skipped — turn-direction decorators for readback)
    """
    if not command_str:
        return []
    parts = command_str.strip().split()
    keystrokes = [parts[0]]
    i = 1
    while i < len(parts):
        tok = parts[i]
        if tok in ('TL', 'TR', 'FH'):
            i += 1
            continue
        if tok.startswith('H'):
            keystrokes.append(tok)
        elif tok.startswith('A') and not tok.startswith('APPR'):
            alt = int(tok[1:])
            if alt >= 180:
                keystrokes.append(f"A{alt:03d}")
            else:
                keystrokes.append(f"D{alt:03d}")
        elif tok.startswith('S') and not tok.startswith('SQ'):
            keystrokes.append(tok)
        elif tok.startswith('SQ'):
            keystrokes.append(tok)
        elif tok.startswith('APPR_'):
            _, appr_type, rwy = tok.split('_', 2)
            fac_code = facility_approach_code(appr_type, rwy)
            if fac_code:
                keystrokes.append(fac_code)
            else:
                prefix = {'ILS': 'CI', 'ILSZ': 'CI', 'RNAV': 'CR',
                          'LOC': 'CL', 'LDA': 'CL', 'VIS': 'CV'}.get(appr_type, 'C')
                keystrokes.append(f"{prefix}{rwy}")
        elif tok.startswith('DCT_'):
            keystrokes.append('D' + tok[4:])
        i += 1
    return keystrokes


def inject_keystrokes(command_str):
    """Type STARS commands into the active window.
    Runs in its own thread so audio processing is never blocked.
    Move mouse to any screen corner to abort (pyautogui FAILSAFE)."""
    global inject_count
    if not PYAUTOGUI_AVAILABLE or not injection_enabled:
        return
    keystrokes = map_to_stars(command_str)
    if not keystrokes:
        return
    time.sleep(1.5)
    try:
        for ks in keystrokes:
            pyautogui.write(ks, interval=0.05)
            pyautogui.press('enter')
            time.sleep(0.3)
        inject_count += 1
        socketio.emit('inject_status', {
            'count': inject_count,
            'last': ' → '.join(keystrokes),
        })
    except Exception as e:
        err = "FAILSAFE: injection aborted" if "FailSafe" in type(e).__name__ else str(e)
        print(f"[!] Injection error: {err}")
        socketio.emit('inject_status', {'error': err})


# ─── TTS Readback ─────────────────────────────────────────────────────────────
def speak_readback(text):
    """Speak the pilot readback in a dedicated thread.
    Uses a lock because pyttsx3 is not thread-safe."""
    if not TTS_AVAILABLE or not tts_enabled or not text:
        return
    with _tts_lock:
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print(f"[!] TTS error: {e}")


# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(CODE_DIR))
app.config['SECRET_KEY'] = 'atc-whisper-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ─── Audio Settings ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
VAD_AGGRESSIVENESS = 2
SILENCE_CHUNKS = 25
MIN_SPEECH_CHUNKS = 10

# ─── Global State ─────────────────────────────────────────────────────────────
audio_queue = queue.Queue()
running = False
ptt_active = False
selected_device = None
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
asr_pipe = None
injection_enabled = False
inject_count = 0
tts_enabled = False
ptt_mode = True
session_log = deque(maxlen=1000)
WORKER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='atc-worker')

# Facility list cache to avoid re-reading JSON from disk on each UI request.
facility_cache = {}


def get_facility_cache():
    global facility_cache
    if facility_cache:
        return facility_cache
    for fid in list_facilities():
        try:
            with open(FACILITY_DIR / f"{fid}.json") as f:
                facility_cache[fid] = json.load(f)
        except Exception:
            facility_cache[fid] = {'facility_name': fid}
    return facility_cache


# ─── Model Loader ────────────────────────────────────────────────────────────
def load_model():
    global asr_pipe
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    print(f"[*] Loading Whisper model from: {MODEL_DIR}")

    if not MODEL_DIR.exists():
        print(f"[!] Model directory not found: {MODEL_DIR}")
        return False

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print(f"[*] Compute device: {device.upper()}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

    print("[*] Model loaded successfully.")
    return True


# ─── Audio Callback ───────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    global ptt_active
    if status:
        print(f"[!] Audio Status: {status}")
    chunk = (indata[:, 0] * 32768).astype(np.int16).tobytes()
    if not ptt_mode or ptt_active:
        audio_queue.put(chunk)


def transcribe(audio_bytes):
    if asr_pipe is None:
        return None
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < SAMPLE_RATE * 0.3:
        return None
    result = asr_pipe(audio, generate_kwargs={"language": "english"})
    text = result["text"].strip()

    if not text or len(text) <= 1 or text.lower() in ["you", "thank you.", "subtitles by", "bye."]:
        return None

    return text


# ─── Transcription Result Handler ─────────────────────────────────────────────
def handle_transcription(text):
    """Central handler for a completed utterance.
    Parses, gates confidence, updates state, emits to UI, fires TTS + injection."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {text}")

    cs, _cmd = parse_atc_command(text)
    _rb = generate_pilot_readback(_cmd)

    tokens = _cmd.split()[1:] if _cmd else []
    confident, reason = get_confidence(cs, tokens)
    print(f"[{'OK' if confident else '!!'}] Confidence: {reason}")

    socketio.emit('transcription', {
        'time': ts,
        'text': text,
        'type': 'atc',
        'command': _cmd,
        'readback': _rb,
        'confident': confident,
    })

    session_log.append({
        'ts': ts,
        'text': text,
        'command': _cmd,
        'readback': _rb,
        'confident': confident,
    })

    if cs and _cmd:
        update_aircraft_state(cs, _cmd)
        socketio.emit('state_update', get_aircraft_states())

    if _rb:
        WORKER_POOL.submit(speak_readback, _rb)

    if _cmd and confident:
        WORKER_POOL.submit(inject_keystrokes, _cmd)
    elif _cmd and not confident:
        socketio.emit('inject_status', {'skipped': reason})


# ─── Processing Loop ──────────────────────────────────────────────────────────
def processing_thread():
    global running, ptt_active
    speech_buf = b""
    silent_count = 0
    speech_count = 0
    in_speech = False
    pending_transcriptions = []

    print(f"[*] Audio processing thread started (Device: {selected_device}).")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                            blocksize=CHUNK_SIZE, device=selected_device, callback=audio_callback):
            while running:
                for future in list(pending_transcriptions):
                    if future.done():
                        pending_transcriptions.remove(future)
                        try:
                            text = future.result()
                        except Exception:
                            continue
                        if text:
                            handle_transcription(text)

                try:
                    chunk = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if ptt_mode and not ptt_active and in_speech:
                        if speech_count >= MIN_SPEECH_CHUNKS:
                            pending_transcriptions.append(WORKER_POOL.submit(transcribe, speech_buf))
                        speech_buf = b""
                        speech_count = 0
                        in_speech = False
                        socketio.emit('vad_status', {'active': False})
                    continue

                try:
                    is_speech = vad.is_speech(chunk, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if is_speech:
                    speech_buf += chunk
                    speech_count += 1
                    silent_count = 0
                    in_speech = True
                    socketio.emit('vad_status', {'active': True})
                elif in_speech:
                    speech_buf += chunk
                    silent_count += 1
                    if silent_count >= SILENCE_CHUNKS:
                        if speech_count >= MIN_SPEECH_CHUNKS:
                            pending_transcriptions.append(WORKER_POOL.submit(transcribe, speech_buf))
                        speech_buf = b""
                        silent_count = 0
                        speech_count = 0
                        in_speech = False
                        socketio.emit('vad_status', {'active': False})
    except Exception as e:
        print(f"[!] Engine Error: {e}")
        running = False
        socketio.emit('status', {'engine_running': False, 'error': str(e)})


# ─── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('atc_ai_interface.html')


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(str(CODE_DIR), path)


@app.route('/facilities')
def get_facilities():
    packs = []
    cache = get_facility_cache()
    for fid in sorted(cache):
        d = cache[fid]
        packs.append({'id': fid, 'name': d.get('facility_name', fid)})
    return jsonify(packs)


@app.route('/load_facility/<facility_id>')
def load_facility_route(facility_id):
    import re
    if not re.match(r'^[A-Za-z0-9_]{1,10}$', facility_id):
        return jsonify({'error': 'invalid facility id'}), 400
    fac = load_facility(facility_id)
    if not fac:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'id': facility_id, 'name': fac.get('facility_name'), 'fixes': len(fac.get('fixes', []))})


@app.route('/download_log')
def download_log():
    lines = ["TIME\tTRANSCRIPT\tCOMMAND\tREADBACK\tCONFIDENT"]
    for e in list(session_log):
        lines.append("\t".join([
            e.get('ts', ''),
            e.get('text', ''),
            e.get('command', '') or '',
            e.get('readback', '') or '',
            str(e.get('confident', '')),
        ]))
    content = "\n".join(lines)
    return Response(
        content,
        mimetype='text/tab-separated-values',
        headers={'Content-Disposition': 'attachment; filename=atc_session_log.tsv'}
    )


@socketio.on('connect')
def handle_connect():
    print("[*] Client connected.")
    devices = []
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if d['max_input_channels'] > 0:
                devices.append({'index': i, 'name': d['name']})
    except Exception:
        pass
    emit('status', {'connected': True, 'engine_running': running, 'devices': devices})


@socketio.on('start_engine')
def handle_start(data=None):
    global running, selected_device
    if data and 'device_index' in data:
        selected_device = data['device_index']

    if not running:
        running = True
        threading.Thread(target=processing_thread, daemon=True).start()
        emit('status', {'engine_running': True})


@socketio.on('ptt_status')
def handle_ptt(data):
    global ptt_active
    ptt_active = data.get('active', False)
    print(f"[*] PTT {'ACTIVE' if ptt_active else 'RELEASED'}")


@socketio.on('stop_engine')
def handle_stop():
    global running
    running = False
    emit('status', {'engine_running': False})


@socketio.on('toggle_vad_mode')
def handle_toggle_vad_mode(data):
    global ptt_mode
    ptt_mode = data.get('ptt', True)
    mode = 'PTT' if ptt_mode else 'CONTINUOUS'
    print(f"[*] VAD mode: {mode}")
    emit('vad_mode_status', {'ptt_mode': ptt_mode})


@socketio.on('toggle_tts')
def handle_toggle_tts(data):
    global tts_enabled
    tts_enabled = data.get('enabled', False)
    state = 'ENABLED' if tts_enabled else 'DISABLED'
    print(f"[*] TTS readback {state}")
    emit('tts_status', {
        'enabled': tts_enabled,
        'available': TTS_AVAILABLE,
    })


@socketio.on('toggle_injection')
def handle_toggle_injection(data):
    global injection_enabled
    injection_enabled = data.get('enabled', False)
    state = 'ENABLED' if injection_enabled else 'DISABLED'
    print(f"[*] Keyboard injection {state}")
    emit('inject_status', {
        'enabled': injection_enabled,
        'available': PYAUTOGUI_AVAILABLE,
    })


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if load_model():
        print("[*] Starting server on http://127.0.0.1:8765")
        socketio.run(app, host='127.0.0.1', port=8765, debug=False)
    else:
        print("[!] Failed to load model. Exiting.")
