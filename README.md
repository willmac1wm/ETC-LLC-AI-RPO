# ETC-LLC AI-RPO (ATC Whisper System)

## Agent Continuity
If you are an AI agent picking up this project, read this entire file before touching anything.

---

## STATUS: LIVE & RUNNING
**As of 2026-04-13** — Server is running, model is loaded, UI is operational.

- Server URL: http://127.0.0.1:8765
- Open that URL in Chrome to use the interface
- Hold SPACE BAR to transmit, release to transcribe

---

## How to Start the Server

```bash
cd ~/Desktop/ETC-LLC-AI-RPO
source venv/bin/activate
python3 code/atc_whisper_server.py
```

Keep that Terminal window open. The server must stay running.

---

## Full Directory Map

```
~/Desktop/ETC-LLC-AI-RPO/
├── code/
│   ├── atc_whisper_server.py   ← Flask + Flask-SocketIO backend (THE SERVER)
│   ├── atc_ai_interface.html   ← Premium radar UI (served at http://127.0.0.1:8765)
│   └── atc_rpo.py              ← Original CLI transcription script (standalone, no server)
├── models/
│   └── atc-whisper/            ← jacktol/whisper-large-v3-finetuned-for-ATC (~6.17GB)
│       ├── model-00001-of-00002.safetensors  (4.99GB)
│       ├── model-00002-of-00002.safetensors  (1.18GB)
│       └── config.json, tokenizer files, etc.
├── dataset/
│   └── atc-asr/                ← jacktol/ATC-ASR-Dataset (~809MB)
├── logs/                       ← Transcription and server logs
├── backups/
├── venv/                       ← Python virtual environment (USE THIS)
├── requirements.txt
├── setup.sh
└── README.md                   ← This file
```

---

## Architecture

```
Controller Mic
    |
    v
Browser (atc_ai_interface.html)
    | Hold SPACE → records audio via MediaRecorder
    | Release → sends audio blob via WebSocket
    v
atc_whisper_server.py  (Flask + Flask-SocketIO on port 8765)
    | Runs audio through Whisper model on MPS (Apple Silicon GPU)
    | Filters hallucinations (single chars, known noise strings)
    v
Transcript returned → displayed in Phraseology Log
```

---

## What Has Been Built (Agent History)

### Session 1 — Claude Cowork (2026-04-13)
- Downloaded jacktol/whisper-large-v3-finetuned-for-ATC to models/atc-whisper/
- Downloaded jacktol/ATC-ASR-Dataset to dataset/atc-asr/
- Built atc_rpo.py (CLI version with VAD + live microphone)
- Planned OpenStars integration for FAA white paper demo
- Built localWhisperService.ts (TypeScript client for OpenStars)
- Wrote OPENSTAIRS_INTEGRATION_STEPS.md integration guide

### Session 2 — Antigravity (2026-04-13)
- Rebuilt server as Flask + Flask-SocketIO (code/atc_whisper_server.py)
- Built premium radar UI (code/atc_ai_interface.html)
- Added Space Bar PTT (hold to transmit, release to process)
- Added audio device selection dropdown (supports WM Microphone)
- Added Voice Activity Monitor (circular waveform display)
- Added hallucination filtering (blocks '!', 'You', single-char noise)
- Configured for MPS (Apple Silicon GPU) — verified working
- Set up venv and requirements.txt
- Verified server runs and transcribes correctly

---

## OpenStars Integration (NEXT MAJOR GOAL)

William wants to demo this system for the FAA as part of a white paper proposal.
The demo: controller speaks into mic → Whisper transcribes → AI pilot reads back correctly.

**OpenStars location:** ~/Desktop/awesome projects/OpenStars-Emulator/
**OpenStars stack:** React + TypeScript + Vite
**Already has:** Gemini 2.0 Flash (pilot AI), ElevenLabs TTS, per-aircraft radio queue

**Files to copy into OpenStars:**
- localWhisperService.ts → OpenStars-Emulator/src/services/localWhisperService.ts

**Integration steps:** See OPENSTAIRS_INTEGRATION_STEPS.md (in Cowork outputs)

**Command format OpenStars uses:** CALLSIGN + CODES e.g. 'AAL123 TL H270 A80'
- TL = Turn Left, TR = Turn Right, FH = Fly Heading
- H = Heading (3 digits), A = Altitude (hundreds), S = Speed (knots)

---

## Next Steps for the Next Agent

1. PILOT READBACK — Add the readback engine to atc_ai_interface.html
   - After Whisper transcribes a command, generate a pilot readback in proper ATC phraseology
   - Currently the UI only shows the transcript — it does NOT speak back yet
   - Options: call local Gemini API, use ElevenLabs, or use browser TTS as fallback

2. THREE-PANEL UI — Upgrade atc_ai_interface.html to show:
   - Panel 1: ATC Spoken Command (what controller said)
   - Panel 2: Pilot Readback in plain English
   - Panel 3: Keyboard Command (parsed OpenStars format e.g. AAL123 TL H270)

3. OPENSTARS WIRE-UP — Copy localWhisperService.ts into OpenStars and modify voiceService.ts
   to call the local server instead of browser Web Speech API

4. RESUME UPDATE — Add AI/technical expertise section to William's resume
   (resume PDF exists, needs new section showing domain knowledge)

---

## About William

- **Name:** William Macomber
- **Company:** ETC LLC (SDVOSB — Service-Disabled Veteran-Owned Small Business)
- **Background:** 17+ years Air Traffic Control, FAA ATC instructor, Navy ATC supervisor
- **Current role:** DTIS/FAA ATC Subject Matter Expert
- **Goal:** FAA white paper proposing AI-Enabled RPO (Radar Position Operator) replacement
- **HuggingFace:** logged in as 'willmac'
- **Style:** Action-oriented, no fluff. Prefers dark premium UIs. Knows ATC deeply.

---

*Last updated by Claude Cowork — 2026-04-13*
