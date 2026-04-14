#!/usr/bin/env python3
"""
ATC-RPO: Air Traffic Control Real-time Phraseology Output
ETC-LLC AI-RPO System

Listens to live audio, detects voice activity, transcribes ATC
communications using the fine-tuned Whisper model, logs all
transcriptions, and optionally reads them back via TTS.

Usage:
    python3 atc_rpo.py              # Basic mode
    python3 atc_rpo.py --tts        # With text-to-speech readback
    python3 atc_rpo.py --file audio.wav  # Transcribe a file
    python3 atc_rpo.py --device 2   # Use specific audio input device
    python3 atc_rpo.py --list-devices   # List available audio devices
"""

import os
import sys
import time
import queue
import logging
import argparse
import threading
import datetime
import numpy as np
import sounddevice as sd
import webrtcvad
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path.home() / "Desktop" / "ETC-LLC-AI-RPO"
MODEL_DIR = BASE_DIR / "models" / "atc-whisper"
LOG_DIR   = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─── Audio Settings ───────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000   # Hz — Whisper expects 16kHz
CHANNELS           = 1
CHUNK_MS           = 30      # VAD frame size (10, 20, or 30ms only)
CHUNK_SIZE         = int(SAMPLE_RATE * CHUNK_MS / 1000)
VAD_AGGRESSIVENESS = 2       # 0 (least) – 3 (most aggressive filtering)
SILENCE_CHUNKS     = 25      # silent frames before end-of-transmission (~750ms)
MIN_SPEECH_CHUNKS  = 10      # ignore clips shorter than this (~300ms)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = LOG_DIR / f"atc_rpo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─── Colours for terminal output ──────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    END     = "\033[0m"


def banner():
    print(f"""{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════╗
║          ATC-RPO  –  ETC-LLC AI System               ║
║    Real-time Air Traffic Control Transcription       ║
╚══════════════════════════════════════════════════════╝{C.END}
""")


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_model(model_dir: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    logger.info(f"Loading model from: {model_dir}")

    if not Path(model_dir).exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    logger.info(f"Compute device: {device.upper()}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

    logger.info("Model ready.")
    return asr_pipe


# ─── Transcriber ──────────────────────────────────────────────────────────────
class ATCTranscriber:
    def __init__(self, model_dir, use_tts=False, device_index=None):
        self.model_dir    = model_dir
        self.use_tts      = use_tts
        self.device_index = device_index
        self.audio_queue  = queue.Queue()
        self.running      = False
        self.vad          = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.tts_engine   = None
        self.pipe         = load_model(model_dir)

        if use_tts:
            self._init_tts()

    def _init_tts(self):
        import pyttsx3
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 155)
        logger.info("TTS engine initialised.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        # Convert float32 → int16 bytes for webrtcvad
        chunk = (indata[:, 0] * 32768).astype(np.int16).tobytes()
        self.audio_queue.put(chunk)

    def _is_speech(self, chunk_bytes: bytes) -> bool:
        try:
            return self.vad.is_speech(chunk_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def _transcribe(self, audio_bytes: bytes):
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < SAMPLE_RATE * 0.3:   # skip < 300 ms
            return None
        result = self.pipe(audio, generate_kwargs={"language": "english"})
        text = result["text"].strip()
        return text if text else None

    def _speak(self, text: str):
        if self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def _print_transcript(self, text: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"{C.GREEN}{C.BOLD}[{ts}]{C.END}  {C.YELLOW}{text}{C.END}"
        print(line)
        logger.info(f"TRANSCRIPT: {text}")

    # ── Live microphone mode ──────────────────────────────────────────────────
    def run_live(self):
        self.running    = True
        speech_buf      = b""
        silent_count    = 0
        speech_count    = 0
        in_speech       = False

        banner()
        print(f"{C.BOLD}Listening… (Ctrl+C to stop){C.END}\n")
        logger.info("Live capture started.")

        stream_kwargs = dict(
            samplerate  = SAMPLE_RATE,
            channels    = CHANNELS,
            dtype       = "float32",
            blocksize   = CHUNK_SIZE,
            callback    = self._audio_callback,
        )
        if self.device_index is not None:
            stream_kwargs["device"] = self.device_index

        with sd.InputStream(**stream_kwargs):
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                is_speech = self._is_speech(chunk)

                if is_speech:
                    speech_buf   += chunk
                    speech_count += 1
                    silent_count  = 0
                    in_speech     = True

                elif in_speech:
                    speech_buf   += chunk
                    silent_count += 1

                    if silent_count >= SILENCE_CHUNKS:
                        if speech_count >= MIN_SPEECH_CHUNKS:
                            text = self._transcribe(speech_buf)
                            if text:
                                self._print_transcript(text)
                                if self.use_tts:
                                    threading.Thread(
                                        target=self._speak, args=(text,), daemon=True
                                    ).start()
                        # Reset
                        speech_buf   = b""
                        silent_count = 0
                        speech_count = 0
                        in_speech    = False

    # ── File transcription mode ───────────────────────────────────────────────
    def run_file(self, filepath: str):
        import scipy.io.wavfile as wav
        logger.info(f"Transcribing file: {filepath}")
        sr, data = wav.read(filepath)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim > 1:
            data = data[:, 0]
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            data = resample(data, int(len(data) * SAMPLE_RATE / sr))
        result = self.pipe({"sampling_rate": SAMPLE_RATE, "raw": data},
                           generate_kwargs={"language": "english"})
        text = result["text"].strip()
        banner()
        print(f"{C.YELLOW}{text}{C.END}\n")
        logger.info(f"TRANSCRIPT: {text}")
        return text

    def stop(self):
        self.running = False
        logger.info("ATC-RPO stopped.")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ATC-RPO: Real-time Air Traffic Control Transcription"
    )
    parser.add_argument("--tts",          action="store_true",
                        help="Enable text-to-speech readback of transcriptions")
    parser.add_argument("--file",         type=str, default=None,
                        help="Transcribe a WAV file instead of live audio")
    parser.add_argument("--device",       type=int, default=None,
                        help="Audio input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--model-dir",    type=str, default=str(MODEL_DIR),
                        help="Path to the ATC Whisper model directory")
    parser.add_argument("--vad",          type=int, default=VAD_AGGRESSIVENESS,
                        choices=[0, 1, 2, 3],
                        help="VAD aggressiveness 0–3 (default: 2)")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    transcriber = ATCTranscriber(
        model_dir    = args.model_dir,
        use_tts      = args.tts,
        device_index = args.device,
    )

    try:
        if args.file:
            transcriber.run_file(args.file)
        else:
            transcriber.run_live()
    except KeyboardInterrupt:
        print(f"\n{C.RED}Shutting down ATC-RPO…{C.END}")
        transcriber.stop()


if __name__ == "__main__":
    main()
