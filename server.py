"""
TTS Flask API Server

Exposes Chatterbox TTS over HTTP for use with Rails backend.
Run with: python server.py
"""

import os
import io
import base64
import tempfile
import logging
from pathlib import Path
from flask import Flask, request, jsonify
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Voice presets directory (relative to this script)
VOICES_DIR = Path(__file__).parent / "voices"

# Global model instance (loaded once at startup)
model = None
model_loaded = False


def get_device():
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load the TTS model at startup."""
    global model, model_loaded

    device = get_device()
    logger.info(f"Loading ChatterboxTTS model on {device}...")

    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        model_loaded = True
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False


def get_available_voices():
    """Get list of available voice presets from voices/ directory."""
    voices = []
    if VOICES_DIR.exists():
        for f in VOICES_DIR.iterdir():
            if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
                voices.append({
                    "id": f.stem,
                    "name": f.stem.replace("_", " ").title(),
                    "file": f.name
                })
    return sorted(voices, key=lambda v: v["name"])


def get_voice_path(voice_id):
    """Get the file path for a voice preset, or None if not found."""
    if not voice_id or not VOICES_DIR.exists():
        return None

    # Check for common audio extensions
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        path = VOICES_DIR / f"{voice_id}{ext}"
        if path.exists():
            return str(path)

    return None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": get_device()
    })


@app.route("/voices", methods=["GET"])
def voices():
    """List available voice presets."""
    return jsonify({
        "voices": get_available_voices()
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate speech from text.

    Request body (JSON):
        - text: string (required) - Text to synthesize
        - exaggeration: float (0.0-1.0, default 0.5) - Speech expressiveness
        - cfg_weight: float (0.0-1.0, default 0.5) - CFG guidance weight
        - voice_preset: string (optional) - Name of preset voice (e.g., "djt")
        - voice_audio: string (optional) - Base64-encoded audio for custom voice cloning

    Note: voice_preset takes precedence over voice_audio if both provided.

    Returns:
        - audio: string - Base64-encoded WAV audio
        - sample_rate: int - Audio sample rate
        - duration: float - Audio duration in seconds
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        exaggeration = float(data.get("exaggeration", 0.5))
        cfg_weight = float(data.get("cfg_weight", 0.5))
        voice_preset = data.get("voice_preset")
        voice_audio_b64 = data.get("voice_audio")

        # Clamp values to valid range
        exaggeration = max(0.0, min(1.0, exaggeration))
        cfg_weight = max(0.0, min(1.0, cfg_weight))

        logger.info(f"Generating speech: text='{text[:50]}...' exag={exaggeration} cfg={cfg_weight}")

        # Determine voice source (preset takes precedence)
        audio_prompt_path = None
        temp_file = None

        if voice_preset:
            # Use preset voice
            preset_path = get_voice_path(voice_preset)
            if preset_path:
                audio_prompt_path = preset_path
                logger.info(f"Using preset voice: {voice_preset}")
            else:
                logger.warning(f"Preset voice '{voice_preset}' not found, using default")
        elif voice_audio_b64:
            # Use uploaded audio
            try:
                audio_bytes = base64.b64decode(voice_audio_b64)
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(audio_bytes)
                temp_file.close()
                audio_prompt_path = temp_file.name
                logger.info("Using uploaded voice audio")
            except Exception as e:
                logger.warning(f"Failed to process voice audio: {e}")

        try:
            # Generate speech
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                audio_prompt_path=audio_prompt_path
            )

            # Convert to WAV bytes
            buffer = io.BytesIO()
            ta.save(buffer, wav, model.sr, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()

            # Calculate duration
            duration = wav.shape[1] / model.sr

            # Encode as base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            logger.info(f"Generated {duration:.2f}s of audio")

            return jsonify({
                "audio": audio_b64,
                "sample_rate": model.sr,
                "duration": duration
            })

        finally:
            # Clean up temp file (only if we created one)
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Load model before starting server
    load_model()

    # Log available voices
    voices = get_available_voices()
    if voices:
        logger.info(f"Available voice presets: {[v['id'] for v in voices]}")
    else:
        logger.info("No voice presets found in voices/ directory")

    # Run Flask server
    # Host 0.0.0.0 to be accessible via Tailscale
    port = int(os.environ.get("TTS_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
