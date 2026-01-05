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
from waitress import serve
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
builtin_voice_conds = None  # Store the default voice to restore later


def get_device():
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load the TTS model at startup."""
    global model, model_loaded, builtin_voice_conds

    device = get_device()
    logger.info(f"Loading ChatterboxTTS model on {device}...")

    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        model_loaded = True

        # Store the built-in voice conditions so we can restore them later
        # This fixes the issue where custom voices "stick" after being used
        if hasattr(model, 'conds') and model.conds is not None:
            builtin_voice_conds = model.conds
            logger.info("Saved built-in voice conditions for restoration")

        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False


def get_available_voices():
    """Get list of available voice presets from voices/ directory, including subdirectories."""
    voices = []
    if not VOICES_DIR.exists():
        return voices

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]

    def scan_directory(directory, category=None):
        for item in directory.iterdir():
            if item.is_dir():
                # Recurse into subdirectory, using folder name as category
                category_name = item.name.replace("_", " ").title()
                scan_directory(item, category_name)
            elif item.suffix.lower() in audio_extensions:
                # Get path relative to VOICES_DIR for the ID
                rel_path = item.relative_to(VOICES_DIR)
                voice_id = str(rel_path.with_suffix(""))  # e.g., "league/jinx"

                voices.append({
                    "id": voice_id,
                    "name": item.stem.replace("_", " ").title(),
                    "category": category or "General",
                    "file": item.name
                })

    scan_directory(VOICES_DIR)

    # Sort by category, then by name
    return sorted(voices, key=lambda v: (v["category"], v["name"]))


def get_voice_path(voice_id):
    """Get the file path for a voice preset, or None if not found.

    voice_id can be a simple name like "jinx" or a path like "league/jinx".
    """
    if not voice_id or not VOICES_DIR.exists():
        return None

    # Check for common audio extensions
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        path = VOICES_DIR / f"{voice_id}{ext}"
        if path.exists():
            return str(path)

    return None


def ensure_mono_audio(audio_path):
    """
    Ensure audio file is mono. If stereo, convert to mono and save to temp file.
    Returns path to mono audio (original if already mono, temp file if converted).
    """
    waveform, sample_rate = ta.load(audio_path)

    # Check if stereo (2 channels)
    if waveform.shape[0] > 1:
        logger.info(f"Converting {waveform.shape[0]}-channel audio to mono")
        # Average channels to get mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ta.save(temp_file.name, waveform, sample_rate)
        return temp_file.name, True  # Return path and flag indicating temp file

    return audio_path, False  # Already mono, no temp file created


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

        # Log what we received
        logger.info(f"Request: text='{text[:50]}...' exag={exaggeration} cfg={cfg_weight}")
        logger.info(f"Request: voice_preset={repr(voice_preset)}, voice_audio={'yes' if voice_audio_b64 else 'no'}")

        # Determine voice source
        audio_prompt_path = None
        temp_files_to_cleanup = []

        if voice_preset:
            # Preset voice selected
            preset_path = get_voice_path(voice_preset)
            if preset_path:
                audio_prompt_path, is_temp = ensure_mono_audio(preset_path)
                if is_temp:
                    temp_files_to_cleanup.append(audio_prompt_path)
                logger.info(f"VOICE: Using preset '{voice_preset}' from {preset_path}")
            else:
                logger.warning(f"VOICE: Preset '{voice_preset}' not found, falling back to default")
        elif voice_audio_b64:
            # Custom uploaded audio
            try:
                audio_bytes = base64.b64decode(voice_audio_b64)
                upload_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                upload_temp.write(audio_bytes)
                upload_temp.close()
                temp_files_to_cleanup.append(upload_temp.name)

                audio_prompt_path, is_temp = ensure_mono_audio(upload_temp.name)
                if is_temp and audio_prompt_path != upload_temp.name:
                    temp_files_to_cleanup.append(audio_prompt_path)

                logger.info(f"VOICE: Using custom uploaded audio ({len(audio_bytes)} bytes)")
            except Exception as e:
                logger.warning(f"VOICE: Failed to process uploaded audio: {e}, using default")
        else:
            # No voice specified - restore the built-in voice
            if builtin_voice_conds is not None:
                model.conds = builtin_voice_conds
                logger.info("VOICE: Restored built-in voice conditions")
            logger.info("VOICE: Using model's DEFAULT voice")

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
            # Clean up all temp files
            for temp_path in temp_files_to_cleanup:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Load model before starting server
    load_model()

    # Log available voices
    voices_list = get_available_voices()
    if voices_list:
        logger.info(f"Available voice presets: {[v['id'] for v in voices_list]}")
    else:
        logger.info("No voice presets found in voices/ directory")

    # Run production server with Waitress
    # Host 0.0.0.0 to be accessible via Tailscale
    port = int(os.environ.get("TTS_PORT", 5000))
    logger.info(f"Starting Waitress production server on http://0.0.0.0:{port}")
    serve(app, host="0.0.0.0", port=port, threads=4)
