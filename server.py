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
from flask import Flask, request, jsonify
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": get_device()
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate speech from text.

    Request body (JSON):
        - text: string (required) - Text to synthesize
        - exaggeration: float (0.0-1.0, default 0.5) - Speech expressiveness
        - cfg_weight: float (0.0-1.0, default 0.5) - CFG guidance weight
        - voice_audio: string (optional) - Base64-encoded audio for voice cloning

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
        voice_audio_b64 = data.get("voice_audio")

        # Clamp values to valid range
        exaggeration = max(0.0, min(1.0, exaggeration))
        cfg_weight = max(0.0, min(1.0, cfg_weight))

        logger.info(f"Generating speech: text='{text[:50]}...' exag={exaggeration} cfg={cfg_weight}")

        # Handle voice cloning if audio provided
        audio_prompt_path = None
        temp_file = None

        if voice_audio_b64:
            try:
                # Decode base64 audio and save to temp file
                audio_bytes = base64.b64decode(voice_audio_b64)
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(audio_bytes)
                temp_file.close()
                audio_prompt_path = temp_file.name
                logger.info(f"Using voice clone from uploaded audio")
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
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Load model before starting server
    load_model()

    # Run Flask server
    # Host 0.0.0.0 to be accessible via Tailscale
    port = int(os.environ.get("TTS_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
