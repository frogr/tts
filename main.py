import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")
text = "I'm an [whisper] AI clone [gasp] of Austin, the famous league of legends streamer [sigh]. Thanks for tuning into the podcast! [woohoo] I just wanted to tell you: Jordan's GAY!! [laughter] Just kidding. No I not. U stupid."
wav = model.generate(text, audio_prompt_path="austin2.wav", exaggeration=0.6, cfg_weight=0.3)

ta.save("jordan.wav", wav, model.sr)
print("Audio saved to output.wav")