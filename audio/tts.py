import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

class LinTalk:
    def __init__(self, model_path="models/kokoro-v1.0.onnx", voices_path="models/voices-v1.0.bin"):
        print("🔧 Initializing Kokoro TTS...")
        self.kokoro = Kokoro(model_path, voices_path)
        self.sample_rate = 24000
        sd.default.samplerate = self.sample_rate
        sd.default.channels = 1

    def speak(self, text, voice="af_heart"):
        if not text:
            return

        try:
            # 1. Generate the speech array
            samples, _ = self.kokoro.create(
                text, 
                voice=voice, 
                speed=1.1,  # Slightly faster for a snappier assistant
                lang="en-us"
            )

            # 2. Prepend 0.4s of silence to wake up the PipeWire hardware
            silence_len = int(self.sample_rate * 0.4)
            silence = np.zeros(silence_len, dtype=np.float32)
            padded_audio = np.concatenate([silence, samples])

            # 3. Play non-blocking, then wait
            sd.play(padded_audio, self.sample_rate)
            sd.wait() 

        except Exception as e:
            print(f"⚠️ TTS Error: {e}")


