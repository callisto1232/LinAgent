import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

class LinTalk:
    def __init__(self, model_path="models/kokoro-v1.0.onnx", voices_path="models/voices-v1.0.bin"):
        self.kokoro = Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def save(self, text, filename, voice="af_heart"):
        samples, _ = self.kokoro.create(text, voice=voice, speed=1.1, lang="en-us")
        sf.write(filename, samples, self.sample_rate)

if __name__ == "__main__":
    LinTalk().save("I'm listening.", "listening1.wav")
    LinTalk().save("Listening", "listening2.wav")
    LinTalk().save("Here", "here.wav")
    LinTalk().save("I'm here", "imhere.wav")
    LinTalk().save("Yes", "yes.wav")

