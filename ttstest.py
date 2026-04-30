import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

class LinTalk:
    def __init__(self, model_path="models/kokoro-v1.0.onnx", voices_path="models/voices-v1.0.bin"):
        self.kokoro = Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def speak(self, text, voice="af_heart"):
        if not text: return
        
        # 1. Generate the speech
        samples, _ = self.kokoro.create(text, voice=voice, speed=1.1, lang="en-us")
        
        # 2. Add 0.3 seconds of silence to wake up PipeWire
        silence_duration = 0.3 
        silence = np.zeros(int(self.sample_rate * silence_duration), dtype=np.float32)
        padded_samples = np.concatenate([silence, samples])
        
        # 3. Play the padded version
        sd.play(padded_samples, self.sample_rate)
        sd.wait()

if __name__ == "__main__":
    talker = LinTalk()
    # The first word 'Testing' shouldn't be cut off now
    talker.speak("Testing the audio pipeline to see if the first word is still clipped.")
