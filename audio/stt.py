import os 
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

os.environ["OMP_NUM_THREADS"] = "4"

class LinVoice:
    def __init__(self, model_path="./models/distil-large-v3"):
        self.fs = 16000
        self.model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=4
        )

    def listen(self, duration=15):
        sys.stderr.write(f"\r[LinAgent Listening ({duration}s)...] ")
        sys.stderr.flush()

        audio = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='int16')
        sd.wait()
        audio_float = audio.flatten().astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float,
            beam_size=1,
            language="en",
            vad_filter=True
        )

        text = " ".join([segment.text.strip() for segment in segments])
        return text.lower()


