from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import sys
import os

# 1. Environment Optimization: Force specific thread count for your i3
os.environ["OMP_NUM_THREADS"] = "4"

MODEL_PATH = "../models/distil-large-v3"

# 2. Optimized Model Loading
model = WhisperModel(
    MODEL_PATH, 
    device="cpu", 
    compute_type="int8", # Keep int8 for i3 efficiency
    cpu_threads=4,       # Match your hardware threads
    num_workers=2        # Parallelize segments
)

def transcribe():
    fs = 16000
    duration = 4 # Increased buffer, but VAD will truncate silence
    
    sys.stderr.write("\r[Listening...] ")
    sys.stderr.flush()
    
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    audio_float = audio.flatten().astype(np.float32) / 32768.0
    
    # 3. VAD Optimization
    # vad_filter=True removes silence blocks before the model even sees them
    segments, _ = model.transcribe(
        audio_float, 
        beam_size=1,        # 1 is much faster than 5, and distil-large is accurate enough
        language="en",
        vad_filter=True,    # Crucial for performance
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    for segment in segments:
        text = segment.text.strip()
        if text:
            # Output for wl-copy
            print(text)
            sys.stderr.write(f"\rDone: {text}\n")

if __name__ == "__main__":
    sys.stderr.write("LinAgent STT Optimized (Int8/VAD). Ctrl+C to stop.\n")
    try:
        while True:
            transcribe()
    except KeyboardInterrupt:
        sys.stderr.write("\nStopped.\n")
