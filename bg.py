import sounddevice as sd
import numpy as np
from openwakeword.model import Model
from openwakeword import models

oww_model = Model([models["hey_jarvis"]["model_path"]]) 

def callback(indata, frames, time, status):
    print(f"Volume: {np.max(np.abs(indata)):.4f}", end="\r")
    audio_frame = (indata.flatten() * 32767).astype(np.int16)
    prediction = oww_model.predict(audio_frame)
    for mdl, score in prediction.items():
        if score > 0.1:
            print(f"Detected {mdl}!")

with sd.InputStream(samplerate=16000, channels=1, blocksize=1280, callback=callback):
    while True:
        sd.sleep(1000)
