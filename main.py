import json
import os
import subprocess
import time
import threading
import numpy as np
import sounddevice as sd
import shlex
from google import genai
from dotenv import load_dotenv
from audio.stt import LinVoice
from rag import LinRAG
from audio.tts import LinTalk
from openwakeword.model import Model
from openwakeword import models
from queue import Queue, Empty
import sys
import logging

# 1. Create a logger object
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)  # Set the minimum logging level

# 2. Create a formatter (how you want your log lines to look)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 3. Console Handler (prints live to your terminal)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 4. File Handler (saves to your log file)
file_handler = logging.FileHandler("app_output.log", mode="a") # "a" to append, "w" to overwrite
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

load_dotenv()
SPEAK_DURATION = 3
WAKE_MODEL = "hey_jarvis"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
THRESHOLD = 0.3
API_MODEL = "gemini-2.5-flash" 

audio_queue = Queue()
wake_word_detected = threading.Event()
reset_flag = threading.Event()

def play_notification_sound():
    try:
        subprocess.Popen(["paplay", "audio/im_here.wav"])
    except Exception as e:
        logger.info(f"⚠️ Sound error: {e}")

class LinAI:
    def __init__(self, skills_json, system_json, vars_json):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = API_MODEL 
        self.rag = LinRAG(skills_json, system_json)
        self.base_instruction = """
        You are LinAgent. Designed as an AI assistant for KDE6 and OpenSuse Tumbleweed, created by callisto1232 and fox7524. 
        Your purpose is to do what the user says and help him use his system, do easy tasks.
        Respond ONLY in valid JSON.
        Structure: {"intent": "intent_name", "parameters": {"param": "val"}, "message": "vocal response"}
        If it is just a question, use intent "chat".

        DO NOT EXECUTE DANGEROUS COMMANDS THAT CAN HARM THE SYSTEM PERMANENTLY
        """

    def decide_action(self, user_prompt):
        relevant_context = self.rag.query(user_prompt, top_k=5)
        dynamic_instruction = f"{self.base_instruction}\nCONTEXT:\n{json.dumps(relevant_context)}"
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=user_prompt,
                config={'system_instruction': dynamic_instruction}
            )
            text = response.text.strip()
            
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            
            return {"intent": "chat", "parameters": {r}, "message": text}
        except Exception as e:
            logger.error(f"❌ API Error: {e}")
            return {"intent": "chat", "parameters": {}, "message": "System error."}

class LinAgentSystem:
    def __init__(self, system_json="data/system.json", vars_json="data/vars.json", skills_json="data/skills.json"):
        self.system_skills = self._load_json_data(system_json, "system_skills")
        self.variables = self._load_json_data(vars_json)
        self.skills = self._load_json_data(skills_json)

    def _load_json_data(self, path, root_key=None):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0: data = data[0]
                return data.get(root_key, {}) if root_key else data
        except: return {}

    def _resolve_variable(self, param_value):
        if not isinstance(param_value, str): return param_value
        for cat in self.variables.values():
            if isinstance(cat, dict) and param_value in cat:
                val = cat[param_value]
                return val.get("ip", val) if isinstance(val, dict) else val
        return os.path.expandvars(param_value)

    def execute_intent(self, intent_name, **kwargs):
        if not intent_name or intent_name == "chat":
            return kwargs.get("message", "I am LinAgent.")

        cmd_tpl = None
        search_list = []
        if isinstance(self.skills, list): search_list.extend(self.skills)
        if isinstance(self.system_skills, dict):
            for cat_list in self.system_skills.values():
                if isinstance(cat_list, list): search_list.extend(cat_list)

        for action in search_list:
            if action.get("intent") == intent_name:
                cmd_tpl = action.get("command")
                break
        
        if not cmd_tpl: return f"Intent {intent_name} not found."
        
        try:
            resolved_kwargs = {k: shlex.quote(str(self._resolve_variable(v))) for k, v in kwargs.items()}
            final_cmd = cmd_tpl.format(**resolved_kwargs)
            if "sudo " in final_cmd: 
                final_cmd = final_cmd.replace('sudo ', '')
                final_cmd = f"kdesu -c {shlex.quote(final_cmd)}"
            
            logger.info(f"🚀 Running: {final_cmd}")
            result = subprocess.run(final_cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() or "Success"
        except Exception as e: return f"Error: {str(e)}"

def sd_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def wake_word_loop():
    oww_model = Model([models[WAKE_MODEL]["model_path"]])
    while True:
        if reset_flag.is_set():
            oww_model.reset() 
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except Empty: break
            reset_flag.clear()
        try:
            chunk = audio_queue.get(timeout=0.1)
            audio_int16 = (np.clip(chunk.flatten(), -1, 1) * 32767).astype(np.int16)
            prediction = oww_model.predict(audio_int16)
            if any(score > THRESHOLD for score in prediction.values()):
                if not wake_word_detected.is_set():
                    play_notification_sound()
                    time.sleep(0.4) 
                    wake_word_detected.set()
        except Empty: continue

def run_linagent(system, lin_ai, stt, tts, stream):
    while True:
        wake_word_detected.wait()
        stream.stop()
        user_input = stt.listen(duration=SPEAK_DURATION)
        if user_input:
            logger.info(f"🎤: {user_input}")
            decision = lin_ai.decide_action(user_input)
            
            intent = decision.get("intent") or "chat"
            params = decision.get("parameters", {})
            
            # Key fix: ensure the AI's intended message is passed to execution
            if "message" in decision:
                params["message"] = decision["message"]
            
            output = system.execute_intent(intent, **params)
            
            if intent == "chat":
                speech = decision.get("message") or output
            else:
                speech = decision.get("message") or f"Executed {intent}. {output[:50]}"

            logger.info(f"🔊 Speaking: {speech}")
            tts.speak(speech)
            time.sleep(len(speech) * 0.12 + 1.2)

        reset_flag.set()
        wake_word_detected.clear()
        time.sleep(0.5)
        stream.start()

if __name__ == "__main__":
    system = LinAgentSystem()
    lin_ai = LinAI(system.skills, system.system_skills, system.variables)
    stt = LinVoice(model_path="models/distil-large-v3")
    tts = LinTalk()
    mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=sd_callback)
    threading.Thread(target=wake_word_loop, daemon=True).start()
    threading.Thread(target=run_linagent, args=(system, lin_ai, stt, tts, mic_stream), daemon=True).start()
    logger.info("--- LinAgent Stable v2.0 ---")
    with mic_stream:
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: pass
