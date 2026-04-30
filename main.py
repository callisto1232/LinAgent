import json
import os
import subprocess
import time
from google import genai
from dotenv import load_dotenv
from audio.stt import LinVoice
from rag import LinRAG
from audio.tts import LinTalk  # Your new Kokoro-based TTS

# Load API Key from .env
load_dotenv()
SPEAK_DURATION = 3

class LinAI:
    def __init__(self, skills_json, system_json, vars_json):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env!")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-2.5-flash' 
        
        self.rag = LinRAG(skills_json, system_json)
        self.vars_json = vars_json
        self.base_instruction = """
        You are LinAgent, a witty and grounded system assistant for openSUSE Tumbleweed and KDE6. You bridge the gap between natural language and terminal execution for a power-user environment.

        CORE CAPABILITIES:
        - System Administration: Manage packages via Zypper (install, remove, update, search) and monitor hardware (USB/Serial device listing).
        - Desktop Management: Control KDE6 workspaces and windows (move, switch, raise, close) and handle screen capture.
        - File Architecture: Execute advanced file operations (move, copy, rename, find latest) across mapped directories like ~/codes and ~/Downloads.
        - Lab Control: Manage media playback and dashboard connectivity for Neptune 4 Pro and Centauri Carbon 3D printers.
        
        OPERATIONAL PROTOCOL:
        1. Intent Mapping: Match requests to the provided skills context. Use the "chat" intent for general conversation.
        2. Variable Resolution: Silently resolve friendly names to their values using the variables context.
        3. Chat Style: Be authentic and direct. Never repeat the user's input. Keep responses concise for TTS output.
        4. Strict Output: You communicate exclusively through valid JSON.
        
        JSON STRUCTURE:
        {
          "intent": "intent_name",
          "parameters": { "key": "value" },
          "thought": "brief reasoning"
        }
        """

    def decide_action(self, user_prompt, retries=5):
        relevant_context = self.rag.query(user_prompt, top_k=5)
        
        dynamic_instruction = f"""
        {self.base_instruction}
        
        RELEVANT SKILLS:
        {json.dumps(relevant_context, indent=2)}

        VARIABLES:
        {json.dumps(self.vars_json, indent=2)}

        PROTOCOL:
        1. For general conversation or questions:
            - Set "intent" to "chat".
            - In "parameters", set "message" to a helpful and authentic response.
        2. For system tasks:
            - Set "intent" to the matching intent name and extract parameters.
        """

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config={'system_instruction': dynamic_instruction}
                )
                
                text_output = response.text.strip()
                
                if "{" in text_output and "}" in text_output:
                    start_idx = text_output.find("{")
                    end_idx = text_output.rfind("}") + 1
                    return json.loads(text_output[start_idx:end_idx])
                
                return {"intent": "chat", "parameters": {"message": text_output}, "thought": "Fallback."}

            except Exception as e:
                error_msg = str(e)
                if any(x in error_msg for x in ["503", "UNAVAILABLE", "429"]):
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ API Busy. Retrying in {wait_time}s... ({attempt+1}/{retries})")
                    time.sleep(wait_time)
                    continue
                return {"error": f"LinAI failed: {e}"}

class LinAgentSystem:
    def __init__(self, system_json="data/system.json", vars_json="data/vars.json", skills_json="data/skills.json"):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.system_path = os.path.join(base_path, system_json)
        self.vars_path = os.path.join(base_path, vars_json)
        self.skills_path = os.path.join(base_path, skills_json)
        
        self.system_skills = self._load_json_data(self.system_path, "system_skills")
        self.variables = self._load_json_data(self.vars_path)
        self.skills = self._load_json_data(self.skills_path)

    def _load_json_data(self, path, root_key=None):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                return data.get(root_key, {}) if root_key else data
        except Exception:
            return {}

    def _resolve_variable(self, param_value):
        if not isinstance(param_value, str):
            return param_value
        for category in self.variables.values():
            if isinstance(category, dict) and param_value in category:
                val = category[param_value]
                return val.get("ip", val) if isinstance(val, dict) else val
        return os.path.expandvars(param_value)

    def execute_intent(self, intent_name, **kwargs):
        if intent_name == "chat":
            return kwargs.get("message", "I'm here.")

        command_template = None
        all_sources = [self.skills, self.system_skills]
        
        for source in all_sources:
            for category, actions in source.items():
                if not isinstance(actions, list): continue
                for action in actions:
                    if isinstance(action, dict) and action.get('intent') == intent_name:
                        command_template = action.get('command')
                        break
                if command_template: break
            if command_template: break
        
        if not command_template:
            return f"Error: Intent '{intent_name}' not found."

        try:
            processed_kwargs = {k: self._resolve_variable(v) for k, v in kwargs.items()}
            final_command = command_template.format(**processed_kwargs)
            
            if "sudo " in final_command:
                clean_cmd = final_command.replace("sudo ", "").replace("--non-interactive ", "")
                final_command = f"kdesu -- {clean_cmd}"
            
            print(f"🚀 Executing: {final_command}")
            result = subprocess.run(final_command, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                return result.stdout.strip() if result.stdout else "Success"
            return f"Error: {result.stderr.strip()}"
        except Exception as e:
            return f"Execution error: {e}"

if __name__ == "__main__":
    system = LinAgentSystem()
    lin_ai = LinAI(system.skills, system.system_skills, system.variables)
    stt = LinVoice(model_path="models/distil-large-v3")
    tts = LinTalk() # Initialize the Kokoro voice

    print(f"--- LinAgent Live (v1.5) ---")
    print(f"Mode: Local RAG | AI: {lin_ai.model_id}")
    
    while True:
        try:
            user_input = input("\n👤 Ahmet: ")
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input: continue
            
            if user_input.lower() == "voice" or user_input.lower() == "v":
                user_input = stt.listen(duration=SPEAK_DURATION)
                if not user_input: continue
                print(f"Voice: {user_input}")

            decision = lin_ai.decide_action(user_input)
            
            if "error" in decision:
                print(f"❌ AI Error: {decision['error']}")
                continue

            intent = decision.get("intent")
            params = decision.get("parameters", {})
            
            print(f"🧠 Thought: {decision.get('thought')}")
            
            # Execute and get system feedback
            output = system.execute_intent(intent, **params)
            print(f"🖥️  System: {output}")

            # If it was a chat or if we want the agent to report success/failure
            if intent == "chat":
                tts.speak(params.get("message"))
            else:
                # Optional: Have the agent voice out the result of the command
                tts.speak(f"Task completed: {intent}")

        except KeyboardInterrupt:
            break
