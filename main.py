import json
import os
import subprocess
import time
from google import genai
from dotenv import load_dotenv
from audio.stt import LinVoice

load_dotenv()

class LinAI:
    def __init__(self, system_skills_json):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env!")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-2.5-flash-lite' 
        
        self.system_instruction = f"""
        You are LinAgent, the intelligent assistant created by Callisto1232 and fox7524 in order to perform system tasks as an artificial intelligence.
        Current Environment: KDE6 Plasma.
        
        Available Skills:
        {json.dumps(system_skills_json, indent=2)}



        PROTOCOL:
        1. Map user prompt to an 'intent' from the JSON.
        2. Extract required 'parameters'.
        3. ALWAYS respond in valid JSON format ONLY.
        4. Use 'chat' intent for general conversation.

        OUTPUT STRUCTURE:
        {{
            "intent": "intent_name",
            "parameters": {{ "key": "value" }},
            "thought": "Brief explanation."
        }}
        """

    def decide_action(self, user_prompt, retries=3):
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config={'system_instruction': self.system_instruction}
                )
                
                text_output = response.text.strip()
                
                # Robust JSON Extraction
                if "{" in text_output and "}" in text_output:
                    start_idx = text_output.find("{")
                    end_idx = text_output.rfind("}") + 1
                    return json.loads(text_output[start_idx:end_idx])
                
                return {"intent": "chat", "parameters": {"message": text_output}, "thought": "Conversational."}

            except Exception as e:
                if "503" in str(e) and attempt < retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                return {"error": f"LinAI failed: {e}"}

class LinAgentSystem:
    def __init__(self, system_json="data/system.json", vars_json="data/vars.json"):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.system_path = os.path.join(base_path, system_json)
        self.vars_path = os.path.join(base_path, vars_json)
        
        self.skills = self._load_json_data(self.system_path, "system_skills")
        self.variables = self._load_json_data(self.vars_path)

    def _load_json_data(self, path, root_key=None):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle the [ { "key": [...] } ] structure from your system.json
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                
                if root_key:
                    return data.get(root_key, {})
                return data
        except Exception as e:
            print(f"Error loading {os.path.basename(path)}: {e}")
            return {}

    def _resolve_variable(self, param_value):
        """Resolves variables like printer IPs or $HOME paths."""
        if not isinstance(param_value, str):
            return param_value
            
        # 1. Check if it's a pointer to vars.json (e.g., "neptune_4_pro")
        # We check the 'printers' and 'directories' keys specifically
        for category in self.variables.values():
            if isinstance(category, dict) and param_value in category:
                val = category[param_value]
                # If it's a dict (like printer info), return the 'ip' or the value itself
                return val.get("ip", val) if isinstance(val, dict) else val
        
        # 2. Expand Linux environment variables ($HOME, etc.)
        return os.path.expandvars(param_value)

    def execute_intent(self, intent_name, **kwargs):
        command_template = None
        
        # Flattened search through categorized skills
        for actions in self.skills.values():
            for action in actions:
                if action.get('intent') == intent_name:
                    command_template = action.get('command')
                    break
            if command_template: break
        
        if not command_template:
            return f"Error: Intent '{intent_name}' not found."

        try:
            # Resolve parameters using vars.json and env expansion
            processed_kwargs = {k: self._resolve_variable(v) for k, v in kwargs.items()}
            
            # Specialized chat handling
            if intent_name == "chat":
                return processed_kwargs.get("message", "...")

            # Inject variables into template
            final_command = command_template.format(**processed_kwargs)
            
            # openSUSE Privilege Handling (kdesu for KDE6)
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
    lin_ai = LinAI(system.skills)
    voice = LinVoice(model_path="models/distil-large-v3")

    print(f"--- LinAgent Live (v1.5) ---")
    print("Mode: System Native (openSUSE) | Target: KDE6")
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            if user_input.lower() == "voice":
                     user_input = voice.listen(duration=3)
                     if not user_input:
                        print("no speech detected")
                        continue
                     print(f"Voice: {user_input}")

            decision = lin_ai.decide_action(user_input)
            
            if "error" in decision:
                print(f"❌ AI Error: {decision['error']}")
                continue

            intent = decision.get("intent")
            params = decision.get("parameters", {})
            print(f"🧠 Thought: {decision.get('thought')}")

            output = system.execute_intent(intent, **params)
            print(f"🖥️  System: {output}")

        except KeyboardInterrupt:
            break
