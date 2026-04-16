import json
import os
import subprocess
import time
from google import genai
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()

class LinAI:
    def __init__(self, system_skills_json):
        """
        The intelligence core of LinAgent.
        Uses Gemini 1.5 Flash with a resilience layer for 503 errors and robust parsing.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env! Please add it.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-2.5-flash-lite'
        
        # Identity and context for the AI
        self.system_instruction = f"""
        You are LinAI, the intelligent assistant for LinAgent on openSUSE Tumbleweed.
        User: Ahmet, an 11th-grade maker/robotics dev in Istanbul.

        Available System Skills:
        {json.dumps(system_skills_json, indent=2)}

        PROTOCOL:
        1. Map user prompt to an 'intent' from the JSON.
        2. Extract required 'parameters'.
        3. ALWAYS respond in valid JSON format ONLY.
        4. If the user is just chatting or asking a question that doesn't need a command, use the 'chat' intent.

        OUTPUT STRUCTURE:
        {{
            "intent": "intent_name",
            "parameters": {{ "key": "value" }},
            "thought": "Briefly explain your choice in English."
        }}
        """

    def decide_action(self, user_prompt, retries=3):
        """Converts natural language to intent with a backoff retry and robust JSON extraction."""
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
                    clean_json = text_output[start_idx:end_idx]
                    return json.loads(clean_json)
                
                # Fallback if AI skips JSON format
                return {
                    "intent": "chat", 
                    "parameters": {"message": text_output}, 
                    "thought": "AI provided a direct conversational response."
                }

            except Exception as e:
                if "503" in str(e) and attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️  AI Busy (503). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return {"error": f"LinAI failed: {e}"}

class LinAgentSystem:
    def __init__(self, json_file="data/system.json"):
        """The execution engine that interacts with the Linux shell."""
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(base_path, json_file)
        self.skills = self._load_skills()

    def _load_skills(self):
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
                # Supports both a direct list or the categorized dictionary structure
                return data.get("system_skills", {})
        except Exception as e:
            print(f"❌ Error loading system.json: {e}")
            return {}

    def execute_intent(self, intent_name, **kwargs):
        """Searches for the intent across all categories and executes the command."""
        command_template = None
        
        # Search through all categories (file_management, window_management, etc.)
        for category_name, actions in self.skills.items():
            for action in actions:
                if action.get('intent') == intent_name:
                    command_template = action.get('command')
                    break
            if command_template: break
        
        if not command_template:
            return f"Error: Intent '{intent_name}' not found."

        try:
            # 1. Expand paths and prepare parameters
            processed_kwargs = {
                k: os.path.expanduser(v) if isinstance(v, str) else v 
                for k, v in kwargs.items()
            }
            
            # 2. Fill the command template
            final_command = command_template.format(**processed_kwargs)
            final_command = final_command.replace("//", "/")

            # 3. Handle 'chat' as a special case or execute as echo
            if intent_name == "chat":
                return processed_kwargs.get("message", "No message provided.")

            # 4. openSUSE/KDE GUI privilege handling
            if final_command.startswith("sudo "):
                # Remove sudo and non-interactive flags for kdesu compatibility
                clean_cmd = final_command.replace("sudo ", "").replace("--non-interactive ", "")
                final_command = f"kdesu -- {clean_cmd}"
            
            print(f"🚀 Executing: {final_command}")
            
            result = subprocess.run(final_command, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                return result.stdout.strip() if result.stdout else "Success"
            else:
                return f"Error: {result.stderr.strip()}"

        except Exception as e:
            return f"Execution error: {e}"

# --- MAIN LOOP ---
if __name__ == "__main__":
    system = LinAgentSystem()
    lin_ai = LinAI(system.skills)

    print("--- LinAgent Live (v1.2) ---")
    print("System: openSUSE Tumbleweed | Brain: Gemini 1.5 Flash")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            # 1. AI Decision
            decision = lin_ai.decide_action(user_input)
            
            if "error" in decision:
                print(f"❌ AI Error: {decision['error']}")
                continue

            # 2. Command Execution
            intent = decision.get("intent")
            params = decision.get("parameters", {})
            thought = decision.get("thought", "Processing...")

            if intent:
                print(f"🧠 LinAI Thought: {thought}")
                output = system.execute_intent(intent, **params)
                
                # Check for multiline output (like ls or search results)
                if "\n" in str(output):
                    print(f"🖥️  System Output:\n{output}")
                else:
                    print(f"🖥️  System: {output}")
            else:
                print("❓ AI could not determine the intent.")

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
