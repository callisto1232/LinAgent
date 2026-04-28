# LinAgent
## AI System Agent For Linux

An intelligent agent built for performing automation tasks, executing scripts, and running system commands for Linux.

### Project Structure
```text
├── audio
│   ├── __init__.py
│   ├── stt.py
│   ├── test_stt.py
│   └── tts.py
├── data
│   ├── skills.json
│   ├── system.json
│   └── vars.json
├── download.py
├── LICENSE
├── main.py
├── memory.py
├── models
│   ├── distil-large-v3
│   └── MiniLM-L6-v2
├── rag.py
├── README.md
├── requirements.txt
├── test
│   ├── stt.py
│   └── test_api.py
└── test_rag.py

```
## Installation
- [ ] Adding...

## Roadmap
- [ ] Command chain for multi-operation support
- [ ] Project will be supported as a system-service
- [ ] Local-AI support after fine tuning Phi3 Model
- [ ] Text-to-speech support
- [x] Speech-to-text support
- [ ] ~Implement [OCR](https://github.com/callisto1232/LinVision) and `ydotool` to perform in-app processes~ cancelled 
- [ ] Cross-DE support by expanding system commands dataset
- [ ] Expand `system.json` for greater system-processes (web processes)
- [ ] Add `apps.json` for in-app operations
- [x] Implement playerctl
- [x] RAG to reduce token usage
- [ ] Log records for debugging 


<img width="1897" height="228" alt="image" src="https://github.com/user-attachments/assets/a54561e7-f8a9-46b3-9d92-d293da900dce" />

<img width="1894" height="520" alt="image" src="https://github.com/user-attachments/assets/b56ca10b-eb7a-42a4-bd81-df17a9b4e97b" />


Added speech to text support.
<img width="928" height="201" alt="image" src="https://github.com/user-attachments/assets/94860d69-8303-4565-bf8f-e64da78d0e4e" />



### Commands Dataset

### `/data/system.json`
```json
{
  "system_skills": {
    "file_management": [
      {
        "intent": "find_latest_file",
        "description": "Finds the most recently modified file",
        "parameters": ["directory", "extension"],
        "command": "ls -t \"{directory}\"/*.{extension} 2>/dev/null | head -n 1",
        "returns": "file_path"
      },
      {
        "intent": "copy_to_directory",
        "description": "Copies a file to a specific folder.",
        "parameters": ["file_path", "directory"],
        "command": "cp -r \"{file_path}\" \"{directory}\" ",
        "returns": "status"
      },
      {
        "intent": "move_to_directory",
        "description": "Moves a file to a specific folder",
        "parameters": ["file_path", "directory"],
        "command": "mv \"{file_path}\" \"{directory}\" ",
        "returns": "status"
      },
      {
        "intent": "rename_file",
        "description": "Renames a file",
        "parameters": ["file_name", "new_name"],
        "command": "mv \"{file_name}\" \"{new_name}\" ",
        "returns": "status"
      },
...
```
`system.json` includes system commands that we feed AI with. There five parameters in this json file, `intent`, `description`, `parameters`, `command` and `returns`. I will expand this dataset further and add `apps.json` to create a dataset of in-app commands for further processes in the future.

### `main.py`
```python
    def __init__(self, system_skills_json):
        """
        The intelligence core of LinAgent.
        Uses Gemini 2.5-lite Flash with a resilience layer for 503 errors and robust parsing.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env! Please add it.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-2.5-flash-lite'
        
        # Identity and context for the AI
        self.system_instruction = f"""
        You are LinAI, the intelligent assistant for LinAgent on openSUSE Tumbleweed.
        User: OpenSUSE Tumbleweed KDE6 user

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
```
In `__init__` function, we feed AI with our data directly, in the future I am thinking of switching to a local AI model and use RAG. 
This function also takes the API key and gives AI instructions of how the output should be and what it's working for.

```python
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
```
In `execute_intent` function, it checks the validity of the intent AI gave and executes the command in system.
