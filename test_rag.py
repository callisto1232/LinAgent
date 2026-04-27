from rag import LinRAG

# Mock data mimicking your skills/system JSON structure
mock_skills = {
    "media_controls": [
        {"intent": "play_music", "description": "Starts playing music or resumes playback."},
        {"intent": "stop_music", "description": "Stops the current music stream."}
    ],
    "workspace_management": [
        {"intent": "change_workspace", "description": "Switches to a different virtual desktop or workspace number."}
    ]
}

mock_system = {
    "power": [
        {"intent": "shutdown", "description": "Powers off the computer immediately."},
        {"intent": "reboot", "description": "Restarts the operating system."}
    ]
}

def run_test():
    print("--- Initializing LinRAG Engine ---")
    # Initializing with our mock dictionaries
    rag = LinRAG(mock_skills, mock_system)
    
    test_queries = [
        "stop the song currently playing",
        "restart my computer",
        "change virtual desktop to 8"
    ]

    print("\n--- Testing Retrieval ---")
    for query in test_queries:
        print(f"\n👤 Query: '{query}'")
        results = rag.query(query, top_k=2)
        
        for i, res in enumerate(results):
            intent = res.get('intent')
            desc = res.get('description')
            print(f"  [{i+1}] Match: {intent} - {desc}")

if __name__ == "__main__":
    run_test()
