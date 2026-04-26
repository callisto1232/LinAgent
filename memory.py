import json
import numpy as np
from sentence_transformers import SentenceTransformer

class LinMemory:
    def __init__(self, json_path="data/system.json"):
        print("--- Initializing LinMemory (Lightweight NumPy Mode) ---")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.entries = []
        self.vectors = None
        self._ingest_json(json_path)

    def _ingest_json(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Since your JSON is a list containing one object with 'system_skills'
            root = data[0].get('system_skills', {})
            
            self.entries = []
            for category, skills in root.items():
                for skill in skills:
                    # Flattening the nested data for the embedding model
                    intent = skill.get('intent', 'unknown')
                    desc = skill.get('description', '')
                    cmd = skill.get('command', 'N/A')
                    
                    # This string is what the AI "reads" to find matches
                    entry_text = f"Intent: {intent} | Description: {desc} | Command: {cmd}"
                    self.entries.append(entry_text)
            
            if not self.entries:
                print("Warning: No skills found in JSON.")
                return

            # Generate embeddings
            embeddings = self.model.encode(self.entries)
            self.vectors = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print(f"--- Successfully indexed {len(self.entries)} skills across {len(root)} categories ---")

        except Exception as e:
            print(f"Error loading JSON: {e}")

    def recall(self, query, k=2):
        if self.vectors is None or len(self.entries) == 0:
            return []
        
        query_vec = self.model.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        similarities = np.dot(self.vectors, query_vec.T).flatten()
        best_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.entries[i] for i in best_indices]

if __name__ == "__main__":
    test_memory = LinMemory() 
    # Testing with one of your specific skills
    print("\nSearch Result for 'how to create a folder named gemini in documents folder':")
    results = test_memory.recall("create a folder")
    for r in results:
        print(f" - {r}")
