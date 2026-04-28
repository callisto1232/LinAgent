import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/MiniLM-L6-v2"
class LinRAG:
    def __init__(self, skills_json, system_json):
        self.model = SentenceTransformer(MODEL_PATH, local_files_only=True)
        self.skills_pool = []
        
        # Unified to single underscore calls
        self._index_data(skills_json)
        self._index_data(system_json)
        # self._index_data(apps_json)  # Ready for future use

    def _index_data(self, data):
        for category, actions in data.items():
            if isinstance(actions, list):
                for action in actions:
                    if isinstance(action, dict):
                        # Combine intent and description for context
                        content = f"{action.get('intent', '')} {action.get('description', '')}"
                        action['embedding'] = self.model.encode(content)
                        self.skills_pool.append(action)

    def query(self, user_prompt, top_k=5):
        query_vec = self.model.encode(user_prompt)
        
        # Calculate cosine similarities
        scores = []
        for skill in self.skills_pool:
            skill_vec = skill.get('embedding')
            if skill_vec is None:
                scores.append(0)
                continue
                
            norm = (np.linalg.norm(query_vec) * np.linalg.norm(skill_vec))
            score = np.dot(query_vec, skill_vec) / norm if norm != 0 else 0
            scores.append(score)
        
        # Get top matching skills
        best_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for i in best_indices:
            # Return a copy without the embedding vector
            item = self.skills_pool[i].copy()
            item.pop('embedding', None)
            results.append(item)
            
        return results
