from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./models/rag_model")
print("Model saved to ./models/rag_model")
