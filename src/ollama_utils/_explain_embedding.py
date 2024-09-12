import sys
from pathlib import Path
import random
import ollama_utils
from chromadb.api.models.Collection import Collection
from src.database.database import get_collection

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

def get_random_embedding(collection: Collection):
    # Get all embeddings from the collection
    all_embeddings = collection.get()
    
    # Choose a random embedding
    random_index = random.randint(0, len(all_embeddings['embeddings']) - 1)
    return all_embeddings['embeddings'][random_index], all_embeddings['documents'][random_index]

def explain_embedding(embedding: list, document: str):
    client = ollama_utils.Client()
    prompt = f"""
    Given the following embedding vector and the corresponding document text, 
    please explain in natural language what this embedding might represent:

    Embedding: {embedding[:10]}... (truncated for brevity)
    Document: {document}

    Explain the potential meaning or content this embedding might capture:
    """
    
    response = client.generate(model='mistral-nemo', prompt=prompt)
    return response['response']

def main():
    # Assuming we're using the 'notion_8d5dc8537d04457fa92a543a83ac397b' collection
    collection_name = "8d5dc8537d04457fa92a543a83ac397b"
    collection = get_collection(collection_name)
    
    embedding, document = get_random_embedding(collection)
    explanation = explain_embedding(embedding, document)
    
    print("Random Document:")
    print(document)
    print("\nEmbedding Explanation:")
    print(explanation)

if __name__ == "__main__":
    main()
