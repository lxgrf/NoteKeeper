import pickle
from pathlib import Path
import sys
from typing import List
from langchain_core.documents import Document
import ollama

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.database.database import store_embeddings, get_existing_ids


def load_docs_from_cache(database_id: str) -> List[Document]:
    cache_file = project_root / "cache" / "notion" / f"{database_id}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return []

def create_embeddings(docs: List[Document]) -> List[List[float]]:
    client = ollama.Client()
    embeddings = []
    for doc in docs:
        response = client.embeddings(model='mistral-nemo', prompt=doc.page_content)
        embeddings.append(response['embedding'])
    return embeddings

def process_and_store_embeddings(database_id: str):
    docs = load_docs_from_cache(database_id)
    if not docs:
        print(f"No documents found for database {database_id}")
        return

    embeddings = create_embeddings(docs)
    documents = [doc.page_content for doc in docs]
    metadata = [doc.metadata for doc in docs]
    
    collection_name = f"notion_{database_id}"
    existing_ids = get_existing_ids(collection_name, documents)  # Pass documents here
    
    store_embeddings(
        collection_name=collection_name,
        documents=documents,
        embeddings=embeddings,
        metadata=metadata,
        ids=existing_ids
    )
    print(f"Stored or updated {len(docs)} embeddings for database {database_id}")

# Example usage:
if __name__ == "__main__":
    database_id = "8d5dc8537d04457fa92a543a83ac397b"
    process_and_store_embeddings(database_id)