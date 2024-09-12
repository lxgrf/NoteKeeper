from pathlib import Path
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import shutil

project_root = Path(__file__).parents[2]

def get_chroma_client():
    persist_directory = str(project_root / "chroma_db")
    return chromadb.PersistentClient(path=persist_directory)

def get_or_create_chroma_collection(collection_name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)

def get_existing_ids_chroma(collection_name: str):
    collection = get_or_create_chroma_collection(collection_name)
    return collection.get(include=['documents'])['ids']

def store_embeddings_chroma(collection, documents: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

def process_and_store_embeddings_chroma(database_id: str, documents: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
    collection_name = f"notion_{database_id}"
    collection = get_or_create_chroma_collection(collection_name)
    
    existing_ids = get_existing_ids_chroma(collection_name)
    
    store_embeddings_chroma(collection, documents, embeddings, metadata)


def reset_database():
    persist_directory = project_root / "chroma_db"
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
        print(f"Removed existing database at {persist_directory}")
    persist_directory.mkdir(parents=True, exist_ok=True)
    print(f"Created new empty database directory at {persist_directory}")
    chromadb.PersistentClient(path=str(persist_directory))
    print("Initialized new Chroma database")

if __name__ == "__main__":
    reset_database()
    print("Database reset completed successfully")