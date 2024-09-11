from pathlib import Path
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

project_root = Path(__file__).parents[2]

def get_chroma_client():
    persist_directory = str(project_root / "chroma_db")
    return chromadb.PersistentClient(path=persist_directory)

def get_or_create_chroma_collection(collection_name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)

def get_existing_ids_chroma(collection):
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
    client = get_chroma_client()
    collection = get_or_create_chroma_collection(client, collection_name)
    
    existing_ids = get_existing_ids_chroma(collection)
    
    store_embeddings_chroma(collection, documents, embeddings, metadata)