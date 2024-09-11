import pickle
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
import ollama
import logging

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.database.database import store_embeddings, get_existing_ids


# Remove the following functions:
# - get_chroma_client()
# - get_or_create_chroma_collection()
# - get_existing_ids_chroma()
# - store_embeddings_chroma()

def load_docs_from_cache(database_id: str) -> List[Document]:
    cache_file = project_root / "cache" / "notion" / f"{database_id}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return []

logging.basicConfig(level=logging.INFO)

def create_embeddings(docs: List[Document]) -> Tuple[List[List[float]], List[int]]:
    client = ollama.Client()
    embeddings = []
    valid_indices = []
    for i, doc in enumerate(docs):
        try:
            logging.info(f"Attempting to create embedding for document {i} with content: {doc.page_content[:100]}...")
            response = client.embeddings(model='mistral-nemo', prompt=doc.page_content)
            embedding = response['embedding']
            if embedding:
                embeddings.append(embedding)
                valid_indices.append(i)
                logging.info(f"Successfully created embedding for document {i}")
            else:
                logging.warning(f"Empty embedding received for document {i}. Skipping this document.")
        except Exception as e:
            logging.error(f"Error creating embedding for document {i}: {str(e)}")
    
    if not embeddings:
        raise ValueError("No valid embeddings were created")
    
    return embeddings, valid_indices

def ensure_valid_metadata(docs: List[Document]) -> List[Document]:
    valid_docs = []
    for i, doc in enumerate(docs):
        if not isinstance(doc.metadata, dict):
            logging.warning(f"Document {i} has no metadata. Setting default metadata.")
            doc.metadata = {"source": "unknown", "index": i}
        else:
            # Ensure all values in metadata are not None
            doc.metadata = {k: v if v is not None else "unknown" for k, v in doc.metadata.items()}
        
        # Overwrite the document content with the 'name' metadata
        if 'name' in doc.metadata:
            doc.page_content = doc.metadata['name']
        else:
            logging.warning(f"Document {i} has no 'name' in metadata. Content remains unchanged.")

        
        valid_docs.append(doc)
    return valid_docs

def process_and_store_embeddings(database_id: str, docs: List[Document] = None):
    if docs is None:
        docs = load_docs_from_cache(database_id)
        if not docs:
            logging.warning(f"No documents found for database {database_id}")
            return

    docs = ensure_valid_metadata(docs)
    
    embeddings, valid_indices = create_embeddings(docs)
    if not embeddings:
        logging.error("No valid embeddings were created")
        return

    valid_docs = [docs[i] for i in valid_indices]
    documents = [doc.page_content for doc in valid_docs]
    metadata = [doc.metadata for doc in valid_docs]
    
    collection_name = f"notion_{database_id}"
    existing_ids = get_existing_ids(collection_name)
    
    store_embeddings(
        collection_name=collection_name,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    logging.info(f"Stored or updated {len(embeddings)} embeddings for database {database_id} in ChromaDB")
    logging.warning(f"Skipped {len(docs) - len(embeddings)} documents due to empty embeddings")

# Example usage:
if __name__ == "__main__":
    process_and_store_embeddings("8d5dc8537d04457fa92a543a83ac397b")
    process_and_store_embeddings("a7c454796df647eaa901d324c74cca67")

