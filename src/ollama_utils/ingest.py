import pickle
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
import logging
import ollama

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.database.database import store_embeddings_chroma, get_existing_ids_chroma, get_or_create_chroma_collection
from src.notion.download import extract_notion_docs  # Add this import

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

# Configure logging at the beginning of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Explicitly set the root logger's level to INFO
logging.getLogger().setLevel(logging.INFO)

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
    logging.info(f"Starting process_and_store_embeddings for database {database_id}")
    
    if docs is None:
        logging.info(f"Extracting Notion docs for database {database_id}")
        docs = extract_notion_docs(database_id)
        logging.info(f"Extracted {len(docs)} documents from Notion")
    
    docs = ensure_valid_metadata(docs)
    logging.info(f"Processed {len(docs)} documents with valid metadata")

    # Log metadata for debugging
    logging.info("Sample of document metadata:")
    for i, doc in enumerate(docs[:5]):  # Log first 5 documents
        logging.info(f"Document {i} metadata: {doc.metadata}")

    # Group documents by 'About NPC' metadata
    npc_groups = {}
    for doc in docs:
        about_npcs = doc.metadata.get('notion_properties', {}).get('About NPC', [])
        if not about_npcs:
            continue  # Skip documents with no 'About NPC' value
        for npc in about_npcs:
            if npc not in npc_groups:
                npc_groups[npc] = []
            npc_groups[npc].append(doc)

    logging.info(f"Created {len(npc_groups)} NPC groups")
    logging.info(f"NPC groups: {list(npc_groups.keys())}")

    # Create synthesized documents
    synthesized_docs = []
    for npc, npc_docs in npc_groups.items():
        content = "\n".join([doc.page_content for doc in npc_docs])
        metadata = {
            "About NPC": npc,
            "document_count": len(npc_docs),
            "source": "synthesized"
        }
        synthesized_docs.append(Document(page_content=content, metadata=metadata))

    logging.info(f"Created {len(synthesized_docs)} synthesized documents")

    # Log synthesized document metadata
    logging.info("Sample of synthesized document metadata:")
    for i, doc in enumerate(synthesized_docs[:5]):  # Log first 5 synthesized documents
        logging.info(f"Synthesized Document {i} metadata: {doc.metadata}")

    # Additional logging to show how many documents were processed vs. skipped
    total_docs = len(docs)
    processed_docs = sum(len(npc_docs) for npc_docs in npc_groups.values())
    skipped_docs = total_docs - processed_docs
    logging.info(f"Total documents: {total_docs}")
    logging.info(f"Processed documents: {processed_docs}")
    logging.info(f"Skipped documents (no 'About NPC'): {skipped_docs}")

    embeddings, valid_indices = create_embeddings(synthesized_docs)
    if not embeddings:
        logging.error("No valid embeddings were created")
        return

    valid_docs = [synthesized_docs[i] for i in valid_indices]
    documents = [doc.page_content for doc in valid_docs]
    metadata = [doc.metadata for doc in valid_docs]
    
    collection_name = f"notion_{database_id}"
    collection = get_or_create_chroma_collection(collection_name)
    
    store_embeddings_chroma(
        collection=collection,
        documents=documents,
        embeddings=embeddings,
        metadata=metadata
    )
    
    logging.info(f"Stored {len(embeddings)} embeddings for database {database_id}")
    if len(synthesized_docs) > len(embeddings):
        logging.warning(f"Skipped {len(synthesized_docs) - len(embeddings)} documents due to empty embeddings")

# Example usage:
if __name__ == "__main__":
    logging.info("Script started")
    ingest_list = ["8d5dc8537d04457fa92a543a83ac397b"]
    for dbase in ingest_list:
        logging.info(f"Processing database: {dbase}")
        process_and_store_embeddings(database_id=dbase)
    logging.info("Script finished")
