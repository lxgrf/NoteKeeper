import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chroma_client = None

def get_chroma_client():
    global chroma_client
    if chroma_client is None:
        persist_directory = str(project_root / "chroma_db")
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client

def list_chroma_collections():
    try:
        # Initialize Chroma client
        client = get_chroma_client()
        
        # Get all collection names
        collections = client.list_collections()
        
        if collections:
            logger.info("Collections in ChromaDB:")
            for collection in collections:
                logger.info(f"- {collection.name} (Number of documents: {collection.count()})")
        else:
            logger.info("No collections found in ChromaDB.")
        
        return collections
    except Exception as e:
        logger.error(f"Error occurred while listing collections: {str(e)}")
        return []

if __name__ == "__main__":
    collections = list_chroma_collections()
    
    if not collections:
        logger.info("No collections to dump.")
        sys.exit(0)
    
    output_file = project_root / "document_dump.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for collection in collections:
            logger.info(f"Dumping documents from collection: {collection.name}")
            
            # Get all documents in the collection
            documents = collection.get(include=["documents", "metadatas"])
            
            # Sort documents by name if available, otherwise by ID
            sorted_docs = sorted(
                zip(documents["ids"], documents["documents"], documents["metadatas"]),
                key=lambda x: x[2].get("name", x[0])  # Sort by name if available, otherwise by ID
            )
            
            f.write(f"Collection: {collection.name}\n")
            f.write("=" * 50 + "\n\n")
            
            for doc_id, content, metadata in sorted_docs:
                f.write(f"Document ID: {doc_id}\n")
                f.write(f"Metadata: {metadata}\n")
                f.write("Content:\n")
                f.write(content + "\n")
                f.write("-" * 50 + "\n\n")
            
            f.write("\n\n")
    
    logger.info(f"Document dump completed. Output saved to {output_file}")
