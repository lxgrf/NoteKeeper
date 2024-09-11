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
    list_chroma_collections()
