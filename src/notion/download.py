import logging
from langchain_community.document_loaders import NotionDBLoader
from dotenv import load_dotenv
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import sys
from pathlib import Path
import pickle

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))
from ollama.ingest import process_and_store_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
config_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=config_path)

def save_docs_locally(docs: List[Document], database_id: str):
    logging.info(f"Attempting to save {len(docs)} documents locally for database {database_id}")
    cache_dir = project_root / "cache" / "notion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{database_id}.pkl"
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(docs, f)
        logging.info(f"Successfully saved {len(docs)} documents to {cache_file}")
    except Exception as e:
        logging.error(f"Error saving documents locally: {e}")

def load_docs_locally(database_id: str) -> List[Document]:
    logging.info(f"Attempting to load documents locally for database {database_id}")
    cache_file = project_root / "cache" / "notion" / f"{database_id}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                docs = pickle.load(f)
            logging.info(f"Successfully loaded {len(docs)} documents from {cache_file}")
            return docs
        except Exception as e:
            logging.error(f"Error loading documents from cache: {e}")
    else:
        logging.info(f"No cache file found for database {database_id}")
    return None

def extract_notion_docs(database_id: str, use_cache: bool = True):
    logging.info(f"Extracting Notion docs for database {database_id}, use_cache={use_cache}")
    if use_cache:
        cached_docs = load_docs_locally(database_id)
        if cached_docs:
            return cached_docs

    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    if not NOTION_API_KEY:
        logging.error("NOTION_API_KEY not found in environment variables")
        return None
    
    try:
        logging.info(f"Initializing NotionDBLoader for database {database_id}")
        loader = NotionDBLoader(
            integration_token=NOTION_API_KEY,
            database_id=database_id,
            request_timeout_sec=30,
        )
        logging.info("Attempting to load documents from Notion")
        docs = loader.load()
        logging.info(f"Successfully loaded {len(docs)} documents from Notion")
        save_docs_locally(docs, database_id)
        return docs
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error occurred: {e}")
        logging.error(f"Response content: {e.response.content}")
    except Exception as e:
        logging.error(f"An error occurred while extracting Notion docs: {e}")
    
    return None

def process_notion_databases(dbases: List[str], use_cache: bool = False) -> Dict[str, bool]:
    logging.info("Starting database processing")
    results = {}
    for dbase in dbases:
        logging.info(f"Processing database: {dbase}")
        docs = extract_notion_docs(dbase, use_cache=use_cache)
        if docs:
            logging.info(f"Processing and storing embeddings for database {dbase}")
            try:
                process_and_store_embeddings(database_id=dbase, docs=docs)
                results[dbase] = True
                logging.info(f"Successfully processed database: {dbase}")
            except Exception as e:
                logging.error(f"Failed to process database {dbase}: {str(e)}")
                results[dbase] = False
        else:
            logging.warning(f"No documents extracted for database {dbase}")
            results[dbase] = False
    logging.info("Database processing completed")
    return results

# Usage example:
if __name__ == "__main__":
    logging.info("Starting main execution")
    test_databases = [
        # "8d5dc8537d04457fa92a543a83ac397b", # Facts
        "a7c454796df647eaa901d324c74cca67"  # Events
    ]
    process_notion_databases(test_databases, use_cache=False)
    logging.info("Main execution completed")


