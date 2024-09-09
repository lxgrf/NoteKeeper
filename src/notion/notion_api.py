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

# Load environment variables
config_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=config_path)

def save_docs_locally(docs: List[Document], database_id: str):
    cache_dir = project_root / "cache" / "notion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{database_id}.pkl"
    
    with open(cache_file, "wb") as f:
        pickle.dump(docs, f)
    print(f"Saved {len(docs)} documents to {cache_file}")

def load_docs_locally(database_id: str) -> List[Document]:
    cache_file = project_root / "cache" / "notion" / f"{database_id}.pkl"
    
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            docs = pickle.load(f)
        print(f"Loaded {len(docs)} documents from {cache_file}")
        return docs
    return None

def extract_notion_docs(database_id: str, use_cache: bool = True):
    if use_cache:
        cached_docs = load_docs_locally(database_id)
        if cached_docs:
            return cached_docs

    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    
    try:
        loader = NotionDBLoader(
            integration_token=NOTION_API_KEY,
            database_id=database_id,
            request_timeout_sec=30,
        )
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} documents")
        save_docs_locally(docs, database_id)
        return docs
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
        print(f"Response content: {e.response.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None

# Usage example:
if __name__ == "__main__":
    for dbase in [ # Test Server
        # "8d5dc8537d04457fa92a543a83ac397b", # Facts
        "a7c454796df647eaa901d324c74cca67" # Events
    ]:
        print(extract_notion_docs(dbase, use_cache=True))

