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
from notion_client import Client

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))
from ollama_utils.ingest import process_and_store_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
config_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=config_path)

def extract_notion_docs(database_id: str):
    logging.info(f"Extracting Notion docs for database {database_id}")

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
            include_child_pages=True,
            include_child_databases=True
        )
        logging.info("Attempting to load documents from Notion")
        docs = loader.load()
        logging.info(f"Successfully loaded {len(docs)} documents from Notion")

        # Initialize Notion client for additional API calls
        notion = Client(auth=NOTION_API_KEY)

        for doc in docs:
            if 'properties' in doc.metadata:
                for prop_name, prop_value in doc.metadata['properties'].items():
                    if prop_value['type'] == 'relation':
                        relation_ids = [relation['id'] for relation in prop_value.get('relation', [])]
                        related_titles = []
                        for relation_id in relation_ids:
                            try:
                                related_page = notion.pages.retrieve(relation_id)
                                related_title = related_page['properties'].get('Name', {}).get('title', [{}])[0].get('plain_text', 'Untitled')
                                related_titles.append(related_title)
                            except Exception as e:
                                logging.error(f"Error retrieving related page {relation_id}: {e}")
                        
                        # Store both the relation IDs and titles in the metadata
                        doc.metadata[f'{prop_name}_ids'] = relation_ids
                        doc.metadata[f'{prop_name}_titles'] = related_titles

            # Add the entire properties object to the metadata
            doc.metadata['notion_properties'] = doc.metadata.get('properties', {})

            logging.info(f"Document metadata after processing: {doc.metadata}")

        return docs
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error occurred: {e}")
        logging.error(f"Response content: {e.response.content}")
    except Exception as e:
        logging.error(f"An error occurred while extracting Notion docs: {e}")
    
    return None

# Usage example:
if __name__ == "__main__":
    logging.info("Starting API key verification")
    
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    if not NOTION_API_KEY:
        logging.error("NOTION_API_KEY not found in environment variables")
        sys.exit(1)
    
    try:
        notion = Client(auth=NOTION_API_KEY)
        # Try to list users, which requires a valid API key
        notion.users.list()
        logging.info("Notion API key is valid")
    except Exception as e:
        logging.error(f"Failed to verify Notion API key: {e}")
        sys.exit(1)
    
    logging.info("API key verification completed")


