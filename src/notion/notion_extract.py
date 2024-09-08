from langchain_community.document_loaders import NotionDBLoader
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import sys
from pathlib import Path

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# Load environment variables
config_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=config_path)


def extract_notion_docs(database_id):
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    
    try:
        loader = NotionDBLoader(
            integration_token=NOTION_API_KEY,
            database_id=database_id,
            request_timeout_sec=30,
        )
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} documents")
        return docs
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
        print(f"Response content: {e.response.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None

def create_embeddings(docs: List[Document]):
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="llama2",  # or any other model you have in Ollama
        base_url="http://localhost:11434"  # default Ollama URL
    )
    # Rest of the embedding logic...
    embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])
    
    print(f"Created {len(embedded_docs)} embeddings")
    return embedded_docs, docs


# Usage example:
if __name__ == "__main__":
    DATABASE_ID = os.getenv("DATABASE_ID")
    docs = extract_notion_docs(DATABASE_ID)
    
    if docs:
        print(docs[0])
        print(docs[0].page_content)

    # Test direct API call
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}"
    headers = {
        "Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}",
        "Notion-Version": "2022-06-28"
    }

    response = requests.get(url, headers=headers)
    print(f"Direct API call status code: {response.status_code}")
    # print(f"Direct API call response: {response.text}")