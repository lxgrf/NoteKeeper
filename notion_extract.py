from langchain_community.document_loaders import NotionDBLoader
from dotenv import load_dotenv
import os
import requests

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("DATABASE_ID")
print(f"Using Database ID: {DATABASE_ID}")

try:
    loader = NotionDBLoader(
        integration_token=NOTION_API_KEY,
        database_id=DATABASE_ID,
        request_timeout_sec=30,
    )
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} documents")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error occurred: {e}")
    print(f"Response content: {e.response.content}")
except Exception as e:
    print(f"An error occurred: {e}")
    for doc in docs:
        print(doc)

# Test direct API call
url = f"https://api.notion.com/v1/databases/{DATABASE_ID}"
headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28"
}

response = requests.get(url, headers=headers)
print(f"Direct API call status code: {response.status_code}")
# print(f"Direct API call response: {response.text}")
print(docs[0])
print(docs[0].page_content)