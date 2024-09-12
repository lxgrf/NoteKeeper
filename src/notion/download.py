import logging
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
from notion_client import Client
from langchain.docstore.document import Document

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
config_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=config_path)

def extract_notion_docs(database_id: str):
    logging.info(f"Starting extract_notion_docs for database {database_id}")
    logging.info(f"Extracting Notion docs for database {database_id}")

    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    if not NOTION_API_KEY:
        logging.error("NOTION_API_KEY not found in environment variables")
        return None
    
    notion = Client(auth=NOTION_API_KEY)
    
    try:
        docs = []
        has_more = True
        next_cursor = None

        while has_more:
            response = notion.databases.query(
                database_id=database_id,
                start_cursor=next_cursor
            )

            for page in response['results']:
                page_id = page['id']
                properties = page['properties']
                
                # Extract content
                content = extract_page_content(notion, page_id)
                
                # Create metadata
                metadata = {
                    'notion_id': page_id,
                    'notion_url': page['url'],
                    'notion_properties': {}
                }

                # Process properties
                for prop_name, prop_value in properties.items():
                    prop_type = prop_value['type']
                    if prop_type == 'title':
                        metadata['title'] = prop_value['title'][0]['plain_text'] if prop_value['title'] else ''
                        metadata['name'] = metadata['title']  # Add this line to set 'name' in metadata
                    elif prop_type == 'relation':
                        relation_ids = [relation['id'] for relation in prop_value['relation']]
                        relation_names = get_relation_names(notion, relation_ids)
                        metadata['notion_properties'][prop_name] = relation_names
                    else:
                        metadata['notion_properties'][prop_name] = prop_value[prop_type]

                logging.info(f"Processing page {page_id}")
                logging.info(f"Metadata before processing: {metadata}")
                
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

                logging.info(f"Metadata after processing: {metadata}")
                
            has_more = response['has_more']
            next_cursor = response['next_cursor']

        logging.info(f"Successfully extracted {len(docs)} documents from Notion")
        return docs

    except Exception as e:
        logging.error(f"An error occurred while extracting Notion docs: {e}")
        return None

def extract_page_content(notion, page_id):
    blocks = notion.blocks.children.list(block_id=page_id)
    content = ""
    for block in blocks['results']:
        if block['type'] == 'paragraph':
            content += block['paragraph']['rich_text'][0]['plain_text'] if block['paragraph']['rich_text'] else ""
        # Add more block types as needed

    return content

def get_relation_names(notion, relation_ids):
    names = []
    for relation_id in relation_ids:
        try:
            page = notion.pages.retrieve(relation_id)
            title_property = next((prop for prop in page['properties'].values() if prop['type'] == 'title'), None)
            if title_property:
                name = title_property['title'][0]['plain_text'] if title_property['title'] else 'Untitled'
                names.append(name)
            else:
                names.append('Untitled')
        except Exception as e:
            logging.error(f"Error retrieving related page {relation_id}: {e}")
            names.append('Error')
    return names

# The rest of your file remains unchanged

