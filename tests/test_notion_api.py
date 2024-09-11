import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from notion.download import list_notion_databases, extract_notion_docs, extract_all_notion_docs, create_embeddings


@pytest.fixture
def mock_env_variables():
    with patch.dict('os.environ', {'NOTION_API_KEY': 'test_api_key'}):
        yield

def test_list_notion_databases(mock_env_variables):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "results": [{"id": "db1", "title": [{"text": {"content": "Database 1"}}]}],
            "has_more": False
        }
        databases = list_notion_databases()
        assert len(databases) == 1
        assert databases[0]["id"] == "db1"
        assert databases[0]["title"][0]["text"]["content"] == "Database 1"

def test_extract_notion_docs(mock_env_variables):
    with patch('langchain_community.document_loaders.NotionDBLoader.load') as mock_load:
        mock_load.return_value = [MagicMock(page_content="Test content")]
        docs = extract_notion_docs("test_db_id")
        assert len(docs) == 1
        assert docs[0].page_content == "Test content"

def test_extract_all_notion_docs(mock_env_variables):
    with patch('notion_api.list_notion_databases') as mock_list_dbs, \
         patch('notion_api.extract_notion_docs') as mock_extract_docs:
        mock_list_dbs.return_value = [
            {"id": "db1", "title": [{"text": {"content": "Database 1"}}]},
            {"id": "db2", "title": [{"text": {"content": "Database 2"}}]}
        ]
        mock_extract_docs.side_effect = [
            [MagicMock(page_content="Content 1")],
            [MagicMock(page_content="Content 2")]
        ]
        all_docs = extract_all_notion_docs()
        assert len(all_docs) == 2
        assert all_docs[0].page_content == "Content 1"
        assert all_docs[1].page_content == "Content 2"

def test_create_embeddings():
    mock_docs = [MagicMock(page_content="Test content 1"), MagicMock(page_content="Test content 2")]
    with patch('langchain_community.embeddings.OllamaEmbeddings.embed_documents') as mock_embed:
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        embedded_docs, original_docs = create_embeddings(mock_docs)
        assert len(embedded_docs) == 2
        assert len(embedded_docs[0]) == 2
        assert original_docs == mock_docs

if __name__ == "__main__":
    pytest.main()