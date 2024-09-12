import chromadb
from chromadb.config import Settings
import ollama
from typing import List, Dict
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import json
from pathlib import Path
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# Global variable for Chroma client
chroma_client = None

def get_chroma_client():
    global chroma_client
    if chroma_client is None:
        persist_directory = str(project_root / "chroma_db")
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def query_all_collections(query: str, n_results: int = 5) -> List[Dict]:
    client = get_chroma_client()
    all_results = []
    
    # Get all collection names
    collection_names = client.list_collections()
    
    for collection_name in collection_names:
        collection = client.get_collection(name=collection_name.name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        all_results.extend(results['documents'][0])
    
    return all_results

def generate_ollama_response(context: str, question: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    client = ollama.Client()
    response = client.generate(model='mistral-nemo', prompt=prompt)
    return response['response']

def answer_question(question: str, database_ids: List[str] = None) -> str:
    try:
        # Initialize Ollama embeddings
        embeddings = OllamaEmbeddings(model="mistral-nemo")
        
        # Test embedding generation
        test_embedding = embeddings.embed_query("Test query")
        logger.info(f"Test embedding generated. Length: {len(test_embedding)}")
        
        # Use the global Chroma client
        client = get_chroma_client()
        
        # Initialize Ollama LLM
        llm = Ollama(model="mistral-nemo")
        
        # Get all collection names
        collection_names = client.list_collections()
        
        if not collection_names:
            logger.warning("No collections found. Returning default message.")
            return "Sorry, I couldn't find any relevant information to answer that question."
        
        # Create a list to store retrievers for each collection
        retrievers = []
        
        for collection_name in collection_names:
            try:
                collection = client.get_collection(name=collection_name.name)
                doc_count = collection.count()
                logger.info(f"Collection '{collection_name.name}' exists with {doc_count} documents")
                
                if doc_count > 0:
                    vectorstore = Chroma(
                        client=client,
                        collection_name=collection_name.name,
                        embedding_function=embeddings
                    )
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 40}
                    )
                    retrievers.append(retriever)
            except ValueError:
                logger.error(f"Error accessing collection '{collection_name.name}'.")
        
        if not retrievers:
            logger.warning("No valid collections found. Returning default message.")
            return "Sorry, I couldn't find any relevant information to answer that question."
        
        # Create an EnsembleRetriever with all the individual retrievers
        ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1]*len(retrievers))
        
        # Retrieve documents using the ensemble retriever
        retrieved_docs = ensemble_retriever.get_relevant_documents(question)
        logger.info(f"Number of retrieved documents: {len(retrieved_docs)}")

        # Log retrieved documents
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"Retrieved document {i+1} content: {doc.page_content[:100]}...")  # Log first 100 chars

        if not retrieved_docs:
            logger.warning("No documents retrieved. Returning default message.")
            return "Sorry, I couldn't find any relevant information to answer that question."

        # Use stuff chain type for processing
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True
        )

        # Generate answer with as much context as possible
        max_context_length = 3500
        context = ""
        for doc in retrieved_docs:
            if len(context) + len(doc.page_content) <= max_context_length:
                context += doc.page_content + "\n\n"
            else:
                break

        result = qa_chain.invoke({
            "query": question,
            "context": context
        })

        # Log the full prompt sent to the LLM
        logger.info(f"Full prompt sent to LLM: {result['source_documents']}")
        
        # Log the raw LLM output
        logger.info(f"Raw LLM output: {result['result']}")
        
        return result["result"]

    except Exception as e:
        logger.error(f"Error occurred while answering question: {str(e)}")
        return "Sorry, I couldn't find an answer to that question."

def extract_metadata(text):
    try:
        # Find the start and end of the metadata JSON
        start = text.index('{')
        end = text.rindex('}') + 1
        metadata_str = text[start:end]
        
        # Parse the metadata
        metadata = json.loads(metadata_str)
        
        # Remove the metadata from the text
        clean_text = text[:start].strip() + ' ' + text[end:].strip()
        
        return clean_text, metadata
    except (ValueError, json.JSONDecodeError):
        # If we can't find or parse the metadata, return the original text
        return text, {}

# Example usage:
if __name__ == "__main__":
    database_ids = ["a7c454796df647eaa901d324c74cca67", "8d5dc8537d04457fa92a543a83ac397b"]
    
    
    question = "Who is Ireena?"
    answer = answer_question(question, database_ids)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
