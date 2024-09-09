import chromadb

chroma_client = chromadb.Client()

def get_existing_ids(collection_name: str, documents: list[str]) -> list[str]:
    """
    Retrieve existing IDs from the ChromaDB collection that match the given documents.
    
    :param collection_name: Name of the collection to search
    :param documents: List of document texts to match
    :return: List of existing IDs that match the documents
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    existing_ids = []
    
    for i, doc in enumerate(documents):
        results = collection.query(query_texts=[doc], n_results=1)
        if results['ids'][0]:
            existing_ids.append(results['ids'][0][0])
        else:
            existing_ids.append(f"doc_{i}")
    
    return existing_ids

def store_embeddings(collection_name: str, documents: list[str], embeddings: list[list[float]], metadata: list[dict] = None, ids: list[str] = None):
    """
    Store embeddings in the ChromaDB database, updating existing entries if IDs are provided.
    
    :param collection_name: Name of the collection to store embeddings in
    :param documents: List of document texts
    :param embeddings: List of embedding vectors
    :param metadata: Optional list of metadata dictionaries for each document
    :param ids: Optional list of IDs for the documents
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    if metadata is None:
        metadata = [{} for _ in documents]
    
    if ids is None:
        ids = get_existing_ids(collection_name, documents)
    
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )