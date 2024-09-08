import chromadb

chroma_client = chromadb.Client()

def store_embeddings(collection_name: str, documents: list[str], embeddings: list[list[float]], metadata: list[dict] = None):
    """
    Store embeddings in the ChromaDB database.
    
    :param collection_name: Name of the collection to store embeddings in
    :param documents: List of document texts
    :param embeddings: List of embedding vectors
    :param metadata: Optional list of metadata dictionaries for each document
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    if metadata is None:
        metadata = [{} for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )