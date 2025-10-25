#
# rag.py: Service layer for Retrieval-Augmented Generation.
# - Manages all interactions with the ChromaDB vector store.
# - Initializes and connects to the persistent ChromaDB client.
# - Provides functions to upsert data and query for relevant chunks.
#
import os
import chromadb
from typing import List, Callable, Dict, Any

# --- ChromaDB Client Initialization ---
# Reads the directory path from environment variables for configuration.
# By using a PersistentClient, ChromaDB saves its data to the specified directory,
# ensuring that embeddings are not lost when the server restarts.
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection(name: str):
    """
    Gets or creates a ChromaDB collection by name. In our design, each document
    gets its own collection named after its unique document_id.
    """
    return client.get_or_create_collection(name=name)

def upsert_chunks(collection_name: str, chunks: List[str], metadatas: List[Dict], ids: List[str], embed_func: Callable):
    """
    Generates embeddings for text chunks and upserts them into a ChromaDB collection.
    'Upsert' will add new documents or update existing ones based on their unique ID.
    """
    if not chunks:
        # Avoids making an empty call to the embedding function or ChromaDB.
        return
        
    # The embedding function is passed in from the caller (e.g., upload router).
    # This decouples the RAG service from the specific embedding model,
    # allowing it to work with either the real OpenAI embedder or the offline hash-based one.
    embeddings = embed_func(chunks)
    
    collection = get_collection(collection_name)
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def get_top_k(collection_name: str, query_text: str, k: int, embed_func: Callable) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant chunks from a specific document collection.
    This is the core "retrieval" step in RAG.
    """
    collection = get_collection(collection_name)
    
    # MVP Requirement: If the query_text is empty (as it is for the initial analysis),
    # we don't perform a similarity search. Instead, we return the first 'k' chunks
    # from the document to provide a general, high-level context for the LLM.
    if not query_text:
        results = collection.get(limit=k, include=["metadatas", "documents"])
        # The output of .get() is a dictionary of lists. We need to reformat it to
        # match the structure of the .query() method's output for consistency.
        return [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(results.get("documents", []), results.get("metadatas", []))
        ]

    # If a specific query is provided, generate an embedding for it and perform a similarity search.
    query_embedding = embed_func([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "documents"]
    )
    
    # The result from .query() is a dictionary containing lists of lists (one for each query).
    # Since we only have one query, we access the first element of each list.
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [{"document": doc, "metadata": meta} for doc, meta in zip(docs, metas)]