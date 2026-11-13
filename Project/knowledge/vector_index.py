"""Turns passages into vectors, stores them, reads a section back and searches them.

This is the only module that talks to the vector store (ChromaDB).
"""

import os

import chromadb

from config import config

# The collection is small (a few hundred passages), so the HNSW graph is built and
# searched with a beam wider than the collection: the nearest passages are found exactly.
_EXACT_SEARCH_SETTINGS = {
    "hnsw:space": config.VECTOR_DISTANCE_METRIC,
    "hnsw:construction_ef": 1000,
    "hnsw:search_ef": 1000,
    "hnsw:M": 64,
}


def _embedding_function():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME,
                                                device=config.EMBEDDING_DEVICE)


def _store_file(store_path):
    return os.path.join(store_path, config.VECTOR_STORE_FILE_NAME)


def build_vector_index(chunk_records, store_path):
    """Build (or rebuild) the persistent knowledge base; returns the passage count."""
    if os.path.isfile(store_path):
        raise ValueError(f"{store_path} is a file, not a folder for the knowledge base")
    if os.path.isdir(store_path) and os.listdir(store_path) and \
            not os.path.isfile(_store_file(store_path)):
        raise ValueError(f"{store_path} holds files but no {config.VECTOR_STORE_FILE_NAME}")

    # Load the embedding model before touching the store: if it cannot be loaded the
    # existing store is left exactly as it was.
    embedding = _embedding_function()
    embedding(["load check"])

    os.makedirs(store_path, exist_ok=True)
    client = chromadb.PersistentClient(path=store_path)
    existing = [getattr(item, "name", item) for item in client.list_collections()]
    if config.VECTOR_COLLECTION_NAME in existing:
        client.delete_collection(config.VECTOR_COLLECTION_NAME)
    collection = client.create_collection(name=config.VECTOR_COLLECTION_NAME,
                                          embedding_function=embedding,
                                          metadata=dict(_EXACT_SEARCH_SETTINGS))
    batch = 128
    for start in range(0, len(chunk_records), batch):
        part = chunk_records[start:start + batch]
        collection.add(
            ids=[record["chunk_id"] for record in part],
            documents=[record["chunk_text"] for record in part],
            metadatas=[{
                "document_name": record["document_name"],
                "section_number": int(record["section_number"]),
                "passage_number": int(record["passage_number"]),
                "carried_characters": int(record["carried_characters"]),
            } for record in part],
        )
    return collection.count()


def open_vector_index(store_path):
    """Open the knowledge base for reading and searching, without changing it."""
    if not os.path.isfile(_store_file(store_path)):
        raise ValueError(f"{store_path} holds no {config.VECTOR_STORE_FILE_NAME}")
    client = chromadb.PersistentClient(path=store_path)
    try:
        return client.get_collection(name=config.VECTOR_COLLECTION_NAME,
                                     embedding_function=_embedding_function())
    except Exception as failure:  # chromadb raises its own error types across versions
        raise ValueError(f"{store_path} holds no {config.VECTOR_COLLECTION_NAME} collection: "
                         f"{failure}") from failure


def read_document_section(collection, document_name, section_number):
    """Read one section of one document back, whole and once."""
    found = collection.get(
        where={"$and": [{"document_name": document_name},
                        {"section_number": int(section_number)}]},
        include=["documents", "metadatas"])
    passages = sorted(zip(found["metadatas"], found["documents"]),
                      key=lambda pair: pair[0]["passage_number"])
    if not passages:
        raise ValueError(config.NO_SECTION_PASSAGES_MESSAGE.format(
            section_number=section_number, document_name=document_name))
    pieces = []
    for metadata, text in passages:
        pieces.append(text[int(metadata["carried_characters"]):] if pieces else text)
    return " ".join(piece for piece in pieces if piece)


def search_documents(collection, query_texts):
    """Ask several questions at once; nearest passages first, casebook never returned."""
    excluded = list(config.CHAT_EXCLUDED_DOCUMENTS)
    where = {"document_name": {"$nin": excluded}} if excluded else None
    found = collection.query(query_texts=list(query_texts),
                             n_results=config.CHAT_RETRIEVAL_RESULT_COUNT,
                             where=where,
                             include=["documents", "metadatas", "distances"])
    results = []
    for ids, texts, metadatas, distances in zip(found["ids"], found["documents"],
                                                found["metadatas"], found["distances"]):
        passages = [{
            "chunk_id": chunk_id,
            "chunk_text": text,
            "document_name": metadata["document_name"],
            "section_number": int(metadata["section_number"]),
            "distance": float(distance),
        } for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances)]
        results.append(sorted(passages, key=lambda passage: passage["distance"]))
    return results
