
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)

from app.core.config import Settings, get_embeddings, get_qdrant_client


def get_vector_store(settings: Settings) -> QdrantVectorStore:
    client = get_qdrant_client(settings)

    if not client.collection_exists(settings.collection_name):
        raise RuntimeError(
            f"Collection {settings.collection_name} not found!"
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(settings),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        vector_name="dense",
        sparse_vector_name="sparse",
    )



