from pathlib import Path

# from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_embeddings, get_qdrant_client, get_settings


def load(path: Path) -> list[Document]:
    return PyPDFLoader(path).load()


def ingest(path: str | None = None) -> int:
    settings = get_settings()

    # 1. Load
    file = Path(path or settings.data_path)
    documents = load(file)

    # 2. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No text found in {file}")

    # 3. Embed: check the model works before the old collection is deleted
    embed = get_embeddings(settings)

    actual_dim = len(embed.embed_query("dimension check"))
    if actual_dim != settings.embed_dim:
        raise ValueError(
            f"RAG_EMBED_DIM is {settings.embed_dim}, "
            f"but {settings.embed_model} returns {actual_dim} dimensions"
        )

    # 4. Store
    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)

        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embed_dim, distance=Distance.COSINE
            ),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embed,
        )
        vector_store.add_documents(chunks)
    finally:
        client.close()

    return len(chunks)



if __name__=="__main__":
    print(ingest(), "chunks indexed")

#to change in documents or embedding model
#cd api
# uv run python -m app.services.ingestion
