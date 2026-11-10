# from functools import lru_cache
# from pathlib import Path

# from langchain.embeddings import init_embeddings
# from langchain_core.embeddings import Embeddings
# from pydantic_settings import BaseSettings, SettingsConfigDict
# from qdrant_client import QdrantClient


# # api/.env
# ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_prefix="RAG_", env_file=ENV_FILE, extra="ignore"
#     )

#     # App
#     app_name: str = "Clause"
#     data_path: str = "data/shop_handbook.pdf"

#     # Chat model
#     chat_provider: str = "ollama"
#     chat_model: str = "gemma3:1b"
#     chat_temperature: float = 0.1

#     # Embeddings
#     embed_provider: str = "ollama"
#     embed_model: str = "nomic-embed-text:latest"
#     embed_dim: int = 768

#     # Indexing
#     chunk_size: int = 600
#     chunk_overlap: int = 100

#     # Retrieval
#     top_k: int = 3
#     score_threshold: float = 0.5

#     # Qdrant
#     collection_name: str = "shop_handbook"
#     qdrant_path: str = "./qdrant_data"
#     qdrant_url: str | None = None

#     # Prompts
#     system_prompt: str = (
#         "You are a helpful Clause assistant. "
#         "Answer only from the provided context. "
#         "If the answer is not in the context, say you don't know."
#     )


# @lru_cache
# def get_settings() -> Settings:
#     return Settings()


# def get_embeddings(settings: Settings) -> Embeddings:
#     return init_embeddings(
#         model=settings.embed_model,
#         provider=settings.embed_provider,
#     )


# def get_qdrant_client(settings: Settings) -> QdrantClient:
#     if settings.qdrant_url:
#         return QdrantClient(url=settings.qdrant_url)
#     return QdrantClient(path=settings.qdrant_path)


from functools import lru_cache
from pathlib import Path

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient


# api/.env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ENV_FILE,
        extra="ignore",
    )

    # ======================================
    # APP
    # ======================================

    app_name: str = "Clause"
    data_path: str = "data/shop_handbook.pdf"

    # ======================================
    # CHAT MODEL
    # ======================================

    chat_provider: str = "ollama"
    chat_model: str = "gemma3:1b"
    chat_temperature: float = 0.1

    # ======================================
    # EMBEDDINGS
    # ======================================

    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # ======================================
    # INDEXING
    # ======================================

    chunk_size: int = 600
    chunk_overlap: int = 100

    # ======================================
    # RETRIEVAL
    # ======================================

    top_k: int = 3
    score_threshold: float = 0.5

    # ======================================
    # QDRANT
    # ======================================

    collection_name: str = "shop_handbook"
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str | None = None

    # ======================================
    # PROMPTS
    # ======================================

    system_prompt: str = (
        "You are a helpful Clause assistant. "
        "Answer only from the provided context. "
        "If the answer is not in the context, say you don't know."
    )


# ==========================================
# SETTINGS
# ==========================================

@lru_cache
def get_settings() -> Settings:
    return Settings()


# ==========================================
# EMBEDDINGS
# ==========================================

def get_embeddings(settings: Settings) -> Embeddings:

    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider,
    )


# ==========================================
# QDRANT CLIENT
# ==========================================

def get_qdrant_client(settings: Settings) -> QdrantClient:

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)

    return QdrantClient(path=settings.qdrant_path)