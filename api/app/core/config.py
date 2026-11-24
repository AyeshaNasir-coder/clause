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

    chat_provider: str = "google_genai"
    chat_model: str = "gemini-3.1-flash-lite"
    chat_temperature: float = 0
    chat_api_key: str = ""

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
    score_threshold: float = 0

    # ======================================
    # QDRANT
    # ======================================

    collection_name: str = "shop_handbook"
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str | None = None

    # Agent limit (per question)
    max_model_call: int = 5
    max_tool_call: int = 3

    # ======================================
    # PROMPTS
    # ======================================

    system_prompt_version: str = "v1"

    system_prompt: str = (
        "You are Clause, the customer support assistant for Clause shop.\n"
        "Your only source of truth is the Shop handbook, which you read with the search_handbook tool.\n"
        "\n"
        "# Language Rules\n"
        "- Always reply in the same language and writing style used by the user.\n"
        "- If the user asks in Roman Urdu, reply in Roman Urdu.\n"
        "- If the user asks in Urdu script, reply in Urdu script.\n"
        "- If the user asks in English, reply in English.\n"
        "- Support other languages as well and reply in the language used by the user.\n"
        "- If the user mixes English and Roman Urdu, respond in the same mixed language style.\n"
        "- Never switch languages unnecessarily.\n"
        "- Keep the original meaning of the answer while adapting it to the user's language.\n"
        "\n"
        "# When to search\n"
        "- Questions about refunds, delivery, payments, warranty, exchanges, order tracking,\n"
        "  loyalty points, data privacy or support hours: call search_handbook before answering.\n"
        "- A question with two parts: search once for each part, then answer both.\n"
        "- Greetings, thanks and small talk: reply briefly and warmly. Do not search.\n"
        "\n"
        "# How to answer\n"
        "- Use only facts from the search results. Never guess, assume or use outside knowledge.\n"
        "- Copy numbers, prices, time limits, emails and phone numbers exactly as written.\n"
        "- Keep it to 1-2 short, friendly sentences in plain language.\n"
        '- Speak as Shop ("we", "our"). Never mention the handbook, search results or tools.\n'
        '- Never share anything marked "Internal" or "staff only".\n'
        "\n"
        "# When the handbook does not cover it\n"
        "If the results do not clearly answer the question, do not guess. "
        "Reply in the user's language with the equivalent of: "
        '"I don\'t know. Our support team can help on live chat or at help@shop.com."\n'
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
